"""Agent 核心 —— 组装组件，提供 process() 入口。

依赖:
    Agent
    ├── EventBus         ← 事件总线
    ├── AgentMemory      ← 纯存储（文件 I/O）
    ├── Compactor        ← 记忆压缩（LLM 提取）
    ├── TokenTracker     ← 用量统计
    ├── LLMClient        ← LLM 调用
    ├── ToolExecutor     ← 工具调度
    └── AgentRunner      ← think-act 编排
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator
from ..ai.config import AppConfig
from .memory import AgentMemory
from .compactor import Compactor
from .tracker import TokenTracker
from ..ai.llm import LLMClient
from .runner import AgentRunner
from .events import EventBus
from ..tools import (
    ToolRegistry, BashTool, FileReadTool, FileWriteTool, FileEditTool,
    WebFetchTool, WebSearchTool, SkillTool, SkillsLoader, TodoWriteTool, SubagentTool,
)
from ..tools.executor import ToolExecutor
from ..tools.subagent_loader import AgentLoader
from .prompts import PromptLoader


class Agent:
    """Agent 核心引擎。

    用法:
        agent = Agent(config)
        for chunk in agent.process("你好"):
            print(chunk, end="")
        agent.shutdown()
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        thinking: str | None = None,
    ):
        self.config = config or AppConfig.from_env()
        cfg = self.config
        root = cfg.root
        client = cfg.create_client()

        # ── 事件总线 ──
        self.bus = EventBus()

        # ── 存储（纯 I/O，不依赖 LLM）──
        self.memory = AgentMemory(memory_dir=cfg.memory_dir)

        # ── 压缩器（LLM 提取，不读写文件）──
        self.compactor = Compactor(client=client, model=cfg.model)

        # ── Token 统计 ──
        self.tracker = TokenTracker(
            log_file=Path(cfg.memory_dir) / "tokens.jsonl"
        )

        # ── 历史 ──
        self.history: list[dict] = self.memory.history

        # ── 技能 / Agent 定义 / Prompt 模板 ──
        skills = SkillsLoader(skill_directory=cfg.skills_dir)
        agent_loader = AgentLoader(cfg.root / "agent" / "subagent")
        self.prompt_loader = PromptLoader(cfg.root / "agent" / "prompts")

        # ── 系统提示词 + 会话恢复 ──
        system_prompt = self._build_system_prompt(skills, agent_loader)
        self.memory.append_history({"role": "system", "content": system_prompt})
        if cfg.restore_session:
            old = self.memory.restore_history()
            if old:
                for msg in old:
                    self.memory.append_history(msg)
                print(f"[Agent] 已恢复 {len(old)} 条历史消息", flush=True)

        # ── 工具注册 ──
        registry = self._build_registry(skills, client, agent_loader)

        # ── LLM / 工具 / 运行器 ──
        self.runner = AgentRunner(
            llm_client=LLMClient(
                client=client, model=cfg.model,
                max_tokens=cfg.max_tokens, thinking=thinking,
            ),
            tool_executor=ToolExecutor(registry=registry, event_bus=self.bus),
            token_tracker=self.tracker,
            event_bus=self.bus,
            max_turns=cfg.max_turns,
        )

        # ── 事件（在组件就绪后注册）──
        self._setup_events()

    # ── 公共 API ──

    def process(self, message: str) -> Generator[str, None, None]:
        """处理用户消息，流式产出 LLM 响应文本。"""
        self.bus.emit("message:received", {"text": message})
        self._add_message("user", message)
        yield from self.runner.step(self.history)

    def shutdown(self) -> dict:
        """关闭会话，返回统计和压缩结果。"""
        self.bus.emit("session:end", {})
        stats = self.tracker.stats_by_model()
        compact_result = self._do_compact()
        return {"token_stats": stats, "compact": compact_result}

    # ── 内部 ──

    def _add_message(self, role: str, content: str) -> None:
        self.memory.append_history({"role": role, "content": content})

    def _should_compact(self) -> bool:
        return self.tracker.should_compact(
            self.config.max_context, self.config.compact_threshold
        )

    def _do_compact(self) -> dict:
        """执行压缩：Compactor 提取 → Memory 写入。"""
        data = self.compactor.compact(self.history)

        summary = data.get("summary", {})
        if any(summary.values()):
            self.memory.append_summary(
                summary.get("critical", "无"),
                summary.get("decision", "无"),
                summary.get("issue", "无"),
            )

        for p in data.get("preferences", []):
            self.memory.add_user(p)

        for f in data.get("facts", []):
            self.memory.add_memory(f)

        if summary or data.get("preferences") or data.get("facts"):
            print("[Memory] 自动压缩完成", flush=True)

        return data

    def _setup_events(self) -> None:
        """注册核心事件监听器。"""
        @self.bus.on("context:high")
        def _on_context_high(event):
            self._do_compact()

        @self.bus.on("turn:end")
        def _on_turn_end(event):
            if self._should_compact():
                self.bus.emit("context:high", {
                    "input_tokens": self.tracker.last_input_tokens(),
                    "threshold": self.config.compact_threshold,
                })

    def _build_system_prompt(self, skills: SkillsLoader, agent_loader: AgentLoader) -> str:
        commands = self.prompt_loader.list_commands()
        parts = ["你是一个智能助手，可以使用各种工具来帮助用户完成任务。"]
        if self.config.context_files:
            parts.append(f"### 项目上下文\n{self.config.context_files}")
        parts.append(f"### 可用技能列表\n{skills.get_description()}")
        parts.append(f"### 可用子代理\n{agent_loader.list_agents()}")
        parts.append(f"### 可用命令\n{commands or '（无）'}")
        parts.append(f"### 长期记忆（最近摘要）\n{self.memory.brief_context()}")
        parts.append(
            f"### 用户偏好（USER.md）\n"
            + ("\n".join(self.memory.user_preferences()) or "（当前没有用户偏好）")
        )
        return "\n\n".join(parts)

    def _build_registry(self, skills, client, agent_loader) -> ToolRegistry:
        cfg = self.config
        registry = ToolRegistry()
        for tool_cls in (BashTool, FileReadTool, FileWriteTool, FileEditTool,
                         WebFetchTool, WebSearchTool, TodoWriteTool):
            registry.register(tool_cls())
        registry.register(SkillTool(skills))

        sub = ToolRegistry()
        for tool_cls in (BashTool, FileReadTool, FileWriteTool, FileEditTool,
                         WebFetchTool, WebSearchTool, TodoWriteTool):
            sub.register(tool_cls())

        registry.register(SubagentTool(
            client=client, model=cfg.model, registry=sub,
            token_tracker=self.tracker, agent_loader=agent_loader,
            system_prompt=(
                "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
                "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
            ),
            max_turns=cfg.subagent_max_turns, sub_model=cfg.subagent_model,
        ))
        return registry
