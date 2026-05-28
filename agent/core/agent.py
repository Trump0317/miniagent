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
from ..tools.subagent import AgentLoader
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
        self._skills = SkillsLoader(skill_directory=cfg.skills_dir)
        self._agent_loader = AgentLoader(cfg.root / "agent" / "subagent")
        self.prompt_loader = PromptLoader(cfg.root / "agent" / "prompts")

        # ── 系统提示词 + 会话恢复 ──
        system_prompt = self._build_system_prompt(self._skills, self._agent_loader)
        self.memory.append_history({"role": "system", "content": system_prompt})
        if cfg.restore_session:
            old = self.memory.restore_history()
            if old:
                # 用 persist=False 避免将已持久化的消息重复写入 JSONL
                for msg in old:
                    self.memory.append_history(msg, persist=False)
                # 注入前导上下文，让 LLM 知道这是历史记录而非待回答的问题
                self.memory.history.insert(
                    1,  # 紧跟 system 消息之后
                    {"role": "user", "content": "[系统] 以下是上次会话的对话记录（仅作上下文参考，不需要回复其中内容）"}
                )
                print(f"[Agent] 已恢复 {len(old)} 条历史消息", flush=True)

        # ── 工具注册 ──
        registry = self._build_registry(self._skills, client, self._agent_loader)

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
        # 无论是否触发压缩，始终同步 JSONL 到当前 history 状态
        self._sync_history_file()
        return {"token_stats": stats, "compact": compact_result}

    # ── 内部 ──

    def _add_message(self, role: str, content: str) -> None:
        self.memory.append_history({"role": role, "content": content})

    def _should_compact(self) -> bool:
        return self.tracker.should_compact(
            self.config.max_context, self.config.compact_threshold
        )

    def _do_compact(self) -> dict:
        """执行压缩：Compactor 提取 → Memory 写入 → 截断历史。"""
        orig_len = len(self.history)
        non_system = [m for m in self.history if m.get("role") != "system"]
        if len(non_system) < 4:
            return {"summary": {}, "preferences": [], "facts": []}

        data = self.compactor.compact(self.history)

        summary = data.get("summary", {})
        if any(summary.values()):
            self.memory.append_summary(
                summary.get("critical", "无"),
                summary.get("decision", "无"),
                summary.get("issue", "无"),
            )

        for p in data.get("preferences", []):
            self.memory.add_user(p)  # add_user 已内置去重

        # 去重：add_memory 内置去重，只统计新增数量
        new_facts = sum(
            1 for f in data.get("facts", [])
            if self.memory.add_memory(f.strip())
        )

        # ★ 关键：截断 self.history，释放上下文
        self._trim_history(data)

        # 同步 JSONL 文件（只保留截断后的 assistant/tool 消息）
        self._sync_history_file()

        compacted = bool(summary or data.get("preferences") or new_facts)
        if compacted:
            usage = self.compactor._last_usage
            cost = f"压缩消耗 {usage.get('input', 0)}+{usage.get('output', 0)} tokens" if usage else ""
            print(
                f"[Memory] 自动压缩: {orig_len} 条 → {len(self.history)} 条"
                + (f" (新增 {new_facts} 条事实)" if new_facts else "")
                + (f" | {cost}" if cost else ""),
                flush=True,
            )

        return data

    def _trim_history(self, data: dict) -> None:
        """将 self.history 截断为：系统提示词（含压缩摘要）+ 最近 2 轮用户对话。"""
        updated_system = self._rebuild_system_prompt_with_summary(data)

        non_system = [m for m in self.history if m.get("role") != "system"]
        # 按 user 消息分段，保留最后 2 个完整用户轮次
        user_indices = [i for i, m in enumerate(non_system) if m.get("role") == "user"]
        if len(user_indices) >= 2:
            recent = non_system[user_indices[-2]:]
        else:
            recent = non_system[-8:]

        # 确保以 user 消息开头（符合 API 消息交替要求）
        while recent and recent[0].get("role") in ("tool", "assistant"):
            recent.pop(0)

        new_history: list[dict] = [
            {"role": "system", "content": updated_system}
        ] + recent

        self.history.clear()
        self.history.extend(new_history)

    def _rebuild_system_prompt_with_summary(self, compaction_data: dict) -> str:
        """重建系统提示词：静态部分不变，动态部分（记忆/偏好）用最新内容。"""
        commands = self.prompt_loader.list_commands()
        parts = ["你是一个智能助手，可以使用各种工具来帮助用户完成任务。"]
        if self.config.context_files:
            parts.append(f"### 项目上下文\n{self.config.context_files}")
        parts.append(f"### 可用技能列表\n{self._skills.get_description()}")
        parts.append(f"### 可用子代理\n{self._agent_loader.list_agents()}")
        parts.append(f"### 可用命令\n{commands or '（无）'}")
        parts.append(f"### 长期记忆（最近摘要）\n{self.memory.brief_context()}")
        parts.append(
            f"### 用户偏好（USER.md）\n"
            + ("\n".join(self.memory.user_preferences()) or "（当前没有用户偏好）")
        )

        # 附加本次压缩摘要
        summary = compaction_data.get("summary", {})
        if any(summary.values()):
            lines = ["### 会话历史摘要（之前对话的压缩记录）"]
            if summary.get("critical"):
                lines.append(f"- 关键事件: {summary['critical']}")
            if summary.get("decision"):
                lines.append(f"- 决策/产出: {summary['decision']}")
            if summary.get("issue"):
                lines.append(f"- 问题: {summary['issue']}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _sync_history_file(self) -> None:
        """将 JSONL 历史文件与当前 self.history 同步（只保留 assistant/tool 消息）。"""
        entries = [
            m for m in self.history
            if m.get("role") in ("assistant", "tool")
        ]
        if entries:
            self.memory._rewrite_history(entries)

    def _setup_events(self) -> None:
        """注册核心事件监听器。在每轮 LLM 调用前检查是否需要压缩。"""
        @self.bus.on("context:high")
        def _on_context_high(event):
            self._do_compact()

        @self.bus.on("turn:start")
        def _on_turn_start(event):
            if self._should_compact():
                self.bus.emit("context:high", {
                    "input_tokens": self.tracker.last_input_tokens(),
                    "threshold": self.config.compact_threshold,
                })

        @self.bus.on("history:appended")
        def _on_history_appended(event):
            msg = event.data.get("message") if event.data else None
            if msg and isinstance(msg, dict):
                self.memory.persist_message(msg)

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
