"""Agent 核心 —— 组装组件，提供 process() 入口。

依赖关系:
    Agent
    ├── EventBus         ← 全局事件总线
    ├── LLMClient        ← 封装 LLM 调用
    ├── ToolExecutor     ← 工具执行调度
    ├── AgentRunner      ← think-act 编排
    ├── Conversation     ← 会话状态
    ├── AgentMemory      ← 持久化存储
    └── TokenTracker     ← 用量统计
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator
from .config import AppConfig
from .conversation import Conversation
from .memory import AgentMemory
from .tokentracker import TokenTracker
from .llm import LLMClient
from .runner import AgentRunner
from .events import EventBus
from .tools import (
    ToolRegistry, BashTool, FileReadTool, FileWriteTool, FileEditTool,
    WebFetchTool, WebSearchTool, SkillTool, SkillsLoader, TodoWriteTool, SubagentTool,
)
from .tools.executor import ToolExecutor
from .tools.SubagentTool.loader import AgentLoader
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

        # ── 技能 ──
        skills = SkillsLoader(skill_directory=cfg.skills_dir)

        # ── Agent 定义 ──
        agent_loader = AgentLoader(cfg.root / "agent" / "subagent")

        # ── Prompt 模板 ──
        self.prompt_loader = PromptLoader(cfg.root / "agent" / "prompts")

        # ── 记忆系统 ──
        memory = AgentMemory(
            memory_dir=cfg.memory_dir, client=client, model=cfg.model
        )
        tracker = TokenTracker(log_file=Path(cfg.memory_dir) / "tokens.jsonl")

        # ── 系统提示词 ──
        system_prompt = self._build_system_prompt(
            skills, memory, agent_loader, self.prompt_loader,
        )

        # ── 对话状态 ──
        self.conversation = Conversation(
            memory=memory,
            token_tracker=tracker,
            system_prompt=system_prompt,
            max_context=cfg.max_context,
            compact_threshold=cfg.compact_threshold,
            restore=cfg.restore_session,
        )

        # ── 工具注册 ──
        registry = self._build_registry(skills, client, agent_loader)

        # ── LLM 客户端 ──
        llm = LLMClient(
            client=client,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            thinking=thinking,
        )

        # ── 工具执行器 ──
        executor = ToolExecutor(registry=registry, event_bus=self.bus)

        # ── 执行引擎 ──
        self.runner = AgentRunner(
            llm_client=llm,
            tool_executor=executor,
            conversation=self.conversation,
            token_tracker=tracker,
            event_bus=self.bus,
            max_turns=cfg.max_turns,
        )

        # ── 事件监听：上下文压力 ──
        @self.bus.on("context:high")
        def _on_context_high(event):
            print(f"\n[Memory] 上下文用量接近上限 (event from {event.source})",
                  flush=True)

    # ── 公共 API ──

    def process(self, message: str) -> Generator[str, None, None]:
        """处理用户消息，流式产出文本。"""
        self.bus.emit("message:received", {"message": message}, source="agent")
        self.conversation.add_user_message(message)
        yield from self.runner.step(self.conversation.history)

    def shutdown(self) -> dict:
        """关闭会话，返回统计。"""
        self.bus.emit("session:end", {}, source="agent")
        stats = self.conversation.token_stats()
        compact_result = self.conversation.compact()
        return {"token_stats": stats, "compact": compact_result}

    # ── 内建方法 ──

    def _build_system_prompt(
        self,
        skills: SkillsLoader,
        memory: AgentMemory,
        agent_loader: AgentLoader,
        prompt_loader: PromptLoader,
    ) -> str:
        commands = prompt_loader.list_commands()
        context_files = self.config.context_files
        parts = [
            "你是一个智能助手，可以使用各种工具来帮助用户完成任务。",
        ]
        if context_files:
            parts.append(f"### 项目上下文\n{context_files}")
        parts.append(f"### 可用技能列表\n{skills.get_description()}")
        parts.append(f"### 可用子代理\n{agent_loader.list_agents()}")
        parts.append(f"### 可用命令\n{commands or '（无）'}")
        parts.append(f"### 长期记忆（最近摘要）\n{memory.brief_context()}")
        parts.append(
            f"### 用户偏好（USER.md）\n"
            + ("\n".join(memory.user_preferences()) or "（当前没有用户偏好）")
        )
        return "\n\n".join(parts)

    def _build_registry(self, skills: SkillsLoader, client, agent_loader: AgentLoader) -> ToolRegistry:
        cfg = self.config

        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(FileEditTool())
        registry.register(WebFetchTool())
        registry.register(WebSearchTool())
        registry.register(SkillTool(skills))
        registry.register(TodoWriteTool())

        sub = ToolRegistry()
        sub.register(BashTool())
        sub.register(FileReadTool())
        sub.register(FileWriteTool())
        sub.register(FileEditTool())
        sub.register(WebFetchTool())
        sub.register(WebSearchTool())
        sub.register(TodoWriteTool())

        registry.register(SubagentTool(
            client=client,
            model=cfg.model,
            registry=sub,
            token_tracker=self.conversation.token_tracker,
            agent_loader=agent_loader,
            system_prompt=(
                "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
                "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
            ),
            max_turns=cfg.subagent_max_turns,
            sub_model=cfg.subagent_model,
        ))

        return registry
