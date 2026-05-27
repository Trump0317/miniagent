"""Agent 主入口 —— 组装配置、记忆、工具和运行器，启动交互循环。"""

from __future__ import annotations
from pathlib import Path
from .config import AppConfig
from .conversation import Conversation
from .runner import AgentRunner
from .memory import AgentMemory
from .tokentracker import TokenTracker
from .tools import (
    ToolRegistry, BashTool, FileReadTool, FileWriteTool, FileEditTool,
    WebFetchTool, WebSearchTool, SkillTool, SkillsLoader, TodoWriteTool, SubagentTool,
)


class Agent:
    """Agent 应用 —— 将配置、状态和执行引擎组装在一起。"""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.from_env()
        cfg = self.config
        root = cfg.root
        client = cfg.create_client()

        # ── 技能 ──
        skills = SkillsLoader(skill_directory=cfg.skills_dir)

        # ── 记忆系统 ──
        memory = AgentMemory(
            memory_dir=cfg.memory_dir, client=client, model=cfg.model
        )
        tracker = TokenTracker(log_file=Path(cfg.memory_dir) / "tokens.jsonl")

        # ── 系统提示词 ──
        system_prompt = self._build_system_prompt(skills, memory)

        # ── 对话状态 ──
        self.conversation = Conversation(
            memory=memory,
            token_tracker=tracker,
            system_prompt=system_prompt,
            max_context=cfg.max_context,
            compact_threshold=cfg.compact_threshold,
        )

        # ── 工具注册 ──
        registry = self._build_registry(skills, client)

        # ── 执行引擎 ──
        self.runner = AgentRunner(
            client=client,
            model=cfg.model,
            tool_registry=registry,
            conversation=self.conversation,
            max_turns=cfg.max_turns,
            max_tokens=cfg.max_tokens,
        )

    # ── 构造方法 ──

    def _build_system_prompt(self, skills: SkillsLoader, memory: AgentMemory) -> str:
        return f"""
        你是一个智能助手，可以使用各种工具来帮助用户完成任务。
        ### 可用技能列表
        {skills.get_description()}
        ### 长期记忆（最近摘要）
        {memory.brief_context()}
        ### 用户偏好（USER.md）
        {"\n".join(memory.user_preferences()) or "（当前没有用户偏好）"}
        """.strip()

    def _build_registry(self, skills: SkillsLoader, client) -> ToolRegistry:
        cfg = self.config

        # 主工具注册表
        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(FileEditTool())
        registry.register(WebFetchTool())
        registry.register(WebSearchTool())
        registry.register(SkillTool(skills))
        registry.register(TodoWriteTool())

        # 子代理注册表（独立拷贝）
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
            system_prompt=(
                "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
                "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
            ),
            max_turns=cfg.subagent_max_turns,
            sub_model=cfg.subagent_model,
        ))

        return registry

    # ── 主循环 ──

    def run(self) -> None:
        """交互式主循环"""
        while True:
            user_input = input("[You] : ")
            command = user_input.strip()
            if command.lower() in {"exit", "quit"}:
                self._shutdown()
                break

            self.conversation.add_user_message(command)

            print("[Assistant] : ", end="", flush=True)
            for chunk in self.runner.step(self.conversation.history):
                print(chunk, end="", flush=True)
            print("\n")

    def _shutdown(self) -> None:
        """退出前：打印统计、压缩记忆"""
        stats = self.conversation.token_stats()
        if stats:
            print("\n[Tokens] 本次会话 Token 消耗统计:")
            for m, s in stats.items():
                print(f"  - {m}: 输入 {s['input']}, 输出 {s['output']}, 缓存命中 {s['cache_hit']}")

        result = self.conversation.compact()
        if result.get("summary") or result.get("preferences"):
            print("[Memory] 已自动压缩并保存本次会话记录")
        print("退出对话")
