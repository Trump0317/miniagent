"""Agent 核心 —— 组装配置、记忆、工具和运行器。

这是一个纯粹的库，不包含任何 I/O 或交互逻辑。
可以嵌入任意外壳（CLI、TUI、RPC、HTTP）中使用。
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator
from .config import AppConfig
from .conversation import Conversation
from .runner import AgentRunner
from .memory import AgentMemory
from .tokentracker import TokenTracker
from .tools import (
    ToolRegistry, BashTool, FileReadTool, FileWriteTool, FileEditTool,
    WebFetchTool, WebSearchTool, SkillTool, SkillsLoader, TodoWriteTool, SubagentTool,
)
from .tools.SubagentTool.loader import AgentLoader
from .prompts import PromptLoader


class Agent:
    """Agent 核心引擎 —— 组装组件，提供单一入口 process()。

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

        # ── 技能 ──
        skills = SkillsLoader(skill_directory=cfg.skills_dir)

        # ── Agent 定义 ──
        agent_loader = AgentLoader(cfg.root / "agent" / "subagent")

        # ── Prompt 模板（暴露给外壳做 /command 展开）──
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

        # ── 执行引擎 ──
        self.runner = AgentRunner(
            client=client,
            model=cfg.model,
            tool_registry=registry,
            conversation=self.conversation,
            max_turns=cfg.max_turns,
            max_tokens=cfg.max_tokens,
            thinking=thinking,
        )

    # ── 公共 API ──

    def process(self, message: str) -> Generator[str, None, None]:
        """处理一条用户消息，流式产出 LLM 响应文本。

        外壳负责：收集、打印、格式化这些文本块。
        """
        self.conversation.add_user_message(message)
        yield from self.runner.step(self.conversation.history)

    def shutdown(self) -> dict:
        """关闭会话：返回 token 统计和压缩结果。"""
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
            agent_loader=agent_loader,
            system_prompt=(
                "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
                "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
            ),
            max_turns=cfg.subagent_max_turns,
            sub_model=cfg.subagent_model,
        ))

        return registry
