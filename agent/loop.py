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
from .tools.SubagentTool.loader import AgentLoader
from .prompts import PromptLoader


class Agent:
    """Agent 应用 —— 将配置、状态和执行引擎组装在一起。"""

    def __init__(
        self,
        config: AppConfig | None = None,
        thinking: str | None = None,
        context_files: str = "",
    ):
        self.config = config or AppConfig.from_env()
        cfg = self.config
        root = cfg.root
        client = cfg.create_client()

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
            skills, memory, agent_loader, self.prompt_loader, context_files
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

    # ── 构造方法 ──

    def _build_system_prompt(
        self,
        skills: SkillsLoader,
        memory: AgentMemory,
        agent_loader: AgentLoader,
        prompt_loader: PromptLoader,
        context_files: str = "",
    ) -> str:
        commands = prompt_loader.list_commands()
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

    # ── 运行入口 ──

    def run(self, initial_message: str = "", print_mode: bool = False) -> None:
        """启动 agent。

        print_mode=True: 处理 initial_message 后打印结果并退出。
        print_mode=False: 进入交互式循环。
        """
        # 启动时打印加载的资源摘要
        self._print_startup_info()

        if print_mode:
            self._run_print_mode(initial_message)
        else:
            self._run_interactive(initial_message)

    def _run_print_mode(self, msg: str) -> None:
        """非交互模式：处理一条消息，输出结果后退出。"""
        msg = self._expand_command(msg)
        self.conversation.add_user_message(msg)

        for chunk in self.runner.step(self.conversation.history):
            print(chunk, end="", flush=True)
        print()

        self._shutdown()

    def _run_interactive(self, initial_message: str = "") -> None:
        """交互式主循环。支持 /command 快捷调用 prompt 模板。"""
        # 启动时展示可用命令
        cmds = self.prompt_loader.list_commands()
        if cmds:
            print(cmds)

        # 如果有初始消息，先处理
        if initial_message:
            msg = self._expand_command(initial_message)
            self.conversation.add_user_message(msg)
            print(f"[You] : {initial_message}")
            print("[Assistant] : ", end="", flush=True)
            for chunk in self.runner.step(self.conversation.history):
                print(chunk, end="", flush=True)
            print("\n")

        while True:
            user_input = input("[You] : ")
            command = user_input.strip()
            if command.lower() in {"exit", "quit"}:
                self._shutdown()
                break

            msg = self._expand_command(command)
            self.conversation.add_user_message(msg)

            print("[Assistant] : ", end="", flush=True)
            for chunk in self.runner.step(self.conversation.history):
                print(chunk, end="", flush=True)
            print("\n")

    def _expand_command(self, user_input: str) -> str:
        """将 /command query 展开为 prompt 模板。普通输入原样返回。"""
        if not user_input.startswith("/"):
            return user_input

        parts = user_input.split(maxsplit=1)
        name = parts[0][1:]  # 去 /
        query = parts[1] if len(parts) > 1 else ""

        resolved = self.prompt_loader.resolve(name, query)
        if resolved:
            print(f"[模板 /{name}] → {resolved[:60]}{'...' if len(resolved) > 60 else ''}")
            return resolved

        return user_input

    def _print_startup_info(self) -> None:
        """启动摘要：上下文文件、思考级别等"""
        lines = [f"[miniagent] 模型: {self.config.model}"]
        if self.runner.thinking:
            lines.append(f"  思考级别: {self.runner.thinking}")
        # 从系统提示词中提取上下文文件信息
        sys_msg = self.conversation.history[0]["content"] if self.conversation.history else ""
        if "### 项目上下文" in sys_msg:
            lines.append("  上下文文件: 已加载 (AGENTS.md)")
        print("\n".join(lines))

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
