"""Agent 核心 —— 组装组件，提供 process() 入口。

依赖:
    Agent
    ├── EventBus         ← 事件总线
    ├── AgentMemory      ← 纯存储（文件 I/O）
    ├── Compactor        ← 记忆压缩（LLM 提取）
    ├── TokenTracker     ← 用量统计
    ├── SystemPrompt     ← 系统提示词构建
    ├── CompactionService ← 压缩编排
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


# ═══════════════════════════════════════════════════════════════
# 内部组件
# ═══════════════════════════════════════════════════════════════

class SystemPrompt:
    """系统提示词构建器。

    持有静态上下文引用，build() 时实时查询 memory 的动态部分。
    支持可选的 compaction_data 注入压缩摘要。
    """

    def __init__(
        self,
        context_files: str,
        skills: SkillsLoader,
        agent_loader: AgentLoader,
        prompt_loader: PromptLoader,
        memory: AgentMemory,
    ):
        self._ctx = context_files
        self._skills = skills
        self._agents = agent_loader
        self._commands = prompt_loader
        self._memory = memory

    def build(self, compaction_data: dict | None = None) -> str:
        """构建完整系统提示词。compaction_data 不为 None 时追加压缩摘要段落。"""
        parts = ["你是一个智能助手，可以使用各种工具来帮助用户完成任务。"]

        if self._ctx:
            parts.append(f"### 项目上下文\n{self._ctx}")

        parts.append(f"### 可用技能列表\n{self._skills.get_description()}")
        parts.append(f"### 可用子代理\n{self._agents.list_agents()}")
        parts.append(f"### 可用命令\n{self._commands.list_commands() or '（无）'}")
        parts.append(f"### 长期记忆（最近摘要）\n{self._memory.brief_context()}")
        parts.append(
            f"### 用户偏好（USER.md）\n"
            + ("\n".join(self._memory.user_preferences()) or "（当前没有用户偏好）")
        )

        if compaction_data:
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


class CompactionService:
    """压缩编排：Compactor 提取 → Memory 分发 → 树压缩 → 重建系统提示词。

    不持有状态（每次 compact() 都是全新执行），只依赖注入的组件。
    """

    def __init__(
        self,
        compactor: Compactor,
        memory: AgentMemory,
        tracker: TokenTracker,
        max_context: int,
        compact_threshold: float,
        prompt: SystemPrompt,
    ):
        self._compactor = compactor
        self._memory = memory
        self._tracker = tracker
        self._max_context = max_context
        self._threshold = compact_threshold
        self._prompt = prompt

    def should_compact(self) -> bool:
        """Token 是否超过压缩阈值。"""
        return self._tracker.should_compact(self._max_context, self._threshold)

    def compact(self) -> tuple[str, dict]:
        """执行一次完整压缩。返回 (新系统提示词, 压缩数据)。"""
        non_system = self._memory.non_system_entries()
        if len(non_system) < 4:
            return self._prompt.build(), {"summary": {}, "preferences": [], "facts": []}

        orig_len = len(non_system)
        data = self._compactor.compact(non_system)

        # ── 1. 分发到三层记忆 ──
        summary = data.get("summary", {})
        if any(summary.values()):
            self._memory.append_summary(
                summary.get("critical", "无"),
                summary.get("decision", "无"),
                summary.get("issue", "无"),
            )

        for p in data.get("preferences", []):
            self._memory.add_user(p)

        new_facts = sum(
            1 for f in data.get("facts", [])
            if self._memory.add_memory(f.strip())
        )

        # ── 2. 树压缩 ──
        tree_entries = self._memory._tree.non_system_entries()
        user_entries = [e for e in tree_entries if e.role == "user"]
        if len(user_entries) >= 2:
            first_kept = user_entries[-2]
        else:
            first_kept = tree_entries[-min(8, len(tree_entries))]

        summary_text = self._format_summary(data)
        self._memory.compress_tree(summary_text, first_kept.id,
                                   self._tracker.last_input_tokens())

        # ── 3. 重建系统提示词 ──
        new_prompt = self._prompt.build(data)

        # ── 4. 日志 ──
        compacted = bool(summary or data.get("preferences") or new_facts)
        if compacted:
            usage = self._compactor._last_usage
            cost = (
                f"压缩消耗 {usage.get('input', 0)}+{usage.get('output', 0)} tokens"
                if usage else ""
            )
            print(
                f"[Memory] 树压缩: {orig_len} 条 → {len(self._memory.history)} 条"
                + (f" (新增 {new_facts} 条事实)" if new_facts else "")
                + (f" | {cost}" if cost else ""),
                flush=True,
            )

        return new_prompt, data

    @staticmethod
    def _format_summary(data: dict) -> str:
        summary = data.get("summary", {})
        parts = []
        if summary.get("critical"):
            parts.append(f"关键事件: {summary['critical']}")
        if summary.get("decision"):
            parts.append(f"决策/产出: {summary['decision']}")
        if summary.get("issue"):
            parts.append(f"问题: {summary['issue']}")
        return "\n".join(parts) if parts else "会话已压缩"


def _build_registry(
    cfg: AppConfig,
    skills: SkillsLoader,
    client,
    agent_loader: AgentLoader,
    tracker: TokenTracker,
) -> ToolRegistry:
    """构建 Agent 的工具注册表（含子代理）。"""
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
        token_tracker=tracker, agent_loader=agent_loader,
        system_prompt=(
            "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
            "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
        ),
        max_turns=cfg.subagent_max_turns, sub_model=cfg.subagent_model,
    ))
    return registry


# ═══════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════

class Agent:
    """Agent 核心引擎 —— 组装组件，编排生命周期。

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
        client = cfg.create_client()

        # ── 基础设施 ──
        self.bus = EventBus()
        self.memory = AgentMemory(memory_dir=cfg.memory_dir)
        self.compactor = Compactor(client=client, model=cfg.model)
        self.tracker = TokenTracker(log_file=Path(cfg.memory_dir) / "tokens.jsonl")

        # ── 技能 / 子代理 / 命令 ──
        skills = SkillsLoader(skill_directory=cfg.skills_dir)
        agent_loader = AgentLoader(cfg.root / "agent" / "subagent")
        self.prompt_loader = PromptLoader(cfg.root / "agent" / "prompts")

        # ── 系统提示词构建器 ──
        self._prompt = SystemPrompt(
            context_files=cfg.context_files,
            skills=skills,
            agent_loader=agent_loader,
            prompt_loader=self.prompt_loader,
            memory=self.memory,
        )
        # 暴露给 cli_helpers（fork 时需要 skills/agent_loader 重建提示词）
        self._skills = skills
        self._agent_loader = agent_loader

        # ── 压缩编排 ──
        self._compaction = CompactionService(
            compactor=self.compactor,
            memory=self.memory,
            tracker=self.tracker,
            max_context=cfg.max_context,
            compact_threshold=cfg.compact_threshold,
            prompt=self._prompt,
        )

        # ── 会话恢复 ──
        restored = self.memory.restore_tree() if cfg.restore_session else False
        self._system_prompt = self._prompt.build()
        self._has_restored_context = restored
        if restored:
            non_sys = [m for m in self.memory.history if m.get("role") != "system"]
            print(f"[Agent] 已恢复会话（{len(non_sys)} 条上下文消息）", flush=True)

        # ── 工具注册 ──
        registry = _build_registry(cfg, skills, client, agent_loader, self.tracker)

        # ── 运行器 ──
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

        # ── 事件 ──
        self._setup_events()

    # ── 公共 API ──

    def process(self, message: str) -> Generator[str, None, None]:
        """处理用户消息，流式产出 LLM 响应文本。"""
        self.bus.emit("message:received", {"text": message})
        self._add_message("user", message)

        context = self._build_context()
        snapshot_len = len(context)

        yield from self.runner.step(context)

        for msg in context[snapshot_len:]:
            self.memory.append_message(msg)

    def shutdown(self) -> dict:
        """关闭会话：统计 + 压缩。"""
        self.bus.emit("session:end", {})
        stats = self.tracker.stats_by_model()
        self._system_prompt, compact_result = self._compaction.compact()
        return {"token_stats": stats, "compact": compact_result}

    # ── 内部 ──

    def _build_context(self) -> list[dict]:
        """构建给 Runner 的上下文：[system] + 可选恢复分隔 + 树历史。"""
        context = [{"role": "system", "content": self._system_prompt}]
        if self._has_restored_context:
            context.append({
                "role": "user",
                "content": "[系统] 以下是上次会话的对话记录（仅作上下文参考，不需要回复其中内容）"
            })
            self._has_restored_context = False
        context.extend(self.memory.history)
        return context

    def _add_message(self, role: str, content: str) -> None:
        self.memory.append_message({"role": role, "content": content})

    def _setup_events(self) -> None:
        """注册事件：每轮 LLM 调用前检查是否需要压缩。"""
        @self.bus.on("context:high")
        def _on_context_high(event):
            self._system_prompt, _ = self._compaction.compact()

        @self.bus.on("turn:start")
        def _on_turn_start(event):
            if self._compaction.should_compact():
                self.bus.emit("context:high", {
                    "input_tokens": self.tracker.last_input_tokens(),
                    "threshold": self.config.compact_threshold,
                })
