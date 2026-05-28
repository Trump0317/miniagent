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
from .compaction import CompactionService
from .system_prompt import SystemPrompt
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
# 工具注册表工厂
# ═══════════════════════════════════════════════════════════════


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
        self.memory = AgentMemory(memory_dir=cfg.memory_dir, session_dir=cfg.session_dir)
        self.compactor = Compactor(client=client, model=cfg.model)
        self.tracker = TokenTracker(log_file=cfg.session_dir / "tokens.jsonl")

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
