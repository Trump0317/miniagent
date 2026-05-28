"""压缩编排 —— Compactor 提取 → Memory 分发 → 树压缩 → 重建系统提示词。

不持有状态，每次 compact() 全新执行。只依赖注入的组件。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import AgentMemory
    from .compactor import Compactor
    from .tracker import TokenTracker
    from .system_prompt import SystemPrompt


class CompactionService:
    """编排一次完整压缩流程。

    流程:
        1. Compactor.compact() → LLM 提取 summary / preferences / facts
        2. Memory 分发 → summaries/ / user.md / memory.md
        3. SessionTree.compact() → 插入 COMPACT 节点 + JSONL 全量持久化
        4. SystemPrompt.build(data) → 重建含压缩摘要的系统提示词
    """

    def __init__(
        self,
        compactor: "Compactor",
        memory: "AgentMemory",
        tracker: "TokenTracker",
        max_context: int,
        compact_threshold: float,
        prompt: "SystemPrompt",
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
