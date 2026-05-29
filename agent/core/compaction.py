"""压缩编排 —— 找切点 → LLM 提取 → 分发 → 树压缩 → 重建提示词。

单一模块覆盖完整压缩流程:
    1. _find_cut_point()     → token-aware 切点（沿用户消息边界，保留完整 turns）
    2. _extract()            → LLM 提取 summary / preferences / facts
    3. _dispatch_to_memory() → 分发到三层记忆（user.md / memory.md / summaries/）
    4. Memory.compress_tree()→ 插入 COMPACT 节点 + JSONL 持久化
    5. SystemPrompt.build()  → 重建系统提示词

不持有状态，每次 compact() 全新执行。
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import json
from openai import OpenAI

if TYPE_CHECKING:
    from .memory import AgentMemory
    from .tracker import TokenTracker
    from .system_prompt import SystemPrompt


class CompactionService:
    """压缩编排器 —— 合并了原 Compactor 的 LLM 提取逻辑。

    对外:
        should_compact() → bool          是否触发压缩
        compact() → (system_prompt, data) 执行完整压缩
    """

    # ── LLM 提取提示词 ──
    _EXTRACT_PROMPT = (
        "你是一个记忆提取专家。请分析以下对话，并提取关键信息。\n"
        "输出必须是严格的 JSON 格式，包含以下字段：\n"
        "1. summary: 对象，包含 critical(关键事件), decision(决策/产出), issue(心得/问题)。\n"
        "   - 要求：每项极简，总字数 < 150 字。\n"
        "2. preferences: 字符串列表。记录用户明确表达的偏好、习惯或要求。\n"
        "3. facts: 字符串列表。记录值得长期记住的核心事实。\n"
        "注意：\n"
        "- 如果没有相关信息，对应的列表应为空，字段不能缺失。\n"
        "- **不要**提取系统已知的静态信息（如工具数量、模型名称、框架名称等），这些不是需要记住的'事实'。\n"
        "- 只提取本次对话中**首次出现**的、对后续任务有指导意义的信息。\n"
        "- 偏好和事实都要尽量简短，每条 < 30 字。"
    )

    # ── 可调参数 ──
    _RESERVE_TOKENS = 16384        # 为 LLM 响应预留的 token 空间
    _KEEP_RECENT_RATIO = 0.15      # 保留最近 15% 的上下文不压缩
    _CHAR_PER_TOKEN = 3            # 粗略 token 估算（中文 ~1.5, 英文 ~4，取中）
    _COMPACT_K = 10                # 最多取最近 k 条消息送给 LLM 提取

    def __init__(
        self,
        client: OpenAI,
        model: str,
        memory: "AgentMemory",
        tracker: "TokenTracker",
        max_context: int,
        compact_threshold: float,
        prompt: "SystemPrompt",
    ):
        self._client = client
        self._model = model
        self._memory = memory
        self._tracker = tracker
        self._max_context = max_context
        self._threshold = compact_threshold
        self._prompt = prompt
        self._keep_recent = int(max_context * self._KEEP_RECENT_RATIO)

        # 最近一次 LLM 提取的 token 用量（供日志使用）
        self._last_usage: dict[str, int] = {}

    # ── 公共 API ──

    def should_compact(self) -> bool:
        """Token 是否超过压缩阈值。"""
        return self._tracker.should_compact(self._max_context, self._threshold)

    def compact(self) -> tuple[str, dict]:
        """执行一次完整压缩。返回 (新系统提示词, 压缩数据)。"""
        entries = self._memory.non_system_entries()
        if len(entries) < 4:
            return self._prompt.build(), {"summary": {}, "preferences": [], "facts": []}

        # ── 步骤 1: token-aware 切点 ──
        first_kept_id, messages_to_summarize = self._find_cut_point()
        if first_kept_id is None or len(messages_to_summarize) < 2:
            return self._prompt.build(), {"summary": {}, "preferences": [], "facts": []}

        orig_len = len(entries)

        # ── 步骤 2: LLM 提取（原 Compactor 逻辑）──
        data = self._extract(messages_to_summarize)

        # ── 步骤 3: 分发到三层记忆 ──
        new_facts = self._dispatch_to_memory(data)

        # ── 步骤 4: 树压缩 ──
        summary_text = self._build_compaction_summary(data)
        tokens_before = self._tracker.last_input_tokens()
        self._memory.compress_tree(summary_text, first_kept_id, tokens_before)

        # ── 步骤 5: 重建系统提示词 ──
        new_prompt = self._prompt.build(data)

        # ── 步骤 6: 日志 ──
        self._log(orig_len, new_facts, data)

        return new_prompt, data

    # ── 内部：切点计算 ──

    def _find_cut_point(self) -> tuple[str | None, list[dict]]:
        """Token-aware 切点：从 leaf 往回走，在用户消息边界切割。

        规则:
          - 沿 path 从后往前累积估算 token
          - 遇到用户消息时检查是否超过 keep_recent 阈值
          - 超过则该用户消息作为 first_kept（保留完整 turn）
          - 该用户消息之前的所有消息作为压缩输入

        返回 (first_kept_id, messages_to_summarize)。
        """
        path = self._memory._tree.path()
        msg_entries = [e for e in path if e.type == "message"]
        if len(msg_entries) < 4:
            return None, []

        accumulated = 0
        cut_index = len(msg_entries)

        for i in range(len(msg_entries) - 1, -1, -1):
            entry = msg_entries[i]
            estimated = max(len(entry.content or "") // self._CHAR_PER_TOKEN, 1)
            accumulated += estimated

            if entry.role == "user" and accumulated >= self._keep_recent:
                cut_index = i
                break

        # 兜底：阈值太高时保留最后 2 条用户消息
        if cut_index >= len(msg_entries):
            user_entries = [e for e in msg_entries if e.role == "user"]
            if len(user_entries) >= 2:
                first_kept = user_entries[-2]
            else:
                first_kept = msg_entries[-min(4, len(msg_entries))]
            cut_index = msg_entries.index(first_kept)

        first_kept = msg_entries[cut_index]
        to_summarize = [
            {"role": e.role, "content": e.content}
            for e in msg_entries[:cut_index]
        ]

        return first_kept.id, to_summarize

    # ── 内部：LLM 提取 ──

    def _extract(self, history: list[dict]) -> dict[str, Any]:
        """调用 LLM 从对话历史中提取摘要、偏好和事实。

        如果树中已有之前的压缩节点，将其摘要作为迭代上下文传入，
        让 LLM 在已有基础上增量更新，而非每次从头提取。
        """
        effective = [m for m in history if m.get("role") != "system"]
        if len(effective) < 2:
            return {"summary": {}, "preferences": [], "facts": []}

        recent = effective[-self._COMPACT_K:]
        self._last_usage = {}

        # ── 迭代压缩：获取上一次压缩摘要作为上下文 ──
        previous = self._get_previous_summary()
        user_parts = []
        if previous:
            user_parts.append(
                "<previous_summary>\n"
                + previous
                + "\n</previous_summary>\n\n"
                + "以上是之前对话的压缩摘要。以下是新的对话内容，"
                + "请在上次摘要的基础上更新/补充提取信息："
            )
        user_parts.append(self._format_messages(recent))

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._EXTRACT_PROMPT},
                    {"role": "user", "content": "\n".join(user_parts)},
                ],
                response_format={"type": "json_object"},
            )
            if hasattr(response, "usage") and response.usage:
                self._last_usage = {
                    "input": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "output": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            data = json.loads(response.choices[0].message.content or "{}")
        except Exception as e:
            print(f"[Compaction] 提取失败: {e}", flush=True)
            return {
                "summary": {"critical": f"提取失败: {e}", "decision": "无", "issue": "无"},
                "preferences": [],
                "facts": [],
            }

        return self._normalize(data)

    def _get_previous_summary(self) -> str | None:
        """从当前树路径中获取最近一次压缩节点的摘要。

        遍历 path（root → leaf），找到最后一个 compaction entry。
        这是最近一次压缩的结果，用作迭代压缩的上下文。
        """
        path = self._memory._tree.path()
        for entry in reversed(path):
            if entry.type == "compaction" and entry.summary:
                return entry.summary
        return None

    @staticmethod
    def _format_messages(history: list[dict]) -> str:
        """将历史消息格式化为 LLM 可读文本。"""
        lines: list[str] = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            if role == "assistant" and msg.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?")
                         for tc in msg["tool_calls"]]
                content = f"[调用工具: {', '.join(names)}] {content}".strip()

            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _normalize(data: dict) -> dict:
        """规范化 LLM 返回，确保字段存在且值合理。"""
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        for key in ("critical", "decision", "issue"):
            val = summary.get(key, "无")
            summary[key] = val[:60] if isinstance(val, str) else "无"

        preferences = data.get("preferences", [])
        if not isinstance(preferences, list):
            preferences = []

        facts = data.get("facts", [])
        if not isinstance(facts, list):
            facts = []

        return {"summary": summary, "preferences": preferences, "facts": facts}

    # ── 内部：记忆分发 ──

    def _dispatch_to_memory(self, data: dict) -> int:
        """分发 LLM 提取结果到三层记忆。返回新增事实条数。"""
        summary = data.get("summary", {})
        if any(summary.values()):
            self._memory.append_summary(
                summary.get("critical", "无"),
                summary.get("decision", "无"),
                summary.get("issue", "无"),
            )

        for p in data.get("preferences", []):
            self._memory.add_user(p)

        return sum(
            1 for f in data.get("facts", [])
            if self._memory.add_memory(f.strip())
        )

    # ── 内部：摘要文本 ──

    def _build_compaction_summary(self, data: dict) -> str:
        """构建 COMPACT 节点的统一摘要文本。"""
        summary = data.get("summary", {})
        parts = []

        if summary.get("critical"):
            parts.append(f"关键事件: {summary['critical']}")
        if summary.get("decision"):
            parts.append(f"决策/产出: {summary['decision']}")
        if summary.get("issue"):
            parts.append(f"问题: {summary['issue']}")

        prefs = data.get("preferences", [])
        if prefs:
            parts.append(f"用户偏好: {'; '.join(prefs[:5])}")

        facts = data.get("facts", [])
        if facts:
            parts.append(f"事实: {'; '.join(facts[:5])}")

        return "\n".join(parts) if parts else "会话已压缩"

    # ── 内部：日志 ──

    def _log(self, orig_len: int, new_facts: int, data: dict) -> None:
        """统一日志输出。"""
        summary = data.get("summary", {})
        facts = data.get("facts", [])
        has_content = bool(
            summary.get("critical") or summary.get("decision") or summary.get("issue")
            or data.get("preferences") or facts
        )

        if not has_content:
            return

        new_len = len(self._memory.history)
        parts = [f"[Memory] 树压缩: {orig_len} 条 → {new_len} 条"]
        if facts:
            parts.append(f"(新增 {len(facts)} 条事实)")
        if self._last_usage:
            u = self._last_usage
            parts.append(f"| 压缩消耗 {u.get('input', 0)}+{u.get('output', 0)} tokens")

        print(" ".join(parts), flush=True)
