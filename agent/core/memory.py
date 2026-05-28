"""记忆存储 —— 纯文件 I/O，不调用 LLM，不关心压缩逻辑。

三层结构:
  history.jsonl  — 对话历史（JSONL 持久化）
  memory.md      — 核心记忆（追加事实）
  summaries/     — 每日摘要（按日期文件）
  user.md        — 用户偏好（列表）
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, ConfigDict, Field


class ConversationEntry(BaseModel):
    """单条历史记录的持久化格式。"""
    model_config = ConfigDict(extra="forbid")
    created_at: str
    role: str
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMemory:
    """纯存储层 —— 管理对话历史、摘要、偏好和长期记忆的文件读写。

    LLM 压缩逻辑已抽离到 Compactor。
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_file = self.memory_dir / "memory.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.summary_dir = self.memory_dir / "summaries"
        self.user_file = self.memory_dir / "user.md"

        # 内存中的消息历史（Runner 直接操作此列表）
        self.history: list[dict] = []

        # 确保目录和文件存在
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        for f, header in [
            (self.memory_file, "# 长期记忆\n\n此文件常驻上下文，记录核心目标、当前任务与关键事实。\n"),
            (self.user_file, "# 用户信息\n\n此文件记录用户的基本信息和偏好。\n"),
        ]:
            if not f.exists():
                f.write_text(header, encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.write_text("", encoding="utf-8")

    # ── 时间工具 ──

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _today_file(self, when: datetime | None = None) -> Path:
        moment = when or datetime.now()
        return self.summary_dir / f"{moment:%Y-%m-%d}.md"

    # ── 历史读写 ──

    def append_history(self, content: Any, persist: bool = True) -> None:
        """追加消息到内存列表，默认同时持久化到 JSONL。

        persist=False 用于恢复历史（消息已存在于 JSONL，避免重复写入）。
        """
        self.history.append(content)
        if persist:
            self._persist_message(content)

    def persist_message(self, content: Any) -> None:
        """只持久化到 JSONL，不修改内存中的历史列表。

        用于 runner 直接 history.append() 后的异步持久化。
        """
        self._persist_message(content)

    def _persist_message(self, content: Any) -> None:
        if not isinstance(content, dict):
            return
        role = content.get("role")
        if role == "system":
            return  # 系统消息每次启动重建，不持久化

        entry = ConversationEntry(
            created_at=self._now(),
            role=role or "unknown",
            content=content.get("content"),
            metadata={
                "reasoning_content": content.get("reasoning_content"),
                "tool_calls": content.get("tool_calls"),
                "tool_call_id": content.get("tool_call_id"),
            } if role in ("assistant", "tool") else {},
        )
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")

    def restore_history(self, max_messages: int = 50) -> list[dict]:
        """从 history.jsonl 恢复上次会话的 assistant/tool 消息作为上下文延续。

        user 消息不恢复（避免模型误认为是新的待回答查询）。
        """
        if not self.history_file.exists():
            return []
        entries = []
        with self.history_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not (line := line.strip()):
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = entry.get("role", "")
                # 只恢复 assistant 和 tool 消息，跳过 user
                if role == "user":
                    continue

                msg: dict = {"role": role, "content": entry.get("content", "")}
                meta = entry.get("metadata", {})

                if role == "tool" and meta.get("tool_call_id"):
                    msg["tool_call_id"] = meta["tool_call_id"]
                if role == "assistant":
                    if meta.get("reasoning_content"):
                        msg["reasoning_content"] = meta["reasoning_content"]
                    if meta.get("tool_calls"):
                        msg["tool_calls"] = meta["tool_calls"]
                    # 修复历史遗留：content 为 None 且无 tool_calls 的消息无法通过 API 校验
                    if msg.get("content") is None and not msg.get("tool_calls"):
                        msg["content"] = meta.get("reasoning_content") or ""

                entries.append(msg)

        # 限制恢复条数，避免超出模型上下文
        if len(entries) > max_messages:
            print(f"[Memory] 历史过长 ({len(entries)} 条)，截断为最近 {max_messages} 条", flush=True)
            entries = entries[-max_messages:]
            # 确保截断后不以孤立的 tool 消息开头（必须有前置 assistant tool_calls）
            while entries and entries[0].get("role") == "tool":
                entries.pop(0)
            # 重写历史文件，只保留截断后的内容
            self._rewrite_history(entries)

        return entries

    def _rewrite_history(self, entries: list[dict]) -> None:
        """重写 history.jsonl，只保留给定的消息。"""
        now = self._now()
        with self.history_file.open("w", encoding="utf-8") as f:
            for msg in entries:
                role = msg.get("role", "")
                meta = {}
                if role == "tool" and msg.get("tool_call_id"):
                    meta["tool_call_id"] = msg["tool_call_id"]
                if role == "assistant":
                    if msg.get("reasoning_content"):
                        meta["reasoning_content"] = msg["reasoning_content"]
                    if msg.get("tool_calls"):
                        meta["tool_calls"] = msg["tool_calls"]
                entry = {
                    "created_at": now,
                    "role": role,
                    "content": msg.get("content"),
                    "metadata": meta,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── 摘要 ──

    def append_summary(self, critical: str, decision: str, issue: str) -> Path:
        """追加每日摘要条目。"""
        today_file = self._today_file()
        timestamp = datetime.now().strftime("%H:%M:%S")
        mode = "a" if today_file.exists() else "w"
        with today_file.open(mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# 摘要 {today_file.stem}\n\n")
            f.write(f"## {timestamp}\n")
            f.write(f"- **关键**: {critical}\n")
            f.write(f"- **决策**: {decision}\n")
            f.write(f"- **问题**: {issue}\n\n")
        return today_file

    def brief_context(self) -> str:
        """获取系统提示词所需的最近背景。"""
        parts: list[str] = []

        if self.memory_file.exists():
            text = self.memory_file.read_text(encoding="utf-8")
            valid = [l.strip() for l in text.split("\n")
                     if l.strip() and not l.strip().startswith("#")]
            if valid:
                parts.append("## 核心记忆")
                parts.extend(valid[-15:])

        if self.summary_dir.exists():
            summaries = sorted(self.summary_dir.glob("*.md"), reverse=True)
            for s_file in summaries[:3]:
                text = s_file.read_text(encoding="utf-8")
                parts.append(f"### {s_file.stem}")
                parts.extend(l for l in text.split("\n")
                             if not l.strip().startswith("#"))

        return "\n".join(parts) if parts else "（暂无历史背景）"

    # ── 偏好与事实 ──

    def user_preferences(self) -> list[str]:
        """读取用户偏好列表。"""
        if not self.user_file.exists():
            return []
        return [line.strip("- ").strip()
                for line in self.user_file.read_text(encoding="utf-8").split("\n")
                if line.strip().startswith("-")]

    def add_user(self, preference: str) -> None:
        """追加用户偏好（自动去重）。"""
        p = preference.strip()
        if not p:
            return
        existing = self.get_existing_preferences()
        if p in existing:
            return
        with self.user_file.open("a", encoding="utf-8") as f:
            f.write(f"- {p}\n")

    def get_existing_preferences(self) -> set[str]:
        """返回 user.md 中已有的所有偏好文本，用于去重。"""
        prefs: set[str] = set()
        for line in self.user_preferences():
            prefs.add(line)
        return prefs

    def add_memory(self, fact: str) -> bool:
        """追加长期记忆事实（自动去重）。返回 True 表示新增，False 表示已存在。"""
        f = fact.strip()
        if not f:
            return False
        existing = self.get_existing_facts()
        if f in existing:
            return False
        with self.memory_file.open("a", encoding="utf-8") as fh:
            fh.write(f"- {f}\n")
        return True

    def get_existing_facts(self) -> set[str]:
        """返回 memory.md 中已有的所有事实文本，用于去重。"""
        if not self.memory_file.exists():
            return set()
        facts: set[str] = set()
        for line in self.memory_file.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line.startswith("- "):
                facts.add(line[2:].strip())
        return facts
