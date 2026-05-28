"""记忆存储 —— 纯文件 I/O，不调用 LLM，不关心压缩逻辑。

树状结构存储:
  - 底层用 SessionTree 管理消息树
  - 对外暴露 history 属性（普通列表），保持与 Runner 兼容
  - 压缩/分叉时从树重建 history
  - JSONL 持久化使用扩展格式（含 id/parent_id/type）

三层记忆:
  history.jsonl  — 对话历史（JSONL 持久化，树结构）
  memory.md      — 核心记忆（追加事实）
  summaries/     — 每日摘要（按日期文件）
  user.md        — 用户偏好（列表）
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from .session_tree import SessionTree, SessionEntry


class AgentMemory:
    """纯存储层 —— 管理对话历史、摘要、偏好和长期记忆的文件读写。

    历史存储:
      - self._tree: SessionTree（树状结构，用于压缩/分叉/导航）
      - self.history: list[dict]（普通列表，Runner 直接操作）
      - 写入时双写（tree + history），分叉/导航/压缩时从 tree 重建 history
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_file = self.memory_dir / "memory.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.summary_dir = self.memory_dir / "summaries"
        self.user_file = self.memory_dir / "user.md"

        # ── 树（新）──
        self._tree = SessionTree()

        # ── 历史列表（Runner 兼容）──
        self._history: list[dict] = []

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

    # ── history 属性 ──

    @property
    def history(self) -> list[dict]:
        """当前对话历史列表（Runner 兼容接口）。

        注意: 返回的是内部列表的引用，Runner 的 append 等操作会直接修改它。
        修改后需通过 persist_message() 同步到树。
        """
        return self._history

    @history.setter
    def history(self, value: list[dict]) -> None:
        self._history = value

    @property
    def tree(self) -> SessionTree:
        """底层会话树（只读）"""
        return self._tree

    @property
    def leaf_id(self) -> str | None:
        """当前 leaf entry id"""
        return self._tree.leaf_id

    # ── 时间工具 ──

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _today_file(self, when: datetime | None = None) -> Path:
        moment = when or datetime.now()
        return self.summary_dir / f"{moment:%Y-%m-%d}.md"

    # ── 历史写入（双写: history 列表 + 树）──

    def append_history(self, content: dict | Any, persist: bool = True) -> None:
        """追加消息到 history 列表 + 树。默认同时持久化到 JSONL。

        persist=False 用于恢复历史（消息已存在于 JSONL）。
        """
        if not isinstance(content, dict):
            return

        self._history.append(content)
        if persist:
            self._append_to_tree(content)
            self._persist_entry(content)

    def persist_message(self, content: Any) -> None:
        """只持久化到 JSONL + 树，不修改 history。

        用于 Runner 直接 history.append() 后的同步。
        """
        if not isinstance(content, dict):
            return
        self._append_to_tree(content)
        self._persist_entry(content)

    def _append_to_tree(self, content: dict) -> str | None:
        """将消息追加到树，返回 entry_id。跳过 system 消息。"""
        role = content.get("role", "")
        if role == "system":
            return None  # system 消息不持久化

        metadata: dict[str, Any] = {}
        if role == "assistant":
            if content.get("tool_calls"):
                metadata["tool_calls"] = content["tool_calls"]
            if content.get("reasoning_content"):
                metadata["reasoning_content"] = content["reasoning_content"]
        elif role == "tool":
            if content.get("tool_call_id"):
                metadata["tool_call_id"] = content["tool_call_id"]

        return self._tree.append(role, content.get("content"), metadata)

    def _persist_entry(self, content: dict) -> None:
        """将单条消息写入 JSONL（新格式，含树结构字段）。

        复用树上最新 entry 的 id/parent_id（刚由 _append_to_tree 创建）。
        """
        role = content.get("role", "")
        if role == "system":
            return  # 不持久化

        entry_id = self._tree.leaf_id
        if not entry_id:
            return

        entry = self._tree.get(entry_id)
        if not entry:
            return

        meta: dict[str, Any] = {}
        if role == "assistant":
            if content.get("tool_calls"):
                meta["tool_calls"] = content["tool_calls"]
            if content.get("reasoning_content"):
                meta["reasoning_content"] = content["reasoning_content"]
        elif role == "tool":
            if content.get("tool_call_id"):
                meta["tool_call_id"] = content["tool_call_id"]

        row = {
            "id": entry.id,
            "parent_id": entry.parent_id,
            "type": entry.type,
            "role": role,
            "content": content.get("content"),
            "metadata": meta,
            "timestamp": entry.timestamp,
        }
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── 树操作（压缩 / 分叉 / 导航）──

    def compress_tree(self, summary: str, first_kept_id: str, tokens_before: int) -> str:
        """压缩树：插入 compaction entry，重建 history，全量持久化 JSONL。

        返回 compaction entry 的 id。
        """
        cid = self._tree.compact(summary, first_kept_id, tokens_before)
        self._history = self._tree.build_context()
        # 全量重写 JSONL（树结构变化后需要 DFS 序）
        self._persist_tree()
        return cid

    def fork(self, entry_id: str) -> None:
        """分叉到指定 entry，重建 history"""
        self._tree.fork(entry_id)
        self._history = self._tree.build_context()

    def navigate(self, entry_id: str) -> None:
        """导航到指定 entry（切换分支），重建 history"""
        self._tree.navigate(entry_id)
        self._history = self._tree.build_context()

    def get_tree_entries(self) -> list[SessionEntry]:
        """获取所有树条目的只读列表"""
        return list(self._tree._entries.values())

    # ── 树恢复（启动时从 JSONL 重建）──

    def restore_tree(self) -> bool:
        """从 JSONL 恢复树结构 + history 列表。

        自动检测格式：
          - 新格式: 含 id/parent_id/type 字段
          - 旧格式: 仅 role/content/metadata，转换为线性的树

        返回 True 表示有历史数据被恢复。
        """
        if not self.history_file.exists():
            return False

        lines: list[str] = []
        with self.history_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        if not lines:
            return False

        # 检测格式
        first = json.loads(lines[0])
        if "id" in first and "parent_id" in first and "type" in first:
            ok = self._load_tree_format(lines)
        else:
            ok = self._load_legacy_format(lines)

        # 确保 history 与树同步
        if self._tree._root_id:
            self._history = self._tree.build_context()

        return ok and bool(self._history)

    def _load_tree_format(self, lines: list[str]) -> bool:
        """从新格式 JSONL 重建树"""
        self._tree = SessionTree()

        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_id = row.get("id", "")
            parent_id = row.get("parent_id")
            entry_type = row.get("type", "message")
            role = row.get("role", "")

            # 手动插入到树（绕过 append 的自动 leaf 逻辑）
            entry = SessionEntry(
                id=entry_id,
                parent_id=parent_id,
                type=entry_type,
                role=role,
                content=row.get("content") if entry_type == "message" else None,
                metadata=row.get("metadata", {}),
                timestamp=row.get("timestamp", 0.0),
                summary=row.get("summary") if entry_type == "compaction" else None,
                first_kept_id=row.get("first_kept_id") if entry_type == "compaction" else None,
                tokens_before=row.get("tokens_before", 0) if entry_type == "compaction" else 0,
            )

            self._tree._entries[entry_id] = entry
            if self._tree._root_id is None:
                self._tree._root_id = entry_id

        # 找 leaf: 从 root 沿最右子节点链走到底
        if self._tree._root_id:
            self._tree._leaf_id = self._tree._root_id
            while True:
                children = self._tree.children_of(self._tree._leaf_id)
                if not children:
                    break
                self._tree._leaf_id = children[-1].id  # 走最右分支

        return True

    def _load_legacy_format(self, lines: list[str]) -> bool:
        """从旧格式 JSONL（无树结构）转换为新格式树"""
        self._tree = SessionTree()

        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = row.get("role", "")
            content = row.get("content")
            meta = row.get("metadata", {})

            # 旧格式会跳过 user 消息，但树结构下需要保留
            # 然而旧格式的 user 消息内容未知，用占位
            if role == "user":
                content = content or "(历史用户消息)"

            msg = {"role": role, "content": content}
            if role == "assistant":
                if meta.get("tool_calls"):
                    msg["tool_calls"] = meta["tool_calls"]
                if meta.get("reasoning_content"):
                    msg["reasoning_content"] = meta["reasoning_content"]
            elif role == "tool":
                if meta.get("tool_call_id"):
                    msg["tool_call_id"] = meta["tool_call_id"]

            self._history.append(msg)
            self._append_to_tree(msg)

        # 转换后持久化为新格式，然后从树重建 history
        self._persist_tree()
        return True

    def restore_history(self, max_messages: int = 50) -> list[dict]:
        """恢复上次会话的历史上下文。

        新实现: 从 JSONL 重建树 → 用 build_context 获取恢复后的消息列表。
        这些消息会作为上下文注入，只需 assistant/tool 部分。
        """
        if not self.restore_tree():
            return []

        # 用于 Agent 注入的恢复消息
        # 从 build_context 中提取非 system 部分
        ctx = self._history  # restore_tree() 已设置 self._history
        non_system = [m for m in ctx if m.get("role") != "system"]

        if not non_system:
            return []

        # 限制条数
        if len(non_system) > max_messages:
            non_system = non_system[-max_messages:]
            # 确保不以孤立的 tool 消息开头
            while non_system and non_system[0].get("role") == "tool":
                non_system.pop(0)

        return non_system

    def _persist_tree(self) -> None:
        """全量持久化树到 JSONL（用于旧格式迁移后）"""
        if not self._tree._root_id:
            return

        with self.history_file.open("w", encoding="utf-8") as f:
            self._write_subtree(self._tree._root_id, f)

    def _write_subtree(self, entry_id: str, f) -> None:
        """DFS 写入子树（按子节点 timestamp 排序）"""
        entry = self._tree.get(entry_id)
        if not entry:
            return

        row: dict[str, Any] = {
            "id": entry.id,
            "parent_id": entry.parent_id,
            "type": entry.type,
            "role": entry.role,
            "timestamp": entry.timestamp,
        }
        if entry.type == "compaction":
            row["summary"] = entry.summary
            row["first_kept_id"] = entry.first_kept_id
            row["tokens_before"] = entry.tokens_before
        else:
            row["content"] = entry.content
            row["metadata"] = entry.metadata

        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        for child in self._tree.children_of(entry.id):
            self._write_subtree(child.id, f)

    # ── 旧接口兼容（压缩后在 Agent 中重建 system prompt 用）──

    def _rewrite_history(self, entries: list[dict]) -> None:
        """重写 history.jsonl 和树，只保留给定的消息。

        用于压缩后的同步（兼容旧调用）。
        """
        if not entries:
            return

        # 重建树
        self._tree = SessionTree()
        self._history = entries
        for msg in entries:
            self._append_to_tree(msg)

        self._persist_tree()

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

    def non_system_entries(self) -> list[dict]:
        """树中非 system 的纯消息列表，供 Compactor 使用。"""
        entries = self._tree.non_system_entries()
        result: list[dict] = []
        for e in entries:
            msg: dict = {"role": e.role, "content": e.content}
            if e.role == "assistant" and e.metadata.get("tool_calls"):
                msg["tool_calls"] = e.metadata["tool_calls"]
            if e.role == "tool" and e.metadata.get("tool_call_id"):
                msg["tool_call_id"] = e.metadata["tool_call_id"]
            result.append(msg)
        return result
