"""记忆存储 —— 纯文件 I/O，不调用 LLM，不关心压缩逻辑。

树状结构存储:
  - 底层用 SessionTree 管理消息树（唯一真相来源）
  - 对外 history 属性 = tree.build_context()（给 Runner 的列表视图）
  - 压缩/分叉后自动反映在 history 中

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

    树是唯一真相来源：
      - self._tree: SessionTree（树状结构，用于压缩/分叉/导航/持久化）
      - self.history: property，返回 tree.build_context()（只读列表视图）
      - 写入统一走 append_message()，一步完成树 + JSONL
    """

    def __init__(self, memory_dir: Path, session_dir: Path | None = None):
        self.memory_dir = Path(memory_dir)
        self._session_dir = Path(session_dir) if session_dir else self.memory_dir

        # 共享文件（跨会话）
        self.memory_file = self.memory_dir / "memory.md"
        self.summary_dir = self.memory_dir / "summaries"
        self.user_file = self.memory_dir / "user.md"

        # 会话私有文件
        self.history_file = self._session_dir / "history.jsonl"

        # ── 树（唯一真相来源）──
        self._tree = SessionTree()
        self._fork_stack: list[str] = []  # fork 跳转栈，记录每次分叉前的位置

        # 确保目录存在（共享文件目录）
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        # 会话目录延迟创建（第一条消息写入时）
        if self.history_file.exists():
            pass  # 已存在则保留
        for f, header in [
            (self.memory_file, "# 长期记忆\n\n此文件常驻上下文，记录核心目标、当前任务与关键事实。\n"),
            (self.user_file, "# 用户信息\n\n此文件记录用户的基本信息和偏好。\n"),
        ]:
            if not f.exists():
                f.write_text(header, encoding="utf-8")
        if not self.history_file.exists():
            pass  # 延迟创建，只在第一条消息写入后才产生

    # ── history 属性（树的可读视图）──

    @property
    def history(self) -> list[dict]:
        """从树计算的消息列表（不含 system），供 Runner 使用。

        每次访问从树实时计算。压缩/分叉后自动反映新路径。
        """
        return self._tree.build_context()

    @property
    def tree(self) -> SessionTree:
        """底层会话树（只读）"""
        return self._tree

    @property
    def leaf_id(self) -> str | None:
        """当前 leaf entry id"""
        return self._tree.leaf_id

    @property
    def has_data(self) -> bool:
        """树中是否有数据（用于判断是否从历史恢复）"""
        return self._tree.root_id is not None

    @property
    def entry_count(self) -> int:
        """树中 entry 总数"""
        return self._tree.entry_count

    # ── 时间工具 ──

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _today_file(self, when: datetime | None = None) -> Path:
        moment = when or datetime.now()
        return self.summary_dir / f"{moment:%Y-%m-%d}.md"

    # ── 唯一写入入口 ──

    def append_message(self, msg: dict) -> str:
        """写入一条消息到树 + JSONL。返回 entry_id。跳过 system 消息。"""
        role = msg.get("role", "")
        if role == "system":
            return ""

        metadata = self._extract_metadata(msg)
        entry_id = self._tree.append(role, msg.get("content"), metadata)
        self._write_jsonl_line(entry_id, msg)
        return entry_id

    @staticmethod
    def _extract_metadata(msg: dict) -> dict[str, Any]:
        """从消息字典提取树节点需要存储的元数据。"""
        role = msg.get("role", "")
        meta: dict[str, Any] = {}
        if role == "assistant":
            if msg.get("tool_calls"):
                meta["tool_calls"] = msg["tool_calls"]
            if msg.get("reasoning_content"):
                meta["reasoning_content"] = msg["reasoning_content"]
        elif role == "tool":
            if msg.get("tool_call_id"):
                meta["tool_call_id"] = msg["tool_call_id"]
        return meta

    def _write_jsonl_line(self, entry_id: str, msg: dict) -> None:
        """将单条消息追加到 JSONL（增量写入）。"""
        entry = self._tree.get(entry_id)
        if not entry:
            return

        self._session_dir.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            "id": entry.id,
            "parent_id": entry.parent_id,
            "type": entry.type,
            "role": msg.get("role", ""),
            "content": msg.get("content"),
            "metadata": self._extract_metadata(msg),
            "timestamp": entry.timestamp,
        }
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── 树操作（压缩 / 分叉 / 导航）──

    def compress_tree(self, summary: str, first_kept_id: str, tokens_before: int) -> str:
        """压缩树：插入 COMPACT 节点 + 全量持久化 JSONL。

        返回 compaction entry 的 id。
        """
        cid = self._tree.compact(summary, first_kept_id, tokens_before)
        self._write_jsonl_full()
        return cid

    def fork(self, entry_id: str) -> None:
        """分叉到指定 entry。history 自动反映新路径。"""
        self._tree.fork(entry_id)

    def navigate(self, entry_id: str) -> None:
        """导航到指定 entry（切换分支）。history 自动反映新路径。"""
        self._tree.navigate(entry_id)

    def get_tree_entries(self) -> list[SessionEntry]:
        """获取所有树条目的只读列表"""
        return list(self._tree.entries().values())

    # ── fork 跳转栈 ──

    def push_fork(self) -> None:
        """保存当前位置到栈（/fork 前调用）。"""
        if self._tree.leaf_id:
            self._fork_stack.append(self._tree.leaf_id)

    def pop_fork(self) -> str | None:
        """弹出栈顶位置并导航过去（/back 使用）。返回目标 entry_id，栈空返回 None。"""
        if not self._fork_stack:
            return None
        target = self._fork_stack.pop()
        self._tree.navigate(target)
        return target

    # ── JSONL 全量持久化 ──

    def _write_jsonl_full(self) -> None:
        """全量持久化树到 JSONL（DFS 序）。压缩后调用。"""
        if not self._tree.root_id:
            return

        self._session_dir.mkdir(parents=True, exist_ok=True)

        with self.history_file.open("w", encoding="utf-8") as f:
            self._write_subtree(self._tree.root_id, f)

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

    # ── 树恢复（启动时从 JSONL 重建）──

    def restore_tree(self) -> bool:
        """从 JSONL 恢复树结构。

        自动检测格式：
          - 新格式: 含 id/parent_id/type 字段
          - 旧格式: 仅 role/content/metadata，转换为新格式

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
            self._load_tree_format(lines)
        else:
            self._load_legacy_format(lines)

        return self.has_data

    def _load_tree_format(self, lines: list[str]) -> None:
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

            if role == "system" and entry_type == "message":
                continue  # system 消息不存入树（与 append_message 一致），但 compaction 保留

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

            self._tree.add_entry(entry)

        # 找 leaf: 从 root 沿最右子节点链走到底
        leaf = self._tree.find_deepest_leaf()
        if leaf:
            self._tree.set_leaf(leaf)

    def _load_legacy_format(self, lines: list[str]) -> None:
        """从旧格式 JSONL（无树结构）转换为新格式树"""
        self._tree = SessionTree()

        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = row.get("role", "")
            content = row.get("content")
            if role == "system":
                continue  # 旧格式可能误存 system 消息

            if role == "user":
                content = content or "(历史用户消息)"

            meta = row.get("metadata", {})
            metadata = self._extract_metadata({"role": role, **meta})

            self._tree.append(role, content, metadata)

        # 转换后持久化为新格式
        self._write_jsonl_full()

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

    def user_preferences(self, max_items: int = 0) -> list[str]:
        """读取用户偏好列表。max_items > 0 时只返回最近 N 条。"""
        if not self.user_file.exists():
            return []
        prefs = [line.strip("- ").strip()
                for line in self.user_file.read_text(encoding="utf-8").split("\n")
                if line.strip().startswith("-")]
        if max_items > 0 and len(prefs) > max_items:
            return prefs[-max_items:]
        return prefs

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
        """树中非 system 的纯消息列表，供 Compactor 使用。

        委托给 SessionTree.to_messages()。
        """
        return self._tree.to_messages()
