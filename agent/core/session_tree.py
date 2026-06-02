"""会话树 —— 管理对话历史的分支结构。

核心概念:
- 每个 entry 是树中的一个节点，通过 parent_id 连成链
- leaf 是当前所在位置，append 从 leaf 后分叉
- compaction entry 标记压缩点，build_context() 利用它跳过已压缩的历史
- fork 将 leaf 移到历史节点，后续 append 在那里创建新分支

用法:
    tree = SessionTree()
    root = tree.append("system", "你是智能助手")
    u1 = tree.append("user", "列出文件")
    a1 = tree.append("assistant", "LICENSE ...")

    tree.fork(u1)       # 回到 "列出文件"，分叉
    a2 = tree.append("assistant", "换个方案 ...")

    path = tree.path()  # root → u1 → a2
    ctx = tree.build_context()  # 给 LLM 的消息列表
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass
class SessionEntry:
    """树中的一个节点。

    id / parent_id: 树结构的核心。parent_id=None 是根节点。

    type 决定同名 node 如何被 build_context() 处理:
      - "message": 普通消息节点，按 role 输出
      - "compaction": 压缩节点，包含 summary + first_kept_id
    """

    id: str
    parent_id: str | None
    type: str                     # "message" | "compaction"
    role: str                     # "user" | "assistant" | "tool" | "system"
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    # === compaction 类型专用字段 ===
    summary: str | None = None      # LLM 生成的压缩摘要文本
    first_kept_id: str | None = None  # compact 之后从哪个 entry 开始保留
    tokens_before: int = 0          # 压缩前的上下文 token 数

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]


class SessionTree:
    """会话树 —— 管理所有 entry 的创建、导航和查询。

    关键语义:
      - append() 总是在当前 leaf 下创建子节点
      - fork() 和 navigate() 只改变 leaf 指针，不修改数据
      - build_context() 沿 path() 走，遇 compaction 跳过已压缩的部分
    """

    def __init__(self):
        self._entries: dict[str, SessionEntry] = {}
        self._root_id: str | None = None
        self._leaf_id: str | None = None

    # ── 只读属性 ──

    @property
    def leaf_id(self) -> str | None:
        """当前所在 leaf 的 id"""
        return self._leaf_id

    @property
    def root_id(self) -> str | None:
        """根节点 id"""
        return self._root_id

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def entries(self) -> dict[str, SessionEntry]:
        """所有 entry 的只读视图"""
        return dict(self._entries)

    # ── 写操作 ──

    def append(
        self,
        role: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """在当前 leaf 后追加新 message entry，返回 entry_id。

        自动成为新的 leaf。root 不存在时自动创建。
        """
        entry_id = SessionEntry.new_id()
        entry = SessionEntry(
            id=entry_id,
            parent_id=self._leaf_id,   # None for root
            type="message",
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._entries[entry_id] = entry

        if self._root_id is None:
            self._root_id = entry_id
        self._leaf_id = entry_id
        return entry_id

    def compact(
        self,
        summary: str,
        first_kept_id: str,
        tokens_before: int,
    ) -> str:
        """压缩树：将 compaction entry 插入到 first_kept 之前。

        树结构调整:
          之前: ... → first_kept_parent → first_kept → ... → leaf
          之后: ... → first_kept_parent → COMPACT → first_kept → ... → leaf (不变)

        COMPACT 的 parent = first_kept 的原 parent
        first_kept 的 parent = COMPACT

        这样 build_context() 从 root 走到 leaf 必然经过 COMPACT，
        自然应用压缩语义。已压缩的 entry 不删除，只是不在主路径上。
        """
        first = self._entries.get(first_kept_id)
        if not first:
            raise ValueError(f"first_kept_id 不存在: {first_kept_id}")

        entry_id = SessionEntry.new_id()
        entry = SessionEntry(
            id=entry_id,
            parent_id=first.parent_id,       # COMPACT 替代 first 在原父节点下的位置
            type="compaction",
            role="system",
            summary=summary,
            first_kept_id=first_kept_id,
            tokens_before=tokens_before,
        )
        self._entries[entry_id] = entry

        # first_kept 重新挂到 COMPACT 下
        first.parent_id = entry_id

        return entry_id

    def fork(self, entry_id: str) -> None:
        """将 leaf 移到 entry_id，后续 append 从那里分叉。

        Raises:
            ValueError: entry_id 不存在
        """
        if entry_id not in self._entries:
            raise ValueError(f"entry_id 不存在: {entry_id}")
        self._leaf_id = entry_id

    def navigate(self, entry_id: str) -> None:
        """将 leaf 移到 entry_id，语义上表示"切换分支"而非"开新分支"。

        与 fork() 功能相同，未来可区分（如 navigate 不记录到 fork 栈）。
        """
        self.fork(entry_id)

    # ── 读操作 ──

    def get(self, entry_id: str) -> SessionEntry | None:
        """按 id 获取 entry"""
        return self._entries.get(entry_id)

    def path(self, to_id: str | None = None) -> list[SessionEntry]:
        """从 root 沿 parent_id 链走到 to_id（默认当前 leaf）。

        返回按时间顺序排列的 entry 列表（root → leaf）。
        """
        target = to_id or self._leaf_id
        if target is None:
            return []

        # 从 leaf 往上回溯
        rev: list[SessionEntry] = []
        cur = target
        visited: set[str] = set()
        while cur is not None:
            if cur in visited:
                break  # 防止循环引用
            visited.add(cur)
            entry = self._entries.get(cur)
            if entry is None:
                break
            rev.append(entry)
            cur = entry.parent_id

        rev.reverse()
        return rev

    def children_of(self, entry_id: str) -> list[SessionEntry]:
        """返回 entry_id 的所有直接子节点（按 timestamp 排序）"""
        children = [
            e for e in self._entries.values()
            if e.parent_id == entry_id
        ]
        children.sort(key=lambda e: e.timestamp)
        return children

    def siblings(self, entry_id: str) -> list[SessionEntry]:
        """返回 entry_id 的兄弟节点"""
        entry = self._entries.get(entry_id)
        if not entry:
            return []
        return [e for e in self.children_of(entry.parent_id or "") if e.id != entry_id]

    # ── 上下文构建 ──

    def build_context(self, to_id: str | None = None) -> list[dict]:
        """构建给 LLM 的消息列表。

        规则:
          1. 沿 path 遍历
          2. 遇到 compaction entry: 用 summary 替换 first_kept_id 之前的所有 entry，
             保留 first_kept_id 及之后的 entry
          3. 遇到 message entry: 转换为 LLM 格式

        多次 compaction 时: 后续 compaction 的 summary 替换前一个。
        """
        path_entries = self.path(to_id)
        result: list[dict] = []
        entry_map: dict[str, int] = {}  # entry_id → index in result

        for entry in path_entries:
            if entry.type == "compaction":
                # COMPACT 始终在 first_kept 之前，前面的内容全部替换为摘要
                result = [{
                    "role": "user",
                    "content": f"[系统] 以下是之前对话的压缩记录:\n{entry.summary}",
                }]
                entry_map = {}
                continue

            if entry.type != "message":
                continue

            entry_map[entry.id] = len(result)
            msg: dict = {"role": entry.role}
            if entry.content is not None:
                msg["content"] = entry.content

            # 恢复 metadata
            if entry.role == "assistant":
                if entry.metadata.get("tool_calls"):
                    msg["tool_calls"] = entry.metadata["tool_calls"]
                if entry.metadata.get("reasoning_content"):
                    msg["reasoning_content"] = entry.metadata["reasoning_content"]
            elif entry.role == "tool":
                if entry.metadata.get("tool_call_id"):
                    msg["tool_call_id"] = entry.metadata["tool_call_id"]

            result.append(msg)

        return result

    def non_system_entries(self, to_id: str | None = None) -> list[SessionEntry]:
        """路径上非 system role 的 entry，供 compactor 使用"""
        return [
            e for e in self.path(to_id)
            if e.role != "system" and e.type == "message"
        ]

    def to_messages(self, to_id: str | None = None) -> list[dict]:
        """路径上的消息纯文本（不含 system），用于传给 LLM 压缩器"""
        entries = self.non_system_entries(to_id)
        result: list[dict] = []
        for e in entries:
            msg = {"role": e.role, "content": e.content}
            if e.role == "assistant" and e.metadata.get("tool_calls"):
                msg["tool_calls"] = e.metadata["tool_calls"]
            if e.role == "tool" and e.metadata.get("tool_call_id"):
                msg["tool_call_id"] = e.metadata["tool_call_id"]
            result.append(msg)
        return result

    # ── 诊断 ──

    def dump_tree(self) -> str:
        """生成树的文本表示，用于调试"""
        if not self._root_id:
            return "(空树)"

        lines: list[str] = []
        self._dump_node(self._root_id, "", lines)
        return "\n".join(lines)

    def _dump_node(self, entry_id: str, prefix: str, lines: list[str]) -> None:
        entry = self._entries.get(entry_id)
        if not entry:
            return

        marker = " ← leaf" if entry_id == self._leaf_id else ""
        content_preview = (entry.content or entry.summary or "")[:40].replace("\n", " ")
        tag = "COMPACT" if entry.type == "compaction" else entry.role
        lines.append(f"{prefix}[{entry.id[:8]}] {tag}: {content_preview}{marker}")

        children = self.children_of(entry_id)
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{prefix}{connector}")
            self._dump_node(child.id, child_prefix, lines)

