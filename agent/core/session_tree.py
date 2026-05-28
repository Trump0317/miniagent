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
        """将 leaf 移到 entry_id（用于 /tree 导航到现有分支）。

        与 fork() 功能相同，语义上表示"切换分支"而非"开新分支"。
        实现上可区分（比如 navigate 不改变 UI 主题色），暂时共用。
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
                # 保留 first_kept_id 及之后的 entry，丢弃之前的
                if entry.first_kept_id and entry.first_kept_id in entry_map:
                    trim_at = entry_map[entry.first_kept_id]
                    kept = result[trim_at:]
                else:
                    kept = []
                # compaction summary 放在最前面
                result = [{
                    "role": "user",
                    "content": f"[系统] 以下是之前对话的压缩记录:\n{entry.summary}",
                }] + kept
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


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: 基本 append + path")
    print("=" * 60)

    tree = SessionTree()
    r = tree.append("system", "你是助手")
    u1 = tree.append("user", "列出文件")
    a1 = tree.append("assistant", "LICENSE README.md agent.py")

    path = tree.path()
    assert len(path) == 3
    assert path[-1].content == "LICENSE README.md agent.py"
    print("✓ path 正确")

    ctx = tree.build_context()
    assert len(ctx) == 3
    assert ctx[0]["role"] == "system"
    assert ctx[2]["role"] == "assistant"
    print("✓ build_context 正确")

    # ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("测试 2: fork + 分支")
    print("=" * 60)

    tree.fork(u1)  # 回到 "列出文件"
    assert tree.leaf_id == u1
    print("✓ fork 到 u1")

    a2 = tree.append("assistant", "可以用 tree 或 find 命令")
    u2 = tree.append("user", "用 ls -la")
    a3 = tree.append("assistant", "total 76 ...")

    path2 = tree.path()
    assert len(path2) == 5  # root, u1, a2, u2, a3
    assert path2[2].content == "可以用 tree 或 find 命令"
    print("✓ 分支 path 正确，不包含旧分支的 a1")

    # 旧分支仍然存在
    children = tree.children_of(u1)
    assert len(children) == 2  # a1 和 a2
    assert children[0].content == "LICENSE README.md agent.py"
    assert children[1].content == "可以用 tree 或 find 命令"
    print("✓ 两个分支都保留")

    # ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("测试 3: compaction")
    print("=" * 60)

    # 回到 root，创建长对话
    tree2 = SessionTree()
    tree2.append("system", "你是助手")
    tree2.append("user", "问题1")
    tree2.append("assistant", "回答1")
    tree2.append("user", "问题2")
    tree2.append("assistant", "回答2")
    tree2.append("user", "问题3")
    tree2.append("assistant", "回答3", {"tool_calls": [{"function": {"name": "bash"}}]})
    tree2.append("tool", "stdout: ...", {"tool_call_id": "xxx"})

    # 压缩：保留 "问题3" 及之后
    all_entries = tree2.path()
    user_entries = [e for e in all_entries if e.role == "user"]
    first_kept = user_entries[-1]  # "问题3"

    cid = tree2.compact(
        summary="会话摘要: 用户问了3个问题",
        first_kept_id=first_kept.id,
        tokens_before=5000,
    )
    print(f"✓ compaction entry 创建: {cid[:8]}...")

    ctx = tree2.build_context()
    print(f"build_context 返回 {len(ctx)} 条消息:")
    for m in ctx:
        preview = str(m.get("content", ""))[:50].replace("\n", " ")
        tc = " [tool_calls]" if "tool_calls" in m else ""
        print(f"  {m['role']}: {preview}{tc}")

    # 验证: 只有 compaction summary + 问题3 + 回答3 + tool
    assert len(ctx) == 4
    assert ctx[0]["role"] == "user"
    assert "压缩记录" in ctx[0]["content"]
    assert ctx[1]["role"] == "user" and "问题3" in ctx[1]["content"]
    assert ctx[2]["role"] == "assistant"
    assert ctx[3]["role"] == "tool"
    print("✓ build_context 跳过已压缩的消息")

    # 压缩后继续对话
    tree2.append("user", "问题4")
    tree2.append("assistant", "回答4")

    ctx2 = tree2.build_context()
    assert len(ctx2) == 6  # summary + q3/a3/tool + q4/a4
    print("✓ 压缩后继续添加消息正常")

    # ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("测试 4: 多次 compaction")
    print("=" * 60)

    tree3 = SessionTree()
    tree3.append("system", "你是助手")
    tree3.append("user", "q1")
    tree3.append("assistant", "a1")
    tree3.append("user", "q2")
    tree3.append("assistant", "a2")

    # 第一次压缩
    e1 = tree3.path()
    tree3.compact("压缩1: q1/q2", e1[2].id, 1000)
    # 后续
    tree3.append("user", "q3")
    tree3.append("assistant", "a3")
    tree3.append("user", "q4")
    tree3.append("assistant", "a4")

    # 第二次压缩
    e2 = tree3.path()
    user_entries2 = [e for e in e2 if e.role == "user"]
    tree3.compact("压缩2: q3/q4", user_entries2[-1].id, 2000)

    ctx3 = tree3.build_context()
    # 第二次压缩替换第一次: summary2 + q4 + a4
    assert len(ctx3) == 3
    assert "压缩2" in ctx3[0]["content"]
    assert "q4" in ctx3[1]["content"]
    print("✓ 多次 compaction，后续 summary 替换前一个")

    # ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("测试 5: dump_tree 可视化")
    print("=" * 60)

    print(tree.dump_tree())
    print()
    print(tree2.dump_tree())

    # ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("测试 6: navigate / fork 到 leaf 已有子节点")
    print("=" * 60)

    tree4 = SessionTree()
    tree4.append("system", "你是助手")
    tree4.append("user", "任务A")
    a_id = tree4.append("assistant", "完成A")
    tree4.append("user", "任务B")
    tree4.append("assistant", "完成B")

    # navigate 回到 a_id（已有子节点"任务B"）
    tree4.navigate(a_id)
    assert tree4.leaf_id == a_id
    p = tree4.path()
    assert len(p) == 3  # system, 任务A, 完成A
    print("✓ navigate 到已有分支节点")

    # ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("测试 7: 边界情况")
    print("=" * 60)

    # 空树
    t = SessionTree()
    assert t.path() == []
    assert t.build_context() == []
    assert t.leaf_id is None
    print("✓ 空树")

    # 单节点
    t.append("system", "")
    assert len(t.path()) == 1
    print("✓ 单节点")

    # fork 到不存在的 id
    try:
        t.fork("no-such-id")
        assert False, "应该抛异常"
    except ValueError:
        print("✓ fork 不存在 id 正确抛异常")

    # non_system_entries
    t2 = SessionTree()
    t2.append("system", "")
    t2.append("user", "hello")
    t2.append("assistant", "hi")
    non_sys = t2.non_system_entries()
    assert len(non_sys) == 2
    assert all(e.role in ("user", "assistant") for e in non_sys)
    print("✓ non_system_entries 过滤正确")

    print("\n" + "=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)
