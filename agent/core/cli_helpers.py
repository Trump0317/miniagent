"""CLI 辅助函数 — 由 agent.py 和测试代码共用。

不依赖 Agent 以外的模块，避免循环导入。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


def handle_tree(agent: Agent) -> None:
    """显示会话分支树。"""
    entries = agent.memory.get_tree_entries()
    if not entries:
        print("(空会话)")
        return

    # id → 序号（只给 user 消息编号）
    sorted_entries = sorted(entries, key=lambda e: e.timestamp)
    id_to_num: dict[str, int] = {}
    user_count = 0
    for e in sorted_entries:
        if e.role == "user":
            user_count += 1
            id_to_num[e.id] = user_count

    leaf_id = agent.memory.leaf_id

    def _print_tree(entry_id: str, prefix: str, depth: int) -> None:
        if depth > 40:
            print(f"{prefix}... (超过深度限制)")
            return

        entry = agent.memory.tree.get(entry_id)
        if not entry:
            return

        marker = " ← 当前" if entry_id == leaf_id else ""

        if entry.type == "compaction":
            summary = (entry.summary or "")[:50].replace("\n", " ")
            print(f"{prefix}[压缩] {summary}{marker}")
        else:
            num = id_to_num.get(entry.id)
            num_str = f"[{num}] " if num else ""
            content = (entry.content or "")[:50].replace("\n", " ")
            print(f"{prefix}{num_str}{entry.role}: {content}{marker}")

        children = agent.memory.tree.children_of(entry_id)
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{prefix}{connector}")
            _print_tree(child.id, child_prefix, depth + 1)

    if agent.memory.tree.root_id:
        _print_tree(agent.memory.tree.root_id, "", 0)


def handle_back(agent: Agent) -> None:
    """返回分叉之前的位置。

    每次 /fork 会保存跳转前的 leaf 位置，/back 弹出最近保存的位置并导航过去。
    """
    target = agent.memory.pop_fork()
    if target:
        entry = agent.memory.tree.get(target)
        content = (entry.content or "")[:60].replace("\n", " ") if entry else "(未知)"
        agent._system_prompt = agent._prompt.build()
        print(f"[back] 已返回: {content}")
    else:
        print("[back] 没有可返回的位置（fork 栈为空）")


def handle_fork(agent: Agent, command: str) -> None:
    """分叉到指定 user 消息。

    用法:
      /fork       → 分叉到最近一次 user 消息
      /fork N     → 分叉到第 N 条 user 消息
    """
    parts = command.split(maxsplit=1)
    arg = parts[1] if len(parts) > 1 else ""

    all_entries = sorted(
        [e for e in agent.memory.get_tree_entries() if e.role == "user"],
        key=lambda e: e.timestamp,
    )

    if not all_entries:
        print("[fork] 没有可 fork 的用户消息")
        return

    if not arg:
        target = all_entries[-2] if len(all_entries) >= 2 else all_entries[-1]
    else:
        try:
            n = int(arg)
        except ValueError:
            print(f"[fork] 无效参数: {arg}，请输入序号")
            return
        if n < 1 or n > len(all_entries):
            print(f"[fork] 序号超出范围: 1-{len(all_entries)}")
            return
        target = all_entries[n - 1]

    agent.memory.push_fork()  # 保存当前位置，以便 /back 返回
    agent.memory.fork(target.id)
    agent._system_prompt = agent._prompt.build()
    content = (target.content or "")[:60].replace("\n", " ")
    print(f"[fork] 已分叉到: {content}")
