"""CLI 辅助函数 — 由 agent.py 和测试代码共用。

不依赖 Agent 以外的模块，避免循环导入。
"""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.agent import Agent


# ═══════════════════════════════════════════════════════════════
# 树视图共用辅助
# ═══════════════════════════════════════════════════════════════

def format_tool_entry(tree, entry) -> str:
    """将工具条目格式化为 [toolname: arg_summary] 形式。"""
    tool_name = "tool"
    arg_summary = ""

    parent = tree.get(entry.parent_id) if entry.parent_id else None
    if parent and parent.role == "assistant" and parent.metadata.get("tool_calls"):
        tool_call_id = entry.metadata.get("tool_call_id")
        for tc in parent.metadata["tool_calls"]:
            if tc.get("id") == tool_call_id:
                tool_name = tc.get("function", {}).get("name", "tool")
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    for key in ("file_path", "path", "command", "query", "url", "content"):
                        if key in args and args[key]:
                            arg_summary = str(args[key])[:60].replace("\n", " ")
                            break
                    if not arg_summary and args:
                        first_val = next(iter(args.values()), "")
                        arg_summary = str(first_val)[:60].replace("\n", " ")
                except (json.JSONDecodeError, StopIteration, TypeError):
                    pass
                break

    if arg_summary:
        return f"[{tool_name}: {arg_summary}]"
    return f"[{tool_name}]"


def format_tree_entry(tree, entry, on_path: bool) -> str:
    """格式化单个树节点为一行显示文本。"""
    if entry.type == "compaction":
        return "[c] " + ((entry.summary or "")[:40].replace("\n", " "))
    if entry.role == "tool":
        return format_tool_entry(tree, entry)
    content = (entry.content or "")[:60].replace("\n", " ")
    bullet = "• " if on_path else "  "
    return f"{bullet}{entry.role}: {content}"


def render_tree_text(tree, leaf_id: str | None) -> str:
    """渲染会话树为 pi 风格的文本。

    主干路径扁平化（所有主干节点同缩进），分支处用 ├⊟/└⊟ 展开。
    """
    if not tree.root_id:
        return "(empty session)"

    # 构建叶子路径集合
    leaf_path_ids: set[str] = set()
    if leaf_id:
        node = tree.get(leaf_id)
        while node:
            leaf_path_ids.add(node.id)
            node = tree.get(node.parent_id) if node.parent_id else None

    lines: list[str] = []
    MAIN = "     "
    LEAF_MARKER = "› "

    def _render_subtree(eid: str, indent: str) -> None:
        """渲染一棵子树。同层节点共享 indent；分支时子节点用连接符+延续缩进。"""
        entry = tree.get(eid)
        if not entry:
            return

        children = tree.children_of(eid)
        on_path = eid in leaf_path_ids
        marker = LEAF_MARKER if eid == leaf_id else ""
        content = format_tree_entry(tree, entry, on_path)
        # off-path 节点在子树中不需要 bullet 占位
        if not on_path and content.startswith("  "):
            content = content[2:]
        # leaf 标记替换缩进前两字符
        if marker:
            display_indent = marker + indent[2:]
        else:
            display_indent = indent
        lines.append(f"{display_indent}{content}")

        if len(children) > 1:
            # 多个子节点 → 使用连接符展开
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                conn = "└⊟ " if is_last else "├⊟ "
                child_on_path = child.id in leaf_path_ids
                child_content = format_tree_entry(tree, child, child_on_path)
                # off-path 节点：连接符取代 bullet 前缀
                if not child_on_path and child_content.startswith("  "):
                    child_content = child_content[2:]
                lines.append(f"{indent}{conn}{child_content}")

                child_cont = indent + ("      " if is_last else "│     ")
                for gc in tree.children_of(child.id):
                    _render_subtree(gc.id, child_cont)
        elif len(children) == 1:
            _render_subtree(children[0].id, indent)

    _render_subtree(tree.root_id, MAIN)
    return "\n".join(lines)


def handle_tree(agent: Agent) -> None:
    """显示会话分支树。"""
    tree = agent.memory.tree
    if not tree.root_id:
        print("(空会话)")
        return
    print(render_tree_text(tree, agent.memory.leaf_id))


def handle_back(agent: Agent) -> None:
    """返回分叉之前的位置。

    每次 /fork 会保存跳转前的 leaf 位置，/back 弹出最近保存的位置并导航过去。
    """
    target = agent.memory.pop_fork()
    if target:
        entry = agent.memory.tree.get(target)
        content = (entry.content or "")[:60].replace("\n", " ") if entry else "(未知)"
        agent.rebuild_system_prompt()
        print(f"[back] 已返回: {content}")
    else:
        print("[back] 没有可返回的位置（fork 栈为空）")


def handle_fork(agent: Agent, command: str) -> None:
    """分叉到指定 user 消息。

    用法:
      /fork       → 分叉到倒数第 2 条用户消息之前（该消息及之后保留在旧分支）
      /fork N     → 分叉到第 N 条用户消息之前

    fork 到目标消息的 parent，而不是目标消息本身。
    这样新分支不包含被 fork 的那条用户消息，
    用户重新输入的问题能在干净的上下文中被 LLM 理解。
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
        # 默认：倒数第 2 条用户消息
        if len(all_entries) >= 2:
            target = all_entries[-2]
            display_num = len(all_entries) - 1  # 1-based: 倒数第 2 条
        else:
            target = all_entries[-1]
            display_num = len(all_entries)
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
        display_num = n

    # fork 到目标用户消息的 parent（即该消息之前），
    # 这样新分支不携带被 fork 的用户消息内容。
    fork_point = target.parent_id or agent.memory.tree.root_id
    if fork_point is None:
        print("[fork] 无法分叉：没有更早的消息")
        return
    agent.memory.push_fork()  # 保存当前位置，以便 /back 返回
    agent.memory.fork(fork_point)
    agent.rebuild_system_prompt()
    content = (target.content or "")[:60].replace("\n", " ")
    print(f"[fork] 已分叉到第 {display_num} 条消息之前: {content}")
