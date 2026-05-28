#!/usr/bin/env python3
"""miniagent CLI 外壳 —— 交互模式 / print 模式的 Harness 层。

用法:
    python agent.py                          # 交互模式
    python agent.py "列出当前目录文件"        # 交互模式（带初始消息）
    python agent.py -p "重构这个文件"         # print 模式：执行并退出
    cat README.md | python agent.py -p "总结" # print 模式 + 管道输入
    python agent.py @file.py "审查这个文件"   # @文件引用展开
    python agent.py --thinking high "..."     # 指定思考级别
    python agent.py -nc                       # 禁用上下文文件
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Harness 工具函数
# ═══════════════════════════════════════════════════════════════

def _expand_at_files(args: list[str]) -> str:
    """将 @file 引用展开为文件内容，其余参数用空格连接。"""
    parts: list[str] = []
    for arg in args:
        if arg.startswith("@"):
            filepath = Path(arg[1:]).expanduser().resolve()
            try:
                content = filepath.read_text(encoding="utf-8")
                parts.append(f"# 文件: {filepath.name}\n\n{content}")
            except Exception as e:
                parts.append(f"[无法读取 {filepath}: {e}]")
        else:
            parts.append(arg)
    return "\n\n".join(parts) if parts else ""


def _read_stdin() -> str:
    """如果 stdin 有管道数据，读取并返回；否则返回空字符串。"""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def _expand_command(prompt_loader, user_input: str) -> str:
    """将 /command query 展开为 prompt 模板。普通输入原样返回。"""
    if not user_input.startswith("/"):
        return user_input

    parts = user_input.split(maxsplit=1)
    name = parts[0][1:]
    query = parts[1] if len(parts) > 1 else ""

    resolved = prompt_loader.resolve(name, query)
    if resolved:
        print(f"[模板 /{name}] → {resolved[:60]}{'...' if len(resolved) > 60 else ''}")
        return resolved

    return user_input


# ═══════════════════════════════════════════════════════════════
# Harness 模式
# ═══════════════════════════════════════════════════════════════

def _print_startup_info(agent) -> None:
    """启动摘要"""
    cfg = agent.config
    lines = [f"[miniagent] 模型: {cfg.model}"]
    if agent.runner.llm.thinking:
        lines.append(f"  思考级别: {agent.runner.llm.thinking}")
    if cfg.context_files:
        lines.append("  上下文文件: 已加载 (AGENTS.md)")
    print("\n".join(lines))


def _run_print_mode(agent, msg: str) -> None:
    """非交互模式：处理一条消息，输出结果后退出。"""
    msg = _expand_command(agent.prompt_loader, msg)
    for chunk in agent.process(msg):
        print(chunk, end="", flush=True)
    print()
    _shutdown(agent)


def _run_interactive(agent, initial_message: str = "") -> None:
    """交互式主循环。"""
    # 展示可用命令
    cmds = agent.prompt_loader.list_commands()
    extra_cmds = "  /fork [n]  — 分叉到第 n 条用户消息\n  /tree      — 显示会话分支树"
    print((cmds + "\n" + extra_cmds) if cmds else extra_cmds)

    # 初始消息
    if initial_message:
        msg = _expand_command(agent.prompt_loader, initial_message)
        print(f"[You] : {initial_message}")
        print("[Assistant] : ", end="", flush=True)
        for chunk in agent.process(msg):
            print(chunk, end="", flush=True)
        print("\n")

    while True:
        try:
            user_input = input("[You] : ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_input.strip()
        if command.lower() in {"exit", "quit"}:
            break

        # ── 内置命令（不经过 LLM）──
        if command.startswith("/tree"):
            _handle_tree(agent)
            continue
        if command.startswith("/fork"):
            _handle_fork(agent, command)
            continue

        msg = _expand_command(agent.prompt_loader, command)

        print("[Assistant] : ", end="", flush=True)
        for chunk in agent.process(msg):
            print(chunk, end="", flush=True)
        print("\n")

    _shutdown(agent)


def _handle_tree(agent) -> None:
    """显示会话分支树。"""
    entries = agent.memory.get_tree_entries()
    if not entries:
        print("(空会话)")
        return

    # 构建 id → 序号映射
    sorted_entries = sorted(entries, key=lambda e: e.timestamp)
    id_to_num: dict[str, int] = {}
    user_count = 0
    for e in sorted_entries:
        if e.role == "user":
            user_count += 1
            id_to_num[e.id] = user_count

    leaf_id = agent.memory.leaf_id

    def _print_tree(entry_id: str, prefix: str, depth: int) -> None:
        entry = agent.memory.tree.get(entry_id)
        if not entry:
            return
        if depth > 30:
            return

        marker = " ← 当前" if entry_id == leaf_id else ""
        tag = "[压缩]" if entry.type == "compaction" else entry.role
        content = (entry.content or entry.summary or "")[:60].replace("\n", " ")
        num = id_to_num.get(entry_id)
        num_str = f"[{num}] " if num else ""
        print(f"{prefix}{num_str}{tag}: {content}{marker}")

        children = agent.memory.tree.children_of(entry_id)
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{prefix}{connector}")
            _print_tree(child.id, child_prefix, depth + 1)

    if agent.memory.tree.root_id:
        _print_tree(agent.memory.tree.root_id, "", 0)


def _handle_fork(agent, command: str) -> None:
    """分叉到指定 user 消息。

    用法:
      /fork       → fork 到最近一次 user 消息
      /fork N     → fork 到第 N 条 user 消息
    """
    parts = command.split(maxsplit=1)
    arg = parts[1] if len(parts) > 1 else ""

    # 找树中所有 user entry（按 timestamp 排序）
    all_entries = sorted(
        [e for e in agent.memory.get_tree_entries() if e.role == "user"],
        key=lambda e: e.timestamp,
    )

    if not all_entries:
        print("[fork] 没有可 fork 的用户消息")
        return

    if not arg:
        # 默认：fork 到倒数第 2 个 user 消息（如果有的话）
        if len(all_entries) >= 2:
            target = all_entries[-2]
        else:
            target = all_entries[-1]
    else:
        try:
            n = int(arg)
        except ValueError:
            print(f"[fork] 无效参数: {arg}，请输入序号")
            return
        # 1-based 序号
        if n < 1 or n > len(all_entries):
            print(f"[fork] 序号超出范围: 1-{len(all_entries)}")
            return
        target = all_entries[n - 1]

    agent.memory.fork(target.id)
    agent.history = agent.memory.history
    # 重建系统提示词
    system = agent._build_system_prompt(agent._skills, agent._agent_loader)
    agent.history.insert(0, {"role": "system", "content": system})

    content = (target.content or "")[:60].replace("\n", " ")
    print(f"[fork] 已分叉到: {content}")


def _shutdown(agent) -> None:
    """退出前：打印统计、压缩记忆"""
    result = agent.shutdown()
    stats = result["token_stats"]
    if stats:
        print("\n[Tokens] 本次会话 Token 消耗统计:")
        for m, s in stats.items():
            print(f"  - {m}: 输入 {s['input']}, 输出 {s['output']}, 缓存命中 {s['cache_hit']}")

    compact = result["compact"]
    if compact.get("summary") or compact.get("preferences"):
        print("[Memory] 已自动压缩并保存本次会话记录")
    print("退出对话")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="miniagent — 基于 LLM 的智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  agent.py                          交互模式
  agent.py "你好"                   交互模式，初始消息
  agent.py -p "列出文件"            print 模式
  cat README.md | agent.py -p "总结" 管道输入
  agent.py @code.py "审查这个文件"   @文件引用
  agent.py --thinking high "..."    思考级别控制
  agent.py -nc                      禁用上下文文件
        """,
    )
    parser.add_argument("-p", "--print", action="store_true",
                        help="非交互模式：执行后打印结果并退出")
    parser.add_argument("--thinking",
                        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
                        default=None,
                        help="思考级别（off / minimal / low / medium / high / xhigh）")
    parser.add_argument("-nc", "--no-context-files", action="store_true",
                        help="禁用 AGENTS.md / CLAUDE.md 上下文文件自动加载")
    parser.add_argument("message", nargs="*",
                        help="初始消息（空格连接）；支持 @file 引用")

    args = parser.parse_args()

    # ── 组装初始消息 ──
    stdin_text = _read_stdin()
    cli_text = _expand_at_files(args.message)
    initial_parts = [p for p in (stdin_text, cli_text) if p]
    initial_message = "\n\n".join(initial_parts) if initial_parts else ""

    # ── 加载上下文文件 ──
    from agent.ai.context import load_context_files
    from agent import AppConfig

    user_dir = Path.home() / ".miniagent"
    ctx = "" if args.no_context_files else load_context_files(user_dir=user_dir)

    # ── 构建 Agent 核心 ──
    from agent import Agent

    agent = Agent(
        config=AppConfig.from_env(
            context_files=ctx,
            restore_session=not args.print,
        ),
        thinking=args.thinking,
    )

    # 交互模式：新会话，重置 token 统计
    if not args.print:
        agent.tracker.reset_session()

    # ── 启动外壳 ──
    _print_startup_info(agent)

    if args.print:
        _run_print_mode(agent, initial_message)
    else:
        _run_interactive(agent, initial_message)


if __name__ == "__main__":
    main()
