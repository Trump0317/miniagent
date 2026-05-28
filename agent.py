#!/usr/bin/env python3
"""miniagent CLI 入口 —— 交互模式和 print 模式。

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
import os
from pathlib import Path


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
    parser.add_argument(
        "-p", "--print",
        action="store_true",
        help="非交互模式：执行后打印结果并退出",
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="思考级别（off / minimal / low / medium / high / xhigh）",
    )
    parser.add_argument(
        "-nc", "--no-context-files",
        action="store_true",
        help="禁用 AGENTS.md / CLAUDE.md 上下文文件自动加载",
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="初始消息（空格连接）；支持 @file 引用",
    )

    args = parser.parse_args()

    # ── 组装初始消息：管道输入 + @file 展开 + 命令行参数 ──
    stdin_text = _read_stdin()
    cli_text = _expand_at_files(args.message)

    initial_parts = []
    if stdin_text:
        initial_parts.append(stdin_text)
    if cli_text:
        initial_parts.append(cli_text)
    initial_message = "\n\n".join(initial_parts) if initial_parts else ""

    # ── 加载上下文文件 ──
    from agent.context import load_context_files

    user_dir = Path.home() / ".miniagent"
    ctx = "" if args.no_context_files else load_context_files(user_dir=user_dir)

    # ── 构建 Agent ──
    from agent.config import AppConfig
    from agent.loop import Agent

    agent = Agent(
        config=AppConfig.from_env(context_files=ctx),
        thinking=args.thinking,
    )
    agent.run(initial_message=initial_message, print_mode=args.print)


if __name__ == "__main__":
    main()
