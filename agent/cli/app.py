"""CLI 应用程序 — 交互/print 模式的完整外壳逻辑。

包含: readline 历史、智能输出缓冲（思考/工具/正文分发）、运行模式、main 入口。
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from ..core.chunks import ChunkType

# ── ANSI 颜色 ──
DIM = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
RESET = "\033[0m"

# ── readline: 命令历史和行编辑 ──
try:
    import readline
    _HISTFILE = Path.home() / ".miniagent" / ".history"
    _HISTFILE.parent.mkdir(parents=True, exist_ok=True)
    if _HISTFILE.exists():
        readline.read_history_file(str(_HISTFILE))
    readline.set_history_length(1000)

    # ── Tab 补全 ──
    _BUILTIN_COMMANDS = [
        "/help", "/clear", "/config", "/model ", "/thinking ",
        "/turns ", "/tree", "/fork ", "/back",
    ]
    _THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"]

    class _Completer:
        def __init__(self, agent_ref):
            self._agent_ref = agent_ref  # 弱引用，避免循环

        def complete(self, text: str, state: int) -> str | None:
            """readline completer 协议。"""
            if state == 0:
                self._matches = self._build_matches(text)
            try:
                return self._matches[state]
            except IndexError:
                return None

        def _build_matches(self, text: str) -> list[str]:
            if not text.startswith("/"):
                return []

            # ── 内置命令补全 ──
            if " " not in text:
                # 正在输入命令名
                return [c for c in _BUILTIN_COMMANDS if c.startswith(text)]

            # ── 子命令参数补全 ──
            parts = text.split(maxsplit=1)
            cmd = parts[0]
            partial = parts[1] if len(parts) > 1 else ""

            if cmd == "/thinking":
                return [l for l in _THINKING_LEVELS if l.startswith(partial)]

            if cmd == "/model":
                # 列出已知模型
                known = ["deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner",
                         "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                return [m for m in known if m.startswith(partial)]

            if cmd == "/fork":
                # 列出可 fork 的用户消息编号
                try:
                    agent = self._agent_ref() if callable(self._agent_ref) else self._agent_ref
                    if agent:
                        entries = agent.memory.get_tree_entries()
                        user_entries = [e for e in entries if e.role == "user"]
                        nums = [str(i) for i in range(1, len(user_entries) + 1)]
                        return [n for n in nums if n.startswith(partial)]
                except Exception:
                    pass
                return []

            return []

    _completer = _Completer(lambda: None)  # 延迟绑定 agent
    readline.set_completer(_completer.complete)
    readline.parse_and_bind("tab: complete")

except ImportError:
    readline = None


# ═══════════════════════════════════════════════════════════════
# 输入处理
# ═══════════════════════════════════════════════════════════════

def expand_at_files(args: list[str]) -> str:
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


def read_stdin() -> str:
    """如果 stdin 有管道数据，读取并返回；否则返回空字符串。"""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def read_multiline(first_line: str) -> str:
    """读取多行输入。空行结束；\\ 续行符号。

    用法：输入第一行后，后续行以空行结束；\\ 结尾表示续行。
    """
    lines = [first_line]
    prompt = "    "  # 续行缩进提示
    while True:
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            break
        if line == "":
            break  # 空行结束
        if line.rstrip().endswith("\\"):
            lines.append(line.rstrip()[:-1])
            continue
        lines.append(line)
    return "\n".join(lines)


def expand_command(prompt_loader, user_input: str) -> str:
    """将 /command query 展开为 prompt 模板。普通输入原样返回。"""
    if not user_input.startswith("/"):
        return user_input

    parts = user_input.split(maxsplit=1)
    name = parts[0][1:]
    query = parts[1] if len(parts) > 1 else ""

    resolved = prompt_loader.resolve(name, query)
    if resolved:
        preview = resolved[:60]
        print(f"[{CYAN}/{name}{RESET}] → {preview}{'...' if len(resolved) > 60 else ''}")
        return resolved

    return user_input


# ═══════════════════════════════════════════════════════════════
# 智能输出处理器 — 缓冲思考内容，格式化工具输出
# ═══════════════════════════════════════════════════════════════

class OutputHandler:
    """状态感知的流式输出处理器。

    将 chunk 流按类型分发到不同渲染通道：
      - REASONING: 灰色斜体，首次出现时打印 Thinking 标签
      - TEXT: 正文直接输出，切换到正文时刷新残留思考标记
      - TOOL_STATUS: 青色工具名，带缩进
      - TOOL_RESULT: 灰色缩进输出，长结果截断
      - DONE: 收尾，打印统计
    """

    def __init__(self, tracker=None):
        self._thinking_active = False   # 是否正在打印思考内容
        self._thinking_started = False  # 本轮是否有过思考
        self._tool_count = 0            # 本轮工具调用数
        self._tracker = tracker

    def write(self, chunk: dict) -> None:
        t = chunk.get("type", "")
        if t == ChunkType.REASONING:
            self._write_reasoning(chunk.get("content", ""))
        elif t == ChunkType.TEXT:
            self._write_text(chunk.get("content", ""))
        elif t == ChunkType.TOOL_STATUS:
            self._write_tool_status(chunk.get("content", ""))
        elif t == ChunkType.TOOL_RESULT:
            self._write_tool_result(chunk)
        elif t == ChunkType.DONE:
            self._write_done()

    def _write_reasoning(self, text: str) -> None:
        if not self._thinking_active:
            print(f"\n  {DIM}{ITALIC}··· Thinking ···{RESET}")
            self._thinking_active = True
            self._thinking_started = True
        print(f"{DIM}{text}{RESET}", end="", flush=True)

    def _write_text(self, text: str) -> None:
        if self._thinking_active:
            print(f"\n  {DIM}───{RESET}")
            self._thinking_active = False
        print(text, end="", flush=True)

    def _write_tool_status(self, content: str) -> None:
        if self._thinking_active:
            print(f"\n  {DIM}───{RESET}")
            self._thinking_active = False
        tool_name = content.strip().replace("\n", " ")
        self._tool_count += 1
        print(f"\n  {CYAN}{BOLD}#{self._tool_count} {tool_name}{RESET}", flush=True)

    def _write_tool_result(self, chunk: dict) -> None:
        result = chunk.get("result", "")
        # 长结果截断
        max_display = 2000
        if len(result) > max_display:
            truncated = result[:max_display]
            print(f"{DIM}{truncated}{RESET}")
            remaining = len(result) - max_display
            print(f"  {DIM}... [{remaining} more chars, {len(result)} total]{RESET}")
        elif result:
            print(f"{DIM}{result}{RESET}")

    def _write_done(self) -> None:
        if self._thinking_active:
            print(f"\n  {DIM}───{RESET}")
            self._thinking_active = False

    def reset(self) -> None:
        """重置本轮状态，准备下一轮。"""
        self._thinking_active = False
        self._thinking_started = False
        self._tool_count = 0


# ═══════════════════════════════════════════════════════════════
# 显示函数
# ═══════════════════════════════════════════════════════════════

def print_help(agent) -> None:
    """显示所有可用命令。"""
    print(f"\n{BOLD}内置命令:{RESET}")
    print("  /help           — 显示此帮助")
    print("  /clear          — 清屏")
    print("  /config         — 显示/修改配置")
    print("  /model [name]   — 显示或切换模型")
    print("  /thinking [lvl] — 显示或切换思考级别")
    print("  /turns [n]      — 显示或设置最大轮数")
    print("  /tree           — 显示会话分支树")
    print("  /fork [n]       — 分叉到第 n 条用户消息之前")
    print("  /back           — 返回分叉前的位置")
    cmds = agent.prompt_loader.list_commands()
    if cmds:
        print(f"\n{BOLD}模板命令:{RESET}")
        print(cmds)
    print(f"\n{BOLD}输入:{RESET} Enter 发送, Ctrl+C/Ctrl+D 退出, 空行结束多行输入")
    print(f"        \\ 续行, 或直接粘贴多行文本\n")


def print_config_info(agent, command: str = "/config") -> None:
    """显示当前配置信息。"""
    info = agent.get_config_info()
    print(f"\n  Provider:       {info['provider']}")
    print(f"  模型:           {info['model']}")
    print(f"  思考级别:       {info['thinking']}")
    if info["max_turns"] is not None:
        print(f"  最大轮数:       {info['max_turns']}")
    else:
        print(f"  最大轮数:       无限制")
    print(f"  max_tokens:     {info['max_tokens']:,}")
    print(f"  max_context:    {agent.config.max_context:,}")
    print(f"  压缩阈值:       {agent.config.compact_threshold:.0%}")
    print(f"  会话 ID:        {info['session_id'][:20]}...")
    print(f"  历史消息数:     {agent.memory.entry_count}")
    stats = agent.tracker.stats_by_model()
    if stats:
        total_in = sum(s["input"] for s in stats.values())
        total_out = sum(s["output"] for s in stats.values())
        total_cache = sum(s.get("cache_hit", 0) for s in stats.values())
        print(f"  Token:          输入 {total_in:,} / 输出 {total_out:,} / 缓存命中 {total_cache:,}")
    print()


def handle_model(agent, command: str) -> None:
    """/model [name] — 显示或切换模型。"""
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        print(f"  当前模型: {agent.config.model}")
        return
    new_model = parts[1].strip()
    old = agent.config.model
    agent.set_model(new_model)
    print(f"  {GREEN}已切换:{RESET} {old} → {new_model}")


def handle_thinking(agent, command: str) -> None:
    """/thinking [level] — 显示或切换思考级别。"""
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        current = agent.runner.llm.thinking or "off"
        print(f"  当前思考级别: {current}")
        print(f"  可选: off, minimal, low, medium, high, xhigh")
        return
    level = parts[1].strip()
    old = agent.runner.llm.thinking or "off"
    try:
        agent.set_thinking(level if level != "off" else None)
        new = agent.runner.llm.thinking or "off"
        print(f"  {GREEN}已切换:{RESET} {old} → {new}")
    except ValueError as e:
        print(f"  {RED}{e}{RESET}")


def handle_turns(agent, command: str) -> None:
    """/turns [n] — 显示或设置最大轮数（0 或不填 = 无限制）。"""
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        current = agent.config.max_turns
        if current is None:
            print(f"  当前最大轮数: 无限制")
        else:
            print(f"  当前最大轮数: {current}")
        return
    arg = parts[1].strip()
    old = agent.config.max_turns
    try:
        n = int(arg)
        agent.set_max_turns(n if n > 0 else None)
        new_label = f"{n}" if n > 0 else "无限制"
        old_label = f"{old}" if old is not None else "无限制"
        print(f"  {GREEN}已切换:{RESET} {old_label} → {new_label}")
    except ValueError:
        print(f"  {RED}无效参数: {arg}，请输入数字{RESET}")
    except Exception as e:
        print(f"  {RED}{e}{RESET}")


def print_startup_info(agent) -> None:
    """启动摘要"""
    cfg = agent.config
    parts = [f"{BOLD}miniagent{RESET} · {cfg.model}"]
    if agent.runner.llm.thinking:
        parts.append(f"思考:{agent.runner.llm.thinking}")
    parts.append(f"会话:{cfg.session_id[:12]}")
    print("  ".join(parts))
    print(f"输入 {CYAN}/help{RESET} 查看命令\n")


# ═══════════════════════════════════════════════════════════════
# 运行模式
# ═══════════════════════════════════════════════════════════════

def shutdown(agent) -> None:
    """退出前：打印统计、压缩记忆"""
    result = agent.shutdown()
    stats = result["token_stats"]
    if stats:
        print(f"\n{BOLD}Token 消耗:{RESET}")
        for m, s in stats.items():
            print(f"  {m}: 输入 {s['input']:,}, 输出 {s['output']:,},"
                  f" 缓存命中 {s.get('cache_hit', 0):,}")

    compact = result["compact"]
    if compact.get("summary") or compact.get("preferences"):
        print(f"{DIM}[Memory] 已自动压缩并保存本次会话记录{RESET}")
    # 保存命令历史
    if readline:
        readline.write_history_file(str(_HISTFILE))
    print("退出对话")


def run_print_mode(agent, msg: str) -> None:
    """非交互模式：处理一条消息，输出结果后退出。"""
    msg = expand_command(agent.prompt_loader, msg)
    handler = OutputHandler()
    for chunk in agent.process(msg):
        handler.write(chunk)
    print()
    shutdown(agent)


def run_interactive(agent, initial_message: str = "") -> None:
    """交互式主循环。"""
    from .helpers import handle_tree, handle_back, handle_fork

    # ── 绑定 completer 到当前 agent ──
    if readline:
        _completer._agent_ref = lambda: agent

    handler = OutputHandler()

    # 初始消息
    if initial_message:
        msg = expand_command(agent.prompt_loader, initial_message)
        print(f"\n{BOLD}You{RESET}  {initial_message}")
        for chunk in agent.process(msg):
            handler.write(chunk)
        print("\n")
        handler.reset()

    while True:
        try:
            user_input = input(f"{BOLD}You{RESET}  ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_input.strip()
        if command.lower() in {"exit", "quit"}:
            break

        # ── 内置命令（不经过 LLM）──
        if command in ("/help", "/?"):
            print_help(agent)
            continue
        if command == "/clear":
            os.system("clear" if os.name == "posix" else "cls")
            continue
        if command == "/session" or command == "/config":
            print_config_info(agent, command)
            continue
        if command.startswith("/model"):
            handle_model(agent, command)
            continue
        if command.startswith("/thinking"):
            handle_thinking(agent, command)
            continue
        if command.startswith("/turns"):
            handle_turns(agent, command)
            continue
        if command.startswith("/tree"):
            handle_tree(agent)
            continue
        if command.startswith("/back"):
            handle_back(agent)
            continue
        if command.startswith("/fork"):
            handle_fork(agent, command)
            continue

        # ── 多行输入检测 ──
        if "\n" in command:
            # 粘贴的多行文本，直接使用
            msg = expand_command(agent.prompt_loader, command)
        elif command.endswith("\\"):
            # 续行符：读取多行
            full = read_multiline(command.rstrip("\\"))
            msg = expand_command(agent.prompt_loader, full)
        else:
            msg = expand_command(agent.prompt_loader, command)

        try:
            for chunk in agent.process(msg):
                handler.write(chunk)
        except Exception as e:
            print(f"\n  {RED}Error: {e}{RESET}")
        print("\n")
        handler.reset()

    shutdown(agent)


# ═══════════════════════════════════════════════════════════════
# main 入口
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
    parser.add_argument("-r", "--restore", action="store_true",
                        help="恢复最近的会话历史")
    parser.add_argument("message", nargs="*",
                        help="初始消息（空格连接）；支持 @file 引用")

    args = parser.parse_args()

    # ── 组装初始消息 ──
    stdin_text = read_stdin()
    cli_text = expand_at_files(args.message)
    initial_parts = [p for p in (stdin_text, cli_text) if p]
    initial_message = "\n\n".join(initial_parts) if initial_parts else ""

    # ── 加载上下文文件 ──
    from agent.ai.context import load_context_files
    from agent import AppConfig

    user_dir = Path.home() / ".miniagent"
    ctx = "" if args.no_context_files else load_context_files(user_dir=user_dir)

    # ── 构建 Agent ──
    from agent import Agent

    agent = Agent(
        config=AppConfig.from_env(
            context_files=ctx,
            restore_session=args.restore,
        ),
        thinking=args.thinking,
    )

    # ── 启动外壳 ──
    print_startup_info(agent)

    if args.print:
        run_print_mode(agent, initial_message)
    else:
        run_interactive(agent, initial_message)
