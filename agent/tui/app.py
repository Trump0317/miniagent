"""miniagent TUI — 基于 prompt_toolkit，对齐 pi-tui 风格。"""

from __future__ import annotations
import argparse
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl
from prompt_toolkit.layout.containers import WindowAlign, ScrollOffsets
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.widgets import TextArea

from agent import Agent, AppConfig
from agent.ai.context import load_context_files
from agent.cli.helpers import render_tree_text
from agent.core.chunks import ChunkType

from .messages import Message, ScrollableChatControl, STYLE
from .render import render_markdown


# ===========================================================
# TuiApp
# ===========================================================

class TuiApp:
    SPINNER = ["◐", "◓", "◑", "◒"]

    def __init__(self, agent: Agent, initial_message: str = ""):
        self._agent = agent
        self._initial = initial_message

        self._messages: list[Message] = []
        self._current: Message | None = None
        self._queue: Queue = Queue()
        self._running = False
        self._spinner = 0
        self._status = ""
        self._showing_tree = False
        self._content_line_count = 0
        self._cursor_y = 0
        self._auto_scroll = True
        self._turn_count = 0
        # 渲染缓存
        self._dirty = True
        self._cached_lines: list[tuple[str, str]] = [
            ("class:thinking-text", "Start a conversation...")
        ]
        self._cached_line_count = 1

        self._app = self._build_app()

        if self._initial:
            self._process(self._initial)

    # -- UI 构建 --

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("escape")
        def _(event: KeyPressEvent):
            if self._showing_tree:
                self._showing_tree = False
                self._cursor_y = 0
                self._auto_scroll = True
                self._dirty = True
                event.app.invalidate()
            elif self._input.text.strip():
                self._input.text = ""

        @kb.add("c-o")
        def _(event: KeyPressEvent):
            pass  # TODO: 展开/折叠工具输出

        self._input = TextArea(
            height=4, prompt="> ",
            style="class:input",
            multiline=False, wrap_lines=True,
        )

        @kb.add("enter")
        def _(event: KeyPressEvent):
            text = self._input.text.strip()
            self._input.text = ""
            if text:
                self._showing_tree = False
                self._dirty = True
                self._handle_input(text)

        self._chat_control = ScrollableChatControl(
            lambda: self._build_chat_text(),
            get_cursor_position=lambda: Point(
                x=0,
                y=max(0, min(self._cursor_y,
                             max(0, self._content_line_count - 1))),
            ),
            scroll_callback=self._handle_scroll,
        )
        self._chat_window = Window(
            content=self._chat_control,
            wrap_lines=True,
            allow_scroll_beyond_bottom=False,
            always_hide_cursor=True,
            scroll_offsets=ScrollOffsets(top=3, bottom=3),
        )

        cfg = self._agent.config
        header = Window(
            content=FormattedTextControl([
                ("class:header",
                 f" miniagent · {cfg.model} · {cfg.session_id[:12]} "),
            ]),
            height=1, style="class:header",
        )

        self._status_window = Window(
            content=FormattedTextControl(self._render_status),
            height=1, style="class:status", align=WindowAlign.LEFT,
        )

        root = HSplit([
            header,
            self._chat_window,
            self._status_window,
            Window(height=1, char="─", style="class:input-border"),
            self._input,
        ])

        return Application(
            layout=Layout(root),
            key_bindings=kb,
            style=STYLE,
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.05,
        )

    # -- 渲染 --

    def _build_chat_text(self) -> list[tuple[str, str]]:
        try:
            self._poll()
        except Exception:
            pass

        if self._showing_tree:
            tree_text = self._build_tree_text()
            self._content_line_count = tree_text.count("\n") + 1
            self._cursor_y = 0
            self._cached_lines = [("", tree_text)]
            self._cached_line_count = self._content_line_count
            self._dirty = False
            return self._cached_lines

        if not self._messages:
            self._content_line_count = 1
            self._cursor_y = 0
            self._cached_lines = [
                ("class:thinking-text", "Start a conversation...")
            ]
            self._cached_line_count = 1
            self._dirty = False
            return self._cached_lines

        # 空闲时直接返回缓存
        if not self._dirty and not self._running:
            self._update_cursor()
            return self._cached_lines

        # 重建
        result: list[tuple[str, str]] = []
        line_count = 0
        first_visible = True

        for msg in self._messages:
            if msg.role == "tool":
                continue  # 默认不展示工具输出
            if not first_visible:
                if msg.role == "user":
                    result.append((
                        "class:separator",
                        "\n──╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌",
                    ))
                    line_count += 1
                else:
                    result.append(("", "\n"))
                    line_count += 1
            first_visible = False
            formatted = self._fmt_message(msg)
            result.extend(formatted)
            for _, text in formatted:
                line_count += text.count("\n")

        self._cached_lines = result
        self._cached_line_count = max(1, line_count)
        self._content_line_count = self._cached_line_count
        self._dirty = False
        self._update_cursor()
        return result

    def _update_cursor(self) -> None:
        if self._auto_scroll:
            self._cursor_y = max(0, self._content_line_count - 1)
        else:
            self._cursor_y = min(
                self._cursor_y, max(0, self._content_line_count - 1)
            )

    @staticmethod
    def _fmt_message(msg: Message) -> list[tuple[str, str]]:
        """格式化单条消息。对 assistant 消息应用 Markdown 渲染。"""
        result: list[tuple[str, str]] = []

        if msg.role == "user":
            result.append(("class:user-label", "\nYou"))
        elif msg.role == "thinking":
            label = "Thinking..." if not msg.done else "Thought"
            style = (
                "class:thinking-label"
                if not msg.done
                else "class:thinking-label-done"
            )
            if msg.thinking_collapsed:
                label += f" ({msg.thinking_char_count} chars)"
            result.append((style, f"\n  {label}"))
        elif msg.role == "tool":
            name = msg.tool_name or "tool"
            status = ""
            if not msg.tool_result:
                status = " running..."
            elif msg.tool_collapsed:
                status = f" [{msg.tool_byte_size} bytes]"
            result.append(("class:tool-title", f"\n  {name}{status}"))
        elif msg.role == "error":
            result.append(("class:error-text", "\n  Error"))
        else:
            result.append(("class:agent-label", "\nAgent"))

        text = msg.display_text
        if text:
            if msg.role == "thinking":
                style = "class:thinking-text"
                prefix = "\n  "
                result.append((style, f"{prefix}{text}"))
            elif msg.role == "error":
                prefix = "\n  "
                result.append(("class:error-text", f"{prefix}{text}"))
            elif msg.role == "tool":
                style = (
                    "class:tool-result"
                    if msg.tool_result
                    else "class:tool-result-dim"
                )
                prefix = "\n  "
                result.append((style, f"{prefix}{text}"))
            elif msg.role == "assistant":
                # Markdown 渲染
                result.append(("", "\n"))
                result.extend(render_markdown(text))
            else:
                result.append(("", f"\n{text}"))
        return result

    def _render_status(self) -> list[tuple[str, str]]:
        if self._running:
            s = self.SPINNER[self._spinner % len(self.SPINNER)]
            return [("class:spinner", f" {s} Thinking...")]
        if self._showing_tree:
            return [("class:status", " Tree view -- Esc to return")]

        cfg = self._agent.config
        stats = self._agent.tracker.stats_by_model()
        token_info = ""
        if stats:
            tin = sum(s["input"] for s in stats.values())
            tout = sum(s["output"] for s in stats.values())
            token_info = f" | Tokens {tin:,}+{tout:,}"

        status = (
            f" {cfg.model} · {self._agent.memory.entry_count} msgs{token_info}"
        )
        return [("class:status", status)]

    # -- 输入处理 --

    def _handle_input(self, text: str):
        if text in ("/quit", "/exit"):
            self._app.exit()
            return
        if text in ("/help", "/?"):
            self._dirty = True
            self._show_help()
            return
        if text == "/clear":
            self._messages.clear()
            self._showing_tree = False
            self._dirty = True
            return
        if text in ("/session", "/config"):
            self._dirty = True
            self._show_config()
            return
        if text.startswith("/model"):
            self._dirty = True
            self._handle_model(text)
            return
        if text.startswith("/thinking"):
            self._dirty = True
            self._handle_thinking(text)
            return
        if text.startswith("/turns"):
            self._dirty = True
            self._handle_turns(text)
            return
        if text.startswith("/tree"):
            self._show_tree()
            return
        if text.startswith("/back"):
            self._do_back()
            return
        if text.startswith("/fork"):
            self._dirty = True
            self._handle_fork(text)
            return
        self._process(text)

    # -- 核心处理 --

    def _process(self, text: str):
        if self._running:
            return
        text = self._expand(text)

        um = Message("user")
        um.content = text
        um.done = True
        self._messages.append(um)
        self._turn_count += 1
        self._auto_scroll = True
        self._dirty = True
        self._app.invalidate()

        self._running = True
        self._current = None
        self._queue = Queue()
        Thread(target=self._run_agent, args=(text,), daemon=True).start()

    def _run_agent(self, text: str):
        try:
            for chunk in self._agent.process(text):
                self._queue.put(chunk)
        except Exception as e:
            self._queue.put({"type": "error", "content": str(e)})
        finally:
            self._queue.put(None)

    def _poll(self) -> bool:
        """处理队列中的 chunk，返回是否有内容变化。"""
        if not self._running:
            return False

        self._spinner += 1
        processed = 0
        updated = False

        while not self._queue.empty() and processed < 20:
            processed += 1
            chunk = self._queue.get_nowait()

            if chunk is None:
                self._running = False
                if self._current:
                    self._current.done = True
                self._current = None
                self._update_status()
                updated = True
                break

            t = chunk.get("type", "")

            if t == ChunkType.TEXT:
                if self._current is None or self._current.role == "thinking":
                    self._current = Message("assistant")
                    self._messages.append(self._current)
                elif self._current.role == "tool":
                    self._current.tool_result = (
                        (self._current.tool_result or "")
                        + chunk.get("content", "")
                    )
                    self._auto_scroll = True
                    updated = True
                    continue
                self._current.content += chunk.get("content", "")
                self._auto_scroll = True
                updated = True

            elif t == ChunkType.REASONING:
                if self._current is None or self._current.role != "thinking":
                    self._current = Message("thinking")
                    self._messages.append(self._current)
                self._current.content += chunk.get("content", "")
                self._auto_scroll = True
                updated = True

            elif t == ChunkType.TOOL_STATUS:
                tool_name = chunk.get("content", "").strip().replace("\n", " ")
                is_done = "✓" in chunk.get("content", "")
                if (
                    self._messages
                    and self._messages[-1].role == "tool"
                    and not self._messages[-1].tool_result
                    and not self._messages[-1].tool_name
                ):
                    self._messages[-1].tool_name = tool_name
                    self._current = self._messages[-1]
                else:
                    m = Message("tool")
                    m.tool_name = tool_name
                    self._messages.append(m)
                    self._current = m
                if is_done:
                    self._current = None
                updated = True

            elif t == ChunkType.TOOL_RESULT:
                result = chunk.get("result", "")
                for m in reversed(self._messages):
                    if m.role == "tool" and not m.tool_result:
                        m.tool_result = result
                        break
                else:
                    m = Message("tool")
                    m.tool_result = result
                    self._messages.append(m)
                self._current = None
                self._auto_scroll = True
                updated = True

            elif t == "error":
                m = Message("tool")
                m.content = f"Error: {chunk.get('content', '')}"
                m.done = True
                self._messages.append(m)
                updated = True

        if updated:
            self._dirty = True
            self._app.invalidate()
        return updated

    # -- 鼠标滚轮 --

    def _handle_scroll(self, direction: int):
        if self._running:
            return
        win_h = (
            self._chat_window.render_info.window_height
            if self._chat_window.render_info
            else 40
        )
        step = max(3, win_h // 3)
        if direction < 0:
            self._cursor_y = max(0, self._cursor_y - step)
            self._auto_scroll = False
        else:
            bottom = max(0, self._content_line_count - 1)
            self._cursor_y = min(bottom, self._cursor_y + step)
            if self._cursor_y >= bottom:
                self._auto_scroll = True
        self._app.invalidate()

    # -- 命令 --

    def _expand(self, text: str) -> str:
        if not text.startswith("/"):
            return text
        parts = text.split(maxsplit=1)
        name = parts[0][1:]
        query = parts[1] if len(parts) > 1 else ""
        r = self._agent.prompt_loader.resolve(name, query)
        return r if r else text

    def _update_status(self):
        cfg = self._agent.config
        mem = self._agent.memory
        t = self._agent.tracker
        stats = t.stats_by_model()
        parts = [f"Session {cfg.session_id[:12]} | {mem.entry_count} msgs"]
        if stats:
            tin = sum(s["input"] for s in stats.values())
            tout = sum(s["output"] for s in stats.values())
            parts.append(f"Tokens {tin}+{tout}")
        self._status = "  ".join(parts)

    def _show_help(self):
        m = Message("assistant")
        cmds = self._agent.prompt_loader.list_commands()
        cmd_text = ""
        if cmds:
            cmd_text = f"\n模板命令:\n{cmds}\n"
        m.content = (
            "**miniagent TUI**\n\n"
            "Esc 清空输入 / 退出树视图 · 鼠标滚轮翻页\n"
            "/help           — 显示此帮助\n"
            "/clear          — 清屏\n"
            "/config         — 显示当前配置和 token 统计\n"
            "/model [name]   — 显示或切换模型\n"
            "/thinking [lvl] — 显示或切换思考级别\n"
            "/turns [n]      — 显示或设置最大轮数（0=无限制）\n"
            "/tree           — 显示会话分支树\n"
            "/fork [n]       — 分叉到第 n 条消息之前\n"
            "/back           — 返回分叉前的位置\n"
            + cmd_text +
            "\nEnter 发送消息，/quit 退出"
        )
        m.done = True
        self._messages.append(m)

    def _show_config(self):
        cfg = self._agent.config
        info = self._agent.get_config_info()
        t = self._agent.tracker
        stats = t.stats_by_model()
        lines = [
            f"Provider: {cfg.provider}",
            f"Model: {info['model']}",
            f"Thinking: {info['thinking']}",
            f"Max turns: {info['max_turns'] or 'unlimited'}",
            f"Max tokens: {info['max_tokens']:,}",
            f"Session: {info['session_id'][:12]}",
        ]
        if stats:
            tin = sum(s["input"] for s in stats.values())
            tout = sum(s["output"] for s in stats.values())
            cache = sum(s.get("cache_hit", 0) for s in stats.values())
            lines.append(f"Tokens: in={tin:,} out={tout:,} cache={cache:,}")
        m = Message("assistant")
        m.content = "\n".join(lines)
        m.done = True
        self._messages.append(m)

    def _handle_model(self, text: str):
        parts = text.split(maxsplit=1)
        m = Message("assistant")
        if len(parts) == 1:
            m.content = f"Current model: {self._agent.config.model}"
        else:
            new_model = parts[1].strip()
            old = self._agent.config.model
            try:
                self._agent.set_model(new_model)
                m.content = f"Model: {old} → {new_model}"
            except Exception as e:
                m.content = f"Error: {e}"
        m.done = True
        self._messages.append(m)

    def _handle_thinking(self, text: str):
        parts = text.split(maxsplit=1)
        m = Message("assistant")
        current = self._agent.runner.llm.thinking or "off"
        if len(parts) == 1:
            m.content = (
                f"Thinking: {current} (off/minimal/low/medium/high/xhigh)"
            )
        else:
            level = parts[1].strip()
            try:
                self._agent.set_thinking(level if level != "off" else None)
                new = self._agent.runner.llm.thinking or "off"
                m.content = f"Thinking: {current} → {new}"
            except ValueError as e:
                m.content = str(e)
        m.done = True
        self._messages.append(m)

    def _handle_turns(self, text: str):
        parts = text.split(maxsplit=1)
        m = Message("assistant")
        current = self._agent.config.max_turns
        if len(parts) == 1:
            m.content = f"Max turns: {current or 'unlimited'}"
        else:
            try:
                n = int(parts[1].strip())
                self._agent.set_max_turns(n if n > 0 else None)
                m.content = (
                    f"Max turns: {current or 'unlimited'}"
                    f" → {n if n > 0 else 'unlimited'}"
                )
            except (ValueError, Exception) as e:
                m.content = f"Error: {e}"
        m.done = True
        self._messages.append(m)

    # -- 分叉 --

    def _handle_fork(self, text: str):
        entries = self._agent.memory.get_tree_entries()
        user_entries = sorted(
            [e for e in entries if e.role == "user" and e.content],
            key=lambda e: e.timestamp,
        )
        if not user_entries:
            return

        parts = text.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        if arg:
            try:
                n = int(arg)
            except ValueError:
                m = Message("assistant")
                m.content = f"Invalid fork target: {arg}"
                m.done = True
                self._messages.append(m)
                return
            if n < 1 or n > len(user_entries):
                m = Message("assistant")
                m.content = f"Fork target out of range: 1-{len(user_entries)}"
                m.done = True
                self._messages.append(m)
                return
            target = user_entries[n - 1]
            fork_point = target.parent_id or self._agent.memory.tree.root_id
            if fork_point is None:
                return
            self._agent.memory.push_fork()
            self._agent.memory.fork(fork_point)
            self._agent.rebuild_system_prompt()
            self._messages.clear()
            self._status = f"Forked to message #{n}"
            return

        m = Message("assistant")
        lines = ["**Fork targets (use /fork [n]):**"]
        for i, e in enumerate(user_entries, 1):
            preview = (e.content or "")[:50].replace("\n", " ")
            lines.append(f"  [{i}] {preview}")
        m.content = "\n".join(lines)
        m.done = True
        self._messages.append(m)

    def _do_back(self):
        target = self._agent.memory.pop_fork()
        if target:
            self._agent.rebuild_system_prompt()
            self._status = "Returned"
        else:
            self._status = "No fork to return from"

    # -- 分支树 --

    def _toggle_tree(self):
        self._showing_tree = not self._showing_tree
        self._dirty = True

    def _show_tree(self):
        self._showing_tree = True
        self._dirty = True
        self._cursor_y = 0
        self._auto_scroll = True

    def _build_tree_text(self) -> str:
        try:
            return self._build_tree_text_impl()
        except Exception as e:
            self._showing_tree = False
            return f"(tree error: {e})"

    def _build_tree_text_impl(self) -> str:
        tree = self._agent.memory.tree
        leaf_id = self._agent.memory.leaf_id
        if not tree.root_id:
            self._showing_tree = False
            return "(empty session)"
        return "**Session Tree** (Esc to return):\n" + render_tree_text(
            tree, leaf_id
        )

    def run(self):
        self._update_status()
        self._app.run()


# ===========================================================
# main
# ===========================================================


def main():
    parser = argparse.ArgumentParser(description="miniagent TUI")
    parser.add_argument(
        "--thinking",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="thinking level",
    )
    parser.add_argument("-nc", "--no-context-files", action="store_true")
    parser.add_argument("-r", "--restore", action="store_true")
    parser.add_argument("message", nargs="*", help="initial message")
    args = parser.parse_args()

    user_dir = Path.home() / ".miniagent"
    ctx = "" if args.no_context_files else load_context_files(user_dir=user_dir)
    initial = " ".join(args.message) if args.message else ""

    agent = Agent(
        config=AppConfig.from_env(context_files=ctx,
                                 restore_session=args.restore),
        thinking=args.thinking,
    )
    TuiApp(agent, initial).run()


if __name__ == "__main__":
    main()
