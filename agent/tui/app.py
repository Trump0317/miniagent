"""miniagent TUI — 基于 prompt_toolkit + Rich 渲染，对齐 pi-tui 风格。"""

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
from prompt_toolkit.mouse_events import MouseEventType, MouseEvent
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from agent import Agent, AppConfig
from agent.ai.context import load_context_files
from agent.core.chunks import ChunkType


# ===========================================================
# Styles -- pi-tui 风格配色
# ===========================================================

STYLE = Style.from_dict({
    "header": "bg:#1a2b4a #ffffff bold",
    "status": "bg:#1a1a2e #888888",
    "input": "bg:#1a1a2e #cccccc",
    "input-border": "#3a3a5e",
    # 消息角色标签
    "user-label": "#5af bold",
    "agent-label": "#5f5 bold",
    "thinking-label": "#666 italic",
    "thinking-label-done": "#888",
    # 消息内容
    "thinking-text": "#888888 italic",
    "tool-title": "#5af",
    "tool-result": "#aaa",
    "tool-result-dim": "#666",
    "error-text": "#f66",
    # 分隔线
    "separator": "#333",
    "separator-text": "#555",
    # 状态
    "spinner": "#ff0",
})


# ===========================================================
# 消息模型
# ===========================================================

class Message:
    """单个聊天消息。工具状态+结果合并为一条。"""

    def __init__(self, role: str):
        self.role = role          # user / assistant / thinking / tool / error
        self.content = ""         # 文本内容
        self.tool_name = ""       # 工具名 (role == "tool")
        self.tool_result = ""     # 工具结果 (role == "tool")
        self.tool_collapsed = True
        self.thinking_collapsed = False
        self.done = False

    @property
    def display_text(self) -> str:
        if self.role == "tool":
            if self.tool_result and self.tool_collapsed:
                preview = self.tool_result[:200].replace("\n", " ")
                if len(self.tool_result) > 200:
                    preview += " ..."
                return preview
            return self.tool_result[:3000] if self.tool_result else self.content
        if self.role == "thinking" and self.thinking_collapsed:
            return self.content[:200].replace("\n", " ") + (" ..." if len(self.content) > 200 else "")
        return self.content

    @property
    def tool_byte_size(self) -> int:
        return len(self.tool_result.encode("utf-8")) if self.tool_result else 0

    @property
    def thinking_char_count(self) -> int:
        return len(self.content) if self.role == "thinking" else 0


# ===========================================================
# ScrollableChatControl -- 鼠标滚轮支持
# ===========================================================

class ScrollableChatControl(FormattedTextControl):
    """拦截鼠标滚轮事件，委托给回调更新光标位置驱动滚动。"""

    def __init__(self, *args, scroll_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._scroll_callback = scroll_callback

    def mouse_handler(self, mouse_event: MouseEvent):
        if self._scroll_callback:
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                self._scroll_callback(-1)
                return None
            elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                self._scroll_callback(1)
                return None
        return super().mouse_handler(mouse_event)


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

        self._app = self._build_app()

        if self._initial:
            self._process(self._initial)

    # -- UI 构建 --

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("c-g")
        def _(event: KeyPressEvent):
            event.app.exit()

        @kb.add("c-f")
        def _(event: KeyPressEvent):
            if not self._running:
                self._fork_dialog()

        @kb.add("c-b")
        def _(event: KeyPressEvent):
            if not self._running:
                self._do_back()

        @kb.add("c-t")
        def _(event: KeyPressEvent):
            if not self._running:
                self._toggle_tree()

        @kb.add("c-o")
        def _(event: KeyPressEvent):
            if not self._running:
                self._toggle_last_block()

        @kb.add("escape")
        def _(event: KeyPressEvent):
            if self._showing_tree:
                self._showing_tree = False
                event.app.invalidate()
            elif self._input.text.strip():
                self._input.text = ""

        @kb.add("pageup")
        def _(event: KeyPressEvent):
            if self._running:
                return
            page = max(5, self._chat_window.render_info.window_height // 2) if self._chat_window.render_info else 10
            self._cursor_y = max(0, self._cursor_y - page)
            self._auto_scroll = False
            event.app.invalidate()

        @kb.add("pagedown")
        def _(event: KeyPressEvent):
            if self._running or self._auto_scroll:
                return
            page = max(5, self._chat_window.render_info.window_height // 2) if self._chat_window.render_info else 10
            bottom = max(0, self._content_line_count - 1)
            self._cursor_y = min(bottom, self._cursor_y + page)
            if self._cursor_y >= bottom:
                self._auto_scroll = True
            event.app.invalidate()

        self._input = TextArea(
            height=3, prompt="> ",
            style="class:input",
            multiline=False, wrap_lines=True,
        )

        @kb.add("enter")
        def _(event: KeyPressEvent):
            text = self._input.text.strip()
            self._input.text = ""
            if text:
                self._showing_tree = False
                self._handle_input(text)

        self._chat_control = ScrollableChatControl(
            lambda: self._build_chat_text(),
            get_cursor_position=lambda: Point(x=0, y=max(0, self._cursor_y)),
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
            content=FormattedTextControl([("class:header", f" miniagent · {cfg.model} · {cfg.session_id[:12]} ")]),
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
        self._poll()

        if self._showing_tree:
            tree_text = self._build_tree_text()
            self._content_line_count = tree_text.count("\n") + 1
            if self._auto_scroll:
                self._cursor_y = max(0, self._content_line_count - 1)
            return [("", tree_text)]

        if not self._messages:
            self._content_line_count = 1
            self._cursor_y = 0
            return [("class:thinking-text", "Start a conversation...")]

        result: list[tuple[str, str]] = []
        line_count = 0

        for i, msg in enumerate(self._messages):
            if i > 0:
                # 在 user 消息前添加分隔线（新回合标记）
                if msg.role == "user":
                    result.append(("class:separator", "\n──╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌"))
                    line_count += 1
                else:
                    result.append(("", "\n"))
                    line_count += 1
            formatted = self._fmt_message(msg)
            result.extend(formatted)
            for _, text in formatted:
                line_count += text.count("\n")

        self._content_line_count = max(1, line_count)
        if self._auto_scroll:
            self._cursor_y = max(0, self._content_line_count - 1)
        else:
            self._cursor_y = min(self._cursor_y, max(0, self._content_line_count - 1))
        return result

    @staticmethod
    def _fmt_message(msg: Message) -> list[tuple[str, str]]:
        """格式化单条消息，对齐 pi-tui 风格。"""
        result: list[tuple[str, str]] = []

        if msg.role == "user":
            result.append(("class:user-label", "\nYou"))
        elif msg.role == "thinking":
            label = "Thinking..." if not msg.done else "Thought"
            style = "class:thinking-label" if not msg.done else "class:thinking-label-done"
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
            elif msg.role == "error":
                style = "class:error-text"
            elif msg.role == "tool":
                style = "class:tool-result" if msg.tool_result else "class:tool-result-dim"
            else:
                style = ""
            # 为每条消息添加左边距
            prefix = "\n  " if msg.role in ("tool", "thinking", "error") else "\n"
            result.append((style, f"{prefix}{text}"))
        return result

    def _render_status(self) -> list[tuple[str, str]]:
        if self._running:
            s = self.SPINNER[self._spinner % len(self.SPINNER)]
            return [("class:spinner", f" {s} Thinking...")]
        if self._showing_tree:
            return [("class:status", " Tree view -- Esc to return")]
        status = f" {self._status}"
        if not self._auto_scroll:
            status += " | PgUp/Dn or scroll to navigate"
        return [("class:status", status)]

    # -- 输入处理 --

    def _handle_input(self, text: str):
        if text in ("/quit", "/exit"):
            self._app.exit()
            return
        if text in ("/help", "/?"):
            self._show_help()
            return
        if text == "/clear":
            self._messages.clear()
            self._showing_tree = False
            return
        if text == "/session":
            self._update_status()
            return
        if text.startswith("/tree"):
            self._show_tree()
            return
        if text.startswith("/back"):
            self._do_back()
            return
        if text.startswith("/fork"):
            self._fork_dialog()
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

    def _poll(self):
        if not self._running:
            return

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
                # 合并到上一条 tool 消息（如果它还没有 result）
                tool_name = chunk.get("content", "").strip().replace("\n", " ")
                if (self._messages and self._messages[-1].role == "tool"
                        and not self._messages[-1].tool_result
                        and not self._messages[-1].tool_name):
                    self._messages[-1].tool_name = tool_name
                else:
                    m = Message("tool")
                    m.tool_name = tool_name
                    self._messages.append(m)
                updated = True

            elif t == ChunkType.TOOL_RESULT:
                result = chunk.get("result", "")
                # 找到最后一条 tool 消息填入结果
                for m in reversed(self._messages):
                    if m.role == "tool" and not m.tool_result:
                        m.tool_result = result
                        break
                else:
                    m = Message("tool")
                    m.tool_result = result
                    self._messages.append(m)
                self._auto_scroll = True
                updated = True

            elif t == "error":
                m = Message("tool")
                m.content = f"Error: {chunk.get('content', '')}"
                m.done = True
                self._messages.append(m)
                updated = True

        if updated:
            self._app.invalidate()

    # -- 鼠标滚轮 --

    def _handle_scroll(self, direction: int):
        if self._running:
            return
        win_h = self._chat_window.render_info.window_height if self._chat_window.render_info else 40
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

    # -- 展开/折叠 --

    def _toggle_last_block(self):
        """Ctrl+O: 切换最后一条工具结果或思考内容的折叠状态。"""
        for msg in reversed(self._messages):
            if msg.role == "tool" and msg.tool_result:
                msg.tool_collapsed = not msg.tool_collapsed
                act = "expanded" if not msg.tool_collapsed else "collapsed"
                self._status = f"Tool {msg.tool_name} {act} ({msg.tool_byte_size} bytes)"
                self._app.invalidate()
                return
            if msg.role == "thinking" and msg.content:
                msg.thinking_collapsed = not msg.thinking_collapsed
                act = "expanded" if not msg.thinking_collapsed else "collapsed"
                self._status = f"Thinking {act} ({msg.thinking_char_count} chars)"
                self._app.invalidate()
                return
        self._status = "Nothing to toggle"

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
        parts.append("Ctrl+G quit  Ctrl+O toggle  PgUp/Dn scroll")
        self._status = "  ".join(parts)

    def _show_help(self):
        m = Message("assistant")
        m.content = (
            "**miniagent TUI**\n\n"
            "Ctrl+G quit  |  Ctrl+F fork  |  Ctrl+B back  |  Ctrl+T tree\n"
            "Ctrl+O toggle (tool/thinking)  |  Esc clear/exit-tree\n"
            "PageUp/Down or mouse wheel to scroll\n\n"
            "Commands: /help  /clear  /session  /fork [n]  /back  /tree"
        )
        m.done = True
        self._messages.append(m)

    # -- 分叉 --

    def _fork_dialog(self):
        entries = self._agent.memory.get_tree_entries()
        user_entries = sorted(
            [(i, e) for i, e in enumerate(
                [e for e in entries if e.role == "user" and e.content], 1
            )],
            key=lambda x: x[1].timestamp,
        )
        if not user_entries:
            return
        m = Message("assistant")
        lines = ["**Fork targets:**"]
        for idx, e in user_entries:
            preview = (e.content or "")[:50].replace("\n", " ")
            lines.append(f"  [{idx}] {preview}")
        lines.append("")
        lines.append("Use /fork [n] to select")
        m.content = "\n".join(lines)
        m.done = True
        self._messages.append(m)
        self._showing_tree = False

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

    def _show_tree(self):
        self._showing_tree = True

    def _build_tree_text(self) -> str:
        entries = self._agent.memory.get_tree_entries()
        if not entries:
            self._showing_tree = False
            return "(empty session)"

        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        id_to_num: dict[str, int] = {}
        for i, e in enumerate([e for e in sorted_entries if e.role == "user"], 1):
            id_to_num[e.id] = i

        leaf_id = self._agent.memory.leaf_id
        lines = ["**Session Tree** (Ctrl+T or Esc to return):\n"]

        def _build(eid, prefix, depth):
            if depth > 40:
                return
            entry = self._agent.memory.tree.get(eid)
            if not entry:
                return
            marker = " <=" if eid == leaf_id else ""
            if entry.type == "compaction":
                s = (entry.summary or "")[:40].replace("\n", " ")
                lines.append(f"{prefix}[c] {s}{marker}")
            elif entry.role == "tool":
                lines.append(f"{prefix}[t]{marker}")
            else:
                num = id_to_num.get(entry.id)
                num_str = f"[{num}] " if num else ""
                c = (entry.content or "")[:50].replace("\n", " ")
                icon = {"user": "U", "assistant": "A"}.get(entry.role, ".")
                lines.append(f"{prefix}{num_str}{icon} {c}{marker}")
            children = self._agent.memory.tree.children_of(eid)
            for i, child in enumerate(children):
                last = i == len(children) - 1
                connector = "└── " if last else "├── "
                child_pf = prefix + ("    " if last else "│   ")
                lines.append(f"{prefix}{connector}")
                _build(child.id, child_pf, depth + 1)

        if self._agent.memory.tree.root_id:
            _build(self._agent.memory.tree.root_id, "", 0)
        return "\n".join(lines)

    def run(self):
        self._update_status()
        self._app.run()


# ===========================================================
# main
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="miniagent TUI")
    parser.add_argument("--thinking",
                        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
                        default=None, help="thinking level")
    parser.add_argument("-nc", "--no-context-files", action="store_true")
    parser.add_argument("-r", "--restore", action="store_true")
    parser.add_argument("message", nargs="*", help="initial message")
    args = parser.parse_args()

    user_dir = Path.home() / ".miniagent"
    ctx = "" if args.no_context_files else load_context_files(user_dir=user_dir)
    initial = " ".join(args.message) if args.message else ""

    agent = Agent(
        config=AppConfig.from_env(context_files=ctx, restore_session=args.restore),
        thinking=args.thinking,
    )
    TuiApp(agent, initial).run()


if __name__ == "__main__":
    main()
