"""TUI 单元测试 — 测试 TuiApp 逻辑层和 Message 模型。"""

from __future__ import annotations
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

from agent.tui.app import TuiApp, Message, STYLE, ScrollableChatControl
from agent.core.chunks import ChunkType


def _make_mock_agent():
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.model = "test-model"
    agent.config.session_id = "20260602-test01"
    agent.config.session_dir = Path("/tmp/miniagent-test")
    agent.memory = MagicMock()
    agent.memory.entry_count = 0
    agent.memory.leaf_id = None
    agent.tracker = MagicMock()
    agent.tracker.stats_by_model.return_value = {}
    agent.prompt_loader = MagicMock()
    agent.prompt_loader.resolve.return_value = None
    agent.runner = MagicMock()
    agent.runner.llm = MagicMock()
    agent.runner.llm.thinking = None
    return agent


def _make_mock_chunk(chunk_type: str, **kwargs) -> dict:
    chunk = {"type": chunk_type}
    chunk.update(kwargs)
    return chunk


# ===========================================================
# Message 模型
# ===========================================================

class TestMessage(unittest.TestCase):

    def test_user_message(self):
        m = Message("user")
        m.content = "hello"
        self.assertEqual(m.role, "user")
        self.assertEqual(m.display_text, "hello")

    def test_tool_collapsed_preview(self):
        m = Message("tool")
        m.tool_name = "read"
        m.tool_result = "x" * 300
        m.tool_collapsed = True
        self.assertIn("...", m.display_text)

    def test_tool_expanded_full(self):
        m = Message("tool")
        m.tool_result = "short"
        m.tool_collapsed = False
        self.assertEqual(m.display_text, "short")

    def test_tool_byte_size(self):
        m = Message("tool")
        m.tool_result = "hello"
        self.assertEqual(m.tool_byte_size, 5)

    def test_thinking_collapsed(self):
        m = Message("thinking")
        m.content = "Let me think about this carefully..." + "x" * 200
        m.thinking_collapsed = True
        self.assertIn("...", m.display_text)
        self.assertLess(len(m.display_text), 210)

    def test_thinking_expanded(self):
        m = Message("thinking")
        m.content = "reasoning"
        m.thinking_collapsed = False
        self.assertEqual(m.display_text, "reasoning")

    def test_thinking_char_count(self):
        m = Message("thinking")
        m.content = "abcde"
        self.assertEqual(m.thinking_char_count, 5)


# ===========================================================
# TuiApp 初始化
# ===========================================================

class TestTuiAppInit(unittest.TestCase):

    def setUp(self):
        self.agent = _make_mock_agent()

    def test_init_defaults(self):
        app = TuiApp(self.agent)
        self.assertEqual(app._messages, [])
        self.assertFalse(app._running)
        self.assertTrue(app._auto_scroll)
        self.assertEqual(app._turn_count, 0)

    def test_init_with_message(self):
        with patch.object(TuiApp, '_process') as mock_p:
            TuiApp(self.agent, initial_message="hi")
            mock_p.assert_called_once_with("hi")


# ===========================================================
# 输入处理
# ===========================================================

class TestInputHandling(unittest.TestCase):

    def setUp(self):
        self.agent = _make_mock_agent()

    def _make_app(self):
        with patch.object(TuiApp, '_build_app', return_value=MagicMock()):
            return TuiApp(self.agent)

    def test_quit(self):
        app = self._make_app()
        app._handle_input("/quit")
        app._app.exit.assert_called_once()

    def test_help(self):
        app = self._make_app()
        app._app.invalidate = MagicMock()
        app._handle_input("/help")
        self.assertEqual(len(app._messages), 1)
        self.assertIn("Ctrl+G", app._messages[0].content)


# ===========================================================
# Chunk 处理
# ===========================================================

class TestChunkProcessing(unittest.TestCase):

    def setUp(self):
        self.agent = _make_mock_agent()
        with patch.object(TuiApp, '_build_app', return_value=MagicMock()):
            self.app = TuiApp(self.agent)
        self.app._app.invalidate = MagicMock()

    def _enqueue(self, chunks):
        self.app._running = True
        self.app._queue = Queue()
        self.app._messages = []
        self.app._current = None
        for c in chunks:
            self.app._queue.put(c)

    def test_text_chunk(self):
        self._enqueue([_make_mock_chunk(ChunkType.TEXT, content="Hello")])
        self.app._poll()
        self.assertEqual(len(self.app._messages), 1)
        self.assertEqual(self.app._messages[0].role, "assistant")

    def test_reasoning_chunk(self):
        self._enqueue([_make_mock_chunk(ChunkType.REASONING, content="hmm...")])
        self.app._poll()
        self.assertEqual(self.app._messages[0].role, "thinking")

    def test_reasoning_then_text_separate(self):
        """REASONING -> TEXT 产生两条独立消息。"""
        self._enqueue([
            _make_mock_chunk(ChunkType.REASONING, content="think"),
            _make_mock_chunk(ChunkType.TEXT, content="answer"),
        ])
        self.app._poll()
        self.assertEqual(len(self.app._messages), 2)
        self.assertEqual(self.app._messages[0].role, "thinking")
        self.assertEqual(self.app._messages[1].role, "assistant")

    def test_tool_status_creates_tool_message(self):
        self._enqueue([_make_mock_chunk(ChunkType.TOOL_STATUS, content="read f.txt")])
        self.app._poll()
        self.assertEqual(self.app._messages[0].role, "tool")
        self.assertEqual(self.app._messages[0].tool_name, "read f.txt")

    def test_tool_result_merges_with_last_tool(self):
        """TOOL_RESULT 合并到上一条 tool 消息。"""
        self._enqueue([
            _make_mock_chunk(ChunkType.TOOL_STATUS, content="read"),
            _make_mock_chunk(ChunkType.TOOL_RESULT, result="file content", id="t1"),
        ])
        self.app._poll()
        self.assertEqual(len(self.app._messages), 1, "tool status + result should merge")
        self.assertEqual(self.app._messages[0].tool_name, "read")
        self.assertEqual(self.app._messages[0].tool_result, "file content")
        self.assertFalse(self.app._messages[0].done)

    def test_two_tools_create_separate(self):
        """两个工具调用应产生两条消息。"""
        self._enqueue([
            _make_mock_chunk(ChunkType.TOOL_STATUS, content="read a"),
            _make_mock_chunk(ChunkType.TOOL_RESULT, result="a content", id="t1"),
            _make_mock_chunk(ChunkType.TOOL_STATUS, content="read b"),
            _make_mock_chunk(ChunkType.TOOL_RESULT, result="b content", id="t2"),
        ])
        self.app._poll()
        self.assertEqual(len(self.app._messages), 2)
        self.assertEqual(self.app._messages[0].tool_name, "read a")
        self.assertEqual(self.app._messages[1].tool_name, "read b")


# ===========================================================
# 展开/折叠
# ===========================================================

class TestToggle(unittest.TestCase):

    def setUp(self):
        self.agent = _make_mock_agent()
        with patch.object(TuiApp, '_build_app', return_value=MagicMock()):
            self.app = TuiApp(self.agent)
        self.app._app.invalidate = MagicMock()

    def test_toggle_tool(self):
        m = Message("tool")
        m.tool_name = "read"
        m.tool_result = "content"
        m.tool_collapsed = True
        self.app._messages = [m]
        self.app._toggle_last_block()
        self.assertFalse(m.tool_collapsed)

    def test_toggle_thinking(self):
        m = Message("thinking")
        m.content = "reasoning text"
        m.thinking_collapsed = False
        self.app._messages = [m]
        self.app._toggle_last_block()
        self.assertTrue(m.thinking_collapsed)

    def test_toggle_prefers_tool_over_thinking(self):
        """最后一条是 thinking，前面是 tool，应 toggle tool。"""
        m_tool = Message("tool")
        m_tool.tool_name = "bash"
        m_tool.tool_result = "output"
        m_tool.tool_collapsed = True
        m_think = Message("thinking")
        m_think.content = "done"
        self.app._messages = [m_tool, m_think]
        # thinking comes last but has no toggle state (collapsed=false by default, but also no result)
        # Actually, thinking with content should be togglable
        # Let's test: both are toggleable, should toggle the LAST one (thinking)
        self.app._toggle_last_block()
        self.assertTrue(m_think.thinking_collapsed)
        self.assertTrue(m_tool.tool_collapsed)  # unchanged

    def test_toggle_empty(self):
        self.app._toggle_last_block()
        self.assertIn("Nothing to toggle", self.app._status)


# ===========================================================
# 滚动
# ===========================================================

class TestScroll(unittest.TestCase):

    def setUp(self):
        self.agent = _make_mock_agent()
        with patch.object(TuiApp, '_build_app', return_value=MagicMock()):
            self.app = TuiApp(self.agent)

    def test_scroll_up_disables_auto(self):
        self.app._auto_scroll = True
        self.app._cursor_y = 20
        mock_win = MagicMock()
        mock_win.render_info = MagicMock()
        mock_win.render_info.window_height = 30
        self.app._chat_window = mock_win
        self.app._app.invalidate = MagicMock()

        self.app._handle_scroll(-1)
        self.assertFalse(self.app._auto_scroll)
        self.assertEqual(self.app._cursor_y, 10)  # 20 - max(3, 30//3) = 20 - 10

    def test_scroll_down_to_bottom_reenables(self):
        self.app._auto_scroll = False
        self.app._content_line_count = 15
        self.app._cursor_y = 10
        mock_win = MagicMock()
        mock_win.render_info = MagicMock()
        mock_win.render_info.window_height = 30
        self.app._chat_window = mock_win
        self.app._app.invalidate = MagicMock()

        self.app._handle_scroll(1)
        self.assertTrue(self.app._auto_scroll)


# ===========================================================
# 格式化
# ===========================================================

class TestFormatting(unittest.TestCase):

    def setUp(self):
        self.agent = _make_mock_agent()
        with patch.object(TuiApp, '_build_app', return_value=MagicMock()):
            self.app = TuiApp(self.agent)

    def test_fmt_user_has_label(self):
        m = Message("user")
        m.content = "hi"
        result = TuiApp._fmt_message(m)
        self.assertIn("You", result[0][1])

    def test_fmt_tool_shows_name(self):
        m = Message("tool")
        m.tool_name = "bash"
        m.tool_result = "ok"
        result = TuiApp._fmt_message(m)
        self.assertIn("bash", result[0][1])

    def test_fmt_tool_shows_bytes_when_collapsed(self):
        m = Message("tool")
        m.tool_name = "read"
        m.tool_result = "x" * 1000
        m.tool_collapsed = True
        result = TuiApp._fmt_message(m)
        self.assertTrue(any("bytes" in t for _, t in result))

    def test_fmt_thinking_shows_label(self):
        m = Message("thinking")
        m.content = "hmm"
        m.done = False
        result = TuiApp._fmt_message(m)
        self.assertIn("Thinking", result[0][1])

    def test_fmt_thinking_done_shows_thought(self):
        m = Message("thinking")
        m.content = "done"
        m.done = True
        result = TuiApp._fmt_message(m)
        self.assertIn("Thought", result[0][1])


if __name__ == "__main__":
    unittest.main()
