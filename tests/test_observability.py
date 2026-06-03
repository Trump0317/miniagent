"""可观测性模块单元测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from agent.core.events import EventBus
from agent.core.observability import Observability


class TestObservability(unittest.TestCase):
    """Observability 事件记录测试。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_bus_and_obs(self) -> tuple[EventBus, Observability]:
        bus = EventBus()
        obs = Observability(log_dir=self.log_dir)
        obs.attach(bus)
        return bus, obs

    def _read_log(self) -> list[dict]:
        path = self.log_dir / "trace.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().strip().splitlines()]

    # ── 基本事件 ──

    def test_request_start_generates_trace_id(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hello"})
        self.assertNotEqual(obs.summary()["trace_id"], "")

    def test_request_start_logs_message_preview(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "a" * 300})
        logs = self._read_log()
        self.assertEqual(logs[0]["event"], "request:start")
        # 截断到 200 字符
        self.assertLess(len(logs[0]["data"]["message"]), 250)

    def test_turn_count_increments(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        self.assertEqual(obs.summary()["turns"], 0)
        bus.emit("turn:start", {"turn": 1})
        self.assertEqual(obs.summary()["turns"], 1)
        bus.emit("turn:start", {"turn": 2})
        self.assertEqual(obs.summary()["turns"], 2)

    def test_tool_call_counting(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        bus.emit("tool:before", {"name": "bash_tool", "args": {}})
        self.assertEqual(obs.summary()["tool_calls"], 1)
        bus.emit("tool:before", {"name": "read_tool", "args": {}})
        self.assertEqual(obs.summary()["tool_calls"], 2)

    def test_tool_failure_detection(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        bus.emit("tool:before", {"name": "test", "args": {}})
        bus.emit("tool:after", {"name": "test", "result": "error: something wrong"})
        self.assertEqual(obs.summary()["tool_failures"], 1)

    def test_tool_success_not_counted_as_failure(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        bus.emit("tool:before", {"name": "test", "args": {}})
        bus.emit("tool:after", {"name": "test", "result": "ok, done"})
        self.assertEqual(obs.summary()["tool_failures"], 0)

    def test_tool_failure_detection_chinese(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        bus.emit("tool:before", {"name": "test", "args": {}})
        bus.emit("tool:after", {"name": "test", "result": "执行错误: 超时"})
        self.assertEqual(obs.summary()["tool_failures"], 1)

    def test_subagent_tool_label(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        bus.emit("tool:before", {
            "name": "subagent_tool",
            "args": {"agent": "scout", "task": "investigate"},
        })
        logs = self._read_log()
        tool_logs = [l for l in logs if l["event"] == "tool:start"]
        self.assertIn("subagent:scout", tool_logs[0]["data"]["tool"])

    # ── 日志文件 ──

    def test_log_file_created(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hello"})
        logs = self._read_log()
        self.assertGreater(len(logs), 0)

    def test_all_entries_have_trace_id(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hello"})
        bus.emit("turn:start", {"turn": 1})
        bus.emit("turn:end", {})
        bus.emit("session:end", {})
        logs = self._read_log()
        for entry in logs:
            self.assertTrue(entry.get("trace"), f"Missing trace: {entry}")

    def test_same_trace_id_across_events(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hello"})
        bus.emit("turn:start", {"turn": 1})
        bus.emit("session:end", {})
        logs = self._read_log()
        trace_ids = {entry["trace"] for entry in logs}
        self.assertEqual(len(trace_ids), 1)

    def test_no_log_dir_no_file(self):
        """没有 log_dir 时不写文件也不崩溃。"""
        bus = EventBus()
        obs = Observability(log_dir=None)
        obs.attach(bus)
        bus.emit("message:received", {"text": "hi"})
        bus.emit("turn:start", {"turn": 1})
        bus.emit("session:end", {})
        self.assertIsNotNone(obs.summary())

    # ── summary ──

    def test_summary_fields(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hello"})
        s = obs.summary()
        for key in ("trace_id", "elapsed_ms", "turns", "tool_calls",
                     "tool_failures", "errors", "tokens_in", "tokens_out"):
            self.assertIn(key, s)

    def test_second_request_resets_counters(self):
        """新的 message:received 重置计数器。"""
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "first"})
        bus.emit("turn:start", {"turn": 1})
        bus.emit("turn:start", {"turn": 2})
        bus.emit("session:end", {})
        self.assertEqual(obs.summary()["turns"], 2)

        bus.emit("message:received", {"text": "second"})
        self.assertEqual(obs.summary()["turns"], 0)

    def test_context_high_event_logged(self):
        bus, obs = self._make_bus_and_obs()
        bus.emit("message:received", {"text": "hi"})
        bus.emit("context:high", {})
        logs = self._read_log()
        events = [l["event"] for l in logs]
        self.assertIn("context:high", events)
