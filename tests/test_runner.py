"""AgentRunner 单元测试."""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.runner import AgentRunner


class TestAgentRunner(unittest.TestCase):
    """AgentRunner think-act 循环测试."""

    def setUp(self):
        self.llm = MagicMock()
        self.tools = MagicMock()
        self.tools.registry = MagicMock()
        self.tools.registry.get_tool_schemas.return_value = []
        self.bus = MagicMock()
        self.tracker = MagicMock()

    def _make_runner(self, max_turns=None):
        return AgentRunner(
            llm_client=self.llm,
            tool_executor=self.tools,
            token_tracker=self.tracker,
            event_bus=self.bus,
            max_turns=max_turns,
        )

    def _make_stream_chunks(self, *chunks):
        """构造 llm.stream 返回值."""
        self.llm.stream.return_value = iter(chunks)

    # ── 纯文本响应 ──

    def test_simple_text_response(self):
        """无工具调用时逐块产出文本并结束."""
        self._make_stream_chunks(
            {"type": "content", "text": "Hello "},
            {"type": "content", "text": "World"},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "hi"}]

        output = list(runner.step(history))
        self.assertEqual(output, ["Hello ", "World"])

        # 验证 assistant 消息被追加
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hello World")

    def test_simple_text_emits_turn_end(self):
        """无工具调用时 emit turn:end."""
        self._make_stream_chunks(
            {"type": "content", "text": "ok"},
        )
        runner = self._make_runner()
        list(runner.step([{"role": "user", "content": "hi"}]))
        self.bus.emit.assert_any_call("turn:end", {"text": "ok"})

    # ── reasoning ──

    def test_reasoning_content(self):
        """reasoning 被累积但不出现在输出中."""
        self._make_stream_chunks(
            {"type": "reasoning", "text": "thinking..."},
            {"type": "content", "text": "answer"},
        )
        runner = self._make_runner()
        history = [{"role": "user",  "content": "hi"}]

        output = list(runner.step(history))
        # reasoning 不出现在 yield 中
        self.assertNotIn("thinking...", output)
        self.assertEqual(output, ["answer"])

        # 但 reasoning 被写入 assistant 消息
        self.assertEqual(history[1]["reasoning_content"], "thinking...")

    # ── 工具调用 ──

    def test_single_tool_call(self):
        """单个工具调用流程."""
        self.tools.execute.return_value = iter([
            {"id": "call_1", "result": "tool output"},
        ])
        self._make_stream_chunks(
            {"type": "tool_call", "index": 0, "id": "call_1", "name": "bash",
             "arguments": '{"cmd": "ls"}'},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "list files"}]

        list(runner.step(history))

        # assistant 消息包含 tool_calls
        self.assertIn("tool_calls", history[1])
        self.assertEqual(history[1]["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(history[1].get("content"), None)

        # tool 消息被追加
        tool_msgs = [m for m in history if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_1")
        self.assertIn("tool output", tool_msgs[0]["content"])

    def test_multiple_tool_calls(self):
        """多个工具调用 — 两次 tool_call 产生两条 tool 消息."""
        self.tools.execute.return_value = iter([
            {"id": "call_1", "result": "output1"},
            {"id": "call_2", "result": "output2"},
        ])
        # 第一次 stream 返回 2 个 tool_call；第二次返回空内容（结束）
        self.llm.stream.side_effect = [
            iter([
                {"type": "tool_call", "index": 0, "id": "call_1", "name": "bash",
                 "arguments": '{"cmd":"ls"}'},
                {"type": "tool_call", "index": 1, "id": "call_2", "name": "read",
                 "arguments": '{"path":"f.txt"}'},
            ]),
            iter([{"type": "content", "text": "done"}]),
        ]
        runner = self._make_runner()
        history = [{"role": "user", "content": "do multiple"}]

        list(runner.step(history))
        # 两个 tool_calls
        self.assertEqual(len(history[1]["tool_calls"]), 2)
        # 两条 tool 消息
        tool_msgs = [m for m in history if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_1")
        self.assertEqual(tool_msgs[1]["tool_call_id"], "call_2")

    def test_tool_call_arguments_concatenation(self):
        """DeepSeek 分片：arguments 逐 chunk 拼接而非覆盖."""
        self.tools.execute.return_value = iter([
            {"id": "call_1", "result": "ok"},
        ])
        self._make_stream_chunks(
            {"type": "tool_call", "index": 0, "id": "call_1", "name": "bash",
             "arguments": '{"cmd":'},
            {"type": "tool_call", "index": 0, "arguments": ' "ls"}'},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "hi"}]

        list(runner.step(history))
        args = history[1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, '{"cmd": "ls"}')

    def test_tool_call_with_content(self):
        """tool_call 和 content 同时存在."""
        self.tools.execute.return_value = iter([
            {"id": "call_1", "result": "ok"},
        ])
        self._make_stream_chunks(
            {"type": "content", "text": "Let me check..."},
            {"type": "tool_call", "index": 0, "id": "call_1", "name": "bash",
             "arguments": '{"cmd":"ls"}'},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "hi"}]

        output = list(runner.step(history))
        self.assertIn("Let me check...", output)
        self.assertEqual(history[1]["content"], "Let me check...")
        self.assertIn("tool_calls", history[1])

    # ── max_turns ──

    def test_max_turns_limit(self):
        """达到最大轮数时熔断."""
        self.tools.execute.side_effect = [
            iter([{"id": "c1", "result": "ok"}]),
            iter([{"id": "c2", "result": "ok"}]),
            iter([{"id": "c3", "result": "ok"}]),
        ]
        self.llm.stream.side_effect = [
            iter([{"type": "tool_call", "index": 0, "id": "c1", "name": "bash",
                   "arguments": "{}"}]),
            iter([{"type": "tool_call", "index": 0, "id": "c2", "name": "bash",
                   "arguments": "{}"}]),
            iter([{"type": "tool_call", "index": 0, "id": "c3", "name": "bash",
                   "arguments": "{}"}]),
        ]
        runner = self._make_runner(max_turns=2)
        history = [{"role": "user", "content": "go"}]

        output = list(runner.step(history))
        output_text = "".join(str(o) for o in output)
        self.assertIn("达到最大轮数", output_text)
        self.assertIn("2", output_text)

    def test_no_max_turns(self):
        """max_turns=None 时不设上限."""
        # 需要有限循环才能测试 — 让 LLM 返回文本结束
        self._make_stream_chunks({"type": "content", "text": "done"})
        runner = self._make_runner(max_turns=None)
        history = [{"role": "user", "content": "hi"}]
        output = list(runner.step(history))
        self.assertEqual(output, ["done"])

    # ── event_bus ──

    def test_turn_start_event(self):
        """每轮 emit turn:start."""
        self._make_stream_chunks({"type": "content", "text": "ok"})
        runner = self._make_runner()
        list(runner.step([{"role": "user", "content": "hi"}]))
        self.bus.emit.assert_any_call("turn:start", {"turn": 1})

    def test_no_event_bus_does_not_crash(self):
        """event_bus=None 不崩溃."""
        runner = AgentRunner(
            llm_client=self.llm,
            tool_executor=self.tools,
        )
        self._make_stream_chunks({"type": "content", "text": "ok"})
        list(runner.step([{"role": "user", "content": "hi"}]))

    # ── token tracking ──

    def test_token_usage_recorded(self):
        """usage chunk 被记录到 tracker，参数正确传递."""
        self.llm.model = "test-model"
        self._make_stream_chunks(
            {"type": "usage", "input": 100, "output": 50,
             "cache_hit": 30, "cache_miss": 20},
            {"type": "content", "text": "ok"},
        )
        runner = self._make_runner()
        list(runner.step([{"role": "user", "content": "hi"}]))
        self.tracker.record.assert_called_once()
        call_args = self.tracker.record.call_args[0]
        self.assertEqual(call_args[0], "test-model")
        self.assertEqual(call_args[1].prompt_tokens, 100)
        self.assertEqual(call_args[1].completion_tokens, 50)
        self.assertEqual(call_args[1].prompt_cache_hit_tokens, 30)
        self.assertEqual(call_args[1].prompt_cache_miss_tokens, 20)

    def test_token_tracking_without_tracker(self):
        """没有 tracker 时不崩溃."""
        runner = AgentRunner(
            llm_client=self.llm,
            tool_executor=self.tools,
        )
        self.llm.model = "test"
        self._make_stream_chunks(
            {"type": "usage", "input": 100, "output": 50},
            {"type": "content", "text": "ok"},
        )
        list(runner.step([{"role": "user", "content": "hi"}]))

    # ── 边界场景 ──

    def test_empty_content_with_tool_calls(self):
        """只有 tool_calls 没有文本内容时 content 为 None."""
        self.tools.execute.return_value = iter([
            {"id": "call_1", "result": "ok"},
        ])
        self._make_stream_chunks(
            {"type": "tool_call", "index": 0, "id": "call_1", "name": "bash",
             "arguments": "{}"},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "hi"}]

        list(runner.step(history))
        self.assertIsNone(history[1]["content"])

    def test_empty_content_no_tool_calls_fallback(self):
        """无文本也无 tool_calls 时用 reasoning 或空字符串回退."""
        self._make_stream_chunks(
            {"type": "reasoning", "text": "hmm..."},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "hi"}]

        list(runner.step(history))
        self.assertEqual(history[1]["content"], "hmm...")

    def test_tool_result_as_string_yielded(self):
        """工具执行返回纯字符串时直接 yield."""
        self.tools.execute.return_value = iter(["streaming output"])
        self._make_stream_chunks(
            {"type": "tool_call", "index": 0, "id": "call_1", "name": "bash",
             "arguments": "{}"},
        )
        runner = self._make_runner()
        history = [{"role": "user", "content": "hi"}]

        output = list(runner.step(history))
        self.assertIn("streaming output", output)


if __name__ == "__main__":
    unittest.main()
