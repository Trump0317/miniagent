"""ToolExecutor 单元测试."""
import unittest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from agent.tools.base import Tool, tool
from agent.tools.registry import ToolRegistry
from agent.tools.executor import ToolExecutor, MAX_RESULT_BYTES, MAX_RESULT_LINES
from agent.core.events import EventBus


# ── 测试用工具 ──

class SimpleArgs(BaseModel):
    value: str = "default"


@tool(name="echo", description="Echo tool", parameters=SimpleArgs)
class EchoTool(Tool):
    parallel_safe = True

    def execute(self, value: str = "default") -> str:
        return f"echo: {value}"


@tool(name="unsafe_tool", description="Not parallel safe", parameters=SimpleArgs)
class UnsafeTool(Tool):
    parallel_safe = False

    def execute(self, value: str = "default") -> str:
        return f"unsafe: {value}"


@tool(name="streamer", description="Streaming tool", parameters=SimpleArgs)
class StreamTool(Tool):
    parallel_safe = True
    supports_streaming = True

    def execute(self, value: str = "default") -> str:
        return f"full: {value}"

    def stream_execute(self, value: str = "default"):
        for c in value:
            yield c


@tool(name="failing", description="Always fails", parameters=SimpleArgs)
class FailingTool(Tool):
    parallel_safe = True

    def execute(self, value: str = "default") -> str:
        raise ValueError("boom")


class RequiredArgs(BaseModel):
    path: str  # 必需，无默认值


@tool(name="need_path", description="Needs path", parameters=RequiredArgs)
class NeedPathTool(Tool):
    parallel_safe = True

    def execute(self, path: str) -> str:
        return f"path: {path}"


@tool(name="big_output", description="Large output", parameters=SimpleArgs)
class BigOutputTool(Tool):
    parallel_safe = True

    def execute(self, value: str = "default") -> str:
        # 生成超过截断阈值的输出
        return "x" * (MAX_RESULT_BYTES + 100)


class TestToolExecutor(unittest.TestCase):
    """ToolExecutor 核心功能测试."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())
        self.registry.register(UnsafeTool())
        self.registry.register(StreamTool())
        self.registry.register(FailingTool())
        self.registry.register(NeedPathTool())
        self.registry.register(BigOutputTool())
        self.bus = EventBus()
        self.executor = ToolExecutor(self.registry, event_bus=self.bus)

    def _tc(self, name: str, args: dict, id: str = "call_1") -> dict:
        return {"id": id, "function": {"name": name, "arguments": json.dumps(args)}}

    # ── 单工具串行 ──

    def test_single_tool_serial(self):
        """单个工具使用串行执行."""
        results = list(self.executor.execute([self._tc("echo", {"value": "hello"})]))
        # 包含状态文本 + 最终结果
        self.assertTrue(any(isinstance(r, dict) for r in results))
        result_dict = [r for r in results if isinstance(r, dict)][0]
        self.assertEqual(result_dict["id"], "call_1")
        self.assertIn("echo: hello", result_dict["result"])

    def test_single_tool_streaming(self):
        """流式工具逐块产出."""
        results = list(self.executor.execute([self._tc("streamer", {"value": "abc"})]))
        # 应该包含逐块输出
        texts = [r for r in results if isinstance(r, str)]
        self.assertIn("a", texts)
        self.assertIn("b", texts)
        self.assertIn("c", texts)

    # ── 多工具并行（全部 safe）──

    def test_parallel_all_safe(self):
        """全部 parallel_safe 的工具使用并行执行."""
        tcs = [
            self._tc("echo", {"value": "a"}, id="1"),
            self._tc("echo", {"value": "b"}, id="2"),
            self._tc("streamer", {"value": "x"}, id="3"),
        ]
        results = list(self.executor.execute(tcs))
        result_dicts = [r for r in results if isinstance(r, dict)]
        self.assertEqual(len(result_dicts), 3)

    # ── 多工具串行（存在非安全工具）──

    def test_serial_when_unsafe_present(self):
        """存在非安全工具时全部降级串行."""
        tcs = [
            self._tc("echo", {"value": "safe"}),
            self._tc("unsafe_tool", {"value": "dangerous"}),
        ]
        results = list(self.executor.execute(tcs))
        # 应该包含串行提示
        texts = "".join(r for r in results if isinstance(r, str))
        self.assertIn("串行执行", texts)

    # ── 截断 ──

    def test_truncate_large_output(self):
        """超大输出被截断并附加截断说明."""
        results = list(self.executor.execute([self._tc("big_output", {"value": "test"})]))
        result_dicts = [r for r in results if isinstance(r, dict)]
        result_str = result_dicts[0]["result"]
        self.assertIn("输出已截断", result_str)

    def test_truncate_no_truncation_needed(self):
        """短输出原样返回不截断."""
        result = ToolExecutor._truncate("short")
        self.assertEqual(result, "short")

    # ── _parse_args ──

    def test_parse_args_valid_json(self):
        """正常 JSON 解析为 dict."""
        tc = {"function": {"arguments": '{"key": "val"}'}}
        result = ToolExecutor._parse_args(tc)
        self.assertEqual(result, {"key": "val"})

    def test_parse_args_empty(self):
        """空 tool_call 返回空 dict."""
        tc = {}
        result = ToolExecutor._parse_args(tc)
        self.assertEqual(result, {})

    def test_parse_args_invalid_json(self):
        """非法 JSON 返回空 dict 不抛异常."""
        tc = {"function": {"arguments": "not json"}}
        result = ToolExecutor._parse_args(tc)
        self.assertEqual(result, {})

    # ── 事件钩子: tool:before 拦截 ──

    def test_tool_before_block(self):
        """tool:before 事件可以拦截工具执行."""
        @self.bus.on("tool:before")
        def blocker(event):
            return {"block": True, "reason": "test block"}

        results = list(self.executor.execute([self._tc("echo", {"value": "test"})]))
        result_dicts = [r for r in results if isinstance(r, dict)]
        self.assertIn("[拦截]", result_dicts[0]["result"])
        self.assertIn("test block", result_dicts[0]["result"])

    # ── 事件钩子: tool:after 修改结果 ──

    def test_tool_after_modify(self):
        """tool:after 事件可以修改工具结果."""
        @self.bus.on("tool:after")
        def modifier(event):
            return "modified result"

        results = list(self.executor.execute([self._tc("echo", {"value": "test"})]))
        result_dicts = [r for r in results if isinstance(r, dict)]
        self.assertEqual(result_dicts[0]["result"], "modified result")

    def test_parallel_with_tool_before_block(self):
        """并行模式下 tool:before 也能拦截工具执行."""
        @self.bus.on("tool:before")
        def blocker(event):
            return {"block": True, "reason": "blocked in parallel"}

        tcs = [
            self._tc("echo", {"value": "a"}, id="1"),
            self._tc("echo", {"value": "b"}, id="2"),
        ]
        results = list(self.executor.execute(tcs))
        result_dicts = [r for r in results if isinstance(r, dict)]
        for d in result_dicts:
            self.assertIn("[拦截]", d["result"])

    # ── 错误工具 ──

    def test_tool_not_found(self):
        """调用未注册工具返回错误提示."""
        results = list(self.executor.execute([self._tc("ghost", {})]))
        result_dicts = [r for r in results if isinstance(r, dict)]
        self.assertIn("未找到工具", result_dicts[0]["result"])

    def test_tool_param_validation_error(self):
        """缺少必需参数触发校验错误."""
        results = list(self.executor.execute([self._tc("need_path", {})]))
        texts = "".join(str(r) for r in results)
        self.assertIn("参数校验错误", texts)

    def test_tool_runtime_error(self):
        """工具 execute 抛 ValueError 被 call_tool 捕获为参数校验错误."""
        results = list(self.executor.execute([self._tc("failing", {})]))
        result_dicts = [r for r in results if isinstance(r, dict)]
        # execute 抛出 ValueError → call_tool 的 ValueError 捕获返回 "参数校验错误"
        self.assertIn("参数校验错误", result_dicts[0]["result"])


class TestToolExecutorWithoutEventBus(unittest.TestCase):
    """无 EventBus 时的行为."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())
        self.executor = ToolExecutor(self.registry, event_bus=None)

    def _tc(self, name: str, args: dict) -> dict:
        return {"id": "call_1", "function": {"name": name, "arguments": str(args).replace("'", '"')}}

    def test_execute_without_bus(self):
        """没有 EventBus 时正常执行不报错."""
        results = list(self.executor.execute([self._tc("echo", {"value": "test"})]))
        result_dicts = [r for r in results if isinstance(r, dict)]
        self.assertIn("echo: test", result_dicts[0]["result"])


if __name__ == "__main__":
    unittest.main()
