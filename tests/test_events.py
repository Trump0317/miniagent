"""EventBus 单元测试."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.events import EventBus, Event


class TestEventBus(unittest.TestCase):
    """EventBus 核心功能测试."""

    def setUp(self):
        self.bus = EventBus()

    # ── 注册 ──

    def test_on_register(self):
        """on() 注册处理器."""
        calls = []
        self.bus.on("test", lambda e: calls.append(e.name))
        self.bus.emit("test")
        self.assertEqual(calls, ["test"])

    def test_on_as_decorator(self):
        """on() 作为装饰器使用."""
        calls = []

        @self.bus.on("decorated")
        def handler(e):
            calls.append(e.name)

        self.bus.emit("decorated")
        self.assertEqual(calls, ["decorated"])

    def test_on_returns_handler(self):
        """on() 返回 handler 自身."""

        def h(e):
            pass

        result = self.bus.on("ret", h)
        self.assertIs(result, h)

    def test_multiple_handlers(self):
        """同一事件多个处理器."""
        calls = []
        self.bus.on("multi", lambda e: calls.append(1))
        self.bus.on("multi", lambda e: calls.append(2))
        self.bus.emit("multi")
        self.assertEqual(calls, [1, 2])

    # ── 一次性监听 ──

    def test_once(self):
        """once() 只触发一次."""
        calls = []
        self.bus.once("one_shot", lambda e: calls.append(1))
        self.bus.emit("one_shot")
        self.bus.emit("one_shot")
        self.assertEqual(calls, [1])  # 第二次不触发

    def test_once_as_decorator(self):
        """once() 作为装饰器."""
        calls = []

        @self.bus.once("once_deco")
        def handler(e):
            calls.append(1)

        self.bus.emit("once_deco")
        self.bus.emit("once_deco")
        self.assertEqual(calls, [1])

    def test_multiple_once_handlers(self):
        """同一事件的多个 once() 处理器各触发一次."""
        calls = []
        self.bus.once("multi_once", lambda e: calls.append(1))
        self.bus.once("multi_once", lambda e: calls.append(2))
        self.bus.emit("multi_once")
        self.bus.emit("multi_once")  # 第二次不应触发
        self.assertEqual(calls, [1, 2])

    # ── 移除 ──

    def test_off(self):
        """off() 移除处理器."""
        calls = []

        def h(e):
            calls.append(1)

        self.bus.on("removable", h)
        self.bus.emit("removable")
        self.bus.off("removable", h)
        self.bus.emit("removable")
        self.assertEqual(calls, [1])  # 只触发一次

    def test_off_nonexistent(self):
        """off() 传入从未注册的 handler 不报错."""
        def h(e):
            pass

        # h 未注册到任何事件上
        self.bus.off("no_such_event", h)  # 不应抛异常

    def test_off_removes_from_both_stores(self):
        """off() 同时从 _handlers 和 _once_handlers 移除."""
        calls = []

        def h(e):
            calls.append(1)

        self.bus.on("dup", h)
        self.bus.once("dup", h)
        self.bus.off("dup", h)
        self.bus.emit("dup")
        self.assertEqual(calls, [])  # 两边都不触发

    def test_off_wildcard_handler(self):
        """off() 能移除通配符注册的处理器."""
        calls = []

        def h(e):
            calls.append(1)

        self.bus.on("tool:*", h)
        self.bus.off("tool:*", h)
        self.bus.emit("tool:before")
        self.assertEqual(calls, [])

    # ── 返回值收集 ──

    def test_emit_returns(self):
        """emit() 收集返回值列表."""
        self.bus.on("calc", lambda e: 42)
        self.bus.on("calc", lambda e: "hello")
        results = self.bus.emit("calc")
        self.assertEqual(results, [42, "hello"])

    def test_emit_returns_includes_none(self):
        """None 返回值也出现在结果列表中."""
        self.bus.on("nop", lambda e: None)
        results = self.bus.emit("nop")
        self.assertEqual(results, [None])

    def test_emit_no_handlers(self):
        """无处理器时返回空列表."""
        results = self.bus.emit("nobody")
        self.assertEqual(results, [])

    # ── Event 对象 ──

    def test_event_attributes(self):
        """Event 对象携带 name / data / source."""
        captured = []

        @self.bus.on("attr_test")
        def h(e):
            captured.append((e.name, e.data, e.source))

        self.bus.emit("attr_test", data={"key": "val"}, source="test_source")
        self.assertEqual(captured[0], ("attr_test", {"key": "val"}, "test_source"))

    def test_event_defaults(self):
        """Event 默认参数."""
        e = Event("test")
        self.assertEqual(e.name, "test")
        self.assertIsNone(e.data)
        self.assertEqual(e.source, "")

    # ── 通配符匹配 ──

    def test_wildcard_single_level(self):
        """"tool:*" 匹配 "tool:before"."""
        calls = []
        self.bus.on("tool:*", lambda e: calls.append(e.name))
        self.bus.emit("tool:before")
        self.bus.emit("tool:after")
        self.assertEqual(calls, ["tool:before", "tool:after"])

    def test_wildcard_not_match_other(self):
        """"tool:*" 不匹配 "other:event"."""
        calls = []
        self.bus.on("tool:*", lambda e: calls.append(e.name))
        self.bus.emit("other:event")
        self.assertEqual(calls, [])

    def test_wildcard_exact_and_pattern(self):
        """精确匹配和通配符同时触发."""
        calls_exact = []
        calls_wild = []
        self.bus.on("tool:before", lambda e: calls_exact.append("exact"))
        self.bus.on("tool:*", lambda e: calls_wild.append("wild"))
        self.bus.emit("tool:before")
        self.assertEqual(calls_exact, ["exact"])
        self.assertEqual(calls_wild, ["wild"])

    def test_wildcard_multi_level(self):
        """'a:b:*' 匹配 'a:b:c' 但不匹配 'a:x'."""
        calls = []
        self.bus.on("a:b:*", lambda e: calls.append(e.name))
        self.bus.emit("a:b:c")
        self.bus.emit("a:b:d")
        self.bus.emit("a:x")
        self.assertEqual(calls, ["a:b:c", "a:b:d"])

    def test_wildcard_shallow_matches_deep(self):
        """'a:*' 匹配深层事件 'a:b:c'."""
        calls = []
        self.bus.on("a:*", lambda e: calls.append(e.name))
        self.bus.emit("a:b:c")
        self.assertEqual(calls, ["a:b:c"])

    def test_wildcard_no_empty_pattern_for_plain(self):
        """'plain' 事件不生成 ':*' 空模式（range(1, ...) 正确跳过 i=0）."""
        calls = []
        # 注册一个监听器验证 plain 事件确实能被 plain:* 通配符匹配
        self.bus.on("plain:*", lambda e: calls.append(e.name))
        self.bus.emit("plain")
        self.assertEqual(calls, ["plain"])

    def test_literal_colon_star_only_matches_exact(self):
        """字面量 ':*' handler 只匹配精确事件名 ':*'，不匹配 'tool:before'."""
        calls = []
        self.bus.on(":*", lambda e: calls.append(e.name))
        self.bus.emit("tool:before")
        self.bus.emit("plain")
        self.assertEqual(calls, [])

    # ── once 与通配符 ──

    def test_once_matched_by_wildcard(self):
        """once('tool:*') 被通配符匹配触发，触发后自动移除."""
        calls = []
        self.bus.once("tool:*", lambda e: calls.append(1))
        self.bus.emit("tool:before")
        self.bus.emit("tool:after")
        self.assertEqual(calls, [1])  # 只触发一次

    # ── 空/边界 ──

    def test_handler_receives_none_data(self):
        """data 为 None 时也能正常传递."""
        captured = []
        self.bus.on("none_data", lambda e: captured.append(e.data))
        self.bus.emit("none_data")
        self.assertEqual(captured, [None])

    def test_handler_exception_propagates(self):
        """处理器异常向上传播.
        
        注意：这是当前实现的行为选择。另见
        test_exception_stops_subsequent_handlers 验证异常后后续处理器不执行。
        """
        def boom(e):
            raise RuntimeError("BOOM")

        self.bus.on("explode", boom)
        with self.assertRaises(RuntimeError):
            self.bus.emit("explode")

    # ── 执行顺序 ──

    def test_handler_execution_order(self):
        """精确匹配先于 once，once 先于通配符."""
        order = []
        self.bus.on("order", lambda e: order.append("exact"))
        self.bus.once("order", lambda e: order.append("once"))
        self.bus.on("order:*", lambda e: order.append("wild"))
        self.bus.emit("order")
        self.assertEqual(order, ["exact", "once", "wild"])

    # ── 多处理器异常 ──

    def test_exception_stops_subsequent_handlers(self):
        """多处理器中第一个抛异常，后续不执行."""
        calls = []

        def boom(e):
            raise ValueError("Boom")

        self.bus.on("chain", boom)
        self.bus.on("chain", lambda e: calls.append("nope"))
        with self.assertRaises(ValueError):
            self.bus.emit("chain")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
