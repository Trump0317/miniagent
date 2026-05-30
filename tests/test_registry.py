"""ToolRegistry 单元测试."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from agent.tools.base import Tool, tool
from agent.tools.registry import ToolRegistry


# ── 测试用工具 ──

class Args1(BaseModel):
    x: int
    y: str = "default"


class Args2(BaseModel):
    path: str


@tool(name="tool_a", description="Tool A", parameters=Args1)
class ToolA(Tool):
    def execute(self, x: int, y: str = "default") -> str:
        return f"A: {x}, {y}"


@tool(name="tool_b", description="Tool B", parameters=Args2)
class ToolB(Tool):
    def execute(self, path: str) -> str:
        return f"B: {path}"


@tool(name="failing", description="Always fails", parameters=Args2)
class FailingTool(Tool):
    def execute(self, path: str) -> str:
        raise RuntimeError("always crash")


@tool(name="error_return", description="Returns error text", parameters=Args2)
class ErrorReturnTool(Tool):
    def execute(self, path: str) -> str:
        return "error: something went wrong"


class TestToolRegistry(unittest.TestCase):
    """ToolRegistry 核心功能测试."""

    def setUp(self):
        self.registry = ToolRegistry()

    # ── register ──

    def test_register_adds_tool(self):
        self.registry.register(ToolA())
        self.assertIsNotNone(self.registry.get_tool("tool_a"))

    def test_register_invalidates_schema_cache(self):
        self.registry.register(ToolA())
        s1 = self.registry.get_tool_schemas()
        self.registry.register(ToolB())
        s2 = self.registry.get_tool_schemas()
        self.assertIsNot(s1, s2)
        self.assertEqual(len(s2), 2)

    def test_register_duplicate_overwrites(self):
        a1 = ToolA()
        a2 = ToolA()
        self.registry.register(a1)
        self.registry.register(a2)
        self.assertIs(self.registry.get_tool("tool_a"), a2)

    # ── get_tool ──

    def test_get_tool_returns_none_for_missing(self):
        self.assertIsNone(self.registry.get_tool("ghost"))

    def test_get_tool_returns_registered_instance(self):
        t = ToolA()
        self.registry.register(t)
        self.assertIs(self.registry.get_tool("tool_a"), t)

    # ── tool_names ──

    def test_tool_names_empty(self):
        self.assertEqual(self.registry.tool_names, [])

    def test_tool_names_sorted(self):
        self.registry.register(ToolB())
        self.registry.register(ToolA())
        self.assertEqual(self.registry.tool_names, ["tool_a", "tool_b"])

    # ── get_tool_schemas ──

    def test_get_tool_schemas_empty(self):
        self.assertEqual(self.registry.get_tool_schemas(), [])

    def test_get_tool_schemas_format(self):
        self.registry.register(ToolA())
        schemas = self.registry.get_tool_schemas()
        self.assertEqual(len(schemas), 1)
        s = schemas[0]
        self.assertEqual(s["type"], "function")
        self.assertEqual(s["function"]["name"], "tool_a")
        self.assertEqual(s["function"]["description"], "Tool A")
        self.assertIn("parameters", s["function"])

    def test_get_tool_schemas_cached(self):
        self.registry.register(ToolA())
        s1 = self.registry.get_tool_schemas()
        s2 = self.registry.get_tool_schemas()
        self.assertIs(s1, s2)

    def test_get_tool_schemas_multiple_tools_sorted(self):
        self.registry.register(ToolB())
        self.registry.register(ToolA())
        schemas = self.registry.get_tool_schemas()
        self.assertEqual(schemas[0]["function"]["name"], "tool_a")
        self.assertEqual(schemas[1]["function"]["name"], "tool_b")

    # ── call_tool ──

    def test_call_tool_success(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", {"x": 42})
        self.assertEqual(result, "A: 42, default")

    def test_call_tool_with_all_args(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", {"x": 1, "y": "custom"})
        self.assertEqual(result, "A: 1, custom")

    def test_call_tool_not_found(self):
        result = self.registry.call_tool("ghost", {})
        self.assertIn("未找到工具", result)
        self.assertIn("分析上述错误", result)

    def test_call_tool_not_found_includes_available(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("ghost", {})
        self.assertIn("tool_a", result)

    def test_call_tool_args_not_dict(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", "bad args")
        self.assertIn("参数格式非对象", result)

    def test_call_tool_args_is_list(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", [1, 2])
        self.assertIn("参数格式非对象", result)

    def test_call_tool_args_is_none(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", None)
        self.assertIn("参数格式非对象", result)

    def test_call_tool_validation_error(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", {})
        self.assertIn("参数校验错误", result)
        self.assertIn("分析上述错误", result)

    def test_call_tool_runtime_error(self):
        self.registry.register(FailingTool())
        result = self.registry.call_tool("failing", {"path": "/tmp"})
        self.assertIn("执行错误", result)
        self.assertIn("always crash", result)
        self.assertIn("分析上述错误", result)

    def test_call_tool_error_text_appends_hint(self):
        self.registry.register(ErrorReturnTool())
        result = self.registry.call_tool("error_return", {"path": "/tmp"})
        self.assertIn("error:", result)
        self.assertIn("分析上述错误", result)

    def test_call_tool_type_coercion(self):
        self.registry.register(ToolA())
        result = self.registry.call_tool("tool_a", {"x": "99"})
        self.assertIn("A: 99", result)

    # ── 边界 ──

    def test_empty_registry_call_tool(self):
        result = self.registry.call_tool("any", {"x": 1})
        self.assertIn("未找到工具", result)

    def test_schema_count_after_multiple_registers(self):
        self.registry.register(ToolA())
        self.assertEqual(len(self.registry.get_tool_schemas()), 1)
        self.registry.register(ToolB())
        self.assertEqual(len(self.registry.get_tool_schemas()), 2)
        self.registry.register(ToolA())
        self.assertEqual(len(self.registry.get_tool_schemas()), 2)


if __name__ == "__main__":
    unittest.main()
