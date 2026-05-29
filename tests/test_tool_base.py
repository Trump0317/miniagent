"""Tool 基类和装饰器单元测试."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from agent.tools.base import _minify_schema, _walk, Tool, tool


# ── 测试用参数模型 ──

class SimpleArgs(BaseModel):
    query: str
    max_results: int = 10


# ── 测试用 Tool 子类 ──

@tool(name="greet", description="问候工具", parameters=SimpleArgs)
class GreetTool(Tool):
    def execute(self, query: str, max_results: int = 10) -> str:
        return f"greet: {query} (max: {max_results})"


@tool(parameters=SimpleArgs)
class NoNameTool(Tool):
    """默认使用类名作为名称."""
    def execute(self, **kwargs) -> str:
        return "ok"


@tool(name="streamer", description="流式工具", parameters=SimpleArgs)
class StreamTool(Tool):
    supports_streaming = True

    def execute(self, **kwargs) -> str:
        return "done"

    def stream_execute(self, **kwargs):
        for part in ["a", "b", "c"]:
            yield part


class TestMinifySchema(unittest.TestCase):
    """_minify_schema 修复 schema."""

    def test_removes_title(self):
        """移除 title 字段."""
        schema = {"type": "object", "title": "MyTool", "properties": {}}
        result = _minify_schema(schema)
        self.assertNotIn("title", result)

    def test_removes_additional_properties(self):
        """移除 additionalProperties 字段."""
        schema = {"type": "object", "additionalProperties": False, "properties": {}}
        result = _minify_schema(schema)
        self.assertNotIn("additionalProperties", result)

    def test_preserves_default(self):
        """保留 default 值."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer", "default": 10}},
        }
        result = _minify_schema(schema)
        self.assertEqual(result["properties"]["count"]["default"], 10)

    def test_preserves_anyof(self):
        """保留 anyOf（Optional 类型）. """
        schema = {
            "type": "object",
            "properties": {"name": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }
        result = _minify_schema(schema)
        self.assertIn("anyOf", result["properties"]["name"])
        self.assertEqual(len(result["properties"]["name"]["anyOf"]), 2)

    def test_recurses_nested_properties(self):
        """递归处理嵌套对象属性."""
        schema = {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "title": "Nested",
                    "properties": {"key": {"type": "string", "title": "Key"}},
                }
            },
        }
        result = _minify_schema(schema)
        self.assertNotIn("title", result["properties"]["options"])
        self.assertNotIn("title", result["properties"]["options"]["properties"]["key"])

    def test_recurses_array_items(self):
        """递归处理数组 items."""
        schema = {
            "type": "array",
            "items": {"type": "object", "title": "Item", "properties": {}},
        }
        result = _minify_schema(schema)
        self.assertNotIn("title", result["items"])

    def test_recurses_defs(self):
        """递归处理 $defs."""
        schema = {
            "type": "object",
            "$defs": {"Ref": {"type": "object", "title": "RefTitle", "properties": {}}},
        }
        result = _minify_schema(schema)
        self.assertNotIn("title", result["$defs"]["Ref"])

    def test_recurses_anyof_types(self):
        """递归处理 anyOf 中非 null 类型."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "object", "title": "Obj", "properties": {}},
                        {"type": "null"},
                    ]
                }
            },
        }
        result = _minify_schema(schema)
        anyof = result["properties"]["value"]["anyOf"]
        self.assertNotIn("title", anyof[0])  # 第一个（非 null）被递归处理
        self.assertEqual(anyof[1]["type"], "null")  # null 不变

    def test_copies_schema(self):
        """不修改原始 schema."""
        original = {"type": "object", "title": "MyTitle", "properties": {}}
        _minify_schema(original)
        self.assertIn("title", original)  # 原 schema 不变


class TestToolDecorator(unittest.TestCase):
    """@tool 装饰器测试."""

    def test_name_injected(self):
        """name 参数被注入."""
        self.assertEqual(GreetTool().name, "greet")

    def test_description_injected(self):
        """description 被注入."""
        self.assertEqual(GreetTool().description, "问候工具")

    def test_args_model_injected(self):
        """args_model 被注入."""
        self.assertEqual(GreetTool().args_model, SimpleArgs)

    def test_name_fallback_to_classname(self):
        """name 未指定时回退到类名."""
        self.assertEqual(NoNameTool().name, "NoNameTool")

    def test_description_fallback_to_docstring(self):
        """description 未指定时回退到 docstring."""
        self.assertIn("默认使用类名", NoNameTool().description)


class TestToolBase(unittest.TestCase):
    """Tool 基类功能测试."""

    def setUp(self):
        self.tool = GreetTool()

    def test_name_property(self):
        self.assertEqual(self.tool.name, "greet")

    def test_description_property(self):
        self.assertEqual(self.tool.description, "问候工具")

    def test_args_model_property(self):
        self.assertEqual(self.tool.args_model, SimpleArgs)

    def test_parallel_safe_default(self):
        """默认 parallel_safe=True."""
        self.assertTrue(self.tool.parallel_safe)

    def test_supports_streaming_default(self):
        """默认 supports_streaming=False."""
        self.assertFalse(self.tool.supports_streaming)

    def test_parameters_returns_schema(self):
        """parameters 返回瘦身后的 JSON Schema."""
        schema = self.tool.parameters
        self.assertNotIn("title", schema)
        self.assertIn("type", schema)
        self.assertIn("properties", schema)

    def test_cast_params_valid(self):
        """cast_params 正确转换参数."""
        result = self.tool.cast_params({"query": "hello"})
        self.assertEqual(result["query"], "hello")
        self.assertEqual(result["max_results"], 10)  # 默认值

    def test_cast_params_invalid_raises(self):
        """无效参数抛 ValueError."""
        with self.assertRaises(ValueError):
            self.tool.cast_params({"query": 123})  # 类型错误

    def test_validate_params_valid(self):
        """validate_params 有效参数不报错."""
        self.tool.validate_params({"query": "ok"})

    def test_validate_params_invalid_raises(self):
        """validate_params 无效参数也抛 ValueError."""
        with self.assertRaises(ValueError):
            self.tool.validate_params({"query": 123})

    def test_execute(self):
        """execute 工作正常."""
        result = self.tool.execute(query="hi")
        self.assertEqual(result, "greet: hi (max: 10)")


class TestToolStreaming(unittest.TestCase):
    """流式执行测试."""

    def test_supports_streaming(self):
        tool = StreamTool()
        self.assertTrue(tool.supports_streaming)

    def test_stream_execute(self):
        """stream_execute 逐块产出."""
        tool = StreamTool()
        parts = list(tool.stream_execute(query="test"))
        self.assertEqual(parts, ["a", "b", "c"])

    def test_default_stream_execute(self):
        """默认 stream_execute 回退到 execute."""
        tool = GreetTool()
        parts = list(tool.stream_execute(query="hi"))
        self.assertEqual(parts, ["greet: hi (max: 10)"])


if __name__ == "__main__":
    unittest.main()
