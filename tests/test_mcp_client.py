"""MCP 客户端模块单元测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.tools.mcp_client import (
    _json_type_to_python,
    _schema_to_pydantic_model,
    _make_mcp_tool,
    McpConnection,
    McpClientManager,
)


class TestSchemaConversion(unittest.TestCase):
    """Schema → Pydantic 模型转换测试。"""

    def test_basic_types(self):
        self.assertIs(_json_type_to_python({"type": "string"}), str)
        self.assertIs(_json_type_to_python({"type": "number"}), float)
        self.assertIs(_json_type_to_python({"type": "integer"}), int)
        self.assertIs(_json_type_to_python({"type": "boolean"}), bool)
        self.assertIs(_json_type_to_python({"type": "object"}), dict)
        self.assertIs(_json_type_to_python({"type": "array"}), list)

    def test_unknown_type_defaults_to_str(self):
        self.assertIs(_json_type_to_python({"type": "unknown"}), str)
        self.assertIs(_json_type_to_python({}), str)

    def test_type_list_skips_null(self):
        self.assertIs(
            _json_type_to_python({"type": ["null", "string"]}), str
        )
        self.assertIs(
            _json_type_to_python({"type": ["string", "null"]}), str
        )

    def test_simple_model_with_required(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        Model = _schema_to_pydantic_model("search", schema)
        m = Model(query="hello")
        self.assertEqual(m.query, "hello")

    def test_simple_model_with_default(self):
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
            },
        }
        Model = _schema_to_pydantic_model("search", schema)
        m = Model()
        self.assertEqual(m.limit, 10)
        m = Model(limit=5)
        self.assertEqual(m.limit, 5)

    def test_mixed_required_and_optional(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        }
        Model = _schema_to_pydantic_model("test", schema)
        m = Model(name="x")
        self.assertEqual(m.name, "x")
        self.assertFalse(m.verbose)

    def test_empty_properties(self):
        schema = {"type": "object", "properties": {}}
        Model = _schema_to_pydantic_model("empty", schema)
        m = Model()
        self.assertIsNotNone(m)

    def test_no_properties_key(self):
        schema = {"type": "object"}
        Model = _schema_to_pydantic_model("empty", schema)
        m = Model()
        self.assertIsNotNone(m)

    def test_empty_schema_dict(self):
        Model = _schema_to_pydantic_model("empty", {})
        m = Model()
        self.assertIsNotNone(m)

    def test_tool_name_sanitization(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        Model = _schema_to_pydantic_model("my-tool.sub", schema)
        self.assertIn("my_tool_sub", Model.__name__)


class TestMcpToolAdapter(unittest.TestCase):
    """MCP 工具 → miniagent Tool 适配测试。"""

    def test_tool_naming(self):
        conn = MagicMock()
        conn.name = "github"
        mcp_tool = MagicMock()
        mcp_tool.name = "search_repos"
        mcp_tool.description = "Search GitHub repos"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

        ToolCls = _make_mcp_tool(conn, mcp_tool)
        inst = ToolCls()
        self.assertEqual(inst.name, "mcp__github__search_repos")
        self.assertIn("[MCP:github]", inst.description)
        self.assertFalse(inst.parallel_safe)

    def test_tool_call_delegates_to_connection(self):
        conn = MagicMock()
        conn.name = "test"
        conn.call_tool.return_value = "result"
        mcp_tool = MagicMock()
        mcp_tool.name = "echo"
        mcp_tool.description = "Echo"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        }

        ToolCls = _make_mcp_tool(conn, mcp_tool)
        inst = ToolCls()
        result = inst.execute(message="hello")
        self.assertEqual(result, "result")
        conn.call_tool.assert_called_once_with("echo", {"message": "hello"})

    def test_tool_without_args(self):
        conn = MagicMock()
        conn.name = "test"
        conn.call_tool.return_value = "ok"
        mcp_tool = MagicMock()
        mcp_tool.name = "status"
        mcp_tool.description = "Get status"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        ToolCls = _make_mcp_tool(conn, mcp_tool)
        inst = ToolCls()
        result = inst.execute()
        self.assertEqual(result, "ok")


class TestMcpClientManager(unittest.TestCase):
    """McpClientManager 配置加载和生命周期测试。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._config_path = Path(self._tmpdir.name) / "mcp.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_config(self, data: dict) -> None:
        self._config_path.write_text(json.dumps(data), encoding="utf-8")

    def test_no_config_file(self):
        mgr = McpClientManager(config_path="/nonexistent/mcp.json")
        self.assertEqual(len(mgr.get_tool_classes()), 0)
        mgr.shutdown()

    def test_empty_servers(self):
        self._write_config({"mcpServers": {}})
        mgr = McpClientManager(config_path=self._config_path)
        self.assertEqual(len(mgr.get_tool_classes()), 0)

    def test_missing_command_skipped(self):
        self._write_config({
            "mcpServers": {
                "bad": {"args": ["--help"]},
            }
        })
        mgr = McpClientManager(config_path=self._config_path)
        self.assertEqual(len(mgr.get_tool_classes()), 0)

    def test_invalid_json(self):
        self._config_path.write_text("not json", encoding="utf-8")
        mgr = McpClientManager(config_path=self._config_path)
        self.assertEqual(len(mgr.get_tool_classes()), 0)

    def test_connection_failure_graceful(self):
        """服务器启动失败应跳过并继续。"""
        self._write_config({
            "mcpServers": {
                "nonexistent": {
                    "command": "/usr/bin/definitely_does_not_exist_12345",
                    "args": [],
                },
            }
        })
        mgr = McpClientManager(config_path=self._config_path)
        self.assertEqual(len(mgr.get_tool_classes()), 0)
        mgr.shutdown()

    def test_shutdown_idempotent(self):
        mgr = McpClientManager()
        mgr.shutdown()
        mgr.shutdown()  # 不抛异常

    def test_none_config_path(self):
        mgr = McpClientManager(config_path=None)
        self.assertEqual(len(mgr.get_tool_classes()), 0)
        mgr.shutdown()
