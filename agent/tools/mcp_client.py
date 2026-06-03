"""MCP (Model Context Protocol) 客户端 —— 连接外部 MCP 服务器，将 MCP 工具适配为 miniagent Tool。

仅支持 stdio 传输。配置文件格式（mcp.json）兼容 Claude Code:
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        }
      }
    }

工具命名规则: mcp__<server_name>__<tool_name>
"""

from __future__ import annotations
import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

# ── Schema 转换 ──────────────────────────────────────────────────

_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _json_type_to_python(prop_schema: dict) -> type:
    """将 JSON Schema 单字段类型映射为 Python 类型。"""
    t = prop_schema.get("type", "string")
    if isinstance(t, list):
        # 取第一个非 null 类型
        for item in t:
            if item != "null":
                return _TYPE_MAP.get(item, str)
    return _TYPE_MAP.get(t, str)


def _schema_to_pydantic_model(tool_name: str, schema: dict) -> type[BaseModel]:
    """从 MCP 工具的 inputSchema 动态生成 Pydantic 模型。"""
    properties = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    if not properties:
        # 无参数工具 → 空模型
        return create_model(f"MCP_{tool_name}_Args")

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        field_type = _json_type_to_python(prop_schema)
        desc = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (field_type, Field(description=desc))
        else:
            default = prop_schema.get("default", None)
            fields[prop_name] = (
                field_type,
                Field(default=default, description=desc),
            )

    safe_name = tool_name.replace("-", "_").replace(".", "_")
    return create_model(f"MCP_{safe_name}_Args", **fields)


# ── McpConnection ─────────────────────────────────────────────────


class McpConnection:
    """管理单个 MCP stdio 服务器连接。

    在后台线程中运行 asyncio event loop，通过 run_coroutine_threadsafe 桥接到主线程。
    """

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.name = name
        self._command = command
        self._args = args or []
        self._env = env
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._tools: list = []
        self._ready = threading.Event()
        self._error: Exception | None = None

    # ── 生命周期 ──

    def start(self) -> None:
        """在后台线程启动连接并发现工具。"""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"mcp-{self.name}"
        )
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise TimeoutError(
                f"MCP server '{self.name}' 连接超时（15s）"
            )
        if self._error:
            raise self._error

    def close(self) -> None:
        """关闭连接，清理子进程。"""
        if self._loop is None:
            return

        async def _shutdown():
            try:
                if self._session_ctx:
                    await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                if self._stdio_ctx:
                    await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            future.result(timeout=5)
        except Exception:
            pass
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── 后台线程 ──

    def _run(self) -> None:
        """后台线程入口：创建 event loop，连接服务器，保持运行。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
            self._loop.run_forever()
        except Exception as e:
            self._error = e
            self._ready.set()
            logger.warning("MCP server '%s' 连接失败: %s", self.name, e)

    async def _connect(self) -> None:
        """异步连接 MCP 服务器并发现工具。"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # 合并环境变量
        import os
        merged_env = os.environ.copy()
        if self._env:
            merged_env.update(self._env)

        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=merged_env,
        )

        # 手动管理上下文（因为需要 run_forever）
        self._stdio_ctx = stdio_client(params)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()

        await self._session.initialize()

        result = await self._session.list_tools()
        self._tools = list(result.tools)
        self._ready.set()
        logger.info(
            "MCP server '%s' 已连接，%d 个工具可用",
            self.name, len(self._tools),
        )

    # ── 工具调用 ──

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """同步调用 MCP 工具（桥接到后台 event loop）。"""
        if self._loop is None or self._session is None:
            raise RuntimeError(f"MCP server '{self.name}' 未连接")

        async def _call():
            return await self._session.call_tool(tool_name, arguments)

        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        result = future.result(timeout=60)

        # 从 MCP CallToolResult 提取文本内容
        texts: list[str] = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
        return "\n".join(texts) if texts else str(result)

    @property
    def tools(self):
        """MCP 原始工具列表（mcp.types.Tool 对象）。"""
        return self._tools


# ── MCP 工具适配器 ───────────────────────────────────────────────


def _make_mcp_tool(connection: McpConnection,
                   mcp_tool) -> type[Tool]:
    """为单个 MCP 工具动态创建一个 miniagent Tool 子类。"""
    tool_name = mcp_tool.name
    full_name = f"mcp__{connection.name}__{tool_name}"
    model = _schema_to_pydantic_model(full_name, mcp_tool.inputSchema)

    class McpTool(Tool):
        _tool_name = full_name
        _tool_description = (
            f"[MCP:{connection.name}] {mcp_tool.description or ''}"
        )
        _args_model = model
        parallel_safe = False  # 跨进程 stdio 不可并行

        def execute(self, **kwargs) -> str:
            return connection.call_tool(tool_name, kwargs)

    # 注入类名方便调试
    safe = full_name.replace("-", "_").replace(".", "_")
    McpTool.__name__ = f"McpTool_{safe}"
    McpTool.__qualname__ = f"McpTool_{safe}"

    return McpTool


# ── McpClientManager ─────────────────────────────────────────────


class McpClientManager:
    """管理所有 MCP 服务器连接，提供工具注册表。"""

    def __init__(self, config_path: str | Path | None = None):
        self._connections: dict[str, McpConnection] = {}
        self._tools: list[type[Tool]] = []
        if config_path:
            self._load_and_connect(Path(config_path))

    # ── 加载配置并连接 ──

    def _load_and_connect(self, config_path: Path) -> None:
        """读取 mcp.json，逐一连接服务器。失败时跳过并警告。"""
        if not config_path.exists():
            logger.debug("MCP 配置文件 %s 不存在，跳过", config_path)
            return

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("MCP 配置文件解析失败: %s", e)
            return

        servers = config.get("mcpServers", {})
        if not servers:
            logger.debug("MCP 配置中无 mcpServers 定义")
            return

        for name, server_cfg in servers.items():
            command = server_cfg.get("command", "")
            if not command:
                logger.warning("MCP server '%s' 缺少 command，跳过", name)
                continue

            args = server_cfg.get("args", [])
            env = server_cfg.get("env")

            conn = McpConnection(
                name=name,
                command=command,
                args=args,
                env=env,
            )
            try:
                conn.start()
                self._connections[name] = conn
                for mcp_tool in conn.tools:
                    tool_cls = _make_mcp_tool(conn, mcp_tool)
                    self._tools.append(tool_cls)
            except Exception as e:
                logger.warning("MCP server '%s' 启动失败，已跳过: %s", name, e)

    # ── 公共 API ──

    def get_tool_classes(self) -> list[type[Tool]]:
        """返回所有 MCP 工具类列表。"""
        return list(self._tools)

    def shutdown(self) -> None:
        """关闭所有 MCP 连接。"""
        for name, conn in self._connections.items():
            try:
                conn.close()
            except Exception as e:
                logger.warning("关闭 MCP server '%s' 时出错: %s", name, e)
        self._connections.clear()
        self._tools.clear()
