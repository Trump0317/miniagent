from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging

from agent.tools.ToolRegisty.base import Tool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    工具注册表：负责管理所有可用的工具，并提供调用和描述生成功能。
    """
    _ERROR_HINT = "\n[分析上述错误并尝试不同的方案。]"

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._schemas_cache: Optional[List[Dict[str, Any]]] = None

    def register(self, tool: Tool) -> None:
        """注册一个工具实例"""
        if tool.name in self._tools:
            logger.warning(f"工具 {tool.name} 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        self._schemas_cache = None

    def get_tool(self, name: str) -> Optional[Tool]:
        """根据名称获取工具实例"""
        return self._tools.get(name)

    @property
    def tool_names(self) -> List[str]:
        """获取所有已注册工具的名称列表"""
        return sorted(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        生成符合标准 Tool Call 格式的描述列表。
        使用格式兼容 OpenAI/DeepSeek 等模型。
        """
        if self._schemas_cache is not None:
            return self._schemas_cache
        
        schemas = []
        for name in sorted(self._tools.keys()):
            tool = self._tools[name]
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        self._schemas_cache = schemas
        return self._schemas_cache

    def call_tool(self, name: str, arguments: Any) -> str:
        """
        统一调用接口：处理参数校验并执行工具。
        """
        tool = self.get_tool(name)
        if not tool:
            return f"错误: 未找到工具 '{name}'。可用: {', '.join(self.tool_names)}{self._ERROR_HINT}"

        if not isinstance(arguments, dict):
            return f"错误: 参数格式非对象 (got {type(arguments).__name__}){self._ERROR_HINT}"

        try:
            # 利用 Pydantic 进行转换和校验
            validated_params = tool.cast_params(arguments)
            result = tool.execute(**validated_params)
            
            if isinstance(result, str) and (result.lower().startswith("error") or result.startswith("错误")):
                return f"{result}{self._ERROR_HINT}"
            return str(result)

        except ValueError as ve:
            return f"参数校验错误: {ve}{self._ERROR_HINT}"
        except Exception as e:
            return f"执行错误 {name}: {e}{self._ERROR_HINT}"
