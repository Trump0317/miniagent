from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging

from agent.tools.base import Tool

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
            
            if isinstance(result, str) and ("error" in result.lower() or result.startswith("错误")):
                return f"{result}{self._ERROR_HINT}"
            return str(result)

        except ValueError as ve:
            return f"参数校验错误: {ve}{self._ERROR_HINT}"
        except Exception as e:
            return f"执行错误 {name}: {e}{self._ERROR_HINT}"


# ═══════════════════════════════════════════════════════════════
# 默认注册表工厂
# ═══════════════════════════════════════════════════════════════

_DEFAULT_SUBAGENT_PROMPT = (
    "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
    "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
)

_DEFAULT_TOOLS = None  # 延迟初始化，避免循环导入


def build_default_registry(
    cfg,           # AppConfig (避免循环导入，不标注类型)
    skills,        # SkillsLoader
    client,        # OpenAI client
    agent_loader,  # AgentLoader
    tracker,       # TokenTracker
) -> ToolRegistry:
    """构建 Agent 的默认工具注册表（含子代理）。

    包含: Bash, FileRead, FileWrite, FileEdit, WebFetch, WebSearch,
          TodoWrite, SkillTool, SubagentTool。
    """
    from .bash import BashTool
    from .file_read import FileReadTool
    from .file_write import FileWriteTool
    from .file_edit import FileEditTool
    from .web_fetch import WebFetchTool
    from .web_search import WebSearchTool
    from .todo import TodoWriteTool
    from .skill import SkillTool
    from .subagent import SubagentTool

    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        _DEFAULT_TOOLS = (BashTool, FileReadTool, FileWriteTool, FileEditTool,
                          WebFetchTool, WebSearchTool, TodoWriteTool)

    registry = ToolRegistry()
    for tool_cls in _DEFAULT_TOOLS:
        registry.register(tool_cls())
    registry.register(SkillTool(skills))

    sub = ToolRegistry()
    for tool_cls in _DEFAULT_TOOLS:
        sub.register(tool_cls())

    registry.register(SubagentTool(
        client=client, model=cfg.model, registry=sub,
        token_tracker=tracker, agent_loader=agent_loader,
        system_prompt=_DEFAULT_SUBAGENT_PROMPT,
        max_turns=cfg.subagent_max_turns, sub_model=cfg.subagent_model,
    ))
    return registry
