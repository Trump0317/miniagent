"""事件钩子 —— 在工具执行前后插入自定义逻辑。

用法:
    hooks = EventHooks()
    hooks.on_tool_call = lambda name, args: print(f"调用 {name}")
    hooks.on_tool_result = lambda name, result: result[:100]  # 截断结果
    
    runner = AgentRunner(..., hooks=hooks)
"""

from __future__ import annotations
from typing import Callable, Any


class EventHooks:
    """可选的钩子回调集合。

    每个钩子都是一个 Callable，接收特定参数，返回可选的控制指令。
    返回 None 表示继续正常流程；返回 dict 可修改参数或拦截执行。
    """

    def __init__(self):
        # on_tool_call: 工具执行前调用
        #   签名: (tool_name: str, args: dict) → dict | None
        #   返回 {"block": True, "reason": "..."} 阻止执行
        #   返回 {"args": {...}} 替换参数
        #   返回 None 继续正常流程
        self.on_tool_call: Callable[[str, dict], dict | None] | None = None

        # on_tool_result: 工具执行后调用
        #   签名: (tool_name: str, result: str) → str | None
        #   返回 str 替换原结果
        #   返回 None 保持原结果不变
        self.on_tool_result: Callable[[str, str], str | None] | None = None


def apply_tool_call_hook(
    hooks: EventHooks | None,
    tool_name: str,
    args: dict,
) -> tuple[dict, str | None]:
    """在工具执行前应用钩子。

    返回 (修改后的 args, 拦截消息)。
    如果拦截消息非 None，则跳过执行，直接返回该消息。
    """
    if not hooks or not hooks.on_tool_call:
        return args, None

    try:
        result = hooks.on_tool_call(tool_name, args.copy())
    except Exception as e:
        return args, f"[钩子异常] on_tool_call: {e}"

    if result is None:
        return args, None

    if result.get("block"):
        return args, result.get("reason", "被钩子拦截")

    if "args" in result:
        return result["args"], None

    return args, None


def apply_tool_result_hook(
    hooks: EventHooks | None,
    tool_name: str,
    result: str,
) -> str:
    """在工具执行后应用钩子，返回（可能被修改的）结果。"""
    if not hooks or not hooks.on_tool_result:
        return result

    try:
        new_result = hooks.on_tool_result(tool_name, result)
    except Exception as e:
        return f"{result}\n[钩子异常] on_tool_result: {e}"

    return new_result if new_result is not None else result
