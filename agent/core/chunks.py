"""统一 chunk 协议 — 定义 agent→consumer 的结构化输出格式。

Runner、Executor 等生产者产出 chunk dict，CLI/Web 等消费者按 type 分发渲染。
"""

from __future__ import annotations
from enum import StrEnum
from typing import Any


class ChunkType(StrEnum):
    """chunk 类型枚举。"""
    TEXT = "text"                # LLM 正文 / 工具实时输出
    REASONING = "reasoning"      # LLM 推理/思考内容
    TOOL_STATUS = "tool_status"  # 工具执行状态
    TOOL_RESULT = "tool_result"  # 工具最终结果
    DONE = "done"                # 本轮结束


def text_chunk(content: str) -> dict[str, Any]:
    return {"type": ChunkType.TEXT, "content": content}


def reasoning_chunk(content: str) -> dict[str, Any]:
    return {"type": ChunkType.REASONING, "content": content}


def tool_status_chunk(content: str) -> dict[str, Any]:
    return {"type": ChunkType.TOOL_STATUS, "content": content}


def tool_result_chunk(id: str, result: str) -> dict[str, Any]:
    return {"type": ChunkType.TOOL_RESULT, "id": id, "result": result}


def done_chunk() -> dict[str, Any]:
    return {"type": ChunkType.DONE}
