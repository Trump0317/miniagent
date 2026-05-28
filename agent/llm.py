"""LLM 客户端 —— 封装 OpenAI 兼容的流式聊天调用。"""

from __future__ import annotations
from typing import Any, Generator
from openai import OpenAI


class LLMClient:
    """封装 LLM 流式调用，产出 (content_chunk | reasoning_chunk | usage_chunk | tool_call_chunk)。

    每个 chunk 是一个 dict，包含 type 和对应的数据。
    """

    # thinking → reasoning_effort 映射
    _THINKING_MAP = {
        "off": None,
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }

    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_tokens: int = 20000,
        thinking: str | None = None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[dict, None, None]:
        """流式调用 LLM。

        产出 chunk 类型:
            {"type": "content", "text": "..."}
            {"type": "reasoning", "text": "..."}
            {"type": "tool_call", "index": 0, "id": "...", "name": "...", "arguments": "..."}
            {"type": "usage", "input": 100, "output": 50, "cache_hit": 0, "cache_miss": 0}
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        effort = self._THINKING_MAP.get(self.thinking or "off")
        if effort:
            kwargs["reasoning_effort"] = effort

        response = self.client.chat.completions.create(**kwargs)

        for chunk in response:
            # Token 用量
            if hasattr(chunk, "usage") and chunk.usage:
                yield {
                    "type": "usage",
                    "input": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "output": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    "cache_hit": getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0,
                    "cache_miss": getattr(chunk.usage, "prompt_cache_miss_tokens", 0) or 0,
                }
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 思维链
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}

            # 正文
            if delta.content:
                yield {"type": "content", "text": delta.content}

            # 工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    data = {"type": "tool_call", "index": tc.index}
                    if tc.id:
                        data["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            data["name"] = tc.function.name
                        if tc.function.arguments:
                            data["arguments"] = tc.function.arguments
                    yield data
