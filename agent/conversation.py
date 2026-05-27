"""对话状态管理 —— 统一管理历史记录、记忆压缩和 token 统计。"""

from __future__ import annotations
from typing import Any
from pathlib import Path
from openai import OpenAI


class Conversation:
    """管理一次对话会话的全部状态：消息历史、持久化记忆、Token 计数。

    之前这些职责分散在 AgentRunner 和 AgentLoop 中，
    现在统一到这里，Runner 只负责"执行"，Conversation 负责"状态"。
    """

    def __init__(
        self,
        memory: Any,           # AgentMemory 实例
        token_tracker: Any,    # TokenTracker 实例
        system_prompt: str,
        max_context: int = 200_000,
        compact_threshold: float = 0.7,
    ):
        self.memory = memory
        self.token_tracker = token_tracker
        self.max_context = max_context
        self.compact_threshold = compact_threshold

        # 启动时将系统提示词注入为第一条消息
        system_msg = {"role": "system", "content": system_prompt}
        self.memory.append_history(system_msg)

    @property
    def history(self) -> list[dict]:
        """当前完整消息历史（含系统提示词）"""
        return self.memory.history

    def add_user_message(self, text: str) -> None:
        """追加用户消息到历史"""
        msg = {"role": "user", "content": text}
        self.memory.append_history(msg)

    def add_assistant_message(self, msg: dict) -> None:
        """追加助手消息到历史"""
        self.memory.append_history(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """追加工具结果到历史"""
        msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        self.memory.append_history(msg)

    def record_tokens(self, model: str, usage: Any) -> None:
        """记录一次 LLM 调用的 Token 消耗"""
        self.token_tracker.record(model, usage)

    def should_compact(self) -> bool:
        """是否需要触发记忆压缩"""
        return self.token_tracker.should_compact(
            self.max_context, self.compact_threshold
        )

    def compact(self) -> dict[str, Any]:
        """执行记忆压缩"""
        return self.memory.compact()

    def token_stats(self) -> dict[str, dict[str, int]]:
        """获取 Token 统计"""
        return self.token_tracker.stats_by_model()
