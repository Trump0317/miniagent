"""Agent 执行引擎 —— think-act 循环编排。

LLM 调用委托给 LLMClient，工具执行委托给 ToolExecutor。
本模块只负责：组装 assistant 消息、管理历史、触发事件。
"""

from __future__ import annotations
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import Conversation
    from .tokentracker import TokenTracker
    from .events import EventBus


class AgentRunner:
    """执行 think-act 循环。

    两种模式：
    1. 主循环模式：传入 conversation，自动管理历史
    2. 子代理模式：不传 conversation，用原始 history list
    """

    def __init__(
        self,
        llm_client,
        tool_executor,
        conversation: Conversation | None = None,
        token_tracker: TokenTracker | None = None,
        event_bus: EventBus | None = None,
        max_turns: int | None = None,
    ):
        self.llm = llm_client
        self.tools = tool_executor
        self.conversation = conversation
        self._token_tracker = token_tracker
        self.bus = event_bus
        self.max_turns = max_turns

    def step(self, history: list[dict]) -> Generator[str, None, None]:
        """执行一轮完整对话（可能多轮 tool-use）。逐块产出文本。"""
        turns = 0
        while True:
            if self.max_turns is not None and turns >= self.max_turns:
                yield f"\n[达到最大轮数 {self.max_turns}，已熔断]\n"
                break

            tool_schemas = self.tools.registry.get_tool_schemas()
            turns += 1

            # 事件：轮次开始
            if self.bus:
                self.bus.emit("turn:start", {"turn": turns}, source="runner")

            # ── 1. LLM 流式调用 ──
            full_content = ""
            full_reasoning = ""
            tool_calls_accum: dict[int, dict] = {}

            for chunk in self.llm.stream(history, tool_schemas):
                if chunk["type"] == "usage":
                    self._record_usage(chunk)

                elif chunk["type"] == "reasoning":
                    full_reasoning += chunk["text"]

                elif chunk["type"] == "content":
                    full_content += chunk["text"]
                    yield chunk["text"]

                elif chunk["type"] == "tool_call":
                    idx = chunk["index"]
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {"id": None, "name": None, "arguments": ""}
                    for key in ("id", "name", "arguments"):
                        if key in chunk and chunk[key]:
                            tool_calls_accum[idx][key] = chunk[key]

            # ── 2. 组装助手消息 ──
            assistant_msg: dict = {"role": "assistant", "content": full_content or None}
            if full_reasoning:
                assistant_msg["reasoning_content"] = full_reasoning

            if tool_calls_accum:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls_accum.values()
                ]

            self._add_to_history(assistant_msg, history)

            # ── 3. 无工具调用 → 结束 ──
            if not tool_calls_accum:
                self._maybe_compact()
                if self.bus:
                    self.bus.emit("turn:end", {"text": full_content}, source="runner")
                return

            # ── 4. 执行工具 ──
            tool_results: dict[str, str] = {}
            for item in self.tools.execute(assistant_msg["tool_calls"]):
                if isinstance(item, dict):
                    tool_results[item["id"]] = item["result"]
                else:
                    yield str(item)

            # ── 5. 写入工具结果 ──
            for tc in assistant_msg["tool_calls"]:
                tc_id = tc["id"]
                if tc_id in tool_results:
                    self._add_tool_result(tc_id, tool_results[tc_id], history)

            continue

    # ── 历史管理 ──

    def _add_to_history(self, msg: dict, history: list[dict]) -> None:
        if self.conversation:
            self.conversation.add_assistant_message(msg)
        else:
            history.append(msg)

    def _add_tool_result(self, call_id: str, content: str, history: list[dict]) -> None:
        if self.conversation:
            self.conversation.add_tool_result(call_id, content)
        else:
            history.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def _record_usage(self, chunk: dict) -> None:
        if self.conversation:
            # 构建兼容 usage 对象
            from types import SimpleNamespace
            usage = SimpleNamespace(
                prompt_tokens=chunk.get("input", 0),
                completion_tokens=chunk.get("output", 0),
                prompt_cache_hit_tokens=chunk.get("cache_hit", 0),
                prompt_cache_miss_tokens=chunk.get("cache_miss", 0),
            )
            self.conversation.record_tokens(self.llm.model, usage)
        elif self._token_tracker:
            self._token_tracker.record(
                self.llm.model,
                SimpleNamespace(
                    prompt_tokens=chunk.get("input", 0),
                    completion_tokens=chunk.get("output", 0),
                    prompt_cache_hit_tokens=chunk.get("cache_hit", 0),
                    prompt_cache_miss_tokens=chunk.get("cache_miss", 0),
                ),
            )

    def _maybe_compact(self) -> None:
        if not self.conversation:
            return
        if not self.conversation.should_compact():
            return

        if self.bus:
            self.bus.emit("context:high", {
                "input_tokens": self.conversation.token_tracker.last_input_tokens(),
                "max_context": self.conversation.max_context,
            }, source="runner")

        print(f"\n[Memory] 上下文用量接近上限, 自动压缩中...", flush=True)
        result = self.conversation.compact()
        if result.get("summary") or result.get("facts"):
            print(f"[Memory] 压缩完成", flush=True)
