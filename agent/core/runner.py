"""Agent 执行引擎 —— think-act 循环编排。

不持有状态，不关心历史存储细节。
LLM → LLMClient，工具 → ToolExecutor，事件 → EventBus。
"""

from __future__ import annotations
from types import SimpleNamespace
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import TokenTracker
    from .events import EventBus


class AgentRunner:
    """执行 think-act 循环。

    用法:
        runner = AgentRunner(llm_client, tool_executor, token_tracker, event_bus)
        for chunk in runner.step(history):
            print(chunk)
    """

    def __init__(
        self,
        llm_client,
        tool_executor,
        token_tracker: TokenTracker | None = None,
        event_bus: EventBus | None = None,
        max_turns: int | None = None,
    ):
        self.llm = llm_client
        self.tools = tool_executor
        self._tracker = token_tracker
        self.bus = event_bus
        self.max_turns = max_turns

    def step(self, history: list[dict]) -> Generator[str, None, None]:
        """执行一轮完整对话。逐块产出文本。"""
        turns = 0
        while True:
            if self.max_turns is not None and turns >= self.max_turns:
                yield f"\n[达到最大轮数 {self.max_turns}，已熔断]\n"
                break

            tool_schemas = self.tools.registry.get_tool_schemas()
            turns += 1

            if self.bus:
                self.bus.emit("turn:start", {"turn": turns})

            # ── 1. LLM 流式调用 ──
            full_content, full_reasoning = "", ""
            tool_calls_accum: dict[int, dict] = {}

            for chunk in self.llm.stream(history, tool_schemas):
                t = chunk["type"]
                if t == "usage":
                    self._record(chunk)
                elif t == "reasoning":
                    full_reasoning += chunk["text"]
                elif t == "content":
                    full_content += chunk["text"]
                    yield chunk["text"]
                elif t == "tool_call":
                    idx = chunk["index"]
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {"id": None, "name": None, "arguments": ""}
                    for k in ("id", "name", "arguments"):
                        if k in chunk and chunk[k]:
                            if k == "arguments":
                                tool_calls_accum[idx][k] += chunk[k]  # 拼接分片
                            else:
                                tool_calls_accum[idx][k] = chunk[k]   # id/name 覆盖

            # ── 2. 写入助手消息 ──
            # 确保 API 兼容：content 和 tool_calls 不能同时为空
            has_tool_calls = bool(tool_calls_accum)
            assistant_content: str | None = full_content or None
            if assistant_content is None and not has_tool_calls:
                assistant_content = full_reasoning or ""  # fallback: 用 reasoning 或空串
            assistant_msg: dict = {"role": "assistant", "content": assistant_content}
            if full_reasoning:
                assistant_msg["reasoning_content"] = full_reasoning
            if has_tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_accum.values()
                ]
            history.append(assistant_msg)
            if self.bus:
                self.bus.emit("history:appended", {"message": assistant_msg})

            # ── 3. 无工具调用 → 结束 ──
            if not tool_calls_accum:
                if self.bus:
                    self.bus.emit("turn:end", {"text": full_content})
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
                tid = tc["id"]
                if tid in tool_results:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": tool_results[tid],
                    }
                    history.append(tool_msg)
                    if self.bus:
                        self.bus.emit("history:appended", {"message": tool_msg})

    def _record(self, chunk: dict) -> None:
        if not self._tracker:
            return
        self._tracker.record(self.llm.model, SimpleNamespace(
            prompt_tokens=chunk.get("input", 0),
            completion_tokens=chunk.get("output", 0),
            prompt_cache_hit_tokens=chunk.get("cache_hit", 0),
            prompt_cache_miss_tokens=chunk.get("cache_miss", 0),
        ))
