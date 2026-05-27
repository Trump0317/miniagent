"""Agent 执行引擎 —— 负责 LLM 流式调用和工具执行编排。

历史管理和 Token 统计不在本模块处理，而是通过 Conversation 对象委托。
这使得 Runner 可以独立测试，也可被 SubagentTool 用原始 history list 复用。
"""

from __future__ import annotations
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from .conversation import Conversation
    from .tools.ToolRegisty.registry import ToolRegistry
    from .tokentracker import TokenTracker

# 工具结果截断参数
MAX_RESULT_BYTES = 50 * 1024
MAX_RESULT_LINES = 2000
MAX_PARALLEL_TOOLS = 8


class AgentRunner:
    """执行 agent 的 think-act 循环。

    两种使用模式：
    1. 主循环模式：传入 conversation，自动管理历史和 token
    2. 子代理模式：不传 conversation，由调用方管理原始 history list
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tool_registry: ToolRegistry | None = None,
        conversation: Conversation | None = None,
        token_tracker: TokenTracker | None = None,
        max_turns: int | None = None,
        max_tokens: int = 20000,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.tool_registry = tool_registry
        self.conversation = conversation
        self._token_tracker = token_tracker
        self.max_turns = max_turns

    # ── 公共入口 ──

    def step(self, history: list[dict]):
        """执行一轮完整的对话（可能包含多个 tool-use 回合）。

        这是一个生成器，逐块产出 LLM 文本和工具执行状态。
        所有产出都是 str 类型，调用方直接 print。
        """
        turns = 0
        while True:
            if self.max_turns is not None and turns >= self.max_turns:
                yield f"\n[达到最大轮数 {self.max_turns}，已熔断]\n"
                break

            tools = self.tool_registry.get_tool_schemas() if self.tool_registry else []
            turns += 1

            # ── 1. LLM 流式调用 ──
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=history,
                tools=tools,
                stream=True,
                stream_options={"include_usage": True},
            )

            full_content = ""
            full_reasoning = ""
            tool_calls_accum: dict[int, dict] = {}

            for chunk in response:
                # Token 用量（流末尾的 usage chunk）
                if hasattr(chunk, "usage") and chunk.usage:
                    self._record_tokens(chunk.usage)
                    continue

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 思维链（DeepSeek R1）
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    full_reasoning += reasoning

                # 正文流式产出
                if delta.content:
                    full_content += delta.content
                    yield delta.content

                # 工具调用累积
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {"id": None, "name": None, "arguments": ""}
                        if tc.id:
                            tool_calls_accum[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_accum[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_accum[idx]["arguments"] += tc.function.arguments

            # ── 2. 构造助手消息并写入历史 ──
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

            # ── 3. 无工具调用 → 对话结束 ──
            if not tool_calls_accum:
                self._maybe_compact()
                return

            # ── 4. 执行工具 ──
            tool_results: dict[str, str] = {}
            for item in self._execute_tools(assistant_msg["tool_calls"]):
                if isinstance(item, dict):
                    tool_results[item["id"]] = item["result"]
                else:
                    yield str(item)

            # ── 5. 工具结果写入历史 ──
            for tc in assistant_msg["tool_calls"]:
                tc_id = tc["id"]
                if tc_id in tool_results:
                    self._add_tool_result(tc_id, tool_results[tc_id], history)

            # 继续循环，让模型根据结果回复
            continue

    # ── 历史管理（委托给 Conversation 或直接操作 list）──

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

    def _record_tokens(self, usage) -> None:
        if self.conversation:
            self.conversation.record_tokens(self.model, usage)
        elif self._token_tracker:
            self._token_tracker.record(self.model, usage)

    def _maybe_compact(self) -> None:
        if not self.conversation:
            return
        if not self.conversation.should_compact():
            return
        print(f"\n[Memory] 上下文用量接近上限, 自动压缩中...", flush=True)
        result = self.conversation.compact()
        if result.get("summary") or result.get("facts"):
            print(f"[Memory] 压缩完成", flush=True)

    # ── 工具执行 ──

    def _execute_tools(self, tool_calls: list[dict]) -> Generator[str | dict, None, None]:
        """执行工具调用。

        策略：
        - 单工具 → 串行
        - 多工具 && 全部 parallel_safe → 并行
        - 多工具 && 存在非安全工具 → 全部降级串行
        """
        count = len(tool_calls)
        if count == 1:
            yield from self._execute_serial(tool_calls)
            return

        all_safe = all(
            self.tool_registry.get_tool(tc["function"]["name"]).parallel_safe
            for tc in tool_calls
            if self.tool_registry.get_tool(tc["function"]["name"])
        )

        if all_safe:
            yield from self._execute_parallel(tool_calls)
        else:
            unsafe_names = [
                tc["function"]["name"]
                for tc in tool_calls
                if self.tool_registry.get_tool(tc["function"]["name"])
                and not self.tool_registry.get_tool(tc["function"]["name"]).parallel_safe
            ]
            yield f"\n[串行执行 {count} 个工具 (含非并发安全: {', '.join(unsafe_names)})...]\n"
            yield from self._execute_serial(tool_calls)

    def _execute_serial(self, tool_calls: list[dict]):
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = self._parse_args(tc)
            yield f"[执行工具: {name}...]\n"
            try:
                result = self.tool_registry.call_tool(name, args)
            except Exception as e:
                result = f"[错误] {name}: {e}"
            yield {"id": tc["id"], "result": self._truncate(result)}

    def _execute_parallel(self, tool_calls: list[dict]):
        count = len(tool_calls)
        workers = min(count, MAX_PARALLEL_TOOLS)
        yield f"\n[并行执行 {count} 个工具 (最多 {workers} 并发)...]\n"

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = self._parse_args(tc)
                futures[pool.submit(self.tool_registry.call_tool, name, args)] = (tc["id"], name)

            for future in as_completed(futures):
                tc_id, name = futures[future]
                try:
                    raw = future.result()
                except Exception as e:
                    raw = f"[错误] {name}: {e}"
                results[tc_id] = self._truncate(raw)
                yield f"[{name}] ✓\n"

        for tc in tool_calls:
            tc_id = tc["id"]
            if tc_id in results:
                yield {"id": tc_id, "result": results[tc_id]}

    # ── 工具函数 ──

    @staticmethod
    def _parse_args(tc: dict) -> dict:
        raw = tc.get("function", {}).get("arguments", "{}")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _truncate(result: str) -> str:
        """截断过长输出"""
        byte_len = len(result.encode("utf-8"))
        lines = result.split("\n")
        num_lines = len(lines)

        if byte_len <= MAX_RESULT_BYTES and num_lines <= MAX_RESULT_LINES:
            return result

        if num_lines > MAX_RESULT_LINES:
            lines = lines[:MAX_RESULT_LINES]
            result = "\n".join(lines)

        if len(result.encode("utf-8")) > MAX_RESULT_BYTES:
            raw = result.encode("utf-8")
            cut = MAX_RESULT_BYTES
            while cut > 0 and (raw[cut] & 0xC0) == 0x80:
                cut -= 1
            result = raw[:cut].decode("utf-8", errors="replace")

        return (
            result
            + f"\n\n[输出已截断: 原始 {byte_len} 字节 / {num_lines} 行, "
            + f"超出上限 {MAX_RESULT_BYTES} 字节 / {MAX_RESULT_LINES} 行]"
        )
