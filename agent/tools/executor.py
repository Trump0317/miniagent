"""工具执行器 —— 串行/并行工具调度，钩子集成，结果截断。

产出统一 chunk 格式（见 agent/core/chunks.py）。
"""

from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, TYPE_CHECKING

from ..core.chunks import text_chunk, tool_status_chunk, tool_result_chunk

if TYPE_CHECKING:
    from ..core.events import EventBus
    from .registry import ToolRegistry

MAX_RESULT_BYTES = 50 * 1024
MAX_RESULT_LINES = 2000
MAX_PARALLEL_TOOLS = 8


class ToolExecutor:
    """负责工具的实际执行：调度策略、钩子回调、结果截断。"""

    def __init__(self, registry: ToolRegistry, event_bus: EventBus | None = None):
        self.registry = registry
        self.bus = event_bus

    def execute(self, tool_calls: list[dict]) -> Generator[dict, None, None]:
        """执行工具调用。产出状态文本和结果字典。

        策略：
        - 单工具 → 串行
        - 多工具 && 全部 parallel_safe → 并行（最多 8 并发）
        - 多工具 && 存在非安全工具 → 全部降级串行
        """
        count = len(tool_calls)
        if count == 1:
            yield from self._serial(tool_calls)
            return

        all_safe = all(
            self.registry.get_tool(tc["function"]["name"]).parallel_safe
            for tc in tool_calls
            if self.registry.get_tool(tc["function"]["name"])
        )

        if all_safe:
            yield from self._parallel(tool_calls)
        else:
            unsafe_names = [
                tc["function"]["name"]
                for tc in tool_calls
                if self.registry.get_tool(tc["function"]["name"])
                and not self.registry.get_tool(tc["function"]["name"]).parallel_safe
            ]
            yield tool_status_chunk(
                f"\n[串行执行 {count} 个工具 (含非并发安全: {', '.join(unsafe_names)})...]\n"
            )
            yield from self._serial(tool_calls)

    def _serial(self, tool_calls: list[dict]):
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = self._parse_args(tc)
            yield tool_status_chunk(f"\n[执行工具: {name}...]\n")

            # before 事件
            blocked_result: str | None = None
            if self.bus:
                responses = self.bus.emit("tool:before", {"name": name, "args": args}, source="executor")
                for resp in responses:
                    if isinstance(resp, dict) and resp.get("block"):
                        blocked_result = f"[拦截] {name}: {resp.get('reason', '被事件拦截')}"
                        break

            if blocked_result:
                yield tool_result_chunk(tc["id"], self._truncate(blocked_result))
                continue

            # 流式执行工具，逐块产出
            chunks: list[str] = []
            for chunk in self._run_one(name, args):
                if chunk:
                    chunks.append(str(chunk))
                    yield text_chunk(str(chunk))

            result = "".join(chunks)

            # after 事件
            if self.bus:
                responses = self.bus.emit("tool:after", {"name": name, "result": result}, source="executor")
                for resp in responses:
                    if isinstance(resp, str):
                        result = resp

            yield tool_result_chunk(tc["id"], self._truncate(result))

    def _parallel(self, tool_calls: list[dict]):
        count = len(tool_calls)
        workers = min(count, MAX_PARALLEL_TOOLS)
        yield tool_status_chunk(f"\n[并行执行 {count} 个工具 (最多 {workers} 并发)...]\n")

        results: dict[str, str] = {}

        def _run_one(tc):
            name = tc["function"]["name"]
            args = self._parse_args(tc)

            if self.bus:
                responses = self.bus.emit("tool:before", {"name": name, "args": args}, source="executor")
                for resp in responses:
                    if isinstance(resp, dict) and resp.get("block"):
                        return tc["id"], name, f"[拦截] {name}: {resp.get('reason', '被事件拦截')}"

            chunks: list[str] = []
            for chunk in self._run_one(name, args):
                if chunk:
                    chunks.append(str(chunk))
            raw = "".join(chunks)

            if self.bus:
                responses = self.bus.emit("tool:after", {"name": name, "result": raw}, source="executor")
                for resp in responses:
                    if isinstance(resp, str):
                        raw = resp

            return tc["id"], name, raw

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, tc): tc for tc in tool_calls}
            for future in as_completed(futures):
                tc_id, name, raw = future.result()
                results[tc_id] = self._truncate(raw)
                yield tool_status_chunk(f"[{name}] ✓\n")

        for tc in tool_calls:
            tc_id = tc["id"]
            if tc_id in results:
                yield tool_result_chunk(tc_id, results[tc_id])

    def _run_one(self, name: str, args: dict):
        """生成器: 逐块产出工具输出。"""
        tool = self.registry.get_tool(name)
        if not tool:
            yield f"[错误] 未找到工具 '{name}'"
            return

        try:
            validated = tool.cast_params(args)
        except ValueError as e:
            yield f"参数校验错误: {e}"
            return

        if tool.supports_streaming:
            try:
                for chunk in tool.stream_execute(**validated):
                    if chunk:
                        yield str(chunk)
            except Exception as e:
                yield f"[错误] {name}: {e}"
        else:
            try:
                yield self.registry.call_tool(name, validated)
            except Exception as e:
                yield f"[错误] {name}: {e}"

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
