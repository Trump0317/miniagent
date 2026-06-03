"""可观测性 —— 结构化日志、请求追踪、耗时统计。

通过订阅 EventBus 事件，自动记录 agent 运行过程中的关键指标。
输出到 sessions/<ts>/trace.jsonl（JSON Lines 格式）。

事件覆盖:
    message:received → request:start
    turn:start       → turn 开始计时
    turn:end         → turn 结束，记录耗时
    tool:before      → 工具调用开始计时
    tool:after       → 工具调用结束，记录耗时+结果
    session:end      → request:end，输出汇总指标
"""

from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventBus


class Observability:
    """可观测性收集器。

    用法:
        obs = Observability(log_dir=config.session_dir)
        obs.attach(event_bus)

        # agent.process() → 自动记录所有事件
        # agent.shutdown() → 自动输出汇总

        stats = obs.summary()  # 获取当前请求的汇总指标
    """

    def __init__(self, log_dir: str | Path | None = None):
        self._log_path: Path | None = None
        if log_dir:
            dir_path = Path(log_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            self._log_path = dir_path / "trace.jsonl"

        self._trace_id = ""
        self._request_start = 0.0
        self._turn_starts: dict[int, float] = {}
        self._tool_starts: dict[str, float] = {}
        self._errors: list[str] = []

        # 指标
        self._turns = 0
        self._tool_calls = 0
        self._tool_failures = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    # ── 公共 API ──

    def attach(self, bus: "EventBus") -> None:
        """订阅 EventBus 事件，自动记录。"""
        bus.on("message:received", self._on_message_received)
        bus.on("turn:start", self._on_turn_start)
        bus.on("turn:end", self._on_turn_end)
        bus.on("tool:before", self._on_tool_before)
        bus.on("tool:after", self._on_tool_after)
        bus.on("session:end", self._on_session_end)
        bus.on("context:high", self._on_context_high)

    def summary(self) -> dict:
        """返回当前请求的汇总指标。"""
        elapsed = time.time() - self._request_start if self._request_start else 0
        return {
            "trace_id": self._trace_id,
            "elapsed_ms": round(elapsed * 1000),
            "turns": self._turns,
            "tool_calls": self._tool_calls,
            "tool_failures": self._tool_failures,
            "errors": len(self._errors),
            "tokens_in": self._total_tokens_in,
            "tokens_out": self._total_tokens_out,
        }

    # ── 事件处理 ──

    def _on_message_received(self, event) -> None:
        self._trace_id = uuid.uuid4().hex[:8]
        self._request_start = time.time()
        self._turns = 0
        self._tool_calls = 0
        self._tool_failures = 0
        self._errors = []
        text = (event.data or {}).get("text", "")
        self._log("request:start", {
            "message": text[:200],
        })

    def _on_turn_start(self, event) -> None:
        turn = (event.data or {}).get("turn", self._turns + 1)
        self._turns = turn
        self._turn_starts[turn] = time.time()
        self._log("turn:start", {"turn": turn})

    def _on_turn_end(self, event) -> None:
        turn = self._turns
        start = self._turn_starts.pop(turn, None)
        duration = self._elapsed(start)
        self._log("turn:end", {"turn": turn, "duration_ms": duration})

    def _on_tool_before(self, event) -> None:
        data = event.data or {}
        name = data.get("name", "unknown")
        if name == "subagent_tool":
            label = "subagent:" + str(data.get("args", {}).get("agent", "?"))
        else:
            label = name
        self._tool_calls += 1
        self._tool_starts[label] = time.time()
        self._log("tool:start", {"tool": label})

    def _on_tool_after(self, event) -> None:
        data = event.data or {}
        name = data.get("name", "unknown")
        if name == "subagent_tool":
            label = "subagent:" + str(data.get("args", {}).get("agent", "?"))
        else:
            label = name
        result = data.get("result", "")
        start = self._tool_starts.pop(label, None)
        duration = self._elapsed(start)
        is_error = "error" in str(result).lower() or "错误" in str(result)
        if is_error:
            self._tool_failures += 1
        # 截断结果，避免日志过大
        result_preview = str(result)[:200]
        self._log("tool:end", {
            "tool": label,
            "duration_ms": duration,
            "ok": not is_error,
            "result": result_preview,
        })

    def _on_session_end(self, _event) -> None:
        s = self.summary()
        # 从 tracker 获取 token 统计（如果有）
        self._log("request:end", s)
        # 汇总
        self._write_line({
            "ts": self._now(),
            "level": "INFO",
            "trace": self._trace_id,
            "event": "request:summary",
            "data": s,
        })

    def _on_context_high(self, _event) -> None:
        self._log("context:high", {"turns": self._turns})

    # ── 内部 ──

    def _log(self, event: str, data: dict | None = None) -> None:
        self._write_line({
            "ts": self._now(),
            "level": "INFO",
            "trace": self._trace_id,
            "event": event,
            "data": data or {},
        })

    def _write_line(self, entry: dict) -> None:
        if self._log_path:
            try:
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    @staticmethod
    def _elapsed(start: float | None) -> int:
        if start is None:
            return 0
        return round((time.time() - start) * 1000)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
