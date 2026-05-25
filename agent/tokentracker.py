"""Token usage tracking — per-call JSONL log + aggregations."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class TokenTracker:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_input_tokens = 0

    def record(self, model: str, usage: Any) -> None:
        """记录 OpenAI/DeepSeek 格式的 token 使用情况"""
        if not usage:
            return

        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "input": getattr(usage, "prompt_tokens", 0) or 0,
            "output": getattr(usage, "completion_tokens", 0) or 0,
            # 处理 DeepSeek 的缓存命中/未命中 (OpenAI 兼容字段)
            "cache_hit": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
            "cache_miss": getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
        }
        
        self._last_input_tokens = row["input"]
        
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush() # 强制刷新到磁盘

    def last_input_tokens(self) -> int:
        return self._last_input_tokens

    def should_compact(self, max_context: int, threshold: float = 0.7) -> bool:
        return self._last_input_tokens > max_context * threshold

    def _iter_rows(self):
        if not self.log_file.exists():
            return
        with self.log_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def stats_by_date(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "cache_hit": 0, "cache_miss": 0})
        for r in self._iter_rows():
            date = r.get("ts", "")[:10]
            for k in ("input", "output", "cache_hit", "cache_miss"):
                out[date][k] += r.get(k, 0)
        return dict(out)

    def stats_by_model(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "cache_hit": 0, "cache_miss": 0})
        for r in self._iter_rows():
            m = r.get("model", "unknown")
            for k in ("input", "output", "cache_hit", "cache_miss"):
                out[m][k] += r.get(k, 0)
        return dict(out)