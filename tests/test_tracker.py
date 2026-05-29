"""TokenTracker 单元测试."""
import unittest
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.tracker import TokenTracker


class MockUsage:
    """模拟 OpenAI/DeepSeek API 返回的 usage 对象."""

    def __init__(self, prompt_tokens=0, completion_tokens=0,
                 prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_cache_hit_tokens = prompt_cache_hit_tokens
        self.prompt_cache_miss_tokens = prompt_cache_miss_tokens


class TestTokenTrackerInit(unittest.TestCase):
    """初始化测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp.name) / "tokens.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_last_input_tokens_initial_zero(self):
        """初始 last_input_tokens 为 0."""
        t = TokenTracker(self.log_file)
        self.assertEqual(t.last_input_tokens(), 0)

    def test_should_compact_initial_false(self):
        """无记录时 should_compact 返回 False."""
        t = TokenTracker(self.log_file)
        self.assertFalse(t.should_compact(100_000))


class TestTokenTrackerRecord(unittest.TestCase):
    """record 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp.name) / "tokens.jsonl"
        self.tracker = TokenTracker(self.log_file)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_creates_directory(self):
        """record 创建日志目录."""
        nested = Path(self.tmp.name) / "deep" / "nested" / "tokens.jsonl"
        t = TokenTracker(nested)
        t.record("deepseek", MockUsage(100, 50))
        self.assertTrue(nested.exists())

    def test_record_writes_jsonl(self):
        """record 写入 JSONL 行."""
        self.tracker.record("deepseek", MockUsage(100, 50))
        with self.log_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["model"], "deepseek")
        self.assertEqual(row["input"], 100)
        self.assertEqual(row["output"], 50)

    def test_record_includes_timestamp(self):
        """record 包含 ISO 时间戳."""
        self.tracker.record("deepseek", MockUsage(100, 50))
        with self.log_file.open() as f:
            row = json.loads(f.readline())
        self.assertIn("ts", row)
        self.assertIn("T", row["ts"])  # ISO 格式

    def test_record_updates_last_input_tokens(self):
        """record 更新 last_input_tokens."""
        self.tracker.record("deepseek", MockUsage(500, 200))
        self.assertEqual(self.tracker.last_input_tokens(), 500)

    def test_record_multiple_appends(self):
        """多次 record 追加而非覆盖."""
        self.tracker.record("deepseek", MockUsage(100, 50))
        self.tracker.record("deepseek", MockUsage(200, 100))
        with self.log_file.open() as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)

    def test_record_none_usage_ignored(self):
        """usage=None 时不写入."""
        self.tracker.record("deepseek", None)
        self.assertEqual(self.tracker.last_input_tokens(), 0)
        self.assertFalse(self.log_file.exists())

    def test_record_cache_tokens(self):
        """record 记录 DeepSeek 缓存命中/未命中."""
        self.tracker.record("deepseek", MockUsage(
            1000, 200,
            prompt_cache_hit_tokens=600,
            prompt_cache_miss_tokens=400,
        ))
        with self.log_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["cache_hit"], 600)
        self.assertEqual(row["cache_miss"], 400)

    def test_record_zero_tokens(self):
        """用量为 0 时也正常记录."""
        self.tracker.record("deepseek", MockUsage(0, 0))
        with self.log_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["input"], 0)
        self.assertEqual(row["output"], 0)

    def test_record_usage_missing_attributes(self):
        """usage 对象缺少缓存属性时 fallback 为 0."""
        class PartialUsage:
            prompt_tokens = 100
            completion_tokens = 50
            # 缺少 prompt_cache_hit_tokens 和 prompt_cache_miss_tokens

        self.tracker.record("deepseek", PartialUsage())
        with self.log_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["cache_hit"], 0)
        self.assertEqual(row["cache_miss"], 0)


class TestTokenTrackerReset(unittest.TestCase):
    """reset_session 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp.name) / "tokens.jsonl"
        self.tracker = TokenTracker(self.log_file)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reset_clears_last_input(self):
        """reset 清除 last_input_tokens."""
        self.tracker.record("deepseek", MockUsage(500, 200))
        self.tracker.reset_session()
        self.assertEqual(self.tracker.last_input_tokens(), 0)

    def test_reset_clears_file(self):
        """reset 清空文件."""
        self.tracker.record("deepseek", MockUsage(100, 50))
        self.tracker.reset_session()
        self.assertTrue(self.log_file.exists())
        self.assertEqual(self.log_file.read_text(), "")

    def test_reset_nonexistent_file(self):
        """reset 在文件不存在时不报错."""
        t = TokenTracker(Path(self.tmp.name) / "nonexistent" / "t.jsonl")
        t.reset_session()  # 不应抛异常


class TestTokenTrackerCompactCheck(unittest.TestCase):
    """should_compact 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp.name) / "tokens.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_below_threshold(self):
        """低于阈值返回 False."""
        t = TokenTracker(self.log_file)
        t.record("deepseek", MockUsage(30_000, 5_000))
        self.assertFalse(t.should_compact(100_000, threshold=0.35))

    def test_above_threshold(self):
        """高于阈值返回 True."""
        t = TokenTracker(self.log_file)
        t.record("deepseek", MockUsage(80_000, 5_000))
        self.assertTrue(t.should_compact(100_000, threshold=0.35))

    def test_exactly_at_threshold(self):
        """等于阈值返回 False（严格大于才触发）."""
        t = TokenTracker(self.log_file)
        t.record("deepseek", MockUsage(35_000, 5_000))
        self.assertFalse(t.should_compact(100_000, threshold=0.35))

    def test_default_threshold(self):
        """默认 threshold=0.35（与 AppConfig 一致）."""
        t = TokenTracker(self.log_file)
        t.record("deepseek", MockUsage(40_000, 5_000))
        # 40000 > 100000 * 0.35 = 35000 → True
        self.assertTrue(t.should_compact(100_000))


class TestTokenTrackerStats(unittest.TestCase):
    """stats_by_date / stats_by_model 聚合测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp.name) / "tokens.jsonl"
        self.tracker = TokenTracker(self.log_file)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats_by_date_empty(self):
        """无记录时返回空字典."""
        self.assertEqual(self.tracker.stats_by_date(), {})

    def test_stats_by_model_empty(self):
        """无记录时返回空字典."""
        self.assertEqual(self.tracker.stats_by_model(), {})

    def test_stats_by_date_aggregation(self):
        """按日期聚合 token."""
        rows = [
            {"ts": "2026-05-29T10:00:00", "model": "deepseek", "input": 100, "output": 50,
             "cache_hit": 30, "cache_miss": 20},
            {"ts": "2026-05-29T11:00:00", "model": "deepseek", "input": 200, "output": 100,
             "cache_hit": 0, "cache_miss": 0},
            {"ts": "2026-05-30T09:00:00", "model": "openai", "input": 300, "output": 150,
             "cache_hit": 0, "cache_miss": 0},
        ]
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        stats = self.tracker.stats_by_date()
        self.assertIn("2026-05-29", stats)
        self.assertIn("2026-05-30", stats)
        self.assertEqual(stats["2026-05-29"]["input"], 300)  # 100 + 200
        self.assertEqual(stats["2026-05-30"]["input"], 300)

    def test_stats_by_model_aggregation(self):
        """按模型聚合 token."""
        rows = [
            {"ts": "2026-05-29T10:00:00", "model": "deepseek", "input": 100, "output": 50,
             "cache_hit": 30, "cache_miss": 20},
            {"ts": "2026-05-29T11:00:00", "model": "deepseek", "input": 200, "output": 100,
             "cache_hit": 10, "cache_miss": 40},
            {"ts": "2026-05-29T12:00:00", "model": "openai", "input": 500, "output": 250,
             "cache_hit": 0, "cache_miss": 0},
        ]
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        stats = self.tracker.stats_by_model()
        self.assertIn("deepseek", stats)
        self.assertIn("openai", stats)
        self.assertEqual(stats["deepseek"]["input"], 300)
        self.assertEqual(stats["deepseek"]["output"], 150)
        self.assertEqual(stats["deepseek"]["cache_hit"], 40)
        self.assertEqual(stats["deepseek"]["cache_miss"], 60)
        self.assertEqual(stats["openai"]["input"], 500)

    def test_stats_skip_malformed_lines(self):
        """跳过格式错误的 JSONL 行."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w") as f:
            f.write('{"ts":"2026-05-29T10:00:00","model":"deepseek","input":100,"output":50,"cache_hit":0,"cache_miss":0}\n')
            f.write('not valid json\n')
            f.write('{"ts":"2026-05-29T11:00:00","model":"deepseek","input":200,"output":100,"cache_hit":0,"cache_miss":0}\n')

        stats = self.tracker.stats_by_model()
        self.assertEqual(stats["deepseek"]["input"], 300)

    def test_stats_nonexistent_file(self):
        """文件不存在时不报错返回空字典."""
        t = TokenTracker(Path(self.tmp.name) / "nofile.jsonl")
        self.assertEqual(t.stats_by_date(), {})
        self.assertEqual(t.stats_by_model(), {})

    def test_stats_blank_lines_between_rows(self):
        """JSONL 中空白行被安全跳过."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w") as f:
            f.write('{"ts":"2026-05-29T10:00:00","model":"d","input":100,"output":50,"cache_hit":0,"cache_miss":0}\n')
            f.write('\n')
            f.write('{"ts":"2026-05-29T11:00:00","model":"d","input":200,"output":100,"cache_hit":0,"cache_miss":0}\n')
        stats = self.tracker.stats_by_model()
        self.assertEqual(stats["d"]["input"], 300)

    def test_stats_missing_ts_field(self):
        """缺少 ts 字段的行，聚合到空字符串 key."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w") as f:
            f.write('{"model":"d","input":100,"output":50,"cache_hit":0,"cache_miss":0}\n')
        stats = self.tracker.stats_by_date()
        self.assertIn("", stats)
        self.assertEqual(stats[""]["input"], 100)

    def test_stats_missing_model_field(self):
        """缺少 model 字段的行，聚合到 'unknown' key."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w") as f:
            f.write('{"ts":"2026-05-29T10:00:00","input":100,"output":50,"cache_hit":0,"cache_miss":0}\n')
        stats = self.tracker.stats_by_model()
        self.assertIn("unknown", stats)
        self.assertEqual(stats["unknown"]["input"], 100)


if __name__ == "__main__":
    unittest.main()
