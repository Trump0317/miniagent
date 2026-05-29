"""CompactionService 单元测试."""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.compaction import CompactionService


class TestNormalize(unittest.TestCase):
    """_normalize 静态方法测试."""

    def test_normalize_complete(self):
        """完整数据不变."""
        data = {
            "summary": {"critical": "关键", "decision": "决策", "issue": "问题"},
            "preferences": ["偏好1"],
            "facts": ["事实1"],
        }
        result = CompactionService._normalize(data)
        self.assertEqual(result["summary"]["critical"], "关键")
        self.assertEqual(result["summary"]["decision"], "决策")
        self.assertEqual(result["summary"]["issue"], "问题")
        self.assertEqual(result["preferences"], ["偏好1"])
        self.assertEqual(result["facts"], ["事实1"])

    def test_normalize_missing_fields(self):
        """缺失字段补默认值."""
        result = CompactionService._normalize({})
        self.assertEqual(result["summary"], {"critical": "无", "decision": "无", "issue": "无"})
        self.assertEqual(result["preferences"], [])
        self.assertEqual(result["facts"], [])

    def test_normalize_summary_not_dict(self):
        """summary 非 dict 时替换为空 dict."""
        result = CompactionService._normalize({"summary": "not a dict"})
        self.assertEqual(result["summary"], {"critical": "无", "decision": "无", "issue": "无"})

    def test_normalize_long_values_truncated(self):
        """过长的 summary 值截断到 60 字符."""
        long_text = "x" * 100
        result = CompactionService._normalize({"summary": {"critical": long_text}})
        self.assertEqual(len(result["summary"]["critical"]), 60)

    def test_normalize_prefs_not_list(self):
        """preferences 非 list 时替换为空列表."""
        result = CompactionService._normalize({"preferences": "not a list"})
        self.assertEqual(result["preferences"], [])

    def test_normalize_facts_not_list(self):
        """facts 非 list 时替换为空列表."""
        result = CompactionService._normalize({"facts": 123})
        self.assertEqual(result["facts"], [])


class TestFormatMessages(unittest.TestCase):
    """_format_messages 静态方法测试."""

    def test_basic_messages(self):
        """普通消息格式化."""
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = CompactionService._format_messages(history)
        self.assertIn("user: hello", result)
        self.assertIn("assistant: hi", result)

    def test_tool_calls_inline(self):
        """assistant 的 tool_calls 内联到内容前."""
        history = [{
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"function": {"name": "bash"}},
                {"function": {"name": "read"}},
            ],
        }]
        result = CompactionService._format_messages(history)
        self.assertIn("[调用工具: bash, read]", result)
        self.assertIn("calling", result)

    def test_empty_content(self):
        """content 为空时使用空字符串."""
        history = [{"role": "user"}]
        result = CompactionService._format_messages(history)
        self.assertIn("user: ", result)

    def test_unknown_role(self):
        """未知 role 显示为 unknown."""
        history = [{"role": "custom", "content": "test"}]
        result = CompactionService._format_messages(history)
        self.assertIn("custom: test", result)


class TestBuildCompactionSummary(unittest.TestCase):
    """_build_compaction_summary 测试."""

    def setUp(self):
        self.m = CompactionService.__new__(CompactionService)

    def test_full_summary(self):
        """完整摘要包含所有字段."""
        data = {
            "summary": {"critical": "关键", "decision": "决策", "issue": "问题"},
            "preferences": ["偏好1", "偏好2"],
            "facts": ["事实1"],
        }
        result = self.m._build_compaction_summary(data)
        self.assertIn("关键事件: 关键", result)
        self.assertIn("决策/产出: 决策", result)
        self.assertIn("问题: 问题", result)
        self.assertIn("用户偏好:", result)
        self.assertIn("事实:", result)

    def test_partial_summary(self):
        """部分字段有值."""
        data = {"summary": {"critical": "关键", "decision": "", "issue": ""}}
        result = self.m._build_compaction_summary(data)
        self.assertIn("关键事件: 关键", result)
        self.assertNotIn("决策/产出:", result)

    def test_empty_summary_with_prefs(self):
        """空 summary 但有偏好和事实."""
        data = {
            "summary": {"critical": "", "decision": "", "issue": ""},
            "preferences": ["偏好"],
            "facts": [],
        }
        result = self.m._build_compaction_summary(data)
        self.assertIn("用户偏好: 偏好", result)

    def test_completely_empty(self):
        """完全空白返回 '会话已压缩'."""
        data = {"summary": {"critical": "", "decision": "", "issue": ""}}
        result = self.m._build_compaction_summary(data)
        self.assertEqual(result, "会话已压缩")


class TestGetPreviousSummary(unittest.TestCase):
    """_get_previous_summary 测试."""

    def _build_service(self):
        m = CompactionService.__new__(CompactionService)
        m._memory = MagicMock()
        return m

    def test_no_compaction_in_path(self):
        """路径中没有 compaction 节点返回 None."""
        svc = self._build_service()
        # 构造 mock path entries
        from agent.core.session_tree import SessionEntry
        entries = [
            SessionEntry(id="1", parent_id=None, type="message", role="user", content="hi"),
        ]
        svc._memory.tree.path.return_value = entries
        self.assertIsNone(svc._get_previous_summary())

    def test_finds_last_compaction(self):
        """找到最近一个 compaction 节点的摘要."""
        svc = self._build_service()
        from agent.core.session_tree import SessionEntry
        entries = [
            SessionEntry(id="1", parent_id=None, type="message", role="user", content="q1"),
            SessionEntry(id="c1", parent_id="1", type="compaction", role="system",
                        summary="摘要1", first_kept_id="2", tokens_before=100),
            SessionEntry(id="2", parent_id="c1", type="message", role="assistant", content="a1"),
            SessionEntry(id="c2", parent_id="2", type="compaction", role="system",
                        summary="摘要2", first_kept_id="3", tokens_before=200),
            SessionEntry(id="3", parent_id="c2", type="message", role="user", content="q2"),
        ]
        svc._memory.tree.path.return_value = entries
        result = svc._get_previous_summary()
        self.assertEqual(result, "摘要2")


class TestDispatchToMemory(unittest.TestCase):
    """_dispatch_to_memory 测试."""

    def _build_service(self):
        m = CompactionService.__new__(CompactionService)
        m._memory = MagicMock()
        return m

    def test_dispatches_summary(self):
        """summary 有值时调用 append_summary."""
        svc = self._build_service()
        data = {"summary": {"critical": "关键", "decision": "决策", "issue": "问题"}}
        svc._dispatch_to_memory(data)
        svc._memory.append_summary.assert_called_once_with("关键", "决策", "问题")

    def test_dispatches_empty_summary(self):
        """summary 全空时不调用 append_summary."""
        svc = self._build_service()
        data = {"summary": {"critical": "", "decision": "", "issue": ""}}
        svc._dispatch_to_memory(data)
        svc._memory.append_summary.assert_not_called()

    def test_dispatches_preferences(self):
        """preferences 列表逐条 add_user."""
        svc = self._build_service()
        data = {"summary": {}, "preferences": ["偏好1", "偏好2"]}
        svc._dispatch_to_memory(data)
        svc._memory.add_user.assert_any_call("偏好1")
        svc._memory.add_user.assert_any_call("偏好2")
        self.assertEqual(svc._memory.add_user.call_count, 2)

    def test_dispatches_facts(self):
        """facts 列表逐条 add_memory."""
        svc = self._build_service()
        svc._memory.add_memory.return_value = True
        data = {"summary": {}, "facts": ["事实1", "事实2"]}
        count = svc._dispatch_to_memory(data)
        svc._memory.add_memory.assert_any_call("事实1")
        svc._memory.add_memory.assert_any_call("事实2")
        self.assertEqual(count, 2)

    def test_facts_count_only_new(self):
        """只统计新增的事实."""
        svc = self._build_service()
        svc._memory.add_memory.side_effect = [True, False, True]
        data = {"summary": {}, "facts": ["f1", "f2", "f3"]}
        count = svc._dispatch_to_memory(data)
        self.assertEqual(count, 2)


class TestFindCutPoint(unittest.TestCase):
    """_find_cut_point 测试."""

    def _build_service(self, keep_recent=30000):
        m = CompactionService.__new__(CompactionService)
        m._memory = MagicMock()
        m._keep_recent = keep_recent
        m._CHAR_PER_TOKEN = 3
        return m

    def _make_entries(self, *roles_and_contents):
        """创建 message entry 列表."""
        from agent.core.session_tree import SessionEntry
        entries = []
        for i, (role, content) in enumerate(roles_and_contents):
            entries.append(SessionEntry(
                id=str(i), parent_id=str(i - 1) if i > 0 else None,
                type="message", role=role, content=content
            ))
        return entries

    def test_too_few_entries(self):
        """少于 4 条消息返回 None."""
        svc = self._build_service()
        entries = self._make_entries(("user", "q1"), ("assistant", "a1"))
        svc._memory.tree.path.return_value = entries
        fid, msgs = svc._find_cut_point()
        self.assertIsNone(fid)

    def test_normal_cut(self):
        """在用户消息边界切割."""
        svc = self._build_service(keep_recent=1)  # 极低阈值，第一个用户消息就切
        entries = self._make_entries(
            ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
            ("user", "q3"), ("assistant", "a3"),
        )
        svc._memory.tree.path.return_value = entries
        fid, msgs = svc._find_cut_point()
        # q3 保留（在阈值内），q1/q2 之前压缩
        self.assertIsNotNone(fid)
        # 验证有消息要压缩且保留了后面
        self.assertGreater(len(msgs), 0)

    def test_high_threshold_triggers_fallback(self):
        """阈值太高时兜底保留最后 2 条用户消息."""
        svc = self._build_service(keep_recent=1000000)
        entries = self._make_entries(
            ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
            ("user", "q3"), ("assistant", "a3"),
        )
        svc._memory.tree.path.return_value = entries
        fid, msgs = svc._find_cut_point()
        self.assertIsNotNone(fid)
        # 兜底：保留最后 2 条用户消息 q2/q3，压缩 q1 之前
        user_msgs_to_keep = [m for m in msgs if m["role"] == "user"]
        # q1 被压缩，q2/q3 保留
        self.assertFalse(any("q2" in m["content"] for m in msgs))


class TestShouldCompact(unittest.TestCase):
    """should_compact 测试."""

    def test_delegates_to_tracker(self):
        """直接委托给 tracker.should_compact."""
        svc = CompactionService.__new__(CompactionService)
        svc._tracker = MagicMock()
        svc._max_context = 100000
        svc._threshold = 0.35
        svc._tracker.should_compact.return_value = True
        self.assertTrue(svc.should_compact())
        svc._tracker.should_compact.assert_called_once_with(100000, 0.35)


class TestCompact(unittest.TestCase):
    """compact 完整流程测试（mock LLM）."""

    def setUp(self):
        self.client = MagicMock()
        self.memory = MagicMock()
        self.tracker = MagicMock()
        self.prompt = MagicMock()

    def _build_service(self):
        return CompactionService(
            client=self.client,
            model="test-model",
            memory=self.memory,
            tracker=self.tracker,
            max_context=100000,
            compact_threshold=0.35,
            prompt=self.prompt,
        )

    def test_too_few_entries_skips(self):
        """条目太少时跳过压缩."""
        self.memory.non_system_entries.return_value = [
            {"role": "user", "content": "q1"},
        ]
        svc = self._build_service()
        prompt, data = svc.compact()
        self.assertEqual(data["summary"], {})
        self.assertEqual(data["facts"], [])

    def test_empty_summarize_triggers_early_return(self):
        """要压缩的消息少于 2 条时跳过压缩（不调用 LLM）."""
        self.memory.non_system_entries.return_value = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        # 让 find_cut_point 返回 None（少于2条要压缩的消息）
        # 模拟树只有4条消息，keep_recent 很大
        from agent.core.session_tree import SessionEntry
        entries = [
            SessionEntry(id="0", parent_id=None, type="message", role="user", content="q1"),
            SessionEntry(id="1", parent_id="0", type="message", role="assistant", content="a1"),
            SessionEntry(id="2", parent_id="1", type="message", role="user", content="q2"),
            SessionEntry(id="3", parent_id="2", type="message", role="assistant", content="a2"),
        ]
        self.memory.tree.path.return_value = entries
        svc = self._build_service()
        prompt, data = svc.compact()
        self.assertEqual(data["summary"], {})

    def test_full_compaction_flow(self):
        """完整压缩流程（mock LLM）."""
        # 准备足够条目触发压缩
        entries = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "q4"},
            {"role": "assistant", "content": "a4"},
        ]
        self.memory.non_system_entries.return_value = entries

        # 构造树路径让 find_cut_point 正常工作
        from agent.core.session_tree import SessionEntry
        path_entries = []
        for i in range(8):
            path_entries.append(SessionEntry(
                id=str(i), parent_id=str(i - 1) if i > 0 else None,
                type="message",
                role=entries[i]["role"],
                content=entries[i]["content"],
            ))
        self.memory.tree.path.return_value = path_entries

        # Mock LLM 响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"summary":{"critical":"关键","decision":"决策","issue":"问题"},"preferences":["偏好1"],"facts":["事实1"]}'
        mock_response.usage = None
        self.client.chat.completions.create.return_value = mock_response

        self.tracker.last_input_tokens.return_value = 50000
        self.memory.history = []  # _log 需要使用
        self.prompt.build.return_value = "new system prompt"

        svc = self._build_service()
        svc._keep_recent = 10  # 极小阈值触发切割

        prompt, data = svc.compact()

        # 验证返回值
        self.assertEqual(prompt, "new system prompt")
        # 验证 LLM 被调用
        self.client.chat.completions.create.assert_called_once()
        # 验证摘要、偏好、事实被分发
        self.memory.append_summary.assert_called_once()
        self.memory.add_user.assert_called_with("偏好1")
        self.memory.add_memory.assert_called_with("事实1")
        # 验证树压缩
        self.memory.compress_tree.assert_called_once()
        # 验证提示词重建
        self.prompt.build.assert_called_once_with(data)

    def test_compact_extract_error(self):
        """LLM 提取失败时返回错误摘要但不崩溃."""
        entries = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        self.memory.non_system_entries.return_value = entries

        from agent.core.session_tree import SessionEntry
        path_entries = []
        for i in range(6):
            path_entries.append(SessionEntry(
                id=str(i), parent_id=str(i - 1) if i > 0 else None,
                type="message",
                role=entries[i]["role"],
                content=entries[i]["content"],
            ))
        self.memory.tree.path.return_value = path_entries

        # LLM 抛异常
        self.client.chat.completions.create.side_effect = Exception("API error")
        self.tracker.last_input_tokens.return_value = 10000
        self.memory.history = []
        self.prompt.build.return_value = "fallback prompt"

        svc = self._build_service()
        svc._keep_recent = 10

        prompt, data = svc.compact()

        # 不应崩溃，返回包含错误信息的摘要
        self.assertIn("提取失败", data["summary"]["critical"])


if __name__ == "__main__":
    unittest.main()
