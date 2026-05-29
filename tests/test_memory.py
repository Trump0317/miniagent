"""AgentMemory 单元测试."""
import unittest
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.memory import AgentMemory


class TestAgentMemoryInit(unittest.TestCase):
    """初始化测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_memory_dir_and_summaries(self):
        """初始化创建 memory_dir 和 summaries 目录."""
        AgentMemory(self.memory_dir)
        self.assertTrue(self.memory_dir.exists())
        self.assertTrue((self.memory_dir / "summaries").exists())

    def test_creates_memory_file_with_header(self):
        """memory.md 不存在时创建含标头的文件."""
        AgentMemory(self.memory_dir)
        content = (self.memory_dir / "memory.md").read_text(encoding="utf-8")
        self.assertIn("# 长期记忆", content)

    def test_creates_user_file_with_header(self):
        """user.md 不存在时创建含标头的文件."""
        AgentMemory(self.memory_dir)
        content = (self.memory_dir / "user.md").read_text(encoding="utf-8")
        self.assertIn("# 用户信息", content)

    def test_existing_files_not_overwritten(self):
        """已有文件不被覆盖."""
        mem_file = self.memory_dir / "memory.md"
        self.memory_dir.mkdir(parents=True)
        mem_file.write_text("自定义内容\n", encoding="utf-8")
        AgentMemory(self.memory_dir)
        self.assertEqual(mem_file.read_text(encoding="utf-8"), "自定义内容\n")

    def test_history_file_not_created_until_write(self):
        """history.jsonl 在首条消息写入前不存在."""
        m = AgentMemory(self.memory_dir)
        self.assertFalse(m.history_file.exists())

    def test_with_session_dir(self):
        """支持独立 session_dir."""
        session_dir = Path(self.tmp.name) / "sessions" / "test"
        m = AgentMemory(self.memory_dir, session_dir=session_dir)
        self.assertEqual(m.history_file, session_dir / "history.jsonl")

    def test_empty_state(self):
        """初始状态为空."""
        m = AgentMemory(self.memory_dir)
        self.assertEqual(m.history, [])
        self.assertEqual(m.entry_count, 0)
        self.assertIsNone(m.leaf_id)
        self.assertFalse(m.has_data)


class TestAgentMemoryAppend(unittest.TestCase):
    """append_message 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_user_message_returns_id(self):
        """追加用户消息返回 entry_id."""
        eid = self.m.append_message({"role": "user", "content": "hello"})
        self.assertIsInstance(eid, str)
        self.assertEqual(len(eid), 12)

    def test_skip_system_message(self):
        """system 消息被跳过，返回空字符串."""
        eid = self.m.append_message({"role": "system", "content": "sys"})
        self.assertEqual(eid, "")
        self.assertEqual(self.m.entry_count, 0)

    def test_append_content_none(self):
        """content=None 也能正常追加（如只有 tool_calls 的 assistant 消息）."""
        eid = self.m.append_message({"role": "user", "content": None})
        self.m.append_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash"}}],
        })
        hist = self.m.history
        self.assertEqual(len(hist), 2)

    def test_append_creates_session_dir(self):
        """首条消息写入后 session_dir 被创建."""
        self.m.append_message({"role": "user", "content": "hello"})
        self.assertTrue(self.m.history_file.parent.exists())

    def test_append_writes_jsonl(self):
        """追加消息写入 JSONL."""
        self.m.append_message({"role": "user", "content": "hello"})
        self.assertTrue(self.m.history_file.exists())
        with self.m.history_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["role"], "user")
        self.assertEqual(row["content"], "hello")
        self.assertEqual(row["type"], "message")
        self.assertIn("id", row)
        self.assertIn("parent_id", row)

    def test_append_chain_parent_id(self):
        """连续追加时 parent_id 链正确."""
        eid1 = self.m.append_message({"role": "user", "content": "q1"})
        eid2 = self.m.append_message({"role": "assistant", "content": "a1"})
        with self.m.history_file.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["parent_id"], rows[0]["id"])

    def test_history_reflects_append(self):
        """history 属性反映追加的消息."""
        self.m.append_message({"role": "user", "content": "hello"})
        self.m.append_message({"role": "assistant", "content": "hi"})
        self.assertEqual(len(self.m.history), 2)
        self.assertEqual(self.m.history[0]["content"], "hello")
        self.assertEqual(self.m.history[1]["content"], "hi")

    def test_append_assistant_tool_calls(self):
        """assistant 消息提取 tool_calls 到 metadata."""
        self.m.append_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash"}}],
        })
        with self.m.history_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["metadata"]["tool_calls"], [{"function": {"name": "bash"}}])

    def test_append_assistant_reasoning(self):
        """assistant 消息提取 reasoning_content."""
        self.m.append_message({
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thinking...",
        })
        with self.m.history_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["metadata"]["reasoning_content"], "thinking...")

    def test_append_assistant_combined_metadata(self):
        """assistant 消息同时携带 tool_calls 和 reasoning_content."""
        self.m.append_message({
            "role": "assistant",
            "content": "answer",
            "tool_calls": [{"function": {"name": "read"}}],
            "reasoning_content": "let me think...",
        })
        with self.m.history_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["metadata"]["tool_calls"], [{"function": {"name": "read"}}])
        self.assertEqual(row["metadata"]["reasoning_content"], "let me think...")

    def test_append_tool_message(self):
        """tool 消息提取 tool_call_id."""
        self.m.append_message({
            "role": "tool",
            "content": "stdout",
            "tool_call_id": "call_123",
        })
        with self.m.history_file.open() as f:
            row = json.loads(f.readline())
        self.assertEqual(row["metadata"]["tool_call_id"], "call_123")

    def test_has_data_after_append(self):
        """追加后 has_data 为 True."""
        self.assertFalse(self.m.has_data)
        self.m.append_message({"role": "user", "content": "hello"})
        self.assertTrue(self.m.has_data)

    def test_entry_count(self):
        """entry_count 正确计数."""
        self.m.append_message({"role": "user", "content": "q1"})
        self.m.append_message({"role": "assistant", "content": "a1"})
        self.m.append_message({"role": "system", "content": "sys"})  # 跳过
        self.assertEqual(self.m.entry_count, 2)


class TestAgentMemoryFork(unittest.TestCase):
    """fork/navigate 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)
        self.m.append_message({"role": "user", "content": "task1"})
        self.a1 = self.m.append_message({"role": "assistant", "content": "answer1"})
        self.m.append_message({"role": "user", "content": "task2"})
        self.m.append_message({"role": "assistant", "content": "answer2"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_push_fork_saves_position(self):
        """push_fork 保存当前 leaf_id."""
        leaf_before = self.m.leaf_id
        self.m.push_fork()
        self.assertEqual(self.m._fork_stack, [leaf_before])

    def test_pop_fork_returns_to_saved(self):
        """pop_fork 回到保存位置."""
        self.m.push_fork()
        self.m.append_message({"role": "user", "content": "new_branch"})
        target = self.m.pop_fork()
        self.assertIsNotNone(target)
        self.assertEqual(self.m.leaf_id, target)

    def test_pop_fork_empty_returns_none(self):
        """空栈 pop_fork 返回 None."""
        self.assertIsNone(self.m.pop_fork())

    def test_history_reflects_fork(self):
        """fork 后 history 反映新路径."""
        self.m.fork(self.a1)
        hist = self.m.history
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[-1]["content"], "answer1")

    def test_navigate_is_fork_alias(self):
        """navigate 是 fork 的直接别名（当前实现）. """
        self.m.navigate(self.a1)
        self.assertEqual(self.m.leaf_id, self.a1)

    def test_fork_nonexistent_raises(self):
        """fork 不存在的 id 抛 ValueError."""
        with self.assertRaises(ValueError):
            self.m.fork("nonexistent")


class TestAgentMemoryCompress(unittest.TestCase):
    """compress_tree 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)
        self.m.append_message({"role": "user", "content": "q1"})
        self.m.append_message({"role": "assistant", "content": "a1"})
        self.q2 = self.m.append_message({"role": "user", "content": "q2"})
        self.m.append_message({"role": "assistant", "content": "a2"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_compress_tree_returns_compaction_entry(self):
        """compress_tree 返回 compaction 类型的 entry id."""
        cid = self.m.compress_tree("摘要", self.q2, 1000)
        entry = self.m.tree.get(cid)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.type, "compaction")
        self.assertEqual(entry.summary, "摘要")

    def test_compress_nonexistent_raises(self):
        """不存在的 first_kept_id 抛 ValueError."""
        with self.assertRaises(ValueError):
            self.m.compress_tree("摘要", "nonexistent", 1000)

    def test_history_after_compress(self):
        """压缩后 history 包含 compaction summary."""
        self.m.compress_tree("对话摘要内容", self.q2, 1000)
        hist = self.m.history
        self.assertIn("[系统]", hist[0]["content"])
        self.assertIn("对话摘要内容", hist[0]["content"])
        self.assertIn("q2", hist[1]["content"])

    def test_jsonl_parent_id_chain_after_compress(self):
        """压缩后 JSONL 中 COMPACT 是 first_kept 的 parent."""
        cid = self.m.compress_tree("摘要", self.q2, 1000)
        with self.m.history_file.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        compaction_rows = [r for r in rows if r["type"] == "compaction"]
        self.assertEqual(len(compaction_rows), 1)
        self.assertEqual(compaction_rows[0]["id"], cid)
        # first_kept 的 parent_id 指向 compaction
        q2_rows = [r for r in rows if r["id"] == self.q2]
        self.assertEqual(q2_rows[0]["parent_id"], cid)


class TestAgentMemoryRestore(unittest.TestCase):
    """restore_tree 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_jsonl(self, session_dir: Path, lines: list[dict]):
        session_dir.mkdir(parents=True)
        with (session_dir / "history.jsonl").open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def test_restore_no_file_returns_false(self):
        """无 history.jsonl 时返回 False."""
        m = AgentMemory(self.memory_dir)
        self.assertFalse(m.restore_tree())

    def test_restore_new_format(self):
        """从新格式 JSONL 恢复树."""
        sid = "a1b2c3d4e5f6"
        uid = "b2c3d4e5f678"
        self._write_jsonl(self.memory_dir, [
            {"id": sid, "parent_id": None, "type": "message", "role": "user",
             "content": "hello", "metadata": {}, "timestamp": 1.0},
            {"id": uid, "parent_id": sid, "type": "message", "role": "assistant",
             "content": "hi", "metadata": {}, "timestamp": 2.0},
        ])
        m = AgentMemory(self.memory_dir)
        self.assertTrue(m.restore_tree())
        self.assertEqual(m.entry_count, 2)
        self.assertEqual(len(m.history), 2)

    def test_restore_new_format_skips_system(self):
        """新格式 JSONL 中的 system 消息被过滤（与 append_message 一致）."""
        self._write_jsonl(self.memory_dir, [
            {"id": "s1", "parent_id": None, "type": "message", "role": "system",
             "content": "sys", "metadata": {}, "timestamp": 1.0},
            {"id": "u1", "parent_id": "s1", "type": "message", "role": "user",
             "content": "hello", "metadata": {}, "timestamp": 2.0},
        ])
        m = AgentMemory(self.memory_dir)
        m.restore_tree()
        self.assertEqual(m.entry_count, 1)

    def test_restore_new_format_with_compaction(self):
        """恢复含 compaction entry 的 JSONL."""
        cid = "comp00000001"
        uid = "user0000001"
        self._write_jsonl(self.memory_dir, [
            {"id": cid, "parent_id": None, "type": "compaction", "role": "system",
             "summary": "压缩摘要", "first_kept_id": uid, "tokens_before": 500,
             "timestamp": 1.0},
            {"id": uid, "parent_id": cid, "type": "message", "role": "user",
             "content": "问题", "metadata": {}, "timestamp": 2.0},
        ])
        m = AgentMemory(self.memory_dir)
        m.restore_tree()
        hist = m.history
        self.assertIn("压缩摘要", hist[0]["content"])
        self.assertIn("问题", hist[1]["content"])

    def test_restore_legacy_format(self):
        """从旧格式 JSONL（无 id/parent_id/type）恢复并迁移."""
        self._write_jsonl(self.memory_dir, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there", "metadata": {}},
        ])
        m = AgentMemory(self.memory_dir)
        self.assertTrue(m.restore_tree())
        self.assertEqual(m.entry_count, 2)

    def test_restore_legacy_skips_system(self):
        """旧格式中的 system 消息被跳过."""
        self._write_jsonl(self.memory_dir, [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ])
        m = AgentMemory(self.memory_dir)
        m.restore_tree()
        self.assertEqual(m.entry_count, 1)

    def test_restore_empty_file_returns_false(self):
        """空 JSONL 文件返回 False."""
        session_dir = self.memory_dir
        session_dir.mkdir(parents=True)
        (session_dir / "history.jsonl").write_text("", encoding="utf-8")
        m = AgentMemory(self.memory_dir)
        self.assertFalse(m.restore_tree())

    def test_restore_jsonl_with_blank_lines(self):
        """JSONL 中空白行被安全跳过."""
        self._write_jsonl(self.memory_dir, [
            {"id": "a", "parent_id": None, "type": "message", "role": "user",
             "content": "hello", "metadata": {}, "timestamp": 1.0},
        ])
        # 手动插入空白行
        with (self.memory_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n\n")
        m = AgentMemory(self.memory_dir)
        self.assertTrue(m.restore_tree())
        self.assertEqual(m.entry_count, 1)

    def test_compress_restore_roundtrip(self):
        """压缩 → 持久化 → 恢复 往返测试."""
        # 1. 创建对话并压缩
        m1 = AgentMemory(self.memory_dir)
        m1.append_message({"role": "user", "content": "q1"})
        m1.append_message({"role": "assistant", "content": "a1"})
        q2 = m1.append_message({"role": "user", "content": "q2"})
        m1.append_message({"role": "assistant", "content": "a2"})
        m1.compress_tree("压缩摘要", q2, 1000)

        # 2. 新建 AgentMemory 并恢复
        m2 = AgentMemory(self.memory_dir)
        self.assertTrue(m2.restore_tree())

        # 3. 验证结构一致
        self.assertEqual(m2.entry_count, m1.entry_count)
        hist1 = m1.history
        hist2 = m2.history
        self.assertEqual(len(hist1), len(hist2))
        self.assertIn("压缩摘要", hist2[0]["content"])


class TestAgentMemoryPreferences(unittest.TestCase):
    """用户偏好测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_user(self):
        """add_user 追加偏好."""
        self.m.add_user("喜欢简洁的代码")
        prefs = self.m.user_preferences()
        self.assertIn("喜欢简洁的代码", prefs)

    def test_add_user_dedup(self):
        """add_user 自动去重."""
        self.m.add_user("喜欢 Python")
        self.m.add_user("喜欢 Python")
        prefs = self.m.user_preferences()
        self.assertEqual(prefs.count("喜欢 Python"), 1)

    def test_add_user_empty_ignored(self):
        """空字符串不写入."""
        count_before = len(self.m.user_preferences())
        self.m.add_user("  ")
        self.assertEqual(len(self.m.user_preferences()), count_before)

    def test_user_preferences_max_items(self):
        """user_preferences(max_items=N) 限制返回数量."""
        for i in range(5):
            self.m.add_user(f"偏好{i}")
        prefs = self.m.user_preferences(max_items=3)
        self.assertEqual(len(prefs), 3)
        self.assertIn("偏好4", prefs)
        self.assertIn("偏好3", prefs)
        self.assertIn("偏好2", prefs)

    def test_user_preferences_empty(self):
        """无偏好时返回空列表."""
        self.assertEqual(self.m.user_preferences(), [])

    def test_get_existing_preferences(self):
        """get_existing_preferences 返回集合."""
        self.m.add_user("偏好A")
        self.m.add_user("偏好B")
        existing = self.m.get_existing_preferences()
        self.assertEqual(existing, {"偏好A", "偏好B"})


class TestAgentMemoryFacts(unittest.TestCase):
    """长期记忆事实测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_memory_returns_true(self):
        """新增事实返回 True."""
        result = self.m.add_memory("项目使用 Python 3.12")
        self.assertTrue(result)

    def test_add_memory_dedup_returns_false(self):
        """重复事实返回 False."""
        self.m.add_memory("重要事实")
        result = self.m.add_memory("重要事实")
        self.assertFalse(result)

    def test_add_memory_empty_returns_false(self):
        """空字符串返回 False."""
        self.assertFalse(self.m.add_memory("  "))

    def test_get_existing_facts(self):
        """get_existing_facts 返回去重集合."""
        self.m.add_memory("事实A")
        self.m.add_memory("事实B")
        facts = self.m.get_existing_facts()
        self.assertEqual(facts, {"事实A", "事实B"})

    def test_get_existing_facts_empty(self):
        """无 memory.md 时返回空集合."""
        m = AgentMemory(self.memory_dir)
        m.memory_file.unlink()
        facts = m.get_existing_facts()
        self.assertEqual(facts, set())


class TestAgentMemorySummaries(unittest.TestCase):
    """摘要测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_summary_creates_file(self):
        """append_summary 创建每日摘要文件."""
        filepath = self.m.append_summary("关键信息", "做了决策", "遇到问题")
        self.assertTrue(filepath.exists())
        content = filepath.read_text(encoding="utf-8")
        self.assertIn("关键信息", content)
        self.assertIn("做了决策", content)
        self.assertIn("遇到问题", content)

    def test_append_summary_twice_appends(self):
        """同一天两次调用 append_summary，追加而非覆盖."""
        self.m.append_summary("第一次关键", "第一次决策", "第一次问题")
        self.m.append_summary("第二次关键", "第二次决策", "第二次问题")
        with self.m._today_file().open() as f:
            content = f.read()
        self.assertIn("第一次关键", content)
        self.assertIn("第二次关键", content)

    def test_brief_context_includes_memory(self):
        """brief_context 包含核心记忆."""
        self.m.add_memory("项目使用 FastAPI")
        context = self.m.brief_context()
        self.assertIn("项目使用 FastAPI", context)

    def test_brief_context_includes_summaries(self):
        """brief_context 包含近期摘要."""
        self.m.append_summary("测试关键", "测试决策", "测试问题")
        context = self.m.brief_context()
        self.assertIn("测试关键", context)

    def test_brief_context_empty(self):
        """无记忆无摘要返回默认文本."""
        self.m.memory_file.unlink()
        context = self.m.brief_context()
        self.assertIn("暂无历史背景", context)


class TestAgentMemoryNonSystem(unittest.TestCase):
    """non_system_entries 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self.tmp.name) / ".memory"
        self.m = AgentMemory(self.memory_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_system_entries_filters_system(self):
        """过滤 system 消息."""
        self.m.append_message({"role": "user", "content": "hello"})
        self.m.append_message({"role": "assistant", "content": "hi"})
        entries = self.m.non_system_entries()
        self.assertEqual(len(entries), 2)
        roles = {e["role"] for e in entries}
        self.assertNotIn("system", roles)

    def test_non_system_entries_with_tool_calls(self):
        """恢复 tool_calls 元数据."""
        self.m.append_message({
            "role": "assistant",
            "content": "calling",
            "tool_calls": [{"function": {"name": "read"}}],
        })
        entries = self.m.non_system_entries()
        self.assertEqual(entries[0]["tool_calls"], [{"function": {"name": "read"}}])

    def test_non_system_entries_with_tool(self):
        """恢复 tool_call_id 元数据."""
        self.m.append_message({
            "role": "tool",
            "content": "output",
            "tool_call_id": "call_abc",
        })
        entries = self.m.non_system_entries()
        self.assertEqual(entries[0]["tool_call_id"], "call_abc")

    def test_get_tree_entries(self):
        """get_tree_entries 返回所有 tree entry."""
        self.m.append_message({"role": "user", "content": "q1"})
        self.m.append_message({"role": "assistant", "content": "a1"})
        entries = self.m.get_tree_entries()
        self.assertEqual(len(entries), 2)


if __name__ == "__main__":
    unittest.main()
