"""SessionTree 单元测试."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.session_tree import SessionTree, SessionEntry


class TestSessionEntry(unittest.TestCase):
    """SessionEntry 数据类测试."""

    def test_default_timestamp(self):
        """未指定 timestamp 时自动设为当前时间."""
        entry = SessionEntry(id="a", parent_id=None, type="message", role="user", content="hi")
        self.assertGreater(entry.timestamp, 0)

    def test_explicit_timestamp(self):
        """指定 timestamp 时保留原值."""
        entry = SessionEntry(
            id="a", parent_id=None, type="message", role="user",
            content="hi", timestamp=100.0
        )
        self.assertEqual(entry.timestamp, 100.0)

    def test_new_id_uniqueness(self):
        """new_id() 生成不同 id."""
        ids = {SessionEntry.new_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_new_id_length(self):
        """new_id() 生成 12 位 hex 字符串."""
        eid = SessionEntry.new_id()
        self.assertEqual(len(eid), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in eid))

    def test_compaction_fields_default(self):
        """compaction 专用字段默认值."""
        entry = SessionEntry(id="a", parent_id=None, type="message", role="user")
        self.assertIsNone(entry.summary)
        self.assertIsNone(entry.first_kept_id)
        self.assertEqual(entry.tokens_before, 0)


class TestSessionTreeEmpty(unittest.TestCase):
    """空树状态测试."""

    def setUp(self):
        self.tree = SessionTree()

    def test_empty_leaf_id(self):
        self.assertIsNone(self.tree.leaf_id)

    def test_empty_root_id(self):
        self.assertIsNone(self.tree.root_id)

    def test_empty_entry_count(self):
        self.assertEqual(self.tree.entry_count, 0)

    def test_empty_entries(self):
        self.assertEqual(self.tree.entries(), {})

    def test_empty_path(self):
        self.assertEqual(self.tree.path(), [])

    def test_empty_build_context(self):
        self.assertEqual(self.tree.build_context(), [])

    def test_empty_dump_tree(self):
        self.assertIn("空树", self.tree.dump_tree())

    def test_empty_get(self):
        self.assertIsNone(self.tree.get("any"))


class TestSessionTreeAppend(unittest.TestCase):
    """append 操作测试."""

    def setUp(self):
        self.tree = SessionTree()

    def test_first_append_becomes_root(self):
        """第一个 append 自动成为 root."""
        eid = self.tree.append("system", "你是助手")
        self.assertEqual(self.tree.root_id, eid)
        self.assertEqual(self.tree.leaf_id, eid)
        self.assertEqual(self.tree.entry_count, 1)

    def test_append_chain(self):
        """连续 append 形成 chain."""
        r = self.tree.append("system", "system msg")
        u = self.tree.append("user", "hello")
        a = self.tree.append("assistant", "hi")

        path = self.tree.path()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0].id, r)
        self.assertEqual(path[1].id, u)
        self.assertEqual(path[2].id, a)

        # 验证 parent_id 链
        entry_u = self.tree.get(u)
        entry_a = self.tree.get(a)
        self.assertEqual(entry_u.parent_id, r)
        self.assertEqual(entry_a.parent_id, u)

    def test_append_content_none(self):
        """content=None 可以正常追加."""
        eid = self.tree.append("user", None)
        entry = self.tree.get(eid)
        self.assertIsNone(entry.content)

    def test_append_with_metadata(self):
        """append 支持 metadata."""
        eid = self.tree.append("assistant", "hi", metadata={"tool_calls": [{}]})
        entry = self.tree.get(eid)
        self.assertEqual(entry.metadata["tool_calls"], [{}])

    def test_append_default_metadata(self):
        """不传 metadata 时为 {}."""
        eid = self.tree.append("user", "hello")
        entry = self.tree.get(eid)
        self.assertEqual(entry.metadata, {})

    def test_append_returns_id(self):
        """append 返回新 entry 的 id."""
        eid = self.tree.append("user", "hello")
        self.assertIsInstance(eid, str)
        self.assertEqual(len(eid), 12)

    def test_append_entry_type_is_message(self):
        """append 创建的 entry type 为 'message'."""
        eid = self.tree.append("user", "hello")
        self.assertEqual(self.tree.get(eid).type, "message")


class TestSessionTreePath(unittest.TestCase):
    """path() 操作测试."""

    def setUp(self):
        self.tree = SessionTree()
        self.r = self.tree.append("system", "s")
        self.u1 = self.tree.append("user", "q1")
        self.a1 = self.tree.append("assistant", "a1")
        self.u2 = self.tree.append("user", "q2")
        self.a2 = self.tree.append("assistant", "a2")

    def test_path_to_current_leaf(self):
        """路径从 root 到当前 leaf."""
        p = self.tree.path()
        self.assertEqual([e.id for e in p], [self.r, self.u1, self.a1, self.u2, self.a2])

    def test_path_to_specific_id(self):
        """path(to_id) 到指定节点."""
        p = self.tree.path(to_id=self.a1)
        self.assertEqual([e.id for e in p], [self.r, self.u1, self.a1])

    def test_path_order_is_root_to_target(self):
        """path 返回顺序是 root → target（正向时间序）."""
        p = self.tree.path()
        for i in range(len(p) - 1):
            self.assertEqual(p[i + 1].parent_id, p[i].id)

    def test_path_to_nonexistent_returns_empty(self):
        """不存在的 to_id 返回空列表."""
        p = self.tree.path(to_id="nonexistent")
        self.assertEqual(p, [])


class TestSessionTreeFork(unittest.TestCase):
    """fork/navigate 操作测试."""

    def setUp(self):
        self.tree = SessionTree()
        self.r = self.tree.append("system", "s")
        self.u1 = self.tree.append("user", "task1")
        self.a1 = self.tree.append("assistant", "answer1")

    def test_fork_updates_leaf(self):
        """fork 后 leaf 指向目标."""
        self.tree.fork(self.u1)
        self.assertEqual(self.tree.leaf_id, self.u1)

    def test_fork_nonexistent_raises(self):
        """fork 不存在的 id 抛 ValueError."""
        with self.assertRaises(ValueError):
            self.tree.fork("nonexistent")

    def test_fork_then_append_creates_branch(self):
        """fork 后 append 创建新分支."""
        self.tree.fork(self.u1)
        a2 = self.tree.append("assistant", "answer2")

        # 新路径: root → u1 → a2（不包含旧 a1）
        p = self.tree.path()
        self.assertEqual([e.id for e in p], [self.r, self.u1, a2])

        # a1 仍然存在但不在当前路径上
        self.assertIsNotNone(self.tree.get(self.a1))

    def test_children_of_shows_all_branches(self):
        """children_of 返回所有直接子节点（含多分支）."""
        self.tree.fork(self.u1)
        self.tree.append("assistant", "answer2")

        children = self.tree.children_of(self.u1)
        self.assertEqual(len(children), 2)
        child_ids = {c.id for c in children}
        self.assertIn(self.a1, child_ids)

    def test_children_of_sorted_by_timestamp(self):
        """children_of 按 timestamp 排序."""
        import time
        self.tree.fork(self.u1)
        time.sleep(0.01)
        a2 = self.tree.append("assistant", "answer2")

        children = self.tree.children_of(self.u1)
        self.assertEqual(children[0].id, self.a1)   # 先创建
        self.assertEqual(children[1].id, a2)          # 后创建

    def test_navigate_same_as_fork(self):
        """navigate 效果与 fork 相同."""
        self.tree.navigate(self.u1)
        self.assertEqual(self.tree.leaf_id, self.u1)
        # navigate 到 u1 后 path 只到 u1，不包含之前的 a1
        path = self.tree.path()
        self.assertEqual(len(path), 2)  # system, u1

    def test_navigate_to_node_with_children(self):
        """navigate 到已有子节点的条目，后续 append 从该处分支."""
        # 先追加子节点
        self.tree.append("user", "q2")
        self.tree.append("assistant", "a2")
        # 再 navigate 回 a1
        self.tree.navigate(self.a1)
        self.assertEqual(self.tree.leaf_id, self.a1)
        path = self.tree.path()
        # 只到 a1，不包含 a1 之后的 q2/a2
        self.assertEqual(len(path), 3)

    def test_siblings(self):
        """返回兄弟节点（不含自己）."""
        self.tree.fork(self.u1)
        a2 = self.tree.append("assistant", "answer2")

        # a1 的兄弟应该是 a2
        sibs = self.tree.siblings(self.a1)
        self.assertEqual(len(sibs), 1)
        self.assertEqual(sibs[0].id, a2)

        # a2 的兄弟应该是 a1
        sibs2 = self.tree.siblings(a2)
        self.assertEqual(len(sibs2), 1)
        self.assertEqual(sibs2[0].id, self.a1)

    def test_siblings_root_none(self):
        """root 没有父节点，siblings 返回空."""
        sibs = self.tree.siblings(self.r)
        self.assertEqual(sibs, [])


class TestSessionTreeCompact(unittest.TestCase):
    """compact 操作测试."""

    def setUp(self):
        self.tree = SessionTree()
        self.tree.append("system", "s")
        self.tree.append("user", "q1")
        self.tree.append("assistant", "a1")
        self.q2 = self.tree.append("user", "q2")
        self.a2 = self.tree.append("assistant", "a2")
        self.tree.append("user", "q3")
        self.tree.append("assistant", "a3", metadata={"tool_calls": [{"function": {"name": "bash"}}]})
        self.tool = self.tree.append("tool", "stdout: ok", metadata={"tool_call_id": "call_1"})

    def test_compact_nonexistent_raises(self):
        """compact 不存在的 first_kept_id 抛异常."""
        with self.assertRaises(ValueError):
            self.tree.compact("摘要", "nonexistent", 1000)

    def test_compact_creates_compaction_entry(self):
        """compact 创建 compaction 类型 entry."""
        cid = self.tree.compact("摘要", self.q2, 1000)
        entry = self.tree.get(cid)
        self.assertEqual(entry.type, "compaction")
        self.assertEqual(entry.role, "system")
        self.assertEqual(entry.summary, "摘要")
        self.assertEqual(entry.first_kept_id, self.q2)
        self.assertEqual(entry.tokens_before, 1000)

    def test_compact_restructures_tree(self):
        """compact 重排 parent_id 链."""
        cid = self.tree.compact("摘要", self.q2, 1000)

        compact_entry = self.tree.get(cid)
        q2_entry = self.tree.get(self.q2)

        # COMPACT 的 parent = q2 原来的 parent
        # q2 原来的 parent 是 a1 的 entry
        # q2 的新 parent = COMPACT
        self.assertEqual(q2_entry.parent_id, cid)
        # COMPACT 在 root → leaf 路径上
        path_ids = [e.id for e in self.tree.path()]
        self.assertIn(cid, path_ids)
        self.assertIn(self.q2, path_ids)
        # COMPACT 在 q2 之前
        compact_idx = path_ids.index(cid)
        q2_idx = path_ids.index(self.q2)
        self.assertLess(compact_idx, q2_idx)

    def test_build_context_with_compaction(self):
        """build_context 跳过被压缩的消息."""
        self.tree.compact("对话摘要: 用户问了3个问题", self.q2, 5000)

        ctx = self.tree.build_context()
        # summary + q2 + a2 + q3 + a3 + tool = 6 条
        # 实际上要看 first_kept 是 q2，所以 q2 之前的（system, q1, a1）被 summary 替代
        self.assertEqual(len(ctx), 6)

        # 第一条是 compaction summary
        self.assertIn("[系统]", ctx[0]["content"])
        self.assertIn("对话摘要", ctx[0]["content"])

        # 第二条是 q2
        self.assertIn("q2", ctx[1]["content"])

        # tool 的 metadata 恢复
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_1")

    def test_build_context_assistant_tool_calls(self):
        """assistant 的 tool_calls metadata 被恢复."""
        ctx = self.tree.build_context()
        # 没有 compaction，直接看 assistant
        a3_msg = [m for m in ctx if m.get("tool_calls")]
        self.assertEqual(len(a3_msg), 1)
        self.assertEqual(a3_msg[0]["tool_calls"], [{"function": {"name": "bash"}}])

    def test_multiple_compactions(self):
        """多次 compact，后续 summary 替换前一个."""
        # 第一次压缩
        self.tree.compact("压缩1", self.q2, 1000)

        # 追加更多
        self.tree.append("user", "q4")
        self.tree.append("assistant", "a4")

        # 第二次压缩
        q4 = [e for e in self.tree.path() if e.role == "user" and e.content == "q4"][0]
        self.tree.compact("压缩2", q4.id, 2000)

        ctx = self.tree.build_context()
        # 只有 summary2 + q4 + a4
        self.assertEqual(len(ctx), 3)
        self.assertIn("压缩2", ctx[0]["content"])
        self.assertIn("q4", ctx[1]["content"])

    def test_compact_preserves_leaf(self):
        """compact 不改变 leaf."""
        before = self.tree.leaf_id
        self.tree.compact("摘要", self.q2, 1000)
        self.assertEqual(self.tree.leaf_id, before)

    def test_append_after_compact_context(self):
        """压缩后追加消息，build_context 包含摘要+保留消息+新消息."""
        self.tree.compact("摘要", self.q2, 1000)
        self.tree.append("user", "q4")
        self.tree.append("assistant", "a4")
        ctx = self.tree.build_context()
        # summary + q2 + a2 + q3 + a3 + tool + q4 + a4 = 8
        self.assertEqual(len(ctx), 8)
        self.assertIn("摘要", ctx[0]["content"])
        self.assertIn("q2", ctx[1]["content"])
        self.assertIn("q4", ctx[-2]["content"])

    def test_compact_first_kept_is_leaf(self):
        """first_kept 也可以是当前 leaf."""
        leaf_id = self.tree.leaf_id
        cid = self.tree.compact("摘要", leaf_id, 1000)
        entry = self.tree.get(cid)
        self.assertEqual(entry.first_kept_id, leaf_id)

        ctx = self.tree.build_context()
        # summary + leaf entry = 2 条
        self.assertEqual(len(ctx), 2)  # summary + leaf entry (tool)
        self.assertIn("摘要", ctx[0]["content"])


class TestSessionTreeBuildContext(unittest.TestCase):
    """build_context 消息构建测试."""

    def setUp(self):
        self.tree = SessionTree()

    def test_normal_messages(self):
        """普通消息正确转换为 LLM 格式."""
        self.tree.append("system", "sys")
        self.tree.append("user", "hello")
        self.tree.append("assistant", "hi there")

        ctx = self.tree.build_context()
        self.assertEqual(ctx[0], {"role": "system", "content": "sys"})
        self.assertEqual(ctx[1], {"role": "user", "content": "hello"})
        self.assertEqual(ctx[2], {"role": "assistant", "content": "hi there"})

    def test_content_none_omitted(self):
        """content=None 时不包含 content 键."""
        self.tree.append("user", None)
        ctx = self.tree.build_context()
        self.assertEqual(ctx[0], {"role": "user"})

    def test_assistant_with_reasoning(self):
        """assistant 的 reasoning_content metadata 被恢复."""
        self.tree.append("assistant", "answer", metadata={"reasoning_content": "thinking..."})
        ctx = self.tree.build_context()
        self.assertEqual(ctx[0]["reasoning_content"], "thinking...")

    def test_tool_with_tool_call_id(self):
        """tool 的 tool_call_id metadata 被恢复."""
        self.tree.append("tool", "result", metadata={"tool_call_id": "call_abc"})
        ctx = self.tree.build_context()
        self.assertEqual(ctx[0]["tool_call_id"], "call_abc")

    def test_non_message_type_skipped(self):
        """非 message 类型（非 compaction）被跳过."""
        entry = SessionEntry(
            id="skip", parent_id=self.tree.leaf_id, type="custom",
            role="user", content="skipped"
        )
        self.tree.add_entry(entry)
        self.tree.set_leaf("skip")
        ctx = self.tree.build_context()
        self.assertEqual(ctx, [])

    def test_build_context_with_to_id(self):
        """build_context(to_id) 限制到指定节点."""
        self.tree.append("system", "sys")
        u = self.tree.append("user", "q")
        self.tree.append("assistant", "a")

        ctx = self.tree.build_context(to_id=u)
        self.assertEqual(len(ctx), 2)
        self.assertEqual(ctx[1]["content"], "q")


class TestSessionTreeHelpers(unittest.TestCase):
    """辅助方法测试."""

    def setUp(self):
        self.tree = SessionTree()
        self.tree.append("system", "sys")
        self.tree.append("user", "q1")
        self.tree.append("assistant", "a1")

    def test_non_system_entries_filters_system(self):
        """过滤掉 system role."""
        non_sys = self.tree.non_system_entries()
        roles = {e.role for e in non_sys}
        self.assertNotIn("system", roles)
        self.assertEqual(len(non_sys), 2)

    def test_non_system_entries_filters_compaction_type(self):
        """过滤掉 compaction 类型的 entry."""
        self.tree.append("user", "q2")
        cid = self.tree.compact("summary", self.tree.leaf_id, 100)
        non_sys = self.tree.non_system_entries()
        # compaction entry 不应该出现
        self.assertNotIn(cid, [e.id for e in non_sys])

    def test_to_messages_format(self):
        """to_messages 返回 LLM 兼容消息列表."""
        msgs = self.tree.to_messages()
        self.assertEqual(msgs[0], {"role": "user", "content": "q1"})
        self.assertEqual(msgs[1], {"role": "assistant", "content": "a1"})

    def test_to_messages_with_tool_calls(self):
        """to_messages 恢复 tool_calls metadata."""
        self.tree.append("assistant", "call", metadata={"tool_calls": [{"function": {"name": "bash"}}]})
        msgs = self.tree.to_messages()
        last = msgs[-1]
        self.assertIn("tool_calls", last)
        self.assertEqual(last["tool_calls"], [{"function": {"name": "bash"}}])

    def test_dump_tree_not_empty(self):
        """dump_tree 非空树输出包含节点信息."""
        output = self.tree.dump_tree()
        self.assertIn("sys", output)
        self.assertIn("leaf", output)  # 标记当前 leaf

    def test_dump_tree_empty(self):
        """空树 dump_tree 返回提示."""
        t = SessionTree()
        self.assertIn("空树", t.dump_tree())


class TestSessionTreeEdges(unittest.TestCase):
    """边界情况测试."""

    def test_path_circular_prevention(self):
        """path 防止循环引用."""
        tree = SessionTree()
        eid = tree.append("user", "hi")
        # 手动制造循环
        tree.get(eid).parent_id = eid
        path = tree.path()
        # 不应该无限循环
        self.assertEqual(len(path), 1)

    def test_children_of_nonexistent(self):
        """不存在的节点 children_of 返回空."""
        tree = SessionTree()
        self.assertEqual(tree.children_of("nonexistent"), [])

    def test_entries_returns_copy(self):
        """entries() 返回副本，修改不影响内部."""
        tree = SessionTree()
        tree.append("user", "hi")
        d = tree.entries()
        d.clear()
        self.assertEqual(tree.entry_count, 1)

    def test_build_context_empty_content_preserved(self):
        """content 为空字符串时仍然保留."""
        tree = SessionTree()
        tree.append("user", "")
        ctx = tree.build_context()
        self.assertEqual(ctx[0]["content"], "")


if __name__ == "__main__":
    unittest.main()
