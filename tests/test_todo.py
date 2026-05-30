"""TodoWriteTool 单元测试."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.todo import TodoWriteTool, TodoStatus


class TestTodoWriteTool(unittest.TestCase):
    """TodoWriteTool 有状态待办管理测试."""

    def setUp(self):
        self.tool = TodoWriteTool()

    # ── 初始状态 ──

    def test_initial_state_empty(self):
        self.assertEqual(self.tool.todos, [])

    def test_initial_render(self):
        self.assertEqual(self.tool.render([]), "")

    # ── replace ──

    def test_replace_creates_list(self):
        result = self.tool.execute(replace_todos=[
            {"content": "task 1", "status": "pending"},
            {"content": "task 2", "status": "completed"},
        ])
        self.assertIn("total=2", result)
        self.assertIn("completed=1", result)
        self.assertEqual(len(self.tool.todos), 2)
        self.assertEqual(self.tool.todos[0]["content"], "task 1")
        self.assertEqual(self.tool.todos[1]["content"], "task 2")

    def test_replace_overwrites_existing(self):
        self.tool.execute(replace_todos=[{"content": "old", "status": "pending"}])
        self.tool.execute(replace_todos=[{"content": "new", "status": "completed"}])
        self.assertEqual(len(self.tool.todos), 1)
        self.assertEqual(self.tool.todos[0]["content"], "new")
        self.assertEqual(self.tool.todos[0]["status"], "completed")

    def test_replace_requires_content(self):
        result = self.tool.execute(replace_todos=[{"id": 1, "status": "pending"}])
        self.assertIn("Error", result)
        self.assertEqual(self.tool.todos, [])

    def test_replace_strips_whitespace_content(self):
        self.tool.execute(replace_todos=[{"content": "  hello  ", "status": "pending"}])
        self.assertEqual(self.tool.todos[0]["content"], "hello")

    def test_replace_empty_content_after_strip(self):
        result = self.tool.execute(replace_todos=[{"content": "   ", "status": "pending"}])
        self.assertIn("Error", result)

    def test_replace_enforces_single_in_progress(self):
        result = self.tool.execute(replace_todos=[
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ])
        self.assertIn("Error", result)
        self.assertIn("in_progress", result)

    def test_replace_default_status_is_pending(self):
        self.tool.execute(replace_todos=[{"content": "task", "status": "unknown"}])
        self.assertEqual(self.tool.todos[0]["status"], "pending")

    # ── add ──

    def test_add_appends_items(self):
        self.tool.execute(replace_todos=[{"content": "existing", "status": "pending", "id": 1}])
        result = self.tool.execute(add_todos=[{"content": "new task", "status": "pending", "id": 2}])
        self.assertIn("total=2", result)
        self.assertEqual(len(self.tool.todos), 2)

    def test_add_generates_auto_ids(self):
        self.tool.execute(add_todos=[{"content": "a"}, {"content": "b"}, {"content": "c"}])
        self.assertEqual(self.tool.todos[0]["id"], 1)
        self.assertEqual(self.tool.todos[1]["id"], 2)
        self.assertEqual(self.tool.todos[2]["id"], 3)

    def test_add_duplicate_id(self):
        self.tool.execute(replace_todos=[{"content": "a", "status": "pending", "id": 5}])
        result = self.tool.execute(add_todos=[{"content": "b", "status": "pending", "id": 5}])
        self.assertIn("Error", result)
        self.assertIn("id 5 已存在", result)

    def test_add_requires_content(self):
        result = self.tool.execute(add_todos=[{"status": "pending"}])
        self.assertIn("Error", result)

    def test_add_without_explicit_ids_when_list_not_empty(self):
        """add_todos 不提供 id 且列表已有内容时的行为.
        
        已知问题: _normalize_items 使用列表位置 i 作为临时 id，
        可能与已有 id 冲突导致误报重复。此测试记录当前实际行为。
        """
        self.tool.execute(replace_todos=[
            {"content": "existing", "status": "pending", "id": 1}
        ])
        result = self.tool.execute(add_todos=[
            {"content": "new task", "status": "pending"}
        ])
        # 当前行为: _normalize_items 设 id=1(位置i=1)，与已有 id=1 冲突
        self.assertIn("Error", result)
        self.assertIn("id 1 已存在", result)
        self.assertEqual(len(self.tool.todos), 1)

    # ── update ──

    def test_update_by_id(self):
        self.tool.execute(replace_todos=[{"content": "task", "status": "pending", "id": 1}])
        result = self.tool.execute(update_todos=[{"id": 1, "status": "completed"}])
        self.assertIn("completed=1", result)
        self.assertEqual(self.tool.todos[0]["status"], "completed")

    def test_update_not_found(self):
        result = self.tool.execute(update_todos=[{"id": 999, "content": "x"}])
        self.assertIn("Error", result)
        self.assertIn("id=999", result)

    def test_update_requires_id(self):
        result = self.tool.execute(update_todos=[{"content": "no id"}])
        self.assertIn("Error", result)
        self.assertIn("id", result)

    def test_update_without_status_defaults_to_pending(self):
        """update_todos 不传 status 时，_normalize_items 默认设为 pending.
        
        注意：这是当前实现行为 —— _normalize_items 总是为每个 item 设置 status，
        默认值为 pending。因此仅更新 content 而不传 status 会导致状态被重置为 pending。
        """
        self.tool.execute(replace_todos=[{"content": "old", "status": "completed", "id": 1}])
        # 不传 status → 被重置为 pending
        self.tool.execute(update_todos=[{"id": 1, "content": "new"}])
        self.assertEqual(self.tool.todos[0]["content"], "new")
        self.assertEqual(self.tool.todos[0]["status"], "pending")

    def test_update_enforces_single_in_progress(self):
        self.tool.execute(replace_todos=[
            {"content": "a", "status": "pending", "id": 1},
            {"content": "b", "status": "completed", "id": 2},
        ])
        # 把两个都改成 in_progress
        result = self.tool.execute(update_todos=[
            {"id": 1, "status": "in_progress"},
            {"id": 2, "status": "in_progress"},
        ])
        self.assertIn("Error", result)
        self.assertIn("in_progress", result)

    # ── remove ──

    def test_remove_by_ids(self):
        self.tool.execute(replace_todos=[
            {"content": "a", "status": "pending", "id": 1},
            {"content": "b", "status": "pending", "id": 2},
            {"content": "c", "status": "pending", "id": 3},
        ])
        result = self.tool.execute(remove_ids=[1, 3])
        self.assertIn("total=1", result)
        self.assertEqual(len(self.tool.todos), 1)
        self.assertEqual(self.tool.todos[0]["id"], 2)

    def test_remove_nonexistent_ids(self):
        self.tool.execute(replace_todos=[{"content": "a", "status": "pending", "id": 1}])
        result = self.tool.execute(remove_ids=[999])
        self.assertIn("total=1", result)
        self.assertEqual(len(self.tool.todos), 1)

    # ── clear ──

    def test_clear_all_completed(self):
        self.tool.execute(replace_todos=[
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
        ])
        result = self.tool.execute(clear=True)
        self.assertIn("已清空", result)
        self.assertEqual(self.tool.todos, [])

    def test_clear_refuses_incomplete(self):
        self.tool.execute(replace_todos=[
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "pending"},
        ])
        result = self.tool.execute(clear=True)
        self.assertIn("Error", result)
        self.assertIn("未完成", result)
        self.assertEqual(len(self.tool.todos), 2)

    def test_clear_empty(self):
        result = self.tool.execute(clear=True)
        self.assertIn("无需清空", result)

    # ── 多操作互斥 ──

    def test_only_one_action_allowed(self):
        result = self.tool.execute(
            replace_todos=[{"content": "a"}],
            add_todos=[{"content": "b"}],
        )
        self.assertIn("Error", result)
        self.assertIn("一次只能执行一种操作", result)

    def test_no_action_selected(self):
        result = self.tool.execute()
        self.assertIn("Error", result)
        self.assertIn("一次只能执行一种操作", result)

    # ── render ──

    def test_render_format(self):
        self.tool.execute(replace_todos=[
            {"content": "task", "status": "in_progress", "id": 1},
        ])
        rendered = self.tool.render(self.tool.todos)
        self.assertIn("[~]", rendered)
        self.assertIn("task", rendered)
        self.assertIn("id=1", rendered)

    # ── 状态图标 ──

    def test_status_icons(self):
        todos = [
            {"content": "p", "status": "pending", "id": 1},
            {"content": "i", "status": "in_progress", "id": 2},
            {"content": "c", "status": "completed", "id": 3},
        ]
        rendered = self.tool.render(todos)
        self.assertIn("[ ]", rendered)
        self.assertIn("[~]", rendered)
        self.assertIn("[x]", rendered)

    # ── 并行安全标志 ──

    def test_parallel_safe_is_false(self):
        self.assertFalse(self.tool.parallel_safe)

    # ── id 自增 ──

    def test_next_id_from_max(self):
        self.tool.execute(replace_todos=[
            {"content": "a", "id": 10},
            {"content": "b", "id": 5},
        ])
        self.tool.execute(add_todos=[{"content": "c", "id": 11}])
        self.assertEqual(self.tool.todos[-1]["id"], 11)

    def test_next_id_default_from_zero(self):
        self.tool.execute(add_todos=[{"content": "first"}])
        self.assertEqual(self.tool.todos[0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
