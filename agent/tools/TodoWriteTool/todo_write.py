from enum import Enum
from typing import Type

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.ToolRegisty.base import Tool, tool


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = Field(default=None, description="待办事项 id")
    content: str | None = Field(default=None, min_length=1, description="待办事项内容")
    status: TodoStatus | None = Field(default=None, description="待办事项状态")


class TodoWriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replace_todos: list[TodoItem] | None = Field(default=None, description="用新列表整体替换当前待办事项")
    add_todos: list[TodoItem] | None = Field(default=None, description="新增待办事项")
    update_todos: list[TodoItem] | None = Field(default=None, description="按 id 更新待办事项")
    remove_ids: list[int] | None = Field(default=None, description="按 id 删除待办事项")
    clear: bool = Field(default=False, description="清空全部待办事项；仅当当前所有待办事项都已完成时才会执行")


@tool(
    name="todo_write",
    description="将待办事项写入 todolist",
    parameters=TodoWriteArgs,
)
class TodoWriteTool(Tool):
    _STATUS_ICON = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

    name: str
    description: str
    args_model: Type[TodoWriteArgs]

    def __init__(self):
        """初始化空的待办列表。"""
        self.todos: list[dict] = []

    def render(self, todos: list[dict]) -> str:
        """将待办事项列表渲染为适合展示给模型或用户的文本。"""
        return "\n".join(
            f"{self._STATUS_ICON.get(item['status'], '[ ]')}  {item['content']} (id={item['id']})"
            for item in todos
        )

    def _next_id(self) -> int:
        """生成当前待办列表中尚未使用的下一个 id。"""
        return max((item["id"] for item in self.todos), default=0) + 1

    def _count_in_progress(self, todos: list[dict]) -> int:
        """统计给定待办列表中处于进行中的任务数量。"""
        return sum(1 for item in todos if item["status"] == "in_progress")

    def _build_summary(self, todos: list[dict]) -> str:
        """生成待办列表的状态摘要，便于模型快速判断当前进度。"""
        completed = pending = in_progress = 0
        for item in todos:
            status = item["status"]
            if status == "completed":
                completed += 1
            elif status == "pending":
                pending += 1
            elif status == "in_progress":
                in_progress += 1

        return (
            f"todos updated: total={len(todos)}, completed={completed}, "
            f"in_progress={in_progress}, pending={pending}"
        )

    def _normalize_items(
        self,
        todos: list[dict],
        *,
        require_content: bool,
        require_id: bool = False,
    ) -> list[dict] | str:
        """校验并规范化输入待办项，失败时返回错误字符串。"""
        cleaned: list[dict] = []
        for i, t in enumerate(todos, start=1):
            todo_id = t.get("id")
            if require_id and todo_id is None:
                return "[TodoWriteTool]: Error: update_todos 中每一项都必须提供 id。"

            content = t.get("content")
            if content is None:
                if require_content:
                    return "[TodoWriteTool]: Error: content 不能为空。"
            elif isinstance(content, str):
                content = content.strip()
                if require_content and not content:
                    return "[TodoWriteTool]: Error: content 不能为空。"

            status = t.get("status", "pending")
            if status not in self._STATUS_ICON:
                status = "pending"

            cleaned.append({
                "id": todo_id if todo_id is not None else i,
                "content": content,
                "status": status,
            })

        if self._count_in_progress(cleaned) > 1:
            return "[TodoWriteTool]: Error: 同一时间只能有一个 in_progress 任务，请重新规划。"

        return cleaned

    def _materialize(self, items: list[dict]) -> list[dict]:
        """把规范化后的待办项补全为可直接存储的内部结构。"""
        return [
            {
                "id": item["id"] if item["id"] is not None else self._next_id(),
                "content": item["content"],
                "status": item["status"],
            }
            for item in items
        ]

    def _render_state(self, action_label: str) -> str:
        """输出当前状态并返回给模型可消费的摘要文本。"""
        print(f"[TodoWriteTool]: {action_label}，共 {len(self.todos)} 条。")
        summary = self._build_summary(self.todos)
        print(f"[TodoWriteTool]: 当前列表：\n" + self.render(self.todos))
        return summary + "\n\n当前列表：\n" + self.render(self.todos)

    def _replace(self, todos: list[dict]) -> str:
        """用一组新待办事项整体替换当前列表。"""
        cleaned = self._normalize_items(todos, require_content=True)
        if isinstance(cleaned, str):
            return cleaned

        self.todos = self._materialize(cleaned)
        return self._render_state("已整体替换待办事项")

    def _add(self, todos: list[dict]) -> str:
        """向当前列表追加新的待办事项。"""
        cleaned = self._normalize_items(todos, require_content=True)
        if isinstance(cleaned, str):
            return cleaned

        existing_ids = {item["id"] for item in self.todos}
        new_items = []
        for item in cleaned:
            item_id = item["id"] if item["id"] is not None else self._next_id()
            if item_id in existing_ids:
                return f"[TodoWriteTool]: Error: 待办事项 id {item_id} 已存在。"
            existing_ids.add(item_id)
            new_items.append({
                "id": item_id,
                "content": item["content"],
                "status": item["status"],
            })

        self.todos.extend(new_items)
        return self._render_state("已新增待办事项")

    def _update(self, todos: list[dict]) -> str:
        """按 id 更新已有待办事项的内容或状态。"""
        cleaned = self._normalize_items(todos, require_content=False, require_id=True)
        if isinstance(cleaned, str):
            return cleaned

        by_id = {item["id"]: item for item in self.todos}
        for patch in cleaned:
            todo_id = patch["id"]
            if todo_id not in by_id:
                return f"[TodoWriteTool]: Error: 未找到 id={todo_id} 的待办事项。"

            current = by_id[todo_id]
            if patch.get("content") is not None:
                current["content"] = patch["content"]
            if patch.get("status") is not None:
                current["status"] = patch["status"]

        if self._count_in_progress(self.todos) > 1:
            return "[TodoWriteTool]: Error: 同一时间只能有一个 in_progress 任务，请重新规划。"

        return self._render_state("已更新待办事项")

    def _remove(self, ids: list[int]) -> str:
        """按 id 从当前列表中移除待办事项。"""
        remove_ids = set(ids)
        self.todos = [item for item in self.todos if item["id"] not in remove_ids]
        removed = len(remove_ids)
        return self._render_state(f"已删除待办事项，实际删除 {removed} 条")

    def _clear(self) -> str:
        """在所有事项已完成时清空待办列表，并返回清空前状态。"""
        if not self.todos:
            return "[TodoWriteTool]: 当前没有待办事项，无需清空。"

        incomplete_items = [item for item in self.todos if item["status"] != "completed"]
        if incomplete_items:
            return (
                "[TodoWriteTool]: Error: 还有未完成的待办事项，不能清空。\n"
                + self._build_summary(self.todos)
                + "\n\n请先检查以下未完成事项：\n"
                + self.render(incomplete_items)
                + "\n\n完整列表：\n"
                + self.render(self.todos)
            )

        snapshot = self.render(self.todos)
        summary = self._build_summary(self.todos)
        self.todos = []
        return (
            "[TodoWriteTool]: 已确认所有待办事项都完成，准备清空。\n"
            + summary
            + "\n\n清空前列表：\n"
            + snapshot
            + "\n\n已清空待办事项。"
        )

    def execute(
        self,
        replace_todos: list[dict] | None = None,
        add_todos: list[dict] | None = None,
        update_todos: list[dict] | None = None,
        remove_ids: list[int] | None = None,
        clear: bool = False,
    ) -> str:
        """分发单个待办操作，且一次只允许执行一种动作。"""
        selected_ops = [
            replace_todos is not None,
            add_todos is not None,
            update_todos is not None,
            remove_ids is not None,
            clear,
        ]
        if sum(1 for op in selected_ops if op) != 1:
            return "[TodoWriteTool]: Error: 一次只能执行一种操作，请在 replace_todos / add_todos / update_todos / remove_ids / clear 中选择一个。"

        if replace_todos is not None:
            return self._replace(replace_todos)
        if add_todos is not None:
            return self._add(add_todos)
        if update_todos is not None:
            return self._update(update_todos)
        if remove_ids is not None:
            return self._remove(remove_ids)
        return self._clear()
    