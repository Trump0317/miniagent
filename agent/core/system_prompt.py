"""系统提示词构建器 —— 持有静态上下文引用，实时查询 memory 的动态部分。

支持可选的 compaction_data 注入压缩摘要段落。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import AgentMemory
    from ..tools import SkillsLoader
    from ..tools.subagent import AgentLoader
    from .prompts import PromptLoader


class SystemPrompt:
    """构建完整系统提示词。

    build() 时实时查询 memory 的动态内容（brief_context / user_preferences），
    保证每次调用都反映最新的三层记忆状态。
    """

    def __init__(
        self,
        context_files: str,
        skills: "SkillsLoader",
        agent_loader: "AgentLoader",
        prompt_loader: "PromptLoader",
        memory: "AgentMemory",
    ):
        self._ctx = context_files
        self._skills = skills
        self._agents = agent_loader
        self._commands = prompt_loader
        self._memory = memory

    def build(self, compaction_data: dict | None = None) -> str:
        """构建完整系统提示词。

        compaction_data 不为 None 时追加压缩摘要段落。
        """
        parts = ["你是一个智能助手，可以使用各种工具来帮助用户完成任务。"]

        if self._ctx:
            parts.append(f"### 项目上下文\n{self._ctx}")

        parts.append(f"### 可用技能列表\n{self._skills.get_description()}")
        parts.append(f"### 可用子代理\n{self._agents.list_agents()}")
        parts.append(f"### 可用命令\n{self._commands.list_commands() or '（无）'}")
        parts.append(f"### 长期记忆（最近摘要）\n{self._memory.brief_context()}")
        parts.append(
            f"### 用户偏好（USER.md）\n"
            + ("\n".join(self._memory.user_preferences()) or "（当前没有用户偏好）")
        )

        if compaction_data:
            summary = compaction_data.get("summary", {})
            if any(summary.values()):
                lines = ["### 会话历史摘要（之前对话的压缩记录）"]
                if summary.get("critical"):
                    lines.append(f"- 关键事件: {summary['critical']}")
                if summary.get("decision"):
                    lines.append(f"- 决策/产出: {summary['decision']}")
                if summary.get("issue"):
                    lines.append(f"- 问题: {summary['issue']}")
                parts.append("\n".join(lines))

        return "\n\n".join(parts)
