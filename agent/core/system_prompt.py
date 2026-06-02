"""系统提示词构建器 —— 从 Markdown 文件加载模板，实时注入动态数据。

支持 {placeholder} 占位符替换和可选 compaction_data。
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import AgentMemory
    from ..tools import SkillsLoader
    from ..tools.subagent import AgentLoader
    from .prompts import PromptLoader


class SystemPrompt:
    """构建完整系统提示词。

    模板文件: agent/core/system_prompt.md
    build() 时实时注入 memory 的动态内容（brief_context / user_preferences），
    保证每次调用都反映最新的三层记忆状态。
    """

    _TEMPLATE_PATH = Path(__file__).parent / "system_prompt.md"

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
        self._template = self._load_template()

    def _load_template(self) -> str:
        try:
            return self._TEMPLATE_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "你是一个智能助手。"

    def build(self, compaction_data: dict | None = None) -> str:
        """构建完整系统提示词。

        compaction_data 不为 None 时追加压缩摘要段落。
        """
        compaction = ""
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
                compaction = "\n".join(lines)

        return self._template.format(
            context_files=f"### 项目上下文\n{self._ctx}" if self._ctx else "",
            skills=f"### 可用技能\n{self._skills.get_description()}",
            agents=f"### 可用子代理\n{self._agents.list_agents()}",
            commands=f"### 可用命令\n{self._commands.list_commands() or '（无）'}",
            memory=f"### 长期记忆（最近摘要）\n{self._memory.brief_context()}",
            preferences=(
                f"### 用户偏好（USER.md，最近 10 条）\n"
                + ("\n".join(self._memory.user_preferences(max_items=10)) or "（当前没有用户偏好）")
            ),
            compaction=compaction,
        )
