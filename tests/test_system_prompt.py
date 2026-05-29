"""SystemPrompt 单元测试."""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.system_prompt import SystemPrompt


class TestSystemPrompt(unittest.TestCase):
    """SystemPrompt 构建测试."""

    def setUp(self):
        self.skills = MagicMock()
        self.agents = MagicMock()
        self.commands = MagicMock()
        self.memory = MagicMock()

        self.skills.get_description.return_value = "可用技能: bash, read, write"
        self.agents.list_agents.return_value = "子代理: scout, reviewer"
        self.commands.list_commands.return_value = "可用命令:\n  /review — 审查\n  /scout — 侦查"
        self.memory.brief_context.return_value = "最近摘要: 会议讨论了架构"
        self.memory.user_preferences.return_value = ["偏好1", "偏好2"]

    def _build_prompt(self, context="", compaction=None):
        sp = SystemPrompt(
            context_files=context,
            skills=self.skills,
            agent_loader=self.agents,
            prompt_loader=self.commands,
            memory=self.memory,
        )
        return sp.build(compaction)

    def test_basic_structure(self):
        """基本提示词包含所有必要段落."""
        result = self._build_prompt("test context")
        self.assertIn("智能助手", result)
        self.assertIn("项目上下文", result)
        self.assertIn("test context", result)
        self.assertIn("可用技能列表", result)
        self.assertIn("可用子代理", result)
        self.assertIn("可用命令", result)
        self.assertIn("长期记忆", result)
        self.assertIn("用户偏好", result)
        # 验证各依赖均被调用
        self.skills.get_description.assert_called_once()
        self.agents.list_agents.assert_called_once()
        self.commands.list_commands.assert_called_once()
        self.memory.brief_context.assert_called_once()
        self.memory.user_preferences.assert_called_once_with(max_items=10)

    def test_empty_context_files(self):
        """空 context_files 不包含项目上下文段落."""
        result = self._build_prompt("")
        self.assertNotIn("项目上下文", result)

    def test_skills_section(self):
        """skills 段落正确输出."""
        result = self._build_prompt()
        self.assertIn("可用技能: bash, read, write", result)

    def test_agents_section(self):
        """agents 段落正确输出."""
        result = self._build_prompt()
        self.assertIn("子代理: scout, reviewer", result)

    def test_commands_section(self):
        """commands 段落正确输出."""
        result = self._build_prompt()
        self.assertIn("/review — 审查", result)
        self.assertIn("/scout — 侦查", result)

    def test_commands_empty(self):
        """commands 为空时显示'(无)'."""
        self.commands.list_commands.return_value = ""
        result = self._build_prompt()
        self.assertIn("（无）", result)

    def test_brief_context_section(self):
        """brief_context 段落正确输出."""
        result = self._build_prompt()
        self.assertIn("最近摘要: 会议讨论了架构", result)

    def test_user_preferences_section(self):
        """user_preferences 段落正确输出."""
        result = self._build_prompt()
        self.assertIn("偏好1", result)
        self.assertIn("偏好2", result)

    def test_user_preferences_calls_max_items(self):
        """user_preferences 以 max_items=10 调用."""
        self._build_prompt()
        self.memory.user_preferences.assert_called_once_with(max_items=10)

    def test_user_preferences_empty(self):
        """user_preferences 为空时显示默认文本."""
        self.memory.user_preferences.return_value = []
        result = self._build_prompt()
        self.assertIn("当前没有用户偏好", result)

    def test_compaction_data_with_all_fields(self):
        """compaction_data 包含全部三个字段."""
        result = self._build_prompt(compaction={
            "summary": {
                "critical": "关键事件内容",
                "decision": "决策内容",
                "issue": "问题内容",
            }
        })
        self.assertIn("会话历史摘要", result)
        self.assertIn("关键事件: 关键事件内容", result)
        self.assertIn("决策/产出: 决策内容", result)
        self.assertIn("问题: 问题内容", result)

    def test_compaction_data_partial(self):
        """compaction_data 只包含部分字段."""
        result = self._build_prompt(compaction={
            "summary": {
                "critical": "关键事件",
                "decision": "",
                "issue": "",
            }
        })
        self.assertIn("会话历史摘要", result)
        self.assertIn("关键事件: 关键事件", result)
        self.assertNotIn("决策/产出:", result)
        self.assertNotIn("- 问题:", result)

    def test_compaction_data_empty_summary(self):
        """summary 全部为空字符串时不显示会话历史摘要."""
        result = self._build_prompt(compaction={
            "summary": {"critical": "", "decision": "", "issue": ""}
        })
        self.assertNotIn("会话历史摘要", result)

    def test_compaction_data_none(self):
        """compaction=None 不报错."""
        result = self._build_prompt(compaction=None)
        self.assertNotIn("会话历史摘要", result)

    def test_compaction_data_no_summary_key(self):
        """compaction_data 无 summary key 不报错."""
        result = self._build_prompt(compaction={"other": "data"})
        self.assertNotIn("会话历史摘要", result)

    def test_sections_separated_by_double_newline(self):
        """段落间用双换行分隔."""
        result = self._build_prompt("ctx")
        sections = result.split("\n\n")
        self.assertGreater(len(sections), 4)

    def test_compaction_empty_dict(self):
        """compaction={} 进入 if 块但不追加摘要."""
        result = self._build_prompt(compaction={})
        self.assertNotIn("会话历史摘要", result)


if __name__ == "__main__":
    unittest.main()
