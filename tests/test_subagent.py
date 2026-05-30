"""Subagent 模块单元测试 — AgentLoader / SubagentTool 配置与调度."""
import unittest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.subagent import (
    TaskItem,
    SubagentArgs,
    AgentDefinition,
    AgentLoader,
    SubagentTool,
    _format_result,
)
from agent.tools.base import Tool, tool
from agent.tools.registry import ToolRegistry


# ═══════════════════════════════════════════════════════════════
# 测试用工具
# ═══════════════════════════════════════════════════════════════

def _make_registry(*names: str) -> ToolRegistry:
    """构建包含指定名称工具的注册表."""
    from pydantic import BaseModel

    class _Args(BaseModel):
        pass

    registry = ToolRegistry()
    for name in names:

        @tool(name=name, description=f"{name} tool", parameters=_Args)
        class _T(Tool):
            def execute(self, **kw) -> str:
                return f"{name}: done"

        registry.register(_T())
    return registry


# ═══════════════════════════════════════════════════════════════
# TaskItem / SubagentArgs
# ═══════════════════════════════════════════════════════════════

class TestTaskItem(unittest.TestCase):
    """TaskItem Pydantic 模型."""

    def test_valid_task(self):
        item = TaskItem(task="analyze code")
        self.assertEqual(item.task, "analyze code")

    def test_missing_task_fails(self):
        """TaskItem 缺少必需字段 task 时抛出 ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            TaskItem()


class TestSubagentArgs(unittest.TestCase):
    """SubagentArgs Pydantic 模型."""

    def test_defaults(self):
        args = SubagentArgs()
        self.assertIsNone(args.agent)
        self.assertIsNone(args.task)
        self.assertIsNone(args.tasks)
        self.assertIsNone(args.chain)
        self.assertEqual(args.max_parallel, 4)

    def test_single_mode(self):
        args = SubagentArgs(agent="reviewer", task="review code")
        self.assertEqual(args.agent, "reviewer")
        self.assertEqual(args.task, "review code")

    def test_parallel_mode(self):
        args = SubagentArgs(tasks=[TaskItem(task="a"), TaskItem(task="b")])
        self.assertEqual(len(args.tasks), 2)
        self.assertEqual(args.tasks[0].task, "a")

    def test_chain_mode(self):
        args = SubagentArgs(chain=[TaskItem(task="step1"), TaskItem(task="step2")])
        self.assertEqual(len(args.chain), 2)

    def test_max_parallel_custom(self):
        args = SubagentArgs(max_parallel=8)
        self.assertEqual(args.max_parallel, 8)


# ═══════════════════════════════════════════════════════════════
# AgentLoader
# ═══════════════════════════════════════════════════════════════

AGENT_MD = """---
name: code-reviewer
description: Review code quality
tools: bash_tool, file_read_tool
model: deepseek-v4-pro
max_turns: 8
---
You are a code reviewer. Be thorough and constructive.
"""

AGENT_MD_MINIMAL = """---
name: minimal
---
Minimal body with no optional fields.
"""

AGENT_MD_NO_NAME = """---
description: Missing name field
---
Body
"""

AGENT_MD_INVALID_MAX_TURNS = """---
name: bad-turns
max_turns: abc
---
Body
"""


class TestAgentLoader(unittest.TestCase):
    """AgentLoader 目录扫描与 MD 解析."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write_agent(self, filename: str, content: str):
        p = os.path.join(self.tmpdir.name, filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

    # ── 加载 ──

    def test_load_single_agent(self):
        self._write_agent("reviewer.md", AGENT_MD)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertIn("code-reviewer", loader.agents)
        agent = loader.agents["code-reviewer"]
        self.assertEqual(agent.name, "code-reviewer")
        self.assertEqual(agent.description, "Review code quality")
        self.assertEqual(agent.tools, ["bash_tool", "file_read_tool"])
        self.assertEqual(agent.model, "deepseek-v4-pro")
        self.assertEqual(agent.max_turns, 8)
        self.assertIn("code reviewer", agent.system_prompt.lower())
        self.assertIn("Be thorough", agent.system_prompt)

    def test_load_minimal_agent(self):
        self._write_agent("minimal.md", AGENT_MD_MINIMAL)
        loader = AgentLoader(Path(self.tmpdir.name))
        agent = loader.agents["minimal"]
        self.assertEqual(agent.name, "minimal")
        self.assertEqual(agent.description, "")
        self.assertEqual(agent.tools, [])
        self.assertEqual(agent.model, "")
        self.assertEqual(agent.max_turns, 10)

    def test_load_minimal_body(self):
        self._write_agent("minimal.md", AGENT_MD_MINIMAL)
        loader = AgentLoader(Path(self.tmpdir.name))
        agent = loader.agents["minimal"]
        self.assertEqual(agent.system_prompt, "Minimal body with no optional fields.")

    def test_skip_file_without_name(self):
        self._write_agent("noname.md", AGENT_MD_NO_NAME)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(len(loader.agents), 0)

    def test_invalid_max_turns_defaults_to_10(self):
        self._write_agent("bad.md", AGENT_MD_INVALID_MAX_TURNS)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.agents["bad-turns"].max_turns, 10)

    def test_load_multiple_agents(self):
        self._write_agent("a.md", AGENT_MD)
        self._write_agent("b.md", AGENT_MD_MINIMAL)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(len(loader.agents), 2)

    def test_load_skips_invalid_md(self):
        self._write_agent("bad.md", "No frontmatter at all, just text.")
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(len(loader.agents), 0)

    def test_load_non_md_files_ignored(self):
        self._write_agent("notes.txt", AGENT_MD)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(len(loader.agents), 0)

    def test_empty_directory(self):
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.agents, {})

    def test_nonexistent_directory(self):
        loader = AgentLoader(Path("/nonexistent/agents"))
        self.assertEqual(loader.agents, {})


    # ── get ──

    def test_get_returns_agent(self):
        self._write_agent("r.md", AGENT_MD)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertIsNotNone(loader.get("code-reviewer"))

    def test_get_returns_none(self):
        self._write_agent("r.md", AGENT_MD)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertIsNone(loader.get("nonexistent"))

    # ── list_agents ──

    def test_list_agents_empty(self):
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertIn("无可用子代理", loader.list_agents())

    def test_list_agents_with_tools(self):
        self._write_agent("r.md", AGENT_MD)
        loader = AgentLoader(Path(self.tmpdir.name))
        listing = loader.list_agents()
        self.assertIn("code-reviewer", listing)
        self.assertIn("Review code quality", listing)
        self.assertIn("bash_tool, file_read_tool", listing)

    def test_list_agents_no_tools_shows_all(self):
        self._write_agent("m.md", AGENT_MD_MINIMAL)
        loader = AgentLoader(Path(self.tmpdir.name))
        listing = loader.list_agents()
        self.assertIn("全部", listing)

    # ── 边界 ──

    def test_tools_with_whitespace(self):
        content = """---
name: ws-tool
tools:  bash ,  read ,  grep  
---
Body
"""
        self._write_agent("ws.md", content)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.agents["ws-tool"].tools, ["bash", "read", "grep"])

    def test_tools_empty_string(self):
        content = """---
name: empty-tools
tools: 
---
Body
"""
        self._write_agent("et.md", content)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.agents["empty-tools"].tools, [])

    def test_source_attribute(self):
        self._write_agent("src.md", AGENT_MD)
        loader = AgentLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.agents["code-reviewer"].source, str(Path(self.tmpdir.name) / "src.md"))


# ═══════════════════════════════════════════════════════════════
# _format_result
# ═══════════════════════════════════════════════════════════════

class TestFormatResult(unittest.TestCase):
    """_format_result 格式化函数."""

    def test_basic_format(self):
        result = _format_result("单", "review code", "All good", {"input": 100, "output": 50})
        self.assertIn("子代理报告", result)
        self.assertIn("(单模式)", result)
        self.assertIn("review code", result)
        self.assertIn("输入 100", result)
        self.assertIn("输出 50", result)
        self.assertIn("All good", result)

    def test_long_task_truncated(self):
        """任务描述超过 100 字符时截断并添加省略号."""
        long_task = "x" * 150
        result = _format_result("并行", long_task, "ok", {"input": 0, "output": 0})
        task_line = [l for l in result.split("\n") if "任务:" in l][0]
        self.assertIn("...", task_line)
        self.assertLess(len(task_line), 110)

    def test_missing_tokens(self):
        result = _format_result("单", "task", "done", {})
        self.assertIn("输入 0", result)
        self.assertIn("输出 0", result)


# ═══════════════════════════════════════════════════════════════
# SubagentTool 配置与调度
# ═══════════════════════════════════════════════════════════════

class TestSubagentToolConfig(unittest.TestCase):
    """SubagentTool _resolve / _filter / execute 模式校验."""

    def setUp(self):
        self.registry = _make_registry("bash_tool", "file_read_tool", "file_write_tool")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.loader = _make_loader(self.tmpdir)
        self.client = MagicMock()
        self.tool = SubagentTool(
            client=self.client,
            model="default-model",
            registry=self.registry,
            agent_loader=self.loader,
            system_prompt="default system prompt",
            max_turns=15,
            sub_model="sub-model",
        )

    # ── _resolve: 默认配置 ──

    def test_resolve_default_no_agent(self):
        cfg = self.tool._resolve(None)
        system_prompt, model, max_turns, registry = cfg
        self.assertEqual(system_prompt, "default system prompt")
        self.assertEqual(model, "sub-model")
        self.assertEqual(max_turns, 15)
        # _registry 是构造函数中 deepcopy 后的对象
        self.assertEqual(registry.tool_names, self.registry.tool_names)

    def test_resolve_unknown_agent_falls_back(self):
        cfg = self.tool._resolve("no-such-agent")
        system_prompt, model, max_turns, registry = cfg
        self.assertEqual(system_prompt, "default system prompt")
        self.assertEqual(model, "sub-model")

    def test_resolve_agent_uses_definition_model(self):
        cfg = self.tool._resolve("code-reviewer")
        _, model, _, _ = cfg
        # Definition has model=deepseek-v4-pro, overrides sub_model
        self.assertEqual(model, "deepseek-v4-pro")

    def test_resolve_agent_uses_definition_max_turns(self):
        cfg = self.tool._resolve("code-reviewer")
        _, _, max_turns, _ = cfg
        self.assertEqual(max_turns, 8)

    def test_resolve_agent_uses_definition_prompt(self):
        cfg = self.tool._resolve("code-reviewer")
        system_prompt, _, _, _ = cfg
        self.assertIn("code reviewer", system_prompt.lower())

    def test_resolve_agent_without_model_falls_back(self):
        cfg = self.tool._resolve("minimal")
        _, model, _, _ = cfg
        self.assertEqual(model, "sub-model")

    def test_resolve_agent_without_max_turns_falls_back(self):
        cfg = self.tool._resolve("minimal")
        _, _, max_turns, _ = cfg
        self.assertEqual(max_turns, 10)  # Definition default

    # ── _resolve: 无 loader ──

    def test_resolve_without_loader_falls_back(self):
        tool_no_loader = SubagentTool(
            client=self.client,
            model="dm",
            registry=self.registry,
            agent_loader=None,
        )
        cfg = tool_no_loader._resolve("any-agent")
        system_prompt, model, _, _ = cfg
        self.assertIn("子代理任务执行者", system_prompt)
        self.assertEqual(model, "dm")

    # ── _filter ──

    def test_filter_empty_tools_returns_all(self):
        definition = AgentDefinition(
            name="full", description="all tools",
            system_prompt="prompt", tools=[],
        )
        result = self.tool._filter(definition)
        self.assertEqual(len(result.tool_names), 3)

    def test_filter_specific_tools(self):
        definition = AgentDefinition(
            name="limited", description="limited tools",
            system_prompt="prompt", tools=["bash_tool", "file_read_tool"],
        )
        result = self.tool._filter(definition)
        self.assertEqual(result.tool_names, ["bash_tool", "file_read_tool"])

    def test_filter_unknown_tool_ignored(self):
        definition = AgentDefinition(
            name="partial", description="partial",
            system_prompt="prompt", tools=["bash_tool", "ghost_tool"],
        )
        result = self.tool._filter(definition)
        self.assertEqual(result.tool_names, ["bash_tool"])

    def test_filter_all_unknown_tools_returns_empty(self):
        definition = AgentDefinition(
            name="none", description="none",
            system_prompt="prompt", tools=["ghost1", "ghost2"],
        )
        result = self.tool._filter(definition)
        self.assertEqual(result.tool_names, [])

    # ── execute: 模式校验 ──

    def test_execute_no_mode_selected(self):
        result = self.tool.execute()
        self.assertIn("请只提供 task、tasks 或 chain 其中之一", result)

    def test_execute_multiple_modes_selected(self):
        result = self.tool.execute(task="a", tasks=[TaskItem(task="b")])
        self.assertIn("请只提供 task、tasks 或 chain 其中之一", result)

    def test_execute_empty_tasks_list(self):
        result = self.tool.execute(tasks=[])
        self.assertIn("请只提供 task、tasks 或 chain 其中之一", result)

    def test_execute_empty_chain_list(self):
        result = self.tool.execute(chain=[])
        self.assertIn("请只提供 task、tasks 或 chain 其中之一", result)

    # ── 并行安全 ──

    def test_parallel_safe_is_true(self):
        self.assertTrue(self.tool.parallel_safe)


def _make_loader(tmpdir: tempfile.TemporaryDirectory) -> AgentLoader:
    """在给定临时目录中创建包含测试 agent 定义的 AgentLoader."""
    reviewer = """---
name: code-reviewer
description: Review code quality
tools: bash_tool, file_read_tool
model: deepseek-v4-pro
max_turns: 8
---
You are a code reviewer. Be thorough.
"""
    minimal = """---
name: minimal
---
Minimal agent.
"""
    with open(os.path.join(tmpdir.name, "reviewer.md"), "w") as f:
        f.write(reviewer)
    with open(os.path.join(tmpdir.name, "minimal.md"), "w") as f:
        f.write(minimal)

    return AgentLoader(Path(tmpdir.name))


if __name__ == "__main__":
    unittest.main()
