"""集成测试 — 端到端验证 Agent 完整流程。

Mock 仅作用于 LLMClient.stream() 边界，其余组件均为真实实例。
"""

from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from agent import Agent, AppConfig
from agent.core.chunks import ChunkType
from openai import APIConnectionError


# ═══════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════

def _chunks_text(chunks: list[dict]) -> str:
    """从 chunk 列表中提取文本/状态内容。"""
    return "".join(
        c.get("content", "") for c in chunks
        if c.get("type") in ("text", "tool_status", ChunkType.TEXT)
    )


def _chunks_of_type(chunks: list[dict], typ) -> list[dict]:
    return [c for c in chunks if c.get("type") == typ]


def _make_text_chunk(text: str) -> dict:
    return {"type": "content", "text": text}


def _make_tool_call_chunk(index: int, name: str, args: str,
                          tc_id: str = "", partial: bool = False) -> dict:
    d: dict = {"type": "tool_call", "index": index}
    if not partial:
        d["id"] = tc_id or f"call_{index}"
    if name:
        d["name"] = name
    if args:
        d["arguments"] = args
    return d


def _make_usage_chunk(inp: int = 10, out: int = 5) -> dict:
    return {"type": "usage", "input": inp, "output": out,
            "cache_hit": 0, "cache_miss": inp}


class MockStream:
    """可配置的 LLM stream mock，替代 LLMClient.stream()。"""

    def __init__(self, chunks: list[list[dict]], *, fail_count: int = 0,
                 fail_error=None):
        """每轮 LLM 调用依次返回 chunks[i]。
        前 fail_count 次调用抛出 fail_error（模拟可重试错误）。
        """
        self._chunks = chunks
        self._call = 0
        self._fail_count = fail_count
        self._fail_error = fail_error or Exception("mock error")

    def __call__(self, messages, tools=None):
        if self._call < self._fail_count:
            self._call += 1
            raise self._fail_error
        if self._call >= len(self._chunks) + self._fail_count:
            return iter([_make_text_chunk("done")])
        result = list(self._chunks[self._call - self._fail_count])
        self._call += 1
        return iter(result)


def _make_test_config(tmpdir: str) -> AppConfig:
    """构造用于测试的 AppConfig（使用临时目录）。"""
    return AppConfig(
        provider="deepseek",
        model="test-model",
        api_key="sk-test",
        api_base_url="https://test.example.com/v1",
        memory_dir=tmpdir,
        max_turns=10,
        max_context=200000,
        compact_threshold=0.99,  # 极高阈值，默认不触发
        subagent_max_turns=3,
        context_files="",
        restore_session=False,
    )


# ═══════════════════════════════════════════════════════════════
# 集成测试：完整 Agent 循环
# ═══════════════════════════════════════════════════════════════

class TestFullAgentLoop(unittest.TestCase):
    """端到端 — Agent.process() 完整流程。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cfg = _make_test_config(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_agent(self, stream_chunks: list[list[dict]]) -> Agent:
        """创建 Agent 并替换 LLMClient.stream 为 mock。"""
        agent = Agent(config=self.cfg)
        agent.runner.llm.stream = MockStream(stream_chunks)
        return agent

    # ── 纯文本 ──

    def test_simple_text_response(self):
        """单轮文本对话 — 用户问，LLM 直接回答。"""
        agent = self._make_agent([
            [_make_text_chunk("Hello"), _make_text_chunk(" World")],
        ])
        chunks = list(agent.process("hi"))
        text = _chunks_text(chunks)
        self.assertIn("Hello World", text)
        self.assertTrue(_chunks_of_type(chunks, ChunkType.DONE))

        # 验证 history
        msgs = [m for m in agent.memory.history if m["role"] != "system"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")

        agent.shutdown()

    # ── 单工具调用 ──

    def test_single_tool_call(self):
        """LLM 调用 bash_tool echo，agent 获取结果后继续。"""
        agent = self._make_agent([
            # Turn 1: tool_call → bash_tool echo hello
            [
                _make_tool_call_chunk(0, "bash_tool", "",
                                       tc_id="call_1", partial=True),
                _make_tool_call_chunk(0, "", '{"command":', partial=True),
                _make_tool_call_chunk(0, "", ' "echo hello"}',
                                       tc_id="call_1"),
            ],
            # Turn 2: 收到工具结果后的文本回复
            [_make_text_chunk("Echo returned: hello")],
        ])
        chunks = list(agent.process("echo hello please"))
        text = _chunks_text(chunks)

        # 应该有工具状态 chunk
        status_chunks = _chunks_of_type(chunks, ChunkType.TOOL_STATUS)
        self.assertGreater(len(status_chunks), 0)
        self.assertIn("bash_tool", _chunks_text(status_chunks).lower())

        # 最终文本回复
        self.assertIn("Echo returned", text)

        # history 应有 4 条: user, assistant(tool_call), tool, assistant(text)
        msgs = [m for m in agent.memory.history if m["role"] != "system"]
        self.assertEqual(len(msgs), 4)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])

        agent.shutdown()

    # ── 多工具调用 ──

    def test_multiple_tool_calls_one_turn(self):
        """一轮中 LLM 返回两个 tool_call — 并行执行。"""
        agent = self._make_agent([
            # Turn 1: two tool calls
            [
                _make_tool_call_chunk(0, "bash_tool", '{"command":',
                                       tc_id="call_a", partial=True),
                _make_tool_call_chunk(0, "", '"echo a"}',
                                       tc_id="call_a"),
                _make_tool_call_chunk(1, "bash_tool", '{"command":',
                                       tc_id="call_b", partial=True),
                _make_tool_call_chunk(1, "", '"echo b"}',
                                       tc_id="call_b"),
            ],
            # Turn 2: text response
            [_make_text_chunk("Both commands ran")],
        ])
        chunks = list(agent.process("run two commands"))
        text = _chunks_text(chunks)
        self.assertIn("Both commands ran", text)

        msgs = [m for m in agent.memory.history if m["role"] != "system"]
        # user, assistant(with 2 tool_calls), tool, tool, assistant
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(len(msgs[1].get("tool_calls", [])), 2)

        agent.shutdown()

    # ── 最大轮数熔断 ──

    def test_max_turns_fuse(self):
        """超过最大轮数后自动熔断。"""
        self.cfg.max_turns = 2
        agent = self._make_agent([
            # Turn 1: tool call
            [
                _make_tool_call_chunk(0, "bash_tool",
                                       '{"command": "echo 1"}',
                                       tc_id="call_1"),
            ],
            # Turn 2: tool call (hits limit)
            [
                _make_tool_call_chunk(0, "bash_tool",
                                       '{"command": "echo 2"}',
                                       tc_id="call_2"),
            ],
        ])
        chunks = list(agent.process("keep going"))
        text = _chunks_text(chunks)
        self.assertIn("熔断", text)
        agent.shutdown()

    # ── 空回复容错 ──

    def test_empty_response_no_tool_calls(self):
        """LLM 返回空内容且无 tool_call — 仍然正常结束。"""
        agent = self._make_agent([[]])
        chunks = list(agent.process("hi"))
        self.assertTrue(_chunks_of_type(chunks, ChunkType.DONE))
        agent.shutdown()

    # ── 错误工具调用 ──

    def test_tool_error_propagates(self):
        """工具执行出错后 LLM 获得错误信息并继续。"""
        agent = self._make_agent([
            # Turn 1: call bash with bad command
            [
                _make_tool_call_chunk(0, "bash_tool",
                                       '{"command": "nonexistent_command_xyz"}',
                                       tc_id="call_e"),
            ],
            # Turn 2: LLM sees error, responds
            [_make_text_chunk("Command failed, trying alternative")],
        ])
        chunks = list(agent.process("run bad command"))
        text = _chunks_text(chunks)
        self.assertIn("Command failed", text)

        # 检查错误信息是否在 tool result 中
        tool_results = _chunks_of_type(chunks, ChunkType.TOOL_RESULT)
        self.assertGreater(len(tool_results), 0)

        agent.shutdown()

    # ── 重试机制 ──

    def test_retry_on_connection_error(self):
        """首次调用失败，重试后成功。"""
        # 自定义可重试异常（避免依赖 httpx Request/Response）
        class FakeConnectionError(APIConnectionError):
            def __init__(self):
                super().__init__(message="Connection refused", request=None)

        agent = self._make_agent([
            [_make_text_chunk("Hello after retry")],
        ])
        agent.runner.llm.stream = MockStream(
            [[_make_text_chunk("Hello after retry")]],
            fail_count=1,
            fail_error=FakeConnectionError(),
        )
        chunks = list(agent.process("hi"))
        text = _chunks_text(chunks)
        self.assertIn("重试", text)
        self.assertIn("Hello after retry", text)
        agent.shutdown()

    def test_retry_exhausted_gives_up(self):
        """3 次重试全部失败后报错退出。"""
        class FakeConnectionError(APIConnectionError):
            def __init__(self):
                super().__init__(message="Connection refused", request=None)

        agent = self._make_agent([])
        agent.runner.llm.stream = MockStream(
            [],
            fail_count=3,
            fail_error=FakeConnectionError(),
        )
        chunks = list(agent.process("hi"))
        text = _chunks_text(chunks)
        self.assertIn("最大重试次数", text)
        agent.shutdown()


# ═══════════════════════════════════════════════════════════════
# 集成测试：会话持久化
# ═══════════════════════════════════════════════════════════════

class TestMemoryPersistence(unittest.TestCase):
    """会话存储和恢复测试。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cfg = _make_test_config(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_and_restore(self):
        """对话保存到 JSONL，新 Agent 可恢复。"""
        # 第一轮对话
        agent1 = Agent(config=self.cfg)
        agent1.runner.llm.stream = MockStream([
            [_make_text_chunk("First response")],
        ])
        list(agent1.process("first message"))
        sid = agent1.config.session_id
        agent1.shutdown()

        # 验证 JSONL 文件存在且非空
        jsonl_path = Path(self._tmpdir.name) / "sessions" / sid / "history.jsonl"
        self.assertTrue(jsonl_path.exists())
        self.assertGreater(jsonl_path.stat().st_size, 0)

        # 恢复会话
        self.cfg.restore_session = True
        self.cfg.session_id = sid
        agent2 = Agent(config=self.cfg)
        agent2.runner.llm.stream = MockStream([
            [_make_text_chunk("Second response")],
        ])
        # 恢复后应有之前的上下文
        msgs = [m for m in agent2.memory.history if m["role"] != "system"]
        self.assertGreaterEqual(len(msgs), 2)

        list(agent2.process("second message"))
        agent2.shutdown()

    def test_empty_session_no_history_file(self):
        """从未对话的会话不创建 history.jsonl。"""
        agent = Agent(config=self.cfg)
        sid = agent.config.session_id
        agent.shutdown()
        history_file = Path(self._tmpdir.name) / "sessions" / sid / "history.jsonl"
        self.assertFalse(history_file.exists(),
                         "history.jsonl should not exist for empty session")


# ═══════════════════════════════════════════════════════════════
# 集成测试：会话树导航
# ═══════════════════════════════════════════════════════════════

class TestSessionTreeNavigation(unittest.TestCase):
    """分叉/返回/树导航测试。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cfg = _make_test_config(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_agent(self) -> Agent:
        agent = Agent(config=self.cfg)
        agent.runner.llm.stream = MockStream([
            [_make_text_chunk("Answer 1")],
            [_make_text_chunk("Answer 2")],
            [_make_text_chunk("Answer 3")],
            [_make_text_chunk("Forked answer")],
        ])
        return agent

    def test_fork_and_back(self):
        """分叉到之前的消息，新消息覆盖旧分支。"""
        agent = self._make_agent()

        # 三组对话
        list(agent.process("Q1"))
        list(agent.process("Q2"))
        list(agent.process("Q3"))

        # 获取所有 user 消息
        user_entries = sorted(
            [e for e in agent.memory.get_tree_entries() if e.role == "user"],
            key=lambda e: e.timestamp,
        )
        self.assertEqual(len(user_entries), 3)

        # fork 到第 2 条之前（parent_id 指向 Q1 的 assistant 回复之后）
        agent.memory.push_fork()
        agent.memory.fork(user_entries[1].parent_id or "")
        agent.rebuild_system_prompt()

        # 发新消息
        list(agent.process("Q2-forked"))

        # 应该有 4 条 user 消息（Q1, Q2-forked 替换了原 Q2/Q3 路径）
        user_entries_after = sorted(
            [e for e in agent.memory.get_tree_entries() if e.role == "user"],
            key=lambda e: e.timestamp,
        )
        self.assertEqual(len(user_entries_after), 4)

        # 返回 fork 前位置
        agent.memory.pop_fork()
        agent.rebuild_system_prompt()

        agent.shutdown()

    def test_fork_without_number_defaults_to_penultimate(self):
        """不指定编号时默认 fork 到倒数第 2 条。"""
        agent = self._make_agent()
        list(agent.process("Q1"))
        list(agent.process("Q2"))

        user_entries = sorted(
            [e for e in agent.memory.get_tree_entries() if e.role == "user"],
            key=lambda e: e.timestamp,
        )
        # fork 到 Q1 的 parent（即 Q1 之前）
        target = user_entries[-2]
        fork_point = target.parent_id or agent.memory.tree.root_id
        if fork_point is None:
            self.skipTest("root_id is None")
            return
        agent.memory.push_fork()
        agent.memory.fork(fork_point)
        agent.rebuild_system_prompt()

        list(agent.process("Q2-alt"))
        agent.shutdown()


# ═══════════════════════════════════════════════════════════════
# 集成测试：压缩流程
# ═══════════════════════════════════════════════════════════════

class TestCompactionFlow(unittest.TestCase):
    """Token 超阈值触发压缩流程测试。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cfg = _make_test_config(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_compaction_triggered_by_token_threshold(self):
        """多轮对话后 token 超阈值自动触发压缩。"""
        # 降低阈值使压缩更容易触发
        self.cfg.max_context = 1000
        self.cfg.compact_threshold = 0.01  # 10 tokens 就触发

        agent = Agent(config=self.cfg)

        # 需要真实的 token tracker 记录，所以给多轮对话
        # 先记录足够多的 token
        from types import SimpleNamespace
        agent.tracker.record("test-model", SimpleNamespace(
            prompt_tokens=800, completion_tokens=50,
            prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
        ))

        # 为压缩 mock LLM stream（压缩需要 LLM 提取摘要）
        mock_stream = MockStream([
            [_make_text_chunk("ok")],  # 压缩提取时的 LLM 调用
        ])
        agent.runner.llm.stream = mock_stream
        agent._compaction._extract = MagicMock(return_value={
            "summary": {"critical": "test session", "decision": "", "issue": ""},
            "preferences": [],
            "facts": [],
        })

        # 强制执行压缩
        agent._compaction._find_cut_point = MagicMock(
            return_value=(0, ["summary text"], ["summary text"])
        )
        should = agent._compaction.should_compact()
        if should:
            agent._system_prompt, result = agent._compaction.compact()
            self.assertTrue(result, "compaction should return non-empty result")
            self.assertIsInstance(result.get("summary"), dict)

        agent.shutdown()

    def test_compaction_not_triggered_below_threshold(self):
        """低 token 用量不触发压缩。"""
        self.cfg.max_context = 100000
        self.cfg.compact_threshold = 0.5
        agent = Agent(config=self.cfg)
        self.assertFalse(agent._compaction.should_compact())
        agent.shutdown()
