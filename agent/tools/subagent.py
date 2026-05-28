from __future__ import annotations
from pydantic import BaseModel, Field
from agent.tools.base import Tool, tool
from agent.tools.registry import ToolRegistry
from typing import Optional, List, TYPE_CHECKING
from types import SimpleNamespace
from openai import OpenAI
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import threading
import tempfile
import atexit
import uuid

if TYPE_CHECKING:
    from agent.core.tracker import TokenTracker


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class TaskItem(BaseModel):
    task: str = Field(description="交给子代理的任务描述。")


class SubagentArgs(BaseModel):
    agent: str | None = Field(default=None, description="子代理名称，不指定则用默认")
    task: str | None = Field(default=None, description="单模式任务")
    tasks: list[TaskItem] | None = Field(default=None, description="并行任务列表")
    chain: list[TaskItem] | None = Field(default=None, description="链式任务列表，{previous} 引用前一步输出")
    max_parallel: int = Field(default=4, description="最大并发数")


@dataclass
class AgentDefinition:
    """一个子代理的完整定义，解析自 agents/*.md 文件。

    文件格式:
        ---
        name: code-reviewer
        description: 审查代码质量和风格
        tools: read, grep, bash
        model: deepseek-v4-pro
        max_turns: 10
        ---
        You are a code reviewer. Focus on...
    """

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = ""
    max_turns: int = 10
    source: str = ""


# ═══════════════════════════════════════════════════════════════
# AgentLoader — 从 Markdown 文件加载子代理定义
# ═══════════════════════════════════════════════════════════════

class AgentLoader:
    """扫描目录，解析所有 *.md 文件为 AgentDefinition。"""

    def __init__(self, directory: Path):
        self._directory = Path(directory)
        self._agents: dict[str, AgentDefinition] = {}
        self._load()

    @property
    def agents(self) -> dict[str, AgentDefinition]:
        return self._agents

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def list_agents(self) -> str:
        if not self._agents:
            return "（无可用子代理定义）"
        lines = ["可用的子代理:"]
        for name, agent in self._agents.items():
            tools_str = ", ".join(agent.tools) if agent.tools else "全部"
            lines.append(f"  - {name}: {agent.description} (工具: {tools_str})")
        return "\n".join(lines)

    def _load(self) -> None:
        if not self._directory.exists() or not self._directory.is_dir():
            return
        for md_file in sorted(self._directory.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            definition = self._parse(content, str(md_file))
            if definition:
                self._agents[definition.name] = definition

    def _parse(self, content: str, source: str) -> AgentDefinition | None:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter = match.group(1)
        body = match.group(2).strip()

        name = self._extract(frontmatter, "name")
        if not name:
            return None

        description = self._extract(frontmatter, "description") or ""
        tools_raw = self._extract(frontmatter, "tools") or ""
        tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        model = self._extract(frontmatter, "model") or ""

        max_turns_str = self._extract(frontmatter, "max_turns") or "10"
        try:
            max_turns = int(max_turns_str)
        except ValueError:
            max_turns = 10

        return AgentDefinition(
            name=name, description=description, system_prompt=body,
            tools=tools, model=model, max_turns=max_turns, source=source,
        )

    @staticmethod
    def _extract(frontmatter: str, key: str) -> str | None:
        m = re.search(rf"^{key}:\s*(.+)$", frontmatter, re.MULTILINE)
        return m.group(1).strip() if m else None


# ═══════════════════════════════════════════════════════════════
# SubagentRunner — 运行一个子代理实例
# ═══════════════════════════════════════════════════════════════

class SubagentRunner:
    """创建并运行一个子代理。

    封装了 LLMClient / ToolExecutor / TokenTracker / AgentRunner 的组装逻辑。
    SubagentTool 只需传入已解析的配置即可。
    """

    def __init__(self, client: OpenAI):
        self._client = client

    def run(
        self,
        *,
        system_prompt: str,
        task: str,
        model: str,
        max_turns: int,
        tool_registry: ToolRegistry,
        parent_tracker: TokenTracker | None = None,
        verbose: bool = True,
        label: str = "子代理",
    ) -> tuple[str, dict]:
        """运行一个子代理。返回 (output_text, {"input": N, "output": N})。

        每次调用创建独立的 LLMClient / ToolExecutor / TokenTracker / AgentRunner，
        确保线程安全。
        """
        from agent.core.runner import AgentRunner
        from agent.core.tracker import TokenTracker
        from agent.ai.llm import LLMClient
        from agent.tools.executor import ToolExecutor

        # ── 独立的 TokenTracker（临时文件）──
        tmpfile = Path(tempfile.gettempdir()) / f"subagent_tokens_{uuid.uuid4().hex}.jsonl"
        atexit.register(lambda: tmpfile.unlink(missing_ok=True))
        sub_tracker = TokenTracker(log_file=tmpfile)

        # ── 独立的 Runner ──
        runner = AgentRunner(
            llm_client=LLMClient(client=self._client, model=model),
            tool_executor=ToolExecutor(registry=deepcopy(tool_registry)),
            token_tracker=sub_tracker,
            max_turns=max_turns,
        )

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        result_parts: list[str] = []
        lock = threading.Lock() if verbose else None

        try:
            if verbose:
                with lock:
                    print(f"\n  ╭─ [{label}] 开始执行 ─", flush=True)

            for chunk in runner.step(history):
                result_parts.append(chunk)
                if verbose and chunk:
                    with lock:
                        if chunk.startswith("\n[执行工具:"):
                            print(f"  │ {chunk.strip()}", flush=True)
                        else:
                            print(chunk, end="", flush=True)

            if verbose:
                with lock:
                    print()
                    print(f"  ╰─ [{label}] 完成 ─", flush=True)

            final_output = "".join(result_parts).strip()

            # 容错：空输出时从 history 的 assistant 消息中取
            if not final_output:
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        final_output = msg.get("content") or msg.get("reasoning_content") or ""
                        if final_output:
                            break

            if not final_output:
                actions = [
                    m["tool_calls"][0]["function"]["name"]
                    for m in history if m.get("tool_calls")
                ]
                if actions:
                    final_output = f"(子代理已执行工具: {', '.join(actions)}，但未给出总结)"
                else:
                    final_output = "(子代理未产出有效回复)"

            # Token 聚合
            stats = sub_tracker.stats_by_model()
            total_input = sum(s["input"] for s in stats.values())
            total_output = sum(s["output"] for s in stats.values())
            tokens = {"input": total_input, "output": total_output}

            # 回写到父 tracker
            if parent_tracker:
                parent_tracker.record(f"subagent:{model}", SimpleNamespace(
                    prompt_tokens=total_input,
                    completion_tokens=total_output,
                    prompt_cache_hit_tokens=0,
                    prompt_cache_miss_tokens=0,
                ))

            return final_output, tokens

        except Exception as e:
            return f"[异常] {e}", {"input": 0, "output": 0}


# ═══════════════════════════════════════════════════════════════
# SubagentTool — 子代理调度器
# ═══════════════════════════════════════════════════════════════

# Agent 配置元组: (system_prompt, model, max_turns, tool_registry)
_AgentConfig = tuple[str, str, int, ToolRegistry]


def _format_result(mode: str, task: str, summary: str, tokens: dict) -> str:
    """格式化子代理执行结果为 LLM 可读的文本报告。"""
    lines = [
        f"═══ 子代理报告 ({mode}模式) ═══",
        f"任务: {task[:100]}{'...' if len(task) > 100 else ''}",
        f"Token: 输入 {tokens.get('input', 0)} / 输出 {tokens.get('output', 0)}",
        "",
        summary,
    ]
    return "\n".join(lines)


@tool(
    name="subagent_tool",
    description=(
        "启动子代理执行任务，支持三种模式："
        "1) 单模式: 传入 task；"
        "2) 并行模式: 传入 tasks 列表；"
        "3) 链式模式: 传入 chain 列表，用 {previous} 引用前一步的输出。"
    ),
    parameters=SubagentArgs,
)
class SubagentTool(Tool):
    """子代理调度工具。

    负责解析 agent 定义、过滤工具注册表、选择执行模式（单/并行/链式）。
    实际的子代理运行委托给 SubagentRunner。
    """

    parallel_safe: bool = True

    def __init__(
        self,
        client: OpenAI,
        model: str,
        registry: ToolRegistry,
        token_tracker: TokenTracker | None = None,
        agent_loader: AgentLoader | None = None,
        system_prompt: str = "你是一个高效的子代理任务执行者。请根据用户的任务要求，利用可用工具完成并给出结论。",
        max_turns: int = 10,
        sub_model: Optional[str] = None,
    ):
        self._runner = SubagentRunner(client)
        self._registry = deepcopy(registry)          # 基础注册表，每次 deepcopy 隔离状态
        self._agent_loader = agent_loader
        self._tracker = token_tracker

        # 默认配置
        self._default_model = sub_model or model
        self._default_prompt = system_prompt
        self._default_max_turns = max_turns

    # ── 公共入口 ──

    def execute(
        self,
        task: Optional[str] = None,
        agent: Optional[str] = None,
        tasks: Optional[List[TaskItem]] = None,
        chain: Optional[List[TaskItem]] = None,
        max_parallel: int = 4,
    ) -> str:
        has_task = task is not None
        has_tasks = tasks is not None and len(tasks) > 0
        has_chain = chain is not None and len(chain) > 0

        if sum([has_task, has_tasks, has_chain]) != 1:
            return "[SubagentTool]: 请只提供 task、tasks 或 chain 其中之一。"

        cfg = self._resolve(agent)

        if has_task:
            return self._run_single(task, cfg)
        elif has_tasks:
            return self._run_parallel(tasks, max_parallel, cfg)
        else:
            return self._run_chain(chain, cfg)

    # ── 配置解析 ──

    def _resolve(self, agent_name: str | None) -> _AgentConfig:
        """将 agent_name 解析为 (system_prompt, model, max_turns, tool_registry)。"""
        if agent_name and self._agent_loader:
            definition = self._agent_loader.get(agent_name)
            if definition:
                return (
                    definition.system_prompt,
                    definition.model or self._default_model,
                    definition.max_turns,
                    self._filter(definition),
                )
        return (
            self._default_prompt,
            self._default_model,
            self._default_max_turns,
            self._registry,
        )

    def _filter(self, definition: AgentDefinition) -> ToolRegistry:
        """根据 AgentDefinition.tools 过滤工具注册表。空列表 = 全部工具。"""
        if not definition.tools:
            return self._registry
        filtered = ToolRegistry()
        for tool_name in definition.tools:
            tool = self._registry.get_tool(tool_name)
            if tool:
                filtered.register(tool)
        return filtered

    # ── 单代理模式 ──

    def _run_single(self, task: str, cfg: _AgentConfig) -> str:
        output, tokens = self._runner.run(
            system_prompt=cfg[0], task=task, model=cfg[1],
            max_turns=cfg[2], tool_registry=cfg[3],
            parent_tracker=self._tracker,
        )
        return _format_result("单", task, output, tokens)

    # ── 并行模式 ──

    def _run_parallel(self, tasks: List[TaskItem], max_parallel: int,
                      cfg: _AgentConfig) -> str:
        start = time.time()
        results: list[tuple[int, str, dict]] = []

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(
                    self._runner.run,
                    system_prompt=cfg[0], task=t.task, model=cfg[1],
                    max_turns=cfg[2], tool_registry=cfg[3],
                    parent_tracker=self._tracker,
                    label=f"子代理{i + 1}",
                ): i
                for i, t in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    summary, tokens = future.result()
                except Exception as e:
                    summary, tokens = f"[异常] {e}", {"input": 0, "output": 0}
                results.append((idx, summary, tokens))

        results.sort(key=lambda x: x[0])

        total_input = sum(t["input"] for _, _, t in results)
        total_output = sum(t["output"] for _, _, t in results)
        elapsed = time.time() - start

        lines = ["═══ 子代理并行报告 ═══"]
        lines.append(f"共 {len(results)} 个任务，耗时 {elapsed:.1f}s")
        lines.append(f"Token 总计: 输入 {total_input} + 输出 {total_output}")
        lines.append("")
        for i, (idx, summary, tokens) in enumerate(results):
            task_text = tasks[idx].task
            lines.append(f"─── 子代理 {idx + 1} ───")
            lines.append(f"任务: {task_text[:80]}{'...' if len(task_text) > 80 else ''}")
            lines.append(f"Token: 输入 {tokens['input']} / 输出 {tokens['output']}")
            lines.append(summary)
            lines.append("")

        return "\n".join(lines)

    # ── 链式模式 ──

    def _run_chain(self, chain: List[TaskItem], cfg: _AgentConfig) -> str:
        previous_output = ""
        all_results: list[dict] = []

        for i, step in enumerate(chain):
            task_with_context = step.task.replace("{previous}", previous_output)
            output, tokens = self._runner.run(
                system_prompt=cfg[0], task=task_with_context, model=cfg[1],
                max_turns=cfg[2], tool_registry=cfg[3],
                parent_tracker=self._tracker,
                label=f"链式-步骤{i + 1}",
            )
            all_results.append({
                "step": i + 1,
                "task": step.task[:60],
                "summary": output,
                "tokens": tokens,
            })
            if output.startswith("[异常]") or output.startswith("[错误]"):
                break
            previous_output = output

        lines = ["═══ 子代理链式报告 ═══"]
        for r in all_results:
            lines.append(f"\n─── 步骤 {r['step']} ───")
            lines.append(f"任务: {r['task']}")
            lines.append(f"Token: 输入 {r['tokens']['input']} / 输出 {r['tokens']['output']}")
            lines.append(r["summary"])
        return "\n".join(lines)
