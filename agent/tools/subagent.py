from __future__ import annotations
from pydantic import BaseModel, Field
from agent.tools.base import Tool, tool
from agent.tools.registry import ToolRegistry
from typing import Type, Optional, List, TYPE_CHECKING
from types import SimpleNamespace
from openai import OpenAI
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import threading
import tempfile
import atexit

if TYPE_CHECKING:
    from agent.core.tracker import TokenTracker
    from .subagent_loader import AgentLoader, AgentDefinition


class TaskItem(BaseModel):
    task: str = Field(description="交给子代理的具体任务。")

class SubagentArgs(BaseModel):
    agent: Optional[str] = Field(default=None, description="要使用的子代理名称（对应 agents/*.md 定义文件）。不指定则使用默认子代理。")
    task: Optional[str] = Field(default=None, description="交给子代理的具体任务（单模式）。")
    tasks: Optional[List[TaskItem]] = Field(default=None, description="并行任务列表。")
    chain: Optional[List[TaskItem]] = Field(default=None, description="链式任务列表，支持 {previous} 占位符引用前一步的输出。")
    max_parallel: int = Field(default=4, description="并行模式下的最大并发数。")

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
    parallel_safe: bool = True  # 每个子代理独立上下文，线程安全

    def __init__(self,
                 client: OpenAI,
                 model: str,
                 registry: ToolRegistry,
                 token_tracker: TokenTracker | None = None,
                 agent_loader: AgentLoader | None = None,
                 system_prompt: str = "你是一个高效的子代理任务执行者。请根据用户的任务要求，利用可用工具完成并给出结论。",
                 max_turns: int = 10,
                 sub_model: Optional[str] = None):
        self._client = client
        self._model = sub_model or model          # 子代理默认模型
        self._registry = deepcopy(registry)        # 深拷贝，状态隔离
        self._agent_loader = agent_loader          # Agent 定义加载器
        self._default_system_prompt = system_prompt
        self._default_max_turns = max_turns
        self._token_tracker = token_tracker
        self._print_lock = threading.Lock()

    def _resolve_agent(self, agent_name: str | None) -> tuple[str, str, int, ToolRegistry]:
        """根据 agent_name 解析子代理配置。

        返回 (system_prompt, model, max_turns, tool_registry)。
        如果 agent_name 为 None，使用默认配置。
        """
        if agent_name and self._agent_loader:
            definition = self._agent_loader.get(agent_name)
            if definition:
                return (
                    definition.system_prompt,
                    definition.model or self._model,
                    definition.max_turns,
                    self._filter_registry(definition),
                )
        # 默认配置
        return (
            self._default_system_prompt,
            self._model,
            self._default_max_turns,
            self._registry,
        )

    def _filter_registry(self, definition: AgentDefinition) -> ToolRegistry:
        """根据 AgentDefinition.tools 过滤工具注册表。

        tools 为空 → 返回完整注册表（所有工具）。
        tools 有值 → 只保留指定的工具。
        """
        if not definition.tools:
            return self._registry
        filtered = ToolRegistry()
        for tool_name in definition.tools:
            tool = self._registry.get_tool(tool_name)
            if tool:
                filtered.register(tool)
        return filtered

    # ──────────────────────────────────────────
    # 公共入口：根据参数决定走哪个模式
    # ──────────────────────────────────────────
    def execute(self, task: Optional[str] = None,
                agent: Optional[str] = None,
                tasks: Optional[List[TaskItem]] = None,
                chain: Optional[List[TaskItem]] = None,
                max_parallel: int = 4) -> str:

        has_task = task is not None
        has_tasks = tasks is not None and len(tasks) > 0
        has_chain = chain is not None and len(chain) > 0

        if sum([has_task, has_tasks, has_chain]) != 1:
            return "[SubagentTool]: 请只提供 task、tasks 或 chain 其中之一。"

        cfg = self._resolve_agent(agent)

        if has_task:
            return self._run_single(task, cfg)
        elif has_tasks:
            return self._run_parallel(tasks, max_parallel, cfg)
        else:
            return self._run_chain(chain, cfg)

    # ──────────────────────────────────────────
    # 单代理模式
    # ──────────────────────────────────────────
    def _run_single(self, task: str, cfg: tuple) -> str:
        summary, sub_tokens = self._run_one(task, cfg, label="子代理")
        return self._format_result("单", task, summary, sub_tokens)

    # ──────────────────────────────────────────
    # 并行模式
    # ──────────────────────────────────────────
    def _run_parallel(self, tasks: List[TaskItem], max_parallel: int, cfg: tuple) -> str:
        start = time.time()
        results: list[tuple[int, str, dict]] = []  # (index, summary, tokens)

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(self._run_one, t.task, cfg, f"子代理{i+1}"): i
                for i, t in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    summary, tokens = future.result()
                except Exception as e:
                    summary, tokens = f"[异常] {e}", {"input": 0, "output": 0}
                results.append((idx, summary, tokens))

        # 按原始顺序排列
        results.sort(key=lambda x: x[0])

        total_input = sum(t["input"] for _, _, t in results)
        total_output = sum(t["output"] for _, _, t in results)
        elapsed = time.time() - start

        lines = [f"═══ 子代理并行报告 ═══"]
        lines.append(f"共 {len(results)} 个任务，{len(tasks)} 个完成，耗时 {elapsed:.1f}s")
        lines.append(f"Token 总计: 输入 {total_input} + 输出 {total_output}")
        lines.append("")
        for i, (idx, summary, tokens) in enumerate(results):
            task_text = tasks[idx].task
            lines.append(f"─── 子代理 {idx+1} ───")
            lines.append(f"任务: {task_text[:80]}{'...' if len(task_text) > 80 else ''}")
            lines.append(f"Token: 输入 {tokens['input']} / 输出 {tokens['output']}")
            lines.append(summary)
            lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────────────────
    # 链式模式
    # ──────────────────────────────────────────
    def _run_chain(self, chain: List[TaskItem], cfg: tuple) -> str:
        previous_output = ""
        all_results: list[dict] = []

        for i, step in enumerate(chain):
            task_with_context = step.task.replace("{previous}", previous_output)
            summary, tokens = self._run_one(task_with_context, cfg, label=f"链式-步骤{i+1}")
            all_results.append({
                "step": i + 1,
                "task": step.task[:60],
                "summary": summary,
                "tokens": tokens,
            })
            # 如果子代理返回了明显的错误，链式终止
            if summary.startswith("[异常]") or summary.startswith("[错误]"):
                break
            previous_output = summary

        lines = ["═══ 子代理链式报告 ═══"]
        for r in all_results:
            lines.append(f"\n─── 步骤 {r['step']} ───")
            lines.append(f"任务: {r['task']}")
            lines.append(f"Token: 输入 {r['tokens']['input']} / 输出 {r['tokens']['output']}")
            lines.append(r["summary"])
        return "\n".join(lines)

    # ──────────────────────────────────────────
    # 核心：运行一个子代理
    # ──────────────────────────────────────────
    def _run_one(self, task: str, cfg: tuple, label: str = "子代理") -> tuple[str, dict]:
        """cfg = (system_prompt, model, max_turns, tool_registry)"""
        system_prompt, agent_model, max_turns, tool_registry = cfg
        from agent.runner import AgentRunner
        from agent.core.tracker import TokenTracker

        # 1. 独立的上下文和独立的 token tracker
        import uuid
        tmpfile = Path(tempfile.gettempdir()) / f"subagent_tokens_{uuid.uuid4().hex}.jsonl"
        atexit.register(lambda: tmpfile.unlink(missing_ok=True))

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        sub_tracker = TokenTracker(log_file=tmpfile)

        # 2. 独立的 Runner 实例
        from agent.llm import LLMClient
        from agent.tools.executor import ToolExecutor
        runner = AgentRunner(
            llm_client=LLMClient(
                client=self._client,
                model=agent_model,
            ),
            tool_executor=ToolExecutor(registry=deepcopy(tool_registry)),
            token_tracker=sub_tracker,
            max_turns=max_turns
        )

        result_content = []
        try:
            # 通知用户子代理开始工作
            with self._print_lock:
                print(f"\n  ╭─ [{label}] 开始执行 ─", flush=True)

            for chunk in runner.step(history):
                # 实时输出到终端，方便用户观察进展
                # runner 内部的工具执行提示 (如 [执行工具: bash]) 也会逐块产出
                if chunk:
                    # 为工具执行提示增加缩进
                    if chunk.startswith("\n[执行工具:"):
                        with self._print_lock:
                            print(f"  │ {chunk.strip()}", flush=True)
                    else:
                        # 正文内容直接追加打印（不换行，流式输出）
                        with self._print_lock:
                            print(chunk, end="", flush=True)
                result_content.append(chunk)

            with self._print_lock:
                print()  # 流式输出结束后换行
                print(f"  ╰─ [{label}] 完成 ─", flush=True)

            final_output = "".join(result_content).strip()

            # 容错：空输出时尝试从历史的 assistant 消息里取
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

            token_stats = sub_tracker.stats_by_model()
            total_input = sum(s["input"] for s in token_stats.values())
            total_output = sum(s["output"] for s in token_stats.values())

            # 子代理的 token 用量也记录到父 tracker（方便最终统计）
            if self._token_tracker:
                self._token_tracker.record(f"subagent:{agent_model}", SimpleNamespace(
                    prompt_tokens=total_input,
                    completion_tokens=total_output,
                    prompt_cache_hit_tokens=0,
                    prompt_cache_miss_tokens=0,
                ))

            return final_output, {"input": total_input, "output": total_output}

        except Exception as e:
            return f"[异常] {e}", {"input": 0, "output": 0}

    # ──────────────────────────────────────────
    # 格式化结果
    # ──────────────────────────────────────────
    def _format_result(self, mode: str, task: str, summary: str, tokens: dict) -> str:
        lines = [
            f"═══ 子代理报告 ({mode}模式) ═══",
            f"任务: {task[:100]}{'...' if len(task) > 100 else ''}",
            f"Token: 输入 {tokens.get('input', 0)} / 输出 {tokens.get('output', 0)}",
            f"",
            summary,
        ]
        return "\n".join(lines)
