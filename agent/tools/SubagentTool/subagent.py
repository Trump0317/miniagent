from __future__ import annotations
from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from agent.tools.ToolRegisty.registry import ToolRegistry
from typing import Type, Optional, List, TYPE_CHECKING
from openai import OpenAI
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import threading

if TYPE_CHECKING:
    from agent.tokentracker import TokenTracker


class TaskItem(BaseModel):
    task: str = Field(description="交给子代理的具体任务。")

class SubagentArgs(BaseModel):
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
    def __init__(self, 
                 client: OpenAI, 
                 model: str, 
                 registry: ToolRegistry, 
                 token_tracker: TokenTracker | None = None,
                 system_prompt: str = "你是一个高效的子代理任务执行者。请根据用户的任务要求，利用可用工具完成并给出结论。",
                 max_turns: int = 10,
                 sub_model: Optional[str] = None):
        self._client = client
        self._model = sub_model or model          # 子代理可用独立模型
        self._registry = deepcopy(registry)        # 深拷贝，状态隔离
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        # token_tracker 只在记录子代理概览 token 时使用
        self._parent_tracker = token_tracker
        # 并行模式下的打印锁，防止输出交错
        self._print_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "subagent_tool"

    @property
    def description(self) -> str:
        return "启动子代理执行任务。支持单模式、并行模式、链式模式。"

    @property
    def args_model(self) -> Type[SubagentArgs]:
        return SubagentArgs

    # ──────────────────────────────────────────
    # 公共入口：根据参数决定走哪个模式
    # ──────────────────────────────────────────
    def execute(self, task: Optional[str] = None,
                tasks: Optional[List[TaskItem]] = None,
                chain: Optional[List[TaskItem]] = None,
                max_parallel: int = 4) -> str:

        has_task = task is not None
        has_tasks = tasks is not None and len(tasks) > 0
        has_chain = chain is not None and len(chain) > 0

        if sum([has_task, has_tasks, has_chain]) != 1:
            return "[SubagentTool]: 请只提供 task、tasks 或 chain 其中之一。"

        if has_task:
            return self._run_single(task)
        elif has_tasks:
            return self._run_parallel(tasks, max_parallel)
        else:
            return self._run_chain(chain)

    # ──────────────────────────────────────────
    # 单代理模式
    # ──────────────────────────────────────────
    def _run_single(self, task: str) -> str:
        summary, sub_tokens = self._run_one(task, label="子代理")
        return self._format_result("单", task, summary, sub_tokens)

    # ──────────────────────────────────────────
    # 并行模式
    # ──────────────────────────────────────────
    def _run_parallel(self, tasks: List[TaskItem], max_parallel: int) -> str:
        start = time.time()
        results: list[tuple[int, str, dict]] = []  # (index, summary, tokens)

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(self._run_one, t.task, f"子代理{i+1}"): i
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
    def _run_chain(self, chain: List[TaskItem]) -> str:
        previous_output = ""
        all_results: list[dict] = []

        for i, step in enumerate(chain):
            task_with_context = step.task.replace("{previous}", previous_output)
            summary, tokens = self._run_one(task_with_context, label=f"链式-步骤{i+1}")
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
    def _run_one(self, task: str, label: str = "子代理") -> tuple[str, dict]:
        """返回 (summary, token_stats_dict)"""
        from agent.runner import AgentRunner
        from agent.tokentracker import TokenTracker

        # 1. 独立的上下文和独立的 token tracker
        history = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task}
        ]
        sub_tracker = TokenTracker(
            log_file=Path(f"/tmp/subagent_tokens_{id(self)}_{hash(task) % 10000}.jsonl")
        )

        # 2. 独立的 Runner 实例
        runner = AgentRunner(
            client=self._client,
            model=self._model,
            tool_registry=self._registry,
            token_tracker=sub_tracker,    # 独立 tracker
            max_turns=self._max_turns
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
            if self._parent_tracker:
                self._parent_tracker.record(f"subagent:{self._model}", type("Usage", (), {
                    "prompt_tokens": total_input,
                    "completion_tokens": total_output,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 0,
                })())

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
