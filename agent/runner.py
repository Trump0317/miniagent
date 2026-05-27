from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .tools.ToolRegisty.registry import ToolRegistry

# 工具结果截断参数
MAX_RESULT_BYTES = 50 * 1024   # 50KB
MAX_RESULT_LINES = 2000
MAX_PARALLEL_TOOLS = 8         # 并行工具上限


class AgentRunner:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tool_registry: ToolRegistry | None = None,
        memory=None,
        token_tracker=None,
        compact_threshold: float = 0.7,
        max_tokens: int = 20000,
        max_context: int = 200_000,
        max_turns: int | None = None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.tool_registry = tool_registry
        self.memory = memory
        self.token_tracker = token_tracker
        self.max_context = max_context
        self.compact_threshold = compact_threshold
        self.max_turns = max_turns

    @staticmethod
    def _truncate_result(result: str) -> str:
        """截断过长的工具输出，防止撑爆上下文窗口。
        
        双阈值：超过 50KB 或 2000 行，优先保证字节不超限。
        截断时自动追加提示信息，让 LLM 知道输出被截了。
        """
        byte_len = len(result.encode('utf-8'))
        lines = result.split('\n')
        num_lines = len(lines)

        if byte_len <= MAX_RESULT_BYTES and num_lines <= MAX_RESULT_LINES:
            return result

        # 先按行截
        if num_lines > MAX_RESULT_LINES:
            lines = lines[:MAX_RESULT_LINES]
            result = '\n'.join(lines)

        # 再按字节截
        if len(result.encode('utf-8')) > MAX_RESULT_BYTES:
            raw = result.encode('utf-8')
            # 从 MAX_RESULT_BYTES 向前找最近的完整 UTF-8 字符边界
            cut = MAX_RESULT_BYTES
            while cut > 0 and (raw[cut] & 0xC0) == 0x80:
                cut -= 1
            result = raw[:cut].decode('utf-8', errors='replace')

        hint = (f"\n\n[输出已截断: 原始 {byte_len} 字节 / {num_lines} 行, "
                f"超出上限 {MAX_RESULT_BYTES} 字节 / {MAX_RESULT_LINES} 行]")
        return result + hint

    @staticmethod
    def _parse_tool_args(tc: dict) -> dict:
        """安全解析工具参数 JSON"""
        raw = tc.get("function", {}).get("arguments", "{}")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def step(self, history: list):
        """执行单轮对话，支持流式输出"""
        turns = 0
        while True:
            if self.max_turns is not None and turns >= self.max_turns:
                print(f"达到最大对话轮数 {self.max_turns}, 触发熔断")
                break

            tools = self.tool_registry.get_tool_schemas() if self.tool_registry else []

            turns += 1
            # 请求模型，设置 stream=True
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=history, # type: ignore
                tools=tools, # type: ignore
                stream=True,
                stream_options={"include_usage": True}
            )

            full_content = ""
            full_reasoning_content = ""
            tool_calls_dict = {}
            for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage: # 检查 chunk 本身是否有 usage 字段
                    if self.token_tracker:
                        self.token_tracker.record(self.model, chunk.usage)
                    continue
                
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                
                # 处理思维链内容 (如 DeepSeek R1)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    full_reasoning_content += reasoning

                if delta.content:
                    content = delta.content
                    full_content += content
                    yield content
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {"id": None, "name": None, "arguments": ""}
                        
                        if tc_chunk.id:
                            tool_calls_dict[idx]["id"] = tc_chunk.id
                        if tc_chunk.function and tc_chunk.function.name:
                            tool_calls_dict[idx]["name"] = tc_chunk.function.name
                        if tc_chunk.function and tc_chunk.function.arguments:
                            tool_calls_dict[idx]["arguments"] += tc_chunk.function.arguments

            
            # 构造消息对象
            assistant_msg = {"role": "assistant", "content": full_content or None}
            if full_reasoning_content:
                assistant_msg["reasoning_content"] = full_reasoning_content

            if tool_calls_dict:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]}
                    } for tc in tool_calls_dict.values()
                ]
            
            if self.memory:
                self.memory.append_history(assistant_msg)
            else:
                history.append(assistant_msg)

            # 如果没有工具调用，说明对话结束，直接退出 yield
            if not tool_calls_dict:
                self._maybe_compact()
                return

            # 3. 执行工具（单工具串行，多工具并行，结果自动截断）
            tool_call_results: dict[str, str] = {}
            for item in self._execute_tools(assistant_msg["tool_calls"]):
                if isinstance(item, dict):
                    # {"id": ..., "result": ...}
                    tool_call_results[item["id"]] = item["result"]
                else:
                    yield str(item)

            # 4. 将工具结果按原始顺序写入历史
            for tc in assistant_msg["tool_calls"]:
                tc_id = tc["id"]
                if tc_id in tool_call_results:
                    msg = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_call_results[tc_id]
                    }
                    if self.memory:
                        self.memory.append_history(msg)
                    else:
                        history.append(msg)

            # 5. 继续循环，让模型根据工具结果回复
            continue

    def _execute_tools(self, tool_calls: list[dict]):
        """执行工具调用。单工具串行执行，多工具并行执行。
        
        并行策略：
        - 工具之间无依赖（OpenAI 的 parallel_tool_calls 模式），可以并发
        - 用 ThreadPoolExecutor 控制并发数，避免系统过载
        - as_completed 让先完成的先报告，用户不用等最慢的那个
        """
        if len(tool_calls) == 1:
            # 单工具：保持原有流程
            tc = tool_calls[0]
            func_name = tc["function"]["name"]
            args = self._parse_tool_args(tc)
            yield f"\n[执行工具: {func_name}...]\n"
            try:
                result = self.tool_registry.call_tool(func_name, args)
            except Exception as e:
                result = f"[错误] {func_name}: {e}"
            yield {"id": tc["id"], "result": self._truncate_result(result)}
        else:
            # 多工具：并行执行
            count = len(tool_calls)
            workers = min(count, MAX_PARALLEL_TOOLS)
            yield f"\n[并行执行 {count} 个工具 (最多 {workers} 并发)...]\n"

            results: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    args = self._parse_tool_args(tc)
                    future = pool.submit(self.tool_registry.call_tool, func_name, args)
                    futures[future] = (tc["id"], func_name)

                for future in as_completed(futures):
                    tc_id, func_name = futures[future]
                    try:
                        raw_result = future.result()
                    except Exception as e:
                        raw_result = f"[错误] {func_name}: {e}"
                    results[tc_id] = self._truncate_result(raw_result)
                    yield f"[{func_name}] ✓\n"

            for tc in tool_calls:
                tc_id = tc["id"]
                if tc_id in results:
                    yield {"id": tc_id, "result": results[tc_id]}

    def _maybe_compact(self) -> None:
        """如果 token 用量超过阈值，自动触发记忆压缩。"""
        if not (self.memory and self.token_tracker):
            return
        if not self.token_tracker.should_compact(self.max_context, self.compact_threshold):
            return
        print(f"\n[Memory] 上下文用量接近上限, 自动压缩中...", flush=True)
        result = self.memory.compact()
        if result.get("summary") or result.get("facts"):
            print(f"[Memory] 压缩完成", flush=True)