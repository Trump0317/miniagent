from openai import OpenAI
import json
from .tools.ToolRegisty.registry import ToolRegistry

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
                            tool_calls_dict[idx] = {"id": tc_chunk.id, "name": tc_chunk.function.name, "arguments": ""}
                        
                        if tc_chunk.function.arguments:
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

            # 如果没有工具调用，说明对话结束，直接退出 yield
            if not tool_calls_dict:
                self._maybe_compact()
                return

            # 3. 执行工具
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                
                yield f"\n[执行工具: {func_name}...]\n"
                
                # 调用重写过的 ToolRegistry
                result = self.tool_registry.call_tool(func_name, args) # type: ignore
                
                # 将结果放入历史，role 必须是 "tool"
                msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                }
                if self.memory:
                    self.memory.append_history(msg)
            
            

            # 4. 继续循环，让模型根据工具结果说话
            continue

    def _maybe_compact(self) -> None:
        if not (self.memory and self.token_tracker):
            return
        if not self.token_tracker.should_compact(self.max_context, self.compact_threshold):
            return
        self.memory.compact()