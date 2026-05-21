from openai import OpenAI
import json
from .tools.ToolRegisty.registry import ToolRegistry

class AgentRunner:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tool_registry: ToolRegistry | None = None,
        memory_store=None,
        token_tracker=None,
        compactor=None,
        compact_threshold: float = 0.7,
        max_tokens: int = 20000,
        max_context: int = 200_000,
        max_turns: int | None = None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.token_tracker = token_tracker
        self.compactor = compactor
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
                stream=True
            )

            full_content = ""
            full_reasoning_content = ""
            tool_calls_dict = {}
            for chunk in response:
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
            
            history.append(assistant_msg)

            # 如果没有工具调用，说明对话结束，直接退出 yield
            if not tool_calls_dict:
                return

            # 3. 执行工具
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                
                yield f"\n[执行工具: {func_name}...]\n"
                
                # 调用你重写过的 ToolRegistry
                result = self.tool_registry.call_tool(func_name, args) # type: ignore
                
                # 将结果放入历史，role 必须是 "tool"
                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })
            
            # 4. 继续循环，让模型根据工具结果说话
            continue
