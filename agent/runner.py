from openai import OpenAI
import os

class AgentRunner:
    def __init__(
        self,
        client: OpenAI,
        model: str,
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
            turns += 1
            # 请求模型，设置 stream=True
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=history,
                stream=True
            )

            full_content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    content = delta.content
                    full_content += content
                    yield content

            # 将完整回复加入历史
            history.append({"role": "assistant", "content": full_content})
            return
            
            # 有工具调用，执行工具，并将结果加入对话历史
            # for tool_call in response_message.tool_calls:
            #     function_name = tool_call.function.name
            #     function_to_call = available_functions.get(function_name)

            #     if function_to_call:
            #         function_args = json.loads(tool_call.function.arguments)
            #         #print(f"[调用工具]: {function_name}{function_args}\n")
            #         result = function_to_call(**function_args)
            #         #print(f"[工具输出]: {result}\n")
                    
            #         # 将工具执行结果存入历史
            #         history.append({
            #             "tool_call_id": tool_call.id,
            #             "role": "tool",
            #             "name": function_name,
            #             "content": str(result),
            #         })