from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Type, Any
from openai import OpenAI
import json

class SubagentArgs(BaseModel):
    task: str = Field(description="交给子代理的具体任务内容。")

@tool(
    name="subagent_tool",
    description="启动一个子代理执行特定任务。子代理拥有独立的思考空间和可选工具集。",
    parameters=SubagentArgs,
)
class SubagentTool(Tool):
    def __init__(self, 
                 client: OpenAI, 
                 model: str, 
                 registry: Any, 
                 token_tracker: Any = None,
                 system_prompt: str = "你是一个高效的子代理任务执行者。请根据用户的任务要求，利用可用工具完成并给出结论。",
                 max_turns: int = 10):
        self._client = client
        self._model = model
        self._registry = registry
        self._token_tracker = token_tracker
        self._system_prompt = system_prompt
        self._max_turns = max_turns

    @property
    def name(self) -> str:
        return "subagent_tool"

    @property
    def description(self) -> str:
        return "启动一个子代理执行特定任务。"

    @property
    def args_model(self) -> Type[SubagentArgs]:
        return SubagentArgs

    def execute(self, task: str) -> str:
        """运行子代理并返回其产出的最终结论"""
        # 延迟导入 AgentRunner 避免可能的循环依赖
        from agent.runner import AgentRunner
        
        # 1. 准备子代理的独立上下文
        history = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task}
        ]
        
        # 2. 实例化子代理 Runner
        runner = AgentRunner(
            client=self._client,
            model=self._model,
            tool_registry=self._registry,
            token_tracker=self._token_tracker,
            max_turns=self._max_turns
        )
        
        result_content = []
        try:
            # 3. 运行子代理并捕获流式输出
            for chunk in runner.step(history):
                result_content.append(chunk)
            
            final_output = "".join(result_content).strip()
            
            # 4. 容错处理：如果流式输出为空，检查历史记录中的最后一条消息
            if not final_output:
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        # 优先取正文内容，其次取思维链（如果有且正文为空）
                        final_output = msg.get("content") or msg.get("reasoning_content") or ""
                        if final_output: break
            
            if not final_output:
                # 依然没有输出，可能仅仅是执行了一堆工具但没总结，或者模型罢工了
                actions = [m.get("tool_calls")[0]["function"]["name"] for m in history if m.get("tool_calls")]
                if actions:
                    return f"[SubagentTool]: 任务执行完毕（或到达轮数限制），但子代理未给出总结性文字。已执行操作: {', '.join(actions)}"
                return "[SubagentTool]: 子代理未产出任何有效回复。"
            
            return f"[Subagent 任务报告]:\n{final_output}"
            
        except Exception as e:
            return f"[SubagentTool]: 运行子代理时发生异常 - {str(e)}"
