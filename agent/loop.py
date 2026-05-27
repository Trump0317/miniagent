from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from .runner import AgentRunner
from .memory import AgentMemory
from .tokentracker import TokenTracker
from .tools import (
    ToolRegistry, BashTool, FileReadTool, FileWriteTool, FileEditTool, WebFetchTool, WebSearchTool,
    SkillTool, SkillsLoader, TodoWriteTool, SubagentTool
)

class AgentLoop:
    def __init__(self, root: Path | None = None, model: str = "deepseek-v4-flash"):
        # 加载 .env 文件
        load_dotenv()
        # 设置项目的根目录
        root = root or Path(__file__).parent.parent
        # 初始化模型客户端
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_BASE_URL")
        )
        # 预加载技能摘要
        skills_loader = SkillsLoader(skill_directory=root / "skills")
        self.memory = AgentMemory(memory_dir=root /"agent"/ ".memory", client=client, model=model)
        self.token_tracker = TokenTracker(log_file=root / "agent" / ".memory" / "tokens.jsonl")
        # 注册工具
        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(FileEditTool())
        registry.register(WebFetchTool())
        registry.register(WebSearchTool())
        registry.register(SkillTool(skills_loader))
        registry.register(TodoWriteTool())

        subregistry = ToolRegistry()
        subregistry.register(BashTool())
        subregistry.register(FileReadTool())
        subregistry.register(FileWriteTool())
        subregistry.register(FileEditTool())
        subregistry.register(WebFetchTool())
        subregistry.register(WebSearchTool())
        subregistry.register(TodoWriteTool())

        registry.register(SubagentTool(
            client=client,
            model=model,
            registry=subregistry,
            token_tracker=self.token_tracker,
            system_prompt=(
                "你是一个专注于执行具体任务的子代理。请详细分析任务，使用工具解决问题。"
                "由于你是作为工具被调用的，请务必在任务完成后给出清晰、完整的总结报告。"
            ),
            max_turns=15,
            sub_model="deepseek-v4-flash"  # 子代理默认用轻量模型，节省成本
        ))

        # 系统提示词，目前硬编码
        system_prompt=f"""
        你是一个智能助手，可以使用各种工具来帮助用户完成任务。
        ### 可用技能列表 {skills_loader.get_description()} ###
        ### 长期记忆（最近摘要） ###
        {self.memory.brief_context()}
        ### 用户偏好（USER.md） ###
        {"\n".join(self.memory.user_preferences()) or "（当前没有用户偏好）"}
        """
        # 历史对话记录
        system_msg = {"role": "system", "content": system_prompt}
        self.memory.append_history(system_msg)

        self.runner = AgentRunner(
            client=client,
            model=model,
            tool_registry=registry,
            memory=self.memory,
            token_tracker=self.token_tracker
        )

        
    def run(self) -> None:
        """主循环，持续接受用户输入并生成回复"""
        while True:
            user_input = input("[You] : ")
            command = user_input.strip()
            if command.lower() in {"exit", "quit"}:
                # 打印 Token 统计
                stats = self.token_tracker.stats_by_model()
                if stats:
                    print("\n[Tokens] 本次会话 Token 消耗统计:")
                    for m, s in stats.items():
                        print(f"  - {m}: 输入 {s['input']}, 输出 {s['output']}, 缓存命中 {s['cache_hit']}")

                result = self.memory.compact()
                if result.get("summary") or result.get("preferences"):
                    print("[Memory] 已自动压缩并保存本次会话记录")
                print("退出对话")
                break
            msg = {"role": "user", "content": command}
            self.memory.append_history(msg)
            
            print("[Assistant] : ", end="", flush=True)
            for chunk in self.runner.step(self.memory.history):
                print(chunk, end="", flush=True)
            print("\n")


