from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from .runner import AgentRunner
from .tools import (
    ToolRegistry, BashTool, FileReadTool, FileWriteTool, FileEditTool, WebFetchTool, WebSearchTool,
    SkillTool, SkillsLoader
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
        # 注册工具
        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(FileEditTool())
        registry.register(WebFetchTool())
        registry.register(WebSearchTool())
        registry.register(SkillTool(skills_loader))

        # 系统提示词，目前硬编码
        system_prompt=f"""
        你是一个智能助手，可以使用各种工具来帮助用户完成任务。
        ### 可用技能列表 {skills_loader.get_description()} ###
        """
        # 历史对话记录
        self.history: list = []
        self.history.append({"role": "system", "content": system_prompt})

        self.runner = AgentRunner(
            client=client,
            model=model,
            tool_registry=registry,
        )

        

    def run(self) -> None:
        """主循环，持续接受用户输入并生成回复"""
        while True:
            user_input = input("[You] : ")
            command = user_input.strip()
            if command.lower() in {"exit", "quit"}:
                print("退出对话")
                break
            self.history.append({"role": "user", "content": command})
            print("[Assistant] : ", end="", flush=True)
            for chunk in self.runner.step(self.history):
                print(chunk, end="", flush=True)
            print("\n")


