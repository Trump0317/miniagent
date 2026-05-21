from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Type



class FileReadArgs(BaseModel):
    file_path: str = Field(description="要读取的文件路径。")
    encoding: str = Field(default="utf-8", description="文件编码格式，默认为 'utf-8'。")

@tool(
    name="file_read_tool",
    description="读取本地文件内容并返回其文本。",
    parameters=FileReadArgs,
)
class FileReadTool(Tool):
    name: str
    description: str
    args_model: Type[FileReadArgs]

    def execute(self, file_path: str, encoding: str = "utf-8") -> str:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            return f"[FileReadTool]: Error occurred while reading file: {str(e)}"