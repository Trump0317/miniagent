from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Optional, Type


class FileWriteArgs(BaseModel):
    file_path: str = Field(description="要写入的文件路径。")
    content: str = Field(description="要写入文件的内容。")


@tool(
    name="file_write_tool",
    description="直接将指定内容写入整个文件。如果文件已存在，将覆盖原有内容；如果文件不存在，将创建新文件并写入内容。",
    parameters=FileWriteArgs,
)
class FileWriteTool(Tool):

    def execute(self, file_path: str, content: str) -> str:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[FileWriteTool]: Content successfully written to {file_path}."
        except Exception as e:
            return f"[FileWriteTool]: Error occurred while writing to file: {str(e)}"