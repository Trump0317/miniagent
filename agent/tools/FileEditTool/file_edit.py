from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Optional, Type

class FileEditArgs(BaseModel):
    file_path: str = Field(description="要编辑的文件的完整路径。")
    old_str: str = Field(description="文件中要被替换的精确字符串（支持多行）。")
    new_str: str = Field(description="替换后的新字符串。")

@tool(
    name="file_edit_tool",
    description="通过搜寻并替换特定文本块的方式编辑文件（补丁模式）。请提供足够唯一的 old_str 以确保精确匹配。",
    parameters=FileEditArgs,
)
class FileEditTool(Tool):
    name: str
    description: str
    args_model: Type[FileEditArgs]

    def execute(self, file_path: str, old_str: str, new_str: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_str not in content:
                return f"[FileEditTool]: Error: old_str not found in {file_path}. Please ensure the old_str matches the file content exactly, including whitespace and indentation."

            occurrence_count = content.count(old_str)
            if occurrence_count > 1:
                return f"[FileEditTool]: Error: old_str found {occurrence_count} times. Please provide more context to uniquely identify the block to replace."

            new_content = content.replace(old_str, new_str)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"[FileEditTool]: Successfully edited {file_path}."
        except Exception as e:
            return f"[FileEditTool]: Error occurred while editing file: {str(e)}"