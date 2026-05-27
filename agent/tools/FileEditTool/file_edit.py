from __future__ import annotations
from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Optional, Type

class FileEditArgs(BaseModel):
    file_path: str = Field(description="要编辑的文件的完整路径。")
    old_str: str = Field(description="文件中要被替换的精确字符串（支持多行）。")
    new_str: str = Field(description="替换后的新字符串。")

@tool(
    name="file_edit_tool",
    description="通过搜寻并替换特定文本块的方式编辑文件（补丁模式）。支持精确匹配和忽略缩进的模糊匹配。",
    parameters=FileEditArgs,
)
class FileEditTool(Tool):

    # ── 匹配策略 ──

    @staticmethod
    def _strip_leading_whitespace(text: str) -> list[str]:
        """将文本按行去除前导空白，用于忽略缩进的匹配"""
        return [line.lstrip() for line in text.split('\n')]

    def _try_indent_relaxed_match(self, content: str, old_str: str, new_str: str) -> tuple[bool, str | None]:
        """策略 2: 忽略每行前导空白后匹配。
        
        原理：先将 old_str 和文件内容都去除每行前导空白，再尝试匹配。
        匹配成功后，用 old_str 的原始文本（带缩进）在匹配位置做替换。
        """
        old_stripped = self._strip_leading_whitespace(old_str)
        content_lines = content.split('\n')
        content_stripped = [line.lstrip() for line in content_lines]

        n = len(old_stripped)
        matches = []
        for i in range(len(content_stripped) - n + 1):
            if content_stripped[i:i + n] == old_stripped:
                matches.append(i)

        if len(matches) == 0:
            return False, None
        if len(matches) > 1:
            return False, (
                f"[FileEditTool]: 忽略缩进后匹配到 {len(matches)} 处，"
                f"请提供更多上下文使 old_str 唯一。"
            )

        # 找到匹配位置，保留原文件的缩进风格
        start_line = matches[0]
        end_line = start_line + n
        matched_block = '\n'.join(content_lines[start_line:end_line])
        new_content = content.replace(matched_block, new_str, 1)
        return True, new_content

    @staticmethod
    def _snippet(content: str) -> str:
        """生成错误提示用的代码片段预览"""
        lines = content.split('\n')
        if len(lines) <= 10:
            numbered = '\n'.join(f"{i+1:>4}|{line}" for i, line in enumerate(lines))
            return f"文件内容:\n{numbered}"
        # 只显示前 20 行
        preview = '\n'.join(f"{i+1:>4}|{line}" for i, line in enumerate(lines[:20]))
        return f"文件前 20 行:\n{preview}\n... (共 {len(lines)} 行)"

    # ── 主执行 ──

    def execute(self, file_path: str, old_str: str, new_str: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 策略 1: 精确匹配
            count = content.count(old_str)
            if count == 1:
                new_content = content.replace(old_str, new_str, 1)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return f"[FileEditTool]: 已成功编辑 {file_path}（精确匹配）。"
            elif count > 1:
                return (
                    f"[FileEditTool]: old_str 匹配到 {count} 处，不唯一。\n"
                    f"请增加上下文（上下各多包含几行）使匹配唯一。"
                )

            # 策略 2: 忽略缩进匹配
            ok, result = self._try_indent_relaxed_match(content, old_str, new_str)
            if ok and result is not None:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(result)
                return f"[FileEditTool]: 已成功编辑 {file_path}（通过忽略缩进匹配）。"
            elif result is not None:
                return result

            # 全部失败：给出有用的错误提示
            snippet = self._snippet(content)
            return (
                f"[FileEditTool]: 在 {file_path} 中未找到 old_str。\n\n"
                f"{snippet}\n\n"
                f"请检查：\n"
                f"  1. old_str 是否与文件内容完全一致（注意空白字符）\n"
                f"  2. 是否遗漏了某些行\n"
                f"  3. 路径是否正确"
            )
        except FileNotFoundError:
            return f"[FileEditTool]: 文件不存在: {file_path}"
        except PermissionError:
            return f"[FileEditTool]: 无权限写入: {file_path}"
        except Exception as e:
            return f"[FileEditTool]: 编辑文件时出错: {str(e)}"