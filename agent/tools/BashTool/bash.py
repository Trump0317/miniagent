from pydantic import BaseModel, Field
import subprocess
import re
from typing import Optional, Type, ClassVar
from agent.tools.ToolRegisty.base import Tool, tool

# ── 安全护栏：禁止执行的命令模式 ──
# 注意：这不是真正的沙箱，只是防止 LLM 意外执行破坏性操作。
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    # (正则, 说明)
    (r"rm\s+(-[rRf]\s*)+\s*/",           "禁止递归强制删除根目录"),
    (r"rm\s+(-[rRf]\s*)+\s*~",           "禁止删除用户目录"),
    (r"rm\s+(-[rRf]\s*)+\s*\$HOME",      "禁止删除 HOME 目录"),
    (r"\bmkfs\.",                          "禁止格式化文件系统"),
    (r"dd\s+if=",                          "禁止直接操作块设备"),
    (r">\s*/dev/sd[a-z]",                 "禁止覆写磁盘设备"),
    (r"chmod\s+(-R\s+)?777\s*/",          "禁止对根目录开放所有权限"),
    (r"chown\s+(-R\s+)?[^\s]+\s*/",       "禁止递归变更根目录所有者"),
    (r":\s*\{\s*:\|:&\s*\}\s*;\s*:",       "禁止 fork 炸弹"),
    (r"wget\s+.*\|\s*(ba)?sh",             "禁止下载并执行远程脚本"),
    (r"curl\s+.*\|\s*(ba)?sh",             "禁止通过 curl 下载并执行远程脚本"),
]

class BashArgs(BaseModel):
    command: str = Field(description="要在终端中执行的有效命令字符串。")
    timeout: Optional[int] = Field(default=300, description="命令执行的最大超时时间（秒），默认为 300 秒。")

@tool(
    name="bash_tool",
    description="在系统终端中执行命令并返回标准输出(stdout)和标准错误(stderr)。支持 Windows (PowerShell) 并兼容 Linux/macOS。",
    parameters=BashArgs,
)
class BashTool(Tool):
    """
    在宿主机环境中执行 shell 命令，附带安全护栏。
    """
    name: str
    description: str
    args_model: Type[BashArgs]

    def _check_safety(self, command: str) -> str | None:
        """检查命令是否安全，返回 None 表示通过，否则返回拒绝原因"""
        for pattern, reason in FORBIDDEN_PATTERNS:
            if re.search(pattern, command):
                return f"[BashTool]: 安全拦截 - {reason}\n匹配模式: {pattern}"
        return None

    def execute(self, command: str, timeout: int = 300) -> str:
        # 1. 安全护栏
        block_reason = self._check_safety(command)
        if block_reason:
            return block_reason

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            output = []
            if result.stdout:
                output.append(f"[BashTool]: STDOUT:\n{result.stdout}")
            if result.stderr:
                output.append(f"[BashTool]: STDERR:\n{result.stderr}")
            
            if not output:
                return f"[BashTool]: 命令执行成功，返回码 {result.returncode}，无输出。"
            
            return "\n".join(output)
            
        except subprocess.TimeoutExpired:
            return f"[BashTool]: Error: 命令超时 ({timeout}s)。"
        except Exception as e:
            return f"[BashTool]: Error: {str(e)}"


