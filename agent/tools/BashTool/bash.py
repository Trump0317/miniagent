from pydantic import BaseModel, Field
import subprocess
import platform
from typing import Optional, Type
from agent.tools.ToolRegisty.base import Tool, tool

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
    在宿主机环境中执行 shell 命令。
    """
    # 添加工具的名称、描述和参数模型避免静态解析报错
    name: str
    description: str
    args_model: Type[BashArgs]

    def execute(self, command: str, timeout: int = 300) -> str:
        # 兼容性处理
        is_windows = platform.system() == "Windows"
        
        try:
            # 执行命令
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
                return f"[BashTool]: Command executed successfully with return code {result.returncode} (No output)."
            
            return "\n".join(output)
            
        except subprocess.TimeoutExpired:
            return f"[BashTool]: Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"[BashTool]: Error occurred while executing command: {str(e)}"


