from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Optional, Type
import urllib.request
import urllib.error

class WebFetchArgs(BaseModel):
    url: str = Field(description="要获取的 URL 地址。")
    timeout: Optional[int] = Field(default=30, description="请求的最大超时时间（秒），默认为 30 秒。")

@tool(
    name="web_fetch_tool",
    description="从指定 URL 获取内容并返回文本。支持 HTTP 和 HTTPS 协议。",
    parameters=WebFetchArgs,
)
class WebFetchTool(Tool):

    def run(self, args: WebFetchArgs) -> str:
        """实现 Tool 的 run 方法"""
        return self.execute(args.url, args.timeout or 30)

    def execute(self, url: str, timeout: int = 30) -> str:
        try:
            # 模拟浏览器 User-Agent
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            return f"[WebFetchTool]: HTTP Error {e.code}"
        except Exception as e:
            return f"[WebFetchTool]: Error: {str(e)}"
