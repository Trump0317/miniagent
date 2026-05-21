from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Type, List, Optional
import urllib.request
import urllib.parse
import re

class WebSearchArgs(BaseModel):
    query: str = Field(description="联网搜索的查询关键词。")
    max_results: Optional[int] = Field(default=5, description="要返回的最大结果数。默认为 5 条。")

@tool(
    name="web_search_tool",
    description="使用 DuckDuckGo 搜索引擎进行联网搜索，并返回结果的标题和摘要。",
    parameters=WebSearchArgs,
)
class WebSearchTool(Tool):
    name: str 
    description: str
    args_model: Type[WebSearchArgs]

    def execute(self, query: str, max_results: int = 5) -> str:
        try:
            # 使用 DuckDuckGo HTML 版本
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8')

            results = []
            # 改进正则，匹配包含 result__body 类的 div
            bodies = re.findall(r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)<div class="clear"></div>', html, re.DOTALL)
            
            for body in bodies[:max_results]:
                # 匹配标题，注意可能存在的 rel="nofollow" 或其他属性
                title_match = re.search(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', body, re.DOTALL)
                # 匹配摘要
                snippet_match = re.search(r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', body, re.DOTALL)
                
                if title_match:
                    title = re.sub(r'<.*?>', '', title_match.group(1)).strip()
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<.*?>', '', snippet_match.group(1)).strip()
                    
                    if title:
                        results.append(f"Title: {title}\nSnippet: {snippet}\n")
            
            if not results:
                # 如果没找到结果，检查是否被反爬虫拦截
                if "ddg-captcha" in html.lower() or "robot" in html.lower():
                    return f"[WebSearchTool]: Access denied by DuckDuckGo (Bot detection). Please try again later or use web_fetch_tool."
                return f"[WebSearchTool]: No results found for '{query}'"
            
            return "---\n" + "\n---\n".join(results)
        except Exception as e:
            return f"[WebSearchTool]: Search error: {str(e)}"
