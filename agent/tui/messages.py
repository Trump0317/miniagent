"""TUI 消息模型 + 控件 + 样式。"""

from __future__ import annotations

from prompt_toolkit.layout import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType, MouseEvent
from prompt_toolkit.styles import Style


# ===========================================================
# Styles
# ===========================================================

STYLE = Style.from_dict({
    "header": "bg:#1a2b4a #ffffff bold",
    "status": "bg:#1a1a2e #888888",
    "input": "bg:#1a1a2e #cccccc",
    "input-border": "#3a3a5e",
    # 消息角色标签
    "user-label": "#5af bold",
    "agent-label": "#5f5 bold",
    "thinking-label": "#666 italic",
    "thinking-label-done": "#888",
    # 消息内容
    "thinking-text": "#888888 italic",
    "tool-title": "#5af",
    "tool-result": "#aaa",
    "tool-result-dim": "#666",
    "error-text": "#f66",
    # 分隔线
    "separator": "#333",
    # 状态
    "spinner": "#ff0",
    # Markdown
    "md-code-block": "#aaa",
    "md-inline-code": "bg:#2a2a3e #5af",
    "md-bold": "bold #fff",
    "md-italic": "italic",
    "md-h1": "bold #5af",
    "md-h2": "bold #5af",
    "md-h3": "bold #5af",
    "md-link": "#5af underline",
})


# ===========================================================
# Message
# ===========================================================

class Message:
    """单个聊天消息。工具状态+结果合并为一条。"""

    def __init__(self, role: str):
        self.role = role          # user / assistant / thinking / tool / error
        self.content = ""         # 文本内容
        self.tool_name = ""       # 工具名 (role == "tool")
        self.tool_result = ""     # 工具结果 (role == "tool")
        self.tool_collapsed = True
        self.thinking_collapsed = False
        self.done = False

    @property
    def display_text(self) -> str:
        if self.role == "tool":
            if self.tool_collapsed:
                return ""
            return self.tool_result[:3000] if self.tool_result else self.content
        if self.role == "thinking" and self.thinking_collapsed:
            return self.content[:200].replace("\n", " ") + (
                " ..." if len(self.content) > 200 else ""
            )
        return self.content

    @property
    def tool_byte_size(self) -> int:
        return len(self.tool_result.encode("utf-8")) if self.tool_result else 0

    @property
    def thinking_char_count(self) -> int:
        return len(self.content) if self.role == "thinking" else 0


# ===========================================================
# ScrollableChatControl
# ===========================================================

class ScrollableChatControl(FormattedTextControl):
    """拦截鼠标滚轮事件，委托给回调更新光标位置驱动滚动。"""

    def __init__(self, *args, scroll_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._scroll_callback = scroll_callback

    def mouse_handler(self, mouse_event: MouseEvent):
        if self._scroll_callback:
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                self._scroll_callback(-1)
                return None
            elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                self._scroll_callback(1)
                return None
        return super().mouse_handler(mouse_event)
