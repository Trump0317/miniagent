"""事件总线 —— 组件间松耦合通信。

替代简单的回调钩子，提供发布/订阅模式。
组件只需 emit 事件，不关心谁来消费。

用法:
    bus = EventBus()
    bus.on("tool:after", lambda event: print(event.data))
    bus.emit("tool:after", data={"name": "bash", "result": "ok"})
"""

from __future__ import annotations
from typing import Callable, Any
from collections import defaultdict


class Event:
    """事件对象。"""
    __slots__ = ("name", "data", "source")

    def __init__(self, name: str, data: Any = None, source: str = ""):
        self.name = name
        self.data = data
        self.source = source

    def __repr__(self):
        return f"Event({self.name!r}, data={self.data!r})"


Handler = Callable[[Event], Any]


class EventBus:
    """轻量级发布/订阅事件总线。

    支持：
    - 通配符匹配（"tool:*" 匹配 "tool:before", "tool:after"）
    - 一次性监听（once）
    - 返回值收集（可用于拦截器模式）
    """

    def __init__(self):
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._once_handlers: dict[str, list[Handler]] = defaultdict(list)

    def on(self, event_name: str, handler: Handler | None = None):
        """注册事件监听器。也可用作装饰器。"""
        if handler is None:
            return lambda h: self.on(event_name, h)
        self._handlers[event_name].append(handler)
        return handler

    def once(self, event_name: str, handler: Handler | None = None):
        """注册一次性监听器。"""
        if handler is None:
            return lambda h: self.once(event_name, h)
        self._once_handlers[event_name].append(handler)
        return handler

    def off(self, event_name: str, handler: Handler) -> None:
        """移除监听器。"""
        for store in (self._handlers, self._once_handlers):
            lst = store.get(event_name, [])
            if handler in lst:
                lst.remove(handler)

    def emit(self, event_name: str, data: Any = None, source: str = "") -> list[Any]:
        """发布事件，返回所有监听器的返回值列表。

        若无监听器返回空列表。
        """
        event = Event(name=event_name, data=data, source=source)
        results: list[Any] = []

        # 精确匹配
        for h in self._handlers.get(event_name, []):
            results.append(h(event))

        # 一次性监听器
        once_list = self._once_handlers.pop(event_name, [])
        for h in once_list:
            results.append(h(event))

        # 通配符匹配
        parts = event_name.split(":")
        for i in range(len(parts)):
            pattern = ":".join(parts[:i]) + ":*"
            for h in self._handlers.get(pattern, []):
                results.append(h(event))

        return results
