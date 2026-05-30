"""会话管理器 — 多会话 Agent 实例生命周期管理."""

from __future__ import annotations
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


class SessionInfo:
    """会话元信息."""

    def __init__(self, session_id: str):
        self.id = session_id
        self.created = datetime.now()
        self.agent: Agent | None = None
        self.model = ""
        self.lock = threading.Lock()


class SessionManager:
    """管理多个会话的 Agent 实例."""

    def __init__(self):
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str | None = None) -> SessionInfo:
        """创建新会话。未指定 ID 时自动生成时间戳。"""
        sid = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        with self._lock:
            # 同名会话复用
            if sid in self._sessions:
                return self._sessions[sid]
            info = SessionInfo(sid)
            self._sessions[sid] = info
        return info

    def get(self, session_id: str) -> SessionInfo | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """返回会话列表（按创建时间倒序）."""
        with self._lock:
            items = []
            for info in sorted(
                self._sessions.values(),
                key=lambda s: s.created,
                reverse=True,
            ):
                items.append({
                    "id": info.id,
                    "created": info.created.isoformat(),
                    "model": info.model,
                })
            return items

    def get_or_create_agent(self, session_id: str) -> "Agent":
        """获取或创建 Agent 实例（线程安全懒加载）."""
        from agent import Agent, AppConfig
        from agent.ai.context import load_context_files

        info = self.get(session_id)
        if not info:
            info = self.create(session_id)

        with info.lock:
            if info.agent is None:
                ctx = load_context_files(user_dir=Path.home() / ".miniagent")
                cfg = AppConfig.from_env(
                    context_files=ctx,
                    session_id=session_id,
                )
                info.agent = Agent(cfg)
                info.model = cfg.model
            return info.agent
