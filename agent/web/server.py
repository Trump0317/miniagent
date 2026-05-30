"""miniagent Web UI — FastAPI + WebSocket 流式对话，支持多会话。

启动:
    python -m agent.web.server
    python -m agent.web.server --port 8080
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import queue
import threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from agent.web.session import SessionManager

logger = logging.getLogger(__name__)

# ── FastAPI app ──

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="miniagent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── 会话管理器 ──

sessions = SessionManager()


@app.get("/")
async def index():
    """返回聊天界面."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ── REST API: 会话管理 ──


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话."""
    return sessions.list_sessions()


@app.post("/api/sessions")
async def create_session():
    """创建新会话."""
    info = sessions.create()
    return {"id": info.id, "created": info.created.isoformat()}


@app.get("/api/sessions/{session_id}/tree")
async def session_tree(session_id: str):
    """获取会话分支树."""
    agent = sessions.get_or_create_agent(session_id)
    entries = agent.memory.get_tree_entries()
    if not entries:
        return {"nodes": [], "leaf_id": None}

    nodes = []
    for e in entries:
        nodes.append({
            "id": e.id,
            "parent_id": e.parent_id,
            "type": e.type,
            "role": e.role,
            "content": (e.content or "")[:60],
            "timestamp": e.timestamp,
        })
    return {"nodes": nodes, "leaf_id": agent.memory.leaf_id}


@app.get("/api/sessions/{session_id}/history")
async def session_history(session_id: str):
    """获取会话对话历史（完整消息）."""
    agent = sessions.get_or_create_agent(session_id)
    # memory.history 返回 LLM 格式的消息列表
    messages = agent.memory.history
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]


@app.post("/api/sessions/{session_id}/fork")
async def session_fork(session_id: str, target_id: str = ""):
    """分叉到指定父节点."""
    agent = sessions.get_or_create_agent(session_id)
    if not target_id:
        return {"ok": False, "error": "缺少 target_id"}
    try:
        agent.memory.push_fork()
        agent.memory.fork(target_id)
        agent._system_prompt = agent._prompt.build()
        return {"ok": True, "leaf_id": agent.memory.leaf_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/sessions/{session_id}/back")
async def session_back(session_id: str):
    """返回分叉前位置."""
    agent = sessions.get_or_create_agent(session_id)
    target = agent.memory.pop_fork()
    if target:
        agent.memory.navigate(target)
        agent._system_prompt = agent._prompt.build()
        return {"ok": True, "leaf_id": agent.memory.leaf_id}
    return {"ok": False, "error": "fork 栈为空"}


# ── WebSocket 端点 ──


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # 队列：agent.process() chunk → WS 推送
    chunk_queue: queue.Queue[str | None] = queue.Queue()

    def _run_agent(agent, message: str):
        """在线程中运行 agent.process()."""
        try:
            for chunk in agent.process(message):
                chunk_queue.put(chunk)
            chunk_queue.put(None)
        except Exception:
            chunk_queue.put(None)
            logger.exception("agent.process 异常")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            if msg_type == "switch":
                # 前端请求切换到指定会话
                sid = data.get("session_id", "")
                if sid:
                    agent = sessions.get_or_create_agent(sid)
                    await ws.send_json({
                        "type": "switched",
                        "session_id": sid,
                        "model": agent.config.model,
                    })

            elif msg_type == "message":
                sid = data.get("session_id", "default")
                content = data.get("content", "")
                if not content:
                    continue

                agent = sessions.get_or_create_agent(sid)

                t = threading.Thread(
                    target=_run_agent, args=(agent, content), daemon=True,
                )
                t.start()

                while True:
                    chunk = await asyncio.to_thread(chunk_queue.get)
                    if chunk is None:
                        await ws.send_json({"type": "done"})
                        break
                    await ws.send_json(chunk)

    except WebSocketDisconnect:
        logger.info("WebSocket 断开")
    except Exception:
        logger.exception("WebSocket 异常")


# ── 启动 ──


def main():
    parser = argparse.ArgumentParser(description="miniagent Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
