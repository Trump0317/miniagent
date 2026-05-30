"""miniagent Web UI — FastAPI + WebSocket 流式对话。

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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from agent import Agent, AppConfig
from agent.ai.context import load_context_files

logger = logging.getLogger(__name__)

# ── FastAPI app ──

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="miniagent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """返回聊天界面."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/favicon.ico")
async def favicon():
    """空图标，避免 404 日志."""
    return Response(status_code=204)


# ── Agent 管理 ──

_agent_lock = threading.Lock()
_agent: Agent | None = None


def _get_or_create_agent() -> Agent:
    """懒加载 Agent 单例（WebSocket 连接复用）。"""
    global _agent
    with _agent_lock:
        if _agent is None:
            ctx = load_context_files(user_dir=Path.home() / ".miniagent")
            _agent = Agent(AppConfig.from_env(context_files=ctx))
        return _agent


# ── WebSocket 端点 ──


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    agent = _get_or_create_agent()

    # 队列桥接：同步 Agent.process() → 异步 WS
    chunk_queue: queue.Queue[str | None] = queue.Queue()

    def _run_agent(message: str):
        """在线程中运行 agent.process()，chunk 写入队列."""
        try:
            for chunk in agent.process(message):
                chunk_queue.put(chunk)
            chunk_queue.put(None)  # 结束信号
        except Exception as e:
            chunk_queue.put(None)
            logger.exception("agent.process 异常")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")
            content = data.get("content", "")

            if msg_type == "message" and content:
                t = threading.Thread(target=_run_agent, args=(content,), daemon=True)
                t.start()

                # 从队列流式推送 chunk
                while True:
                    chunk = await asyncio.to_thread(chunk_queue.get)
                    if chunk is None:
                        await ws.send_json({"type": "done"})
                        break
                    await ws.send_json({"type": "text", "content": chunk})

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
