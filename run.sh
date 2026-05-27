#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

# 如果 .venv 不存在，自动创建
if [ ! -d "$VENV" ]; then
    echo "[miniagent] 创建虚拟环境..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet -r "$ROOT/requirements.txt"
    echo "[miniagent] 环境初始化完成"
fi

# 如果不存在 .env，从 .env.example 复制
if [ ! -f "$ROOT/.env" ] && [ -f "$ROOT/.env.example" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "[miniagent] .env 已从 .env.example 创建，请编辑填入 API Key"
fi

cd "$ROOT"
exec "$VENV/bin/python" agent.py
