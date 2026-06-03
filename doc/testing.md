# 全流程测试指南

> 合并到 main 之前执行，验证全部功能完整性。

## 环境准备

```bash
# 全新虚拟环境
python3 -m venv /tmp/test_venv
source /tmp/test_venv/bin/activate
pip install -e ".[all]"
```

## 1. 启动检查（无需 API Key）

```bash
# CLI
miniagent --help
# 预期: 显示参数列表（-p, --tui, --web, --thinking 等）

# TUI
miniagent --tui --help 2>&1 | head -5
# 预期: 显示 TUI 参数或正常退出

# Web
timeout 2 miniagent --web --port 9999 2>&1
# 预期: Uvicorn running on http://127.0.0.1:9999
```

## 2. 单元 + 集成测试

```bash
python -m unittest discover tests
# 预期: Ran 552 tests ... OK
```

若有个别测试失败，运行对应模块排查：

```bash
python -m unittest tests.test_runner -v
python -m unittest tests.test_integration -v
python -m unittest tests.test_mcp_client -v
python -m unittest tests.test_observability -v
```

## 3. Web API 测试

```bash
miniagent --web --port 9999 &
sleep 2

# 首页可访问
curl -s http://127.0.0.1:9999/ | head -5
# 预期: <!DOCTYPE html>

# 会话列表
curl -s http://127.0.0.1:9999/api/sessions
# 预期: []

# 创建会话
curl -s -X POST http://127.0.0.1:9999/api/sessions | python -m json.tool
# 预期: {"id": "20260603-...", "created": "..."}

# 文件上传
echo "test content" > /tmp/test_upload.txt
curl -s -X POST -F "file=@/tmp/test_upload.txt" http://127.0.0.1:9999/api/upload | python -m json.tool
# 预期: {"ok": true, "filename": "test_upload.txt", "ref": "@/home/...", ...}

# @引用解析
UPLOAD_PATH=$(curl -s -X POST -F "file=@/tmp/test_upload.txt" http://127.0.0.1:9999/api/upload | python -c "import sys,json; print(json.load(sys.stdin)['path'])")
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"path\": \"$UPLOAD_PATH\"}" \
  http://127.0.0.1:9999/api/upload/resolve | python -m json.tool
# 预期: {"ok": true, "filename": "...", "content": "test content\n", ...}

kill %1
rm -f /tmp/test_upload.txt
```

## 4. MCP 功能测试（需 npx）

```bash
cat > /tmp/test_mcp.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
EOF

export DEEPSEEK_API_KEY=sk-test
python -c "
from agent.tools.mcp_client import McpClientManager
mgr = McpClientManager('/tmp/test_mcp.json')
tools = mgr.get_tool_classes()
print(f'MCP tools loaded: {len(tools)}')
for t in tools:
    print(f'  {t._tool_name}')
mgr.shutdown()
"
# 预期: MCP tools loaded: 14

rm -f /tmp/test_mcp.json
```

## 5. 端到端 LLM 交互（需真实 API Key）

```bash
export DEEPSEEK_API_KEY=sk-your-real-key

# 单次对话
echo "你好，请用一句话介绍自己" | miniagent -p
# 预期: 正常返回 LLM 回复

# 工具调用（列出当前目录文件）
miniagent -p "列出当前目录的文件"
# 预期: agent 自动调用 bash_tool ls 并返回结果

# 多行输入
printf "hello\nworld\n" | miniagent -p "翻译以下内容为中文"
# 预期: 翻译结果

# WebSocket 测试（需 websocket-client）
pip install websocket-client
python -c "
import json, websocket
ws = websocket.create_connection('ws://127.0.0.1:9999/ws')
ws.send(json.dumps({'type': 'message', 'session_id': 'test', 'content': 'hi'}))
resp = json.loads(ws.recv())
print('WS response type:', resp.get('type'))
ws.close()
"
```

## 6. 路径验证

```bash
python -c "
from agent import AppConfig
cfg = AppConfig.from_env()
print('root:', cfg.root)
print('memory_dir:', cfg.memory_dir)
print('skills_dir:', cfg.skills_dir)
# 预期: root 和 memory_dir/skills_dir 都在当前工作目录下
"
```

## 7. 清理

```bash
deactivate
rm -rf /tmp/test_venv /tmp/.memory /tmp/skills /tmp/sessions
```

## 通过标准

| 检查项 | 标准 |
|--------|------|
| 启动检查 | `--help` 正常输出 |
| 单元测试 | 552 tests OK |
| Web API | 首页返回 HTML，API 端点返回 JSON |
| MCP 功能 | 14 个工具加载成功 |
| LLM 交互 | 正常回复，工具可调用（有 Key 时） |
| 路径验证 | 用户数据在项目目录，不在 site-packages |
