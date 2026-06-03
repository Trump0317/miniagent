# miniagent

基于 LLM 的智能助手框架。ReAct 模式 + 事件驱动 + 树状会话，约 5000 行 Python。

支持 CLI / TUI / Web 三种交互方式，内置工具调用、MCP 协议、子代理、会话分叉/压缩、三层记忆。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key

# CLI
python agent.py                        # 交互模式
python agent.py -p "你好"              # 单次模式

# TUI
python agent.py --tui                  # 终端界面

# Web UI
python agent.py --web                  # http://127.0.0.1:8000
python agent.py --web --port 8080      # 自定义端口

# 恢复会话
python agent.py -r
```

## 架构

```
agent.py                         ← 入口（--tui / --web 分发）
└── agent/
    ├── ai/                       # LLM 调用封装 + Provider 配置
    ├── cli/                      # CLI 交互外壳
    ├── tui/                      # TUI 终端界面（prompt_toolkit）
    ├── core/                     # 核心引擎
    │   ├── agent.py              #   Agent 装配层
    │   ├── runner.py             #   think-act 循环（含 LLM 重试）
    │   ├── session_tree.py       #   树状会话
    │   ├── memory.py             #   三层记忆（树为唯一数据源）
    │   ├── compaction.py         #   压缩编排
    │   ├── events.py             #   事件总线（通配符 + once）
    │   ├── observability.py      #   结构化日志 + trace + 耗时
    │   ├── chunks.py             #   统一 chunk 协议
    │   ├── system_prompt.py/md   #   系统提示词
    │   ├── tracker.py            #   Token 统计
    │   └── prompts.py            #   /command 模板
    ├── tools/                    # 工具集（扁平，每个工具一个文件）
    │   ├── bash.py, file_read.py, file_write.py, file_edit.py
    │   ├── web_fetch.py, web_search.py
    │   ├── todo.py, skill.py, subagent.py
    │   ├── mcp_client.py         #   MCP 客户端
    │   ├── base.py, registry.py, executor.py
    ├── subagent/                 # 子代理定义（.md + YAML）
    ├── prompts/                  # 命令模板
    └── web/                      # Web UI（FastAPI + WebSocket）
```

## 功能清单

| 类别 | 功能 |
|------|------|
| **核心** | ReAct 循环、流式输出、思维链、事件驱动 |
| **韧性** | LLM 自动重试（指数退避，最多 3 次）、熔断 |
| **可观测** | Trace ID、结构化日志（JSONL）、耗时统计、token 汇总 |
| **CLI** | readline 历史、内置命令、多行输入、智能输出着色 |
| **TUI** | 对话气泡、鼠标翻页、工具折叠、快捷键 |
| **Web** | FastAPI + WebSocket 流式、多会话、分支树、文件上传、模型切换 |
| **工具** | Bash（安全护栏）、文件读写编辑、网络搜索/抓取、Todo、Skill |
| **MCP** | stdio 传输、自动工具发现、Claude 兼容 mcp.json |
| **子代理** | 单/并行/链式、Markdown 定义、独立模型和工具 |
| **会话** | 树状分叉/返回、三层记忆、自动压缩、隔离存储 |
| **配置** | 运行时切换模型/思考级别/轮数（/model /thinking /turns） |
| **扩展** | 外部提示词模板、子代理文件定义、自定义 Provider |

## 三种界面

```bash
python agent.py                  # CLI: readline 交互
python agent.py --tui            # TUI: 对话气泡 + 快捷键
python agent.py --web            # Web: 浏览器 http://127.0.0.1:8000
```

### CLI 内置命令

```
/help             显示帮助
/config           显示/修改配置
/model [name]     显示/切换模型
/thinking [level] 显示/切换思考级别 (off/minimal/low/medium/high/xhigh)
/turns [n]        显示/设置最大轮数（0 = 无限制）
/tree             显示会话分支树
/fork [n]         分叉到第 n 条消息之前
/back             返回分叉前位置
/clear            清屏
```

### TUI 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+Q | 退出 |
| Ctrl+F | 分叉 |
| Ctrl+B | 返回 |
| Ctrl+T | 分支树 |
| Ctrl+E | 展开/折叠工具结果 |
| Esc | 退出树视图 / 清空输入 |

## MCP 支持

在项目根目录创建 `mcp.json`（兼容 Claude Code 格式）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    }
  }
}
```

启动时自动连接，工具以 `mcp__<server>__<tool>` 前缀注册。连接失败跳过并警告。

## 子代理

在 `agent/subagent/` 下创建 `.md` 文件：

```markdown
---
name: my-agent
description: 我的代理
tools: bash_tool, file_read_tool
model: deepseek-v4-flash
max_turns: 10
---

你是我的自定义子代理...
```

LLM 通过 `subagent_tool` 调用：`/scout` 展开为用 scout 侦查代码，`/review` 展开为用 reviewer 审查。

## 可观测性

每次请求自动生成 trace ID，记录到 `sessions/<ts>/trace.jsonl`：

```json
{"ts":"2026-06-03T10:48:12","level":"INFO","trace":"a32d8df6","event":"request:start","data":{"message":"hello"}}
{"ts":"2026-06-03T10:48:12","level":"INFO","trace":"a32d8df6","event":"turn:start","data":{"turn":1}}
{"ts":"2026-06-03T10:48:12","level":"INFO","trace":"a32d8df6","event":"tool:end","data":{"tool":"bash_tool","duration_ms":15,"ok":true}}
{"ts":"2026-06-03T10:48:12","level":"INFO","trace":"a32d8df6","event":"request:summary","data":{"turns":1,"tool_calls":1,...}}
```

## 配置 (.env)

```bash
# DeepSeek（默认）
DEEPSEEK_API_KEY=sk-xxx

# 或 OpenAI
# OPENAI_API_KEY=sk-xxx

# 或自定义
# API_KEY=xxx
# API_BASE_URL=http://localhost:8000/v1
```

## 测试

```bash
python -m unittest discover tests    # 552 个测试
```

| 层级 | 覆盖 |
|------|------|
| 单元测试 | 21 个模块，全组件覆盖 |
| 集成测试 | Agent 完整流程、持久化、树导航、压缩、重试 |

## 项目文档

- `doc/agent_logic.md` — 核心逻辑详解（架构、数据流、持久化、压缩）
- `doc/multi-agent-analysis.md` — Multi-Agent 可行性分析
- `doc/commit.md` — 提交规范
- `AGENTS.md` — 项目上下文（供 AI 编码助手使用）
