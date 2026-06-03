# miniagent 核心逻辑文档

> 面向开发者和维护者，描述架构设计、模块职责、数据流和扩展方式。

## 目录

1. [架构总览](#1-架构总览)
2. [关键组件](#2-关键组件)
3. [请求处理流程](#3-请求处理流程)
4. [会话持久化](#4-会话持久化)
5. [压缩流程](#5-压缩流程)
6. [事件系统](#6-事件系统)
7. [MCP 支持](#7-mcp-支持)
8. [LLM 重试](#8-llm-重试)
9. [可观测性](#9-可观测性)
10. [CLI / TUI / Web](#10-cli--tui--web)
11. [测试](#11-测试)

---

## 1. 架构总览

```
agent.py                         ← 入口（--tui / --web 分发）
└── agent/
    ├── ai/                       # Provider 配置 + LLM 调用
    │   ├── config.py             #   AppConfig: 多 Provider + 会话隔离
    │   ├── llm.py                #   LLMClient: 流式调用封装
    │   └── context.py            #   AGENTS.md 自动加载
    ├── cli/                      # CLI 外壳
    │   ├── app.py                #   交互/print 模式、readline、智能输出
    │   └── helpers.py            #   handle_tree/fork/back
    ├── tui/                      # TUI 终端界面
    │   └── app.py                #   prompt_toolkit 对话气泡
    ├── core/                     # 核心引擎
    │   ├── agent.py              #   Agent: 装配层
    │   ├── runner.py             #   AgentRunner: think-act + 重试
    │   ├── session_tree.py       #   SessionTree: 树状会话
    │   ├── memory.py             #   AgentMemory: 三层记忆
    │   ├── compaction.py         #   CompactionService: 压缩编排
    │   ├── events.py             #   EventBus: 发布/订阅 + 通配符
    │   ├── observability.py      #   结构化日志 + trace + 耗时
    │   ├── chunks.py             #   ChunkType 枚举 + 工厂函数
    │   ├── system_prompt.py/md   #   系统提示词
    │   ├── tracker.py            #   TokenTracker
    │   └── prompts.py            #   /command 模板加载
    ├── tools/                    # 工具集（扁平布局）
    │   ├── base.py               #   Tool 基类 + @tool 装饰器
    │   ├── registry.py           #   ToolRegistry
    │   ├── executor.py           #   ToolExecutor: 串行/并行
    │   ├── mcp_client.py         #   MCP 客户端
    │   ├── bash.py               #   终端命令
    │   ├── file_read/write/edit  #   文件操作
    │   ├── web_fetch/search      #   网络工具
    │   ├── todo.py               #   待办管理
    │   ├── skill.py              #   技能加载
    │   └── subagent.py           #   子代理
    ├── subagent/                 # 子代理定义 (.md)
    ├── prompts/                  # 命令模板
    └── web/                      # Web UI
        ├── server.py             #   FastAPI + WebSocket
        ├── session.py            #   多会话管理
        └── static/index.html     #   前端界面
```

### 设计原则

1. **树状会话** — SessionTree 管理分支，分叉不丢数据，压缩插入 COMPACT 节点
2. **无列表双写** — AgentMemory 以树为唯一数据源，`history` 实时计算
3. **事件驱动** — EventBus 发布/订阅 + 通配符，解耦组件
4. **统一 chunk** — 所有组件产出 `{type, content}` 格式，CLI/TUI/Web 共用

---

## 2. 关键组件

### 2.1 AppConfig

- `PROVIDER_PRESETS`: deepseek / openai / custom 三组预设
- `from_env()`: 按 `DEEPSEEK_API_KEY → OPENAI_API_KEY → API_KEY` 自动检测
- 会话隔离：每会话独立 `sessions/<ts>/` 目录
- 新增 `mcp_config_path` 字段

### 2.2 LLMClient

封装 OpenAI 兼容流式调用，产出结构化 chunk：

| chunk type | 说明 |
|-----------|------|
| `content` | LLM 正文 |
| `reasoning` | 推理/思考（DeepSeek thinking） |
| `tool_call` | 工具调用（id/name/arguments） |
| `usage` | token 用量 |

支持 `--thinking off/minimal/low/medium/high/xhigh`。

### 2.3 AgentRunner

think-act 循环编排，产出统一 chunk。核心能力：

- **LLM 流式调用**（含自动重试，见 §8）
- **工具调用分片修复**：DeepSeek 流式 arguments 逐字符拼接
- **工具调度**：委托给 ToolExecutor（串行/并行）
- **熔断**：达到 max_turns 自动停止
- **不完整工具过滤**：id 或 name 缺失的工具调用被丢弃

### 2.4 SessionTree

树状会话结构：

- `SessionEntry`: id/parent_id/type/role/content/metadata/timestamp
- `type`: "message"（普通消息）或 "compaction"（压缩节点）
- 核心操作: `append()` / `compact()` / `fork()` / `navigate()` / `build_context()`
- `build_context()` 遇 COMPACT 节点自动跳过已压缩消息

### 2.5 CompactionService

完整压缩编排：

1. `should_compact()`: token 阈值检查
2. `_find_cut_point()`: token-aware 切点
3. `_extract()`: LLM 提取 summary/preferences/facts
4. `_dispatch_to_memory()`: 分发到三层记忆
5. `memory.compress_tree()`: 插入 COMPACT 节点
6. `prompt.build(data)`: 重建系统提示词

### 2.6 SystemPrompt

从 `agent/core/system_prompt.md` 加载模板，`build()` 时通过 `{placeholder}` 注入动态内容：context_files / skills / agents / commands / memory / preferences / compaction。

---

## 3. 请求处理流程

```
用户输入
  → agent.py: 内置命令直接处理 (/help /config /model /thinking /turns
               /tree /fork /back /clear)
  → agent.py: /command 展开为模板
  → Agent.process(message)
     → EventBus.emit("message:received")          ← 可观测性起点
     → memory.append_message(user msg)
     → Agent._build_context() → [system] + history
     → AgentRunner.step(context)
        → EventBus.emit("turn:start")
        → LLMClient.stream() → chunk 生成器
        → 无工具调用 → yield done → 返回
        → ToolExecutor.execute()
           → tool:before / tool:after 事件钩子
           → 单工具: 串行流式 → 并行: ThreadPoolExecutor
        → context.append(assistant/tool msg)
        → 继续循环
     → 同步增量到 memory
  → Agent.shutdown()
     → bus.emit("session:end")
     → CompactionService.compact()
     → McpClientManager.shutdown()
     → 返回 {token_stats, compact, observability}
```

---

## 4. 会话持久化

```
agent/.memory/
├── memory.md              ← 长期记忆（跨会话）
├── user.md                ← 用户偏好（跨会话）
├── summaries/             ← 每日摘要（跨会话）
└── sessions/
    └── <session_id>/
        ├── history.jsonl  ← 树结构对话历史
        ├── tokens.jsonl   ← Token 消耗日志
        └── trace.jsonl    ← 可观测性日志
```

- 每条消息追加时增量写入 JSONL
- 压缩时全量重写（DFS 序，保证 parent 在 child 前）
- `restore_tree()` 自动检测旧格式并迁移
- `restore_session=True` 恢复最近会话

---

## 5. 压缩流程

Token 超阈值（默认 `max_context * 35%` = 70K tokens）时触发：

```
压缩前: a1 → q2 → a2 → tool → q3 → a3
          ↑ first_kept
压缩后: a1 → COMPACT → q2 → a2 → tool → q3 → a3
build_context: [compaction_summary] + [q2, a2, tool, q3, a3]
```

提取内容分发：
- 偏好 → user.md（去重追加）
- 事实 → memory.md（去重追加）
- 摘要 → summaries/YYYY-MM-DD.md

---

## 6. 事件系统

EventBus 支持精确匹配和通配符（`tool:*` 匹配 `tool:before` + `tool:after`）。

| 事件 | 触发位置 | data | 说明 |
|------|---------|------|------|
| `message:received` | agent.py | `{text}` | 请求起点 |
| `turn:start` | runner.py | `{turn}` | 每轮开始，触发压缩检查 |
| `turn:end` | runner.py | `{text}` | 本轮无工具调用 |
| `tool:before` | executor.py | `{name, args}` | 返回 `{block, reason}` 可拦截 |
| `tool:after` | executor.py | `{name, result}` | 返回字符串可替换结果 |
| `context:high` | agent.py | event | 压缩即将触发 |
| `session:end` | agent.py | `{}` | 会话关闭 |

Observability 订阅全部事件自动记录。

---

## 7. MCP 支持

通过 `mcp.json` 配置文件连接外部 MCP 服务器（stdio 传输）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    }
  }
}
```

- 启动时自动连接并发现工具
- 工具命名：`mcp__<server>__<tool>`
- 连接失败跳过并警告，不影响内置工具
- 使用 `asyncio.run_coroutine_threadsafe` 桥接异步 SDK
- 仅主 Agent 持有 MCP 工具，子代理不挂载

---

## 8. LLM 重试

Runner 内置指数退避重试：

| 参数 | 值 |
|------|-----|
| 可重试错误 | APIConnectionError, RateLimitError, APITimeoutError, InternalServerError |
| 最多重试 | 3 次 |
| 退避策略 | 2s → 4s → 放弃 |
| 不可重试 | AuthenticationError, BadRequestError（直接抛出） |

重试前回滚 history 快照，丢弃本轮部分写入。重试过程中产出 tool_status_chunk 通知用户。

---

## 9. 可观测性

每次请求自动生成 trace ID（8 位 hex），记录到 `trace.jsonl`：

```json
{"ts":"...","level":"INFO","trace":"a32d8df6","event":"request:start","data":{"message":"hello"}}
{"ts":"...","level":"INFO","trace":"a32d8df6","event":"turn:start","data":{"turn":1}}
{"ts":"...","level":"INFO","trace":"a32d8df6","event":"tool:start","data":{"tool":"bash_tool"}}
{"ts":"...","level":"INFO","trace":"a32d8df6","event":"tool:end","data":{"tool":"bash_tool","duration_ms":15,"ok":true}}
{"ts":"...","level":"INFO","trace":"a32d8df6","event":"request:summary","data":{"turns":1,"tool_calls":1,"tokens_in":100,"tokens_out":50,...}}
```

通过 EventBus 订阅实现，零侵入。`summary()` 返回当前请求的汇总指标。

---

## 10. CLI / TUI / Web

### CLI

内置命令：`/help` `/config` `/model` `/thinking` `/turns` `/tree` `/fork` `/back` `/clear`

特性：readline 历史、`-p` print 模式、`-r` 恢复会话、`--thinking` 控制、`@file` 引用、管道输入。

### TUI

基于 prompt_toolkit，对话气泡 + 快捷键：

| 键 | 功能 |
|----|------|
| Ctrl+Q | 退出 |
| Ctrl+F | 分叉 |
| Ctrl+B | 返回 |
| Ctrl+T | 分支树 |
| Ctrl+E | 展开/折叠工具 |

### Web UI

FastAPI + WebSocket，单文件 HTML 前端。

API 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/sessions` | 会话列表 |
| POST | `/api/sessions` | 新建会话 |
| GET | `/api/sessions/{id}/settings` | 获取配置 |
| POST | `/api/sessions/{id}/settings` | 修改模型/思考/轮数 |
| POST | `/api/upload` | 文件上传 |
| POST | `/api/upload/resolve` | @引用解析 |
| WS | `/ws` | 流式对话 |

前端功能：模型切换、文件上传/拖放、@引用解析、分支树、Markdown 渲染。

---

## 11. 测试

552 个测试，全绿。

```bash
python -m unittest discover tests
```

| 类型 | 覆盖 |
|------|------|
| 单元测试 | 21 个模块，523 用例 |
| 集成测试 | Agent 完整流程、持久化、树导航、压缩、重试，14 用例 |
| observability | 日志、trace、指标，15 用例 |
