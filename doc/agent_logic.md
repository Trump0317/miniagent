# miniagent 核心逻辑文档

> 面向开发者和 AI 编码助手维护者，详细描述系统的设计决策、模块职责、数据流和扩展方式。

## 目录

1. [架构总览](#1-架构总览)
2. [关键组件详解](#2-关键组件详解)
3. [数据流](#3-数据流)
4. [会话持久化](#4-会话持久化)
5. [压缩流程](#5-压缩流程)
6. [事件系统](#6-事件系统)
7. [CLI 命令](#7-cli-命令)
8. [Web UI](#8-web-ui)
9. [测试](#9-测试)

---

## 1. 架构总览

```
agent.py                    ← CLI 入口（交互 / print 模式 + 内置命令）
├── agent/
│   ├── system_prompt.md   ← 系统提示词模板（外部文件，可独立修改）
│   ├── ai/                 # AI 层 — Provider 配置 + LLM 调用 + 上下文加载
│   ├── cli/                # CLI 外壳 — 交互/print 模式
│   │   ├── app.py          #   应用程序（readline、OutputHandler 智能输出、多行输入）
│   │   └── helpers.py      #   handle_tree / handle_fork / handle_back
│   ├── tui/                # TUI 终端界面 — prompt_toolkit, pi-tui 风格
│   │   └── app.py          #   对话气泡、流式输出、鼠标滚轮翻页、折叠展开
│   │   ├── config.py       #   AppConfig: 多 Provider 配置 + 会话隔离
│   │   ├── llm.py          #   LLMClient: 流式调用封装
│   │   └── context.py      #   load_context_files: AGENTS.md 加载
│   ├── core/               # 核心引擎 — 组装 + 运行 + 存储 + 压缩
│   │   ├── agent.py        #   Agent: 纯装配层 (~130 行)
│   │   ├── runner.py       #   AgentRunner: think-act 循环，产出统一 chunk
│   │   ├── session_tree.py #   SessionTree: 树状会话（分叉/导航/压缩）
│   │   ├── system_prompt.py#   SystemPrompt: 从文件加载模板 + 动态注入
│   │   ├── compaction.py   #   CompactionService: 压缩编排
│   │   ├── events.py       #   EventBus: 发布/订阅事件总线
│   │   ├── memory.py       #   AgentMemory: 纯存储层（树为唯一数据源）
│   │   ├── tracker.py      #   TokenTracker: Token 消耗统计
│   │   ├── prompts.py      #   PromptLoader: 命令模板加载
│   │   └── cli_helpers.py  #   handle_tree/fork/back
│   ├── tools/              # 工具层 — 扁平布局，每工具一个文件
│   │   ├── base.py         #   Tool 基类 + Schema 瘦身
│   │   ├── registry.py     #   ToolRegistry: 工具注册表
│   │   ├── executor.py     #   ToolExecutor: 串行/并行调度
│   │   ├── bash.py         #   终端命令
│   │   ├── file_read.py    #   文件读取
│   │   ├── file_write.py   #   文件写入
│   │   ├── file_edit.py    #   文件编辑
│   │   ├── web_fetch.py    #   网页抓取
│   │   ├── web_search.py   #   网络搜索
│   │   ├── skill.py        #   技能加载
│   │   ├── todo.py         #   待办管理
│   │   └── subagent.py     #   SubagentRunner + SubagentTool
│   ├── subagent/           # 子代理定义（Markdown + YAML）
│   ├── prompts/            # 命令模板
│   ├── web/                # Web UI — FastAPI + WebSocket
│   │   ├── server.py       #   FastAPI 服务 + REST + WS 端点
│   │   ├── session.py      #   SessionManager 多会话管理
│   │   └── static/
│   │       └── index.html  #   聊天界面
│   └── tests/              # 503 个单元测试
```

### 设计原则

1. **树状会话** — SessionTree 管理对话分支，分叉不丢数据，压缩插入 COMPACT 节点
2. **无列表双写** — AgentMemory 以 SessionTree 为唯一数据源，`history` 是 `tree.build_context()` 的实时计算
3. **事件驱动** — EventBus 解耦各组件，通过 `turn:start` / `context:high` / `tool:before` / `tool:after` 等事件通信
4. **存储与压缩分离** — AgentMemory 只做纯 I/O + 树操作，CompactionService 编排全部压缩流程
5. **扁平工具** — 每个工具一个 .py 文件
6. **统一 chunk** — Agent 产出 `{"type":"text/reasoning/tool_status/tool_result/done"}` 格式，CLI/TUI/Web 按类型消费

---

## 2. 关键组件详解

### 2.1 AppConfig (`agent/ai/config.py`)

集中管理所有可配置项，支持多 Provider 和会话隔离。

- **多 Provider 支持**：`PROVIDER_PRESETS` 定义 `deepseek` / `openai` / `custom` 三组预设
- **自动检测**：`from_env()` 按 `DEEPSEEK_API_KEY → OPENAI_API_KEY → API_KEY` 顺序检测
- **会话隔离**：每个会话独立 `sessions/<session_id>/` 目录
- **配置项**：`provider` / `model` / `max_turns` / `compact_threshold` / `subagent_model` 等

### 2.2 LLMClient (`agent/ai/llm.py`)

封装 OpenAI 兼容的流式聊天调用，产出结构化 chunk：

```python
# chunk 类型:
# {"type": "content", "text": "..."}
# {"type": "reasoning", "text": "..."}
# {"type": "tool_call", "index": 0, "id": "...", "name": "...", "arguments": "..."}
# {"type": "usage", "input": 100, "output": 50}
```

支持 `--thinking` 控制推理强度（off / minimal / low / medium / high / xhigh），映射到 `reasoning_effort` 参数。

### 2.3 AgentRunner (`agent/core/runner.py`)

执行 think-act 循环，产出统一 chunk 格式：

```python
{"type": "text", "content": "..."}        # LLM 正文
{"type": "reasoning", "content": "..."}   # 推理/思考内容（DeepSeek thinking）
{"type": "tool_status", "content": "..."}  # 工具执行状态
{"type": "tool_result", "id": "...", "result": "..."}  # 工具最终结果
{"type": "done"}                           # 本轮结束
```

- 自动处理 DeepSeek 流式 tool_call arguments 的分片拼接
- 并行/串行工具调度委托给 ToolExecutor
- 达到 max_turns 后自动熔断

### 2.4 SessionTree (`agent/core/session_tree.py`)

树状会话结构，替代线性列表：

- `SessionEntry`: 树节点（id/parent_id/type/role/content/metadata/timestamp）
- `type` 有两种: `"message"`（普通消息）、`"compaction"`（压缩节点）
- 核心操作: `append()` / `compact()` / `fork(id)` / `navigate(id)` / `build_context()`
- 压缩语义: 插入 COMPACT 节点 + 重排 parent_id，`build_context()` 遇 COMPACT 自动跳过

### 2.5 CompactionService (`agent/core/compaction.py`)

单一模块覆盖完整压缩流程：

1. `should_compact()`: 检查 token 是否超过阈值
2. `compact()`: 编排一次完整压缩
   - `_find_cut_point()`: token-aware 切点
   - `_extract()`: LLM 提取 summary / preferences / facts
   - `_dispatch_to_memory()`: 分发到三层记忆
   - `memory.compress_tree()`: 树压缩
   - `prompt.build(data)`: 重建提示词

### 2.6 SystemPrompt (`agent/core/system_prompt.py`)

从 `agent/system_prompt.md` 加载模板，`build()` 时通过 `str.format()` 注入动态内容：

- `{context_files}` — 项目上下文
- `{skills}` — 可用技能列表
- `{agents}` — 可用子代理
- `{commands}` — 可用命令
- `{memory}` — 长期记忆
- `{preferences}` — 用户偏好
- `{compaction}` — 压缩摘要（可选）

模板包含行为准则、任务规划、工具策略、错误处理和记忆利用 5 个章节。

---

## 3. 数据流

### 3.1 请求处理完整流程

```
用户输入
  → agent.py: 内置命令直接处理 (/help /session /clear /tree /fork /back)
  → agent.py: _expand_command() → /command 展开为模板
  → Agent.process(message)
     → EventBus.emit("message:received")
     → memory.append_message(user msg) → 树追加 + JSONL 增量写
     → Agent._build_context() → [system] + memory.history 快照
     → AgentRunner.step(context)              # think-act 循环
        → EventBus.emit("turn:start")
        → LLMClient.stream() → chunk 生成器
        → tool_call arguments 逐字符拼接
        → 无工具调用 → yield {"type":"done"} → 返回
        → ToolExecutor.execute()
           → 产出统一 chunk: text / tool_status / tool_result
        → context.append(assistant/tool msg)
        → 继续循环
     → 扫描 context 增量 → memory.append_message(msg)
  → Agent.shutdown()
     → CompactionService.compact() → LLM 提取 → 分发 → 树压缩 → 重建提示词
```

### 3.2 Chunk 流转

```
LLMClient.stream()
  → {"type":"content","text":"..."}  → Runner → {"type":"text","content":"..."}
  → {"type":"reasoning","text":"..."} → Runner → 累积到 assistant 消息
  → {"type":"tool_call",...}          → Runner → 触发工具执行

ToolExecutor.execute()
  → {"type":"tool_status","content":"[执行工具: bash...]"}  → 透传
  → {"type":"text","content":"file.py\n"}                  → 透传
  → {"type":"tool_result","id":"...","result":"..."}       → Runner 内部消费

CLI: _output_chunk() → type∈{text,tool_status} → print(content)
Web: ws.send_json(chunk) → 前端按 type 渲染
```

---

## 4. 会话持久化

- 会话隔离: 每会话独立 `sessions/<ts>/history.jsonl` + `tokens.jsonl`
- 每条消息追加时增量写入 JSONL
- 压缩时全量重写 JSONL（DFS 序）
- 启动时 `restore_tree()` 自动检测旧格式并迁移

```
agent/.memory/
├── memory.md              ← 长期记忆（跨会话）
├── user.md                ← 用户偏好（跨会话）
├── summaries/             ← 每日摘要（跨会话）
└── sessions/
    └── <session_id>/      ← 会话私有目录
        ├── history.jsonl
        └── tokens.jsonl
```

---

## 5. 压缩流程

（略，参见 AGENTS.md 压缩语义章节）

---

## 6. 事件系统

| 事件 | 触发位置 | 用途 |
|------|----------|------|
| `message:received` | `Agent.process()` | 新消息通知 |
| `turn:start` | `AgentRunner.step()` | 每轮开始，触发 token 阈值检查 |
| `context:high` | `turn:start` 回调 | 超过阈值时执行压缩 |
| `tool:before` | `ToolExecutor.execute()` | 工具拦截/修改 |
| `tool:after` | `ToolExecutor.execute()` | 工具结果修改 |
| `session:end` | `Agent.shutdown()` | 会话结束通知 |

---

## 7. CLI 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示所有内置命令和模板命令 |
| `/session` | 显示会话 ID、模型、Token 用量、存储路径 |
| `/clear` | 清屏 |
| `/tree` | 显示会话分支树可视化 |
| `/fork [n]` | 分叉到第 n 条用户消息之前 |
| `/back` | 返回分叉前的位置 |
| `/scout <query>` | 展开为 scout 子代理侦查任务 |
| `/review <query>` | 展开为 reviewer 子代理审查任务 |

其他特性：

- **readline 历史**：↑↓回溯命令历史，退出时持久化到 `~/.miniagent/.history`
- **`-r` / `--restore`**：启动时恢复最近会话
- **`-p`**：print 模式，非交互执行
- **`--thinking`**：控制推理强度

---

## 9. Web UI

基于 FastAPI + WebSocket 的浏览器端交互界面。

### 启动

```bash
python -m agent.web.server
# 打开 http://127.0.0.1:8000
```

### 组件

| 组件 | 文件 | 职责 |
|------|------|------|
| FastAPI 服务 | `server.py` | REST API + WebSocket 端点 |
| 会话管理 | `session.py` | `SessionManager` — 多会话 Agent 实例生命周期 |
| 聊天界面 | `static/index.html` | Markdown 渲染、分支树、多会话 |

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天界面 |
| GET | `/api/sessions` | 列出所有会话 |
| POST | `/api/sessions` | 创建新会话 |
| GET | `/api/sessions/{id}/history` | 获取会话历史 |
| GET | `/api/sessions/{id}/tree` | 获取分支树 |
| POST | `/api/sessions/{id}/fork?target_id=` | 分叉 |
| POST | `/api/sessions/{id}/back` | 返回分叉前 |
| WS | `/ws` | WebSocket 流式对话 |

### WebSocket 协议

```
发送: {"type":"message","session_id":"xxx","content":"你好"}
      {"type":"switch","session_id":"xxx"}

接收: {"type":"text","content":"..."}
      {"type":"tool_status","content":"[执行工具: bash...]"}
      {"type":"done"}
```

### 前端特性

- Markdown 渲染（marked.js）：代码块、列表、表格
- 代码块悬停复制按钮
- 工具状态 ⚙ 旋转动画 / ✓ 完成图标
- 工具输出可折叠面板
- 侧栏多会话管理（新建/切换/删除）
- 分支树可视化 + 点击分叉 + ← 返回
- 会话历史从后端 API 加载，刷新不丢失
- Inter 字体、毛玻璃风格、消息动画

---

## 11. 测试

测试采用先单元后集成的策略，每个模块经过三重审查（自审 → Subagent 审 → 人工审）后提交。

### 覆盖状况

截至 2026-05-30，已覆盖 **20 个模块，共 503 个单元测试**：

| 模块 | 测试文件 | 测试数 |
|------|----------|--------|
| EventBus | `test_events.py` | 28 |
| SessionTree | `test_session_tree.py` | 56 |
| AgentMemory | `test_memory.py` | 59 |
| TokenTracker | `test_tracker.py` | 27 |
| PromptLoader | `test_prompts.py` | 21 |
| SystemPrompt | `test_system_prompt.py` | 17 |
| CompactionService | `test_compaction.py` | 29 |
| AgentRunner | `test_runner.py` | 16 |
| Tool 基类 + @tool 装饰器 | `test_tool_base.py` | 28 |
| ToolRegistry | `test_registry.py` | 24 |
| FileReadTool | `test_file_read.py` | 7 |
| FileWriteTool | `test_file_write.py` | 7 |
| FileEditTool | `test_file_edit.py` | 16 |
| SkillsLoader + SkillTool | `test_skill.py` | 16 |
| TodoWriteTool | `test_todo.py` | 30 |
| BashTool | `test_bash.py` | 20 |
| ToolExecutor | `test_executor.py` | 19 |
| WebFetchTool | `test_web_fetch.py` | 7 |
| WebSearchTool | `test_web_search.py` | 7 |
| AgentLoader + SubagentTool | `test_subagent.py` | 45 |

### 运行

```bash
# 全部测试
python -m unittest discover tests

# 单个模块
python -m unittest tests.test_events
```

### 审查流程

```
编写测试 → 自审 → Subagent(reviewer) 审 → 人工审 → 提交
```

每个审查环节发现问题会立即修复，审核通过后才进行下一模块的编写。
