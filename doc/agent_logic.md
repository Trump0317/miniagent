# Agent 逻辑说明文档

本文档详细介绍了 explore 分支重构后的架构设计与核心逻辑。

## 1. 核心架构概述

项目采用 **ReAct 模式 + 事件驱动架构**，将 AI 调用、核心引擎和工具系统三层解耦：

```
agent/
├── __init__.py         # 公共导出
├── ai/                 # AI 层 — Provider + LLM 调用 + 上下文
│   ├── config.py       #   AppConfig: 多 Provider 智能检测
│   ├── llm.py          #   LLMClient: 流式调用封装
│   └── context.py      #   load_context_files: AGENTS.md 加载
├── core/               # 核心引擎 — 组装 + 运行 + 存储 + 压缩
│   ├── agent.py        #   Agent: 组件组装入口 + 压缩编排
│   ├── runner.py       #   AgentRunner: think-act 循环
│   ├── events.py       #   EventBus: 发布/订阅事件总线
│   ├── memory.py       #   AgentMemory: 纯存储层（文件 I/O）
│   ├── compactor.py    #   Compactor: LLM 提取摘要/偏好/事实
│   ├── tracker.py      #   TokenTracker: Token 消耗统计
│   └── prompts.py      #   PromptLoader: 命令模板加载
└── tools/              # 工具层 — 扁平布局，每工具一个文件
    ├── base.py         #   Tool 基类 + JSON Schema 瘦身
    ├── registry.py     #   ToolRegistry: 工具注册表
    ├── executor.py     #   ToolExecutor: 串行/并行调度
    ├── bash.py         #   终端命令
    ├── file_read.py    #   文件读取
    ├── file_write.py   #   文件写入
    ├── file_edit.py    #   文件编辑（模糊匹配）
    ├── web_fetch.py    #   网页抓取
    ├── web_search.py   #   网络搜索
    ├── skill.py        #   技能加载
    ├── todo.py         #   待办管理
    └── subagent.py     #   子代理（单/并行/链式）
```

### 设计原则

1. **事件驱动** — `EventBus` 解耦各组件，通过事件（`turn:start`、`context:high`、`history:appended` 等）通信，组件只 emit 不关心谁来消费
2. **存储与压缩分离** — `AgentMemory` 只做纯文件 I/O，`Compactor` 只做 LLM 提取，互不依赖
3. **零中间层** — 无 `Conversation` / `Hooks` 抽象，`Agent` 直接编排 `Runner` + `Memory` + `Compactor`
4. **扁平工具** — 每个工具一个 `.py` 文件，不再嵌套子目录；`base.py` / `registry.py` / `executor.py` 独立模块

### 核心文件对照（重构前 → 重构后）

| 重构前 | 重构后 | 变化 |
|--------|--------|------|
| `config.py` (115 行) | `ai/config.py` (141 行) | 集中配置 + Provider 预设 |
| `conversation.py` (80 行) | **已删除** | 职责由 Memory + Compactor + EventBus 分担 |
| `loop.py` (140 行) | `core/agent.py` (254 行) | Agent 组装 + 压缩编排 + 事件注册 |
| `runner.py` (265 行) | `core/runner.py` (119 行) | 纯 think-act 循环，不持有状态 |
| `memory.py` (265 行) | `core/memory.py` (269 行) | 纯存储 + 去重 |
| `hooks.py` (65 行) | **已删除** | 事件钩子改用 EventBus 事件 |
| — | `core/events.py` (94 行) | **新增**: 事件总线 |
| — | `core/compactor.py` (104 行) | **新增**: LLM 压缩提取 |
| `prompts.py` (90 行) | `core/prompts.py` (97 行) | 无大变化 |
| `tokentracker.py` (70 行) | `core/tracker.py` (69 行) | 新增 `reset_session()` |
| — | `ai/llm.py` (91 行) | **新增**: LLM 流式客户端封装 |
| — | `ai/context.py` (57 行) | **新增**: 上下文文件加载 |
| `tools/BashTool/...` | `tools/bash.py` | 扁平化 |
| — | `tools/base.py` | 新增 Schema 瘦身 |
| — | `tools/executor.py` | **新增**: 工具执行器独立 |

---

## 2. 关键组件详解

### 2.1 AppConfig (`agent/ai/config.py`)

集中管理所有可配置项，替代之前散落各处的硬编码参数。

- **多 Provider 支持**：`PROVIDER_PRESETS` 定义 `deepseek` / `openai` / `custom` 三组预设
- **智能检测**：`from_env()` 自动根据环境变量（`DEEPSEEK_API_KEY` → `OPENAI_API_KEY` → `API_KEY`）选择 Provider
- **可覆盖**：所有参数都可通过 `overrides` 传入

```python
# 自动检测
config = AppConfig.from_env()

# 显式指定
config = AppConfig.from_env(provider="openai", model="gpt-4o")

# 完整覆盖
config = AppConfig(api_key="sk-xxx", api_base_url="https://api.openai.com/v1", model="gpt-4o")
```

关键配置项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `provider` | deepseek | Provider 名称 |
| `model` | 根据 Provider | 默认模型 |
| `max_tokens` | 20_000 | LLM 单次最大输出 |
| `max_context` | 200_000 | 上下文窗口上限 |
| `compact_threshold` | 0.35 | 上下文使用率阈值，触发压缩 |
| `restore_session` | True | 启动时恢复上次会话 |
| `subagent_model` | 同 model | 子代理默认模型 |
| `subagent_max_turns` | 15 | 子代理最大轮数 |

### 2.2 LLMClient (`agent/ai/llm.py`)

封装 OpenAI 兼容的流式聊天调用，产出结构化 chunk：

```python
# chunk 类型:
{"type": "content",     "text": "..."}           # 正文
{"type": "reasoning",   "text": "..."}           # 思维链 (DeepSeek R1)
{"type": "tool_call",   "index": 0, "id": "...", "name": "...", "arguments": "..."}
{"type": "usage",       "input": 100, "output": 50, "cache_hit": 0, "cache_miss": 0}
```

支持 `thinking` 参数控制推理努力程度（`off` / `minimal` / `low` / `medium` / `high` / `xhigh`），映射到 DeepSeek 的 `reasoning_effort`。

### 2.3 Agent (`agent/core/agent.py`)

核心组装器，创建并连接所有组件：

```
Agent.__init__
├── AppConfig          — 配置
├── EventBus           — 事件总线
├── AgentMemory        — 纯存储（文件 I/O）
├── Compactor          — 压缩器（LLM 提取）
├── TokenTracker       — Token 统计
├── SkillsLoader       — 技能加载
├── AgentLoader        — 子代理定义加载
├── PromptLoader       — 命令模板加载
├── ToolRegistry       — 工具注册（主 + 子代理）
├── LLMClient          — LLM 客户端
├── ToolExecutor       — 工具执行器
├── AgentRunner        — 执行引擎
└── _setup_events()    — 注册事件监听
```

#### 压缩编排 (`_do_compact()`)

压缩流程是 `Agent` 的核心职责：

```
_do_compact()
├── compactor.compact(history)          → 提取 summary / preferences / facts
├── memory.append_summary(...)          → 写入 summaries/*.md
├── memory.add_user(p)                  → 写入 user.md（内置去重）
├── memory.add_memory(f)                → 写入 memory.md（内置去重，返回是否新增）
├── _trim_history(data)                 → 截断 self.history
│   ├── _rebuild_system_prompt_with_summary()  → 重建系统提示词
│   └── history.clear() + extend(系统 + 最近 2 轮)
└── _sync_history_file()                → 同步 history.jsonl
```

#### 历史截断 (`_trim_history()`)

压缩后将 `self.history` 截断为：
1. **系统提示词** — 包含压缩摘要、最新记忆、用户偏好
2. **最近 2 轮用户对话** — 保留上下文连续性

这样既释放了窗口空间，又保留了必要上下文。

#### 动态系统提示词 (`_rebuild_system_prompt_with_summary()`)

压缩后重建系统提示词，包含：
- 基础角色描述
- 项目上下文文件（AGENTS.md）
- 可用技能 / 子代理 / 命令列表
- 长期记忆最近摘要（`memory.md` + `summaries/`）
- 用户偏好（`USER.md`）
- **本次压缩摘要**（critical / decision / issue）

#### 事件绑定 (`_setup_events()`)

| 事件 | 触发时机 | 处理 |
|------|----------|------|
| `context:high` | 上下文达到阈值 | 执行 `_do_compact()` |
| `turn:start` | 每轮 LLM 调用前 | 检查 `_should_compact()`，必要时 emit `context:high` |
| `history:appended` | Runner 写入历史消息后 | 持久化到 JSONL |

### 2.4 EventBus (`agent/core/events.py`)

轻量级发布/订阅事件总线，替代旧的 `hooks.py` 回调模式。

特性：
- **通配符匹配** — `"tool:*"` 匹配 `"tool:before"` / `"tool:after"`
- **一次性监听** — `once()` 注册只触发一次的监听器
- **返回值收集** — `emit()` 返回所有监听器返回值，实现拦截器模式
- **装饰器友好** — `@bus.on("event:name")` 直接注册

```python
bus = EventBus()

# 注册
@bus.on("tool:before")
def intercept(event):
    if event.data["name"] == "bash":
        return {"block": True, "reason": "只读模式"}

# 一次性
bus.once("startup", lambda e: print("ready"))

# 发布
results = bus.emit("tool:before", {"name": "bash", "args": {}})
```

### 2.5 AgentRunner (`agent/core/runner.py`)

纯粹的执行引擎，**不持有会话状态**，`history` 作为参数传入。

关键流程 (`step` 方法)：

```
for turn in range(max_turns):
    1. emit("turn:start")
    2. LLM 流式调用 → 逐 chunk 产出 text + tool_calls
    3. 写入 assistant 消息到 history
    4. emit("history:appended", {"message": assistant_msg})
    5. 无 tool_calls → emit("turn:end") → return
    6. tool_executor.execute(tool_calls) → 串行/并行调度
    7. 写入 tool 结果到 history
    8. emit("history:appended", {"message": tool_msg})
    9. 循环
```

**API 兼容处理**：当 assistant message 的 `content` 为 None 且无 `tool_calls` 时，用 `reasoning_content` 或空字符串填充，避免 API 校验失败。

### 2.6 AgentMemory (`agent/core/memory.py`)

纯存储层，**不调用 LLM，不关心压缩逻辑**。

三层记忆结构：

```
agent/.memory/
├── history.jsonl     # 对话历史（JSONL 持久化，只存 assistant/tool 消息）
├── memory.md         # 长期记忆（追加事实，系统提示词引用）
├── summaries/        # 每日摘要（按日期 YYYY-MM-DD.md）
└── user.md           # 用户偏好（列表，系统提示词引用）
```

关键方法：

| 方法 | 说明 |
|------|------|
| `append_history(msg, persist=True)` | 追加到内存 + JSONL |
| `persist_message(msg)` | 只持久化到 JSONL（事件触发用） |
| `restore_history(max=50)` | 从 JSONL 恢复上次会话 |
| `_rewrite_history(entries)` | 重写 JSONL（压缩后同步） |
| `add_memory(fact)` → `bool` | 追加事实，自动去重，返回是否新增 |
| `add_user(pref)` | 追加偏好，自动去重 |
| `get_existing_facts()` | 读取已有事实集合 |
| `get_existing_preferences()` | 读取已有偏好集合 |
| `brief_context()` | 获取系统提示词所需的最近背景 |

**去重机制**：`add_memory()` 和 `add_user()` 写入前先检查 `get_existing_facts()` / `get_existing_preferences()` 集合，避免重复存储。

**会话恢复**：只恢复 `assistant` 和 `tool` 消息（不恢复 `user`），通过插入上下文分隔提示消息让 LLM 区分历史与待回答内容。

### 2.7 Compactor (`agent/core/compactor.py`)

利用 LLM 从对话历史中提取关键信息。**不操作文件**，只返回结构化结果。

```python
result = compactor.compact(history)
# {
#   "summary": {"critical": "关键事件", "decision": "决策", "issue": "问题"},
#   "preferences": ["用户偏好1", "用户偏好2"],
#   "facts": ["核心事实1", "核心事实2"]
# }
```

提取提示词约束：
- 只分析最近 `k` 条消息（默认 10）
- 总字数 < 150 字，每条偏好/事实 < 30 字
- 不提取系统已知静态信息（工具数量、模型名称等）
- 记录压缩的 token 消耗（`_last_usage`）

### 2.8 TokenTracker (`agent/core/tracker.py`)

Token 消耗统计，JSONL 格式持久化。

| 方法 | 说明 |
|------|------|
| `record(model, usage)` | 记录单次调用的 token 用量 |
| `reset_session()` | 清空日志，开始新会话统计 |
| `last_input_tokens()` | 获取上次调用的输入 token 数 |
| `should_compact(max, threshold)` | 判断是否需要压缩 |
| `stats_by_model()` | 按模型聚合统计 |
| `stats_by_date()` | 按日期聚合统计 |

### 2.9 参数上下文文件 (`agent/ai/context.py`)

`load_context_files()` 从目录链自动加载上下文文件：

1. 用户全局 `~/.miniagent/AGENTS.md`
2. 从根到工作目录沿途的 `AGENTS.md` / `CLAUDE.md`

拼接后注入系统提示词，CLI 可通过 `-nc` / `--no-context-files` 禁用。

---

## 3. 工具系统

### 3.1 Tool 基类 (`agent/tools/base.py`)

```python
@tool(name="my_tool", description="...", parameters=MyArgs)
class MyTool(Tool):
    parallel_safe: bool = True          # 是否可并发
    supports_streaming: bool = False    # 是否支持流式执行

    def execute(self, **kwargs) -> str:
        ...

    def stream_execute(self, **kwargs) -> Generator[str]:
        ...
```

**JSON Schema 瘦身** (`_minify_schema()`)：剔除 LLM 不需要的冗余字段（`title` / `default` / `additionalProperties`），简化 `anyOf[{type:X}, {type:null}]` 结构，缩减 tool schema 的 token 消耗。

### 3.2 ToolRegistry (`agent/tools/registry.py`)

管理所有工具实例，提供注册、查询、schema 生成和调用功能。

```python
registry = ToolRegistry()
registry.register(BashTool())
registry.register(FileReadTool())

schemas = registry.get_tool_schemas()  # 缓存优化
result = registry.call_tool("bash_tool", {"command": "ls"})
```

### 3.3 ToolExecutor (`agent/tools/executor.py`)

负责工具的实际执行策略：

1. **串行策略** — 单个工具，或存在非 `parallel_safe` 工具时降级串行
2. **并行策略** — 多工具全部 `parallel_safe` 时，使用 `ThreadPoolExecutor` 并发（最多 8 并发）
3. **事件拦截** — 执行前后 emit `tool:before` / `tool:after` 事件，监听器可 `{"block": True}` 拦截
4. **结果截断** — 自动截断超过 50KB 或 2000 行的输出

### 3.4 工具一览

| 工具 | 文件 | 特性 |
|------|------|------|
| **BashTool** | `bash.py` | 安全护栏 + 流式输出（Popen 逐行） |
| **FileReadTool** | `file_read.py` | 文件读取，支持 offset/limit |
| **FileWriteTool** | `file_write.py` | 创建或覆盖文件 |
| **FileEditTool** | `file_edit.py` | 精确匹配 + 忽略缩进的模糊匹配 |
| **WebFetchTool** | `web_fetch.py` | 网页抓取 |
| **WebSearchTool** | `web_search.py` | 网络搜索 |
| **SkillTool** | `skill.py` | 加载预定义技能 |
| **TodoWriteTool** | `todo.py` | 待办列表 CRUD，`parallel_safe=False` |
| **SubagentTool** | `subagent.py` | 单/并行/链式子代理 |

---

## 4. 子代理系统 (`agent/tools/subagent.py`)

### 4.1 AgentLoader

扫描 `agent/subagent/*.md`，解析 Markdown + YAML frontmatter 为 `AgentDefinition`：

```markdown
---
name: scout
description: 快速侦查代码库
tools: bash_tool, file_read_tool
model: deepseek-v4-flash
max_turns: 10
---
你是代码库侦查员...
```

### 4.2 SubagentTool

支持三种调用模式：

```
单模式:   subagent_tool(task="分析代码", agent="scout")
并行模式: subagent_tool(tasks=[{task:"A"}, {task:"B"}, {task:"C"}])
链式模式: subagent_tool(chain=[{task:"Step1"}, {task:"Step2+{previous}"}])
```

每个子代理拥有：
- 独立的 `history` 列表和 `TokenTracker`
- 独立的 `ToolRegistry`（或根据定义过滤）
- 独立的 `AgentRunner` 实例
- 子代理 token 消耗汇总到父 tracker

---

## 5. 工作流示意

```
用户: /scout agent/core/agent.py
        │
        ▼
agent.py._expand_command()
   → PromptLoader.resolve("scout", "agent/core/agent.py")
   → "先用 scout 子代理快速侦查 agent/core/agent.py..."
        │
        ▼
Agent.process(message)
   ├── bus.emit("message:received")
   ├── memory.append_history({"role": "user", ...})
   └── runner.step(history)
         │
         ▼
AgentRunner.step(history)
   ┌── bus.emit("turn:start")
   │     └── Agent 检查 → should_compact? → emit("context:high") → _do_compact()
   ├── LLM 流式调用 ──→ yield 文本
   ├── 解析 tool_calls: [{name: "subagent_tool", args: {agent: "scout", task: "..."}}]
   ├── bus.emit("history:appended", {assistant_msg})
   │     └── Agent 持久化到 JSONL
   ├── tool_executor.execute(tool_calls)
   │     ├── bus.emit("tool:before") → 可拦截
   │     ├── SubagentTool.execute()
   │     │     ├── AgentLoader.get("scout") → AgentDefinition
   │     │     └── 独立 AgentRunner(registry=filtered, model=deepseek-v4-flash)
   │     └── bus.emit("tool:after") → 可修改结果
   ├── bus.emit("history:appended", {tool_msg})
   │     └── Agent 持久化到 JSONL
   └── 循环直到无 tool_calls
         │
         ▼
   产出响应文本
         │
         ▼
Agent.shutdown()
   ├── stats = tracker.stats_by_model()
   ├── _do_compact()          # 最终压缩
   └── _sync_history_file()   # 同步 JSONL
```

---

## 6. 事件系统全景

| 事件 | 来源 | 消费者 | 用途 |
|------|------|--------|------|
| `message:received` | `Agent.process()` | 外部监听 | 记录用户输入 |
| `turn:start` | `Runner.step()` 开始 | `Agent._setup_events()` | 每轮检查是否需要压缩 |
| `turn:end` | `Runner.step()` 结束 | 外部监听 | 轮次统计 |
| `context:high` | `Agent._setup_events()` | `Agent._setup_events()` | 触发压缩 |
| `history:appended` | `Runner.step()` 写入后 | `Agent._setup_events()` | 持久化到 JSONL |
| `tool:before` | `ToolExecutor` 执行前 | 外部监听（拦截器） | 拦截/修改工具调用 |
| `tool:after` | `ToolExecutor` 执行后 | 外部监听 | 修改工具结果 |
| `session:end` | `Agent.shutdown()` | 外部监听 | 会话结束通知 |
