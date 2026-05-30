# Agent 逻辑说明文档

本文档详细介绍重构后的架构设计与核心逻辑。

## 1. 核心架构概述

项目采用 **ReAct 模式 + 事件驱动架构**，将 AI 调用、核心引擎和工具系统三层解耦：

```
agent/
├── __init__.py         # 公共导出
├── ai/                 # AI 层 — Provider + LLM 调用 + 上下文
│   ├── config.py       #   AppConfig: 多 Provider 智能检测 + 会话隔离
│   ├── llm.py          #   LLMClient: 流式调用封装
│   └── context.py      #   load_context_files: AGENTS.md 加载
├── core/               # 核心引擎 — 组装 + 运行 + 存储 + 压缩
│   ├── agent.py        #   Agent: 纯装配层 (~130 行)
│   ├── runner.py       #   AgentRunner: think-act 循环
│   ├── session_tree.py #   SessionTree: 树状会话（分叉/导航/压缩）
│   ├── system_prompt.py#   SystemPrompt: 系统提示词构建（实时查询 memory）
│   ├── compaction.py   #   CompactionService: 压缩编排（含 LLM 提取 + 分发 + 树压缩 + 重建提示词）
│   ├── events.py       #   EventBus: 发布/订阅事件总线
│   ├── memory.py       #   AgentMemory: 纯存储层（树为唯一数据源 + 三层记忆）
│   ├── tracker.py      #   TokenTracker: Token 消耗统计
│   ├── prompts.py      #   PromptLoader: 命令模板加载
│   └── cli_helpers.py  #   handle_tree/fork/back
└── tools/              # 工具层 — 扁平布局，每工具一个文件
    ├── base.py         #   Tool 基类 + Schema 瘦身（保留 anyOf+default）
    ├── registry.py     #   ToolRegistry: 工具注册表 + 错误纠错
    ├── executor.py     #   ToolExecutor: 串行/并行调度
    ├── bash.py         #   终端命令（空输出确认）
    ├── file_read.py    #   文件读取
    ├── file_write.py   #   文件写入
    ├── file_edit.py    #   文件编辑（模糊匹配）
    ├── web_fetch.py    #   网页抓取
    ├── web_search.py   #   网络搜索
    ├── skill.py        #   技能加载
    ├── todo.py         #   待办管理
    └── subagent.py     #   SubagentRunner + SubagentTool（单/并行/链式）
```

### 设计原则

1. **树状会话** — SessionTree 管理对话分支，分叉不丢数据，压缩插入 COMPACT 节点
2. **无列表双写** — AgentMemory 以 SessionTree 为唯一数据源，history 是 tree.build_context() 的实时计算
3. **事件驱动** — EventBus 解耦各组件，通过 turn:start / context:high / tool:before / tool:after 等事件通信
4. **存储与压缩分离** — AgentMemory 只做纯 I/O + 树操作，CompactionService 编排全部压缩流程（含 LLM 提取）
5. **扁平工具** — 每个工具一个 .py 文件

---

## 2. 关键组件详解

### 2.1 AppConfig (`agent/ai/config.py`)

集中管理所有可配置项，支持多 Provider 和会话隔离。

- **多 Provider 支持**：`PROVIDER_PRESETS` 定义 `deepseek` / `openai` / `custom` 三组预设
- **智能检测**：`from_env()` 自动根据环境变量（`DEEPSEEK_API_KEY` → `OPENAI_API_KEY` → `API_KEY`）选择 Provider
- **会话隔离**：每个会话自动生成 `session_id`（时间戳 + 微秒），会话文件存储在 `sessions/<session_id>/` 目录

```python
# 自动检测
config = AppConfig.from_env()

# 显式指定
config = AppConfig.from_env(provider="openai", model="gpt-4o")

# 恢复最近会话
config = AppConfig.from_env(restore_session=True)
```

关键配置项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `provider` | deepseek | Provider 名称 |
| `model` | 根据 Provider | 默认模型 |
| `max_tokens` | 20_000 | LLM 单次最大输出 |
| `max_context` | 200_000 | 上下文窗口上限 |
| `compact_threshold` | 0.35 | 上下文使用率阈值，触发压缩 |
| `restore_session` | False | 默认不恢复，用 -r 恢复 |
| `session_id` | 自动生成 | 会话标识 |
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

支持 `thinking` 参数控制推理努力程度（`off` / `minimal` / `low` / `medium` / `high` / `xhigh`）。

### 2.3 Agent (`agent/core/agent.py`)

纯装配层（~130 行），创建并连接所有组件：

```
Agent.__init__
├── AppConfig          — 配置
├── EventBus           — 事件总线
├── AgentMemory        — 纯存储（树 + 三层记忆）
├── TokenTracker       — Token 统计
├── SystemPrompt       — 系统提示词构建器
├── CompactionService  — 压缩编排（含 LLM 提取）
├── SkillsLoader       — 技能加载
├── AgentLoader        — 子代理定义加载
├── PromptLoader       — 命令模板加载
├── ToolRegistry       — 工具注册（_build_registry 工厂函数）
├── LLMClient          — LLM 客户端
├── ToolExecutor       — 工具执行器
├── AgentRunner        — 执行引擎
└── _setup_events()    — 注册事件监听
```

#### 内部组件

**SystemPrompt** (`system_prompt.py`): 持有静态上下文引用，`build(compaction_data=None)` 实时查询 memory 的动态部分（brief_context、user_preferences），保证每次调用反映最新状态。

**CompactionService** (`compaction.py`): 编排完整压缩流程（已合并原 Compactor 的 LLM 提取逻辑）：
```
compact()
├── _find_cut_point()                  → token-aware 切点（沿用户消息边界累积估算）
├── _extract(history)                  → LLM 提取 summary / preferences / facts
│   └── _get_previous_summary()        → 迭代压缩：传入前次压缩摘要作为上下文
├── memory.append_summary(...)          → 写入 summaries/*.md
├── memory.add_user(p)                  → 写入 user.md（内置去重）
├── memory.add_memory(f)                → 写入 memory.md（内置去重）
├── memory.compress_tree(summary, first_kept_id, tokens)
│   └── tree.compact() → 插入 COMPACT 节点 + JSONL 全量持久化
└── prompt.build(data) → 重建系统提示词 ← 返回给 Agent
```

#### 请求处理

```
Agent.process(message)
├── bus.emit("message:received")
├── memory.append_message(user msg)       → 树 + JSONL
├── context = _build_context()            → [system] + memory.history 快照
├── runner.step(context)                  → Runner 在快照上工作
└── 扫描 context 增量 → memory.append_message(...) → 持久化
```

#### 事件绑定 (`_setup_events()`)

| 事件 | 触发时机 | 处理 |
|------|----------|------|
| `turn:start` | 每轮 LLM 调用前 | 检查 should_compact()，达到阈值则 emit context:high + compact() |
| `context:high` | 压缩触发（扩展钩子） | 外部监听，可在压缩前后执行自定义逻辑 |

### 2.4 SessionTree (`agent/core/session_tree.py`)

树状会话结构，替代线性列表。核心操作：

| 操作 | 说明 |
|------|------|
| `append()` | 在 leaf 下创建子节点 |
| `compact()` | 插入 COMPACT 节点 + 重排 parent_id |
| `fork(id)` / `navigate(id)` | 移动 leaf 指针（不修改数据） |
| `path()` | 从 root 沿 parent_id 走到 leaf |
| `build_context()` | 构建 LLM 消息列表，遇 COMPACT 跳过已压缩消息 |

### 2.5 AgentMemory (`agent/core/memory.py`)

**树为唯一数据源**，无列表双写：
- `history` (property) → `tree.build_context()` 实时计算
- **唯一写入入口**: `append_message(msg)` → 树追加 + JSONL 增量写
- **fork 跳转栈**: `push_fork()` / `pop_fork()` 支持 `/back` 返回

三层记忆结构：

```
agent/.memory/
├── memory.md              ← 长期记忆（跨会话，追加事实 + 自动去重）
├── user.md                ← 用户偏好（跨会话，追加偏好 + 自动去重）
├── summaries/             ← 每日摘要（跨会话，按日期 YYYY-MM-DD.md）
└── sessions/
    └── <session_id>/      ← 会话私有目录
        ├── history.jsonl  ← 树结构对话历史
        └── tokens.jsonl   ← Token 消耗日志
```

### 2.6 AgentRunner (`agent/core/runner.py`)

纯粹的执行引擎，**不持有会话状态**，不关心持久化。

```
for turn in range(max_turns):
    1. emit("turn:start")
    2. LLM 流式调用 → 逐 chunk 产出 text + tool_calls
    3. context.append(assistant_msg)           # 在 Agent 给的快照上操作
    4. 无 tool_calls → 返回
    5. tool_executor.execute(tool_calls) → 串行/并行调度
    6. context.append(tool_msg)
    7. 循环
```

### 2.7 TokenTracker (`agent/core/tracker.py`)

Token 消耗统计，目录延迟创建（首次 record() 时）。支持 `should_compact()` / `stats_by_model()` / `stats_by_date()`。

---

## 3. 工具系统

### 3.1 Tool 基类 (`agent/tools/base.py`)

```python
@tool(name="my_tool", description="...", parameters=MyArgs)
class MyTool(Tool):
    parallel_safe: bool = True
    supports_streaming: bool = False

    def execute(self, **kwargs) -> str: ...
    def stream_execute(self, **kwargs) -> Generator[str]: ...
```

**Schema 瘦身** (`_minify_schema()`): 仅删除 `title` / `additionalProperties`，**保留** `anyOf[{type}, {null}]` 和 `default` 值，帮助 LLM 准确识别可选参数。

### 3.2 ToolRegistry (`agent/tools/registry.py`)

管理工具实例，提供注册、schema 生成和调用功能。错误检测使用 `"error" in result.lower()` 覆盖所有工具的错误格式，自动追加纠错提示 `[分析上述错误并尝试不同的方案。]`。

### 3.3 ToolExecutor (`agent/tools/executor.py`)

执行策略：
1. **串行** — 单个工具，或存在非 `parallel_safe` 工具时降级
2. **并行** — 全部 `parallel_safe` 时 ThreadPoolExecutor 并发（最多 8）
3. **事件拦截** — tool:before / tool:after 事件钩子
4. **结果截断** — 50KB / 2000 行

### 3.4 BashTool (`agent/tools/bash.py`)

安全护栏 + 流式输出。`stream_execute` 无输出时返回 `"[命令执行成功，无输出。]"` 避免 LLM 困惑。

---

## 4. 子代理系统 (`agent/tools/subagent.py`)

### 4.1 AgentLoader

扫描 `agent/subagent/*.md`，解析 Markdown + YAML frontmatter 为 `AgentDefinition`。

### 4.2 SubagentRunner

封装子代理的创建和运行逻辑（LLMClient + ToolExecutor + TokenTracker + AgentRunner 组装）。SubagentTool 通过它运行单个子代理，避免组装逻辑重复。

### 4.3 SubagentTool

三种调用模式：
```
单模式:   subagent_tool(task="分析代码", agent="scout")
并行模式: subagent_tool(tasks=[{task:"A"}, {task:"B"}])
链式模式: subagent_tool(chain=[{task:"Step1"}, {task:"Step2+{previous}"}])
```

每个子代理拥有独立的 history / TokenTracker / ToolRegistry / AgentRunner 实例。

---

## 5. 工作流示意

```
用户输入 → Agent.process(message)
   ├── memory.append_message(user msg)         → 树 + JSONL
   ├── _build_context()                        → [system] + tree.build_context()
   ├── runner.step(context)
   │     ├── turn:start → should_compact? → compact()
   │     ├── LLM.stream() → yield 文本
   │     ├── ToolExecutor.execute()
   │     └── 循环
   ├── 扫描 context 增量 → memory.append_message(...)
   └── 返回响应

Agent.shutdown()
   └── CompactionService.compact()
        ├── _find_cut_point() → token-aware 切点
        ├── _extract() → LLM 提取（含迭代压缩上下文）
        ├── Memory 分发（三层记忆）
        ├── memory.compress_tree() → COMPACT + JSONL
        └── SystemPrompt.build(data) → 重建提示词
```

---

## 6. 事件系统

| 事件 | 来源 | 消费者 | 用途 |
|------|------|--------|------|
| `message:received` | `Agent.process()` | 外部监听 | 记录用户输入 |
| `turn:start` | `Runner.step()` | `Agent._setup_events()` | 每轮检查压缩 |
| `context:high` | `Agent._setup_events()` | `Agent._setup_events()` | 触发压缩 |
| `tool:before` | `ToolExecutor` | 外部监听 | 拦截/修改工具调用 |
| `tool:after` | `ToolExecutor` | 外部监听 | 修改工具结果 |
| `session:end` | `Agent.shutdown()` | 外部监听 | 会话结束通知 |

（`history:appended` 事件已移除 — Runner 不再负责持久化，Agent 在 process() 末端统一处理）

---

## 7. CLI 命令

| 命令 | 说明 |
|------|------|
| `/fork [n]` | 分叉到第 n 条用户消息之前，新分支不含该消息 |
| `/back` | 弹出跳转栈，返回分叉前的位置 |
| `/tree` | 显示会话分支树可视化 |
| `/scout <query>` | 展开为 scout 子代理侦查任务 |
| `/review <query>` | 展开为 reviewer 子代理审查任务 |

`-r` / `--restore` 启动时恢复最近一次有内容的会话。

---

## 8. 测试

测试采用先单元后集成的策略，每个模块经过三重审查（自审 → Subagent 审 → 人工审）后提交。

### 覆盖状况

截至 2026-05-30，已覆盖 **20 个模块，共 473 个单元测试**：

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
