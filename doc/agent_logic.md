# Agent 逻辑说明文档

本文档详细介绍了 `agent/` 目录下的核心逻辑与重构后的架构设计。

## 1. 核心架构概述

项目实现了基于 LLM 的智能助手框架，采用 ReAct 模式。重构后的架构将配置、状态、执行三层解耦：

```
AppConfig  →  Agent  →  Conversation (状态)
  (配置)        │         ├── AgentMemory (记忆)
                │         └── TokenTracker (统计)
                │
                ├── AgentRunner (执行引擎)
                │     ├── 流式 LLM 调用
                │     ├── 串行/并行工具调度
                │     ├── 流式工具输出
                │     └── 事件钩子
                │
                └── AgentLoader / PromptLoader
                      ├── 子代理定义 (subagent/*.md)
                      └── 命令模板 (prompts/*.md)
```

### 核心文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `config.py` | 115 | 多 Provider 配置，`.env` 智能检测 |
| `conversation.py` | 80 | 会话状态：历史、记忆、Token 统一管理 |
| `loop.py` | 140 | Agent 组装入口 + `/command` 交互循环 |
| `runner.py` | 265 | 执行引擎：LLM 流式调用 + 工具编排 |
| `memory.py` | 265 | 三层记忆 + 自动压缩 + 历史恢复 |
| `hooks.py` | 65 | 事件钩子：`on_tool_call` / `on_tool_result` |
| `prompts.py` | 90 | Prompt 模板加载器 |
| `tokentracker.py` | 70 | Token 消耗统计 |

---

## 2. 关键组件详解

### 2.1 AppConfig (`agent/config.py`)

集中管理所有可配置项，替代了之前散落在各处的硬编码参数。

- **多 Provider 支持**：`PROVIDER_PRESETS` 定义 deepseek / openai / custom 三组预设
- **智能检测**：`from_env()` 自动根据环境变量（`DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `API_KEY`）选择 Provider
- **可覆盖**：所有参数都可通过 `overrides` 传入

```python
# 自动检测
config = AppConfig.from_env()

# 显式指定
config = AppConfig.from_env(provider="openai", model="gpt-4o")

# 完全手动
config = AppConfig(provider="custom", model="llama3", api_base_url="http://localhost:8000/v1")
```

### 2.2 Agent (`agent/loop.py`)

从 `AgentLoop` 重命名为 `Agent`，职责从"上帝方法"变为"组装器"：

- 创建 `AppConfig` → 创建客户端
- 创建 `AgentMemory` + `TokenTracker` → 包装为 `Conversation`
- 创建 `ToolRegistry`（主工具 + 子代理工具）
- 创建 `AgentRunner`（绑定 `Conversation`）
- `run()` 主循环支持 `/command` 模板展开

### 2.3 AgentRunner (`agent/runner.py`)

执行引擎，支持两种模式：

- **主循环模式**：传入 `conversation`，自动管理历史和 Token
- **子代理模式**：不传 `conversation`，用原始 `history` list

**关键流程** (`step` 方法)：

1. LLM 流式调用 → 逐 chunk 产出文本
2. 解析 `tool_calls` → 串行或并行执行
3. `_execute_tools` → 根据 `parallel_safe` 标记决定策略：
   - 全部安全 → `ThreadPoolExecutor` 并行（最多 8 并发）
   - 存在不安全 → 全部降级串行
4. 支持流式工具执行（`supports_streaming` + `stream_execute`）
5. 结果自动截断（50KB / 2000 行）

### 2.4 Conversation (`agent/conversation.py`)

统一管理会话状态，替代 Runner 直接操作 `memory` 和 `token_tracker`：

- `add_user_message()` / `add_assistant_message()` / `add_tool_result()`
- `record_tokens()` — 记录 Token 消耗
- `should_compact()` / `compact()` — 压缩触发
- `restore` 参数 — 启动时是否从 `history.jsonl` 恢复历史

### 2.5 记忆系统 (`agent/memory.py`)

三层记忆结构：

- **短期记忆**：当前对话 `history` 列表
- **历史摘要**：每日对话总结（`summaries/*.md`）
- **长期记忆**：核心事实（`memory.md`）+ 用户偏好（`user.md`）

新增 `restore_history()` 方法，从 `history.jsonl` 加载上次会话（含 tool 消息）。

### 2.6 工具系统 (`agent/tools/`)

#### Tool 基类 (`ToolRegisty/base.py`)

重构后不再使用 `__abstractmethods__` hack：

- `@tool` 装饰器直接注入 `_tool_name` / `_tool_description` / `_args_model` 类属性
- 子类无需冗余的 `name: str` 等类注解
- `parallel_safe` — 声明是否可并发
- `supports_streaming` + `stream_execute()` — 可选择流式执行

#### 具体工具

| 工具 | 特性 |
|------|------|
| **BashTool** | 安全护栏（正则黑名单）+ 流式输出（Popen 逐行） |
| **FileEditTool** | 三级匹配：精确 → 忽略缩进 → 失败提示 |
| **SubagentTool** | 单/并行/链式，支持命名 Agent 定义，实时输出 |
| **TodoWriteTool** | `parallel_safe=False`，防止状态竞争 |

### 2.7 事件钩子 (`agent/hooks.py`)

两个钩子点：

- `on_tool_call(name, args)` → 返回 `{"block": True}` 拦截 / `{"args": ...}` 修改参数
- `on_tool_result(name, result)` → 返回修改后的结果

```python
hooks = EventHooks()
hooks.on_tool_call = lambda name, args: (
    {"block": True, "reason": "只读模式"}
    if name not in READ_ONLY else None
)
agent.runner.hooks = hooks
```

### 2.8 Agent 定义 (`agent/subagent/*.md`)

Markdown + YAML frontmatter 定义子代理，无需改代码：

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

### 2.9 Prompt 模板 (`agent/prompts/*.md`)

用户输入 `/name query` 自动展开为模板：

```markdown
---
description: 用 scout 先侦查再回答
---
先用 scout 子代理快速侦查 {query}，然后根据侦查结果回答。
```

---

## 3. 工作流示意

```
用户输入 "/scout agent/runner.py"
        │
        ▼
Agent._expand_command()
   → PromptLoader.resolve("scout", "agent/runner.py")
   → "先用 scout 子代理快速侦查 agent/runner.py..."
        │
        ▼
AgentRunner.step(history)
   ┌── LLM 流式调用 ──→ yield 文本 chunks
   ├── 解析 tool_calls: [{name: "subagent_tool", args: {agent: "scout", ...}}]
   ├── before hook → 可拦截/修改
   ├── _execute_tools → 并行/串行调度
   │     └── SubagentTool._run_one
   │           ├── AgentLoader.get("scout") → AgentDefinition
   │           └── 独立 AgentRunner(registry=filtered, model=deepseek-v4-flash)
   ├── after hook → 可修改结果
   └── 循环直到 LLM 不再调用工具
        │
        ▼
   "[Assistant]: ## 找到的文件..."
```
