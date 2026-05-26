# Agent 逻辑说明文档

本文档详细介绍了 `agent/` 目录下的核心逻辑与架构设计。

## 1. 核心架构概述

该项目实现了一个基于 LLM（大语言模型）的智能助手框架，采用了经典的 **ReAct (Reasoning and Acting)** 模式。系统能够通过工具调用（Tool Calling）与外部环境交互，并根据执行结果调整后续行为。

主要组件分布在 `agent/` 目录下：

- **`loop.py`**: 对话主循环管理，负责初始化环境、注册工具并处理用户输入。
- **`runner.py`**: 核心执行器，负责与模型交互、处理思维链（Reasoning）以及调度工具执行。
- **`memory.py`**: 记忆系统，负责对话历史的持久化、长期记忆维护及自动压缩总结。
- **`tokentracker.py`**: Token 计数器，记录每轮对话的消耗，支持缓存统计。
- **`tools/`**: 工具包，包含工具注册表、基类定义以及各种具体功能的实现。

---

## 2. 关键组件详解

### 2.1 AgentLoop (`agent/loop.py`)
`AgentLoop` 是系统的入口点，其逻辑流程如下：

1. **初始化**:
   - 加载配置文件（`.env`）。
   - 初始化 OpenAI 兼容的客户端（如 DeepSeek）。
   - 初始化 **Memory** 和 **TokenTracker**。
   - 加载技能摘要（`SkillsLoader`）。
   - **工具注册**: 实例化并向 `ToolRegistry` 注册各类工具（Bash、FileRead、FileEdit、TodoWrite 等）。
2. **状态管理**:
   - 维护 `history` 列表（通过 `AgentMemory` 管理），存储系统提示词（System Prompt）、用户输入、模型回复及工具返回结果。
   - 系统提示词会动态注入 **长期记忆** 和 **用户偏好**。
3. **主循环 (`run`)**:
   - 监听控制台输入。
   - 调用 `AgentRunner` 执行对话并流式显示结果。
   - 退出时打印 Token 统计并触发记忆压缩。

### 2.2 AgentRunner (`agent/runner.py`)
`AgentRunner` 是模型交互的核心处理类，支持多轮迭代。

- **流式处理**: 支持流式获取模型输出，并能区分普通内容（`content`）和思维链内容（`reasoning_content`）。
- **Token 监控**: 在流式输出过程中实时记录 Token 使用情况，支持 DeepSeek 缓存命中统计。
- **自动压缩**: 根据 `max_context` 和 `compact_threshold` 自动判断是否需要调用 `memory.compact()` 进行历史瘦身。
- **迭代循环 (`step`)**:
  1. 向模型发送当前对话历史及可用工具 Schema。
  2. 接收响应。如果模型返回 `tool_calls`（工具调用请求），则暂停文本生成。
  3. **工具执行**: 解析工具名和参数，通过 `ToolRegistry` 调用对应工具。
  4. **反馈闭环**: 将工具执行结果作为 `role: tool` 的消息加入历史。
  5. **继续迭代**: 重新请求模型，让其根据工具结果给出最终结论或进行下一步行动，直到模型不再要求调用工具。

### 2.3 记忆系统 (`agent/memory.py`)
`AgentMemory` 提供了三层记忆存储结构，确保 Agent 在长对话中保持连贯性：

- **短期记忆**: 当前对话的 `history` 列表。
- **历史摘要** (`summaries/`): 每天生成的对话关键点总结。
- **长期记忆** (`memory.md`): 记录核心目标、当前任务与关键事实。
- **用户偏好** (`user.md`): 记录用户的习惯和个性化要求。
- **压缩机制**: 当历史消息达到阈值时，自动调用 LLM 提取关键信息并更新摘要/长期记忆，清空冗余历史。

### 2.4 工具系统 (`agent/tools/`)

#### ToolRegistry (`ToolRegisty/registry.py`)
- **管理中心**: 负责存储所有已注册的工具实例。
- **Schema 生成**: 将工具定义的参数转换为符合模型（OpenAI/DeepSeek）要求的 JSON Schema 格式。
- **安全分发**: 提供 `call_tool` 接口，在执行前进行参数校验和错误捕获。

#### Tool 基类 (`ToolRegisty/base.py`)
- **Pydantic 驱动**: 每个工具都必须定义 `args_model`（继承自 `pydantic.BaseModel`），利用 Pydantic 的强类型校验能力确保 LLM 传参的准确性。
- **标准化接口**: 强制要求实现 `execute` 方法，确保所有工具的调用行为一致。

#### 具体工具实现
- **BashTool**: 执行系统命令。
- **FileTools** (`FileRead`, `FileWrite`, `FileEdit`): 处理文件 IO 及增量编辑。
- **WebTools** (`WebFetch`, `WebSearch`): 获取网页内容或进行网络搜索。
- **SkillTool**: 动态加载并使用预定义的复杂操作序列（Skills）。
- **TodoWriteTool**: 管理待办事项列表，支持增删改查。

---

## 3. 工作流示意

1. **用户输入** -> `AgentLoop` 接收 -> `AgentRunner` 启动。
2. **模型请求** -> `AgentRunner` 封装历史 + 工具 Schema -> 发送。
3. **决策分歧**:
   - **如果输出文本**: 直接反馈给用户。
   - **如果调用工具**: 
     - 执行器根据 `tool_calls` 找到对应工具类。
     - Pydantic 进行参数校验 -> 执行 `execute`。
     - 结果存入历史 -> 回到步骤 2 循环。
4. **对话结束** -> 模型给出最终回复 -> 等待下一次用户输入。
