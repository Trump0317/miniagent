# miniagent

一个基于 LLM 的智能助手框架，采用 ReAct 模式 + 事件驱动架构，支持工具调用、多 Provider 切换、子代理、会话分叉、上下文压缩、三层记忆等功能。

## 快速开始

```bash
# 1. 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 API Key（支持 DeepSeek / OpenAI / 自定义）

# 3. 启动（新会话）
./run.sh

# 恢复最近会话
./run.sh -r
```

## 架构

```
agent/
├── __init__.py         # 公共导出（Agent, EventBus, AppConfig, LLMClient）
├── ai/                 # AI 层（Provider 配置 + LLM 调用 + 上下文加载）
│   ├── config.py       #   多 Provider 配置 + 会话隔离
│   ├── llm.py          #   LLM 客户端封装
│   └── context.py      #   项目上下文文件加载
├── core/               # 核心引擎（Agent 装配 + 运行器 + 存储）
│   ├── agent.py        #   Agent 核心（纯装配层，~130 行）
│   ├── runner.py       #   执行引擎（LLM think-act 迭代）
│   ├── session_tree.py #   树状会话（分叉/导航/压缩节点）
│   ├── system_prompt.py#   系统提示词构建器（实时查询 memory）
│   ├── compaction.py   #   压缩编排服务（含 LLM 提取 + 分发 + 树压缩 + 提示词重建）
│   ├── events.py       #   事件总线（发布/订阅，组件解耦）
│   ├── memory.py       #   纯存储层（树为唯一数据源 + 三层记忆）
│   ├── tracker.py      #   Token 消耗统计
│   ├── prompts.py      #   Prompt 模板加载器
│   └── cli_helpers.py  #   CLI 辅助（handle_tree/fork/back）
├── tools/              # 工具集（扁平布局，每个工具一个文件）
│   ├── base.py         #   工具基类 + Schema 瘦身（保留 anyOf+default）
│   ├── registry.py     #   工具注册表
│   ├── executor.py     #   工具执行器（串行/并行调度）
│   ├── bash.py         #   终端命令（安全护栏 + 空输出确认）
│   ├── file_read.py    #   文件读取
│   ├── file_write.py   #   文件写入
│   ├── file_edit.py    #   文件编辑（模糊匹配）
│   ├── web_fetch.py    #   网页抓取
│   ├── web_search.py   #   网络搜索
│   ├── skill.py        #   技能加载
│   ├── todo.py         #   待办管理
│   └── subagent.py     #   SubagentRunner + SubagentTool
├── subagent/           # 子代理定义文件（Markdown + YAML）
│   ├── scout.md        #   代码侦查员
│   └── reviewer.md     #   代码审查员
└── prompts/            # Prompt 模板文件
    ├── scout.md        #   /scout 命令
    └── review.md       #   /review 命令
```

**设计原则：**
- **树状会话** — SessionTree 管理对话分支，分叉不丢数据，压缩插入 COMPACT 节点
- **无列表双写** — `AgentMemory` 以树为唯一数据源，`history` 是 `tree.build_context()` 的实时计算
- **事件驱动** — EventBus 解耦各组件
- **扁平工具** — 每个工具一个 py 文件

## 功能清单

| 类别 | 功能 | 说明 |
|------|------|------|
| **核心** | ReAct 循环 | LLM think-act 迭代，自动工具调用 |
| | 流式输出 | LLM 文本逐 token 显示 |
| | 思维链 | 支持 DeepSeek R1 reasoning_content |
| | 事件驱动 | EventBus 发布/订阅，组件完全解耦 |
| **工具** | Bash | 终端命令，安全护栏拦截危险操作，空输出确认 |
| | 文件读写 | read / write / edit（模糊匹配） |
| | 网络 | fetch / search |
| | Todo | 待办列表增删改查 |
| | Skill | 加载预定义技能 |
| | Subagent | 单 / 并行 / 链式子代理（SubagentRunner 独立封装） |
| **安全** | 安全护栏 | 正则拦截 `rm -rf /` 等危险命令 |
| | 工具拦截 | 事件钩子可阻止任意工具执行 |
| | 错误纠错 | 所有工具错误自动附带重试提示 |
| **性能** | 并行工具 | 多工具并发执行（ThreadPoolExecutor） |
| | 流式工具 | BashTool 边执行边显示输出 |
| | 结果截断 | 50KB/2000 行自动截断 |
| | Schema 瘦身 | 仅删 title/additionalProperties，保留 anyOf+default |
| **模型** | 多 Provider | DeepSeek / OpenAI / 自定义 API |
| | 子代理模型 | 子代理可独立指定模型 |
| **会话** | 会话隔离 | 每会话独立 sessions/<ts>/ 目录，互不干扰 |
| | --restore | `-r` 标志恢复最近会话 |
| | 分叉回退 | `/fork` 分叉 + `/back` 返回，跳转栈不丢位置 |
| | 主动压缩 | 每轮检查 token 阈值（默认 35%），提前压缩 |
| | 动态提示词 | 压缩后 SystemPrompt.build(data) 注入压缩摘要 |
| | 三层记忆 | 短期对话 / 每日摘要 / 长期记忆（memory.md） |
| | 记忆去重 | 偏好和事实自动去重 |
| **扩展** | 事件钩子 | context:high / turn:start / tool:before / tool:after |
| | Agent 定义 | Markdown + YAML 文件配置子代理 |
| | Prompt 模板 | `/scout`, `/review` 等快捷命令 |
| | Token 统计 | 按模型/日期聚合统计 |

## 使用示例

### 基本对话
```
[You]: 你好
[Assistant]: 你好！有什么可以帮你的？😊
```

### 工具调用
```
[You]: 列出当前目录的文件
[Assistant]: [执行工具: bash_tool...]
LICENSE  README.md  agent/  agent.py  ...
```

### 会话管理
```
[You]: /fork           # 分叉到上一次用户消息之前
[You]: /fork 3         # 分叉到第 3 条用户消息之前
[You]: /back           # 返回分叉前的位置
[You]: /tree           # 查看分支树
```

### 子代理
```
[You]: 用 scout 子代理找一下 agent/core/runner.py 里的方法
[Assistant]: [执行工具: subagent_tool...]
  ╭─ [子代理] 开始执行 ─
  ...侦查结果...
  ╰─ [子代理] 完成 ─
```

### 命令模板
```
[You]: /scout agent/core/agent.py     # 展开为"用 scout 侦查 agent/core/agent.py"
[You]: /review agent/core/runner.py   # 展开为"用 reviewer 审查 agent/core/runner.py"
```

### 自定义子代理

在 `agent/subagent/` 下创建 `.md` 文件即可，无需改代码：

```markdown
---
name: my-agent
description: 我的自定义代理
tools: bash_tool, file_read_tool
model: deepseek-v4-flash
max_turns: 8
---

你是我的自定义子代理...
```

## 配置 (.env)

```bash
# DeepSeek（默认）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1

# 或 OpenAI
# OPENAI_API_KEY=sk-xxx
# OPENAI_API_BASE_URL=https://api.openai.com/v1

# 或自定义（Ollama / vLLM 等）
# API_KEY=xxx
# API_BASE_URL=http://localhost:8000/v1
```
