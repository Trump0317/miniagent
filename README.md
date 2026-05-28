# miniagent

一个基于 LLM 的智能助手框架，采用 ReAct 模式 + 事件驱动架构，支持工具调用、多 Provider 切换、子代理、会话恢复、上下文压缩等功能。

## 快速开始

```bash
# 1. 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 API Key（支持 DeepSeek / OpenAI / 自定义）

# 3. 启动
./run.sh
```

## 架构

```
agent/
├── __init__.py         # 公共导出（Agent, EventBus, AppConfig, LLMClient）
├── ai/                 # AI 相关（Provider 配置 + LLM 调用 + 上下文加载）
│   ├── config.py       #   多 Provider 配置（DeepSeek/OpenAI/自定义）
│   ├── llm.py          #   LLM 客户端封装
│   └── context.py      #   项目上下文文件加载
├── core/               # 核心引擎（Agent 组装 + 运行器 + 存储）
│   ├── agent.py        #   Agent 核心（组件组装 + process 入口 + 压缩编排）
│   ├── runner.py       #   执行引擎（LLM think-act 迭代）
│   ├── events.py       #   事件总线（发布/订阅，组件解耦）
│   ├── memory.py       #   纯存储层（JSONL 历史 + memory.md + 偏好管理）
│   ├── compactor.py    #   压缩器（LLM 提取摘要/偏好/事实，不操作文件）
│   ├── tracker.py      #   Token 消耗统计
│   └── prompts.py      #   Prompt 模板加载器
├── tools/              # 工具集（扁平布局，每个工具一个文件）
│   ├── base.py         #   工具基类 + JSON Schema 瘦身
│   ├── registry.py     #   工具注册表
│   ├── executor.py     #   工具执行器
│   ├── bash.py         #   终端命令
│   ├── file_read.py    #   文件读取
│   ├── file_write.py   #   文件写入
│   ├── file_edit.py    #   文件编辑（模糊匹配）
│   ├── web_fetch.py    #   网页抓取
│   ├── web_search.py   #   网络搜索
│   ├── skill.py        #   技能加载
│   ├── todo.py         #   待办管理
│   └── subagent.py     #   子代理（单/并行/链式）
├── subagent/           # 子代理定义文件（Markdown + YAML）
│   ├── scout.md        #   代码侦查员
│   └── reviewer.md     #   代码审查员
└── prompts/            # Prompt 模板文件
    ├── scout.md        #   /scout 命令
    └── review.md       #   /review 命令
```

**设计原则：**
- **事件驱动** — EventBus 解耦各组件，通过 `context:high` / `turn:start` / `history:appended` 等事件协作
- **存储与压缩分离** — `Memory` 只做纯 I/O，`Compactor` 只做 LLM 提取，互不依赖
- **零中间层** — 无 Conversation/Hooks 抽象，Agent 直接编排 Runner + Memory + Compactor
- **扁平工具** — 每个工具一个 py 文件，不再嵌套子目录

## 功能清单

| 类别 | 功能 | 说明 |
|------|------|------|
| **核心** | ReAct 循环 | LLM think-act 迭代，自动工具调用 |
| | 流式输出 | LLM 文本逐 token 显示 |
| | 思维链 | 支持 DeepSeek R1 reasoning_content |
| | 事件驱动 | EventBus 发布/订阅，组件完全解耦 |
| **工具** | Bash | 终端命令，安全护栏拦截危险操作 |
| | 文件读写 | read / write / edit（模糊匹配） |
| | 网络 | fetch / search |
| | Todo | 待办列表增删改查 |
| | Skill | 加载预定义技能 |
| | Subagent | 单 / 并行 / 链式子代理 |
| **安全** | 安全护栏 | 正则拦截 `rm -rf /` 等危险命令 |
| | 工具拦截 | 事件钩子可阻止任意工具执行 |
| **性能** | 并行工具 | 多工具并发执行（ThreadPoolExecutor） |
| | 流式工具 | BashTool 边执行边显示输出 |
| | 结果截断 | 50KB/2000 行自动截断 |
| | Schema 瘦身 | JSON Schema 剔除冗余字段（title/default/anyOf null），缩减 token |
| **模型** | 多 Provider | DeepSeek / OpenAI / 自定义 API |
| | 子代理模型 | 子代理可独立指定模型 |
| **会话** | 会话恢复 | 重启自动加载上次对话历史（注入上下文分隔提示） |
| | 主动压缩 | 每轮 turn:start 检查 token 阈值（默认 35%），提前触发压缩 |
| | 历史截断 | 压缩后保留系统提示词 + 最近 2 轮用户对话，释放上下文窗口 |
| | 动态提示词 | 压缩后重建系统提示词，注入压缩摘要、最新记忆和用户偏好 |
| | 三层记忆 | 短期对话（JSONL）/ 每日摘要 / 长期记忆（memory.md） |
| | 记忆去重 | 偏好和事实自动去重，避免重复存储 |
| | JSONL 同步 | 压缩后自动同步 JSONL 文件，保持与内存 history 一致 |
| **扩展** | 事件钩子 | context:high / turn:start / history:appended 等事件 |
| | Agent 定义 | Markdown + YAML 文件配置子代理 |
| | Prompt 模板 | `/scout`, `/review` 等快捷命令 |
| | Token 统计 | 按模型/日期聚合统计 + 新会话重置 |

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

### 子代理
```
[You]: 用 scout 子代理找一下 agent/core/runner.py 里的方法
[Assistant]: [执行工具: subagent_tool...]
  ╭─ [子代理] 开始执行 ─
  ...侦查结果...
  ╰─ [子代理] 完成 ─
## 找到的文件...
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
