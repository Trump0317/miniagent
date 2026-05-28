# miniagent

一个基于 LLM (大语言模型) 的智能助手框架，采用 ReAct 模式，支持工具调用、多 Provider 切换、子代理、会话恢复等功能。

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
├── config.py           # 多 Provider 配置（DeepSeek/OpenAI/自定义）
├── conversation.py     # 会话状态管理 + 历史恢复
├── loop.py             # Agent 组装入口 + /command 命令
├── runner.py           # 执行引擎（LLM + 工具编排）
├── memory.py           # 三层记忆系统 + 自动压缩
├── hooks.py            # 事件钩子（拦截/修改工具调用）
├── prompts.py           # Prompt 模板加载器
├── tokentracker.py     # Token 消耗统计
├── subagent/           # 子代理定义文件（Markdown + YAML）
│   ├── scout.md        #   代码侦查员
│   └── reviewer.md     #   代码审查员
├── prompts/            # Prompt 模板文件
│   ├── scout.md        #   /scout 命令
│   └── review.md       #   /review 命令
└── tools/              # 工具集
    ├── BashTool/       #   终端命令（安全护栏 + 流式输出）
    ├── FileReadTool/   #   文件读取
    ├── FileWriteTool/  #   文件写入
    ├── FileEditTool/   #   文件编辑（模糊匹配）
    ├── WebFetchTool/   #   网页抓取
    ├── WebSearchTool/  #   网络搜索
    ├── SkillTool/      #   技能加载
    ├── TodoWriteTool/  #   待办管理
    └── SubagentTool/   #   子代理（单/并行/链式）
```

## 功能清单

| 类别 | 功能 | 说明 |
|------|------|------|
| **核心** | ReAct 循环 | LLM think-act 迭代，自动工具调用 |
| | 流式输出 | LLM 文本逐 token 显示 |
| | 思维链 | 支持 DeepSeek R1 reasoning_content |
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
| **模型** | 多 Provider | DeepSeek / OpenAI / 自定义 API |
| | 子代理模型 | 子代理可独立指定模型 |
| **会话** | 会话恢复 | 重启自动加载上次对话历史 |
| | 记忆压缩 | Token 超阈值自动压缩 |
| | 三层记忆 | 短期对话 / 每日摘要 / 长期记忆 |
| **扩展** | 事件钩子 | on_tool_call / on_tool_result |
| | Agent 定义 | Markdown + YAML 文件配置子代理 |
| | Prompt 模板 | `/scout`, `/review` 等快捷命令 |

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
[You]: 用 scout 子代理找一下 agent/runner.py 里的方法
[Assistant]: [执行工具: subagent_tool...]
  ╭─ [子代理] 开始执行 ─
  ...侦查结果...
  ╰─ [子代理] 完成 ─
## 找到的文件...
```

### 命令模板
```
[You]: /scout agent/loop.py       # 展开为"用 scout 侦查 agent/loop.py"
[You]: /review agent/runner.py    # 展开为"用 reviewer 审查 agent/runner.py"
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
