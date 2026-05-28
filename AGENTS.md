# AGENTS.md — miniagent 项目上下文

## 项目概述

**miniagent** 是一个基于 LLM 的智能助手框架（约 2500 行 Python），采用 **ReAct 模式 + 事件驱动架构**。支持工具调用、多 Provider 切换、子代理、会话恢复、上下文压缩等功能。项目定位于轻量级 Harness，强调零中间层和组件解耦。

- **语言**: Python 3.12+
- **依赖**: openai >= 2.0, pydantic >= 2.0, python-dotenv >= 1.0
- **入口**: `agent.py`（CLI Harness 层）
- **虚拟环境**: `.venv`

## 架构总览

```
agent.py                          ← CLI 入口（交互/print 模式 + @文件引用 + 管道输入）
└── agent/
    ├── __init__.py                ← 公共导出
    ├── ai/                        ← AI 层（Provider 配置 + LLM 调用 + 上下文加载）
    │   ├── config.py              ← AppConfig: 多 Provider（DeepSeek/OpenAI/自定义）
    │   ├── llm.py                 ← LLMClient: OpenAI 兼容流式调用封装
    │   └── context.py             ← 自动加载 AGENTS.md/CLAUDE.md
    ├── core/                      ← 核心引擎
    │   ├── agent.py               ← Agent: 组件组装 + process() + shutdown()
    │   ├── runner.py              ← AgentRunner: think-act 循环编排
    │   ├── events.py              ← EventBus: 发布/订阅 + 通配符 + 一次性监听
    │   ├── memory.py              ← AgentMemory: 三层记忆（JSONL/memory.md/user.md）
    │   ├── compactor.py           ← Compactor: LLM 提取摘要/偏好/事实
    │   ├── tracker.py             ← TokenTracker: JSONL 日志 + 聚合统计
    │   └── prompts.py             ← PromptLoader: /command 模板加载
    ├── tools/                     ← 扁平工具集（每个工具一个 py 文件）
    │   ├── base.py                ← Tool 基类 + @tool 装饰器 + schema 瘦身
    │   ├── registry.py            ← ToolRegistry: 工具注册/查找/schema 生成
    │   ├── executor.py            ← ToolExecutor: 串行/并行调度 + 安全钩子
    │   ├── bash.py                ← BashTool: 终端命令 + 安全护栏 + 流式输出
    │   ├── file_read.py           ← 文件读取
    │   ├── file_write.py          ← 文件写入
    │   ├── file_edit.py           ← 文件编辑（精确匹配 + 忽略缩进模糊匹配）
    │   ├── web_fetch.py           ← 网页抓取
    │   ├── web_search.py          ← 网络搜索
    │   ├── skill.py               ← SkillsLoader + SkillTool
    │   ├── todo.py                ← TodoWriteTool: 有状态待办管理
    │   └── subagent.py            ← SubagentTool: 单/并行/链式子代理
    ├── subagent/                  ← 子代理定义（Markdown + YAML frontmatter）
    │   ├── scout.md               ← 代码侦查员
    │   └── reviewer.md            ← 代码审查员
    └── prompts/                   ← 命令模板
        ├── scout.md               ← /scout 展开
        └── review.md              ← /review 展开
```

## 核心设计原则

1. **事件驱动解耦** — EventBus 替代回调链：`turn:start` / `context:high` / `history:appended` / `tool:before` / `tool:after`
2. **存储与压缩分离** — `AgentMemory` 只做纯 I/O，`Compactor` 只做 LLM 提取，互不依赖
3. **零中间层** — 无 Conversation/Hooks 抽象，`Agent` 直接编排 `Runner` + `Memory` + `Compactor`
4. **扁平工具** — 每个工具一个 py 文件，用 `@tool` 装饰器注入 name/description/args_model
5. **Agent 定义为文件** — 子代理通过 Markdown + YAML frontmatter 配置，无需改代码

## 请求处理流程

```
用户输入
  → agent.py: _expand_command()        # /command 展开为模板
  → Agent.process(message)
     → EventBus.emit("message:received")
     → history.append(user message)
     → AgentRunner.step(history)        # think-act 循环
        → EventBus.emit("turn:start")   # 触发 token 阈值检查
        → LLMClient.stream() → chunk 生成器
        → 无工具调用 → 返回
        → ToolExecutor.execute()
           → 单工具: 串行
           → 多工具+全部parallel_safe: 并行(ThreadPoolExecutor, max 8)
           → 多工具+非全部安全: 全部降级串行
           → tool:before / tool:after 事件钩子
        → history.append(tool_results) + 继续循环
  → Agent.shutdown()
     → Compactor.compact() # 如达阈值
     → _sync_history_file() # JSONL 同步
```

## 关键模块详解

### Agent（`agent/core/agent.py`）
- 装配所有组件，提供 `process()` 和 `shutdown()` 两个公共 API
- `process()` 返回 Generator[str]，支持流式输出
- `_build_system_prompt()` 构造系统提示词（项目上下文 + 技能 + 子代理 + 命令 + 记忆 + 偏好）
- `_trim_history()` 截断为"系统提示词 + 最近 2 轮用户对话"
- `_rebuild_system_prompt_with_summary()` 压缩后动态重建系统提示词

### AgentMemory（`agent/core/memory.py`）
三层记忆体系：
| 层 | 文件 | 用途 |
|---|---|---|
| 短期 | `history.jsonl` | 对话历史，JSONL 持久化 |
| 中期 | `summaries/YYYY-MM-DD.md` | 每日压缩摘要 |
| 长期 | `memory.md` | 核心事实，常驻上下文 |
| 用户 | `user.md` | 偏好列表，自动去重 |

- `restore_history()` 只恢复 assistant/tool 消息（跳过 user，避免误解）
- `add_memory()` / `add_user()` 内置去重

### Compactor（`agent/core/compactor.py`）
- 不读写文件，纯 LLM 提取
- 提取三类信息：`summary`（critical/decision/issue）、`preferences`、`facts`
- 只处理最近 k 条非系统消息
- 结果由 Agent 分发到 Memory（摘要→summaries/，偏好→user.md，事实→memory.md）

### TokenTracker（`agent/core/tracker.py`）
- 按调用的 JSONL 日志（ts/model/input/output/cache_hit/cache_miss）
- `should_compact()`: 当 last_input_tokens > max_context * threshold（默认 35%）触发
- 支持 `stats_by_model()` / `stats_by_date()` 聚合

### 工具系统
- **@tool 装饰器**: 注入 `_tool_name` / `_tool_description` / `_args_model`
- **Schema 瘦身** (`_minify_schema`): 移除 title/default/additionalProperties，展开 anyOf null 模式，递归处理
- **ToolExecutor**: 支持并行执行（ThreadPoolExecutor）+ 安全钩子（tool:before / tool:after 事件）
- **SubagentTool**: 单/并行/链式三种模式，每个子代理独立上下文、独立 TokenTracker

### EventBus（`agent/core/events.py`）
核心事件：
- `turn:start` → 检查 token 阈值，触发压缩
- `context:high` → 执行压缩
- `history:appended` → 异步持久化到 JSONL
- `tool:before` / `tool:after` → 工具拦截/修改钩子
- `message:received` / `session:end`

## 配置（`agent/ai/config.py`）

```python
@dataclass
class AppConfig:
    provider: str          # deepseek / openai / custom
    model: str             # 默认模型
    max_turns: int | None  # 最大轮数
    max_tokens: int = 20000
    max_context: int = 200000
    compact_threshold: float = 0.35  # 35% 时触发压缩
    subagent_model: str    # 子代理默认模型
    subagent_max_turns: int = 15
    restore_session: bool = True
    context_files: str     # AGENTS.md 内容
```

Provider 自动检测：DEEPSEEK_API_KEY → OPENAI_API_KEY → API_KEY（custom）

## CLI 特性

- `-p` print 模式（非交互）
- `--thinking off|minimal|low|medium|high|xhigh` 思维链级别
- `-nc` 禁用上下文文件
- `@file.py` 文件引用展开
- 管道输入支持（`cat README.md | python agent.py -p "总结"`）
- `/command` 模板展开

## 安全措施

- BashTool: 正则拦截 `rm -rf /`, `mkfs`, `dd`, `chmod 777 /`, fork 炸弹等
- ToolExecutor: `tool:before` 事件可被消费者阻断任意工具执行
- 子代理: 独立 ToolRegistry 深拷贝，线程隔离

## 扩展点

1. **新工具**: 在 `agent/tools/` 下创建新 py 文件，用 `@tool` 装饰器，在 `agent/__init__.py` 和 `agent/core/agent.py` 的 `_build_registry()` 中注册
2. **新子代理**: 在 `agent/subagent/` 下创建 .md 文件（YAML frontmatter + Markdown body）
3. **新命令模板**: 在 `agent/prompts/` 下创建 .md 文件
4. **新事件监听器**: 在 `Agent._setup_events()` 中注册
5. **自定义 Provider**: 在 `PROVIDER_PRESETS` 中添加预设

## 注意事项

- `.memory/` 目录在 `.gitignore` 中被排除（用户数据）
- `skills/` 目录在 `.gitignore` 中被排除（外部技能）
- 系统消息不持久化（每次启动重建）
- 历史恢复只保留 assistant/tool 消息，最多 50 条
- Tool result 截断上限: 50KB / 2000 行
- 压缩阈值: 上下文 35% → 触发压缩 → 保留最近 2 轮 + 重建系统提示词
- `max_turns` 为 None 时不设上限

## 会话持久化

- 退出时自动触发压缩 + JSONL 同步
- 下次启动自动恢复上次对话（注入上下文分隔提示）
- 压缩提取的偏好写入 `user.md`，事实写入 `memory.md`，摘要写入 `summaries/每日.md`
- Token 统计按会话重置（`tracker.reset_session()`）
