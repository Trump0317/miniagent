# AGENTS.md — miniagent 项目上下文

## 项目概述

**miniagent** 是一个基于 LLM 的智能助手框架（约 3000 行 Python），采用 **ReAct 模式 + 事件驱动 + 树状会话**架构。支持工具调用、多 Provider 切换、子代理、会话分支/分叉、上下文压缩、三层记忆等功能。

- **语言**: Python 3.12+
- **依赖**: openai >= 2.0, pydantic >= 2.0, python-dotenv >= 1.0
- **入口**: `agent.py`（CLI Harness 层）
- **虚拟环境**: `.venv`
- **模型**: DeepSeek V4 Flash（默认）

## 架构总览

```
agent.py                          ← CLI 入口（交互/print 模式 + @文件引用 + 管道输入 + /fork /tree）
└── agent/
    ├── __init__.py                ← 公共导出
    ├── ai/                        ← AI 层
    │   ├── config.py              ← AppConfig: 多 Provider（DeepSeek/OpenAI/自定义）
    │   ├── llm.py                 ← LLMClient: OpenAI 兼容流式调用封装
    │   └── context.py             ← 自动加载 AGENTS.md/CLAUDE.md
    ├── core/                      ← 核心引擎
    │   ├── agent.py               ← Agent: 组件组装 + process() + shutdown()
    │   ├── runner.py              ← AgentRunner: think-act 循环编排
    │   ├── session_tree.py        ← SessionTree: 树状会话（分叉/导航/压缩节点）
    │   ├── events.py              ← EventBus: 发布/订阅 + 通配符 + 一次性监听
    │   ├── memory.py              ← AgentMemory: 树存储 + 三层记忆 + JSONL 持久化
    │   ├── compactor.py           ← Compactor: LLM 提取摘要/偏好/事实
    │   ├── tracker.py             ← TokenTracker: JSONL 日志 + 聚合统计
    │   ├── prompts.py             ← PromptLoader: /command 模板加载
    │   └── cli_helpers.py         ← handle_tree / handle_fork CLI 辅助函数
    ├── tools/                     ← 扁平工具集（每个工具一个 py 文件）
    │   ├── base.py                ← Tool 基类 + @tool 装饰器 + schema 瘦身
    │   ├── registry.py            ← ToolRegistry: 工具注册/查找/schema 生成
    │   ├── executor.py            ← ToolExecutor: 流式输出（yield）+ 串行/并行调度
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
2. **树状会话** — 底层 SessionTree，history 是树的当前视图；压缩插入 COMPACT 节点而非截断；分叉不丢数据
3. **存储与压缩分离** — `AgentMemory` 只做纯 I/O + 树操作，`Compactor` 只做 LLM 提取，互不依赖
4. **零中间层** — 无 Conversation/Hooks 抽象，`Agent` 直接编排 `Runner` + `Memory` + `Compactor`
5. **扁平工具** — 每个工具一个 py 文件，用 `@tool` 装饰器注入 name/description/args_model
6. **Agent 定义为文件** — 子代理通过 Markdown + YAML frontmatter 配置，无需改代码

## 请求处理流程

```
用户输入 (/command | 普通文本)
  → agent.py: 内置命令 (/fork /tree) 直接处理，不调 LLM
  → agent.py: _expand_command() → /command 展开为模板
  → Agent.process(message)
     → EventBus.emit("message:received")
     → memory.append_history(user message) → 双写: history 列表 + 树
     → AgentRunner.step(history)             # think-act 循环
        → EventBus.emit("turn:start")        # 触发 token 阈值检查
        → LLMClient.stream() → chunk 生成器   # 文本/tool_call/reasoning 逐块产出
        → tool_call arguments 逐字符拼接      # DeepSeek 分片修复
        → 无工具调用 → 返回
        → ToolExecutor.execute()
           → 单工具: 串行 + 流式 yield 实时透传
           → 多工具+全部parallel_safe: 并行(ThreadPoolExecutor, max 8) 静默收集
           → 多工具+非全部安全: 全部降级串行
           → tool:before / tool:after 事件钩子
        → history:appended → persist_message() → 树同步 + JSONL
        → 继续循环
  → Agent.shutdown()
     → Compactor.compact() 提取摘要/偏好/事实
     → memory.compress_tree() 插入 COMPACT 节点 + 全量持久化 JSONL
     → 偏好→user.md, 事实→memory.md, 摘要→summaries/每日.md
```

## 关键模块详解

### Agent（`agent/core/agent.py`）
- 装配所有组件，提供 `process()` 和 `shutdown()` 两个公共 API
- `process()` 返回 Generator[str]，支持流式输出
- `_build_system_prompt()` 构造系统提示词
- `_do_compact()` 调用 Compactor 提取 → `memory.compress_tree()` 插入 COMPACT 节点（替代旧的 `_trim_history` 截断）
- `_rebuild_system_prompt_with_summary()` 压缩后动态重建系统提示词
- `_format_compaction_summary()` 将提取结果格式化为摘要文本

### SessionTree（`agent/core/session_tree.py`）
**树状会话结构**，替代线性列表：
- `SessionEntry`: 树节点（id/parent_id/type/role/content/metadata/timestamp）
- `type` 有两种: `"message"`（普通消息）、`"compaction"`（压缩节点）
- 核心操作:
  - `append()` → 在 leaf 下创建子节点
  - `compact()` → 插入 COMPACT 节点 + 重排 parent_id（first_kept 重新挂到 COMPACT 下）
  - `fork(id)` / `navigate(id)` → 移动 leaf 指针
  - `path()` → 从 root 沿 parent_id 走到 leaf
  - `build_context()` → 构建 LLM 消息列表，遇 COMPACT 自动跳过已压缩消息

压缩语义:
```
  压缩前: a1 → q2 → a2 → tool → q3 → a3
            ↑ first_kept
  压缩后: a1 → COMPACT → q2 → a2 → tool → q3 → a3
  build_context 输出: [compaction_summary] + [q2, a2, tool, q3, a3]
```

### AgentMemory（`agent/core/memory.py`）
底层用 SessionTree，对外暴露 history 列表（Runner 兼容）：
- **写入双写**: `append_history()` → history 列表 + 树 + JSONL
- **压缩/分叉重建**: `compress_tree()` / `fork()` → 树操作 → `_history = tree.build_context()`
- **JSONL 格式**: `{id, parent_id, type, role, content, metadata, timestamp}`，自动检测并迁移旧格式
- **三层记忆**（不变）:
  | 层 | 文件 | 用途 |
  |---|---|---|
  | 短期 | `history.jsonl` | 树结构对话历史 |
  | 中期 | `summaries/YYYY-MM-DD.md` | 每日压缩摘要 |
  | 长期 | `memory.md` | 核心事实，常驻上下文 |
  | 用户 | `user.md` | 偏好列表，自动去重 |

### AgentRunner（`agent/core/runner.py`）
- think-act 循环编排，不持有状态
- tool_call arguments 修复: DeepSeek 流式返回 arguments 是逐字符分片的，Runner 用 `+=` 拼接而非 `=` 覆盖
- 工具输出: 串行逐块 yield（实时流式），并行静默收集

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
- **Schema 瘦身** (`_minify_schema`): 移除 title/default/additionalProperties，展开 anyOf null 模式
- **ToolExecutor**: `_run_one()` 从 `return str` 改为 `yield str`（生成器），串行模式逐块透传实时输出，并行模式静默收集
- **SubagentTool**: 单/并行/链式三种模式，每个子代理独立上下文和 TokenTracker

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
- 管道输入（`cat README.md | python agent.py -p "总结"`）
- `/command` 模板展开（`/scout`, `/review`）
- **`/fork [n]`** — 分叉到第 n 条用户消息（默认倒数第 2 条），创建新分支
- **`/tree`** — 显示会话分支树可视化

### 内置命令流程

`/fork` 和 `/tree` 不经过 LLM，直接在 CLI 层处理：
1. `handle_fork()` → `agent.memory.fork(id)` → 树 leaf 指针移动 → `agent.history = memory.history`
2. `handle_tree()` → 遍历树节点 → 打印带缩进的树状视图（含 `[压缩]` 和 `← 当前` 标记）

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

## 会话持久化

- 每条消息追加时增量写入 JSONL（树格式: id/parent_id/type）
- 压缩时全量重写 JSONL（DFS 序，保证 parent 在 child 前）
- 启动时 `restore_tree()` 自动检测旧格式并迁移
- 压缩提取: 偏好→user.md, 事实→memory.md, 摘要→summaries/每日.md
- Token 统计按会话重置（`tracker.reset_session()`）

## 注意事项

- `.memory/` 和 `skills/` 在 `.gitignore` 中排除
- 系统消息不持久化到树（每次启动重建 system prompt）
- Tool result 截断: 50KB / 2000 行
- 压缩阈值: `last_input_tokens > max_context * compact_threshold`（默认 200000 * 0.35 = 70K tokens）
- `max_turns` 为 None 时不设上限
- DeepSeek 流式 tool_call arguments 是逐字符分片，Runner 需用 `+=` 拼接
