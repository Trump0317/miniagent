# AGENTS.md — miniagent 项目上下文

## 项目概述

**miniagent** 是一个基于 LLM 的智能助手框架（约 3000 行 Python），采用 **ReAct 模式 + 事件驱动 + 树状会话**架构。支持工具调用、多 Provider 切换、子代理、会话分支/分叉、上下文压缩、三层记忆等功能。

- **语言**: Python 3.12+
- **核心依赖**: openai >= 2.0, pydantic >= 2.0, python-dotenv >= 1.0
- **可选依赖（TUI）**: prompt_toolkit >= 3.0, rich >= 13.0
- **可选依赖（Web）**: fastapi, websockets
- **入口**: `agent.py`（CLI / TUI 分发）
- **虚拟环境**: `.venv`
- **模型**: DeepSeek V4 Flash（默认）

## 架构总览

```
agent.py                          ← 轻量入口（CLI --tui 分发）
└── agent/
    ├── __init__.py                ← 公共导出
    ├── ai/                        ← AI 层
    │   ├── config.py              ← AppConfig: 多 Provider（DeepSeek/OpenAI/自定义）+ 会话隔离
    │   ├── llm.py                 ← LLMClient: OpenAI 兼容流式调用封装
    │   └── context.py             ← 自动加载 AGENTS.md/CLAUDE.md
    ├── cli/                       ← CLI 外壳
    │   ├── __init__.py            ← 公共导出
    │   ├── app.py                 ← CLI 应用程序（交互/print 模式、readline、argparse）
    │   └── helpers.py             ← handle_tree / handle_fork / handle_back CLI 辅助函数
    ├── tui/                       ← TUI 终端界面（基于 prompt_toolkit）
    │   ├── __init__.py
    │   └── app.py                 ← TuiApp: 对话气泡、流式输出、分叉/返回
    ├── core/                      ← 核心引擎
    │   ├── agent.py               ← Agent: 组装组件 + process() + shutdown()（160 行）
    │   ├── chunks.py              ← ChunkType 枚举 + text_chunk() 等工厂函数
    │   ├── runner.py              ← AgentRunner: think-act 循环编排
    │   ├── session_tree.py        ← SessionTree: 树状会话（分叉/导航/压缩节点）
    │   ├── system_prompt.py       ← SystemPrompt: 系统提示词构建（实时查询 memory 动态部分）
    │   ├── system_prompt.md       ← 系统提示词模板
    │   ├── compaction.py          ← CompactionService: 压缩编排
    │   ├── events.py              ← EventBus: 发布/订阅 + 通配符 + 一次性监听
    │   ├── memory.py              ← AgentMemory: 树存储（单源）+ 三层记忆 + JSONL 持久化
    │   ├── tracker.py             ← TokenTracker: JSONL 日志 + 聚合统计
    │   └── prompts.py             ← PromptLoader: /command 模板加载
    ├── tools/                     ← 扁平工具集（每个工具一个 py 文件）
    │   ├── base.py                ← Tool 基类 + @tool 装饰器 + schema 瘦身
    │   ├── registry.py            ← ToolRegistry + build_default_registry()
    │   ├── executor.py            ← ToolExecutor: 流式输出 + 串行/并行调度
    │   ├── bash.py                ← BashTool
    │   ├── file_read.py           ← 文件读取
    │   ├── file_write.py          ← 文件写入
    │   ├── file_edit.py           ← 文件编辑
    │   ├── web_fetch.py           ← 网页抓取
    │   ├── web_search.py          ← 网络搜索
    │   ├── skill.py               ← SkillsLoader + SkillTool
    │   ├── todo.py                ← TodoWriteTool
    │   └── subagent.py            ← SubagentRunner + SubagentTool
    ├── subagent/                  ← 子代理定义（Markdown + YAML frontmatter）
    │   ├── scout.md
    │   └── reviewer.md
    └── prompts/                   ← 命令模板
        ├── scout.md
        └── review.md
```

## 核心设计原则

1. **事件驱动解耦** — EventBus 替代回调链：`turn:start` / `context:high` / `tool:before` / `tool:after`
2. **树状会话** — 底层 SessionTree，history 是树的实时计算视图；压缩插入 COMPACT 节点而非截断；分叉不丢数据
3. **存储与压缩分离** — `AgentMemory` 只做纯 I/O + 树操作，`CompactionService` 包含 LLM 提取 + 分发 + 树压缩 + 提示词重建全部编排
4. **无列表双写** — `AgentMemory` 以 SessionTree 为唯一数据源，`history` 是 `tree.build_context()` 的实时计算结果
5. **扁平工具** — 每个工具一个 py 文件，用 `@tool` 装饰器注入 name/description/args_model
6. **Agent 定义为文件** — 子代理通过 Markdown + YAML frontmatter 配置，无需改代码

## 请求处理流程

```
用户输入 (/command | 普通文本)
  → agent.py: 内置命令 (/fork /back /tree) 直接处理，不调 LLM
  → agent.py: _expand_command() → /command 展开为模板
  → Agent.process(message)
     → EventBus.emit("message:received")
     → memory.append_message(user msg) → 树追加 + JSONL 增量写
     → Agent._build_context() → [system] + memory.history 快照
     → AgentRunner.step(context)              # think-act 循环
        → EventBus.emit("turn:start")         # 触发 token 阈值检查
        → LLMClient.stream() → chunk 生成器    # 文本/tool_call/reasoning 逐块产出
        → tool_call arguments 逐字符拼接       # DeepSeek 分片修复
        → 无工具调用 → 返回
        → ToolExecutor.execute()
           → 单工具: 串行 + 流式 yield 实时透传
           → 多工具+全部parallel_safe: 并行(ThreadPoolExecutor, max 8) 静默收集
           → 多工具+非全部安全: 全部降级串行
           → tool:before / tool:after 事件钩子
        → context.append(assistant/tool msg)  # Runner 在快照上直接操作
        → 继续循环
     → 扫描 context 增量 → memory.append_message(msg) → 树 + JSONL
  → Agent.shutdown()
     → CompactionService.compact()
        → _find_cut_point() token-aware 切点
        → _extract() LLM 提取摘要/偏好/事实（含迭代压缩上下文）
        → memory 分发: 偏好→user.md, 事实→memory.md, 摘要→summaries/
        → memory.compress_tree() 插入 COMPACT 节点 + 全量持久化 JSONL
        → SystemPrompt.build(data) 重建含压缩摘要的系统提示词
```

## 关键模块详解

### Agent（`agent/core/agent.py`）
- 纯装配层（~160 行），创建并连接所有组件
- `process()` 流式处理用户消息，Runner 在 context 快照上工作，末端统一同步到 memory
- `shutdown()` 调用 CompactionService.compact() 执行压缩

**内部组件**（独立文件）：
- **SystemPrompt** (`system_prompt.py`): 持有静态上下文引用，`build(compaction_data)` 实时查询 memory 动态部分
- **CompactionService** (`compaction.py`): 编排完整压缩流程（提取 → 分发 → 树压缩 → 重建提示词）
- **`build_default_registry()`** (`tools/registry.py`): 构建 Agent 和子代理的工具注册表

### Chunk 协议（`agent/core/chunks.py`）
- `ChunkType` (StrEnum): `TEXT` / `REASONING` / `TOOL_STATUS` / `TOOL_RESULT` / `DONE`
- `text_chunk()` / `reasoning_chunk()` / `tool_status_chunk()` / `tool_result_chunk()` / `done_chunk()` 工厂函数
- Runner 和 Executor 产出 chunk，CLI/Web/TUI 消费者按 type 分发
- `REASONING` chunk 用于 LLM 思考/推理内容（DeepSeek thinking），TUI 以斜体灰色展示

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
**树为唯一数据源**，无列表双写：
- `history` (property) → `tree.build_context()` 实时计算，分叉/压缩后自动反映
- **唯一写入入口**: `append_message(msg)` → 树追加 + JSONL 增量写
- **fork 跳转栈**: `push_fork()` / `pop_fork()` 支持 `/back` 返回
- **JSONL 格式**: `{id, parent_id, type, role, content, metadata, timestamp}`，自动检测并迁移旧格式
- **三层记忆**:
  | 层 | 文件 | 用途 |
  |---|---|---|
  | 短期 | `sessions/<ts>/history.jsonl` | 树结构对话历史（按会话隔离） |
  | 短期 | `sessions/<ts>/tokens.jsonl` | Token 消耗日志（按会话隔离） |
  | 中期 | `summaries/YYYY-MM-DD.md` | 每日压缩摘要（跨会话共享） |
  | 长期 | `memory.md` | 核心事实，常驻上下文（跨会话共享） |
  | 用户 | `user.md` | 偏好列表，自动去重（跨会话共享） |

### AgentRunner（`agent/core/runner.py`）
- think-act 循环编排，不持有状态
- 不关心持久化（Agent 在 process() 末端统一处理）
- tool_call arguments 修复: DeepSeek 流式返回 arguments 是逐字符分片的，Runner 用 `+=` 拼接而非 `=` 覆盖
- 工具输出: 串行逐块 yield（实时流式），并行静默收集

### CompactionService（`agent/core/compaction.py`）
- 单一模块覆盖完整压缩流程，已合并原 Compactor 的 LLM 提取逻辑
- `should_compact()`: 检查 token 是否超过阈值
- `compact()`: 编排一次完整压缩
  1. `_find_cut_point()` — token-aware 切点：从 leaf 往回累积 token 估算，在用户消息边界切割
  2. `_extract()` — LLM 提取 summary / preferences / facts（支持迭代压缩：后续压缩传入前次摘要作为上下文）
  3. `_dispatch_to_memory()` — 分发到三层记忆（偏好→user.md，事实→memory.md，摘要→summaries/）
  4. `memory.compress_tree()` — 插入 COMPACT 节点 + JSONL 持久化
  5. `prompt.build(data)` — 重建系统提示词

### TokenTracker（`agent/core/tracker.py`）
- 按调用的 JSONL 日志（ts/model/input/output/cache_hit/cache_miss）
- 目录延迟创建（首次 record() 时）
- `should_compact()`: 当 last_input_tokens > max_context * threshold（默认 35%）触发
- 支持 `stats_by_model()` / `stats_by_date()` 聚合

### 工具系统
- **@tool 装饰器**: 注入 `_tool_name` / `_tool_description` / `_args_model`
- **Schema 瘦身** (`_minify_schema`): 仅删除 title/additionalProperties，**保留** anyOf[{X},{null}] 和 default 值，帮助 LLM 准确识别可选参数
- **ToolExecutor**: `_run_one()` 使用生成器 `yield str`，串行模式逐块透传实时输出，并行模式静默收集
- **SubagentTool**: SubagentRunner 封装子代理创建/运行逻辑；Tool 自身只负责配置解析和模式调度
- **错误提示**: Registry.call_tool 检测 `"error" in result.lower()`，为所有工具错误追加纠错提示
- **BashTool**: 流式执行无输出时返回 `"[命令执行成功，无输出。]"` 避免 LLM 困惑

### EventBus（`agent/core/events.py`）
核心事件：
- `turn:start` → 检查 token 阈值，触发压缩
- `context:high` → 执行压缩
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
    restore_session: bool = False  # 默认不恢复，用 -r 恢复
    session_id: str        # 会话标识（自动生成时间戳）
    context_files: str     # AGENTS.md 内容
```

Provider 自动检测：DEEPSEEK_API_KEY → OPENAI_API_KEY → API_KEY（custom）

### 会话隔离

每个会话独立存储目录：
```
agent/.memory/
├── memory.md              ← 长期记忆（跨会话）
├── user.md                ← 用户偏好（跨会话）
├── summaries/             ← 每日摘要（跨会话）
└── sessions/
    ├── 20260528-093000/   ← 会话 1（自动生成时间戳）
    │   ├── history.jsonl
    │   └── tokens.jsonl
    └── 20260528-150000/   ← 会话 2
        └── history.jsonl
```

- 默认每次启动为新会话（`restore_session=False`）
- `-r` / `--restore` 恢复最近一次有内容的会话
- 空会话不创建目录

## CLI 特性

CLI 代码已从 `agent.py` 提取到 `agent/cli/`:
- `app.py`: 交互/print 模式、readline 历史、argparse、/command 展开、管道输入
- `helpers.py`: `handle_tree()` / `handle_fork()` / `handle_back()`

`agent.py` 现为轻量入口（7 行），通过 `--tui` 标志分发到 CLI 或 TUI 模式。

- `-p` print 模式（非交互）
- `-r` / `--restore` 恢复最近会话历史
- `--tui` 启动 TUI 模式
- `--thinking off|minimal|low|medium|high|xhigh` 思维链级别
- `-nc` 禁用上下文文件
- `@file.py` 文件引用展开
- 管道输入（`cat README.md | python agent.py -p "总结"`）
- `/command` 模板展开（`/scout`, `/review`）
- **`/fork [n]`** — 分叉到第 n 条用户消息之前
- **`/back`** — 返回分叉前的位置
- **`/tree`** — 显示会话分支树可视化

## TUI 特性

基于 prompt_toolkit 的终端界面（`agent/tui/app.py`），对齐 pi-tui 风格：
- 对话气泡，不同角色用不同颜色标签：**You** 蓝 / **Agent** 绿 / **Thinking** 灰斜体 / **Tool** 按状态着色
- 思考内容实时流式显示（`REASONING` chunk），斜体灰色，完成时标签变 **Thought**
- 工具结果按状态着色：执行中蓝底、成功绿底、错误红底
- 流式输出 + spinner 动画
- 工具结果可折叠（Ctrl+E 展开/折叠）
- 自动滚动到底部（光标驱动），鼠标滚轮 + PageUp/PageDown 翻页
- 快捷键：Ctrl+Q 退出、Ctrl+F 分叉、Ctrl+B 返回、Ctrl+T 分支树、Ctrl+E 展开工具
- Esc 退出树视图 / 清空输入
- Header 显示模型 + 会话 ID

### 内置命令流程

`/fork`、`/back` 和 `/tree` 不经过 LLM，直接在 CLI 层处理：
1. `handle_fork()` → `memory.push_fork()` 保存当前位置 → `memory.fork(target.parent_id)` → leaf 移到目标消息之前
2. `handle_back()` → `memory.pop_fork()` 弹栈 → `memory.navigate(id)` 返回
3. `handle_tree()` → 遍历树节点 → 打印带缩进的树状视图（含 `[压缩]` 和 `← 当前` 标记）

## 安全措施

- BashTool: 正则拦截 `rm -rf /`, `mkfs`, `dd`, `chmod 777 /`, fork 炸弹等
- ToolExecutor: `tool:before` 事件可被消费者阻断任意工具执行
- 子代理: 独立 ToolRegistry 深拷贝，线程隔离

## 扩展点

1. **新工具**: 在 `agent/tools/` 下创建新 py 文件，用 `@tool` 装饰器，在 `build_default_registry()` 中注册
2. **新子代理**: 在 `agent/subagent/` 下创建 .md 文件（YAML frontmatter + Markdown body）
3. **新命令模板**: 在 `agent/prompts/` 下创建 .md 文件
4. **新事件监听器**: 在 `Agent._setup_events()` 中注册
5. **自定义 Provider**: 在 `PROVIDER_PRESETS` 中添加预设

## 会话持久化

- 会话隔离: 每会话独立 `sessions/<ts>/history.jsonl` + `tokens.jsonl`
- 每条消息追加时增量写入 JSONL（树格式: id/parent_id/type）
- 压缩时全量重写 JSONL（DFS 序，保证 parent 在 child 前）
- 启动时 `restore_tree()` 自动检测旧格式并迁移
- 压缩提取: 偏好→user.md, 事实→memory.md, 摘要→summaries/每日.md
- 空会话不创建目录，不残留文件

## 注意事项

- `.memory/` 和 `skills/` 在 `.gitignore` 中排除
- 系统消息不持久化到树（每次启动由 SystemPrompt.build() 重建）
- Tool result 截断: 50KB / 2000 行
- 压缩阈值: `last_input_tokens > max_context * compact_threshold`（默认 200000 * 0.35 = 70K tokens）
- `max_turns` 为 None 时不设上限
- DeepSeek 流式 tool_call arguments 是逐字符分片，Runner 需用 `+=` 拼接

## 测试

测试采用先单元后集成的策略，每个模块编写后经过三重审查（自审 → Subagent 审 → 人工审）。

### 运行

```bash
# 运行全部测试
python -m unittest discover tests

# 运行单个模块
python -m unittest tests.test_events
```

### 已覆盖模块（截至 2026-06-02）

| 模块 | 文件 | 测试数 | 类型 |
|------|------|--------|------|
| EventBus | `test_events.py` | 28 | 单元 |
| SessionTree | `test_session_tree.py` | 58 | 单元 |
| AgentMemory | `test_memory.py` | 59 | 单元 |
| TokenTracker | `test_tracker.py` | 27 | 单元 |
| PromptLoader | `test_prompts.py` | 21 | 单元 |
| SystemPrompt | `test_system_prompt.py` | 17 | 单元 |
| CompactionService | `test_compaction.py` | 29 | 单元 |
| AgentRunner | `test_runner.py` | 16 | 单元 |
| Tool 基类 | `test_tool_base.py` | 28 | 单元 |
| ToolRegistry | `test_registry.py` | 24 | 单元 |
| FileReadTool | `test_file_read.py` | 7 | 单元 |
| FileWriteTool | `test_file_write.py` | 7 | 单元 |
| FileEditTool | `test_file_edit.py` | 16 | 单元 |
| SkillsLoader + SkillTool | `test_skill.py` | 16 | 单元 |
| TodoWriteTool | `test_todo.py` | 30 | 单元 |
| BashTool | `test_bash.py` | 20 | 单元 |
| ToolExecutor | `test_executor.py` | 19 | 单元 |
| WebFetchTool | `test_web_fetch.py` | 7 | 单元 |
| WebSearchTool | `test_web_search.py` | 7 | 单元 |
| AgentLoader + SubagentTool | `test_subagent.py` | 45 | 单元 |
| TUI | `test_tui.py` | 74 | 单元 |
| **合计** | | **550** | |

## 待办计划

### 1. 优化提示词 ✅
- [x] 审查并优化 SystemPrompt 的系统提示词内容 → 外部模板文件
- [x] 优化子代理（scout / reviewer）的提示词 → 结构化流程+清单
- [x] 优化命令模板（/scout / /review）的展开模板 → 细化指引
- [x] 考虑添加任务分解/规划相关的提示词引导

### 2. 实现 Web UI ✅
- [x] FastAPI + WebSocket 流式对话
- [x] Agent 核心统一 chunk 格式（text/tool_status/done），CLI/Web 共用
- [x] 多会话管理、实时流式输出、Markdown 渲染
- [x] 文件上传和 @引用
- [x] 模型/Provider 切换

### 3. 优化 CLI ✅
- [x] CLI 代码提取到 `agent/cli/`，`agent.py` 瘦身为 7 行入口
- [x] readline 命令历史、/help /session /clear 命令
- [x] 配置管理命令

### 4. Core 重构 ✅
- [x] `session_tree.py`: 删除内嵌测试（532→326 行）
- [x] `memory.py`: 关闭 SessionTree 私有成员访问（add_entry/set_leaf/find_deepest_leaf）
- [x] `memory.py`: `non_system_entries()` 委托给 `tree.to_messages()`
- [x] `agent.py`: 提取 `_build_registry()` → `tools/registry.py`
- [x] `memory.py`: 提取 `_entry_to_row()` 消除 JSONL 序列化重复
- [x] `runner.py` + `executor.py`: `ChunkType` 枚举 + 工厂函数
- [x] `compaction.py`: `_last_usage` 隐式状态 → 显式参数

### 5. TUI 终端界面 ✅
- [x] 创建 `agent/tui/` 模块（基于 prompt_toolkit）
- [x] 对话气泡、流式输出、spinner 动画
- [x] 快捷键：Ctrl+Q 退出、Ctrl+F 分叉、Ctrl+B 返回、Ctrl+T 分支树、Ctrl+E 展开工具
- [x] 修复自动滚动到底部（render_info 动态计算 scroll offset）
- [x] 工具结果折叠/展开交互（Ctrl+E 切换）
- [x] 树视图 Esc 退出、发送消息自动退出
- [x] 测试覆盖（65 个测试）
