# AGENTS.md — miniagent 项目上下文

## 项目概述

**miniagent** 是一个基于 LLM 的智能助手框架（约 5000 行 Python），采用 **ReAct 模式 + 事件驱动 + 树状会话**架构。支持 CLI / TUI / Web 三种交互方式。

- **语言**: Python 3.12+
- **安装**: `pip install -e ".[all]"`，入口命令 `miniagent`
- **核心依赖**: openai >= 2.0, pydantic >= 2.0, python-dotenv >= 1.0
- **可选依赖**: prompt_toolkit（TUI）、fastapi + websockets（Web）、mcp（MCP）
- **模型**: DeepSeek V4 Flash（默认），支持 OpenAI / 自定义 Provider
- **测试**: 552 个（523 单元 + 14 集成 + 15 可观测性）

## 架构总览

```
agent.py                          ← 兼容入口（pyproject.toml 提供 miniagent 命令）
└── agent/
    ├── __init__.py                ← main() 入口 + 公共 API 导出
    ├── ai/                        ← AI 层
    │   ├── config.py              ← AppConfig: 多 Provider + 会话隔离 + MCP 配置
    │   ├── llm.py                 ← LLMClient: OpenAI 兼容流式调用封装
    │   └── context.py             ← 自动加载 AGENTS.md/CLAUDE.md
    ├── cli/                       ← CLI 外壳
    │   ├── app.py                 ← CLI 应用程序（交互/print 模式、readline、argparse）
    │   └── helpers.py             ← handle_tree / handle_fork / handle_back
    ├── tui/                       ← TUI 终端界面（基于 prompt_toolkit）
    │   └── app.py                 ← TuiApp: 对话气泡、流式输出、分叉/返回
    ├── core/                      ← 核心引擎
    │   ├── agent.py               ← Agent: 装配层 + set_model/set_thinking/set_max_turns
    │   ├── chunks.py              ← ChunkType 枚举 + 工厂函数
    │   ├── runner.py              ← AgentRunner: think-act 循环 + LLM 重试
    │   ├── session_tree.py        ← SessionTree: 树状会话（分叉/导航/压缩节点）
    │   ├── system_prompt.py       ← SystemPrompt: 系统提示词构建（实时查询 memory）
    │   ├── system_prompt.md       ← 系统提示词模板
    │   ├── compaction.py          ← CompactionService: 压缩编排
    │   ├── events.py              ← EventBus: 发布/订阅 + 通配符 + 一次性监听
    │   ├── memory.py              ← AgentMemory: 树存储（单源）+ 三层记忆 + JSONL
    │   ├── observability.py       ← 结构化日志 + trace ID + 耗时统计
    │   ├── tracker.py             ← TokenTracker: JSONL 日志 + 聚合统计
    │   └── prompts.py             ← PromptLoader: /command 模板加载
    ├── tools/                     ← 扁平工具集（每个工具一个 py 文件）
    │   ├── base.py                ← Tool 基类 + @tool 装饰器 + schema 瘦身
    │   ├── registry.py            ← ToolRegistry + build_default_registry()
    │   ├── executor.py            ← ToolExecutor: 流式输出 + 串行/并行调度
    │   ├── mcp_client.py          ← MCP 客户端（stdio 传输 + 工具发现 + 异步桥接）
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
    ├── prompts/                   ← 命令模板
    │   ├── scout.md
    │   └── review.md
    └── web/                       ← Web UI（FastAPI + WebSocket）
        ├── server.py              ← FastAPI 服务 + REST + WS 端点
        ├── session.py             ← 多会话管理
        └── static/index.html      ← 聊天界面
```

## 核心设计原则

1. **事件驱动解耦** — EventBus 替代回调链，支持通配符（`tool:*`）
2. **树状会话** — SessionTree 管理分支，history 实时计算；压缩插入 COMPACT 节点不截断
3. **存储与压缩分离** — AgentMemory 只做 I/O + 树操作，CompactionService 含 LLM 提取全编排
4. **无列表双写** — AgentMemory 以树为唯一数据源
5. **扁平工具** — 每个工具一个 py 文件，`@tool` 装饰器注入元数据
6. **Agent 定义为文件** — 子代理用 Markdown + YAML，MCP 用 mcp.json
7. **统一 chunk** — 所有组件产出 `{type, content}` 格式，CLI/TUI/Web 按 type 消费

## 请求处理流程

```
用户输入 (/command | 普通文本)
  → 内置命令 (/help /config /model /thinking /turns /tree /fork /back /clear) 直接处理
  → /command 展开为模板
  → Agent.process(message)
     → EventBus.emit("message:received")       ← 可观测性起点
     → memory.append_message(user msg) → 树 + JSONL
     → Agent._build_context() → [system] + memory.history
     → AgentRunner.step(context)
        → EventBus.emit("turn:start")          ← 压缩阈值检查
        → LLMClient.stream() → chunk 生成器     ← 含自动重试（最多 3 次，指数退避）
        → tool_call arguments 逐字符拼接        ← DeepSeek 分片修复
        → 无工具调用 → done → 返回
        → ToolExecutor.execute()
           → 单工具: 串行流式实时透传
           → 多工具+全部parallel_safe: 并行(ThreadPoolExecutor, max 8)
           → 多工具+非全部安全: 全部降级串行
           → tool:before / tool:after 事件钩子
        → context.append(assistant/tool msg)
        → 继续循环
     → 扫描增量 → memory.append_message(msg)
  → Agent.shutdown()
     → bus.emit("session:end") → observability 输出汇总
     → CompactionService.compact() → LLM 提取 → 分发 → 压缩 → 提示词重建
     → McpClientManager.shutdown()
```

## 关键模块详解

### Agent（`agent/core/agent.py`）
- 纯装配层，创建并连接所有组件
- `process()` 流式处理，`shutdown()` 执行压缩 + MCP 清理
- 运行时配置方法：`set_model()` / `set_thinking()` / `set_max_turns()`

### Chunk 协议（`agent/core/chunks.py`）
- `ChunkType` (StrEnum): `TEXT` / `REASONING` / `TOOL_STATUS` / `TOOL_RESULT` / `DONE`
- 工厂函数产出 `{"type": ..., "content": ...}` 格式
- CLI/TUI/Web 消费者按 type 分发

### SessionTree（`agent/core/session_tree.py`）
- `SessionEntry`: id/parent_id/type/role/content/metadata/timestamp
- `type`: "message" | "compaction"
- `build_context()` 遇 COMPACT 跳过已压缩消息

### AgentMemory（`agent/core/memory.py`）
- `history` → `tree.build_context()` 实时计算
- fork 跳转栈: `push_fork()` / `pop_fork()`
- 三层记忆: 短期（history.jsonl/tokens.jsonl/trace.jsonl）、中期（summaries/）、长期（memory.md/user.md）

### AgentRunner（`agent/core/runner.py`）
- think-act 循环编排
- **LLM 重试**: APIConnectionError/RateLimitError/APITimeoutError/InternalServerError 自动重试 3 次
- 不完整工具调用（id/name=None）自动过滤

### CompactionService（`agent/core/compaction.py`）
- Token 超阈值（默认 200000 × 35% = 70K）触发
- 提取 → 分发 → 树压缩 → 提示词重建

### Observability（`agent/core/observability.py`）
- 通过 EventBus 订阅实现，零侵入
- trace ID（8 位 hex）贯穿全请求
- JSON Lines 输出到 `trace.jsonl`
- 记录：request/start、turn/start、tool/start、tool/end、request/summary

### MCP 支持（`agent/tools/mcp_client.py`）
- stdio 传输，配置文件 `mcp.json`（Claude Code 兼容格式）
- 后台线程运行 asyncio event loop，异步 SDK 桥接
- 工具命名：`mcp__<server>__<tool>`
- 连接失败跳过并警告

### EventBus（`agent/core/events.py`）
| 事件 | 触发 | data |
|------|------|------|
| `message:received` | agent.py | `{text}` |
| `turn:start` | runner.py | `{turn}` |
| `turn:end` | runner.py | `{text}` |
| `tool:before` | executor.py | `{name, args}` |
| `tool:after` | executor.py | `{name, result}` |
| `context:high` | agent.py | event |
| `session:end` | agent.py | `{}` |

## 配置（`agent/ai/config.py`）

```python
@dataclass
class AppConfig:
    root: Path = Path.cwd()            # 用户项目根目录
    provider: str = "deepseek"
    model: str
    max_turns: int | None
    max_tokens: int = 20000
    max_context: int = 200000
    compact_threshold: float = 0.35
    subagent_model: str
    subagent_max_turns: int = 15
    restore_session: bool = False
    session_id: str
    context_files: str
    mcp_config_path: str = ""          # mcp.json 路径
```

### 会话隔离（用户数据在项目根目录）

```
.memory/                         ← 项目根目录
├── memory.md                    ← 长期记忆
├── user.md                      ← 用户偏好
├── summaries/                   ← 每日摘要
└── sessions/
    └── <session_id>/
        ├── history.jsonl        ← 树结构历史
        ├── tokens.jsonl         ← Token 日志
        └── trace.jsonl          ← 可观测性日志
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示所有命令 |
| `/config` | 显示当前配置和 token 统计 |
| `/model [name]` | 显示/切换模型 |
| `/thinking [level]` | 显示/切换思考级别 |
| `/turns [n]` | 显示/设置最大轮数（0=无限制） |
| `/tree` | 显示分支树 |
| `/fork [n]` | 分叉到第 n 条消息前 |
| `/back` | 返回分叉前 |
| `/clear` | 清屏 |

命令行参数：`-p` print 模式，`-r` 恢复会话，`--tui` TUI 模式，`--web` Web 模式，`--thinking` 级别，`-nc` 禁用上下文，`@file` 引用，管道输入。

## TUI 快捷键

| 键 | 功能 |
|----|------|
| Ctrl+Q | 退出 |
| Ctrl+F | 分叉 |
| Ctrl+B | 返回 |
| Ctrl+T | 分支树 |
| Ctrl+E | 展开/折叠工具 |
| Esc | 退出树/清空输入 |

## Web UI

FastAPI + WebSocket，单文件 HTML 前端。

API 端点：`/api/sessions`、`/api/sessions/{id}/settings`、`/api/upload`、`/api/upload/resolve`、`/ws`

前端功能：模型切换、文件上传/拖放、@引用解析、分支树、Markdown 渲染。

## 安全措施

- BashTool: 正则拦截 `rm -rf /`, `mkfs`, `dd`, `chmod 777 /`, fork 炸弹
- ToolExecutor: `tool:before` 事件可阻断工具执行
- 子代理: 独立 ToolRegistry 深拷贝，线程隔离
- MCP: 连接失败跳过，不影响内置工具

## 扩展点

1. **新工具**: `agent/tools/` 下创建 py 文件，`@tool` 装饰器，`build_default_registry()` 注册
2. **新子代理**: `agent/subagent/` 下创建 .md 文件（YAML + Markdown）
3. **新命令模板**: `agent/prompts/` 下创建 .md 文件
4. **MCP 服务器**: 项目根目录创建 `mcp.json`
5. **自定义 Provider**: `PROVIDER_PRESETS` 添加预设

## 注意事项

- `.memory/` 和 `skills/` 在 `.gitignore` 排除
- 子代理/MCP 工具不挂载到子代理
- Tool result 截断: 50KB / 2000 行
- 压缩阈值: 70K tokens（200000 × 0.35）
- DeepSeek 流式 tool_call arguments 需用 `+=` 拼接
- 用户数据都在 `root`（`Path.cwd()`），框架资源从包目录加载

## 测试

```bash
python -m unittest discover tests    # 552 个测试
```

| 类型 | 文件 | 用例 |
|------|------|------|
| 单元 | 21 个模块 | 523 |
| 集成 | test_integration.py | 14 |
| 可观测性 | test_observability.py | 15 |
| MCP 客户端 | test_mcp_client.py | 20 |
