# Multi-Agent 可行性分析

> 探索日期: 2026-06-03
> 状态: 已归档（当前不采用）

## 现有架构：已经是「主 Agent 通过 ReAct 调度子 Agent」

miniagent 当前架构本质就是主 Agent 通过 ReAct 循环中的工具调用来调度子 Agent：

```
用户 → 主 Agent (ReAct 循环)
         │
         ├─ Thought: "需要侦查代码结构"
         ├─ Action:  subagent_tool(agent="scout", task="侦查项目")
         ├─ Observe:  ← 侦查报告
         ├─ Thought: "让 reviewer 审查核心文件"
         ├─ Action:  subagent_tool(agent="reviewer", task="审查 runner.py")
         ├─ Observe:  ← 审查报告
         └─ Answer:  汇总回复用户
```

**已支持模式**（`SubagentTool`）：

| 模式 | 说明 |
|------|------|
| 单代理 | 派发一个任务给指定子代理 |
| 并行 | ThreadPoolExecutor，多个子代理并发执行 |
| 链式 | 串行执行，`{previous}` 传递上下文 |

## 「真正 Multi-Agent」需要的额外能力

| 能力 | 现状 | 缺口 |
|------|------|------|
| 子代理间直接通信 | ❌ | 只能对主 Agent 汇报 |
| 子代理递归调用 | ❌ | 子代理不能调子代理 |
| 辩论/迭代循环 | ❌ | 只有硬编码的 3 种模式 |
| 有状态子代理 | ❌ | 每次调用创建全新实例 |
| 共享上下文/黑板 | ❌ | 纯文本传递 |

## 子代理递归的风险

不加限制的递归会导致指数级 token 消耗和死锁：

```
主 Agent → 子代理 A → 子代理 B → 子代理 C → ... 无限递归
子代理 → 发现自己做不好 → 调用同名子代理（循环自身）
```

现有机制的保护不足：

| 现有保护 | 缺口 |
|----------|------|
| `max_turns` | 只管单代理内轮数，不管跨代理递归深度 |
| `subagent_max_turns` | 只限一层，深度 N × 每层 15 轮 = 爆炸 |
| `compact_threshold` | 触发时已经烧了大量 tokens |

### 必要护栏（如果未来开启递归）

1. **深度限制**（`_max_depth: int = 2`）— 硬阻断
2. **白名单制**（`allow_subagent: false` 默认关闭）
3. **循环检测**（链中同名 agent 直接阻断）
4. **总 token 预算**（超阈值全链熔断）

## 行业调研：主流 Coding Agent 全部是 ReAct

| 产品 | 架构 |
|------|------|
| Claude Code | 纯 ReAct |
| Cursor Agent | 纯 ReAct |
| GitHub Copilot Agent | 纯 ReAct |
| Aider | 纯 ReAct |
| Cline / Roo Code | 纯 ReAct |
| Windsurf Cascade | 纯 ReAct |
| OpenAI Codex CLI | 纯 ReAct |
| Amazon Q Developer | 纯 ReAct |
| Devin | 传言多 Agent（闭源未证实） |

## 为什么行业选 ReAct 而不是 Multi-Agent

1. **LLM 本身是编排器** — ReAct 中 LLM 在统一 context 里看全部历史，自主决策。多 Agent 反而打碎上下文。
2. **上下文是稀缺资源** — 多 Agent 意味着信息分片传递、压缩失真。单 Agent 一个 context 看全部。
3. **复杂度回报不成正比** — 多 Agent 唯一真实增量价值是并行执行，而 ToolExecutor 的并行模式已实现这一点（多工具并行，不需要多 Agent）。

## 结论

- miniagent 现有 ReAct + SubagentTool 架构**与行业主流一致**
- 子代理定位为**专用工具**（scout/reviewer），不扩展为 peer-to-peer 多 Agent
- 多 Agent 不是能力缺口，**ReAct 循环就是编排器，子代理就是专用工具**
- 优先级更高的方向：优化 prompt、工具并行执行、上下文压缩（已实现）
