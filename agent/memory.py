from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

from openai import OpenAI

from pydantic import BaseModel, ConfigDict, Field


class ConversationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    role: str
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMemory:
    """agent记忆系统,包含记忆的存储和压缩功能"""

    def __init__(self, memory_dir: Path, client: OpenAI, model: str, compact_k: int = 10):
        # 用于压缩的模型客户端和模型名称
        self.client = client
        self.model = model
        # 长期记忆存储，基于文件系统，三层记忆结构加上用户偏好
        self.memory_dir = Path(memory_dir)
        self.memory_file = self.memory_dir / "memory.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.summary_dir = self.memory_dir / "summaries"
        self.user_file = self.memory_dir / "user.md"
        self.history = []
        self.k = compact_k  # 每次压缩的历史消息数量
        # 确保目录和文件存在
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n此文件常驻上下文，记录核心目标、当前任务与关键事实。\n", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.write_text("", encoding="utf-8")
        if not self.user_file.exists():
            self.user_file.write_text("# 用户信息\n\n此文件记录用户的基本信息和偏好。\n", encoding="utf-8")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def _today_file(self, when: datetime | None = None) -> Path:
        moment = when or datetime.now()
        return self.summary_dir / f"{moment:%Y-%m-%d}.md"
    
    def brief_context(self) -> str:
        """获取系统提示词所需的上下文背景"""
        content = []
        # 1. 长期记忆核心内容 (memory.md)
        if self.memory_file.exists():
            text = self.memory_file.read_text(encoding="utf-8")
            valid_lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("#")]
            if valid_lines:
                content.append("## 核心记忆")
                content.extend(valid_lines[-15:])

        # 2. 最近的摘要 (summaries/*.md)
        if self.summary_dir.exists():
            summaries = sorted(self.summary_dir.glob("*.md"), reverse=True)
            if summaries:
                content.append("## 最近历史摘要")
                for s_file in summaries[:3]:
                    day = s_file.stem
                    text = s_file.read_text(encoding="utf-8")
                    content.append(f"### {day}")
                    # 去掉摘要文件中的大标题，直接显示内容
                    summary_lines = [l for l in text.split("\n") if not l.strip().startswith("#")]
                    content.append("\n".join(summary_lines).strip())

        return "\n".join(content) if content else "（暂无历史背景）"

    def user_preferences(self) -> list[str]:
        """获取用户偏好列表"""
        if not self.user_file.exists():
            return []
        lines = self.user_file.read_text(encoding="utf-8").split("\n")
        return [line.strip("- ").strip() for line in lines if line.strip().startswith("-")]

    def append_history(self, content: Any):
        """记录对话历史"""
        self.history.append(content)
        if not isinstance(content, dict):
            return

        role = content.get("role")
        if role == "system" or role == "tool":
            # 系统消息和工具中间结果不记入持久化日志，保持日志可读性
            return

        entry = ConversationEntry(
            created_at=self._now(),
            role=role or "unknown",
            content=content.get("content"),
            metadata={
                "reasoning_content": content.get("reasoning_content"),
                "tool_calls": content.get("tool_calls"),
            } if role == "assistant" else {}
        )

        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")

    def append_summary(self, critical: str, decision: str, issue: str) -> Path:
        """追加每日摘要"""
        today_file = self._today_file()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        mode = "a" if today_file.exists() else "w"
        with today_file.open(mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# 摘要 {today_file.stem}\n\n")
            f.write(f"## {timestamp}\n")
            f.write(f"- **关键**: {critical}\n")
            f.write(f"- **决策**: {decision}\n")
            f.write(f"- **问题**: {issue}\n\n")
        return today_file
    
    def add_memory(self, content: str) -> None:
        """添加长期记忆"""
        with self.memory_file.open("a", encoding="utf-8") as f:
            f.write(f"- {content.strip()}\n")

    def add_user(self, preference: str) -> None:
        """添加用户偏好"""
        with self.user_file.open("a", encoding="utf-8") as f:
            f.write(f"- {preference.strip()}\n")

    # --- 压缩与提取逻辑 ---

    def compact(self) -> dict[str, Any]:
        """执行压缩机制：总结对话并提取偏好"""
        # 如果历史太短（剔除系统消息后）则跳过
        effective_history = [m for m in self.history if m.get("role") != "system"]
        if len(effective_history) < 2:
            return {}

        # 1. 调用 LLM 统一提取
        data = self._extract_knowledge_with_llm(effective_history)
        
        # 2. 处理摘要 (Summary) -> 写入每日文件
        summary_info = data.get("summary", {})
        critical = self._truncate(summary_info.get("critical", "无"))
        decision = self._truncate(summary_info.get("decision", "无"))
        issue = self._truncate(summary_info.get("issue", "无"))
        summary_path = self.append_summary(critical, decision, issue)

        # 3. 处理用户偏好 (Preferences) -> 写入 user.md
        preferences = data.get("preferences", [])
        for p in preferences:
            self.add_user(p)

        # 4. 处理核心事实 (Facts) -> 写入 memory.md
        facts = data.get("facts", [])
        for f in facts:
            self.add_memory(f)

        return {
            "summary": {"critical": critical, "decision": decision, "issue": issue, "path": str(summary_path)},
            "preferences": preferences,
            "facts": facts,
        }

    def _formathistory(self, history: list[dict]) -> str:
        """格式化历史记录用于 LLM 总结"""
        lines: list[str] = []
        for msg in history:
            role = msg.get("role")
            if role == "system":
                continue
            
            content = msg.get("content") or ""
            
            # 特殊处理助手消息中的工具调用
            if role == "assistant" and msg.get("tool_calls"):
                tool_names = [tc.get("function", {}).get("name", "unknown") for tc in msg["tool_calls"]]
                content = f"[调用工具: {', '.join(tool_names)}] {content}".strip()
            
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _extract_knowledge_with_llm(self, history: list[dict]) -> dict[str, Any]:
        """利用 LLM 一次性提取摘要、偏好和事实"""

        prompt = f"""
            你是一个记忆提取专家。请分析以下对话，并提取关键信息。
            输出必须是严格的 JSON 格式，包含以下字段：
            1. summary: 对象，包含 critical(关键事件), decision(决策/产出), issue(心得/问题)。
            - 要求：每项极简，总字数 < 150 字。
            2. preferences: 字符串列表。记录用户明确表达的偏好、习惯或要求（如 '以后请用简短风格回答'）。
            3. facts: 字符串列表。记录值得长期记住的核心事实（如用户的职业、当前正在进行的大型项目名称等）。
            要分析的历史消息: {history[:-self.k]}
            注意：如果没有相关信息，对应的列表应为空，字段不能缺失。
        """
        # 历史消息过长时，只保留最近的 k 条进行分析，确保在模型上下文限制内
        history = history[-self.k :]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": self._formathistory(history)},
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content or "{}")
        except Exception as e:
            print(f"[Memory] 提取知识失败: {e}")
            return {
                "summary": {"critical": f"提取失败: {e}", "decision": "无", "issue": "无"},
                "preferences": [],
                "facts": []
            }

    def _truncate(self, text: str, limit: int = 60) -> str:
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[: limit - 1].rstrip() + "…"

    


    