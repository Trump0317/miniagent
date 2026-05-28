"""记忆压缩器 —— 从对话历史中提取摘要、偏好和事实。

纯处理逻辑：接收历史，返回结构化结果。不涉及文件 I/O。
"""

from __future__ import annotations
from typing import Any
import json
from openai import OpenAI


class Compactor:
    """利用 LLM 从对话历史中提取关键信息。

    用法:
        compactor = Compactor(client, model)
        result = compactor.compact(history)
        # result = {"summary": {...}, "preferences": [...], "facts": [...]}
    """

    _PROMPT = (
        "你是一个记忆提取专家。请分析以下对话，并提取关键信息。\n"
        "输出必须是严格的 JSON 格式，包含以下字段：\n"
        "1. summary: 对象，包含 critical(关键事件), decision(决策/产出), issue(心得/问题)。\n"
        "   - 要求：每项极简，总字数 < 150 字。\n"
        "2. preferences: 字符串列表。记录用户明确表达的偏好、习惯或要求。\n"
        "3. facts: 字符串列表。记录值得长期记住的核心事实。\n"
        "注意：如果没有相关信息，对应的列表应为空，字段不能缺失。"
    )

    def __init__(self, client: OpenAI, model: str, compact_k: int = 10):
        self.client = client
        self.model = model
        self.k = compact_k

    def compact(self, history: list[dict]) -> dict[str, Any]:
        """分析历史，返回结构化提取结果。

        不修改传入的 history，不执行任何 I/O。
        """
        # 剔除系统消息，只保留最近的 k 条
        effective = [m for m in history if m.get("role") != "system"]
        if len(effective) < 2:
            return {"summary": {}, "preferences": [], "facts": []}

        recent = effective[-self.k:]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._PROMPT},
                    {"role": "user", "content": self._format(recent)},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
        except Exception as e:
            print(f"[Compactor] 提取失败: {e}", flush=True)
            return {
                "summary": {"critical": f"提取失败: {e}", "decision": "无", "issue": "无"},
                "preferences": [],
                "facts": [],
            }

        return self._normalize(data)

    @staticmethod
    def _format(history: list[dict]) -> str:
        """将历史消息列表格式化为 LLM 可读文本。"""
        lines: list[str] = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            # 助手消息中的工具调用合并显示
            if role == "assistant" and msg.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?")
                         for tc in msg["tool_calls"]]
                content = f"[调用工具: {', '.join(names)}] {content}".strip()

            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _normalize(data: dict) -> dict:
        """规范化 LLM 返回结果，确保字段存在且值合理。"""
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        for key in ("critical", "decision", "issue"):
            val = summary.get(key, "无")
            summary[key] = val[:60] if isinstance(val, str) else "无"

        preferences = data.get("preferences", [])
        if not isinstance(preferences, list):
            preferences = []

        facts = data.get("facts", [])
        if not isinstance(facts, list):
            facts = []

        return {"summary": summary, "preferences": preferences, "facts": facts}
