"""Agent 定义加载器 —— 从 Markdown + YAML 文件中加载子代理配置。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass
class AgentDefinition:
    """一个子代理的完整定义，解析自 agents/*.md 文件。

    文件格式:
        ---
        name: code-reviewer
        description: 审查代码质量和风格
        tools: read, grep, bash
        model: deepseek-v4-pro
        max_turns: 10
        ---
        You are a code reviewer. Focus on...
    """

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = ""
    max_turns: int = 10
    source: str = ""  # 文件路径，方便调试


class AgentLoader:
    """扫描目录，解析所有 *.md 文件为 AgentDefinition。"""

    def __init__(self, directory: Path):
        self._directory = Path(directory)
        self._agents: dict[str, AgentDefinition] = {}
        self._load()

    @property
    def agents(self) -> dict[str, AgentDefinition]:
        return self._agents

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def list_agents(self) -> str:
        """生成供 LLM 参考的可用子代理列表"""
        if not self._agents:
            return "（无可用子代理定义）"
        lines = ["可用的子代理:"]
        for name, agent in self._agents.items():
            tools_str = ", ".join(agent.tools) if agent.tools else "全部"
            lines.append(f"  - {name}: {agent.description} (工具: {tools_str})")
        return "\n".join(lines)

    def _load(self) -> None:
        if not self._directory.exists() or not self._directory.is_dir():
            return

        for md_file in sorted(self._directory.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            definition = self._parse(content, str(md_file))
            if definition:
                self._agents[definition.name] = definition

    def _parse(self, content: str, source: str) -> AgentDefinition | None:
        # 解析 YAML frontmatter --- ... ---
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter = match.group(1)
        body = match.group(2).strip()

        # 解析 name（必填）
        name = self._extract(frontmatter, "name")
        if not name:
            return None

        description = self._extract(frontmatter, "description") or ""
        tools_raw = self._extract(frontmatter, "tools") or ""
        tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        model = self._extract(frontmatter, "model") or ""
        max_turns_str = self._extract(frontmatter, "max_turns") or "10"

        try:
            max_turns = int(max_turns_str)
        except ValueError:
            max_turns = 10

        return AgentDefinition(
            name=name,
            description=description,
            system_prompt=body,
            tools=tools,
            model=model,
            max_turns=max_turns,
            source=source,
        )

    @staticmethod
    def _extract(frontmatter: str, key: str) -> str | None:
        m = re.search(rf"^{key}:\s*(.+)$", frontmatter, re.MULTILINE)
        return m.group(1).strip() if m else None
