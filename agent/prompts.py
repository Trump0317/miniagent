"""Prompt 模板加载器 —— 从 Markdown 文件加载可复用的提示词模板。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class PromptTemplate:
    """一个提示词模板，解析自 prompts/*.md 文件。

    文件格式:
        ---
        description: 审查代码
        ---
        请审查文件 {query}，重点关注安全问题和代码质量。

    用户通过 /name 调用，{query} 替换为用户输入。
    """

    name: str           # 模板名（文件名不含扩展名）
    description: str    # 简短描述
    content: str        # 模板正文，{query} 替换为用户输入


class PromptLoader:
    """扫描目录，加载所有 *.md 文件为 PromptTemplate。"""

    def __init__(self, directory: Path):
        self._directory = Path(directory)
        self._templates: dict[str, PromptTemplate] = {}
        self._load()

    @property
    def templates(self) -> dict[str, PromptTemplate]:
        return self._templates

    def get(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def list_commands(self) -> str:
        """生成可供用户参考的命令列表"""
        if not self._templates:
            return ""
        lines = ["可用命令:"]
        for name, tmpl in self._templates.items():
            lines.append(f"  /{name} — {tmpl.description}")
        return "\n".join(lines)

    def resolve(self, name: str, query: str) -> str | None:
        """展开模板，返回替换后的提示词。未找到模板返回 None。"""
        tmpl = self._templates.get(name)
        if not tmpl:
            return None
        return tmpl.content.replace("{query}", query)

    def _load(self) -> None:
        if not self._directory.exists() or not self._directory.is_dir():
            return

        for md_file in sorted(self._directory.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            tmpl = self._parse(content, md_file.stem)
            if tmpl:
                self._templates[tmpl.name] = tmpl

    def _parse(self, content: str, filename: str) -> PromptTemplate | None:
        # 解析 YAML frontmatter
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter = match.group(1)
        body = match.group(2).strip()

        description = self._extract(frontmatter, "description") or filename

        return PromptTemplate(
            name=filename,
            description=description,
            content=body,
        )

    @staticmethod
    def _extract(frontmatter: str, key: str) -> str | None:
        m = re.search(rf"^{key}:\s*(.+)$", frontmatter, re.MULTILINE)
        return m.group(1).strip() if m else None
