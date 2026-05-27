from __future__ import annotations
from pydantic import BaseModel, Field
from agent.tools.ToolRegisty.base import Tool, tool
from typing import Type, Optional
import os
import re
from pathlib import Path


class SkillArgs(BaseModel):
    name: str = Field(description="要调用的skill名称。")

@tool(
    name="skill_tool",
    description="加载指定技能的详细知识内容，在回答相关问题前调用",
    parameters=SkillArgs,
)
class SkillTool(Tool):

    def __init__(self, skills_loader: SkillsLoader) -> None:
        self.skills_loader = skills_loader

    def get_description(self) -> str:
        """返回技能列表描述，供 LLM 参考"""
        return self.skills_loader.get_description()

    def execute(self, name: str) -> str:
        """加载指定技能：返回技能的 SKILL.md 内容"""
        content = self.skills_loader.get_content(name)
        if content.startswith("错误"):
            return f"[SkillTool]: {content}"
        
        return f"[SkillTool]: 已加载技能 '{name}' 的详细说明：\n\n{content}"


class SkillsLoader:
    def __init__(self, skill_directory: Path):
        self._directory = Path(skill_directory)
        self.skills = self._load()

    def _read_file(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"[SkillTool]: 读取文件 {file_path.name} 时出错: {str(e)}"

    def _load(self) -> dict:
        """扫描目录并加载所有 SKILL.md 技能定义"""
        skills = {}
        if not self._directory.exists() or not self._directory.is_dir():
            return skills

        # 遍历所有子目录下的 SKILL.md
        for skill_file in self._directory.glob("*/SKILL.md"):
            content = self._read_file(skill_file)
            name, desc = self._extract_metadata(content)
            
            # 优先级：YAML 中的 name > 目录名
            skill_id = name or skill_file.parent.name
            skills[skill_id] = {
                "content": content,
                "description": desc or f"{skill_id} 技能说明",
                "directory": str(skill_file.parent)
            }
        return skills

    def _extract_metadata(self, content: str) -> tuple[Optional[str], Optional[str]]:
        """解析 SKILL.md 顶部的 YAML 元数据块"""
        # 匹配 --- ... --- 块
        match = re.search(r'^---\s*\n(.*?)\n---\s*', content, re.DOTALL | re.MULTILINE)
        if not match:
            return None, None

        yaml_text = match.group(1)
        # 使用更灵活的正则匹配 Key: Value
        name = re.search(r'^name:\s*(.+)$', yaml_text, re.MULTILINE)
        desc = re.search(r'^description:\s*(.+)$', yaml_text, re.MULTILINE)
        
        return (
            name.group(1).strip() if name else None,
            desc.group(1).strip() if desc else None
        )

    def get_description(self) -> str:
        """生成供 LLM 参考的技能列表描述"""
        if not self.skills:
            return "（当前无可用技能）"
        
        lines = ["可用技能列表:"]
        for name, info in self.skills.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    def get_content(self, skill_name: str) -> str:
        """获取特定技能的完整定义内容"""
        skill_data = self.skills.get(skill_name)
        return skill_data["content"] if skill_data else f"错误：技能 '{skill_name}' 未找到。"
