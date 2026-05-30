"""SkillsLoader / SkillTool 单元测试."""
import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.skill import SkillsLoader, SkillTool


class TestSkillsLoader(unittest.TestCase):
    """SkillsLoader 技能目录扫描与解析."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _mkdir(self, name: str) -> str:
        p = os.path.join(self.tmpdir.name, name)
        os.makedirs(p, exist_ok=True)
        return p

    def _write_skill(self, dir_name: str, content: str) -> str:
        d = self._mkdir(dir_name)
        skill_path = os.path.join(d, "SKILL.md")
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(content)
        return skill_path

    # ── 空目录 ──

    def test_empty_directory(self):
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.skills, {})
        self.assertIn("无可用技能", loader.get_description())

    def test_nonexistent_directory(self):
        loader = SkillsLoader(Path("/nonexistent/skills"))
        self.assertEqual(loader.skills, {})

    # ── 加载单个技能 ──

    def test_load_single_skill(self):
        content = """---
name: test-skill
description: A test skill for unit testing
---
# Test Skill

This is the skill body.
"""
        self._write_skill("test", content)
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertIn("test-skill", loader.skills)
        self.assertEqual(loader.skills["test-skill"]["description"], "A test skill for unit testing")

    def test_load_skill_body(self):
        content = """---
name: skill1
description: desc
---
This is the body content.
"""
        self._write_skill("s1", content)
        loader = SkillsLoader(Path(self.tmpdir.name))
        skill_data = loader.skills["skill1"]
        self.assertIn("This is the body content.", skill_data["content"])

    # ── 无 YAML frontmatter ──

    def test_no_frontmatter_uses_dirname(self):
        content = """# No YAML header
Just some markdown content.
"""
        self._write_skill("dirname-skill", content)
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertIn("dirname-skill", loader.skills)
        self.assertIn("dirname-skill 技能说明", loader.skills["dirname-skill"]["description"])

    def test_no_name_field_uses_dirname(self):
        content = """---
description: Has desc but no name
---
Body
"""
        self._write_skill("fallback-name", content)
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertIn("fallback-name", loader.skills)

    def test_no_description_is_default(self):
        content = """---
name: nodef
---
Body
"""
        self._write_skill("nodef", content)
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertIn("nodef 技能说明", loader.skills["nodef"]["description"])

    # ── 多技能 ──

    def test_multiple_skills(self):
        self._write_skill("a", "---\nname: skill-a\ndescription: first\n---\nbody a\n")
        self._write_skill("b", "---\nname: skill-b\ndescription: second\n---\nbody b\n")
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertEqual(len(loader.skills), 2)
        self.assertIn("skill-a", loader.skills)
        self.assertIn("skill-b", loader.skills)

    # ── get_description ──

    def test_get_description_lists_all_skills(self):
        self._write_skill("a", "---\nname: skill-a\ndescription: first skill\n---\nbody\n")
        self._write_skill("b", "---\nname: skill-b\ndescription: second skill\n---\nbody\n")
        loader = SkillsLoader(Path(self.tmpdir.name))
        desc = loader.get_description()
        self.assertIn("skill-a", desc)
        self.assertIn("skill-b", desc)
        self.assertIn("first skill", desc)
        self.assertIn("second skill", desc)

    # ── get_content ──

    def test_get_content_returns_skill_body(self):
        content = "---\nname: my-skill\ndescription: mine\n---\n# Title\n\nSkill instructions here.\n"
        self._write_skill("my", content)
        loader = SkillsLoader(Path(self.tmpdir.name))
        result = loader.get_content("my-skill")
        self.assertIn("Skill instructions here", result)

    def test_get_content_not_found(self):
        loader = SkillsLoader(Path(self.tmpdir.name))
        result = loader.get_content("nonexistent")
        self.assertIn("错误", result)
        self.assertIn("未找到", result)

    # ── 忽略非 SKILL.md 文件 ──

    def test_ignores_non_skill_files(self):
        d = self._mkdir("extra")
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("not a skill")
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertEqual(loader.skills, {})

    # ── 子目录下的 SKILL.md ──

    def test_only_one_level_deep_discovered(self):
        """只在 */SKILL.md 一层子目录匹配，不递归更深."""
        self._write_skill("one-deep", "---\nname: shallow-skill\ndescription: one level\n---\nbody\n")
        # 两层深的子目录应不被加载
        deep_dir = os.path.join(self.tmpdir.name, "two", "deep")
        os.makedirs(deep_dir, exist_ok=True)
        with open(os.path.join(deep_dir, "SKILL.md"), "w") as f:
            f.write("---\nname: deep-skill\ndescription: hidden\n---\nbody\n")
        loader = SkillsLoader(Path(self.tmpdir.name))
        self.assertIn("shallow-skill", loader.skills)
        self.assertNotIn("deep-skill", loader.skills)


class TestSkillTool(unittest.TestCase):
    """SkillTool execute 测试."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self._write_skill(
            "test-skill",
            "---\nname: test-skill\ndescription: test\n---\n# Skill Body\nInstructions here.\n",
        )
        self.skills = SkillsLoader(Path(self.tmpdir.name))
        self.tool = SkillTool(self.skills)

    def _write_skill(self, dir_name: str, content: str):
        d = os.path.join(self.tmpdir.name, dir_name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(content)

    def test_execute_returns_skill_content(self):
        result = self.tool.execute(name="test-skill")
        self.assertIn("已加载技能", result)
        self.assertIn("Instructions here", result)

    def test_execute_not_found(self):
        result = self.tool.execute(name="unknown")
        self.assertIn("SkillTool", result)
        self.assertIn("错误", result)

    def test_get_description_delegates(self):
        desc = self.tool.get_description()
        self.assertIn("test-skill", desc)


if __name__ == "__main__":
    unittest.main()
