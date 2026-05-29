"""PromptLoader 单元测试."""
import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.core.prompts import PromptLoader, PromptTemplate


class TestPromptTemplate(unittest.TestCase):
    """PromptTemplate 数据类测试."""

    def test_fields(self):
        """基本字段赋值."""
        t = PromptTemplate(name="review", description="审查代码", content="审查 {query}")
        self.assertEqual(t.name, "review")
        self.assertEqual(t.description, "审查代码")
        self.assertEqual(t.content, "审查 {query}")


class TestPromptLoaderInit(unittest.TestCase):
    """加载测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prompts_dir = Path(self.tmp.name) / "prompts"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_directory(self):
        """空目录返回空模板."""
        self.prompts_dir.mkdir()
        loader = PromptLoader(self.prompts_dir)
        self.assertEqual(loader.templates, {})

    def test_nonexistent_directory(self):
        """目录不存在时不报错."""
        loader = PromptLoader(self.prompts_dir)  # 未创建
        self.assertEqual(loader.templates, {})

    def test_loads_md_files(self):
        """加载 *.md 文件为模板."""
        self.prompts_dir.mkdir()
        (self.prompts_dir / "review.md").write_text(
            "---\ndescription: 审查代码\n---\n请审查 {query}，关注安全问题。\n",
            encoding="utf-8"
        )
        loader = PromptLoader(self.prompts_dir)
        self.assertIn("review", loader.templates)
        tmpl = loader.templates["review"]
        self.assertEqual(tmpl.description, "审查代码")
        self.assertIn("{query}", tmpl.content)

    def test_ignores_non_md_files(self):
        """忽略非 .md 文件."""
        self.prompts_dir.mkdir()
        (self.prompts_dir / "notes.txt").write_text("not a template", encoding="utf-8")
        (self.prompts_dir / "empty.py").write_text("", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        self.assertEqual(loader.templates, {})

    def test_skips_malformed_md(self):
        """跳过无 YAML frontmatter 的 .md 文件."""
        self.prompts_dir.mkdir()
        (self.prompts_dir / "bad.md").write_text("没有 frontmatter\n", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        self.assertNotIn("bad", loader.templates)

    def test_missing_description_uses_filename(self):
        """缺少 description 时使用文件名."""
        self.prompts_dir.mkdir()
        (self.prompts_dir / "scout.md").write_text(
            "---\ntool: search\n---\n搜索 {query}\n",
            encoding="utf-8"
        )
        loader = PromptLoader(self.prompts_dir)
        self.assertEqual(loader.templates["scout"].description, "scout")

    def test_multiple_files(self):
        """加载多个模板文件."""
        self.prompts_dir.mkdir()
        (self.prompts_dir / "review.md").write_text(
            "---\ndescription: 审查\n---\n审查 {query}\n", encoding="utf-8")
        (self.prompts_dir / "scout.md").write_text(
            "---\ndescription: 侦查\n---\n侦查 {query}\n", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        self.assertEqual(len(loader.templates), 2)

    def test_read_error_graceful(self):
        """文件读取异常时跳过而非崩溃."""
        self.prompts_dir.mkdir()
        bad_file = self.prompts_dir / "broken.md"
        bad_file.write_text("---\ndescription: test\n---\ncontent\n", encoding="utf-8")
        bad_file.chmod(0o000)
        try:
            loader = PromptLoader(self.prompts_dir)
            self.assertNotIn("broken", loader.templates)
        finally:
            bad_file.chmod(0o644)  # 恢复权限，否则 tearDown 删不掉


class TestPromptLoaderGet(unittest.TestCase):
    """get 方法测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prompts_dir = Path(self.tmp.name) / "prompts"
        self.prompts_dir.mkdir()
        (self.prompts_dir / "review.md").write_text(
            "---\ndescription: 审查\n---\n审查 {query}\n", encoding="utf-8")
        self.loader = PromptLoader(self.prompts_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_get_existing(self):
        """获取已存在的模板."""
        tmpl = self.loader.get("review")
        self.assertIsNotNone(tmpl)
        self.assertEqual(tmpl.name, "review")

    def test_get_nonexistent(self):
        """获取不存在的模板返回 None."""
        self.assertIsNone(self.loader.get("nonexistent"))


class TestPromptLoaderListCommands(unittest.TestCase):
    """list_commands 测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prompts_dir = Path(self.tmp.name) / "prompts"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_templates(self):
        """空模板返回空字符串."""
        self.prompts_dir.mkdir()
        loader = PromptLoader(self.prompts_dir)
        self.assertEqual(loader.list_commands(), "")

    def test_with_templates(self):
        """有模板时生成命令列表."""
        self.prompts_dir.mkdir()
        (self.prompts_dir / "review.md").write_text(
            "---\ndescription: 审查代码\n---\n审查 {query}\n", encoding="utf-8")
        (self.prompts_dir / "scout.md").write_text(
            "---\ndescription: 侦查代码\n---\n侦查 {query}\n", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        output = loader.list_commands()
        self.assertIn("可用命令:", output)
        self.assertIn("/review — 审查代码", output)
        self.assertIn("/scout — 侦查代码", output)


class TestPromptLoaderResolve(unittest.TestCase):
    """resolve 方法测试."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prompts_dir = Path(self.tmp.name) / "prompts"
        self.prompts_dir.mkdir()
        (self.prompts_dir / "review.md").write_text(
            "---\ndescription: 审查\n---\n请审查文件 {query}，重点关注安全问题和代码质量。\n",
            encoding="utf-8")
        self.loader = PromptLoader(self.prompts_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_replaces_query(self):
        """resolve 替换 {query}."""
        result = self.loader.resolve("review", "agent.py")
        self.assertIn("agent.py", result)
        self.assertNotIn("{query}", result)

    def test_resolve_nonexistent(self):
        """resolve 不存在的模板返回 None."""
        self.assertIsNone(self.loader.resolve("nonexistent", "test"))

    def test_resolve_multiple_query(self):
        """多个 {query} 全部替换."""
        (self.prompts_dir / "multi.md").write_text(
            "---\ndescription: 多次\n---\n处理 {query} 和 {query}\n", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        result = loader.resolve("multi", "test.py")
        self.assertEqual(result, "处理 test.py 和 test.py")

    def test_resolve_no_query_placeholder(self):
        """无 {query} 时返回原文."""
        (self.prompts_dir / "static.md").write_text(
            "---\ndescription: 静态\n---\n固定内容，无占位符\n", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        result = loader.resolve("static", "ignored")
        self.assertEqual(result, "固定内容，无占位符")

    def test_resolve_empty_body(self):
        """空 body（只有 frontmatter）返回空字符串."""
        (self.prompts_dir / "empty.md").write_text(
            "---\ndescription: 空\n---\n", encoding="utf-8")
        loader = PromptLoader(self.prompts_dir)
        result = loader.resolve("empty", "test")
        self.assertEqual(result, "")


class TestPromptLoaderExtract(unittest.TestCase):
    """_extract 静态方法测试."""

    def test_extract_existing_key(self):
        """提取存在的 key."""
        result = PromptLoader._extract("description: 测试\nother: value", "description")
        self.assertEqual(result, "测试")

    def test_extract_missing_key(self):
        """提取不存在的 key 返回 None."""
        result = PromptLoader._extract("description: 测试", "nonexistent")
        self.assertIsNone(result)

    def test_extract_multiline_frontmatter(self):
        """多行 frontmatter 中提取."""
        fm = "description: 审查代码\ntool: search\nmodel: deepseek"
        self.assertEqual(PromptLoader._extract(fm, "description"), "审查代码")
        self.assertEqual(PromptLoader._extract(fm, "tool"), "search")
        self.assertEqual(PromptLoader._extract(fm, "model"), "deepseek")


if __name__ == "__main__":
    unittest.main()
