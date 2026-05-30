"""FileEditTool 单元测试."""
import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.file_edit import FileEditTool


class TestFileEditTool(unittest.TestCase):
    """FileEditTool 文本替换功能测试."""

    def setUp(self):
        self.tool = FileEditTool()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name: str) -> str:
        return os.path.join(self.tmpdir.name, name)

    def _write(self, name: str, content: str) -> str:
        p = self._path(name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    def _read(self, name: str) -> str:
        with open(self._path(name), "r", encoding="utf-8") as f:
            return f.read()

    # ── 精确匹配（策略 1）──

    def test_exact_match_single(self):
        path = self._write("f.py", "old line\nother line\n")
        result = self.tool.execute(file_path=path, old_str="old line", new_str="new line")
        self.assertIn("精确匹配", result)
        self.assertEqual(self._read("f.py"), "new line\nother line\n")

    def test_exact_match_multiline(self):
        path = self._write("f.py", "line1\nline2\nline3\n")
        result = self.tool.execute(file_path=path, old_str="line1\nline2", new_str="line_a\nline_b")
        self.assertIn("精确匹配", result)
        self.assertEqual(self._read("f.py"), "line_a\nline_b\nline3\n")

    def test_exact_match_multiple_occurrences(self):
        path = self._write("f.py", "dup\ndup\ndup\n")
        result = self.tool.execute(file_path=path, old_str="dup", new_str="unique")
        self.assertIn("不唯一", result)
        self.assertEqual(self._read("f.py"), "dup\ndup\ndup\n")

    def test_exact_match_not_found(self):
        path = self._write("f.py", "content\n")
        result = self.tool.execute(file_path=path, old_str="missing", new_str="replaced")
        self.assertIn("未找到 old_str", result)

    # ── 忽略缩进匹配（策略 2）──

    def test_indent_relaxed_match(self):
        """old_str 不是文件内容的子串但去掉缩进后匹配."""
        path = self._write("f.py", "    indented line\n    other\n")
        # "indented line\nother" 不是文件内容的子串(文件是 "    indented line\n    other")
        # 但去掉缩进后可以匹配
        result = self.tool.execute(file_path=path, old_str="indented line\nother", new_str="replaced line\nnew")
        self.assertIn("忽略缩进匹配", result)
        # new_str 不带缩进，直接替换掉带缩进的 matched_block
        self.assertEqual(self._read("f.py"), "replaced line\nnew\n")

    def test_indent_relaxed_multiline(self):
        """缩进匹配后 new_str 不带缩进，直接替换带缩进的 matched_block."""
        path = self._write("f.py", "    def foo():\n        pass\n    def bar():\n")
        result = self.tool.execute(
            file_path=path,
            old_str="def foo():\n    pass",
            new_str="def baz():\n    return 1",
        )
        # 原始匹配块是 "    def foo():\n        pass"，替换为 new_str（无缩进）
        self.assertIn("忽略缩进匹配", result)
        self.assertEqual(self._read("f.py"), "def baz():\n    return 1\n    def bar():\n")

    def test_indent_relaxed_multiple_matches(self):
        """精确匹配先命中（多匹配），不会进入缩进策略."""
        path = self._write("f.py",
            "def foo():\n    pass\n"
            "def bar():\n    pass\n"
        )
        result = self.tool.execute(file_path=path, old_str="pass", new_str="return")
        # "pass" 作为精确子串匹配到 2 处
        self.assertIn("匹配到 2 处", result)
        self.assertIn("不唯一", result)

    # ── 错误路径 ──

    def test_file_not_found(self):
        result = self.tool.execute(
            file_path="/nonexistent/file.py",
            old_str="a", new_str="b",
        )
        self.assertIn("文件不存在", result)

    def test_permission_error(self):
        """文件只读时写入失败."""
        path = self._path("readonly.txt")
        with open(path, "w") as f:
            f.write("test\n")
        os.chmod(path, 0o444)
        try:
            result = self.tool.execute(file_path=path, old_str="test", new_str="new")
            self.assertIn("无权限写入", result)
        finally:
            os.chmod(path, 0o644)

    # ── strip_leading_whitespace ──

    def test_strip_leading_whitespace(self):
        result = self.tool._strip_leading_whitespace("  a\n    b\nc")
        self.assertEqual(result, ["a", "b", "c"])

    def test_strip_leading_whitespace_empty_lines(self):
        result = self.tool._strip_leading_whitespace("  a\n\n  b")
        self.assertEqual(result, ["a", "", "b"])

    # ── _snippet ──

    def test_snippet_short_file(self):
        snippet = self.tool._snippet("line1\nline2\nline3\n")
        self.assertIn("line1", snippet)
        self.assertIn("line2", snippet)

    def test_snippet_long_file(self):
        lines = [f"line {i}" for i in range(1, 30)]
        content = "\n".join(lines)
        snippet = self.tool._snippet(content)
        self.assertIn("前 20 行", snippet)
        self.assertIn("共 29 行", snippet)


if __name__ == "__main__":
    unittest.main()
