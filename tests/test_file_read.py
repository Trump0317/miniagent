"""FileReadTool 单元测试."""
import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.file_read import FileReadTool


class TestFileReadTool(unittest.TestCase):
    """FileReadTool 基本功能测试."""

    def setUp(self):
        self.tool = FileReadTool()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write(self, name: str, content: str) -> str:
        p = os.path.join(self.tmpdir.name, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    # ── 正常路径 ──

    def test_read_existing_file(self):
        path = self._write("test.txt", "hello world")
        result = self.tool.execute(file_path=path)
        self.assertEqual(result, "hello world")

    def test_read_multiline_file(self):
        path = self._write("multi.txt", "line1\nline2\nline3")
        result = self.tool.execute(file_path=path)
        self.assertEqual(result, "line1\nline2\nline3")

    def test_read_empty_file(self):
        path = self._write("empty.txt", "")
        result = self.tool.execute(file_path=path)
        self.assertEqual(result, "")

    def test_read_utf8_file(self):
        path = self._write("utf8.txt", "中文测试")
        result = self.tool.execute(file_path=path)
        self.assertEqual(result, "中文测试")

    # ── 错误路径 ──

    def test_file_not_found(self):
        result = self.tool.execute(file_path="/nonexistent/path/file.txt")
        self.assertIn("Error", result)
        self.assertIn("FileReadTool", result)

    def test_read_directory(self):
        result = self.tool.execute(file_path=self.tmpdir.name)
        self.assertIn("Error", result)
        self.assertIn("FileReadTool", result)

    # ── 默认参数 ──

    def test_default_encoding_is_utf8(self):
        """验证 encoding 参数默认值为 utf-8."""
        self.assertEqual(self.tool.args_model.model_fields["encoding"].default, "utf-8")


if __name__ == "__main__":
    unittest.main()
