"""FileWriteTool 单元测试."""
import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.file_write import FileWriteTool


class TestFileWriteTool(unittest.TestCase):
    """FileWriteTool 基本功能测试."""

    def setUp(self):
        self.tool = FileWriteTool()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name: str) -> str:
        return os.path.join(self.tmpdir.name, name)

    def _read(self, name: str) -> str:
        with open(self._path(name), "r", encoding="utf-8") as f:
            return f.read()

    # ── 正常写入 ──

    def test_write_new_file(self):
        path = self._path("new.txt")
        result = self.tool.execute(file_path=path, content="hello")
        self.assertIn("successfully written", result)
        self.assertEqual(self._read("new.txt"), "hello")

    def test_write_overwrites_existing(self):
        path = self._path("existing.txt")
        with open(path, "w") as f:
            f.write("old content")
        result = self.tool.execute(file_path=path, content="new content")
        self.assertIn("successfully written", result)
        self.assertEqual(self._read("existing.txt"), "new content")

    def test_write_multiline_content(self):
        path = self._path("multi.txt")
        content = "line1\nline2\nline3"
        result = self.tool.execute(file_path=path, content=content)
        self.assertIn("successfully written", result)
        self.assertEqual(self._read("multi.txt"), content)

    def test_write_empty_content(self):
        path = self._path("empty.txt")
        result = self.tool.execute(file_path=path, content="")
        self.assertIn("successfully written", result)
        self.assertEqual(self._read("empty.txt"), "")

    def test_write_utf8_content(self):
        path = self._path("utf8.txt")
        result = self.tool.execute(file_path=path, content="中文内容 🎉")
        self.assertIn("successfully written", result)
        self.assertEqual(self._read("utf8.txt"), "中文内容 🎉")

    def test_write_to_nonexistent_dir(self):
        """写入不存在的目录时返回错误."""
        path = os.path.join(self.tmpdir.name, "sub", "deep", "file.txt")
        result = self.tool.execute(file_path=path, content="deep")
        self.assertIn("Error", result)
        self.assertIn("FileWriteTool", result)

    # ── 错误路径 ──

    def test_write_to_readonly_dir(self):
        """写入无权限目录."""
        ro_dir = os.path.join(self.tmpdir.name, "readonly")
        os.makedirs(ro_dir)
        os.chmod(ro_dir, 0o555)
        path = os.path.join(ro_dir, "file.txt")
        try:
            result = self.tool.execute(file_path=path, content="x")
            self.assertIn("Error", result)
            self.assertIn("FileWriteTool", result)
        finally:
            os.chmod(ro_dir, 0o755)


if __name__ == "__main__":
    unittest.main()
