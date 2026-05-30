"""WebFetchTool 单元测试."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.web_fetch import WebFetchTool


class TestWebFetchTool(unittest.TestCase):
    """WebFetchTool 测试."""

    def setUp(self):
        self.tool = WebFetchTool()

    @patch("urllib.request.urlopen")
    def test_fetch_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html><body>Hello</body></html>"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(url="https://example.com")
        self.assertIn("Hello", result)

    @patch("urllib.request.urlopen")
    def test_fetch_sets_user_agent(self, mock_urlopen):
        """验证请求设置了 Mozilla User-Agent."""
        import urllib.request
        mock_response = MagicMock()
        mock_response.read.return_value = b"ok"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        self.tool.execute(url="https://example.com")
        call_args = mock_urlopen.call_args[0][0]
        self.assertIsInstance(call_args, urllib.request.Request)
        self.assertEqual(call_args.get_header('User-agent'), 'Mozilla/5.0')

    @patch("urllib.request.urlopen")
    def test_fetch_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, None,
        )

        result = self.tool.execute(url="https://example.com/404")
        self.assertIn("HTTP Error 404", result)

    @patch("urllib.request.urlopen")
    def test_fetch_generic_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")

        result = self.tool.execute(url="https://unreachable.com")
        self.assertIn("Error", result)
        self.assertIn("connection refused", result)

    @patch("urllib.request.urlopen")
    def test_fetch_decodes_utf8(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = "中文内容".encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(url="https://example.com")
        self.assertIn("中文内容", result)

    def test_default_timeout(self):
        """验证 timeout 的默认值."""
        # 不实际调用，只确保默认参数设置正确
        self.assertEqual(self.tool.args_model.model_fields["timeout"].default, 30)


if __name__ == "__main__":
    unittest.main()
