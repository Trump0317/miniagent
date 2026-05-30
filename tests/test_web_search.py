"""WebSearchTool 单元测试."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.web_search import WebSearchTool


# ── DuckDuckGo HTML 模拟 ──

SINGLE_RESULT_HTML = """<html><body>
<div class="result__body">
    <a class="result__a" href="https://example.com">Example Title</a>
    <a class="result__snippet" href="https://example.com">A brief description of the result.</a>
<div class="clear"></div>
</div>
</body></html>"""

MULTI_RESULT_HTML = """<html><body>
<div class="result__body">
    <a class="result__a">Result One</a>
    <a class="result__snippet">First snippet.</a>
<div class="clear"></div>
</div>
<div class="result__body">
    <a class="result__a">Result Two</a>
    <a class="result__snippet">Second snippet.</a>
<div class="clear"></div>
</div>
<div class="result__body">
    <a class="result__a">Result Three</a>
    <a class="result__snippet">Third snippet.</a>
<div class="clear"></div>
</div>
</body></html>"""

CAPTCHA_HTML = """<html><body>
<div class="ddg-captcha">Please verify you are human</div>
</body></html>"""

NO_RESULTS_HTML = """<html><body>
<div>No results found.</div>
</body></html>"""


class TestWebSearchTool(unittest.TestCase):
    """WebSearchTool 测试."""

    def setUp(self):
        self.tool = WebSearchTool()

    @patch("urllib.request.urlopen")
    def test_search_single_result(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = SINGLE_RESULT_HTML.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(query="test")
        self.assertIn("Example Title", result)
        self.assertIn("brief description", result)

    @patch("urllib.request.urlopen")
    def test_search_multiple_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = MULTI_RESULT_HTML.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(query="test", max_results=5)
        self.assertIn("Result One", result)
        self.assertIn("Result Two", result)
        self.assertIn("Result Three", result)

    @patch("urllib.request.urlopen")
    def test_search_limits_max_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = MULTI_RESULT_HTML.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(query="test", max_results=2)
        self.assertIn("Result One", result)
        self.assertIn("Result Two", result)
        self.assertNotIn("Result Three", result)

    @patch("urllib.request.urlopen")
    def test_search_captcha_detected(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = CAPTCHA_HTML.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(query="test")
        self.assertIn("Access denied", result)

    @patch("urllib.request.urlopen")
    def test_search_no_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = NO_RESULTS_HTML.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        result = self.tool.execute(query="xyzabc123nonexistent")
        self.assertIn("No results found", result)

    @patch("urllib.request.urlopen")
    def test_search_exception(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network down")

        result = self.tool.execute(query="test")
        self.assertIn("Search error", result)

    def test_default_max_results(self):
        self.assertEqual(self.tool.args_model.model_fields["max_results"].default, 5)


if __name__ == "__main__":
    unittest.main()
