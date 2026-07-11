import functools
import http.server
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../src"))
)

import requests

from pneuma_seeker.services.core.ir_system.retriever.impl.web_crawler import WebCrawler
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Text


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        # Simple config object with a max char limit
        self.config = Config()
        self.config.WEB_CRAWL_MAX_CHARS = 1000

    def _mock_response(self, status_code=200, text=""):
        mock = Mock()
        mock.status_code = status_code
        mock.text = text

        def raise_for_status():
            if not (200 <= status_code < 300):
                raise requests.HTTPError(f"status {status_code}")

        mock.raise_for_status = raise_for_status
        return mock

    @patch("requests.Session.get")
    def test_retrieve_success(self, mock_get):
        """When robots.txt is inaccessible (404) and page returns HTML, retrieve returns Text doc."""

        robots_resp = self._mock_response(status_code=404, text="")
        page_html = (
            "<html><body><h1>Example Domain</h1><p>This is example</p></body></html>"
        )
        page_resp = self._mock_response(status_code=200, text=page_html)

        # session.get will be called twice: once for robots.txt, once for the page
        mock_get.side_effect = [robots_resp, page_resp]

        crawler = WebCrawler("uX", "cX", self.config, MagicMock(), MagicMock())
        results = crawler.retrieve("http://www.example.com", 1, False)

        # Should return a list with one Text document
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        doc = results[0]
        self.assertIsInstance(doc, Text)
        self.assertEqual(doc.doc_id, "web_crawl")
        self.assertEqual(doc.retriever_type, RetrieverType.WEB_CRAWL)
        self.assertIn("Example Domain", doc.content)

    @patch("requests.Session.get")
    def test_retrieve_blocked_by_robots(self, mock_get):
        """When robots.txt disallows the path, retrieve returns a disallow message."""

        # robots.txt disallows everything
        robots_text = "User-agent: *\nDisallow: /"
        robots_resp = self._mock_response(status_code=200, text=robots_text)

        mock_get.return_value = robots_resp

        crawler = WebCrawler("uX", "cX", self.config, MagicMock(), MagicMock())
        out = crawler.retrieve("http://www.example.com/anypath", 1, False)

        # When disallowed, retrieve returns a string error message
        self.assertIsInstance(out[0], Text)
        self.assertIn("disallowed by robots.txt", out[0].content)


class WebCrawlerLiveServerTests(unittest.TestCase):
    """
    Serves real files (HTML pages + robots.txt) from a local HTTP server so
    the crawler's HTTP fetch, robots.txt parsing, and HTML-to-text extraction
    run against genuine responses instead of a mocked Session.get.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp()
        root = Path(cls.temp_dir)

        (root / "robots.txt").write_text("User-agent: *\nDisallow: /private/\n")

        (root / "index.html").write_text(
            "<html><head><title>Home</title>"
            "<style>body { color: red; }</style>"
            "<script>console.log('should be stripped');</script>"
            "</head><body>"
            "<h1>Welcome</h1>"
            "<p>This   is   a   test   page  with   irregular   spacing.</p>"
            "<noscript>Enable JS</noscript>"
            "</body></html>"
        )

        private_dir = root / "private"
        private_dir.mkdir()
        (private_dir / "secret.html").write_text(
            "<html><body><p>Top secret content.</p></body></html>"
        )

        public_dir = root / "public"
        public_dir.mkdir()
        (public_dir / "page.html").write_text(
            "<html><body><p>Public page content.</p></body></html>"
        )

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=cls.temp_dir
        )
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self) -> None:
        self.config = Config()
        self.config.WEB_CRAWL_MAX_CHARS = 1000
        self.crawler = WebCrawler("uX", "cX", self.config, MagicMock(), MagicMock())

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_crawls_allowed_page_and_extracts_clean_text(self):
        results = self.crawler.retrieve(self._url("/index.html"), 1, False)

        self.assertEqual(len(results), 1)
        doc = results[0]
        self.assertIsInstance(doc, Text)
        self.assertEqual(doc.retriever_type, RetrieverType.WEB_CRAWL)
        self.assertIn("Welcome", doc.content)
        self.assertIn("This is a test page with irregular spacing.", doc.content)
        # script/style/noscript content must be stripped out entirely
        self.assertNotIn("should be stripped", doc.content)
        self.assertNotIn("color: red", doc.content)
        self.assertNotIn("Enable JS", doc.content)

    def test_robots_txt_blocks_disallowed_path(self):
        results = self.crawler.retrieve(self._url("/private/secret.html"), 1, False)

        self.assertIn("disallowed by robots.txt", results[0].content)
        self.assertNotIn("Top secret", results[0].content)

    def test_robots_txt_allows_non_disallowed_path(self):
        results = self.crawler.retrieve(self._url("/public/page.html"), 1, False)

        self.assertIn("Public page content.", results[0].content)

    def test_max_chars_truncates_extracted_text(self):
        self.config.WEB_CRAWL_MAX_CHARS = 20
        crawler = WebCrawler("uX", "cX", self.config, MagicMock(), MagicMock())

        results = crawler.retrieve(self._url("/index.html"), 1, False)

        self.assertLessEqual(len(results[0].content), 20)

    def test_missing_page_returns_error_message_instead_of_raising(self):
        results = self.crawler.retrieve(self._url("/does-not-exist.html"), 1, False)

        self.assertIsInstance(results[0], Text)
        self.assertIn("Error fetching content", results[0].content)


if __name__ == "__main__":
    unittest.main()
