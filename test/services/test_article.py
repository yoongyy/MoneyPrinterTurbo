import subprocess
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import requests

# 测试文件直接运行时，也能从仓库根目录导入 app 包。
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import article


class TestArticleService(unittest.TestCase):
    def test_extract_article_prefers_article_content_and_title(self):
        html = """
        <html><head><title>Example Story</title></head><body>
          <nav>Navigation should not be included.</nav>
          <article>
            <h1>Example Story</h1>
            <p>This is the first meaningful paragraph with enough useful context for a video.</p>
            <p>This is the second meaningful paragraph explaining the conclusion of the story.</p>
            <p>This final paragraph adds enough detail to make extraction clearly useful for generation.</p>
          </article>
          <footer>Footer should not be included.</footer>
        </body></html>
        """

        title, content, cover_image_url = article.extract_article_text(html)

        self.assertEqual(title, "Example Story")
        self.assertIn("first meaningful paragraph", content)
        self.assertNotIn("Navigation", content)
        self.assertNotIn("Footer", content)
        self.assertEqual(cover_image_url, "")

    @patch("app.services.article.socket.getaddrinfo")
    def test_validate_public_url_rejects_private_addresses(self, getaddrinfo):
        getaddrinfo.return_value = [(None, None, None, None, ("127.0.0.1", 80))]

        with self.assertRaisesRegex(ValueError, "public internet"):
            article._validate_public_url("http://internal.example/article")

    @patch("app.services.article._fetch_text")
    @patch("app.services.article._validate_public_url")
    def test_fetch_url_content_returns_extracted_article(self, validate, fetch):
        validate.return_value = "https://example.com/story"
        fetch.return_value = (
            "https://example.com/story",
            "text/html; charset=utf-8",
            "<title>Story</title><article><p>" + "Useful article sentence. " * 20 + "</p></article>",
        )

        result = article.fetch_url_content("https://example.com/story")

        self.assertEqual(result.title, "Story")
        self.assertIn("Useful article sentence", result.content)
        fetch.assert_called_once_with("https://example.com/story")

    @patch("app.services.article._fetch_text")
    @patch("app.services.article._validate_public_url")
    def test_fetch_url_content_uses_jina_fallback_cover_image(self, validate, fetch):
        validate.return_value = "https://example.com/story"
        fetch.side_effect = [
            (
                "https://example.com/story",
                "text/html; charset=utf-8",
                "<title>Story</title><article><p>Short text.</p></article>",
            ),
            (
                "https://r.jina.ai/http://example.com/story",
                "text/plain; charset=utf-8",
                "Title: Story\nMarkdown Content:\n\n![Cover](https://example.com/cover.jpg)\n\n"
                + "Readable text. " * 10,
            ),
        ]

        result = article.fetch_url_content("https://example.com/story")

        self.assertEqual(result.title, "Story")
        self.assertEqual(result.cover_image_url, "https://example.com/cover.jpg")
        self.assertIn("Readable text", result.content)

    @patch("app.services.article._fetch_rendered_html", side_effect=ValueError("no chromium"))
    @patch("app.services.article._fetch_text")
    @patch("app.services.article._validate_public_url")
    def test_fetch_url_content_tolerates_jina_block_when_title_exists(
        self, validate, fetch, fetch_rendered_html
    ):
        validate.return_value = "https://example.com/story"
        fetch.side_effect = [
            (
                "https://example.com/story",
                "text/html; charset=utf-8",
                "<title>Story</title><article><p>Short text.</p></article>",
            ),
            requests.HTTPError("401 Client Error: Unauthorized"),
        ]

        result = article.fetch_url_content("https://example.com/story")

        self.assertEqual(result.title, "Story")
        self.assertEqual(result.content, "Short text.")
        self.assertEqual(result.cover_image_url, "")
        fetch_rendered_html.assert_called_once_with("https://example.com/story")

    @patch("app.services.article._fetch_rendered_html")
    @patch("app.services.article._fetch_text")
    @patch("app.services.article._validate_public_url")
    def test_fetch_url_content_uses_chromium_fallback_when_jina_is_blocked(
        self, validate, fetch, fetch_rendered_html
    ):
        validate.return_value = "https://example.com/story"
        fetch.side_effect = [
            (
                "https://example.com/story",
                "text/html; charset=utf-8",
                "<title>Story</title><article><p>Short text.</p></article>",
            ),
            requests.HTTPError("401 Client Error: Unauthorized"),
        ]
        fetch_rendered_html.return_value = """
        <html>
          <head>
            <title>Rendered Story</title>
            <meta property="og:image" content="https://example.com/cover.jpg">
          </head>
          <body>
            <article>
              <p>Rendered useful sentence. Rendered useful sentence. Rendered useful sentence.</p>
              <p>Rendered useful sentence. Rendered useful sentence. Rendered useful sentence.</p>
              <p>Rendered useful sentence. Rendered useful sentence. Rendered useful sentence.</p>
              <p>Rendered useful sentence. Rendered useful sentence. Rendered useful sentence.</p>
              <p>Rendered useful sentence. Rendered useful sentence. Rendered useful sentence.</p>
            </article>
          </body>
        </html>
        """

        result = article.fetch_url_content("https://example.com/story")

        self.assertEqual(result.title, "Rendered Story")
        self.assertIn("Rendered useful sentence", result.content)
        self.assertEqual(result.cover_image_url, "https://example.com/cover.jpg")
        fetch_rendered_html.assert_called_once_with("https://example.com/story")

    @patch("app.services.article._fetch_rendered_html")
    @patch("app.services.article._fetch_text")
    @patch("app.services.article._validate_public_url")
    def test_fetch_url_content_skips_jina_for_toutiao_when_chromium_is_available(
        self, validate, fetch, fetch_rendered_html
    ):
        validate.return_value = "https://www.toutiao.com/article/123"
        fetch.return_value = (
            "https://www.toutiao.com/article/123",
            "text/html; charset=utf-8",
            "<title>Toutiao</title><article><p>Short text.</p></article>",
        )
        fetch_rendered_html.return_value = """
        <html>
          <head>
            <title>Rendered Toutiao</title>
            <meta property="og:image" content="https://example.com/cover.jpg">
          </head>
          <body>
            <article>
              <p>Rendered Toutiao sentence. Rendered Toutiao sentence. Rendered Toutiao sentence.</p>
              <p>Rendered Toutiao sentence. Rendered Toutiao sentence. Rendered Toutiao sentence.</p>
              <p>Rendered Toutiao sentence. Rendered Toutiao sentence. Rendered Toutiao sentence.</p>
              <p>Rendered Toutiao sentence. Rendered Toutiao sentence. Rendered Toutiao sentence.</p>
              <p>Rendered Toutiao sentence. Rendered Toutiao sentence. Rendered Toutiao sentence.</p>
            </article>
          </body>
        </html>
        """

        result = article.fetch_url_content(
            "https://www.toutiao.com/article/123?log_from=test"
        )

        self.assertEqual(result.title, "Rendered Toutiao")
        self.assertIn("Rendered Toutiao sentence", result.content)
        self.assertEqual(result.cover_image_url, "https://example.com/cover.jpg")
        fetch.assert_called_once_with("https://www.toutiao.com/article/123")
        fetch_rendered_html.assert_called_once_with("https://www.toutiao.com/article/123")

    @patch("app.services.article._chromium_executable", return_value="/usr/bin/chromium")
    @patch("app.services.article.subprocess.run")
    def test_fetch_rendered_html_retries_after_timeout(self, run, chromium_executable):
        class Completed:
            stdout = "<html><body><article><p>Rendered article text.</p></article></body></html>"
            stderr = ""

        run.side_effect = [
            subprocess.TimeoutExpired(cmd=["chromium"], timeout=25),
            Completed(),
        ]

        rendered_html = article._fetch_rendered_html("https://example.com/story")

        self.assertIn("Rendered article text", rendered_html)
        self.assertEqual(run.call_count, 2)
        chromium_executable.assert_called_once()
