import unittest
from unittest.mock import patch

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

        title, content = article.extract_article_text(html)

        self.assertEqual(title, "Example Story")
        self.assertIn("first meaningful paragraph", content)
        self.assertNotIn("Navigation", content)
        self.assertNotIn("Footer", content)

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
