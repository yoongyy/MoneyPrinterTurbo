import html
import ipaddress
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.parse import urlunparse

import requests
from loguru import logger

from app.config import config


MAX_ARTICLE_BYTES = 4 * 1024 * 1024
MAX_ARTICLE_CHARACTERS = 60_000
MAX_REDIRECTS = 5
MIN_USEFUL_CONTENT_CHARACTERS = 200


@dataclass(frozen=True)
class ArticleContent:
    source_url: str
    final_url: str
    title: str
    content: str
    cover_image_url: str = ""


class _ArticleHTMLParser(HTMLParser):
    _SKIPPED_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "form",
        "nav",
        "footer",
        "header",
    }
    _CONTENT_TAGS = {
        "article",
        "main",
        "p",
        "h1",
        "h2",
        "h3",
        "li",
        "blockquote",
        "section",
    }
    _BLOCK_TAGS = {"p", "h1", "h2", "h3", "li", "blockquote", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._content_depth = 0
        self._preferred_depth = 0
        self._in_title = False
        self._title_fragments: list[str] = []
        self._general_fragments: list[str] = []
        self._preferred_fragments: list[str] = []
        self._meta_title = ""
        self._meta_image = ""

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "title":
            self._in_title = True
        if tag in {"article", "main"}:
            self._preferred_depth += 1
        if tag in self._CONTENT_TAGS:
            self._content_depth += 1
        if tag == "meta":
            values = {str(key).lower(): str(value) for key, value in attrs if value}
            if values.get("property", "").lower() in {"og:title", "twitter:title"}:
                self._meta_title = values.get("content", "").strip()
            if values.get("property", "").lower() in {"og:image", "twitter:image", "twitter:image:src"}:
                self._meta_image = values.get("content", "").strip()
            if values.get("name", "").lower() in {"twitter:image", "twitter:image:src"}:
                self._meta_image = values.get("content", "").strip()

    def handle_startendtag(self, tag: str, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._skip_depth:
            if tag in self._SKIPPED_TAGS:
                self._skip_depth -= 1
            return

        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            if self._content_depth:
                self._general_fragments.append("\n")
            if self._preferred_depth:
                self._preferred_fragments.append("\n")
        if tag in self._CONTENT_TAGS and self._content_depth:
            self._content_depth -= 1
        if tag in {"article", "main"} and self._preferred_depth:
            self._preferred_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        value = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if not value:
            return
        if self._in_title:
            self._title_fragments.append(value)
        if self._content_depth:
            self._general_fragments.append(value)
        if self._preferred_depth:
            self._preferred_fragments.append(value)

    @staticmethod
    def _clean_fragments(fragments: list[str]) -> str:
        raw = " ".join(fragments).replace(" \n ", "\n").replace("\n ", "\n")
        paragraphs = []
        seen = set()
        for line in raw.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line or line in seen:
                continue
            seen.add(line)
            paragraphs.append(line)
        return "\n\n".join(paragraphs)

    def result(self) -> tuple[str, str, str]:
        title = self._meta_title or " ".join(self._title_fragments).strip()
        preferred = self._clean_fragments(self._preferred_fragments)
        general = self._clean_fragments(self._general_fragments)
        content = preferred if len(preferred) >= MIN_USEFUL_CONTENT_CHARACTERS else general
        return title, content, self._meta_image.strip()


def extract_article_text(html_content: str) -> tuple[str, str, str]:
    parser = _ArticleHTMLParser()
    parser.feed(html_content)
    parser.close()
    return parser.result()


def _validate_public_url(url: str) -> str:
    normalized = (url or "").strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("article URL must use http or https")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("article URL must contain a public hostname without credentials")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise ValueError("article URL hostname could not be resolved") from exc

    if not addresses:
        raise ValueError("article URL hostname did not resolve to an address")
    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
                raise ValueError("article URL must resolve only to public internet addresses")
        except ValueError as exc:
            if "public internet" in str(exc):
                raise
            raise ValueError("article URL resolved to an invalid address") from exc
    return normalized


def safe_display_url(url: str) -> str:
    parsed = urlparse(url or "")
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, "", "")
    )


def _read_response_text(response: requests.Response) -> str:
    total = 0
    chunks = []
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_ARTICLE_BYTES:
            raise ValueError("article response is larger than the configured limit")
        chunks.append(chunk)
    encoding = response.encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


def _fetch_text(url: str) -> tuple[str, str, str]:
    current_url = _validate_public_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36 MoneyPrinterTurbo/1.3"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1",
    }
    verify_tls = config.app.get("tls_verify", True)
    if isinstance(verify_tls, str):
        verify_tls = verify_tls.strip().lower() not in {"0", "false", "no", "off"}

    with requests.Session() as session:
        for _ in range(MAX_REDIRECTS + 1):
            response = session.get(
                current_url,
                headers=headers,
                timeout=(10, 30),
                verify=verify_tls,
                allow_redirects=False,
                stream=True,
            )
            if 300 <= response.status_code < 400:
                location = response.headers.get("Location", "")
                response.close()
                if not location:
                    raise ValueError("article URL redirected without a destination")
                current_url = _validate_public_url(urljoin(current_url, location))
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not any(
                value in content_type for value in ("text/", "html", "json", "xml")
            ):
                response.close()
                raise ValueError(f"article URL returned unsupported content type: {content_type}")
            try:
                return current_url, content_type, _read_response_text(response)
            finally:
                response.close()

    raise ValueError(f"article URL exceeded {MAX_REDIRECTS} redirects")


def _jina_reader_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    # Query strings frequently contain tracking or signed-access tokens. They are
    # required for the direct request but must not be forwarded to a fallback service.
    return f"https://r.jina.ai/http://{parsed.netloc}{path}"


def _should_skip_jina_fallback(url: str) -> bool:
    hostname = (urlparse(url or "").hostname or "").lower().rstrip(".")
    return hostname == "toutiao.com" or hostname.endswith(".toutiao.com")


def _extract_first_image_url(text: str) -> str:
    patterns = (
        r"!\[[^\]]*?\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        r'<img[^>]+src=["\']([^"\']+)["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_jina_content(text: str) -> tuple[str, str, str]:
    title_match = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
    content_match = re.search(r"Markdown Content:\s*(.*)$", text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    content = content_match.group(1).strip() if content_match else text.strip()
    cover_image_url = _extract_first_image_url(content) or _extract_first_image_url(text)
    return title, content, cover_image_url


def _chromium_executable() -> str:
    for candidate in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
    ):
        executable = shutil.which(candidate)
        if executable:
            return executable
    return ""


def _fetch_rendered_html(url: str) -> str:
    chromium = _chromium_executable()
    if not chromium:
        raise ValueError("headless Chromium is not available")

    last_error: Exception | None = None
    max_attempts = 3
    for attempt in range(max_attempts):
        command = [
            chromium,
            "--headless",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--mute-audio",
            "--virtual-time-budget=10000",
            "--dump-dom",
            url,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=18,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            logger.warning(
                f"Chromium render timed out, attempt {attempt + 1}/{max_attempts}"
            )
            continue
        except OSError as exc:
            raise ValueError(f"headless Chromium could not be started: {exc}") from exc

        rendered_html = (result.stdout or "").strip()
        if rendered_html:
            return rendered_html

        stderr = (result.stderr or "").strip()
        if stderr:
            logger.warning(f"Chromium render produced no output: {stderr[:500]}")
        last_error = ValueError("headless Chromium did not return rendered HTML")

    if isinstance(last_error, subprocess.TimeoutExpired):
        raise ValueError("headless Chromium timed out while rendering the article") from last_error
    if last_error:
        raise ValueError(str(last_error)) from last_error
    raise ValueError("headless Chromium did not return rendered HTML")


def fetch_url_content(url: str) -> ArticleContent:
    source_url = _validate_public_url(url)
    logger.info(f"fetching article content from: {safe_display_url(source_url)}")
    final_url, content_type, raw = _fetch_text(source_url)

    if "html" in content_type or "xml" in content_type or "<html" in raw[:500].lower():
        title, content, cover_image_url = extract_article_text(raw)
    else:
        title, content, cover_image_url = "", raw.strip(), ""

    if (
        len(content) < MIN_USEFUL_CONTENT_CHARACTERS
        and config.app.get("url_content_use_jina_fallback", True)
    ):
        if _should_skip_jina_fallback(final_url):
            logger.info(
                "direct article extraction was sparse, skipping Jina Reader for a "
                "known blocked domain and trying the local Chromium render fallback"
            )
        else:
            logger.info(
                "direct article extraction was sparse, trying the Jina Reader fallback"
            )
            try:
                _, _, jina_raw = _fetch_text(_jina_reader_url(final_url))
            except Exception as exc:
                logger.warning(f"Jina Reader fallback failed: {exc}")
            else:
                jina_title, jina_content, jina_cover_image_url = _extract_jina_content(
                    jina_raw
                )
                if len(jina_content) > len(content):
                    title = jina_title or title
                    content = jina_content
                if not cover_image_url and jina_cover_image_url:
                    cover_image_url = jina_cover_image_url

    if len(content) < MIN_USEFUL_CONTENT_CHARACTERS:
        logger.info(
            "article extraction was still sparse, trying the local Chromium render fallback"
        )
        try:
            rendered_html = _fetch_rendered_html(final_url)
        except Exception as exc:
            logger.warning(f"Chromium render fallback failed: {exc}")
        else:
            chromium_title, chromium_content, chromium_cover_image_url = (
                extract_article_text(rendered_html)
            )
            if len(chromium_content) > len(content):
                title = chromium_title or title
                content = chromium_content
            if not cover_image_url and chromium_cover_image_url:
                cover_image_url = chromium_cover_image_url

    content = content.strip()[:MAX_ARTICLE_CHARACTERS]
    if len(content) < 80:
        if not title.strip():
            raise ValueError("article URL did not contain enough readable content")
        logger.warning(
            "article URL yielded sparse content; continuing with the extracted title"
        )

    if cover_image_url:
        cover_image_url = urljoin(final_url, cover_image_url)

    return ArticleContent(
        source_url=source_url,
        final_url=final_url,
        title=title.strip(),
        content=content,
        cover_image_url=cover_image_url,
    )
