import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoParams
from app.services import content_source, llm, task as task_service, upload_post
from app.services import state as sm
from app.utils import utils

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
except ImportError:  # pragma: no cover - dependency is installed by normal setup
    YoutubeDL = None

    class DownloadError(Exception):
        pass


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_READY_DIR = "storage/ready_to_upload"
DEFAULT_STATE_DB = "storage/automation/automation.sqlite3"
DEFAULT_MAX_CANDIDATES_PER_SOURCE = 12
DEFAULT_MAX_TOTAL_CANDIDATES = 40
DEFAULT_MIN_CONTENT_CHARS = 120
DEFAULT_REQUEST_TIMEOUT = 25
SOURCE_TYPE_ARTICLE = content_source.SOURCE_ARTICLE
SOURCE_TYPE_YOUTUBE = content_source.SOURCE_YOUTUBE
SOURCE_TYPE_TIKTOK = content_source.SOURCE_TIKTOK
SUPPORTED_CRAWL_MODES = {"links", "yt-dlp", "single"}
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "spm",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
VIDEO_PARAM_CONFIG_KEYS = {
    "video_aspect",
    "script_style",
    "video_layout_style",
    "news_card_background_color",
    "video_duration_minutes",
    "video_clip_duration",
    "video_count",
    "video_source",
    "video_language",
    "voice_name",
    "voice_name_2",
    "voice_volume",
    "voice_rate",
    "dialogue_pause",
    "bgm_type",
    "bgm_file",
    "bgm_volume",
    "subtitle_enabled",
    "subtitle_position",
    "custom_position",
    "font_name",
    "text_fore_color",
    "text_background_color",
    "rounded_subtitle_background",
    "font_size",
    "stroke_color",
    "stroke_width",
    "n_threads",
    "generate_video_cover",
    "cover_generation_provider",
    "paragraph_number",
    "video_script_prompt",
    "custom_system_prompt",
    "match_materials_to_script",
}


@dataclass(frozen=True)
class SourceCandidate:
    source_name: str
    source_type: str
    url: str
    title: str = ""
    summary: str = ""


@dataclass(frozen=True)
class SelectedSource:
    candidate: SourceCandidate
    reason: str = ""


@dataclass(frozen=True)
class ValidatedSource:
    source: content_source.SourceContent
    title: str
    reason: str = ""


@dataclass(frozen=True)
class AutomationRunResult:
    success: bool
    task_id: str = ""
    title: str = ""
    source_url: str = ""
    ready_dir: str = ""
    manifest_path: str = ""
    video_paths: tuple[str, ...] = ()
    error: str = ""


class SeenStore:
    def __init__(self, db_path: str):
        self.db_path = str(_resolve_project_path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_sources (
                    url TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    title TEXT,
                    task_id TEXT,
                    ready_dir TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(seen_sources)").fetchall()
            }
            if "content_hash" not in columns:
                conn.execute("ALTER TABLE seen_sources ADD COLUMN content_hash TEXT")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_seen_sources_content_hash
                ON seen_sources(content_hash)
                """
            )

    def has_success(self, url: str) -> bool:
        normalized = normalize_url(url)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM seen_sources WHERE url = ?", (normalized,)
            ).fetchone()
        return bool(row and row[0] == "complete")

    def should_skip(self, url: str, retry_failed: bool = False) -> bool:
        normalized = normalize_url(url)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM seen_sources WHERE url = ?", (normalized,)
            ).fetchone()
        if not row:
            return False
        if row[0] in {"complete", "processing", "duplicate"}:
            return True
        return not retry_failed

    def reserve(
        self,
        url: str,
        title: str = "",
        retry_failed: bool = False,
        content_hash: str = "",
    ) -> bool:
        normalized = normalize_url(url)
        if not normalized:
            return False

        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM seen_sources WHERE url = ?", (normalized,)
            ).fetchone()
            if row:
                status = row[0]
                if status == "failed" and retry_failed:
                    conn.execute(
                        """
                        UPDATE seen_sources
                        SET status = ?, title = ?, error = ?, content_hash = ?,
                            updated_at = ?
                        WHERE url = ?
                        """,
                        ("processing", title, "", content_hash, now, normalized),
                    )
                    return True
                return False

            conn.execute(
                """
                INSERT INTO seen_sources (
                    url, status, title, task_id, ready_dir, error, created_at,
                    updated_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (normalized, "processing", title, "", "", "", now, now, content_hash),
            )
            return True

    def has_content_hash(self, content_hash: str, retry_failed: bool = False) -> bool:
        value = (content_hash or "").strip()
        if not value:
            return False

        if retry_failed:
            statuses = ("complete", "processing", "duplicate")
        else:
            statuses = ("complete", "processing", "duplicate", "failed")
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1 FROM seen_sources
                WHERE content_hash = ? AND status IN ({placeholders})
                LIMIT 1
                """,
                (value, *statuses),
            ).fetchone()
        return bool(row)

    def upsert(
        self,
        url: str,
        status: str,
        title: str = "",
        task_id: str = "",
        ready_dir: str = "",
        error: str = "",
        content_hash: str = "",
    ) -> None:
        normalized = normalize_url(url)
        if not normalized:
            return
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO seen_sources (
                    url, status, title, task_id, ready_dir, error, created_at,
                    updated_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    status = excluded.status,
                    title = excluded.title,
                    task_id = excluded.task_id,
                    ready_dir = excluded.ready_dir,
                    error = excluded.error,
                    content_hash = COALESCE(NULLIF(excluded.content_hash, ''), seen_sources.content_hash),
                    updated_at = excluded.updated_at
                """,
                (
                    normalized,
                    status,
                    title,
                    task_id,
                    ready_dir,
                    error,
                    now,
                    now,
                    content_hash,
                ),
            )

    def upsert_many(
        self,
        urls: list[str],
        status: str,
        title: str = "",
        task_id: str = "",
        ready_dir: str = "",
        error: str = "",
        content_hash: str = "",
    ) -> None:
        for url in _unique_urls(urls):
            self.upsert(
                url=url,
                status=status,
                title=title,
                task_id=task_id,
                ready_dir=ready_dir,
                error=error,
                content_hash=content_hash,
            )


class LinkExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[dict[str, str]] = []
        self._current_href = ""
        self._current_title = ""
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        href = values.get("href", "").strip()
        if not href:
            return
        self._flush_current()
        self._current_href = urljoin(self.base_url, href)
        self._current_title = values.get("title", "").strip()
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._flush_current()

    def close(self) -> None:
        self._flush_current()
        super().close()

    def _flush_current(self) -> None:
        if not self._current_href:
            return
        text = " ".join(part.strip() for part in self._current_text if part.strip())
        title = re.sub(r"\s+", " ", self._current_title or text).strip()
        self.links.append({"url": self._current_href, "title": title})
        self._current_href = ""
        self._current_title = ""
        self._current_text = []


def _resolve_project_path(path_value: str | os.PathLike) -> Path:
    path = Path(str(path_value or "")).expanduser()
    if path.is_absolute():
        return path
    return Path(utils.root_dir()) / path


def normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return (url or "").strip()
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
        and not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            (parsed.path or "/").rstrip("/") or "/",
            parsed.params,
            query,
            "",
        )
    )


def _unique_urls(urls: list[str]) -> list[str]:
    result = []
    seen = set()
    for url in urls:
        normalized = normalize_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _source_urls(source: content_source.SourceContent, candidate_url: str = "") -> list[str]:
    return _unique_urls(
        [
            candidate_url,
            source.source_url,
            source.final_url,
        ]
    )


def source_content_hash(source: content_source.SourceContent) -> str:
    content = re.sub(r"\s+", " ", source.content or "").strip().lower()
    title = re.sub(r"\s+", " ", source.title or "").strip().lower()
    if not content and not title:
        return ""
    digest_text = content[:50_000] if content else title
    digest_source = f"{source.source_type}\n{digest_text}"
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()


def _config_bool(key: str, default: bool = False) -> bool:
    value = _automation_config_value(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _config_int(key: str, default: int) -> int:
    try:
        return int(_automation_config_value(key, default))
    except (TypeError, ValueError):
        return default


def _config_list(key: str) -> list[Any]:
    value = _automation_config_value(key, [])
    return value if isinstance(value, list) else []


def _automation_config_value(key: str, default: Any = None) -> Any:
    raw_config = getattr(config, "_cfg", {})
    automation_cfg = raw_config.get("automation", {}) if isinstance(raw_config, dict) else {}
    if isinstance(automation_cfg, dict):
        if key in automation_cfg:
            return automation_cfg[key]
        if key.startswith("automation_"):
            short_key = key.removeprefix("automation_")
            if short_key in automation_cfg:
                return automation_cfg[short_key]
    return config.app.get(key, default)


def _automation_config_choice(key: str, default: Any = None) -> Any:
    value = _automation_config_value(key, default)
    if not isinstance(value, list):
        return value

    choices = [item for item in value if str(item or "").strip()]
    if not choices:
        return default
    return random.choice(choices)


def _compile_patterns(values: Any) -> list[re.Pattern]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    patterns = []
    for value in values:
        try:
            patterns.append(re.compile(str(value)))
        except re.error as exc:
            logger.warning(f"invalid automation URL pattern '{value}': {exc}")
    return patterns


def _matches_any(patterns: list[re.Pattern], value: str) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def _same_hostname(left: str, right: str) -> bool:
    return (urlparse(left).hostname or "").lower() == (urlparse(right).hostname or "").lower()


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value[:240]


def _dedupe_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    result = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_url(candidate.url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(
            SourceCandidate(
                source_name=candidate.source_name,
                source_type=candidate.source_type,
                url=normalized,
                title=_clean_title(candidate.title),
                summary=_clean_title(candidate.summary),
            )
        )
    return result


def _source_type_from_config(source_cfg: dict[str, Any]) -> str:
    source_type = source_cfg.get("source_type", source_cfg.get("type", SOURCE_TYPE_ARTICLE))
    return content_source.normalize_source_type(str(source_type))


def _crawl_mode_from_config(source_cfg: dict[str, Any], source_type: str) -> str:
    mode = str(source_cfg.get("crawl", "") or "").strip().lower()
    if not mode:
        mode = "links" if source_type == SOURCE_TYPE_ARTICLE else "yt-dlp"
    if mode not in SUPPORTED_CRAWL_MODES:
        raise ValueError(f"unsupported automation crawl mode: {mode}")
    return mode


def crawl_link_page(source_cfg: dict[str, Any]) -> list[SourceCandidate]:
    source_url = str(source_cfg.get("url", "") or "").strip()
    if not source_url:
        return []

    source_name = str(source_cfg.get("name", "") or source_url).strip()
    source_type = _source_type_from_config(source_cfg)
    limit = int(source_cfg.get("max_candidates", DEFAULT_MAX_CANDIDATES_PER_SOURCE) or 0)
    limit = limit if limit > 0 else DEFAULT_MAX_CANDIDATES_PER_SOURCE
    include_patterns = _compile_patterns(source_cfg.get("include_patterns", []))
    exclude_patterns = _compile_patterns(source_cfg.get("exclude_patterns", []))
    allow_external = bool(source_cfg.get("allow_external", False))

    response = requests.get(
        source_url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    response.encoding = response.encoding or response.apparent_encoding

    parser = LinkExtractor(source_url)
    parser.feed(response.text)
    parser.close()

    candidates = []
    for item in parser.links:
        url = normalize_url(item["url"])
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if not allow_external and not _same_hostname(source_url, url):
            continue
        if include_patterns and not _matches_any(include_patterns, url):
            continue
        if exclude_patterns and _matches_any(exclude_patterns, url):
            continue
        title = _clean_title(item.get("title", ""))
        if len(title) < int(source_cfg.get("min_title_chars", 4) or 4):
            continue
        candidates.append(
            SourceCandidate(
                source_name=source_name,
                source_type=source_type,
                url=url,
                title=title,
            )
        )
        if len(candidates) >= limit:
            break

    return _dedupe_candidates(candidates)


def _media_candidate_url(source_type: str, entry: dict[str, Any]) -> str:
    for key in ("webpage_url", "original_url", "url"):
        value = str(entry.get(key, "") or "").strip()
        if value.startswith(("http://", "https://")):
            return value

    video_id = str(entry.get("id", "") or "").strip()
    if source_type == SOURCE_TYPE_YOUTUBE and video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return ""


def crawl_ytdlp_source(source_cfg: dict[str, Any]) -> list[SourceCandidate]:
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is required for YouTube/TikTok automation sources")

    source_url = str(source_cfg.get("url", "") or "").strip()
    if not source_url:
        return []

    source_name = str(source_cfg.get("name", "") or source_url).strip()
    source_type = _source_type_from_config(source_cfg)
    limit = int(source_cfg.get("max_candidates", DEFAULT_MAX_CANDIDATES_PER_SOURCE) or 0)
    limit = limit if limit > 0 else DEFAULT_MAX_CANDIDATES_PER_SOURCE
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": limit,
        "ignoreerrors": True,
    }
    cookie_file = str(config.app.get("media_source_cookie_file", "") or "").strip()
    if cookie_file:
        options["cookiefile"] = cookie_file

    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(source_url, download=False)
    except DownloadError as exc:
        raise ValueError(f"could not crawl media source: {exc}") from exc

    entries = info.get("entries", []) if isinstance(info, dict) else []
    if not entries and isinstance(info, dict):
        entries = [info]

    candidates = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = normalize_url(_media_candidate_url(source_type, entry))
        if not url:
            continue
        title = _clean_title(str(entry.get("title", "") or ""))
        candidates.append(
            SourceCandidate(
                source_name=source_name,
                source_type=source_type,
                url=url,
                title=title,
            )
        )
        if len(candidates) >= limit:
            break
    return _dedupe_candidates(candidates)


def crawl_single_source(source_cfg: dict[str, Any]) -> list[SourceCandidate]:
    source_url = normalize_url(str(source_cfg.get("url", "") or "").strip())
    if not source_url:
        return []
    return [
        SourceCandidate(
            source_name=str(source_cfg.get("name", "") or source_url).strip(),
            source_type=_source_type_from_config(source_cfg),
            url=source_url,
            title=_clean_title(str(source_cfg.get("title", "") or "")),
        )
    ]


def crawl_sources(store: SeenStore | None = None) -> list[SourceCandidate]:
    sources = _config_list("automation_sources")
    if not sources:
        logger.warning("automation_sources is empty; nothing to crawl")
        return []

    all_candidates: list[SourceCandidate] = []
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            logger.warning(f"skipping invalid automation source config: {raw_source}")
            continue
        enabled = raw_source.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
        if not enabled:
            continue

        try:
            source_type = _source_type_from_config(raw_source)
            crawl_mode = _crawl_mode_from_config(raw_source, source_type)
            if crawl_mode == "single":
                candidates = crawl_single_source(raw_source)
            elif crawl_mode == "yt-dlp":
                candidates = crawl_ytdlp_source(raw_source)
            else:
                candidates = crawl_link_page(raw_source)
        except Exception as exc:
            logger.warning(
                f"failed to crawl automation source '{raw_source.get('name', raw_source.get('url', ''))}': {exc}"
            )
            continue

        if store is not None:
            retry_failed = _config_bool("automation_retry_failed_sources", False)
            candidates = [
                item for item in candidates if not store.should_skip(item.url, retry_failed)
            ]
        logger.info(
            f"crawled {len(candidates)} candidate(s) from {raw_source.get('name', raw_source.get('url', 'source'))}"
        )
        all_candidates.extend(candidates)

    max_total = _config_int("automation_max_total_candidates", DEFAULT_MAX_TOTAL_CANDIDATES)
    return _dedupe_candidates(all_candidates)[:max_total]


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object")
    return payload


def select_candidate_with_ai(candidates: list[SourceCandidate]) -> SelectedSource:
    if not candidates:
        raise ValueError("no automation candidates available")
    if len(candidates) == 1:
        return SelectedSource(candidate=candidates[0], reason="only candidate")

    topic = str(_automation_config_value("automation_topic", "") or "").strip()
    compact_candidates = [
        {
            "index": index,
            "source": candidate.source_name,
            "type": candidate.source_type,
            "title": candidate.title,
            "url": candidate.url,
        }
        for index, candidate in enumerate(candidates)
    ]
    prompt = f"""
Choose exactly one source item for a short-form news video.

Rules:
- Prefer recent, concrete, factual news or explainable current events.
- Avoid ads, category pages, tag pages, navigation pages, duplicate items, and vague teaser titles.
- Avoid topics that cannot be responsibly summarized from a single source.
- If a topic preference is provided, prefer items matching it.
- Return only minified JSON with keys: "index", "reason".

Topic preference: {topic or "general news suitable for a short vertical video"}

Candidates:
{json.dumps(compact_candidates, ensure_ascii=False)}
""".strip()

    response = llm._generate_response(prompt)
    if isinstance(response, str) and response.startswith("Error: "):
        logger.warning(f"AI candidate selection failed, using first candidate: {response}")
        return SelectedSource(candidate=candidates[0], reason="AI selection unavailable")

    try:
        payload = parse_json_object(response)
        index = int(payload.get("index", 0))
        if index < 0 or index >= len(candidates):
            raise ValueError("selected index out of range")
        reason = str(payload.get("reason", "") or "").strip()
        return SelectedSource(candidate=candidates[index], reason=reason)
    except Exception as exc:
        logger.warning(f"failed to parse AI candidate selection, using first candidate: {exc}")
        return SelectedSource(candidate=candidates[0], reason="AI selection parse failed")


def _simple_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?。！？])\s+", text.strip())[0][:160]


def validate_source_with_ai(source: content_source.SourceContent) -> ValidatedSource:
    min_content_chars = _config_int("automation_min_content_chars", DEFAULT_MIN_CONTENT_CHARS)
    title = _clean_title(source.title)
    content = (source.content or "").strip()
    if len(content) < min_content_chars:
        raise ValueError(
            f"source content is too short: {len(content)} chars, need at least {min_content_chars}"
        )

    topic = str(_automation_config_value("automation_topic", "") or "").strip()
    prompt = f"""
Validate this source for an automated short-form news video.

Rules:
- valid must be false if the content is empty, unreadable, mostly boilerplate, or not suitable for a factual short video.
- valid must be false if the title and content are unrelated.
- If the title is weak, vague, clickbait, or not aligned with the content, rewrite it.
- The rewritten title must be concise, factual, and in the same language as the source.
- Do not invent facts that are not supported by the content.
- Return only minified JSON with keys: "valid", "title", "reason".

Topic preference: {topic or "general news suitable for a short vertical video"}
URL: {source.final_url}
Original title: {title}
Content preview:
{content[:6000]}
""".strip()

    response = llm._generate_response(prompt)
    if isinstance(response, str) and response.startswith("Error: "):
        logger.warning(f"AI source validation failed, using heuristic validation: {response}")
        return ValidatedSource(source=source, title=title or _simple_sentence(content))

    try:
        payload = parse_json_object(response)
        if not bool(payload.get("valid", False)):
            raise ValueError(str(payload.get("reason", "AI rejected source")))
        rewritten_title = _clean_title(str(payload.get("title", "") or ""))
        if not rewritten_title:
            rewritten_title = title or _simple_sentence(content)
        return ValidatedSource(
            source=source,
            title=rewritten_title,
            reason=str(payload.get("reason", "") or "").strip(),
        )
    except Exception as exc:
        raise ValueError(f"source validation failed: {exc}") from exc


def _build_video_params(validated: ValidatedSource) -> VideoParams:
    automation_cover_provider = str(
        _automation_config_value("automation_cover_generation_provider", "") or ""
    ).strip()
    cover_generation_provider = (
        automation_cover_provider
        or str(config.app.get("cover_generation_provider", "") or "").strip()
        or "local"
    )
    params_data: dict[str, Any] = {
        "video_subject": validated.title,
        "source_type": validated.source.source_type,
        "source_url": validated.source.source_url or validated.source.final_url,
        "source_title": validated.title or validated.source.title,
        "source_cover_image_url": validated.source.cover_image_url,
        "video_duration_minutes": _config_int("automation_video_duration_minutes", 1),
        "video_language": str(_automation_config_value("automation_video_language", "") or ""),
        "script_style": str(_automation_config_value("automation_script_style", "narration") or "narration"),
        "video_source": str(
            _automation_config_value(
                "automation_video_source", config.app.get("video_source", "pexels")
            )
            or "pexels"
        ),
        "video_aspect": str(_automation_config_value("automation_video_aspect", "9:16") or "9:16"),
        "video_layout_style": str(
            _automation_config_value(
                "automation_video_layout_style",
                config.app.get("video_layout_style", "full"),
            )
            or "full"
        ),
        "news_card_background_color": str(
            _automation_config_choice(
                "automation_news_card_background_color",
                config.app.get("news_card_background_color", "#CBEFF7"),
            )
            or "#CBEFF7"
        ),
        "voice_name": str(
            _automation_config_choice(
                "automation_voice_name",
                config.ui.get("voice_name", "zh-CN-XiaoxiaoNeural-Female"),
            )
            or "zh-CN-XiaoxiaoNeural-Female"
        ),
        "voice_name_2": str(
            _automation_config_value(
                "automation_voice_name_2", config.ui.get("voice_name_2", "")
            )
            or ""
        ),
        "bgm_type": str(_automation_config_value("automation_bgm_type", "random") or "random"),
        "bgm_volume": float(_automation_config_value("automation_bgm_volume", 0.2) or 0.2),
        "subtitle_enabled": _config_bool("automation_subtitle_enabled", True),
        "font_name": str(
            _automation_config_value(
                "automation_font_name", config.ui.get("font_name", "STHeitiMedium.ttc")
            )
            or "STHeitiMedium.ttc"
        ),
        "text_fore_color": str(
            _automation_config_value(
                "automation_text_fore_color",
                config.ui.get("text_fore_color", "#FFFFFF"),
            )
            or "#FFFFFF"
        ),
        "stroke_color": str(_automation_config_value("automation_stroke_color", "#000000") or "#000000"),
        "stroke_width": float(_automation_config_value("automation_stroke_width", 1.5) or 1.5),
        "video_clip_duration": _config_int("automation_video_clip_duration", 3),
        "video_count": _config_int("automation_video_count", 1),
        "n_threads": _config_int("automation_n_threads", 2),
        "match_materials_to_script": _config_bool("automation_match_materials_to_script", True),
        "generate_video_cover": _config_bool("automation_generate_video_cover", True),
        "cover_generation_provider": cover_generation_provider,
        "paragraph_number": _config_int("automation_paragraph_number", 1),
        "video_script_prompt": str(_automation_config_value("automation_video_script_prompt", "") or ""),
        "custom_system_prompt": str(_automation_config_value("automation_custom_system_prompt", "") or ""),
    }

    overrides = _automation_config_value("automation_video_params", {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in VIDEO_PARAM_CONFIG_KEYS:
                params_data[key] = value
            else:
                logger.warning(f"ignoring unsupported automation_video_params key: {key}")

    return VideoParams(**params_data)


def _slugify(value: str, max_length: int = 80) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value or "", flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip(".-")
    return value[:max_length].strip(".-") or "video"


def _ready_video_filename(title: str, index: int, total: int, suffix: str) -> str:
    stem = _slugify(title)
    sequence = f"-{index}" if total > 1 else ""
    extension = suffix if suffix else ".mp4"
    return f"{stem}{sequence}{extension}"


def _copy_outputs_to_ready_dir(
    task_id: str,
    validated: ValidatedSource,
    task_result: dict[str, Any],
    selected_reason: str,
) -> tuple[Path, Path, tuple[str, ...]]:
    ready_root = _resolve_project_path(
        str(_automation_config_value("automation_ready_dir", DEFAULT_READY_DIR) or DEFAULT_READY_DIR)
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ready_dir = ready_root / f"{timestamp}-{_slugify(validated.title)}"
    ready_dir.mkdir(parents=True, exist_ok=True)

    video_outputs = task_result.get("videos") or []
    copied_videos = []
    for index, video_path in enumerate(video_outputs, start=1):
        source_path = Path(str(video_path))
        if not source_path.is_file():
            continue
        target_path = ready_dir / _ready_video_filename(
            validated.title,
            index=index,
            total=len(video_outputs),
            suffix=source_path.suffix,
        )
        shutil.copy2(source_path, target_path)
        copied_videos.append(str(target_path))

    cover_target = ""
    cover_path = task_result.get("cover_image")
    if cover_path and Path(str(cover_path)).is_file():
        cover_source = Path(str(cover_path))
        target_path = ready_dir / f"cover{cover_source.suffix or '.jpg'}"
        shutil.copy2(cover_source, target_path)
        cover_target = str(target_path)

    metadata = llm.generate_social_metadata(
        video_subject=validated.title,
        video_script=str(task_result.get("script", "") or ""),
        language=str(_automation_config_value("automation_video_language", "") or "auto"),
        platform="tiktok",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": task_id,
        "title": validated.title,
        "source": asdict(validated.source),
        "selection_reason": selected_reason,
        "validation_reason": validated.reason,
        "videos": copied_videos,
        "cover_image": cover_target,
        "script": task_result.get("script", ""),
        "terms": task_result.get("terms", []),
        "social_metadata": metadata,
        "publish_mode": "manual",
        "notes": "Prepared for manual TikTok upload. No automatic TikTok upload was attempted.",
    }
    manifest_path = ready_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ready_dir, manifest_path, tuple(copied_videos)


def generate_video_for_source(
    validated: ValidatedSource,
    selected_reason: str,
    store: SeenStore,
    source_urls: list[str],
    content_hash: str,
) -> AutomationRunResult:
    params = _build_video_params(validated)
    task_id = utils.get_uuid()
    original_auto_upload = upload_post.upload_post_service.auto_upload
    try:
        upload_post.upload_post_service.auto_upload = False
        sm.state.update_task(task_id)
        task_result = task_service.start(task_id, params, stop_at="video") or {}
    finally:
        upload_post.upload_post_service.auto_upload = original_auto_upload

    task_state = sm.state.get_task(task_id) or {}
    if task_state.get("state") == const.TASK_STATE_FAILED or not task_result.get("videos"):
        error = "video generation failed"
        store.upsert_many(
            source_urls,
            "failed",
            validated.title,
            task_id,
            error=error,
            content_hash=content_hash,
        )
        return AutomationRunResult(
            success=False,
            task_id=task_id,
            title=validated.title,
            source_url=validated.source.final_url,
            error=error,
        )

    ready_dir, manifest_path, video_paths = _copy_outputs_to_ready_dir(
        task_id=task_id,
        validated=validated,
        task_result=task_result,
        selected_reason=selected_reason,
    )
    store.upsert_many(
        source_urls,
        "complete",
        validated.title,
        task_id,
        str(ready_dir),
        content_hash=content_hash,
    )
    return AutomationRunResult(
        success=True,
        task_id=task_id,
        title=validated.title,
        source_url=validated.source.final_url,
        ready_dir=str(ready_dir),
        manifest_path=str(manifest_path),
        video_paths=video_paths,
    )


def run_once() -> AutomationRunResult:
    if not _config_bool("automation_enabled", False):
        return AutomationRunResult(
            success=False,
            error="automation_enabled is false in config.toml",
        )

    store = SeenStore(str(_automation_config_value("automation_state_db", DEFAULT_STATE_DB)))
    candidates = crawl_sources(store=store)
    if not candidates:
        return AutomationRunResult(success=False, error="no new source candidates found")

    max_attempts = max(1, _config_int("automation_selection_attempts", 5))
    attempted_urls = set()
    last_error = ""
    remaining = candidates
    for _ in range(min(max_attempts, len(candidates))):
        selection = select_candidate_with_ai(remaining)
        candidate = selection.candidate
        attempted_urls.add(candidate.url)
        retry_failed = _config_bool("automation_retry_failed_sources", False)
        if not store.reserve(candidate.url, candidate.title, retry_failed=retry_failed):
            logger.info(f"skip already-seen automation source: {candidate.url}")
            remaining = [item for item in remaining if item.url not in attempted_urls]
            if not remaining:
                break
            continue
        current_source_urls = [candidate.url]
        current_content_hash = ""
        try:
            logger.info(f"selected automation source: {candidate.title or candidate.url}")
            source = content_source.fetch_source_content(candidate.source_type, candidate.url)
            source_urls = _source_urls(source, candidate.url)
            current_source_urls = source_urls
            duplicate_url = next(
                (
                    url
                    for url in source_urls
                    if url != normalize_url(candidate.url)
                    and store.should_skip(url, retry_failed=retry_failed)
                ),
                "",
            )
            if duplicate_url:
                last_error = f"duplicate source URL already processed: {duplicate_url}"
                logger.info(last_error)
                store.upsert_many(
                    source_urls,
                    "duplicate",
                    source.title or candidate.title,
                    error=last_error,
                )
                remaining = [item for item in remaining if item.url not in attempted_urls]
                if not remaining:
                    break
                continue

            content_hash = source_content_hash(source)
            current_content_hash = content_hash
            if store.has_content_hash(content_hash, retry_failed=retry_failed):
                last_error = "duplicate source content already processed"
                logger.info(last_error)
                store.upsert_many(
                    source_urls,
                    "duplicate",
                    source.title or candidate.title,
                    error=last_error,
                    content_hash=content_hash,
                )
                remaining = [item for item in remaining if item.url not in attempted_urls]
                if not remaining:
                    break
                continue

            store.upsert_many(
                source_urls,
                "processing",
                source.title or candidate.title,
                content_hash=content_hash,
            )
            validated = validate_source_with_ai(source)
            return generate_video_for_source(
                validated=validated,
                selected_reason=selection.reason,
                store=store,
                source_urls=source_urls,
                content_hash=content_hash,
            )
        except Exception as exc:
            last_error = str(exc)
            logger.warning(f"automation candidate failed, trying another: {last_error}")
            store.upsert_many(
                current_source_urls,
                "failed",
                candidate.title,
                error=last_error,
                content_hash=current_content_hash,
            )
            remaining = [item for item in remaining if item.url not in attempted_urls]
            if not remaining:
                break

    return AutomationRunResult(
        success=False,
        error=last_error or "all selected candidates failed",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one automated source crawl -> video generation cycle."
    )
    parser.add_argument(
        "--ignore-disabled",
        action="store_true",
        help="Run even when automation_enabled=false.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.ignore_disabled:
        config.app["automation_enabled"] = True

    result = run_once()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
