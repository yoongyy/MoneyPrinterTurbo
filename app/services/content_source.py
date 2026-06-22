import html
import mimetypes
import json
import ipaddress
import os
import re
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import subprocess
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from loguru import logger
from pydub import AudioSegment

from app.config import config
from app.services import article, subtitle
from app.utils import utils

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
except ImportError:  # pragma: no cover - exercised only in incomplete installations
    YoutubeDL = None

    class DownloadError(Exception):
        pass


SOURCE_ARTICLE = "article"
SOURCE_YOUTUBE = "youtube"
SOURCE_TIKTOK = "tiktok"
SUPPORTED_SOURCE_TYPES = (SOURCE_ARTICLE, SOURCE_YOUTUBE, SOURCE_TIKTOK)

MAX_TRANSCRIPT_CHARACTERS = 60_000
MIN_TRANSCRIPT_CHARACTERS = 20
DEFAULT_MAX_MEDIA_DURATION_MINUTES = 120
DEFAULT_MAX_MEDIA_DOWNLOAD_MB = 250

_PLATFORM_DOMAINS = {
    SOURCE_YOUTUBE: ("youtube.com", "youtu.be", "youtube-nocookie.com"),
    SOURCE_TIKTOK: ("tiktok.com",),
}
_CAPTION_EXTENSIONS = {
    ".json",
    ".json3",
    ".srv1",
    ".srv2",
    ".srv3",
    ".srt",
    ".ttml",
    ".vtt",
}
_GEMINI_TRANSCRIPTION_PROMPT = (
    "Transcribe the provided audio verbatim. Return only the transcript text, "
    "with no markdown, bullet points, timestamps, or summary. Preserve the "
    "spoken language and keep punctuation natural."
)
_GEMINI_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
_DNS_FALLBACK_CACHE: dict[str, list[str]] = {}


@dataclass(frozen=True)
class SourceContent:
    source_type: str
    source_url: str
    final_url: str
    title: str
    content: str
    extraction_method: str


class _YTDLPLogger:
    def debug(self, message):
        if not str(message).startswith("[debug]"):
            logger.debug(message)

    def warning(self, message):
        logger.warning(message)

    def error(self, message):
        logger.error(message)


def normalize_source_type(source_type: str) -> str:
    value = (
        str(getattr(source_type, "value", source_type) or SOURCE_ARTICLE)
        .strip()
        .lower()
    )
    if value not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"unsupported source type: {value}; expected article, youtube, or tiktok"
        )
    return value


def _hostname_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def validate_media_url(source_type: str, url: str) -> str:
    source_type = normalize_source_type(source_type)
    normalized = (url or "").strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{source_type} URL must use http or https")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(
            f"{source_type} URL must contain a public hostname without credentials"
        )

    hostname = parsed.hostname.lower().rstrip(".")
    if not any(
        _hostname_matches(hostname, domain) for domain in _PLATFORM_DOMAINS[source_type]
    ):
        raise ValueError(f"the selected source requires a valid {source_type} URL")
    return normalized


def safe_media_url(source_type: str, url: str) -> str:
    """Remove tracking data without dropping YouTube's required video ID."""
    source_type = normalize_source_type(source_type)
    parsed = urlparse(url or "")
    query = ""
    if source_type == SOURCE_YOUTUBE and _hostname_matches(
        (parsed.hostname or "").lower(), "youtube.com"
    ):
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
        if video_id:
            query = urlencode({"v": video_id})
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, "")
    )


def _positive_config_int(key: str, default: int) -> int:
    try:
        value = int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _proxy_url() -> str:
    proxy_cfg = config.proxy if isinstance(config.proxy, dict) else {}
    for key in ("https", "http", "all"):
        value = str(proxy_cfg.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _resolve_hostname_with_dig(hostname: str) -> list[str]:
    hostname = (hostname or "").strip().rstrip(".").lower()
    if not hostname or _is_ip_address(hostname):
        return [hostname] if hostname else []

    cached = _DNS_FALLBACK_CACHE.get(hostname)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            ["/usr/bin/dig", "+short", hostname, "@1.1.1.1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        logger.debug(f"DNS fallback failed for {hostname}: {exc}")
        _DNS_FALLBACK_CACHE[hostname] = []
        return []

    resolved = []
    for line in result.stdout.splitlines():
        candidate = line.strip().rstrip(".")
        if not candidate or not _is_ip_address(candidate):
            continue
        if candidate not in resolved:
            resolved.append(candidate)

    _DNS_FALLBACK_CACHE[hostname] = resolved
    return resolved


@contextmanager
def _dns_fallback_context():
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def _build_getaddrinfo_result(host: str, port, family, type_, proto, ips: list[str]):
        results = []
        for ip in ips:
            try:
                ip_obj = ipaddress.ip_address(ip)
            except ValueError:
                continue

            addr_family = socket.AF_INET6 if ip_obj.version == 6 else socket.AF_INET
            if family not in (0, socket.AF_UNSPEC, addr_family):
                continue

            sockaddr = (ip, port, 0, 0) if addr_family == socket.AF_INET6 else (ip, port)
            socktype = type_ or socket.SOCK_STREAM
            protocol = proto or (socket.IPPROTO_TCP if socktype == socket.SOCK_STREAM else 0)
            results.append((addr_family, socktype, protocol, "", sockaddr))
        return results

    def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return original_getaddrinfo(host, port, family, type, proto, flags)
        except socket.gaierror:
            if not isinstance(host, str):
                raise
            normalized = host.strip().rstrip(".").lower()
            if not normalized or normalized == "localhost" or _is_ip_address(normalized):
                raise
            resolved = _resolve_hostname_with_dig(normalized)
            fallback = _build_getaddrinfo_result(
                normalized, port, family, type, proto, resolved
            )
            if fallback:
                logger.debug(
                    f"resolved {normalized} via dig fallback: {', '.join(resolved)}"
                )
                return fallback
            raise

    def patched_gethostbyname(host):
        try:
            return original_gethostbyname(host)
        except socket.gaierror:
            resolved = _resolve_hostname_with_dig(host)
            if resolved:
                return resolved[0]
            raise

    def patched_gethostbyname_ex(host):
        try:
            return original_gethostbyname_ex(host)
        except socket.gaierror:
            resolved = _resolve_hostname_with_dig(host)
            if resolved:
                return (host, [], resolved)
            raise

    socket.getaddrinfo = patched_getaddrinfo
    socket.gethostbyname = patched_gethostbyname
    socket.gethostbyname_ex = patched_gethostbyname_ex
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo
        socket.gethostbyname = original_gethostbyname
        socket.gethostbyname_ex = original_gethostbyname_ex


def _ytdlp_options(work_dir: Path) -> dict:
    options = {
        "logger": _YTDLPLogger(),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "restrictfilenames": True,
    }
    proxy_url = _proxy_url()
    if proxy_url:
        options["proxy"] = proxy_url

    cookie_file = str(config.app.get("media_source_cookie_file", "") or "").strip()
    if cookie_file:
        if not os.path.isfile(cookie_file):
            raise ValueError("configured media_source_cookie_file does not exist")
        options["cookiefile"] = cookie_file

    if shutil.which("node"):
        options["js_runtimes"] = {"node": {}}
    return options


def _require_ytdlp():
    if YoutubeDL is None:
        raise RuntimeError(
            "YouTube/TikTok import requires yt-dlp. Run 'uv sync --frozen' first."
        )


def _extract_media_info(url: str, work_dir: Path) -> dict:
    _require_ytdlp()
    options = _ytdlp_options(work_dir)
    options["skip_download"] = True
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except DownloadError as exc:
        raise ValueError(f"could not read media metadata: {exc}") from exc

    if not isinstance(info, dict):
        raise ValueError("the media URL did not return usable metadata")
    if info.get("_type") in {"playlist", "multi_video"} or info.get("entries"):
        raise ValueError("playlists are not supported; provide one video URL")
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        raise ValueError("live or upcoming videos are not supported")

    duration = info.get("duration")
    max_duration = (
        _positive_config_int(
            "media_source_max_duration_minutes", DEFAULT_MAX_MEDIA_DURATION_MINUTES
        )
        * 60
    )
    if duration:
        try:
            duration_seconds = float(duration)
        except (TypeError, ValueError):
            duration_seconds = 0
        if duration_seconds > max_duration:
            raise ValueError(
                f"media duration exceeds the configured {max_duration // 60}-minute limit"
            )

    return info


def _preferred_caption_language(info: dict, captions: dict) -> str:
    available = [str(key) for key in captions if str(key).lower() != "live_chat"]
    if not available:
        return ""

    preferred = [
        info.get("original_language"),
        info.get("language"),
        "en",
    ]
    for candidate in preferred:
        normalized = str(candidate or "").lower().replace("_", "-")
        if not normalized:
            continue
        for language in available:
            current = language.lower().replace("_", "-")
            if (
                current == normalized
                or current.split("-", 1)[0] == normalized.split("-", 1)[0]
            ):
                return language
    return available[0]


def _clean_caption_lines(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        value = html.unescape(re.sub(r"<[^>]+>", "", str(line or "")))
        value = re.sub(r"\s+", " ", value).strip()
        if not value or (cleaned and value == cleaned[-1]):
            continue
        cleaned.append(value)
    return " ".join(cleaned).strip()[:MAX_TRANSCRIPT_CHARACTERS]


def parse_caption_text(raw: str | bytes, extension: str = "") -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = str(raw or "").lstrip("\ufeff")
    extension = extension.lower().lstrip(".")

    if extension in {"json", "json3"} or raw.lstrip().startswith("{"):
        try:
            payload = json.loads(raw)
            fragments = []
            for event in payload.get("events", []):
                fragments.extend(
                    segment.get("utf8", "")
                    for segment in event.get("segs", [])
                    if isinstance(segment, dict)
                )
            text = _clean_caption_lines(fragments)
            if text:
                return text
        except (AttributeError, json.JSONDecodeError):
            pass

    lines = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.isdigit()
            or line.startswith(
                ("WEBVTT", "NOTE", "STYLE", "REGION", "Kind:", "Language:")
            )
            or "-->" in line
        ):
            continue
        lines.append(line)
    return _clean_caption_lines(lines)


def _inline_caption(info: dict, caption_group: str, language: str) -> str:
    entries = (info.get(caption_group) or {}).get(language) or []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("data"):
            continue
        text = parse_caption_text(entry["data"], str(entry.get("ext", "")))
        if len(text) >= MIN_TRANSCRIPT_CHARACTERS:
            return text
    return ""


def _download_caption(url: str, work_dir: Path, language: str, automatic: bool) -> str:
    options = _ytdlp_options(work_dir)
    options.update(
        {
            "skip_download": True,
            "writesubtitles": not automatic,
            "writeautomaticsub": automatic,
            "subtitleslangs": [language],
            "subtitlesformat": "json3/vtt/srt/best",
        }
    )
    try:
        with YoutubeDL(options) as downloader:
            downloader.download([url])
    except DownloadError as exc:
        logger.warning(
            f"caption download failed, falling back to speech recognition: {exc}"
        )
        return ""

    for caption_file in sorted(work_dir.iterdir()):
        if caption_file.suffix.lower() not in _CAPTION_EXTENSIONS:
            continue
        text = parse_caption_text(caption_file.read_bytes(), caption_file.suffix)
        if len(text) >= MIN_TRANSCRIPT_CHARACTERS:
            return text
    return ""


def _extract_caption(info: dict, url: str, work_dir: Path) -> str:
    for caption_group, automatic in (
        ("subtitles", False),
        ("automatic_captions", True),
    ):
        captions = info.get(caption_group) or {}
        language = _preferred_caption_language(info, captions)
        if not language:
            continue
        inline_text = _inline_caption(info, caption_group, language)
        if inline_text:
            return inline_text
        downloaded_text = _download_caption(url, work_dir, language, automatic)
        if downloaded_text:
            return downloaded_text
    return ""


def _download_audio(url: str, work_dir: Path) -> Path:
    max_download_mb = _positive_config_int(
        "media_source_max_download_mb", DEFAULT_MAX_MEDIA_DOWNLOAD_MB
    )
    options = _ytdlp_options(work_dir)
    options.update(
        {
            "format": "bestaudio/best",
            "max_filesize": max_download_mb * 1024 * 1024,
            "overwrites": True,
        }
    )
    try:
        with YoutubeDL(options) as downloader:
            downloader.download([url])
    except DownloadError as exc:
        raise ValueError(f"could not download media audio: {exc}") from exc

    ignored_extensions = _CAPTION_EXTENSIONS | {
        ".description",
        ".jpg",
        ".part",
        ".png",
        ".webp",
        ".ytdl",
    }
    candidates = [
        item
        for item in work_dir.iterdir()
        if item.is_file() and item.suffix.lower() not in ignored_extensions
    ]
    if not candidates:
        raise ValueError(
            "the media audio could not be downloaded; it may be private, restricted, or require cookies"
    )
    return max(candidates, key=lambda item: item.stat().st_size)


def _prepare_audio_for_gemini(audio_file: str, work_dir: Path) -> Path:
    source_path = Path(audio_file)
    if source_path.suffix.lower() in _GEMINI_AUDIO_EXTENSIONS:
        return source_path

    converted_path = work_dir / "source-gemini.mp3"
    AudioSegment.converter = utils.get_ffmpeg_binary()
    audio_segment = AudioSegment.from_file(str(source_path))
    audio_segment.export(converted_path, format="mp3", bitrate="128k")
    logger.info(
        f"converted source media audio for Gemini: {source_path.name} -> {converted_path.name}"
    )
    return converted_path


def _configure_gemini():
    api_key = str(config.app.get("gemini_api_key", "") or "").strip()
    if not api_key:
        return None

    import google.generativeai as genai

    base_url = str(config.app.get("gemini_base_url", "") or "").strip()
    if base_url:
        genai.configure(
            api_key=api_key,
            transport="rest",
            client_options={"api_endpoint": base_url},
        )
    else:
        genai.configure(api_key=api_key, transport="rest")
    return genai


def _wait_for_gemini_file(genai, uploaded_file):
    deadline = time.monotonic() + 60
    current = uploaded_file
    while True:
        state = getattr(getattr(current, "state", None), "name", "")
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError(f"Gemini file upload failed: {current}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Gemini file upload timed out")
        time.sleep(1)
        current = genai.get_file(current.name)


def _transcribe_audio_with_gemini(audio_file: str, work_dir: Path) -> str:
    genai = _configure_gemini()
    if genai is None:
        return ""

    uploaded_file = None
    try:
        gemini_audio_file = _prepare_audio_for_gemini(audio_file, work_dir)
        logger.info("transcribing source media audio with Gemini")
        mime_type, _ = mimetypes.guess_type(str(gemini_audio_file))
        upload_kwargs = {}
        if mime_type:
            upload_kwargs["mime_type"] = mime_type
        uploaded_file = genai.upload_file(str(gemini_audio_file), **upload_kwargs)
        uploaded_file = _wait_for_gemini_file(genai, uploaded_file)

        model_name = str(config.app.get("gemini_model_name", "") or "").strip()
        if not model_name:
            model_name = "gemini-2.5-flash"
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            [_GEMINI_TRANSCRIPTION_PROMPT, uploaded_file],
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 8192,
            },
        )
        transcript = getattr(response, "text", "") or ""
        return transcript.strip()
    except Exception as exc:
        logger.warning(f"Gemini transcription failed, falling back to Whisper: {exc}")
        return ""
    finally:
        if uploaded_file is not None:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass


def _fetch_media_content(source_type: str, url: str) -> SourceContent:
    source_url = validate_media_url(source_type, url)
    logger.info(
        f"extracting {source_type} source: {safe_media_url(source_type, source_url)}"
    )
    with tempfile.TemporaryDirectory(prefix="mpt-source-") as temp_dir:
        work_dir = Path(temp_dir)
        with _dns_fallback_context():
            info = _extract_media_info(source_url, work_dir)
            transcript = _extract_caption(info, source_url, work_dir)
            extraction_method = "captions"
            if not transcript:
                logger.info("captions unavailable; downloading audio for transcription")
                audio_file = _download_audio(source_url, work_dir)
                transcript = _transcribe_audio_with_gemini(str(audio_file), work_dir)
                extraction_method = "gemini-speech-to-text" if transcript else "speech-to-text"
                if not transcript:
                    logger.info("Gemini transcription unavailable; using local Whisper")
                    transcript = subtitle.transcribe_text(str(audio_file))

    transcript = (transcript or "").strip()[:MAX_TRANSCRIPT_CHARACTERS]
    if len(transcript) < MIN_TRANSCRIPT_CHARACTERS:
        raise ValueError("the media did not contain enough speech to generate a script")

    final_url = safe_media_url(
        source_type,
        str(info.get("webpage_url") or info.get("original_url") or source_url),
    )
    return SourceContent(
        source_type=source_type,
        source_url=safe_media_url(source_type, source_url),
        final_url=final_url,
        title=str(info.get("title") or "").strip(),
        content=transcript,
        extraction_method=extraction_method,
    )


def fetch_source_content(source_type: str, url: str) -> SourceContent:
    source_type = normalize_source_type(source_type)
    if source_type == SOURCE_ARTICLE:
        result = article.fetch_url_content(url)
        return SourceContent(
            source_type=source_type,
            source_url=result.source_url,
            final_url=article.safe_display_url(result.final_url),
            title=result.title,
            content=result.content,
            extraction_method="article",
        )
    return _fetch_media_content(source_type, url)
