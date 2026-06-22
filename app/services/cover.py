import io
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.models.schema import VideoAspect
from app.utils import utils


DEFAULT_COVER_TITLE_MAX_CHARS = 28
DEFAULT_COVER_FONT_SIZE_RATIO = 0.048
DEFAULT_COVER_PANEL_ALPHA = 170
DEFAULT_COVER_PANEL_HEIGHT_RATIO = 0.36
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36 MoneyPrinterTurbo/1.3"
    )
}


def clean_cover_title(text: str, max_chars: int = DEFAULT_COVER_TITLE_MAX_CHARS) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -_.,:;，。！？、")
    if not value:
        return ""

    if " " in value:
        selected_words = []
        for word in value.split():
            candidate = " ".join((*selected_words, word))
            if len(candidate) > max_chars:
                break
            selected_words.append(word)
        value = " ".join(selected_words) or value

    value = value[:max_chars].strip(" -_.,:;，。！？、")
    return value


def _resolve_font_path(font_name: str | None) -> str:
    candidates = []
    if font_name:
        candidates.append(str(font_name))
    candidates.extend(
        [
            "MicrosoftYaHeiBold.ttf",
            "MicrosoftYaHeiBold.ttc",
            "STHeitiMedium.ttc",
            "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        font_path = Path(utils.font_dir()) / candidate
        if font_path.exists():
            return str(font_path)
    return ""


def _load_font(font_name: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _resolve_font_path(font_name)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as exc:
            logger.debug(f"failed to load cover font {font_path}: {exc}")
    return ImageFont.load_default()


def _text_size(font, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text or "")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_text(text: str, font, max_width: int) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    if _text_size(font, text)[0] <= max_width:
        return text

    def split_long_token(token: str) -> list[str]:
        pieces = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            if current and _text_size(font, candidate)[0] > max_width:
                pieces.append(current)
                current = char
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    lines: list[str] = []
    current = ""
    tokens = text.split(" ") if " " in text else [text]
    for token in tokens:
        candidate = f"{current} {token}".strip() if current else token
        if candidate and _text_size(font, candidate)[0] <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        if _text_size(font, token)[0] <= max_width:
            current = token
        else:
            split_tokens = split_long_token(token)
            if split_tokens:
                lines.extend(split_tokens[:-1])
                current = split_tokens[-1]
            else:
                current = ""

    if current:
        lines.append(current)

    return "\n".join(line.strip() for line in lines if line.strip()).strip()


def _open_source_image(source_image_ref: str) -> Image.Image:
    source_image_ref = str(source_image_ref or "").strip()
    if not source_image_ref:
        raise ValueError("source image is empty")

    parsed = urlparse(source_image_ref)
    if parsed.scheme in {"http", "https"}:
        response = requests.get(
            source_image_ref,
            headers=_REQUEST_HEADERS,
            timeout=(10, 30),
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    path = Path(source_image_ref)
    if not path.exists():
        raise FileNotFoundError(f"source image not found: {source_image_ref}")
    return Image.open(path)


def generate_video_cover(
    source_image_ref: str,
    title: str,
    output_file: str,
    video_aspect: str | VideoAspect = VideoAspect.portrait.value,
    font_name: str | None = None,
) -> str:
    source_image_ref = str(source_image_ref or "").strip()
    if not source_image_ref:
        return ""

    aspect = VideoAspect(getattr(video_aspect, "value", video_aspect))
    width, height = aspect.to_resolution()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with _open_source_image(source_image_ref) as image:
            background = ImageOps.fit(
                image.convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.35),
            ).convert("RGBA")

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        panel_top = int(height * (1 - DEFAULT_COVER_PANEL_HEIGHT_RATIO))
        draw.rectangle(
            [0, panel_top, width, height],
            fill=(0, 0, 0, DEFAULT_COVER_PANEL_ALPHA),
        )
        draw.rectangle(
            [0, panel_top - 6, width, panel_top],
            fill=(255, 255, 255, 24),
        )

        clean_title = clean_cover_title(title)
        if not clean_title:
            clean_title = clean_cover_title(source_image_ref)

        font_size = max(44, int(height * DEFAULT_COVER_FONT_SIZE_RATIO))
        font = _load_font(font_name, font_size)
        max_text_width = int(width * 0.84)
        wrapped_title = _wrap_text(clean_title, font, max_text_width)
        if not wrapped_title:
            wrapped_title = clean_title[:DEFAULT_COVER_TITLE_MAX_CHARS]

        line_spacing = max(8, int(font_size * 0.25))
        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped_title,
            font=font,
            spacing=line_spacing,
            align="left",
            stroke_width=max(2, font_size // 22),
        )
        text_height = bbox[3] - bbox[1]
        x = int(width * 0.08)
        y = panel_top + max(24, (height - panel_top - text_height) // 2)

        draw.multiline_text(
            (x, y),
            wrapped_title,
            font=font,
            fill=(255, 255, 255, 255),
            spacing=line_spacing,
            align="left",
            stroke_width=max(2, font_size // 22),
            stroke_fill=(0, 0, 0, 190),
        )

        cover = Image.alpha_composite(background, overlay).convert("RGB")
        cover.save(output_path, format="JPEG", quality=94, optimize=True)
        logger.info(f"generated video cover: {output_path}")
        return str(output_path)
    except Exception as exc:
        logger.warning(f"failed to generate video cover from source image: {exc}")
        return ""
