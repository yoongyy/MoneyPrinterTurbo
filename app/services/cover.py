import base64
import io
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger
from openai import OpenAI
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils


DEFAULT_COVER_TITLE_MAX_CHARS = 28
DEFAULT_COVER_FONT_SIZE_RATIO = 0.06
DEFAULT_COVER_PANEL_ALPHA = 188
DEFAULT_COVER_PANEL_HEIGHT_RATIO = 0.42
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1"
DEFAULT_OPENROUTER_IMAGE_MODEL = "openai/gpt-image-1"
PREFERRED_OPENROUTER_IMAGE_MODELS = (
    "google/gemini-2.5-flash-image",
    "black-forest-labs/flux.2-klein-4b",
    "openai/gpt-image-1",
    "black-forest-labs/flux.2-pro",
    "google/gemini-2.5-flash-image-preview",
)
AI_COVER_PROVIDER_LOCAL = "local"
AI_COVER_PROVIDER_OPENAI = "openai"
AI_COVER_PROVIDER_OPENROUTER = "openrouter"
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36 MoneyPrinterTurbo/1.3"
    )
}


def clean_cover_title(text: str, max_chars: int = DEFAULT_COVER_TITLE_MAX_CHARS) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    raw_segments = [
        segment.strip()
        for segment in re.split(r"[!！?？,，。；;、~～/|&]+", value)
        if segment.strip()
    ]
    if re.search(r"[\u4e00-\u9fff]", value) and len(raw_segments) > 1:
        selected_segments = []
        total_length = 0
        for segment in raw_segments:
            cleaned_segment = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", segment)
            cleaned_segment = re.sub(r"\s+", " ", cleaned_segment).strip(
                " -_.,:;，。！？、"
            )
            if not cleaned_segment:
                continue
            next_length = total_length + len(cleaned_segment)
            if selected_segments:
                next_length += 1
            if selected_segments and next_length > max_chars:
                break
            selected_segments.append(cleaned_segment)
            total_length = next_length
            if len(selected_segments) >= 3:
                break
        if selected_segments:
            value = " ".join(selected_segments)

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


def _multiline_text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    spacing: int,
    stroke_width: int,
) -> tuple[int, int]:
    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
        align="left",
        stroke_width=stroke_width,
    )
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_wrapped_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    font_name: str | None,
    max_width: int,
    max_height: int,
    initial_size: int,
) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
    min_size = 42
    for font_size in range(initial_size, min_size - 1, -4):
        font = _load_font(font_name, font_size)
        spacing = max(8, int(font_size * 0.18))
        stroke_width = max(2, font_size // 24)
        wrapped = _wrap_text(title, font, max_width)
        if not wrapped:
            continue
        _, text_height = _multiline_text_size(
            draw, wrapped, font, spacing, stroke_width
        )
        if text_height <= max_height:
            return wrapped, font, spacing

    font = _load_font(font_name, min_size)
    return _wrap_text(title, font, max_width), font, max(8, int(min_size * 0.18))


def _compose_cover_background(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    fitted = ImageOps.fit(
        image.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.42),
    )
    blurred = fitted.filter(ImageFilter.GaussianBlur(radius=max(18, width // 34)))
    muted = ImageEnhance.Color(blurred).enhance(0.42)
    muted = ImageEnhance.Contrast(muted).enhance(0.82)
    muted = ImageEnhance.Brightness(muted).enhance(0.62).convert("RGBA")

    shade = Image.new("RGBA", size, (8, 14, 20, 72))
    draw = ImageDraw.Draw(shade)
    for y in range(height):
        alpha = int(20 + 125 * (y / max(1, height - 1)))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    for x in range(width):
        alpha = int(88 * (1 - x / max(1, width - 1)))
        draw.line([(x, 0), (x, height)], fill=(0, 21, 32, alpha))

    return Image.alpha_composite(muted, shade)


def _compose_title_card_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    background = Image.new("RGBA", size, (13, 18, 24, 255))
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for y in range(height):
        ratio = y / max(1, height - 1)
        alpha = int(50 + 120 * ratio)
        draw.line([(0, y), (width, y)], fill=(8, 12, 16, alpha))

    accent_color = (0, 214, 185, 210)
    draw.rounded_rectangle(
        [int(width * 0.08), int(height * 0.14), int(width * 0.92), int(height * 0.88)],
        radius=max(18, width // 40),
        outline=(24, 34, 48, 220),
        width=max(2, width // 240),
    )
    draw.rectangle(
        [int(width * 0.08), int(height * 0.14), int(width * 0.11), int(height * 0.88)],
        fill=accent_color,
    )
    draw.ellipse(
        [int(width * 0.72), int(height * 0.12), int(width * 1.05), int(height * 0.45)],
        fill=(255, 255, 255, 18),
    )
    draw.ellipse(
        [int(width * -0.10), int(height * 0.72), int(width * 0.32), int(height * 1.08)],
        fill=(255, 255, 255, 12),
    )
    return Image.alpha_composite(background, overlay)


def _cover_aspect_ratio(video_aspect: str | VideoAspect) -> str:
    aspect = VideoAspect(getattr(video_aspect, "value", video_aspect))
    if aspect == VideoAspect.landscape:
        return "16:9"
    if aspect == VideoAspect.square:
        return "1:1"
    return "9:16"


def _openai_image_size(video_aspect: str | VideoAspect) -> str:
    aspect_ratio = _cover_aspect_ratio(video_aspect)
    if aspect_ratio == "16:9":
        return "1536x1024"
    if aspect_ratio == "1:1":
        return "1024x1024"
    return "1024x1536"


def _build_openrouter_headers() -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.app.get('openrouter_api_key', '')}",
        "Content-Type": "application/json",
    }
    http_referer = str(config.app.get("openrouter_http_referer", "")).strip()
    app_title = str(config.app.get("openrouter_app_title", "")).strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-OpenRouter-Title"] = app_title
    return headers


def _build_ai_cover_prompt(title: str, video_aspect: str | VideoAspect) -> str:
    aspect_ratio = _cover_aspect_ratio(video_aspect)
    clean_title = clean_cover_title(title, max_chars=42) or str(title or "").strip()
    return (
        "Create a high-impact video thumbnail image for a short video. "
        f"Topic headline: {clean_title}. "
        f"Canvas aspect ratio: {aspect_ratio}. "
        "Style: premium social media thumbnail, photorealistic or highly detailed "
        "editorial image, sharp focus, strong foreground subject, clear scene detail, "
        "strong depth separation, rich colors, cinematic lighting, and a visually "
        "obvious composition related to the topic. "
        "Avoid abstract blur-only art, empty scenery, low-detail compositions, "
        "single-tone gradients, and generic wallpaper-like images. "
        "The image should look finished on its own, not like a background plate. "
        "Do not include any words, letters, numbers, watermarks, captions, UI elements, "
        "logos, brand marks, or misspelled text."
    )


def _save_image_response_data(image_data: dict, output_file: str) -> str:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b64_json = image_data.get("b64_json") if isinstance(image_data, dict) else None
    image_url = image_data.get("url") if isinstance(image_data, dict) else None
    if b64_json:
        raw = base64.b64decode(b64_json)
    elif image_url:
        response = requests.get(image_url, headers=_REQUEST_HEADERS, timeout=(10, 60))
        response.raise_for_status()
        raw = response.content
    else:
        raise ValueError("image response did not contain b64_json or url")

    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(output_path, format="PNG", optimize=True)
    return str(output_path)


def _coerce_sdk_image_data(item) -> dict:
    if isinstance(item, dict):
        return item
    return {
        "b64_json": getattr(item, "b64_json", None),
        "url": getattr(item, "url", None),
    }


def _generate_openai_cover_background(
    prompt: str,
    output_file: str,
    video_aspect: str | VideoAspect,
) -> str:
    api_key = str(config.app.get("openai_api_key", "")).strip()
    if not api_key:
        raise ValueError("openai_api_key is not set")

    kwargs = {"api_key": api_key}
    base_url = str(config.app.get("openai_base_url", "")).strip()
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    model_name = (
        str(config.app.get("openai_image_model_name", "")).strip()
        or DEFAULT_OPENAI_IMAGE_MODEL
    )
    response = client.images.generate(
        model=model_name,
        prompt=prompt,
        size=_openai_image_size(video_aspect),
        quality=str(config.app.get("openai_image_quality", "high")).strip() or "high",
        n=1,
    )
    data = getattr(response, "data", None)
    if not data:
        raise ValueError("OpenAI image generation returned empty data")
    return _save_image_response_data(_coerce_sdk_image_data(data[0]), output_file)


def _post_openrouter_image(url: str, headers: dict[str, str], payload: dict) -> dict:
    response = requests.post(url, headers=headers, json=payload, timeout=(10, 180))
    if response.status_code == 400:
        minimal_payload = {
            "model": payload["model"],
            "prompt": payload["prompt"],
        }
        response = requests.post(
            url, headers=headers, json=minimal_payload, timeout=(10, 180)
        )
    if response.status_code >= 400:
        message = response.text.strip()
        raise ValueError(
            f"OpenRouter image request failed with {response.status_code}: {message}"
        )
    return response.json()


def _generate_openrouter_cover_background(
    prompt: str,
    output_file: str,
    video_aspect: str | VideoAspect,
) -> str:
    api_key = str(config.app.get("openrouter_api_key", "")).strip()
    if not api_key:
        raise ValueError("openrouter_api_key is not set")

    base_url = str(config.app.get("openrouter_base_url", "")).strip()
    if not base_url:
        base_url = "https://openrouter.ai/api/v1"
    configured_model_name = (
        str(config.app.get("openrouter_image_model_name", "")).strip()
        or DEFAULT_OPENROUTER_IMAGE_MODEL
    )
    supported_models = _get_openrouter_image_models(base_url)
    concrete_supported_models = tuple(
        model_id for model_id in supported_models if model_id != "openrouter/auto"
    )
    model_attempts = _build_openrouter_model_attempts(
        configured_model_name, concrete_supported_models
    )
    if not model_attempts:
        example_models = ", ".join(concrete_supported_models[:6])
        raise ValueError(
            "No usable OpenRouter image model is configured for the images API. "
            f"Use a supported model such as: {example_models}"
        )
    url = f"{base_url.rstrip('/')}/images"
    last_error = None
    for model_name in model_attempts:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "aspect_ratio": _cover_aspect_ratio(video_aspect),
            "n": 1,
            "quality": str(config.app.get("openrouter_image_quality", "high")).strip()
            or "high",
            "output_format": "png",
        }
        try:
            result = _post_openrouter_image(url, _build_openrouter_headers(), payload)
            data = result.get("data") if isinstance(result, dict) else None
            if not data:
                raise ValueError("OpenRouter image generation returned empty data")
            return _save_image_response_data(data[0], output_file)
        except Exception as exc:
            last_error = exc
            if model_name != configured_model_name:
                logger.warning(
                    f"OpenRouter image model fallback failed for {model_name}: {exc}"
                )
            continue

    raise ValueError(str(last_error) if last_error else "OpenRouter image generation failed")


def _build_openrouter_model_attempts(
    configured_model_name: str,
    supported_models: tuple[str, ...],
) -> list[str]:
    attempts: list[str] = []
    seen = set()

    def add(model_name: str):
        model_name = str(model_name or "").strip()
        if not model_name or model_name in seen:
            return
        seen.add(model_name)
        attempts.append(model_name)

    if configured_model_name and configured_model_name != "openrouter/auto":
        if supported_models:
            if configured_model_name in supported_models:
                add(configured_model_name)
            else:
                logger.warning(
                    f'configured OpenRouter image model "{configured_model_name}" '
                    "is not available on the images API; using concrete fallbacks"
                )
        else:
            add(configured_model_name)

    if supported_models:
        for model_name in PREFERRED_OPENROUTER_IMAGE_MODELS:
            if model_name in supported_models:
                add(model_name)
        for model_name in supported_models:
            add(model_name)

    return attempts


def _generate_ai_cover_background(
    provider: str,
    prompt: str,
    output_file: str,
    video_aspect: str | VideoAspect,
) -> str:
    provider = str(provider or "").strip().lower()
    if provider == AI_COVER_PROVIDER_OPENAI:
        return _generate_openai_cover_background(prompt, output_file, video_aspect)
    if provider == AI_COVER_PROVIDER_OPENROUTER:
        return _generate_openrouter_cover_background(prompt, output_file, video_aspect)
    raise ValueError(f"unsupported AI cover provider: {provider}")


@lru_cache(maxsize=4)
def _get_openrouter_image_models(base_url: str) -> tuple[str, ...]:
    url = f"{base_url.rstrip('/')}/images/models"
    try:
        response = requests.get(
            url,
            headers=_build_openrouter_headers(),
            timeout=(10, 30),
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return tuple(
            str(item.get("id", "")).strip()
            for item in data
            if str(item.get("id", "")).strip()
        )
    except Exception as exc:
        logger.warning(f"failed to fetch OpenRouter image models: {exc}")
        return ()


def _split_cover_headline(title: str, max_lines: int = 3) -> list[str]:
    words = [word for word in str(title or "").split(" ") if word]
    if not words:
        return []
    if len(words) <= max_lines:
        return words

    lines = words[: max_lines - 1]
    lines.append(" ".join(words[max_lines - 1 :]))
    return lines


def _fit_headline_font(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font_name: str | None,
    max_width: int,
    max_height: int,
    initial_size: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
    min_size = 44
    for font_size in range(initial_size, min_size - 1, -4):
        font = _load_font(font_name, font_size)
        spacing = max(8, int(font_size * 0.12))
        stroke_width = max(4, font_size // 12)
        wrapped = "\n".join(lines)
        text_width, text_height = _multiline_text_size(
            draw, wrapped, font, spacing, stroke_width
        )
        if text_width <= max_width and text_height <= max_height:
            return font, spacing, stroke_width

    font = _load_font(font_name, min_size)
    return font, max(8, int(min_size * 0.12)), max(4, min_size // 12)


def _compose_ai_thumbnail_cover(
    background_image_ref: str,
    title: str,
    output_file: str,
    video_aspect: str | VideoAspect,
    font_name: str | None = None,
) -> str:
    aspect = VideoAspect(getattr(video_aspect, "value", video_aspect))
    width, height = aspect.to_resolution()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_source_image(background_image_ref) as image:
        background = ImageOps.fit(
            image.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    background = ImageEnhance.Color(background).enhance(1.12)
    background = ImageEnhance.Contrast(background).enhance(1.12).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for y in range(height):
        alpha = int(20 + 145 * (y / max(1, height - 1)))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    draw.rectangle([0, 0, width, int(height * 0.18)], fill=(0, 0, 0, 92))

    clean_title = clean_cover_title(title, max_chars=42) or str(title or "").strip()
    lines = _split_cover_headline(clean_title, max_lines=3)
    if not lines:
        return ""

    initial_size = int(height * (0.12 if aspect == VideoAspect.landscape else 0.075))
    font, spacing, stroke_width = _fit_headline_font(
        draw,
        lines,
        font_name,
        int(width * 0.88),
        int(height * 0.42),
        max(48, initial_size),
    )

    line_boxes = [
        draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        for line in lines
    ]
    line_heights = [box[3] - box[1] for box in line_boxes]
    total_height = sum(line_heights) + spacing * (len(lines) - 1)
    y = int(height * (0.57 if aspect == VideoAspect.landscape else 0.50))
    y = min(y, int(height * 0.91) - total_height)
    y = max(int(height * 0.12), y)

    current_y = y
    for index, line in enumerate(lines):
        bbox = line_boxes[index]
        line_width = bbox[2] - bbox[0]
        x = int((width - line_width) / 2)
        fill = (
            (255, 219, 40, 255)
            if index == len(lines) - 1
            else (255, 255, 255, 255)
        )
        draw.text(
            (x, current_y),
            line,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 230),
        )
        current_y += line_heights[index] + spacing

    cover = Image.alpha_composite(background, overlay).convert("RGB")
    cover.save(output_path, format="JPEG", quality=94, optimize=True)
    logger.info(f"generated AI video cover: {output_path}")
    return str(output_path)


def generate_title_card_cover(
    title: str,
    output_file: str,
    video_aspect: str | VideoAspect = VideoAspect.portrait.value,
    font_name: str | None = None,
) -> str:
    aspect = VideoAspect(getattr(video_aspect, "value", video_aspect))
    width, height = aspect.to_resolution()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    background = _compose_title_card_background((width, height)).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    clean_title = clean_cover_title(title, max_chars=42) or str(title or "").strip()
    lines = _split_cover_headline(clean_title, max_lines=3)
    if not lines:
        return ""

    initial_size = int(height * (0.12 if aspect == VideoAspect.landscape else 0.08))
    font, spacing, stroke_width = _fit_headline_font(
        draw,
        lines,
        font_name,
        int(width * 0.78),
        int(height * 0.56),
        max(48, initial_size),
    )

    line_boxes = [
        draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        for line in lines
    ]
    line_heights = [box[3] - box[1] for box in line_boxes]
    total_height = sum(line_heights) + spacing * (len(lines) - 1)
    current_y = int((height - total_height) / 2)
    current_y = max(int(height * 0.18), current_y)

    for index, line in enumerate(lines):
        bbox = line_boxes[index]
        line_width = bbox[2] - bbox[0]
        x = int((width - line_width) / 2)
        draw.text(
            (x, current_y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 220),
        )
        current_y += line_heights[index] + spacing

    cover = Image.alpha_composite(background, overlay).convert("RGB")
    cover.save(output_path, format="JPEG", quality=94, optimize=True)
    logger.info(f"generated fallback title card cover: {output_path}")
    return str(output_path)


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
            background = _compose_cover_background(image, (width, height))

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        clean_title = clean_cover_title(title)
        if not clean_title:
            clean_title = clean_cover_title(source_image_ref)

        font_size = max(44, int(height * DEFAULT_COVER_FONT_SIZE_RATIO))
        max_text_width = int(width * 0.78)
        max_text_height = int(height * DEFAULT_COVER_PANEL_HEIGHT_RATIO)
        wrapped_title, font, line_spacing = _fit_wrapped_title(
            draw,
            clean_title,
            font_name,
            max_text_width,
            max_text_height,
            font_size,
        )
        if not wrapped_title:
            wrapped_title = clean_title[:DEFAULT_COVER_TITLE_MAX_CHARS]

        actual_font_size = getattr(font, "size", font_size)
        stroke_width = max(2, actual_font_size // 24)
        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped_title,
            font=font,
            spacing=line_spacing,
            align="left",
            stroke_width=stroke_width,
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = int(width * 0.11)
        y = int(height * 0.47)
        if y + text_height > int(height * 0.86):
            y = int(height * 0.86) - text_height
        y = max(int(height * 0.30), y)

        panel_padding_x = max(44, int(width * 0.045))
        panel_padding_y = max(42, int(height * 0.024))
        panel_left = max(0, x - panel_padding_x)
        panel_top = max(0, y - panel_padding_y)
        panel_right = min(width, x + text_width + panel_padding_x)
        panel_bottom = min(height, y + text_height + panel_padding_y)

        draw.rounded_rectangle(
            [panel_left, panel_top, panel_right, panel_bottom],
            radius=max(18, width // 36),
            fill=(2, 10, 16, DEFAULT_COVER_PANEL_ALPHA),
        )
        accent_width = max(10, width // 72)
        draw.rounded_rectangle(
            [panel_left, panel_top, panel_left + accent_width, panel_bottom],
            radius=max(8, accent_width // 2),
            fill=(0, 214, 185, 235),
        )

        draw.multiline_text(
            (x, y),
            wrapped_title,
            font=font,
            fill=(255, 255, 255, 255),
            spacing=line_spacing,
            align="left",
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 205),
        )

        cover = Image.alpha_composite(background, overlay).convert("RGB")
        cover.save(output_path, format="JPEG", quality=94, optimize=True)
        logger.info(f"generated video cover: {output_path}")
        return str(output_path)
    except Exception as exc:
        logger.warning(f"failed to generate video cover from source image: {exc}")
        return ""


def generate_ai_video_cover(
    title: str,
    output_file: str,
    video_aspect: str | VideoAspect = VideoAspect.portrait.value,
    font_name: str | None = None,
    provider: str | None = None,
) -> str:
    provider = (
        str(provider or config.app.get("cover_generation_provider", "local"))
        .strip()
        .lower()
    )
    if provider in {"", AI_COVER_PROVIDER_LOCAL}:
        return ""

    output_path = Path(output_file)
    background_path = output_path.with_name(f"{output_path.stem}.ai-background.png")
    prompt = _build_ai_cover_prompt(title, video_aspect)
    generated_background = _generate_ai_cover_background(
        provider=provider,
        prompt=prompt,
        output_file=str(background_path),
        video_aspect=video_aspect,
    )
    return _compose_ai_thumbnail_cover(
        background_image_ref=generated_background,
        title=title,
        output_file=output_file,
        video_aspect=video_aspect,
        font_name=font_name,
    )
