import re
from dataclasses import dataclass


DIALOGUE_SCRIPT_STYLES = {"podcast", "talkshow", "crosstalk"}


@dataclass(frozen=True)
class DialogueSegment:
    speaker: str
    text: str


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DialogueSubtitleTimeline:
    cues: list[SubtitleCue]


def is_dialogue_style(script_style: str | None) -> bool:
    value = getattr(script_style, "value", script_style)
    return str(value or "narration").strip().lower() in DIALOGUE_SCRIPT_STYLES


def parse_dialogue_script(script: str, script_style: str | None = None) -> list[DialogueSegment]:
    script = (script or "").strip()
    if not script:
        return []
    if not is_dialogue_style(script_style):
        return [DialogueSegment(speaker="Narrator", text=script)]

    bracket_pattern = re.compile(
        r"\[(?P<speaker>[^\]\n]{1,40})\]\s*(?P<text>.*?)"
        r"(?=(?:\n\s*)?\[[^\]\n]{1,40}\]\s*|\Z)",
        re.DOTALL,
    )
    segments = []
    for match in bracket_pattern.finditer(script):
        speaker = re.sub(r"\s+", " ", match.group("speaker")).strip()
        text = re.sub(r"\s+", " ", match.group("text")).strip()
        if speaker and text:
            segments.append(DialogueSegment(speaker=speaker, text=text))
    if segments:
        return segments

    colon_pattern = re.compile(r"^([^:\n]{1,40}):\s*(.+)$")
    for line in script.splitlines():
        match = colon_pattern.match(line.strip())
        if not match:
            continue
        speaker = re.sub(r"\s+", " ", match.group(1)).strip()
        text = re.sub(r"\s+", " ", match.group(2)).strip()
        if speaker and text:
            segments.append(DialogueSegment(speaker=speaker, text=text))
    if segments:
        return segments

    return [DialogueSegment(speaker="Speaker 1", text=script)]


def strip_speaker_labels(script: str, script_style: str | None = None) -> str:
    return "\n".join(segment.text for segment in parse_dialogue_script(script, script_style))


def assign_speaker_voices(
    segments: list[DialogueSegment], primary_voice: str, secondary_voice: str
) -> dict[str, str]:
    voices = [primary_voice, secondary_voice or primary_voice]
    result: dict[str, str] = {}
    for segment in segments:
        if segment.speaker not in result:
            result[segment.speaker] = voices[len(result) % len(voices)]
    return result
