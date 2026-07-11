import math
import os.path
import re
from os import path

from loguru import logger
from pydub import AudioSegment

from app.config import config
from app.models import const
from app.models.schema import VideoAspect, VideoConcatMode, VideoParams
from app.services import (
    content_source,
    cover,
    dialogue,
    llm,
    material,
    subtitle,
    video,
    voice,
    upload_post,
)
from app.services import state as sm
from app.utils import file_security, utils


_WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_TARGET_AUDIO_DURATION_OVERAGE_RATIO = 1.15
_TARGET_AUDIO_DURATION_OVERAGE_SECONDS = 15
_MAX_SCRIPT_AUDIO_REWRITE_ATTEMPTS = 2


def subject_video_filename(subject: str, index: int = 1, total: int = 1) -> str:
    """Return a portable final-video filename based on the video subject."""
    stem = re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]+', "-", str(subject or ""))
    stem = re.sub(r"\s+", " ", stem).strip(" .-")
    if not stem:
        stem = "video"
    if stem.split(".", 1)[0].upper() in _WINDOWS_RESERVED_FILENAMES:
        stem = f"_{stem}"

    sequence = f"-{index}" if total > 1 else ""
    max_stem_bytes = 240 - len(sequence.encode("utf-8"))
    stem = stem.encode("utf-8")[:max_stem_bytes].decode("utf-8", errors="ignore")
    stem = stem.rstrip(" .-") or "video"
    return f"{stem}{sequence}.mp4"


def _target_audio_duration_seconds(params: VideoParams) -> int:
    try:
        minutes = int(getattr(params, "video_duration_minutes", 1) or 1)
    except (TypeError, ValueError):
        minutes = 1
    return max(1, minutes) * 60


def _max_audio_duration_seconds(params: VideoParams) -> float:
    target_seconds = _target_audio_duration_seconds(params)
    return max(
        target_seconds * _TARGET_AUDIO_DURATION_OVERAGE_RATIO,
        target_seconds + _TARGET_AUDIO_DURATION_OVERAGE_SECONDS,
    )


def _audio_exceeds_target(params: VideoParams, audio_duration: float | int | None) -> bool:
    if not audio_duration:
        return False
    if getattr(params, "custom_audio_file", None):
        return False
    return float(audio_duration) > _max_audio_duration_seconds(params)


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    source_content = ""
    source_title = getattr(params, "source_title", "") or ""
    source_url = (getattr(params, "source_url", "") or "").strip()
    source_cover_image_url = getattr(params, "source_cover_image_url", "") or ""

    resolved_source = None
    if source_url:
        source_type = getattr(params, "source_type", "article")
        if not video_script:
            try:
                resolved_source = content_source.fetch_source_content(
                    source_type, source_url
                )
            except Exception as exc:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                logger.error(f"failed to extract source content: {str(exc)}")
                return None
            source_content = resolved_source.content
        elif not source_title or not source_cover_image_url or not params.video_subject:
            try:
                resolved_source = content_source.fetch_source_metadata(
                    source_type, source_url
                )
            except Exception as exc:
                logger.warning(f"failed to resolve source metadata: {str(exc)}")

        if resolved_source:
            source_title = resolved_source.title or source_title
            source_url = resolved_source.final_url or source_url
            source_cover_image_url = (
                resolved_source.cover_image_url or source_cover_image_url
            )
            params.source_url = source_url
            params.source_title = source_title
            params.source_cover_image_url = source_cover_image_url
            if not params.video_subject:
                params.video_subject = source_title or source_url

    if not video_script:
        generate_kwargs = dict(
            video_subject=params.video_subject,
            language=params.video_language,
            video_duration_minutes=getattr(params, "video_duration_minutes", 1),
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
        if source_content or dialogue.is_dialogue_style(
            getattr(params, "script_style", "narration")
        ):
            generate_kwargs.update(
                script_style=getattr(params, "script_style", "narration"),
                source_content=source_content,
                source_title=source_title,
                source_url=source_url,
            )
        video_script = llm.generate_script(**generate_kwargs)
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        searchable_script = dialogue.strip_speaker_labels(
            video_script, getattr(params, "script_style", "narration")
        )
        # 开启素材按文案顺序匹配后，关键词本身也必须按脚本叙事顺序生成；
        # 否则后续即使顺序下载和顺序拼接，也只能复用一组全局主题词，
        # 无法改善“后面内容的画面提前出现”的问题。
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=searchable_script,
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def resolve_custom_audio_file(task_id: str, custom_audio_file: str | None) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _parse_srt_timestamp(value: str) -> float:
    hours, minutes, seconds = value.strip().replace(".", ",").split(":")
    whole_seconds, milliseconds = seconds.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(whole_seconds)
        + int(milliseconds) / 1000
    )


def _format_srt_timestamp(value: float) -> str:
    total_milliseconds = max(0, round(value * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _write_dialogue_subtitle(
    timeline: dialogue.DialogueSubtitleTimeline, subtitle_path: str
):
    items = []
    for index, cue in enumerate(timeline.cues, start=1):
        items.append(
            f"{index}\n{_format_srt_timestamp(cue.start)} --> "
            f"{_format_srt_timestamp(cue.end)}\n{cue.text}\n"
        )
    with open(subtitle_path, "w", encoding="utf-8") as file:
        file.write("\n".join(items))


def _segment_subtitle_cues(
    sub_maker, segment: dialogue.DialogueSegment, segment_file: str, duration: float
) -> list[dialogue.SubtitleCue]:
    segment_srt = f"{segment_file}.srt"
    try:
        voice.create_subtitle(
            text=segment.text, sub_maker=sub_maker, subtitle_file=segment_srt
        )
        cues = []
        for _, timing, text in subtitle.file_to_subtitles(segment_srt):
            start_text, end_text = [item.strip() for item in timing.split("-->", 1)]
            cues.append(
                dialogue.SubtitleCue(
                    start=_parse_srt_timestamp(start_text),
                    end=_parse_srt_timestamp(end_text),
                    text=f"{segment.speaker}: {text}",
                )
            )
        if cues:
            return cues
    except Exception as exc:
        logger.warning(f"failed to build detailed dialogue subtitle cue: {str(exc)}")
    finally:
        if os.path.exists(segment_srt):
            os.remove(segment_srt)

    return [
        dialogue.SubtitleCue(
            start=0.0,
            end=max(0.1, duration),
            text=f"{segment.speaker}: {segment.text}",
        )
    ]


def _generate_dialogue_audio(task_id, params, video_script):
    segments = dialogue.parse_dialogue_script(
        video_script, getattr(params, "script_style", "narration")
    )
    if not segments:
        return None, None, None

    primary_voice = voice.parse_voice_name(params.voice_name)
    secondary_voice = voice.parse_voice_name(
        getattr(params, "voice_name_2", "") or params.voice_name
    )
    speaker_voices = dialogue.assign_speaker_voices(
        segments, primary_voice, secondary_voice
    )
    pause_seconds = float(getattr(params, "dialogue_pause", 0.2) or 0.0)
    pause_milliseconds = round(max(0.0, min(2.0, pause_seconds)) * 1000)

    AudioSegment.converter = utils.get_ffmpeg_binary()
    combined_audio = AudioSegment.empty()
    timeline_cues: list[dialogue.SubtitleCue] = []
    segment_files = []

    try:
        for index, segment in enumerate(segments, start=1):
            segment_file = path.join(
                utils.task_dir(task_id), f"dialogue-{index:03}.mp3"
            )
            segment_files.append(segment_file)
            segment_sub_maker = voice.tts(
                text=segment.text,
                voice_name=speaker_voices[segment.speaker],
                voice_rate=params.voice_rate,
                voice_file=segment_file,
            )
            if segment_sub_maker is None or not os.path.exists(segment_file):
                raise ValueError(
                    f"failed to synthesize dialogue audio for speaker {segment.speaker}"
                )

            segment_audio = AudioSegment.from_file(segment_file)
            offset_seconds = len(combined_audio) / 1000
            segment_duration = len(segment_audio) / 1000
            for cue in _segment_subtitle_cues(
                segment_sub_maker, segment, segment_file, segment_duration
            ):
                timeline_cues.append(
                    dialogue.SubtitleCue(
                        start=offset_seconds + cue.start,
                        end=offset_seconds + cue.end,
                        text=cue.text,
                    )
                )
            combined_audio += segment_audio
            if index < len(segments) and pause_milliseconds:
                combined_audio += AudioSegment.silent(duration=pause_milliseconds)

        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        exported_audio = combined_audio.export(audio_file, format="mp3", bitrate="192k")
        exported_audio.close()
        audio_duration = len(combined_audio) / 1000
        return (
            audio_file,
            audio_duration,
            dialogue.DialogueSubtitleTimeline(cues=timeline_cues),
        )
    except Exception as exc:
        logger.error(f"failed to generate dialogue audio: {str(exc)}")
        return None, None, None
    finally:
        for segment_file in segment_files:
            if os.path.exists(segment_file):
                os.remove(segment_file)


def generate_audio(task_id, params, video_script):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # /audio 和 /subtitle 请求模型不包含 custom_audio_file，
    # 这里统一做兼容读取，避免直调接口时抛属性错误。
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id, requested_custom_audio_file
        )
    except ValueError as exc:
        logger.error(
            "custom audio file is invalid, "
            f"task_id: {task_id}, path: {requested_custom_audio_file}, error: {str(exc)}"
        )
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return None, None, None

    if not custom_audio_file:
        logger.info("no custom audio file provided, using TTS to generate audio.")
        if dialogue.is_dialogue_style(getattr(params, "script_style", "narration")):
            return _generate_dialogue_audio(task_id, params, video_script)

        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
            """.strip()
            )
            return None, None, None
        audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration.")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration from custom audio file.")
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled or sub_maker is None:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    subtitle_fallback = False
    if subtitle_provider == "edge":
        if isinstance(sub_maker, dialogue.DialogueSubtitleTimeline):
            _write_dialogue_subtitle(sub_maker, subtitle_path)
        else:
            voice.create_subtitle(
                text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
            )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(
            subtitle_file=subtitle_path,
            video_script=dialogue.strip_speaker_labels(
                video_script, getattr(params, "script_style", "narration")
            ),
        )

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def get_video_materials(task_id, params, video_terms, audio_duration):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        # 顺序匹配模式只在用户显式开启时生效。这里强制素材下载按关键词顺序
        # 轮询，避免某个早期关键词下载太多素材，把后续脚本主题挤出最终时间线。
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_concat_mode=(
                VideoConcatMode.sequential
                if params.match_materials_to_script
                else params.video_concat_mode
            ),
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
            match_script_order=params.match_materials_to_script,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path
):
    final_video_paths = []
    combined_video_paths = []
    # 多视频生成默认会打散素材以增加差异；但“按文案顺序匹配素材”追求的是
    # 时间线稳定性和可解释性，所以开启后所有输出都使用顺序拼接。
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        combine_video_aspect = (
            VideoAspect.landscape
            if getattr(params, "video_layout_style", "full") == "news_card"
            else params.video_aspect
        )
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=combine_video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(
            utils.task_dir(task_id),
            subject_video_filename(
                params.video_subject,
                index=index,
                total=params.video_count,
            ),
        )

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate audio first, then enforce the requested target duration against
    # the real TTS duration. LLMs can over-generate even when prompted for
    # "3 minutes"; without this guard, ffmpeg will build a video as long as the
    # oversized voiceover.
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)
    audio_file, audio_duration, sub_maker = None, None, None
    for rewrite_attempt in range(_MAX_SCRIPT_AUDIO_REWRITE_ATTEMPTS + 1):
        audio_file, audio_duration, sub_maker = generate_audio(
            task_id, params, video_script
        )
        if not audio_file:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

        if not _audio_exceeds_target(params, audio_duration):
            break

        if rewrite_attempt >= _MAX_SCRIPT_AUDIO_REWRITE_ATTEMPTS:
            logger.warning(
                "generated audio is still longer than the target duration after "
                f"{_MAX_SCRIPT_AUDIO_REWRITE_ATTEMPTS} rewrite attempt(s): "
                f"audio={audio_duration:.2f}s, target={_target_audio_duration_seconds(params)}s"
            )
            break

        logger.warning(
            "generated audio exceeds target duration; shortening script and "
            f"regenerating audio. audio={audio_duration:.2f}s, "
            f"max_allowed={_max_audio_duration_seconds(params):.2f}s"
        )
        shortened_script = llm.shorten_script_to_duration(
            video_script=video_script,
            video_duration_minutes=getattr(params, "video_duration_minutes", 1),
            script_style=getattr(params, "script_style", "narration"),
        )
        if not shortened_script or shortened_script.strip() == video_script.strip():
            logger.warning("script shortening did not produce a shorter replacement")
            break
        video_script = shortened_script.strip()

    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 3. Generate terms after any duration rewrite so the downloaded visuals
    # match the final script instead of an oversized draft.
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 仅完整视频生成流程才需要处理视频拼接模式；
    # 这样可以避免 /subtitle 和 /audio 这类请求访问不存在的字段。
    if isinstance(params.video_concat_mode, str):
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    cover_image_path = ""
    cover_error = ""
    cover_title = cover.clean_cover_title(
        params.video_subject or getattr(params, "source_title", "")
    )
    if getattr(params, "generate_video_cover", True):
        cover_provider = (
            getattr(params, "cover_generation_provider", "")
            or config.app.get("cover_generation_provider", "local")
            or "local"
        )
        cover_output_file = path.join(utils.task_dir(task_id), "cover.jpg")
        cover_title_text = cover_title or params.video_subject
        cover_source_image = getattr(params, "source_cover_image_url", "") or ""
        if cover_provider and cover_provider != cover.AI_COVER_PROVIDER_LOCAL:
            try:
                cover_image_path = cover.generate_ai_video_cover(
                    title=cover_title_text,
                    output_file=cover_output_file,
                    video_aspect=params.video_aspect,
                    font_name=params.font_name,
                    provider=cover_provider,
                )
            except Exception as exc:
                cover_error = str(exc)
                logger.warning(
                    f"failed to generate AI cover via {cover_provider}: {cover_error}"
                )
        if not cover_image_path:
            if not cover_source_image:
                cover_source_image = material.find_stock_cover_image(
                    task_id=task_id,
                    search_term=cover_title_text or getattr(params, "source_url", ""),
                    source=getattr(params, "video_source", "pexels") or "pexels",
                    video_aspect=params.video_aspect,
                )
            if cover_source_image:
                cover_image_path = cover.generate_video_cover(
                    source_image_ref=cover_source_image,
                    title=cover_title_text,
                    output_file=cover_output_file,
                    video_aspect=params.video_aspect,
                    font_name=params.font_name,
                )
        if not cover_image_path:
            logger.warning("no usable cover image available; skipping generated cover")

    # 7. Cross-post to social platforms (if enabled)
    cross_post_results = []
    if (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    ):
        platforms = upload_post.upload_post_service.platforms
        logger.info(f"\n\n## cross-posting videos to {', '.join(platforms)}")

        youtube_extra = None
        if "youtube" in platforms:
            metadata = llm.generate_social_metadata(
                video_subject=params.video_subject,
                video_script=video_script,
                language=params.video_language or "",
                platform="youtube_shorts",
            )
            youtube_extra = {
                "youtube_title": metadata.get("title", params.video_subject),
                "youtube_description": metadata.get("caption", ""),
                "tags": metadata.get("hashtags", []),
                "privacyStatus": upload_post.upload_post_service.youtube_privacy_status,
                "containsSyntheticMedia": True,
                "thumbnail_path": cover_image_path,
                "tiktok_cover_timestamp": 1000,
            }

        upload_title = cover_title or params.video_subject or "Check out this video! #shorts #viral"
        is_short_form = str(getattr(params.video_aspect, "value", params.video_aspect)) == VideoAspect.portrait.value

        for video_path in final_video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=upload_title,
                youtube_extra=youtube_extra,
                cover_image_path=cover_image_path,
                is_short_form=is_short_form,
            )
            cross_post_results.append(result)
            if result.get("success"):
                logger.info(f"✅ Cross-posted: {video_path}")
            else:
                logger.warning(
                    f"⚠️ Failed to cross-post: {video_path} - {result.get('error', 'Unknown error')}"
                )

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cover_image": cover_image_path or None,
        "cover_error": cover_error or None,
        "cross_post_results": cross_post_results if cross_post_results else None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
