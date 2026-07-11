import unittest
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestTaskService(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_generate_script_forwards_advanced_prompt_options(self):
        """
        任务生成入口和 WebUI/API 共用 VideoParams。这里验证自动生成文案时，
        高级提示词参数会继续传到 LLM 服务层，避免只在 /scripts 接口生效。
        """
        params = VideoParams(
            video_subject="咖啡",
            video_script="",
            video_language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

        with patch.object(
            tm.llm, "generate_script", return_value="生成的文案"
        ) as generate:
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "生成的文案")
        generate.assert_called_once_with(
            video_subject="咖啡",
            language="zh-CN",
            video_duration_minutes=1,
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

    def test_subject_video_filename(self):
        self.assertEqual(
            tm.subject_video_filename("My Video Subject"), "My Video Subject.mp4"
        )
        self.assertEqual(
            tm.subject_video_filename("News: AI / 2026", index=2, total=3),
            "News- AI - 2026-2.mp4",
        )
        self.assertEqual(tm.subject_video_filename("../"), "video.mp4")
        self.assertEqual(tm.subject_video_filename("CON"), "_CON.mp4")

    def test_generate_final_video_uses_subject_filename(self):
        params = VideoParams(video_subject="Example Subject", video_count=1)
        with (
            patch.object(tm.utils, "task_dir", return_value="/tmp/task-id"),
            patch.object(tm.video, "combine_videos"),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _ = tm.generate_final_videos(
                "task-id", params, ["clip.mp4"], "audio.mp3", "subtitle.srt"
            )

        expected = "/tmp/task-id/Example Subject.mp4"
        self.assertEqual(final_paths, [expected])
        self.assertEqual(generate_video.call_args.kwargs["output_file"], expected)

    def test_cover_generation_prefers_edited_video_subject_over_source_title(self):
        params = VideoParams(
            video_subject="Edited Video Subject",
            source_title="Captured URL Title",
            source_cover_image_url="https://example.com/cover.jpg",
            video_script="ready script",
            video_terms=["term"],
            video_count=1,
            generate_video_cover=True,
        )

        with tempfile.TemporaryDirectory() as task_dir:
            with (
                patch.object(tm.utils, "task_dir", return_value=task_dir),
                patch.object(tm, "generate_script", return_value="ready script"),
                patch.object(tm, "generate_terms", return_value=["term"]),
                patch.object(
                    tm, "generate_audio", return_value=("audio.mp3", 1.0, object())
                ),
                patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
                patch.object(tm, "get_video_materials", return_value=["material.mp4"]),
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"]),
                ),
                patch.object(tm.cover, "generate_ai_video_cover", return_value=""),
                patch.object(
                    tm.cover,
                    "generate_video_cover",
                    return_value=os.path.join(task_dir, "cover.jpg"),
                ) as generate_cover,
                patch.object(
                    tm.upload_post.upload_post_service,
                    "is_configured",
                    return_value=False,
                ),
                patch.object(tm.sm.state, "update_task"),
            ):
                tm.start("task-id", params)

        self.assertEqual(generate_cover.call_args.kwargs["title"], "Edited Video Subject")

    def test_ai_cover_failure_uses_source_image_fallback_and_skips_title_card(self):
        params = VideoParams(
            video_subject="Edited Video Subject",
            source_title="Captured URL Title",
            source_cover_image_url="https://example.com/cover.jpg",
            video_script="ready script",
            video_terms=["term"],
            video_count=1,
            generate_video_cover=True,
            cover_generation_provider="openrouter",
        )

        with tempfile.TemporaryDirectory() as task_dir:
            with (
                patch.object(tm.utils, "task_dir", return_value=task_dir),
                patch.object(tm, "generate_script", return_value="ready script"),
                patch.object(tm, "generate_terms", return_value=["term"]),
                patch.object(
                    tm, "generate_audio", return_value=("audio.mp3", 1.0, object())
                ),
                patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
                patch.object(tm, "get_video_materials", return_value=["material.mp4"]),
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"]),
                ),
                patch.object(
                    tm.cover,
                    "generate_ai_video_cover",
                    side_effect=ValueError("No model found"),
                ),
                patch.object(
                    tm.cover,
                    "generate_video_cover",
                    return_value=os.path.join(task_dir, "cover.jpg"),
                ) as generate_cover,
                patch.object(
                    tm.cover,
                    "generate_title_card_cover",
                ) as generate_title_card,
                patch.object(
                    tm.upload_post.upload_post_service,
                    "is_configured",
                    return_value=False,
                ),
                patch.object(tm.sm.state, "update_task"),
            ):
                result = tm.start("task-id", params)

        self.assertEqual(
            generate_cover.call_args.kwargs["source_image_ref"],
            "https://example.com/cover.jpg",
        )
        self.assertEqual(
            generate_cover.call_args.kwargs["title"], "Edited Video Subject"
        )
        generate_title_card.assert_not_called()
        self.assertEqual(result["cover_error"], "No model found")

    def test_start_shortens_script_when_tts_audio_exceeds_target_duration(self):
        params = VideoParams(
            video_subject="Long Article",
            video_script="original long script",
            video_terms=["term"],
            video_duration_minutes=3,
            video_source="local",
            generate_video_cover=False,
        )

        generated_audio = [
            ("audio.mp3", 600.0, object()),
            ("audio.mp3", 175.0, object()),
        ]

        with tempfile.TemporaryDirectory() as task_dir:
            with (
                patch.object(tm.utils, "task_dir", return_value=task_dir),
                patch.object(tm, "generate_script", return_value="original long script"),
                patch.object(
                    tm, "generate_audio", side_effect=generated_audio
                ) as generate_audio,
                patch.object(
                    tm.llm,
                    "shorten_script_to_duration",
                    return_value="short script",
                ) as shorten_script,
                patch.object(tm, "generate_subtitle", return_value="subtitle.srt") as generate_subtitle,
                patch.object(tm, "get_video_materials", return_value=["material.mp4"]),
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"]),
                ),
                patch.object(
                    tm.upload_post.upload_post_service,
                    "is_configured",
                    return_value=False,
                ),
                patch.object(tm.sm.state, "update_task"),
            ):
                tm.start("task-id", params)

        self.assertEqual(generate_audio.call_count, 2)
        shorten_script.assert_called_once_with(
            video_script="original long script",
            video_duration_minutes=3,
            script_style=params.script_style,
        )
        generate_subtitle.assert_called_once()
        self.assertEqual(generate_subtitle.call_args.args[2], "short script")

    def test_generate_script_fetches_url_content_and_derives_subject(self):
        params = VideoParams(
            source_url="https://example.com/story",
            script_style="podcast",
            video_subject="",
        )
        source_content = tm.content_source.SourceContent(
            source_type="article",
            source_url=params.source_url,
            final_url=params.source_url,
            title="Example Story",
            content="Grounded article content.",
            extraction_method="article",
        )

        with (
            patch.object(
                tm.content_source, "fetch_source_content", return_value=source_content
            ) as fetch_source,
            patch.object(
                tm.llm, "generate_script", return_value="[Host] A\n[Guest] B"
            ) as generate,
        ):
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "[Host] A\n[Guest] B")
        self.assertEqual(params.video_subject, "Example Story")
        fetch_source.assert_called_once_with(params.source_type, params.source_url)
        generate.assert_called_once_with(
            video_subject="Example Story",
            language="",
            video_duration_minutes=1,
            paragraph_number=1,
            video_script_prompt="",
            custom_system_prompt="",
            script_style=params.script_style,
            source_content="Grounded article content.",
            source_title="Example Story",
            source_url="https://example.com/story",
        )

    def test_generate_script_resolves_source_metadata_for_existing_script(self):
        params = VideoParams(
            source_url="https://example.com/story",
            video_script="A ready-made script.",
            video_subject="",
            source_title="",
            source_cover_image_url="",
        )
        source_meta = tm.content_source.SourceContent(
            source_type="article",
            source_url="https://example.com/story",
            final_url="https://example.com/story?ref=1",
            title="Example Story",
            content="",
            extraction_method="article-metadata",
            cover_image_url="https://example.com/cover.jpg",
        )

        with (
            patch.object(
                tm.content_source,
                "fetch_source_metadata",
                return_value=source_meta,
            ) as fetch_source_metadata,
            patch.object(tm.llm, "generate_script") as generate,
        ):
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "A ready-made script.")
        self.assertEqual(params.source_title, "Example Story")
        self.assertEqual(params.source_cover_image_url, "https://example.com/cover.jpg")
        self.assertEqual(params.source_url, "https://example.com/story?ref=1")
        self.assertEqual(params.video_subject, "Example Story")
        fetch_source_metadata.assert_called_once_with(
            params.source_type, "https://example.com/story"
        )
        generate.assert_not_called()

    def test_generate_terms_uses_script_order_mode_when_enabled(self):
        """
        默认模式不受影响；只有用户显式开启素材按文案顺序匹配时，任务层才
        要求 LLM 生成有序关键词，并适当增加关键词数量以覆盖更多脚本片段。
        """
        params = VideoParams(
            video_subject="城市通勤",
            video_script="",
            match_materials_to_script=True,
        )

        with patch.object(
            tm.llm, "generate_terms", return_value=["city", "train"]
        ) as generate:
            result = tm.generate_terms("task-id", params, "先城市，再地铁")

        self.assertEqual(result, ["city", "train"])
        generate.assert_called_once_with(
            video_subject="城市通勤",
            video_script="先城市，再地铁",
            amount=8,
            match_script_order=True,
        )

    def test_generate_audio_uses_custom_file_inside_task_directory(self):
        task_id = "test-custom-audio-safe"
        task_dir = utils.task_dir(task_id)
        custom_audio_file = os.path.join(task_dir, "custom-audio.mp3")
        with open(custom_audio_file, "wb") as audio:
            audio.write(b"fake audio")

        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=custom_audio_file,
            voice_name="test-voice",
        )

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.voice, "get_audio_duration", return_value=7),
            ):
                audio_file, audio_duration, sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(custom_audio_file))
        self.assertEqual(audio_duration, 7)
        self.assertIsNone(sub_maker)
        tts.assert_not_called()

    def test_generate_audio_accepts_server_side_custom_file(self):
        task_id = "test-custom-audio-server-side"
        task_dir = utils.task_dir(task_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            server_audio.write(b"fake audio")
            server_audio.flush()
            params = VideoParams(
                video_subject="custom audio",
                video_script="",
                custom_audio_file=server_audio.name,
                voice_name="test-voice",
            )

            try:
                with (
                    patch.object(tm.voice, "tts") as tts,
                    patch.object(tm.voice, "get_audio_duration", return_value=6),
                ):
                    audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                        task_id, params, "script"
                    )
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(server_audio.name))
        self.assertEqual(audio_duration, 6)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()

    def test_generate_audio_rejects_missing_custom_file_without_tts(self):
        task_id = "test-custom-audio-missing"
        task_dir = utils.task_dir(task_id)
        missing_audio_file = os.path.join(task_dir, "missing.mp3")
        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=missing_audio_file,
            voice_name="test-voice",
        )

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.sm.state, "update_task") as update_task,
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()
        update_task.assert_called_with(task_id, state=tm.const.TASK_STATE_FAILED)

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials = []
        for i in range(1, 4):
            video_materials.append(
                MaterialInfo(
                    provider="local",
                    url=os.path.join(resources_dir, f"{i}.png"),
                    duration=0,
                )
            )

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1,
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)


if __name__ == "__main__":
    unittest.main()
