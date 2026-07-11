import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

# 测试文件直接运行时，也能从仓库根目录导入 app 包。
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import article, content_source


class TestContentSourceService(unittest.TestCase):
    def test_article_source_preserves_existing_extraction(self):
        article_content = article.ArticleContent(
            source_url="https://example.com/story",
            final_url="https://example.com/story?tracking=1",
            title="Example Story",
            content="Grounded article content.",
            cover_image_url="https://example.com/cover.jpg",
        )

        with patch.object(
            content_source.article,
            "fetch_url_content",
            return_value=article_content,
        ) as fetch:
            result = content_source.fetch_source_content(
                "article", "https://example.com/story"
            )

        fetch.assert_called_once_with("https://example.com/story")
        self.assertEqual(result.extraction_method, "article")
        self.assertEqual(result.title, "Example Story")
        self.assertEqual(result.final_url, "https://example.com/story")
        self.assertEqual(result.cover_image_url, "https://example.com/cover.jpg")

    def test_media_url_must_match_selected_platform(self):
        with self.assertRaisesRegex(ValueError, "valid youtube URL"):
            content_source.validate_media_url(
                "youtube", "https://www.tiktok.com/@creator/video/123"
            )

    def test_youtube_safe_url_preserves_video_id_only(self):
        safe_url = content_source.safe_media_url(
            "youtube",
            "https://www.youtube.com/watch?v=abc123&utm_source=test&feature=share",
        )

        self.assertEqual(safe_url, "https://www.youtube.com/watch?v=abc123")

    def test_parse_vtt_removes_timing_and_duplicate_lines(self):
        transcript = content_source.parse_caption_text(
            """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello <c>world</c>

00:00:02.000 --> 00:00:04.000
Hello world

00:00:04.000 --> 00:00:06.000
This is the transcript.
""",
            "vtt",
        )

        self.assertEqual(transcript, "Hello world This is the transcript.")

    def test_prepare_audio_for_gemini_transcodes_webm_to_mp3(self):
        class FakeResult:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, capture_output=True, text=True):
            Path(command[-1]).write_bytes(b"mp3-audio")
            return FakeResult()

        with patch.object(content_source.subprocess, "run", side_effect=fake_run) as run:
            with tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / "source.webm"
                source.write_bytes(b"audio")

                result = content_source._prepare_audio_for_gemini(
                    str(source), Path(temp_dir)
                )

                self.assertEqual(result.name, "source-gemini.mp3")
                self.assertEqual(result.read_bytes(), b"mp3-audio")
        run.assert_called_once()

    @patch.object(content_source.config, "proxy", {"https": "http://proxy.local:8080"})
    def test_ytdlp_options_includes_proxy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            options = content_source._ytdlp_options(Path(temp_dir))

        self.assertEqual(options["proxy"], "http://proxy.local:8080")

    def test_download_audio_prefers_h264_when_audio_only_is_unavailable(self):
        captured = {}

        class FakeYoutubeDL:
            def __init__(self, options):
                captured["format"] = options["format"]
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def download(self, urls):
                output = Path(self.options["outtmpl"].replace("%(ext)s", "mp4"))
                output.write_bytes(b"media")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(content_source, "YoutubeDL", FakeYoutubeDL):
                result = content_source._download_audio(
                    "https://www.tiktok.com/@creator/video/123", Path(temp_dir)
                )

        self.assertEqual(result.name, "source.mp4")
        self.assertIn("bestaudio", captured["format"])
        self.assertIn("best[vcodec=h264][acodec!=none]", captured["format"])
        self.assertNotEqual(captured["format"], "bestaudio/best")

    def test_pick_thumbnail_url_prefers_best_available_image(self):
        info = {
            "thumbnails": [
                {"url": "https://example.com/1.jpg", "width": 320, "height": 180},
                {"url": "https://example.com/2.jpg", "width": 1280, "height": 720},
            ]
        }

        self.assertEqual(
            content_source._pick_thumbnail_url(info),
            "https://example.com/2.jpg",
        )

    @patch.object(content_source, "_extract_media_info")
    def test_fetch_source_metadata_for_media_uses_thumbnail_without_transcription(
        self, extract_info
    ):
        extract_info.return_value = {
            "title": "Video title",
            "webpage_url": "https://www.youtube.com/watch?v=abc",
            "thumbnails": [
                {"url": "https://example.com/thumb-small.jpg", "width": 320, "height": 180},
                {"url": "https://example.com/thumb.jpg", "width": 1280, "height": 720},
            ],
        }

        result = content_source.fetch_source_metadata("youtube", "https://youtu.be/abc")

        self.assertEqual(result.title, "Video title")
        self.assertEqual(result.cover_image_url, "https://example.com/thumb.jpg")
        self.assertEqual(result.content, "")
        self.assertEqual(result.extraction_method, "metadata")

    @patch.object(content_source.subtitle, "transcribe_text")
    @patch.object(content_source, "_transcribe_audio_with_gemini", return_value="")
    @patch.object(content_source, "_download_audio")
    @patch.object(content_source, "_extract_caption", return_value="")
    @patch.object(content_source, "_extract_media_info")
    def test_media_falls_back_to_audio_transcription(
        self,
        extract_info,
        extract_caption,
        download_audio,
        transcribe_audio_with_gemini,
        transcribe_text,
    ):
        extract_info.return_value = {
            "title": "Video title",
            "webpage_url": "https://www.youtube.com/watch?v=abc",
            "duration": 30,
            "thumbnail": "https://example.com/thumb.jpg",
        }

        def create_audio(_url, work_dir):
            audio = Path(work_dir) / "source.m4a"
            audio.write_bytes(b"audio")
            return audio

        download_audio.side_effect = create_audio
        transcribe_text.return_value = (
            "A sufficiently useful locally generated transcript."
        )

        result = content_source.fetch_source_content("youtube", "https://youtu.be/abc")

        self.assertEqual(result.extraction_method, "speech-to-text")
        self.assertEqual(result.title, "Video title")
        self.assertEqual(result.cover_image_url, "https://example.com/thumb.jpg")
        transcribe_audio_with_gemini.assert_called_once()
        transcribe_text.assert_called_once()

    @patch.object(content_source.subtitle, "transcribe_text")
    @patch.object(
        content_source,
        "_transcribe_audio_with_gemini",
        return_value="A Gemini generated transcript.",
    )
    @patch.object(content_source, "_download_audio")
    @patch.object(content_source, "_extract_caption", return_value="")
    @patch.object(content_source, "_extract_media_info")
    def test_media_prefers_gemini_transcription_before_whisper(
        self,
        extract_info,
        extract_caption,
        download_audio,
        transcribe_audio_with_gemini,
        transcribe_text,
    ):
        extract_info.return_value = {
            "title": "Video title",
            "webpage_url": "https://www.youtube.com/watch?v=abc",
            "duration": 30,
        }

        def create_audio(_url, work_dir):
            audio = Path(work_dir) / "source.m4a"
            audio.write_bytes(b"audio")
            return audio

        download_audio.side_effect = create_audio

        result = content_source.fetch_source_content("youtube", "https://youtu.be/abc")

        self.assertEqual(result.extraction_method, "gemini-speech-to-text")
        self.assertEqual(result.content, "A Gemini generated transcript.")
        transcribe_audio_with_gemini.assert_called_once()
        transcribe_text.assert_not_called()

    @patch.object(content_source.subtitle, "transcribe_text")
    @patch.object(content_source, "_download_audio")
    @patch.object(
        content_source,
        "_extract_caption",
        return_value="A sufficiently useful platform caption transcript.",
    )
    @patch.object(content_source, "_extract_media_info")
    def test_media_prefers_platform_captions(
        self, extract_info, extract_caption, download_audio, transcribe_text
    ):
        extract_info.return_value = {
            "title": "TikTok title",
            "webpage_url": "https://www.tiktok.com/@creator/video/123",
            "duration": 20,
        }

        result = content_source.fetch_source_content(
            "tiktok", "https://www.tiktok.com/@creator/video/123"
        )

        self.assertEqual(result.extraction_method, "captions")
        download_audio.assert_not_called()
        transcribe_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
