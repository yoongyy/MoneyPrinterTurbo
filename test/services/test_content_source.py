import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.services import article, content_source


class TestContentSourceService(unittest.TestCase):
    def test_article_source_preserves_existing_extraction(self):
        article_content = article.ArticleContent(
            source_url="https://example.com/story",
            final_url="https://example.com/story?tracking=1",
            title="Example Story",
            content="Grounded article content.",
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

    @patch.object(content_source.AudioSegment, "from_file")
    def test_prepare_audio_for_gemini_transcodes_webm_to_mp3(self, from_file):
        class FakeSegment:
            def export(self, output_path, format="mp3", bitrate=None):
                Path(output_path).write_bytes(b"mp3-audio")

        from_file.return_value = FakeSegment()

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.webm"
            source.write_bytes(b"audio")

            result = content_source._prepare_audio_for_gemini(str(source), Path(temp_dir))
            self.assertEqual(result.name, "source-gemini.mp3")
            self.assertEqual(result.read_bytes(), b"mp3-audio")
        from_file.assert_called_once_with(str(source))

    @patch.object(content_source.config, "proxy", {"https": "http://proxy.local:8080"})
    def test_ytdlp_options_includes_proxy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            options = content_source._ytdlp_options(Path(temp_dir))

        self.assertEqual(options["proxy"], "http://proxy.local:8080")

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
