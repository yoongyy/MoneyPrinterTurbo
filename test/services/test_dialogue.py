import os
import tempfile
import types
import unittest
from unittest.mock import patch

from pydub import AudioSegment

from app.models.schema import VideoParams
from app.services import dialogue
from app.services import task as task_service


class TestDialogueService(unittest.TestCase):
    def test_parse_bracketed_dialogue_and_assign_voices(self):
        segments = dialogue.parse_dialogue_script(
            "[Host] Welcome to the show.\n[Guest] Thanks for having me.\n[Host] Let us begin.",
            "podcast",
        )

        self.assertEqual([segment.speaker for segment in segments], ["Host", "Guest", "Host"])
        self.assertEqual(
            dialogue.assign_speaker_voices(segments, "voice-a", "voice-b"),
            {"Host": "voice-a", "Guest": "voice-b"},
        )

    def test_strip_speaker_labels_keeps_spoken_content(self):
        result = dialogue.strip_speaker_labels(
            "[Host] First point.\n[Guest] Second point.", "talkshow"
        )

        self.assertEqual(result, "First point.\nSecond point.")

    def test_write_dialogue_subtitle_uses_combined_timeline(self):
        timeline = dialogue.DialogueSubtitleTimeline(
            cues=[
                dialogue.SubtitleCue(0.0, 1.25, "Host: Hello"),
                dialogue.SubtitleCue(1.45, 2.5, "Guest: Hi"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "dialogue.srt")
            task_service._write_dialogue_subtitle(timeline, output)
            with open(output, encoding="utf-8") as subtitle_file:
                content = subtitle_file.read()

        self.assertIn("00:00:00,000 --> 00:00:01,250", content)
        self.assertIn("00:00:01,450 --> 00:00:02,500", content)
        self.assertIn("Host: Hello", content)

    def test_generate_dialogue_audio_assigns_two_voices_and_offsets_subtitles(self):
        used_voices = []

        def fake_tts(text, voice_name, voice_rate, voice_file, **_kwargs):
            used_voices.append(voice_name)
            exported_audio = AudioSegment.silent(duration=300).export(
                voice_file, format="mp3"
            )
            exported_audio.close()
            return types.SimpleNamespace()

        def fake_create_subtitle(text, sub_maker, subtitle_file):
            with open(subtitle_file, "w", encoding="utf-8") as output:
                output.write(f"1\n00:00:00,000 --> 00:00:00,300\n{text}\n")

        params = VideoParams(
            script_style="podcast",
            voice_name="voice-a",
            voice_name_2="voice-b",
            dialogue_pause=0.2,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            task_service.utils, "task_dir", return_value=directory
        ), patch.object(task_service.voice, "tts", side_effect=fake_tts), patch.object(
            task_service.voice, "create_subtitle", side_effect=fake_create_subtitle
        ):
            audio_file, duration, timeline = task_service._generate_dialogue_audio(
                "task-id", params, "[Host] Hello.\n[Guest] Hi."
            )

            self.assertTrue(os.path.exists(audio_file))
            self.assertGreaterEqual(duration, 0.75)
            self.assertEqual(used_voices, ["voice-a", "voice-b"])
            self.assertEqual(len(timeline.cues), 2)
            self.assertGreaterEqual(timeline.cues[1].start, 0.49)
