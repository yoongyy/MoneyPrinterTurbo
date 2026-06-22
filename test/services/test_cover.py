import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.services import cover


class TestCoverService(unittest.TestCase):
    def test_clean_cover_title_removes_special_characters_and_shortens(self):
        result = cover.clean_cover_title(
            "下周全球资本市场将迎来一场“多线大乱斗”！AI/美元/黄金&PCE 数据全面对决"
        )

        self.assertNotIn("！", result)
        self.assertNotIn("/", result)
        self.assertLessEqual(len(result), 28)

    def test_generate_video_cover_from_local_source_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.jpg"
            output_path = Path(temp_dir) / "cover.jpg"
            Image.new("RGB", (1600, 900), color=(20, 80, 140)).save(source_path)

            result = cover.generate_video_cover(
                source_image_ref=str(source_path),
                title="A very long title with special characters! / #test",
                output_file=str(output_path),
                video_aspect="9:16",
            )

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
