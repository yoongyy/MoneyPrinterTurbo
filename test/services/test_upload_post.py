import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services import upload_post


class TestUploadPostService(unittest.TestCase):
    def test_upload_video_includes_thumbnail_and_tiktok_cover_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            cover_path = Path(temp_dir) / "cover.jpg"
            video_path.write_bytes(b"fake video")
            cover_path.write_bytes(b"fake image")

            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"success": True}

            with (
                patch.object(upload_post.upload_post_service, "is_configured", return_value=True),
                patch.object(upload_post.requests, "post", return_value=response) as post,
            ):
                result = upload_post.upload_post_service.upload_video(
                    str(video_path),
                    "Clean Title",
                    platforms=["tiktok", "youtube"],
                    youtube_extra={
                        "youtube_title": "YT title",
                        "youtube_description": "Description",
                        "tags": ["tag1"],
                        "privacyStatus": "public",
                        "thumbnail_path": str(cover_path),
                        "tiktok_cover_timestamp": 1500,
                    },
                    cover_image_path=str(cover_path),
                    is_short_form=False,
                )

            self.assertTrue(result["success"])
            _, kwargs = post.call_args
            self.assertIn(("cover_timestamp", "1500"), kwargs["data"])
            self.assertIn(("youtube_title", "YT title"), kwargs["data"])
            file_fields = [item[0] for item in kwargs["files"]]
            self.assertIn("thumbnail", file_fields)

    def test_upload_video_skips_youtube_thumbnail_for_short_form(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            cover_path = Path(temp_dir) / "cover.jpg"
            video_path.write_bytes(b"fake video")
            cover_path.write_bytes(b"fake image")

            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"success": True}

            with (
                patch.object(upload_post.upload_post_service, "is_configured", return_value=True),
                patch.object(upload_post.requests, "post", return_value=response) as post,
            ):
                upload_post.upload_post_service.upload_video(
                    str(video_path),
                    "Clean Title",
                    platforms=["youtube"],
                    youtube_extra={
                        "thumbnail_path": str(cover_path),
                    },
                    cover_image_path=str(cover_path),
                    is_short_form=True,
                )

            _, kwargs = post.call_args
            file_fields = [item[0] for item in kwargs["files"]]
            self.assertNotIn("thumbnail", file_fields)


if __name__ == "__main__":
    unittest.main()
