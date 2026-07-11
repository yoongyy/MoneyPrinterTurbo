import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.config import config
from app.services import cover


class TestCoverService(unittest.TestCase):
    def test_clean_cover_title_removes_special_characters_and_shortens(self):
        result = cover.clean_cover_title(
            "下周全球资本市场将迎来一场“多线大乱斗”！AI/美元/黄金&PCE 数据全面对决"
        )

        self.assertNotIn("！", result)
        self.assertNotIn("/", result)
        self.assertLessEqual(len(result), 28)

    def test_clean_cover_title_keeps_chinese_phrase_boundaries(self):
        result = cover.clean_cover_title(
            "三重暴击！ 引来跳水大戏！巨头行情看点，催化剂？微软MSFT ～Meta META"
        )

        self.assertEqual(result, "三重暴击 引来跳水大戏 巨头行情看点")

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

    def test_generate_ai_video_cover_uses_openrouter_images_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cover.jpg"
            image_buffer = io.BytesIO()
            Image.new("RGB", (640, 360), color=(24, 36, 54)).save(
                image_buffer, format="PNG"
            )
            encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

            class FakeResponse:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": [{"b64_json": encoded_image}]}

            original_app = dict(config.app)
            config.app.update(
                {
                    "openrouter_api_key": "test-key",
                    "openrouter_base_url": "https://openrouter.ai/api/v1",
                    "openrouter_image_model_name": "openai/gpt-image-1",
                    "openrouter_image_quality": "high",
                }
            )
            try:
                with (
                    patch.object(
                        cover,
                        "_get_openrouter_image_models",
                        return_value=("openai/gpt-image-1",),
                    ),
                    patch("app.services.cover.requests.post") as mock_post,
                ):
                    mock_post.return_value = FakeResponse()

                    result = cover.generate_ai_video_cover(
                        title="三重暴击！ 引来跳水大戏！巨头行情看点",
                        output_file=str(output_path),
                        video_aspect="16:9",
                        provider="openrouter",
                    )
            finally:
                config.app.clear()
                config.app.update(original_app)

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            request_payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(request_payload["model"], "openai/gpt-image-1")
            self.assertEqual(request_payload["aspect_ratio"], "16:9")
            self.assertIn("prompt", request_payload)

    def test_generate_ai_video_cover_falls_back_from_unknown_openrouter_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cover.jpg"
            image_buffer = io.BytesIO()
            Image.new("RGB", (640, 360), color=(24, 36, 54)).save(
                image_buffer, format="PNG"
            )
            encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

            class FakeResponse:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": [{"b64_json": encoded_image}]}

            original_app = dict(config.app)
            config.app.update(
                {
                    "openrouter_api_key": "test-key",
                    "openrouter_base_url": "https://openrouter.ai/api/v1",
                    "openrouter_image_model_name": "stabilityai/stable-diffusion-3.5-large",
                    "openrouter_image_quality": "high",
                }
            )
            try:
                with (
                    patch.object(
                        cover,
                        "_get_openrouter_image_models",
                        return_value=("google/gemini-2.5-flash-image",),
                    ),
                    patch("app.services.cover.requests.post") as mock_post,
                ):
                    mock_post.return_value = FakeResponse()
                    result = cover.generate_ai_video_cover(
                        title="Test",
                        output_file=str(output_path),
                        provider="openrouter",
                    )
            finally:
                config.app.clear()
                config.app.update(original_app)

            self.assertEqual(result, str(output_path))
            request_payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(
                request_payload["model"], "google/gemini-2.5-flash-image"
            )

    def test_generate_ai_video_cover_falls_back_from_openrouter_auto_to_concrete_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cover.jpg"
            image_buffer = io.BytesIO()
            Image.new("RGB", (640, 360), color=(24, 36, 54)).save(
                image_buffer, format="PNG"
            )
            encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

            class FakeResponse:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": [{"b64_json": encoded_image}]}

            original_app = dict(config.app)
            config.app.update(
                {
                    "openrouter_api_key": "test-key",
                    "openrouter_base_url": "https://openrouter.ai/api/v1",
                    "openrouter_image_model_name": "openrouter/auto",
                    "openrouter_image_quality": "high",
                }
            )
            try:
                with (
                    patch.object(
                        cover,
                        "_get_openrouter_image_models",
                        return_value=(
                            "openrouter/auto",
                            "google/gemini-2.5-flash-image",
                        ),
                    ),
                    patch("app.services.cover.requests.post") as mock_post,
                ):
                    mock_post.return_value = FakeResponse()

                    result = cover.generate_ai_video_cover(
                        title="Test",
                        output_file=str(output_path),
                        video_aspect="9:16",
                        provider="openrouter",
                    )
            finally:
                config.app.clear()
                config.app.update(original_app)

            self.assertEqual(result, str(output_path))
            request_payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(
                request_payload["model"], "google/gemini-2.5-flash-image"
            )


if __name__ == "__main__":
    unittest.main()
