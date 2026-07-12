import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import toml

import app.config.config as config_module
from app.config.config import _migrate_upload_post_config


class TestUploadPostConfigMigration(unittest.TestCase):
    def test_moves_upload_post_settings_from_ui_to_app(self):
        cfg = {
            "app": {},
            "ui": {
                "hide_log": True,
                "upload_post_enabled": True,
                "upload_post_api_key": "api-key",
                "upload_post_username": "user",
                "upload_post_platforms": ["youtube"],
                "upload_post_auto_upload": True,
                "upload_post_youtube_privacy_status": "public",
            },
        }

        migrated = _migrate_upload_post_config(cfg)

        self.assertEqual(migrated["app"]["upload_post_enabled"], True)
        self.assertEqual(migrated["app"]["upload_post_api_key"], "api-key")
        self.assertEqual(migrated["app"]["upload_post_username"], "user")
        self.assertEqual(migrated["app"]["upload_post_platforms"], ["youtube"])
        self.assertEqual(migrated["app"]["upload_post_auto_upload"], True)
        self.assertEqual(migrated["app"]["upload_post_youtube_privacy_status"], "public")
        self.assertEqual(migrated["ui"], {"hide_log": True})

    def test_preserves_existing_app_values_and_drops_ui_copy(self):
        cfg = {
            "app": {
                "upload_post_enabled": True,
                "upload_post_username": "kept-user",
            },
            "ui": {
                "upload_post_enabled": False,
                "upload_post_username": "stale-user",
            },
        }

        migrated = _migrate_upload_post_config(cfg)

        self.assertEqual(migrated["app"]["upload_post_enabled"], True)
        self.assertEqual(migrated["app"]["upload_post_username"], "kept-user")
        self.assertNotIn("upload_post_enabled", migrated.get("ui", {}))
        self.assertNotIn("upload_post_username", migrated.get("ui", {}))

    def test_save_config_keeps_file_upload_post_values_over_stale_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            with config_path.open("w", encoding="utf-8") as fp:
                toml.dump(
                    {
                        "app": {
                            "upload_post_enabled": True,
                            "upload_post_api_key": "file-key",
                            "upload_post_username": "file-user",
                        },
                        "azure": {},
                        "siliconflow": {},
                        "ui": {"hide_log": True},
                    },
                    fp,
                )

            original_state = {
                "config_file": config_module.config_file,
                "_cfg": config_module._cfg,
                "app": config_module.app,
                "azure": config_module.azure,
                "siliconflow": config_module.siliconflow,
                "ui": config_module.ui,
            }

            try:
                config_module.config_file = str(config_path)
                config_module._cfg = {
                    "app": {"upload_post_enabled": False, "upload_post_api_key": "stale-key"},
                    "azure": {},
                    "siliconflow": {},
                    "ui": {"hide_log": False},
                }
                config_module.app = config_module._cfg["app"]
                config_module.azure = config_module._cfg["azure"]
                config_module.siliconflow = config_module._cfg["siliconflow"]
                config_module.ui = config_module._cfg["ui"]

                config_module.save_config()

                saved = toml.load(config_path)
                self.assertTrue(saved["app"]["upload_post_enabled"])
                self.assertEqual(saved["app"]["upload_post_api_key"], "file-key")
                self.assertEqual(saved["app"]["upload_post_username"], "file-user")
            finally:
                config_module.config_file = original_state["config_file"]
                config_module._cfg = original_state["_cfg"]
                config_module.app = original_state["app"]
                config_module.azure = original_state["azure"]
                config_module.siliconflow = original_state["siliconflow"]
                config_module.ui = original_state["ui"]

    def test_save_config_preserves_automation_values_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            with config_path.open("w", encoding="utf-8") as fp:
                toml.dump(
                    {
                        "app": {"video_source": "pexels"},
                        "azure": {},
                        "siliconflow": {},
                        "ui": {"hide_log": True},
                        "automation": {
                            "automation_news_card_background_color": [
                                "#CBEFF7",
                                "#E8F5E9",
                            ],
                            "automation_voice_name": [
                                "zh-CN-XiaoxiaoNeural-Female",
                                "zh-CN-YunxiNeural-Male",
                            ],
                        },
                    },
                    fp,
                )

            original_state = {
                "config_file": config_module.config_file,
                "_cfg": config_module._cfg,
                "app": config_module.app,
                "azure": config_module.azure,
                "siliconflow": config_module.siliconflow,
                "ui": config_module.ui,
            }

            try:
                config_module.config_file = str(config_path)
                config_module._cfg = {
                    "app": {"video_source": "local"},
                    "azure": {},
                    "siliconflow": {},
                    "ui": {"hide_log": False},
                    "automation": {
                        "automation_news_card_background_color": "#000000",
                        "automation_voice_name": "stale-voice",
                    },
                }
                config_module.app = config_module._cfg["app"]
                config_module.azure = config_module._cfg["azure"]
                config_module.siliconflow = config_module._cfg["siliconflow"]
                config_module.ui = config_module._cfg["ui"]

                config_module.save_config()

                saved = toml.load(config_path)
                self.assertEqual(
                    saved["automation"]["automation_news_card_background_color"],
                    ["#CBEFF7", "#E8F5E9"],
                )
                self.assertEqual(
                    saved["automation"]["automation_voice_name"],
                    ["zh-CN-XiaoxiaoNeural-Female", "zh-CN-YunxiNeural-Male"],
                )
            finally:
                config_module.config_file = original_state["config_file"]
                config_module._cfg = original_state["_cfg"]
                config_module.app = original_state["app"]
                config_module.azure = original_state["azure"]
                config_module.siliconflow = original_state["siliconflow"]
                config_module.ui = original_state["ui"]


if __name__ == "__main__":
    unittest.main()
