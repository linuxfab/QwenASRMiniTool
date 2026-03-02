"""
test_config.py — config.py 單元測試

覆蓋：AppSettings、SettingsSchema、load/save/patch/load_with_defaults
"""
from pathlib import Path

import pytest

from config import AppSettings, SettingsSchema


class TestSettingsSchema:
    def test_defaults(self):
        s = SettingsSchema()
        assert s.backend == "openvino"
        assert s.device == "CPU"
        assert s.cpu_model_size == "0.6B"
        assert s.output_simplified is False
        assert s.appearance_mode == "dark"
        assert s.vad_threshold is None


class TestAppSettings:
    def test_load_nonexistent(self, tmp_path: Path):
        cfg = AppSettings()
        cfg.path = tmp_path / "no_such.json"
        assert cfg.load() == {}

    def test_save_and_load(self, tmp_path: Path):
        cfg = AppSettings()
        cfg.path = tmp_path / "test_settings.json"
        data = {"backend": "chatllm", "device": "GPU:0"}
        cfg.save(data)
        loaded = cfg.load()
        assert loaded["backend"] == "chatllm"
        assert loaded["device"] == "GPU:0"

    def test_patch(self, tmp_path: Path):
        cfg = AppSettings()
        cfg.path = tmp_path / "test_settings.json"
        cfg.save({"backend": "openvino", "device": "CPU"})
        cfg.patch("device", "GPU.0")
        loaded = cfg.load()
        assert loaded["device"] == "GPU.0"
        assert loaded["backend"] == "openvino"  # 其他 key 未被影響

    def test_load_with_defaults(self, tmp_path: Path):
        cfg = AppSettings()
        cfg.path = tmp_path / "partial.json"
        cfg.save({"backend": "cuda"})
        full = cfg.load_with_defaults()
        assert full["backend"] == "cuda"       # 使用者的值
        assert full["device"] == "CPU"         # 預設值填補
        assert full["appearance_mode"] == "dark"

    def test_variant_gpu(self, tmp_path: Path):
        cfg = AppSettings("gpu")
        # 相對路徑名稱正確
        assert cfg.path.name == "settings-gpu.json"

    def test_variant_default(self):
        cfg = AppSettings()
        assert cfg.path.name == "settings.json"

    def test_save_unicode(self, tmp_path: Path):
        cfg = AppSettings()
        cfg.path = tmp_path / "unicode.json"
        cfg.save({"note": "中文測試"})
        loaded = cfg.load()
        assert loaded["note"] == "中文測試"
