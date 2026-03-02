"""
config.py — 統一設定檔管理

集中 settings.json 的讀寫邏輯，取代散落在 app.py / app-gpu.py 的重複實作。

用法：
    from config import AppSettings
    cfg = AppSettings()          # 預設 settings.json
    cfg = AppSettings("gpu")     # settings-gpu.json

    s = cfg.load()               # → dict
    cfg.save(s)                   # 寫入完整 dict
    cfg.patch("key", value)       # 讀→改→寫單一 key
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── 路徑 ──────────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent


@dataclass
class SettingsSchema:
    """settings.json 的 schema 定義。

    所有欄位都有預設值，缺少的 key 會自動補上。
    新增欄位時只需在此加一行即可。
    """
    # 推理後端
    backend:           str  = "openvino"     # "openvino" | "chatllm" | "cuda"
    device:            str  = "CPU"
    cpu_model_size:    str  = "0.6B"         # "0.6B" | "1.7B"

    # 模型路徑
    model_dir:         str  = ""             # OpenVINO 模型資料夾
    model_path:        str  = ""             # chatllm .bin 模型路徑
    chatllm_dir:       str  = ""             # chatllm DLL 目錄

    # UI 偏好
    output_simplified: bool = False          # True=簡體 / False=繁體
    appearance_mode:   str  = "dark"         # "dark" | "light"
    vad_threshold:     float | None = None   # 自訂 VAD 閾值


class AppSettings:
    """設定檔讀寫管理器。

    Parameters
    ----------
    variant : str
        設定檔變體名稱。
        ""  → settings.json（預設）
        "gpu" → settings-gpu.json
    """

    def __init__(self, variant: str = ""):
        suffix = f"-{variant}" if variant else ""
        self.path = BASE_DIR / f"settings{suffix}.json"

    def load(self) -> dict:
        """讀取設定檔，回傳 dict。檔案不存在或解析失敗時回傳空 dict。"""
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save(self, settings: dict) -> None:
        """儲存完整設定 dict。"""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def patch(self, key: str, value: Any) -> None:
        """讀取現有設定、更新單一 key，再寫回。"""
        s = self.load()
        s[key] = value
        self.save(s)

    def load_with_defaults(self) -> dict:
        """讀取設定並以 SettingsSchema 預設值填補缺少的 key。"""
        defaults = asdict(SettingsSchema())
        saved = self.load()
        # 已存在的 key 覆蓋預設值；缺少的 key 用預設值補上
        return {**defaults, **saved}
