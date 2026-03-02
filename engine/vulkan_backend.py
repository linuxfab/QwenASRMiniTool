"""
engine/vulkan_backend.py — Vulkan GPU 推理後端（chatllm.cpp）

使用 chatllm_engine.py 中的 _DLLASRRunner / _ChatLLMRunner 作為底層推理，
本模組只負責統一介面封裝。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from asr_utils import SAMPLE_RATE, LANG_CODE
from .base import ASREngineBase, BASE_DIR

# _LANG_CODE 已移至 asr_utils.LANG_CODE，保留別名供內部相容
_LANG_CODE = LANG_CODE


class VulkanEngine(ASREngineBase):
    """chatllm.cpp + Vulkan GPU 推理後端。

    優先使用 DLL 模式（_DLLASRRunner）：
      - 模型常駐 GPU 記憶體，每 chunk ~0.23s
      - 若 libchatllm.dll 不存在，自動回退到 subprocess 模式（_ChatLLMRunner）

    processor = None（chatllm 不使用 LightProcessor）
    """

    max_chunk_secs = 30

    def __init__(self):
        super().__init__()
        self._runner = None
        self._use_dll = False

    def load(
        self,
        *,
        model_path: "str | Path",
        chatllm_dir: "str | Path",
        n_gpu_layers: int = 99,
        device_id: int = 0,
        cb=None,
    ):
        """載入 Vulkan 模型。"""
        from chatllm_engine import _DLLASRRunner, _ChatLLMRunner

        def _s(msg):
            if cb:
                cb(msg)

        self._model_path = Path(model_path)
        self._chatllm_dir = Path(chatllm_dir)
        self._n_gpu_layers = n_gpu_layers

        # ── 共用：VAD + Diarize + OpenCC ────────────────────────────
        # Vulkan 使用 ov_models/ 或 GPUModel/ 中的 VAD 模型
        model_dir = BASE_DIR / "ov_models"
        self._load_vad(model_dir, cb=cb)
        self._load_diarization(model_dir, cb=cb)
        self._load_opencc()

        # ── 驗證路徑 ─────────────────────────────────────────────────
        if not self._model_path.exists():
            raise FileNotFoundError(f"模型不存在：{self._model_path}")

        # ── 建立 Runner：優先 DLL，後備 subprocess ───────────────
        dll_path = self._chatllm_dir / "libchatllm.dll"
        if dll_path.exists():
            try:
                _s("載入 chatllm 模型（DLL 模式，Vulkan 全層 GPU）…")
                self._runner = _DLLASRRunner(
                    model_path=model_path,
                    chatllm_dir=chatllm_dir,
                    n_gpu_layers=n_gpu_layers,
                    device_id=device_id,
                    cb=cb,
                )
                self._use_dll = True
                self.ready = True
                _s("ChatLLM DLL 載入完成（模型常駐 GPU，每 chunk ~0.23s）")
                return
            except Exception as e:
                _s(f"DLL 模式失敗（{e}），改用 subprocess 模式…")

        _s("驗證 chatllm 模型（subprocess 模式）…")
        self._runner = _ChatLLMRunner(
            model_path=model_path,
            chatllm_dir=chatllm_dir,
            n_gpu_layers=n_gpu_layers,
            device_id=device_id,
        )
        self._use_dll = False
        self.ready = True
        _s("ChatLLM 載入完成（subprocess 模式，Vulkan GPU）")

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """16kHz float32 → 轉錄文字。"""
        import soundfile as sf

        # 語系 → system prompt
        sys_prompt: str | None = None
        if language and language != "自動偵測":
            code = _LANG_CODE.get(language, language.lower()[:2])
            sys_prompt = (
                f"The audio language is {language}. "
                f"Transcribe it and output strictly in this format: "
                f"language {code}<asr_text>[transcription]. "
                f"Output only {language} text after <asr_text>, no translation."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
            text = self._runner.transcribe(tmp_path, sys_prompt=sys_prompt)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # OpenCC 簡→繁轉換
        if self.cc and text and not self.output_simplified:
            text = self.cc.convert(text)

        return text

    def __del__(self):
        pass   # DLL runner 由 GC 自然回收
