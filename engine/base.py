"""
engine/base.py — ASR 引擎抽象基底類別

定義所有後端（OpenVINO / Vulkan / CUDA）的統一介面。
process_file() 在此實作（統一 Pipeline），子類別只需覆寫：
  - load()       : 載入模型
  - transcribe() : 單段音訊 → 文字
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

from asr_utils import (
    SAMPLE_RATE, detect_speech_groups, split_to_lines, srt_ts, assign_ts,
)

# ── 路徑 ──────────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent.parent    # engine/ 的上一層

SRT_DIR = BASE_DIR / "subtitles"
SRT_DIR.mkdir(exist_ok=True)


class ASREngineBase:
    """所有 ASR 後端的統一介面。

    子類別必須覆寫：
      - load(**kwargs)
      - transcribe(audio, *, max_tokens, language, context) -> str

    可選覆寫：
      - process_file()  — 若後端有特殊需求（如 ForcedAligner）
    """

    max_chunk_secs: int = 30
    output_simplified: bool = False

    def __init__(self):
        self.ready       = False
        self._lock       = threading.Lock()
        self.vad_sess    = None
        self.diar_engine = None
        self.processor   = None   # LightProcessor 實例（OpenVINO）或 None（Vulkan/CUDA）
        self.cc          = None   # OpenCC 實例

    # ── 子類別必須實作 ────────────────────────────────────

    def load(self, *, cb=None, **kwargs):
        """載入模型。cb(msg) 用於更新 UI 狀態。"""
        raise NotImplementedError

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """將 16kHz float32 音訊轉錄為文字。"""
        raise NotImplementedError

    # ── 共用：VAD + Diarize 載入 ─────────────────────────

    def _load_vad(self, model_dir: Path, cb=None):
        """載入 VAD 模型（Silero VAD v4, ONNX CPU）。"""
        import onnxruntime as ort

        if cb:
            cb("載入 VAD 模型…")

        vad_candidates = [
            model_dir / "silero_vad_v4.onnx",
            BASE_DIR / "ov_models" / "silero_vad_v4.onnx",
            BASE_DIR / "GPUModel"  / "silero_vad_v4.onnx",
        ]
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            vad_candidates.insert(
                0, Path(sys._MEIPASS) / "ov_models" / "silero_vad_v4.onnx"
            )
        vad_path = next((p for p in vad_candidates if p.exists()), None)
        if vad_path is None:
            raise FileNotFoundError("找不到 silero_vad_v4.onnx")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

    def _load_diarization(self, model_dir: Path, cb=None):
        """載入說話者分離模型（可選，失敗時靜默跳過）。"""
        if cb:
            cb("載入說話者分離模型…")
        try:
            from diarize import DiarizationEngine
            diar_candidates = [
                model_dir / "diarization",
                BASE_DIR / "ov_models" / "diarization",
                BASE_DIR / "GPUModel"  / "diarization",
            ]
            diar_dir = next((p for p in diar_candidates if p.exists()), None)
            if diar_dir:
                eng = DiarizationEngine(diar_dir)
                self.diar_engine = eng if eng.ready else None
        except Exception:
            self.diar_engine = None

    def _load_opencc(self):
        """載入 OpenCC 簡→繁轉換。"""
        try:
            import opencc
            self.cc = opencc.OpenCC("s2twp")
        except Exception:
            self.cc = None

    # ── 共用：chunk 長度限制 ──────────────────────────────

    def _enforce_chunk_limit(
        self,
        groups: list[tuple[float, float, np.ndarray, "str | None"]],
    ) -> list[tuple[float, float, np.ndarray, "str | None"]]:
        """將超過 max_chunk_secs 的音訊段落切分為等長子片段。"""
        max_samples = self.max_chunk_secs * SAMPLE_RATE
        result = []
        for t0, t1, chunk, spk in groups:
            if len(chunk) <= max_samples:
                result.append((t0, t1, chunk, spk))
            else:
                pos = 0
                while pos < len(chunk):
                    piece = chunk[pos: pos + max_samples]
                    if len(piece) < SAMPLE_RATE:   # 不足 1 秒的殘餘片段跳過
                        break
                    piece_t0 = t0 + pos / SAMPLE_RATE
                    piece_t1 = min(t1, piece_t0 + len(piece) / SAMPLE_RATE)
                    result.append((piece_t0, piece_t1, piece, spk))
                    pos += max_samples
        return result

    # ── 統一 Pipeline：音檔 → SRT ─────────────────────────

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language: str | None = None,
        context: str | None = None,
        diarize: bool = False,
        n_speakers: int | None = None,
    ) -> Path | None:
        """音檔 → SRT，回傳 SRT 路徑。

        language   : 強制語系（如 "Chinese"），None 表示自動偵測
        context    : 辨識提示（歌詞/關鍵字），放入 system message
        diarize    : True 時用說話者分離取代 VAD，SRT 加說話者前綴
        n_speakers : 指定說話者人數（None=自動偵測）
        """
        from ffmpeg_utils import decode_audio_to_numpy, find_ffmpeg

        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            raise RuntimeError("找不到 ffmpeg，請重新啟動程式或手動安裝。")
        audio = decode_audio_to_numpy(audio_path, ffmpeg_exe, sr=SAMPLE_RATE)

        # ── 分段策略：說話者分離 vs 傳統 VAD ─────────────────────
        use_diar = diarize and self.diar_engine is not None and self.diar_engine.ready
        if use_diar:
            diar_segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
            if not diar_segs:
                return None
            groups_spk = [
                (t0, t1,
                 audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)],
                 spk)
                for t0, t1, spk in diar_segs
            ]
        else:
            vad_groups = detect_speech_groups(
                audio, self.vad_sess, self.max_chunk_secs
            )
            if not vad_groups:
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        # 強制切分超過 max_chunk_secs 的片段
        groups_spk = self._enforce_chunk_limit(groups_spk)

        # ── ASR 逐段轉錄 ─────────────────────────────────────────
        all_subs: list[tuple[float, float, str, str | None]] = []
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total,
                            f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")
            max_tok = 400 if language == "Japanese" else 300
            text = self.transcribe(
                chunk, max_tokens=max_tok, language=language, context=context
            )
            if not text:
                continue
            lines = split_to_lines(text)
            all_subs.extend(
                (s, e, line, spk) for s, e, line in assign_ts(lines, g0, g1)
            )

        if not all_subs:
            return None

        if progress_cb:
            progress_cb(total, total, "寫入 SRT…")

        SRT_DIR.mkdir(exist_ok=True)
        out = SRT_DIR / (audio_path.stem + ".srt")
        with open(out, "w", encoding="utf-8") as f:
            for idx, (s, e, line, spk) in enumerate(all_subs, 1):
                prefix = f"{spk}：" if spk else ""
                f.write(f"{idx}\n{srt_ts(s)} --> {srt_ts(e)}\n{prefix}{line}\n\n")
        return out
