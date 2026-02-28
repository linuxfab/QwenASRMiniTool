"""
engine/cuda_backend.py — PyTorch CUDA 推理後端

使用 qwen_asr 官方 API，支援 CUDA / CPU。
僅供 Source 版本使用（需 torch），EXE 不包含。

注意：此後端的 process_file() 覆寫了基底版本，
因為它包含 ForcedAligner 精確時間軸的分支邏輯。
"""
from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np

from asr_utils import (
    SAMPLE_RATE, detect_speech_groups, split_to_lines, srt_ts, assign_ts,
)
from .base import ASREngineBase, BASE_DIR, SRT_DIR

GPU_MODEL_DIR      = BASE_DIR / "GPUModel"
OV_MODEL_DIR       = BASE_DIR / "ov_models"
ASR_MODEL_NAME     = "Qwen3-ASR-1.7B"
ALIGNER_MODEL_NAME = "Qwen3-ForcedAligner-0.6B"

# 中文子句結束標點（不保留，切行後隱藏）
_ZH_CLAUSE_END = frozenset('，。？！；：…—、·')
# 英文子句結束標點（含逗號，讓英文逗號也觸發切行）
_EN_SENT_END   = frozenset('.,!?;')

MAX_CHARS = 20


def _find_vad_model():
    """依序在 GPUModel/ 和 ov_models/ 尋找 Silero VAD ONNX。"""
    candidates = [
        GPU_MODEL_DIR / "silero_vad_v4.onnx",
        OV_MODEL_DIR  / "silero_vad_v4.onnx",
    ]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.insert(0, Path(sys._MEIPASS) / "ov_models" / "silero_vad_v4.onnx")
    return next((p for p in candidates if p.exists()), None)


def _ts_to_subtitle_lines(
    ts_list,
    raw_text: str,
    chunk_offset: float,
    spk: "str | None",
    cc,
    simplified: bool,
):
    """ForcedAligner token（詞級別）+ ASR 原文（含標點）→ 字幕行。

    ── FA token 與 raw_text 的對應關係 ────────────────────────────────
    FA 的 tokenize_space_lang():
      "Hello World" → ['HELLO', 'WORLD']
      "你好世界"     → ['你', '好', '世', '界']
    raw_text（ASR 全段原始輸出）：
      "Hello, World!" 或 "你好，世界！"
    兩者差異：FA 沒有標點與空格。
    ── 匹配策略 ──────────────────────────────────────────────────────
    用 FA token 序列對齊 raw_text 字元，非 token 匹配字元視為分隔符。
    """
    def _is_latin_word(w: str) -> bool:
        return bool(re.match(r'^[A-Za-z]', w))

    def _tokenize_text(text: str) -> list[str]:
        """模擬 FA 的 tokenize_space_lang：空格切詞 + 中文逐字。"""
        tokens: list[str] = []
        buf = []
        for ch in text:
            if ch == ' ':
                if buf:
                    tokens.append(''.join(buf))
                    buf = []
            elif '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                if buf:
                    tokens.append(''.join(buf))
                    buf = []
                tokens.append(ch)
            else:
                buf.append(ch)
        if buf:
            tokens.append(''.join(buf))
        return tokens

    def _emit(seg_tokens: list, seg_words: list[str]) -> None:
        if not seg_tokens:
            return
        start_s = seg_tokens[0].get('t0', 0)
        end_s   = seg_tokens[-1].get('t1', start_s + 0.1)
        txt     = ''.join(seg_words)
        txt     = txt if simplified else cc.convert(txt)
        subs.append((
            chunk_offset + start_s,
            chunk_offset + end_s,
            txt.strip(),
            spk,
        ))

    subs: list[tuple[float, float, str, "str | None"]] = []
    text_tokens = _tokenize_text(raw_text)

    if len(ts_list) != len(text_tokens):
        # 數量不對 → 放棄精確對齊
        return []

    seg_tok: list = []
    seg_wrd: list[str] = []
    char_count = 0

    for i, (fa_tok, word) in enumerate(zip(ts_list, text_tokens)):
        seg_tok.append(fa_tok)
        seg_wrd.append(word)
        char_count += len(word)

        # 是否在此切行
        last_ch = word[-1] if word else ''
        if last_ch in _ZH_CLAUSE_END or last_ch in _EN_SENT_END:
            _emit(seg_tok, seg_wrd)
            seg_tok = []
            seg_wrd = []
            char_count = 0
        elif char_count >= MAX_CHARS:
            _emit(seg_tok, seg_wrd)
            seg_tok = []
            seg_wrd = []
            char_count = 0

    _emit(seg_tok, seg_wrd)
    return subs


class CUDAEngine(ASREngineBase):
    """PyTorch CUDA/CPU 推理引擎。

    使用 qwen_asr 官方 API，支援 Qwen3-ASR-1.7B + ForcedAligner。
    """

    max_chunk_secs = 30   # PyTorch 模型沒有固定長度限制

    def __init__(self):
        super().__init__()
        self.model = None
        self.aligner = None
        self.use_aligner = False
        self.device = "cuda"

    def load(
        self,
        *,
        device: str = "cuda",
        model_dir: Path = None,
        cb=None,
        use_aligner: bool = True,
    ):
        """載入 PyTorch ASR 模型。"""
        import torch
        import onnxruntime as ort
        from qwen_asr import Qwen3ASRModel

        if model_dir is None:
            model_dir = GPU_MODEL_DIR

        asr_path = model_dir / ASR_MODEL_NAME
        aligner_path = model_dir / ALIGNER_MODEL_NAME

        def _s(msg):
            if cb:
                cb(msg)

        # ── VAD + Diarize + OpenCC ────────────────────────────────
        vad_path = _find_vad_model()
        if vad_path is None:
            raise FileNotFoundError(
                "找不到 Silero VAD 模型 (silero_vad_v4.onnx)。\n"
                f"請將模型放入 {GPU_MODEL_DIR} 或先執行 CPU 版本下載。"
            )
        _s("載入 VAD 模型…")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )
        self._load_diarization(OV_MODEL_DIR, cb=cb)
        self._load_opencc()

        # ── PyTorch ASR 模型 ──────────────────────────────────────
        _s(f"載入 ASR 模型（{asr_path.name}）…")
        if not asr_path.exists():
            raise FileNotFoundError(
                f"找不到 ASR 模型：{asr_path}\n"
                f"請將 {ASR_MODEL_NAME} 放入 {model_dir}"
            )

        self.device = device.lower()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        _s(f"編譯模型（{device.upper()}，{str(dtype).split('.')[-1]}）…")
        self.model = Qwen3ASRModel.from_pretrained(
            str(asr_path),
            device_map=self.device,
            dtype=dtype,
        )

        # ── ForcedAligner（可選）────────────────────────────────────
        self.aligner = None
        self.use_aligner = False
        if use_aligner and aligner_path.exists():
            try:
                _s(f"載入時間軸對齊模型（{ALIGNER_MODEL_NAME}）…")
                from qwen_asr import Qwen3ForcedAligner
                self.aligner = Qwen3ForcedAligner.from_pretrained(
                    str(aligner_path),
                    device_map=self.device,
                    dtype=dtype,
                )
                self.use_aligner = True
                _s(f"時間軸對齊模型就緒（{device.upper()}）")
            except Exception as _e:
                _s(f"⚠ ForcedAligner 載入失敗（{_e}），改用比例估算")
                self.aligner = None
                self.use_aligner = False

        self.ready = True
        aligner_info = "  + ForcedAligner" if self.use_aligner else ""
        _s(f"就緒（{device.upper()}  {ASR_MODEL_NAME}{aligner_info}）")

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """PyTorch 推理。"""
        with self._lock:
            results = self.model.transcribe(
                [(audio, SAMPLE_RATE)],
                language=language,
                context=context or "",
            )
            text = (results[0].text if results else "").strip()
            return text if self.output_simplified else self.cc.convert(text)

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language: str | None = None,
        context: str | None = None,
        diarize: bool = False,
        n_speakers: int | None = None,
    ) -> Path | None:
        """覆寫版本：支援 ForcedAligner 精確時間軸。"""
        from ffmpeg_utils import decode_audio_to_numpy, find_ffmpeg

        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            raise RuntimeError("找不到 ffmpeg，請重新啟動程式或手動安裝。")
        audio = decode_audio_to_numpy(audio_path, ffmpeg_exe, sr=SAMPLE_RATE)

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
            vad_groups = detect_speech_groups(audio, self.vad_sess)
            if not vad_groups:
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        all_subs: list[tuple[float, float, str, str | None]] = []
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total, f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")

            # ── ASR 轉錄 ──────────────────────────────────────────
            with self._lock:
                results = self.model.transcribe(
                    [(chunk, SAMPLE_RATE)],
                    language=language,
                    context=context or "",
                )
            raw_text = (results[0].text if results else "").strip()
            if not raw_text:
                continue

            # ── ForcedAligner ──────────────────────────────────────
            aligned = False
            if self.use_aligner and self.aligner is not None:
                try:
                    align_lang = language or "Chinese"
                    align_results = self.aligner.align(
                        audio=(chunk, SAMPLE_RATE),
                        text=raw_text,
                        language=align_lang,
                    )
                    ts_list = align_results[0] if align_results else []
                    if ts_list:
                        subs = _ts_to_subtitle_lines(
                            ts_list, raw_text, g0, spk,
                            self.cc, self.output_simplified
                        )
                        if subs:
                            all_subs.extend(subs)
                            aligned = True
                except Exception:
                    aligned = False

            if not aligned:
                # ── 比例估算 Fallback ──────────────────────────────
                text = (
                    raw_text if self.output_simplified
                    else self.cc.convert(raw_text)
                )
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
