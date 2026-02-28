"""
engine/openvino_backend.py — OpenVINO INT8 推理後端

包含兩個引擎：
  - OpenVINO06BEngine : Qwen3-ASR-0.6B, stateful decoder
  - OpenVINO17BEngine : Qwen3-ASR-1.7B, KV-cache decoder
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import ASREngineBase, BASE_DIR

_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"


class OpenVINO06BEngine(ASREngineBase):
    """Qwen3-ASR-0.6B OpenVINO INT8 引擎（stateful decoder）。"""

    max_chunk_secs = 30

    def load(self, *, device: str = "CPU", model_dir: Path = None, cb=None):
        """載入 0.6B 模型。"""
        import openvino as ov
        from processor_numpy import LightProcessor

        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        ov_dir = model_dir / "qwen3_asr_int8"

        def _s(msg):
            if cb:
                cb(msg)

        # ── 共用：VAD + Diarize + OpenCC ────────────────────────────
        self._load_vad(model_dir, cb=cb)
        self._load_diarization(model_dir, cb=cb)
        self._load_opencc()

        # ── 編譯 ASR 模型 ──────────────────────────────────────────
        _s(f"編譯 ASR 模型（{device}）…")
        core = ov.Core()

        cache_dir = model_dir / "ov_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})

        self.audio_enc = core.compile_model(
            ov_dir / "audio_encoder_model.xml", device
        )
        self.embedder = core.compile_model(
            ov_dir / "thinker_embeddings_model.xml", device
        )
        dec_comp = core.compile_model(
            ov_dir / "decoder_model.xml", device
        )
        self.dec_req = dec_comp.create_infer_request()

        # ── Processor ──────────────────────────────────────────────
        _s("載入 Processor（純 numpy）…")
        self.processor = LightProcessor(ov_dir)
        self.pad_id = self.processor.pad_id
        self.ready = True
        _s(f"編譯完成（{device}）")

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """stateful decoder 自回歸推理。"""
        with self._lock:
            # ── 前處理（純 numpy）────────────────────────────────────
            mel, ids = self.processor.prepare(
                audio, language=language, context=context
            )

            # ── 音頻編碼 + 文字 Embedding ────────────────────────────
            ae = list(self.audio_enc({"mel": mel}).values())[0]
            te = list(self.embedder({"input_ids": ids}).values())[0]

            # ── 音頻特徵填入 pad 位置 ────────────────────────────────
            combined = te.copy()
            mask = ids[0] == self.pad_id
            np_ = int(mask.sum())
            na = ae.shape[1]
            if np_ != na:
                mn = min(np_, na)
                combined[0, np.where(mask)[0][:mn]] = ae[0, :mn]
            else:
                combined[0, mask] = ae[0]

            # ── Decoder 自回歸生成 ────────────────────────────────────
            L = combined.shape[1]
            pos = np.arange(L, dtype=np.int64)[np.newaxis, :]
            self.dec_req.reset_state()
            out = self.dec_req.infer({0: combined, "position_ids": pos})
            logits = list(out.values())[0]

            eos = self.processor.eos_id
            eot = self.processor.eot_id
            gen: list[int] = []
            nxt = int(np.argmax(logits[0, -1, :]))
            cur = L
            while nxt not in (eos, eot) and len(gen) < max_tokens:
                gen.append(nxt)
                emb = list(
                    self.embedder(
                        {"input_ids": np.array([[nxt]], dtype=np.int64)}
                    ).values()
                )[0]
                out = self.dec_req.infer(
                    {0: emb, "position_ids": np.array([[cur]], dtype=np.int64)}
                )
                logits = list(out.values())[0]
                nxt = int(np.argmax(logits[0, -1, :]))
                cur += 1

            # ── 解碼 ────────────────────────────────────────────────
            raw = self.processor.decode(gen)
            if "<asr_text>" in raw:
                raw = raw.split("<asr_text>", 1)[1]
            text = raw.strip()
            return text if self.output_simplified else self.cc.convert(text)


class OpenVINO17BEngine(ASREngineBase):
    """Qwen3-ASR-1.7B OpenVINO KV-cache 引擎。

    模型目錄：ov_models/qwen3_asr_1p7b_kv_int8/
      audio_encoder_model.xml       — mel(128,1000)  → audio_embeds(1,130,2048)
      thinker_embeddings_model.xml  — input_ids      → token_embeds
      decoder_prefill_kv_model.xml  — prefill pass   → logit + past_keys + past_vals
      decoder_kv_model.xml          — decode step    → logit + new_keys  + new_vals
    """

    _OV_SUBDIR = "qwen3_asr_1p7b_kv_int8"
    max_chunk_secs = 10   # audio_encoder 匯出固定 T=1000（10s）

    def __init__(self):
        super().__init__()
        self.pf_model = None   # compiled prefill model
        self.dc_model = None   # compiled decode-step model

    def load(self, *, device: str = "CPU", model_dir: Path = None, cb=None):
        """載入 1.7B 模型。"""
        import openvino as ov
        from processor_numpy import LightProcessor

        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        kv_dir = model_dir / self._OV_SUBDIR

        def _s(msg):
            if cb:
                cb(msg)

        # ── 共用：VAD + Diarize + OpenCC ────────────────────────────
        self._load_vad(model_dir, cb=cb)
        self._load_diarization(model_dir, cb=cb)
        self._load_opencc()

        # ── 編譯 ASR 模型 ──────────────────────────────────────────
        _s(f"編譯 1.7B ASR 模型（{device}）…")
        core = ov.Core()

        cache_dir = model_dir / "ov_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})

        self.audio_enc = core.compile_model(
            kv_dir / "audio_encoder_model.xml", device
        )
        self.embedder = core.compile_model(
            kv_dir / "thinker_embeddings_model.xml", device
        )
        self.pf_model = core.compile_model(
            kv_dir / "decoder_prefill_kv_model.xml", device
        )
        self.dc_model = core.compile_model(
            kv_dir / "decoder_kv_model.xml", device
        )

        # ── Processor ──────────────────────────────────────────────
        _s("載入 Processor（純 numpy，1.7B 10s）…")
        self.processor = LightProcessor(kv_dir)
        self.pad_id = self.processor.pad_id
        self.ready = True
        _s(f"1.7B 編譯完成（{device}）")

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 256,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """KV-cache 貪婪解碼：O(L²) prefill + O(n) decode。"""
        with self._lock:
            # 1. 前處理（10s 音訊）
            mel, ids = self.processor.prepare(
                audio, language=language, context=context
            )
            # audio_encoder 輸入 mel[0] 去除 batch dim → (128, 1000)
            ae = list(self.audio_enc({"mel": mel[0]}).values())[0]   # (1, 130, 2048)
            te = list(self.embedder({"input_ids": ids}).values())[0]  # (1, L, 2048)

            # 2. 合併音頻特徵
            combined = te.copy()
            mask = ids[0] == self.pad_id
            n_pad = int(mask.sum())
            n_ae = ae.shape[1]
            if n_pad != n_ae:
                mn = min(n_pad, n_ae)
                combined[0, np.where(mask)[0][:mn]] = ae[0, :mn]
            else:
                combined[0, mask] = ae[0]

            # 3. Prefill
            seq_len = combined.shape[1]
            pos_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
            pf_out = self.pf_model(
                {"input_embeds": combined, "position_ids": pos_ids}
            )
            pf_vals = list(pf_out.values())
            logits = pf_vals[0]    # (1, 1, vocab)
            past_k = pf_vals[1]    # (28, 1, 8, L, 128)
            past_v = pf_vals[2]

            eos = self.processor.eos_id
            eot = self.processor.eot_id
            next_tok = int(np.argmax(logits[0, -1, :]))
            if next_tok in (eos, eot):
                return ""

            gen = [next_tok]
            cur_pos = seq_len

            # 4. Decode loop
            for _ in range(max_tokens - 1):
                new_ids = np.array([[next_tok]], dtype=np.int64)
                new_emb = list(
                    self.embedder({"input_ids": new_ids}).values()
                )[0]
                new_pos = np.array([[cur_pos]], dtype=np.int64)

                dc_out = self.dc_model({
                    "new_embed":   new_emb,
                    "new_pos":     new_pos,
                    "past_keys":   past_k,
                    "past_values": past_v,
                })
                dc_vals = list(dc_out.values())
                logits = dc_vals[0]
                past_k = dc_vals[1]
                past_v = dc_vals[2]

                next_tok = int(np.argmax(logits[0, -1, :]))
                if next_tok in (eos, eot):
                    break
                gen.append(next_tok)
                cur_pos += 1

            # 5. 解碼
            raw = self.processor.decode(gen)
            if "<asr_text>" in raw:
                raw = raw.split("<asr_text>", 1)[1]
            text = raw.strip()
            return text if self.output_simplified else self.cc.convert(text)
