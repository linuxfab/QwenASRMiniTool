"""
asr_utils.py — 共用工具函式與常數

集中 app.py、app-gpu.py、chatllm_engine.py、streamlit_vulkan.py 的
重複程式碼，統一維護。

匯出：
  常數  : SAMPLE_RATE, VAD_CHUNK, VAD_THRESHOLD, MAX_GROUP_SEC,
          MAX_CHARS, MIN_SUB_SEC, GAP_SEC, RT_SILENCE_CHUNKS, RT_MAX_BUFFER_CHUNKS
  函式  : detect_speech_groups, split_to_lines, srt_ts, assign_ts
  類別  : RealtimeManager
"""
from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# ══════════════════════════════════════════════════════════════════════
# 共用常數
# ══════════════════════════════════════════════════════════════════════

SAMPLE_RATE          = 16000
VAD_CHUNK            = 512
VAD_THRESHOLD        = 0.5
MAX_GROUP_SEC        = 20
MAX_CHARS            = 20
MIN_SUB_SEC          = 0.6
GAP_SEC              = 0.08

RT_SILENCE_CHUNKS    = 25    # ~0.8s 靜音後觸發轉錄
RT_MAX_BUFFER_CHUNKS = 600   # ~19s 上限強制轉錄


# ══════════════════════════════════════════════════════════════════════
# VAD 分段
# ══════════════════════════════════════════════════════════════════════

def detect_speech_groups(
    audio: np.ndarray,
    vad_sess,
    max_group_sec: int = MAX_GROUP_SEC,
) -> list[tuple[float, float, np.ndarray]]:
    """Silero VAD 分段，回傳 [(start_s, end_s, chunk), ...]"""
    h  = np.zeros((2, 1, 64), dtype=np.float32)
    c  = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    n  = len(audio) // VAD_CHUNK
    probs = []
    for i in range(n):
        chunk = audio[i*VAD_CHUNK:(i+1)*VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c, "sr": sr})
        probs.append(float(out[0, 0]))
    if not probs:
        return [(0.0, len(audio) / SAMPLE_RATE, audio)]

    MIN_CH = 16; PAD = 5; MERGE = 16
    raw: list[tuple[int, int]] = []
    in_sp = False; s0 = 0
    for i, p in enumerate(probs):
        if p >= VAD_THRESHOLD and not in_sp:
            s0 = i; in_sp = True
        elif p < VAD_THRESHOLD and in_sp:
            if i - s0 >= MIN_CH:
                raw.append((max(0, s0-PAD), min(n, i+PAD)))
            in_sp = False
    if in_sp and n - s0 >= MIN_CH:
        raw.append((max(0, s0-PAD), n))
    if not raw:
        return []

    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= MERGE:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    mx_samp = max_group_sec * SAMPLE_RATE
    groups: list[tuple[int, int]] = []
    gs = merged[0][0] * VAD_CHUNK
    ge = merged[0][1] * VAD_CHUNK
    for seg in merged[1:]:
        s = seg[0] * VAD_CHUNK; e = seg[1] * VAD_CHUNK
        if e - gs > mx_samp:
            groups.append((gs, ge)); gs = s
        ge = e
    groups.append((gs, ge))

    result = []
    for gs, ge in groups:
        ns = max(1, int((ge - gs) // SAMPLE_RATE))
        ch = audio[gs: gs + ns * SAMPLE_RATE].astype(np.float32)
        if len(ch) < SAMPLE_RATE:
            continue
        result.append((gs / SAMPLE_RATE, gs / SAMPLE_RATE + ns, ch))
    return result


# ══════════════════════════════════════════════════════════════════════
# 字幕斷句
# ══════════════════════════════════════════════════════════════════════

# 中文、英文標點統一觸發切行（含英文逗號）
_PUNCT = frozenset('，。？！；：…—、.,!?;:')


def split_to_lines(text: str) -> list[str]:
    """以標點符號切分短句，移除標點，每句獨立成行。

    斷句規則（英文/中文統一）：
    1. 所有標點（,.!?;: 及中文，。？！；：…—）→ 立即切行，標點不輸出
    2. 英文整字為最小單位，詞前補空格（詞界）
    3. MAX_CHARS 保護：超限才強制換行
    """
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = text.strip()
    if not text:
        return []

    lines: list[str] = []
    buf   = ""

    i = 0
    while i < len(text):
        ch = text[i]

        # ── 標點符號：切行，標點不加入輸出（隱藏）────────────────────
        if ch in _PUNCT:
            if buf.strip():
                lines.append(buf.strip())
            buf = ""
            i += 1
            continue

        # ── 英文單字：整字收集，詞前補空格（詞界）────────────────────
        if ch.isalpha() and ord(ch) < 128:
            j = i
            while j < len(text) and text[j].isalpha() and ord(text[j]) < 128:
                j += 1
            word = text[i:j]
            prefix = " " if buf and not buf.endswith(" ") else ""
            if len(buf) + len(prefix) + len(word) > MAX_CHARS and buf.strip():
                lines.append(buf.strip())
                buf = word
            else:
                buf += prefix + word
            i = j
            continue

        # ── 空格：保留分詞間距 ────────────────────────────────────────
        if ch == " ":
            if buf and not buf.endswith(" "):
                buf += " "
            i += 1
            if len(buf.rstrip()) >= MAX_CHARS:
                lines.append(buf.strip())
                buf = ""
            continue

        # ── 中文/日文/數字等：逐字累積 ────────────────────────────────
        buf += ch
        i += 1
        if len(buf) >= MAX_CHARS:
            lines.append(buf.strip())
            buf = ""

    if buf.strip():
        lines.append(buf.strip())
    return [l for l in lines if l.strip()]


# ══════════════════════════════════════════════════════════════════════
# SRT 時間軸
# ══════════════════════════════════════════════════════════════════════

def srt_ts(s: float) -> str:
    """秒數轉 SRT 時間戳格式 HH:MM:SS,mmm"""
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def assign_ts(
    lines: list[str], g0: float, g1: float,
) -> list[tuple[float, float, str]]:
    """依字數比例分配時間軸。"""
    if not lines:
        return []
    total = sum(len(l) for l in lines)
    if total == 0:
        return []
    dur = g1 - g0; res = []; cur = g0
    for i, line in enumerate(lines):
        end = cur + max(MIN_SUB_SEC, dur * len(line) / total)
        if i == len(lines) - 1:
            end = max(end, g1)
        res.append((cur, end, line))
        cur = end + GAP_SEC
    return res


# ══════════════════════════════════════════════════════════════════════
# 即時轉錄管理員
# ══════════════════════════════════════════════════════════════════════

class RealtimeManager:
    """sounddevice 串流 + VAD + 緩衝轉錄。

    需要 ASR 引擎有以下屬性：
      - vad_sess : ONNX InferenceSession（Silero VAD）
      - transcribe(audio, max_tokens=..., language=..., context=...) -> str
      - max_chunk_secs : int（可選，預設 19）

    on_text 回呼簽名：on_text(text: str, start_sec: float, end_sec: float)
      start_sec / end_sec 為相對於錄音開始的真實秒數。
    """

    def __init__(
        self,
        asr,
        device_idx: int,
        on_text,
        on_status,
        language: str | None = None,
        context: str | None = None,
    ):
        self.asr       = asr
        self.dev_idx   = device_idx
        self.on_text   = on_text    # callback(text: str, start_sec: float, end_sec: float)
        self.on_status = on_status  # callback(msg: str)
        self.language  = language
        self.context   = context
        self._q        = queue.Queue()
        self._running  = False
        self._stream   = None

    def start(self):
        import sounddevice as sd
        self._running = True
        # 查詢裝置原生聲道數：立體聲混音等 loopback 裝置需要 2ch
        dev_info      = sd.query_devices(self.dev_idx, "input")
        self._native_ch = max(1, int(dev_info["max_input_channels"]))
        self._stream  = sd.InputStream(
            device=self.dev_idx,
            samplerate=SAMPLE_RATE,
            channels=self._native_ch,
            blocksize=VAD_CHUNK,
            dtype="float32",
            callback=self._audio_cb,
        )
        threading.Thread(target=self._loop, daemon=True).start()
        self._stream.start()
        self.on_status("🔴 錄音中…")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.on_status("⏹ 已停止")

    def _audio_cb(self, indata, frames, time_info, status):
        # 多聲道混音取平均轉 mono（立體聲混音 / WASAPI loopback 2ch）
        mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
        self._q.put(mono.copy())

    def _loop(self):
        h   = np.zeros((2, 1, 64), dtype=np.float32)
        c   = np.zeros((2, 1, 64), dtype=np.float32)
        sr  = np.array(SAMPLE_RATE, dtype=np.int64)
        buf: list[np.ndarray] = []
        sil = 0
        # 累計取樣數 → 真實時間（每收到一個 chunk +VAD_CHUNK）
        total_chunks   = 0     # 自錄音開始的總 chunk 數
        buf_start_chunk = 0    # 當前 buf 第一個 chunk 的全域位置

        while self._running:
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            total_chunks += 1

            out, h, c = self.asr.vad_sess.run(
                None,
                {"input": chunk[np.newaxis, :].astype(np.float32), "h": h, "c": c, "sr": sr},
            )
            prob = float(out[0, 0])

            if prob >= VAD_THRESHOLD:
                if not buf:
                    buf_start_chunk = total_chunks - 1
                buf.append(chunk); sil = 0
            elif buf:
                buf.append(chunk); sil += 1
                rt_max_buf = int(getattr(self.asr, "max_chunk_secs", 19) * SAMPLE_RATE / VAD_CHUNK)
                if sil >= RT_SILENCE_CHUNKS or len(buf) >= rt_max_buf:
                    audio = np.concatenate(buf)
                    n = max(1, len(audio) // SAMPLE_RATE) * SAMPLE_RATE
                    # 計算真實時間軸
                    start_sec = buf_start_chunk * VAD_CHUNK / SAMPLE_RATE
                    end_sec   = start_sec + n / SAMPLE_RATE
                    _max_tok = 400 if self.language == "Japanese" else 300
                    try:
                        text = self.asr.transcribe(
                            audio[:n],
                            max_tokens=_max_tok,
                            language=self.language,
                            context=self.context,
                        )
                        if text:
                            self.on_text(text, start_sec, end_sec)
                    except Exception as _e:
                        self.on_status(f"⚠ 轉錄錯誤：{_e}")
                    buf = []; sil = 0
                    h = np.zeros((2, 1, 64), dtype=np.float32)
                    c = np.zeros((2, 1, 64), dtype=np.float32)
