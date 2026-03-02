"""
chatllm_engine.py — ChatLLM.cpp + Vulkan 推理後端

兩種執行模式：
  1. DLL 模式（優先）：ctypes 直接呼叫 libchatllm.dll，模型常駐記憶體
     - 每 chunk 約 0.23s（GPU shader 暖機後），免去 subprocess 啟動 overhead
  2. Subprocess 模式（後備）：每 chunk 啟動 main.exe 子程序
     - 模型每次重載，但不需要 DLL

輸出格式：language {lang}<asr_text>{transcription}
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ── 輸出語系旗標（由 app.py / app-gpu.py 切換時同步設定）──────────────
# True = 直接輸出模型原始簡體；False = 經 OpenCC s2twp 轉為繁體
_output_simplified: bool = False

# ── 共用常數與工具函式（統一在 asr_utils.py）────────────────────────
from asr_utils import (
    SAMPLE_RATE, VAD_CHUNK, VAD_THRESHOLD, MAX_GROUP_SEC,
    MAX_CHARS, MIN_SUB_SEC, GAP_SEC,
    LANG_CODE,
    detect_speech_groups, split_to_lines, srt_ts, assign_ts,
)

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Windows 旗標：建立子程序時不彈出主控台視窗（防止辨識時畫面閃爍）
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# STARTUPINFO：額外強制隱藏子程序視窗（搭配 CREATE_NO_WINDOW 雙重保護）
# CREATE_NO_WINDOW 阻止 console 分配，STARTF_USESHOWWINDOW+SW_HIDE 隱藏主視窗
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = None
if sys.platform == "win32":
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUP_INFO.wShowWindow = 0  # SW_HIDE


def _to_path_bytes(path: "str | Path") -> bytes:
    """路徑轉 bytes，適合傳給 Windows C DLL（ANSI API 相容）。

    Windows C 函式庫的 fopen() / LoadLibrary() 等 ANSI 函式期望系統碼頁
    （CP936/CP950）編碼的路徑，而 Python 預設 .encode() 是 UTF-8，
    兩者在中文路徑上不相容。

    解法優先順序：
      1. GetShortPathNameW → 8.3 短路徑（純 ASCII，任何 C API 都能處理）
      2. 若 8.3 短路徑仍含非 ASCII → 改用 GetACP() 系統碼頁編碼
      3. 最後回退 UTF-8
    """
    p = str(path)
    if sys.platform != "win32":
        return p.encode("utf-8")
    # 嘗試 GetShortPathNameW 取得 ASCII 8.3 短路徑
    try:
        n = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, n) > 0:
                try:
                    return buf.value.encode("ascii")
                except UnicodeEncodeError:
                    p = buf.value   # 短路徑仍有非 ASCII → 繼續往下
    except Exception:
        pass
    # 回退：系統 ANSI 碼頁（C fopen/CreateFileA 期望的編碼）
    try:
        cp = ctypes.windll.kernel32.GetACP()
        return p.encode(f"cp{cp}")
    except (UnicodeEncodeError, LookupError):
        return p.encode("utf-8")


def _short_path_str(path: "str | Path") -> str:
    """回傳 8.3 短路徑字串（盡量 ASCII，用於嵌入 DLL 訊息字串中）。"""
    p = str(path)
    if sys.platform != "win32":
        return p
    try:
        n = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, n) > 0:
                return buf.value
    except Exception:
        pass
    return p

# _LANG_CODE 已移至 asr_utils.LANG_CODE，保留別名供內部相容
_LANG_CODE = LANG_CODE

SRT_DIR = BASE_DIR / "subtitles"


# ══════════════════════════════════════════════════════
# Vulkan 裝置偵測
# ══════════════════════════════════════════════════════

def detect_vulkan_devices(chatllm_dir: str | Path) -> list[dict]:
    """執行 main.exe --show_devices，解析所有非 CPU 的計算裝置。

    輸出格式（每裝置兩行）：
      0: Vulkan - VulkanO (AMD Radeon(TM) Graphics)
         type: ACCEL
         memory free: 7957908736 B
      1: CPU - CPU (AMD Ryzen 5 9600X 6-Core Processor)
         type: CPU

    判斷邏輯：
      - 行首 backend 欄位（Vulkan/CPU 等）決定裝置類型
      - backend == "CPU" → 跳過；其餘（Vulkan, Metal, CUDA…）均列出
      - 不依賴 type: 行，避免 NVIDIA/AMD/Intel 格式差異

    回傳: [{'id': 0, 'name': 'AMD Radeon(TM) Graphics', 'vram_free': 7957908736}, ...]
    失敗時回傳空清單。
    """
    exe = Path(chatllm_dir) / "main.exe"
    if not exe.exists():
        return []
    try:
        result = subprocess.run(
            [str(exe), "--show_devices"],
            capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=10,
            cwd=str(chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = result.stdout + result.stderr
        pending: list[dict] = []   # 尚未確認 vram_free 的裝置
        current: dict | None = None

        for line in output.splitlines():
            # 裝置標頭行：「0: Vulkan - VulkanO (AMD Radeon(TM) Graphics)」
            m = re.match(r"\s*(\d+):\s*(\S+)\s+-\s+\S+\s+\((.+)\)", line)
            if m:
                backend = m.group(2).upper()   # "VULKAN", "CPU", "METAL" …
                current = {
                    "id":        int(m.group(1)),
                    "name":      m.group(3).strip(),
                    "vram_free": 0,
                    "_skip":     backend == "CPU",   # 只排除純 CPU 裝置
                }
                pending.append(current)
            elif "memory free" in line and current is not None:
                mf = re.search(r"(\d+)\s*B", line)
                if mf:
                    current["vram_free"] = int(mf.group(1))

        return [
            {"id": d["id"], "name": d["name"], "vram_free": d["vram_free"]}
            for d in pending if not d["_skip"]
        ]
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# main.exe 子程序包裝
# ══════════════════════════════════════════════════════

class _ChatLLMRunner:
    """
    以一次性模式執行 main.exe（每個音訊 chunk 一次呼叫）。

    使用 `-mgl main N` 而非 `-ngl N`：
      - Transformer 放 GPU（Vulkan 加速）
      - 音訊 encoder（FFmpeg + GGML audio）留在 CPU（Vulkan 不支援 audio encoder）

    輸出格式：language {lang}<asr_text>{transcription}
    """

    def __init__(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        device_id:    int = 0,
    ):
        self._model_path   = Path(model_path).resolve()   # 必須解析為絕對路徑
        self._chatllm_dir  = Path(chatllm_dir).resolve()
        self._n_gpu_layers = n_gpu_layers
        self._device_id    = device_id
        self._lock         = threading.Lock()

        exe = self._chatllm_dir / "main.exe"
        if not exe.exists():
            raise FileNotFoundError(f"main.exe 不存在：{exe}")
        self._exe = exe

        # 驗證：執行 --show 確認模型可載入
        # 注意：用 -ngl 0 驗證（不上 GPU），避免驗證步驟佔用顯存
        r = subprocess.run(
            [str(exe), "-m", str(self._model_path), "-ngl", "0",
             "--hide_banner", "--show"],
            capture_output=True, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            timeout=30, cwd=str(self._chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = r.stdout + r.stderr
        if "Qwen3-ASR" not in output:
            raise RuntimeError(f"模型驗證失敗（rc={r.returncode}）：{output[:300]}")

    def transcribe(self, wav_path: str, sys_prompt: str | None = None) -> str:
        """送入 WAV 路徑（絕對路徑），回傳轉錄文字。"""
        # -ngl all = -ngl 99999,prolog,epilog：把全部 layer（含 audio encoder Conv2D）放 GPU
        # 比 -mgl main N 快 2.7×（GPU 加速 audio encoder + Transformer 兩段）
        gpu_args = ["-ngl", "all"] if self._n_gpu_layers > 0 else ["-ngl", "0"]
        cmd = [
            str(self._exe),
            "-m",    str(self._model_path),
            *gpu_args,
            "--hide_banner",
            "-p",    wav_path,
        ]
        if sys_prompt:
            cmd += ["-s", sys_prompt]

        with self._lock:
            r = subprocess.run(
                cmd,
                capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=120, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUP_INFO,
            )
        output = r.stdout + r.stderr

        # 正常輸出必含 <asr_text>；若缺失代表裝置錯誤，立即中止，不回傳垃圾字幕
        if "<asr_text>" not in output:
            preview = output.strip()[:300] or "(無輸出)"
            raise RuntimeError(
                f"GPU 推理失敗，未取得語音輸出。\n"
                f"可能原因：裝置不相容、模型錯誤或記憶體不足。\n"
                f"chatllm 輸出：{preview}"
            )
        return output.split("<asr_text>", 1)[1].strip()


# ══════════════════════════════════════════════════════
# DLL 模式包裝（ctypes，模型常駐記憶體）
# ══════════════════════════════════════════════════════

class _DLLASRRunner:
    """
    libchatllm.dll ctypes 包裝，模型常駐 GPU 記憶體。

    每 chunk 呼叫 transcribe()：
      chatllm_restart → 寫 WAV → chatllm_user_input("{{audio:path}}")
      第一次因 Vulkan shader 編譯約 8s；後續 ~0.23s（43× 實時）
    """

    def __init__(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        device_id:    int = 0,
        cb=None,
    ):
        self._chatllm_dir = Path(chatllm_dir).resolve()
        self._lock        = threading.Lock()

        # ── 凍結視窗模式：預配置隱藏主控台（防止 DLL 閃出黑色視窗）────
        # 問題根源：--windowed PyInstaller EXE 沒有主控台，
        # libchatllm.dll 的 MSVC C runtime 每次呼叫 chatllm_restart() /
        # chatllm_user_input() 寫 stderr/stdout 時，發現 handle 無效，
        # 就會自行呼叫 AllocConsole() 建立主控台視窗（黑色視窗閃爍）。
        # 解法：在 LoadLibrary 前搶先 AllocConsole() 並立即隱藏，
        # 讓 DLL C runtime 找到合法 handle，不再自行建立可見視窗。
        # source 模式（python app.py）從 cmd.exe 繼承主控台，不觸發此問題。
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            _k32 = ctypes.windll.kernel32
            _u32 = ctypes.windll.user32
            if not _k32.GetConsoleWindow():          # 目前無主控台
                if _k32.AllocConsole():              # 分配一個
                    _hwnd = _k32.GetConsoleWindow()
                    if _hwnd:
                        _u32.ShowWindow(_hwnd, 0)    # SW_HIDE：立即隱藏

        dll_path = self._chatllm_dir / "libchatllm.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"libchatllm.dll 不存在：{dll_path}")

        # ── DLL 相依解析修復（PyInstaller EXE 關鍵）──────────────
        # libchatllm.dll 內部用 plain LoadLibrary("ggml-vulkan.dll")
        # （不帶 LOAD_LIBRARY_SEARCH_* 旗標），走傳統 DLL 搜尋順序：
        #   模組目錄 → CWD → System32 → PATH
        # AddDllDirectory()（os.add_dll_directory）只影響有旗標的 LoadLibraryEx，
        # 對傳統搜尋無效。EXE 的 CWD ≠ chatllm/，PATH 也不含 chatllm/，
        # 所以 ggml-vulkan.dll 等找不到 → DLL 初始化失敗 → fallback subprocess。
        # 解法：暫時把 chatllm_dir 插到 PATH 最前面，chatllm_start() 後還原。
        _saved_path = os.environ.get("PATH", "")
        _chatllm_dir_str = str(self._chatllm_dir)
        os.environ["PATH"] = _chatllm_dir_str + os.pathsep + _saved_path

        os.add_dll_directory(_chatllm_dir_str)
        lib = ctypes.windll.LoadLibrary(str(dll_path))

        # ── 函式原型 ─────────────────────────────────────────────
        PRINTFUNC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
        ENDFUNC   = ctypes.WINFUNCTYPE(None, ctypes.c_void_p)

        lib.chatllm_append_init_param.argtypes = [ctypes.c_char_p]
        lib.chatllm_append_init_param.restype  = None
        lib.chatllm_init.argtypes              = []
        lib.chatllm_init.restype               = ctypes.c_int
        lib.chatllm_create.argtypes            = []
        lib.chatllm_create.restype             = ctypes.c_void_p
        lib.chatllm_append_param.argtypes      = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_append_param.restype       = None
        lib.chatllm_start.argtypes             = [ctypes.c_void_p, PRINTFUNC, ENDFUNC, ctypes.c_void_p]
        lib.chatllm_start.restype              = ctypes.c_int
        lib.chatllm_restart.argtypes           = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_restart.restype            = None
        lib.chatllm_user_input.argtypes        = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_user_input.restype         = ctypes.c_int

        self._lib       = lib
        self._PRINTFUNC = PRINTFUNC
        self._ENDFUNC   = ENDFUNC

        # ── chatllm 全域初始化（--ggml_dir 告知後端 DLL 位置）────
        # 必須用 ANSI/ASCII 相容的路徑；DLL 的 C 函式庫用 ANSI fopen/LoadLibrary，
        # 若傳 UTF-8 中文路徑會找不到 ggml-*.dll，使用 _to_path_bytes() 解決此問題。
        lib.chatllm_append_init_param(b"--ggml_dir")
        lib.chatllm_append_init_param(_to_path_bytes(self._chatllm_dir))
        r = lib.chatllm_init()
        if r != 0:
            raise RuntimeError(f"chatllm_init() failed: {r}")

        # ── 建立 LLM object ───────────────────────────────────────
        chat = lib.chatllm_create()
        if not chat:
            raise RuntimeError("chatllm_create() returned NULL")
        self._chat = chat

        # ── 模型參數：必須加 --multimedia_file_tags {{ }} ─────────
        # 若缺少此參數，chat->history 的 mm_opening/closing 為空字串，
        # Content::push_back() 會把 {{audio:path}} 當純文字儲存，不做音訊解析。
        # 模型路徑同樣需要 ANSI/ASCII 相容編碼（GetShortPathNameW）。
        gpu_arg = "all" if n_gpu_layers > 0 else "0"
        model_path_bytes = _to_path_bytes(Path(model_path).resolve())
        for p_b in [
            b"-m", model_path_bytes,
            b"-ngl", gpu_arg.encode(),
            b"--multimedia_file_tags", b"{{", b"}}",
        ]:
            lib.chatllm_append_param(chat, p_b)

        # ── 回呼（必須存為 instance attribute 防止 GC 回收）────────
        self._chunks: list[str] = []
        self._error:  str | None = None

        @PRINTFUNC
        def on_print(user_data, print_type, s_ptr):
            text = s_ptr.decode("utf-8", errors="replace") if s_ptr else ""
            if print_type == 0:       # PRINT_CHAT_CHUNK
                self._chunks.append(text)
            elif print_type == 2:     # PRINTLN_ERROR
                self._error = text

        @ENDFUNC
        def on_end(user_data):
            pass

        self._on_print = on_print
        self._on_end   = on_end

        # ── 載入模型（Vulkan 全層 GPU）───────────────────────────
        if cb:
            cb("載入 chatllm 模型（Vulkan GPU，-ngl all）…")
        r = lib.chatllm_start(chat, on_print, on_end, ctypes.c_void_p(0))
        # chatllm_start() 後 DLL 已完全初始化，相依 DLL 也已載入記憶體，可還原 PATH
        os.environ["PATH"] = _saved_path
        if r != 0:
            raise RuntimeError(f"chatllm_start() failed: {r}")

    def transcribe(self, wav_path: str, sys_prompt: str | None = None) -> str:
        """送入 WAV 路徑（絕對路徑），回傳轉錄文字。"""
        # 取得 8.3 短路徑（ASCII），避免中文路徑無法被 DLL 的 C fopen 開啟
        # 例：C:\Users\陳小明\AppData\Local\Temp\xxx.wav
        #   → C:\Users\CHEN~1\AppData\Local\Temp\xxx.wav（純 ASCII）
        safe_path = _short_path_str(str(Path(wav_path).resolve()))
        fwd = safe_path.replace("\\", "/")
        # 若 _short_path_str 仍有非 ASCII（8.3 名稱停用），改用 ANSI 碼頁
        try:
            path_b = fwd.encode("ascii")
        except UnicodeEncodeError:
            cp = ctypes.windll.kernel32.GetACP() if sys.platform == "win32" else 65001
            path_b = fwd.encode(f"cp{cp}", errors="replace")
        msg = b"{{audio:" + path_b + b"}}"
        sys_bytes = sys_prompt.encode("utf-8") if sys_prompt else None

        with self._lock:
            self._lib.chatllm_restart(
                self._chat,
                ctypes.c_char_p(sys_bytes) if sys_bytes else ctypes.c_char_p(None),
            )
            self._chunks.clear()
            self._error = None

            r = self._lib.chatllm_user_input(self._chat, msg)

        if r != 0:
            raise RuntimeError(f"chatllm_user_input() failed: {r}")
        if self._error:
            raise RuntimeError(f"DLL 錯誤：{self._error}")

        full = "".join(self._chunks)
        if "<asr_text>" not in full:
            preview = full.strip()[:300] or "(無輸出)"
            raise RuntimeError(
                f"GPU 推理失敗，未取得語音輸出。\n"
                f"可能原因：裝置不相容、模型錯誤或記憶體不足。\n"
                f"DLL 輸出：{preview}"
            )
        return full.split("<asr_text>", 1)[1].strip()


