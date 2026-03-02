"""
Qwen3 ASR 字幕生成器 - CustomTkinter 前端

功能：
  1. 音檔上傳 → SRT 字幕（支援 OpenVINO CPU / GPU）
  2. 即時轉換：偵測音訊輸入裝置，邊說邊顯示字幕
"""
from __future__ import annotations

# ── UTF-8 模式：在所有其他 import 之前設定 ────────────────────────────
# 解決 Traditional Chinese Windows（cp950）上第三方套件用系統預設編碼
# 讀取 UTF-8 檔案時出現 "utf-8 codec can't decode byte 0xa6" 的問題。
# PYTHONUTF8=1 等效於 `python -X utf8`，讓所有 open() 預設使用 UTF-8。
import os as _os, sys as _sys, io as _io
_os.environ.setdefault("PYTHONUTF8", "1")
# 同步修正 stdout/stderr（避免 print 中文在 cp950 console 出錯）
for _stream_name in ("stdout", "stderr"):
    _s = getattr(_sys, _stream_name)
    if hasattr(_s, "buffer") and _s.encoding.lower() not in ("utf-8", "utf8"):
        setattr(_sys, _stream_name,
                _io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace"))
del _os, _sys, _io, _stream_name, _s

import json
import os
import re
import sys
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk

# ── 統一引擎模組 ──────────────────────────────────────────────────────
from engine import create_engine

# ── chatllm Vulkan 裝置偵測（可選）────────────────────────────────────
try:
    from chatllm_engine import detect_vulkan_devices
    _CHATLLM_AVAILABLE = True
except Exception:
    _CHATLLM_AVAILABLE = False
    def detect_vulkan_devices(_): return []

# ── 路徑 ──────────────────────────────────────────────
# PyInstaller 凍結時，模型應放在 EXE 旁邊（非 _internal/）
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"
from config import AppSettings
SRT_DIR            = BASE_DIR / "subtitles"
_CHATLLM_DIR       = BASE_DIR / "chatllm"
# .bin 優先找 ov_models/（開發期），再找 GPUModel/（打包後下載位置）
_BIN_PATH          = next(
    (p for p in [
        BASE_DIR / "ov_models"  / "qwen3-asr-1.7b.bin",
        BASE_DIR / "GPUModel"   / "qwen3-asr-1.7b.bin",
    ] if p.exists()),
    BASE_DIR / "GPUModel" / "qwen3-asr-1.7b.bin",  # 預設（未下載時）
)
SRT_DIR.mkdir(exist_ok=True)

# ── 共用常數與工具函式（統一在 asr_utils.py）────────────────────────
from asr_utils import (
    SAMPLE_RATE, VAD_CHUNK, VAD_THRESHOLD, MAX_GROUP_SEC,
    MAX_CHARS, MIN_SUB_SEC, GAP_SEC,
    RT_SILENCE_CHUNKS, RT_MAX_BUFFER_CHUNKS,
    detect_speech_groups, split_to_lines, srt_ts, assign_ts,
    RealtimeManager,
)




# ══════════════════════════════════════════════════════
# （ASREngine / ASREngine1p7B 已搬移至 engine/ 模組）
# ══════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_MONO  = ("Consolas", 12)
FONT_TITLE = ("Microsoft JhengHei", 22, "bold")


# ══════════════════════════════════════════════════════════════════════
# 字幕驗證 & 編輯視窗（共用模組 subtitle_editor.py）
# ══════════════════════════════════════════════════════════════════════
from subtitle_editor import SubtitleDetailEditor, SubtitleEditorWindow  # noqa: F401


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Qwen3 ASR 字幕生成器")
        self.geometry("960x700")
        self.minsize(800, 580)

        self.engine       = None
        self._rt_mgr: RealtimeManager | None = None
        self._rt_log: list[tuple[str, float, float]] = []   # (text, start_sec, end_sec)
        self._rt_lock                        = threading.Lock()  # 保護 _rt_log
        self._audio_file: Path | None        = None
        self._srt_output: Path | None        = None
        self._converting                     = False
        self._convert_lock                   = threading.Lock()  # 保護 _converting
        self._dev_idx_map: dict[str, int]    = {}
        self._model_dir: Path | None         = None   # 使用者選定的模型路徑
        self._lang_list: list[str]           = []     # 載入後填入
        self._selected_language: str | None  = None   # 目前選定的語系
        self._settings: dict                 = {}     # 目前生效的設定
        self._cfg                            = AppSettings()   # 統一設定讀寫
        self._all_devices: dict              = {}     # 偵測到的所有裝置
        self._file_hint: str | None          = None   # 音檔轉字幕 hint
        self._file_diarize: bool             = False  # 說話者分離開關
        self._file_n_speakers: int | None    = None   # 指定說話者人數（None=自動）

        self._build_ui()
        self._detect_all_devices()
        self._refresh_audio_devices()   # 音訊裝置獨立初始化，不依賴模型載入
        threading.Thread(target=self._startup_check, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 建構 ────────────────────────────────────────

    def _build_ui(self):
        # 標題列
        title_bar = ctk.CTkFrame(self, height=54, corner_radius=0)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar, text="  🎙 Qwen3 ASR 字幕生成器",
            font=FONT_TITLE, anchor="w"
        ).pack(side="left", padx=16, pady=8)

        # 裝置選擇列
        dev_bar = ctk.CTkFrame(self, height=46)
        dev_bar.pack(fill="x", padx=10, pady=(6, 0))
        dev_bar.pack_propagate(False)

        ctk.CTkLabel(dev_bar, text="模型：", font=FONT_BODY).pack(
            side="left", padx=(14, 4), pady=12
        )
        self.model_var   = ctk.StringVar(value="Qwen3-ASR-0.6B")
        self.model_combo = ctk.CTkComboBox(
            dev_bar,
            values=["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B INT8"],
            variable=self.model_var,
            width=160, state="readonly", font=FONT_BODY,
        )
        self.model_combo.pack(side="left", pady=12)

        ctk.CTkLabel(dev_bar, text="推理裝置：", font=FONT_BODY).pack(
            side="left", padx=(12, 4), pady=12
        )
        self.device_var   = ctk.StringVar(value="CPU")
        self.device_combo = ctk.CTkComboBox(
            dev_bar, values=["CPU"], variable=self.device_var,
            width=110, state="disabled", font=FONT_BODY,
        )
        self.device_combo.pack(side="left", pady=12)

        self.reload_btn = ctk.CTkButton(
            dev_bar, text="重新載入", width=90, state="disabled",
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._on_reload_models,
        )
        self.reload_btn.pack(side="left", padx=8, pady=12)

        ctk.CTkLabel(dev_bar, text="語系：", font=FONT_BODY).pack(
            side="left", padx=(12, 2), pady=12
        )
        self.lang_var   = ctk.StringVar(value="自動偵測")
        self.lang_combo = ctk.CTkComboBox(
            dev_bar, values=["自動偵測"], variable=self.lang_var,
            width=130, state="disabled", font=FONT_BODY,
        )
        self.lang_combo.pack(side="left", pady=12)

        self.status_dot = ctk.CTkLabel(
            dev_bar, text="⏳ 啟動中…",
            font=FONT_BODY, text_color="#AAAAAA", anchor="w"
        )
        self.status_dot.pack(side="left", padx=12, pady=12)

        # 下載進度條（正常情況下隱藏）
        self.dl_bar = ctk.CTkProgressBar(dev_bar, width=200, height=12)
        self.dl_bar.set(0)
        # 啟動時不 pack，由 _show_dl_bar / _hide_dl_bar 控制

        # 分頁
        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        self.tabs.add("  音檔轉字幕  ")
        self.tabs.add("  批次辨識  ")
        self.tabs.add("  即時轉換  ")
        self.tabs.add("  設定  ")

        self._build_file_tab(self.tabs.tab("  音檔轉字幕  "))
        self._build_batch_tab(self.tabs.tab("  批次辨識  "))
        self._build_rt_tab(self.tabs.tab("  即時轉換  "))

        from setting import SettingsTab
        self._settings_tab = SettingsTab(
            self.tabs.tab("  設定  "), self,
            # 編譯版（frozen）不含 streamlit，服務 tab 僅在開發模式顯示
            show_service=not getattr(sys, "frozen", False))
        self._settings_tab.pack(fill="both", expand=True)

    # ── 批次辨識 tab ───────────────────────────────────

    def _build_batch_tab(self, parent):
        from batch_tab import BatchTab
        tab_frame = ctk.CTkFrame(parent, fg_color="transparent")
        tab_frame.pack(fill="both", expand=True)
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)
        self._batch_tab = BatchTab(
            tab_frame,
            engine=None,   # 引擎於模型載入完成後注入（_on_models_ready）
            open_subtitle_cb=lambda srt, audio, dz:
                SubtitleEditorWindow(self, srt, audio, dz),
        )
        self._batch_tab.grid(row=0, column=0, sticky="nsew")

    # ── 音檔轉字幕 tab ─────────────────────────────────

    def _build_file_tab(self, parent):
        # 選檔列
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", padx=8, pady=(12, 4))

        self.file_entry = ctk.CTkEntry(
            row1, placeholder_text="選擇或拖曳音訊檔案…",
            font=FONT_BODY, height=34,
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(
            row1, text="瀏覽…", width=80, height=34, font=FONT_BODY,
            command=self._on_browse,
        ).pack(side="left")

        # 操作按鈕列
        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=4)

        self.convert_btn = ctk.CTkButton(
            row2, text="▶  開始轉換", width=130, height=36,
            font=FONT_BODY, state="disabled",
            command=self._on_convert,
        )
        self.convert_btn.pack(side="left", padx=(0, 10))

        self.open_dir_btn = ctk.CTkButton(
            row2, text="📁  開啟輸出資料夾", width=150, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="gray35", hover_color="gray25",
            command=lambda: os.startfile(str(SRT_DIR)),
        )
        self.open_dir_btn.pack(side="left")

        self.verify_btn = ctk.CTkButton(
            row2, text="🔍  字幕驗證", width=120, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="#1A3050", hover_color="#265080",
            command=self._on_verify,
        )
        self.verify_btn.pack(side="left", padx=(8, 0))

        self._diarize_var = ctk.BooleanVar(value=False)
        self.diarize_chk = ctk.CTkCheckBox(
            row2, text="說話者分離", variable=self._diarize_var,
            font=FONT_BODY, state="disabled",
            command=self._on_diarize_toggle,
        )
        self.diarize_chk.pack(side="left", padx=(20, 0))

        ctk.CTkLabel(row2, text="人數：", font=FONT_BODY,
                     text_color="#AAAAAA").pack(side="left", padx=(8, 2))
        self.n_spk_combo = ctk.CTkComboBox(
            row2,
            values=["自動", "2", "3", "4", "5", "6", "7", "8"],
            width=76, state="disabled", font=FONT_BODY,
        )
        self.n_spk_combo.set("自動")
        self.n_spk_combo.pack(side="left")

        # 辨識提示（Hint / Context）
        hint_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hint_hdr.pack(fill="x", padx=8, pady=(6, 0))
        # 右側按鈕要在左側標籤之前 pack，才能正確定位
        ctk.CTkButton(
            hint_hdr, text="讀入 TXT…", width=100, height=26,
            font=("Microsoft JhengHei", 11),
            fg_color="gray35", hover_color="gray25",
            command=lambda: self._load_hint_txt(self.hint_box),
        ).pack(side="right")
        ctk.CTkLabel(
            hint_hdr, text="辨識提示（可選）：", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            hint_hdr,
            text="貼入歌詞、關鍵字或背景說明，可提升辨識準確度",
            font=("Microsoft JhengHei", 11),
            text_color="#555555",
        ).pack(side="left", padx=(6, 0))

        self.hint_box = ctk.CTkTextbox(
            parent, font=FONT_MONO, height=72,
        )
        self.hint_box.pack(fill="x", padx=8, pady=(2, 4))
        self._bind_ctx_menu(self.hint_box._textbox, is_text=True)

        # 進度
        prog_frame = ctk.CTkFrame(parent, fg_color="transparent")
        prog_frame.pack(fill="x", padx=8, pady=(4, 2))

        self.prog_label = ctk.CTkLabel(
            prog_frame, text="", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        )
        self.prog_label.pack(fill="x")

        self.prog_bar = ctk.CTkProgressBar(prog_frame, height=10)
        self.prog_bar.pack(fill="x", pady=(2, 0))
        self.prog_bar.set(0)

        # 記錄
        ctk.CTkLabel(
            parent, text="轉換記錄", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.file_log = ctk.CTkTextbox(
            parent, font=FONT_MONO, state="disabled",
        )
        self.file_log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── 即時轉換 tab ───────────────────────────────────

    def _build_rt_tab(self, parent):
        # 裝置選擇列
        dev_row = ctk.CTkFrame(parent, fg_color="transparent")
        dev_row.pack(fill="x", padx=8, pady=(12, 4))

        ctk.CTkLabel(dev_row, text="音訊輸入裝置：", font=FONT_BODY).pack(
            side="left", padx=(0, 8)
        )
        self.rt_dev_combo = ctk.CTkComboBox(
            dev_row, values=["偵測中…"], width=380, font=FONT_BODY,
        )
        self.rt_dev_combo.pack(side="left")

        ctk.CTkButton(
            dev_row, text="重新整理", width=80, height=30,
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._refresh_audio_devices,
        ).pack(side="left", padx=8)

        # Hint 輸入列（即時模式）
        hint_row = ctk.CTkFrame(parent, fg_color="transparent")
        hint_row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(hint_row, text="辨識提示：", font=FONT_BODY,
                     text_color="#AAAAAA").pack(side="left", padx=(0, 6))
        # 右側按鈕先 pack
        ctk.CTkButton(
            hint_row, text="讀入 TXT…", width=90, height=26,
            font=("Microsoft JhengHei", 11),
            fg_color="gray35", hover_color="gray25",
            command=lambda: self._load_hint_txt(self.rt_hint_entry, is_textbox=False),
        ).pack(side="right")
        self.rt_hint_entry = ctk.CTkEntry(
            hint_row,
            placeholder_text="（可選）貼入歌詞、關鍵字或說明文字…",
            font=FONT_BODY, height=30,
        )
        self.rt_hint_entry.pack(side="left", fill="x", expand=True)
        self._bind_ctx_menu(self.rt_hint_entry._entry, is_text=False)

        # 控制按鈕列
        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=4)

        self.rt_start_btn = ctk.CTkButton(
            btn_row, text="▶  開始錄音", width=130, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="#2E7D32", hover_color="#1B5E20",
            command=self._on_rt_start,
        )
        self.rt_start_btn.pack(side="left", padx=(0, 10))

        self.rt_stop_btn = ctk.CTkButton(
            btn_row, text="■  停止錄音", width=130, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="#C62828", hover_color="#B71C1C",
            command=self._on_rt_stop,
        )
        self.rt_stop_btn.pack(side="left", padx=(0, 14))

        self.rt_status_lbl = ctk.CTkLabel(
            btn_row, text="", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        )
        self.rt_status_lbl.pack(side="left")

        ctk.CTkLabel(
            btn_row,
            text="（會在說話停頓中處理辨識）",
            font=("Microsoft JhengHei", 11),
            text_color="#666666",
        ).pack(side="left", padx=(12, 0))

        # 字幕顯示
        ctk.CTkLabel(
            parent, text="即時字幕", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.rt_textbox = ctk.CTkTextbox(
            parent, font=("Microsoft JhengHei", 15), state="disabled",
        )
        self.rt_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        # 操作列
        act_row = ctk.CTkFrame(parent, fg_color="transparent")
        act_row.pack(fill="x", padx=8, pady=(0, 10))

        ctk.CTkButton(
            act_row, text="清除", width=80, height=32,
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._on_rt_clear,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            act_row, text="💾  儲存 SRT", width=120, height=32,
            font=FONT_BODY, command=self._on_rt_save,
        ).pack(side="left")

    # ── 說話者分離 UI 輔助 ───────────────────────────────────────────

    def _on_diarize_toggle(self):
        """說話者分離 checkbox 切換時，同步啟用／停用人數選擇器。"""
        state = "readonly" if self._diarize_var.get() else "disabled"
        self.n_spk_combo.configure(state=state)

    # ── Hint 輸入輔助 ─────────────────────────────────────────────────

    def _bind_ctx_menu(self, native_widget, is_text: bool = False):
        """為原生 tkinter widget 綁定右鍵貼上選單（支援 Text 與 Entry）。"""
        def show(event):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(
                label="貼上",
                command=lambda: native_widget.event_generate("<<Paste>>"),
            )
            if is_text:
                menu.add_command(
                    label="全選",
                    command=lambda: native_widget.tag_add("sel", "1.0", "end"),
                )
                menu.add_separator()
                menu.add_command(
                    label="清除全部",
                    command=lambda: native_widget.delete("1.0", "end"),
                )
            else:
                menu.add_command(
                    label="全選",
                    command=lambda: native_widget.select_range(0, "end"),
                )
                menu.add_separator()
                menu.add_command(
                    label="清除全部",
                    command=lambda: native_widget.delete(0, "end"),
                )
            menu.tk_popup(event.x_root, event.y_root)
        native_widget.bind("<Button-3>", show)

    def _load_hint_txt(self, target, is_textbox: bool = True):
        """開啟 TXT 檔案，將內容填入 hint 輸入框。
        target     : CTkTextbox（is_textbox=True）或 CTkEntry（is_textbox=False）
        """
        path = filedialog.askopenfilename(
            title="選擇提示文字檔",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="cp950", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                messagebox.showerror("讀取失敗", str(e))
                return
        if is_textbox:
            target.delete("1.0", "end")
            target.insert("1.0", text)
        else:
            target.delete(0, "end")
            target.insert(0, text)

    def _refresh_model_combo(self, model_dir: Path):
        """主執行緒：固定顯示所有模型選項（下載邏輯由 _load_models 處理）。"""
        available = ["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B INT8"]
        self.model_combo.configure(values=available)
        if self.model_var.get() not in available:
            self.model_var.set(available[0])

    def _refresh_model_combo_from_settings(self, settings: dict):
        """主執行緒：依 settings.backend 顯示對應的模型 combo 狀態。"""
        backend = settings.get("backend", "openvino")
        if backend == "chatllm":
            self.model_combo.configure(
                values=["1.7B Q8_0 (Vulkan GPU)"], state="disabled"
            )
            self.model_var.set("1.7B Q8_0 (Vulkan GPU)")
        else:
            sz = settings.get("cpu_model_size", "0.6B")
            self.model_combo.configure(
                values=["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B INT8"],
                state="readonly",
            )
            self.model_var.set(
                "Qwen3-ASR-1.7B INT8" if sz == "1.7B" else "Qwen3-ASR-0.6B"
            )

    def _detect_all_devices(self):
        """同時偵測 OpenVINO（CPU / Intel iGPU）與 Vulkan（NVIDIA / AMD）裝置。
        結果儲存在 self._all_devices，並更新 device_combo 選單。
        """
        # ── OpenVINO 裝置 ───────────────────────────────────────────────
        ov_labels = ["CPU"]
        igpu_list: list[dict] = []
        try:
            import openvino as ov
            core = ov.Core()
            for d in core.available_devices:
                if not d.startswith("GPU"):
                    continue
                try:
                    name = core.get_property(d, "FULL_DEVICE_NAME")
                except Exception:
                    name = d
                if "Intel" in name:
                    label = f"{d} ({name})"
                    ov_labels.append(label)
                    igpu_list.append({"device": d, "name": name, "label": label})
        except Exception:
            pass

        # ── Vulkan 裝置（NVIDIA / AMD）──────────────────────────────────
        nvidia_amd: list[dict] = []
        if _CHATLLM_AVAILABLE:
            chatllm_dir = str(_CHATLLM_DIR)
            if not _CHATLLM_DIR.exists():
                # 嘗試 chatllmtest 目錄（開發模式）
                chatllm_dir = str(BASE_DIR / "chatllmtest" / "chatllm_win_x64" / "bin")
            nvidia_amd = detect_vulkan_devices(chatllm_dir)

        self._all_devices = {
            "cpu":       True,
            "igpu":      igpu_list,
            "nvidia_amd": nvidia_amd,
        }

        # ── 更新 device_combo ────────────────────────────────────────────
        all_labels = list(ov_labels)
        for dev in nvidia_amd:
            all_labels.append(f"GPU:{dev['id']} ({dev['name']}) [Vulkan]")

        self.device_combo.configure(values=all_labels)
        self.device_var.set(all_labels[0])

    # ── 設定檔讀寫（記住模型路徑）──────────────────────────────────────

    def _load_settings(self) -> dict:
        return self._cfg.load()

    def _save_settings(self, settings: dict):
        """儲存完整設定 dict 到 settings.json。
        schema:
          backend       : "openvino" | "chatllm"
          device        : "CPU" | "GPU.0 (Intel UHD...)" | "GPU:0 (NVIDIA...) [Vulkan]"
          cpu_model_size: "0.6B" | "1.7B"
          model_dir     : OpenVINO 模型資料夾
          model_path    : chatllm .bin 模型路徑（chatllm 後端用）
          chatllm_dir   : chatllm DLL 目錄
        """
        self._cfg.save(settings)

    def _patch_setting(self, key: str, value):
        """讀取現有設定、更新單一 key，再寫回。"""
        self._cfg.patch(key, value)

    def _apply_ui_prefs(self, settings: dict):
        """主執行緒：根據儲存的偏好設定同步 UI 控件與外觀。"""
        global VAD_THRESHOLD
        mode = settings.get("appearance_mode", "dark")
        ctk.set_appearance_mode(mode)
        # VAD 閾值：從設定還原
        vad = settings.get("vad_threshold")
        if vad is not None:
            VAD_THRESHOLD = float(vad)
        if hasattr(self, "_settings_tab"):
            self._settings_tab.sync_prefs(settings)

    def _on_chinese_mode_change(self, value: str):
        """輸出模式切換：繁體（OpenCC）or 簡體（直接輸出）。"""
        is_simplified = (value == "簡體")
        if self.engine:
            self.engine.output_simplified = is_simplified
        self._patch_setting("output_simplified", is_simplified)

    def _on_appearance_change(self, value: str):
        """主題切換：深色 🌑 or 淺色 ☀。"""
        mode = "light" if value == "☀" else "dark"
        ctk.set_appearance_mode(mode)
        self._patch_setting("appearance_mode", mode)

    def _settings_valid(self, s: dict) -> bool:
        """檢查設定是否足夠完整（不需要重新引導）。"""
        if not s:
            return False
        backend = s.get("backend", "")
        if backend == "chatllm":
            mdl  = s.get("model_path", "") or s.get("gguf_path", "")
            cdir = s.get("chatllm_dir", "")
            return bool(mdl and cdir and Path(mdl).exists() and Path(cdir).exists())
        elif backend == "openvino":
            model_dir = s.get("model_dir", "")
            if not model_dir:
                return False
            # 至少 0.6B 必須存在
            from downloader import quick_check
            return quick_check(Path(model_dir))
        return False

    def _resolve_model_dir(self) -> Path | None:
        """
        依序檢查：
          1. 預設 portable 路徑（EXE 旁邊的 ov_models/）
          2. settings.json 記住的路徑
        回傳第一個模型完整的路徑，或 None（需要詢問使用者）。
        """
        from downloader import quick_check
        if quick_check(_DEFAULT_MODEL_DIR):
            return _DEFAULT_MODEL_DIR
        saved = self._load_settings().get("model_dir")
        if saved:
            p = Path(saved)
            if quick_check(p):
                return p
        return None

    # ── 啟動檢查：設定有效 → 直接載入；否則 → 引導畫面 ────────────────

    def _startup_check(self):
        """背景執行緒：確認設定有效性 → 必要時顯示引導畫面 → 載入模型。"""
        settings = self._load_settings()

        if not self._settings_valid(settings):
            # 顯示引導畫面（主執行緒）
            chosen = [None]
            evt = threading.Event()
            self.after(0, lambda: self._run_onboarding(chosen, evt))
            evt.wait()

            if chosen[0] is None:
                # 使用者取消 → 嘗試 CPU + 0.6B 預設值
                default_dir = _DEFAULT_MODEL_DIR
                from downloader import quick_check
                if quick_check(default_dir):
                    settings = {
                        "backend":        "openvino",
                        "device":         "CPU",
                        "cpu_model_size": "0.6B",
                        "model_dir":      str(default_dir),
                    }
                else:
                    self.after(0, lambda: self._set_status("⚠ 已取消，模型未載入"))
                    return
            else:
                settings = chosen[0]

            self._save_settings(settings)

        self._settings = settings

        # 套用 simplied/traditional 偏好
        self._output_simplified = settings.get("output_simplified", False)
        self.after(0, lambda s=settings: self._apply_ui_prefs(s))

        # 同步 device_combo 到已儲存的裝置
        saved_dev = settings.get("device", "CPU")
        def _sync_device():
            vals = self.device_combo.cget("values")
            if saved_dev in vals:
                self.device_var.set(saved_dev)
        self.after(0, _sync_device)

        # 更新模型選單
        self.after(0, lambda: self._refresh_model_combo_from_settings(settings))

        self._set_status("⏳ 模型載入中…")
        self._load_models()

    # ── 引導畫面：硬體偵測 + 後端選擇 + 下載 ────────────────────────────

    def _run_onboarding(self, chosen: list, evt: threading.Event):
        """主執行緒：顯示初始設定引導畫面（modal）。
        chosen[0] = 選定設定 dict（或 None 表示取消）。
        """
        from onboarding import run_onboarding
        run_onboarding(
            self, chosen, evt,
            all_devices=self._all_devices,
            default_model_dir=_DEFAULT_MODEL_DIR,
            bin_path=_BIN_PATH,
            chatllm_dir=_CHATLLM_DIR,
            set_status=self._set_status,
            load_settings=self._load_settings,
        )

    def _on_dl_progress(self, pct: float, msg: str):
        self.after(0, lambda: self.dl_bar.set(pct))
        self.after(0, lambda: self._set_status(f"⬇ {msg} ({pct*100:.0f}%)"))

    def _show_dl_bar(self):
        self.dl_bar.pack(side="left", padx=(0, 8), pady=12)

    def _hide_dl_bar(self):
        self.dl_bar.pack_forget()

    def _load_models(self):
        import gc

        # ── 釋放舊引擎記憶體 ───────────────────────────────────────────
        if self.engine is not None:
            for attr in ("audio_enc", "embedder", "dec_req", "vad_sess",
                         "pf_model", "dc_model", "_llm"):
                if hasattr(self.engine, attr):
                    setattr(self.engine, attr, None)
        gc.collect()

        # ── 讀取設定：先用儲存的，再 fallback 至 UI 選擇 ───────────────
        settings       = self._settings or self._load_settings()
        backend        = settings.get("backend", "openvino")
        device_label   = settings.get("device", self.device_var.get())
        # 解析 OV 裝置名（如 "GPU.0 (Intel...)" → "GPU.0"）
        ov_device      = device_label.split(" (")[0].split(" [")[0]

        if backend == "chatllm":
            # ── chatllm / Vulkan 路線 ──────────────────────────────────
            if not _CHATLLM_AVAILABLE:
                self.after(0, lambda: self._on_models_failed(
                    "chatllm", "chatllm_engine 無法載入，請確認 chatllm/ 目錄"
                ))
                return

            # 向下相容：新 key=model_path，舊 key=gguf_path
            _saved_mdl  = settings.get("model_path") or settings.get("gguf_path") or str(_BIN_PATH)
            model_path  = Path(_saved_mdl)
            chatllm_dir = Path(settings.get("chatllm_dir", str(_CHATLLM_DIR)))

            # chatllm .bin 是否存在
            if not model_path.exists():
                self.after(0, self._show_dl_bar)
                self._set_status("⬇ 下載 chatllm 模型（~2.3 GB）…")
                try:
                    import urllib.request
                    from downloader import _ssl_ctx
                    url = ("https://huggingface.co/dseditor/Collection"
                           "/resolve/main/qwen3-asr-1.7b.bin")
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0 (compatible; QwenASR)"}
                    )
                    with urllib.request.urlopen(req, context=_ssl_ctx()) as resp, \
                         open(str(model_path) + ".tmp", "wb") as out:
                        total = int(resp.headers.get("Content-Length", 0))
                        done  = 0
                        while True:
                            block = resp.read(65536)
                            if not block:
                                break
                            out.write(block)
                            done += len(block)
                            if total > 0:
                                self._on_dl_progress(done / total,
                                    f"模型 {done/1_048_576:.0f}/{total/1_048_576:.0f} MB")
                    import os as _os
                    _os.replace(str(model_path) + ".tmp", str(model_path))
                    self.after(0, self._hide_dl_bar)
                except Exception as e:
                    msg = str(e)
                    self.after(0, self._hide_dl_bar)
                    self.after(0, lambda: messagebox.showerror(
                        "下載失敗",
                        f"chatllm 模型下載失敗：\n{msg}\n\n請確認網路連線後點「重新載入」重試。",
                    ))
                    self.after(0, lambda: self._set_status("❌ 下載失敗"))
                    self.after(0, lambda: self.reload_btn.configure(state="normal"))
                    return

            # 持久化完整設定（確保下次啟動不會重觸 onboarding）
            settings["model_path"]  = str(model_path)
            settings["chatllm_dir"] = str(chatllm_dir)
            self._settings = settings
            self._save_settings(settings)

            # 設定 _model_dir 供 diarization 下載確認流程使用
            self._model_dir = Path(settings.get("model_dir", str(BASE_DIR / "ov_models")))

            # 從 device_label 解析 Vulkan device ID
            # 格式：「GPU:0 (AMD Radeon(TM) Graphics) [Vulkan]」
            _vk_dev_id = 0
            _m = re.search(r"GPU:(\d+)", device_label)
            if _m:
                _vk_dev_id = int(_m.group(1))


            self.engine = create_engine("chatllm")
            self.engine.output_simplified = getattr(self, '_output_simplified', False)
            try:
                self.engine.load(
                    model_path  = model_path,
                    chatllm_dir = chatllm_dir,
                    n_gpu_layers= 99,
                    device_id   = _vk_dev_id,
                    cb          = self._set_status,
                )
                self.after(0, self._on_models_ready)
            except Exception as e:
                first_line = str(e).splitlines()[0][:120]
                self.after(0, lambda r=first_line: self._on_models_failed("chatllm", r))

        else:
            # ── OpenVINO 路線 ──────────────────────────────────────────
            model_dir  = Path(settings.get("model_dir", str(_DEFAULT_MODEL_DIR)))
            model_size = settings.get("cpu_model_size", self.model_var.get())
            self._model_dir = model_dir

            # 1.7B 按需下載
            use_17b = "1.7B" in model_size
            if use_17b:
                from downloader import quick_check_1p7b, download_1p7b
                if not quick_check_1p7b(model_dir):
                    self.after(0, self._show_dl_bar)
                    self._set_status("⬇ 下載 1.7B 模型（約 4.3 GB）…")
                    try:
                        download_1p7b(model_dir, progress_cb=self._on_dl_progress)
                    except Exception as e:
                        msg = str(e)
                        self.after(0, self._hide_dl_bar)
                        self.after(0, lambda: self.reload_btn.configure(state="normal"))
                        self.after(0, lambda: messagebox.showerror(
                            "下載失敗",
                            f"1.7B 模型下載失敗：\n{msg}\n\n"
                            "請確認網路連線後點「重新載入」重試。",
                        ))
                        self.after(0, lambda: self._set_status("❌ 下載失敗"))
                        return
                    self.after(0, self._hide_dl_bar)

            model_size_key = "1.7B" if use_17b else "0.6B"
            self.engine = create_engine("openvino", model_size_key)
            self.engine.output_simplified = getattr(self, '_output_simplified', False)
            try:
                self.engine.load(device=ov_device, model_dir=model_dir, cb=self._set_status)
                self.after(0, self._on_models_ready)
            except Exception as e:
                first_line = str(e).splitlines()[0][:120]
                self.after(0, lambda d=ov_device, r=first_line: self._on_models_failed(d, r))

    def _on_models_ready(self):
        self.device_combo.configure(state="readonly")
        self.reload_btn.configure(state="normal")
        self.convert_btn.configure(state="normal")
        self.rt_start_btn.configure(state="normal")
        # 注入引擎到批次辨識頁籤
        if hasattr(self, "_batch_tab"):
            self._batch_tab.set_engine(self.engine)

        settings = self._settings or {}
        backend  = settings.get("backend", "openvino")
        device   = self.device_var.get()

        # ── model_combo 依後端顯示 ─────────────────────────────────────
        if backend == "chatllm":
            # Vulkan GPU：顯示固定標籤，combo 唯讀
            self.model_combo.configure(
                values=["1.7B Q8_0 (Vulkan GPU)"], state="disabled"
            )
            self.model_var.set("1.7B Q8_0 (Vulkan GPU)")
            self._set_status(f"✅ 就緒（Vulkan GPU）")
        else:
            # OpenVINO：顯示 0.6B / 1.7B，可切換
            self.model_combo.configure(
                values=["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B INT8"],
                state="readonly",
            )
            sz = settings.get("cpu_model_size", "0.6B")
            self.model_var.set(
                "Qwen3-ASR-1.7B INT8" if sz == "1.7B" else "Qwen3-ASR-0.6B"
            )
            self._set_status(f"✅ 就緒（{device}）")

        # 填入語系清單（模型載入後才知道 supported_languages）
        if self.engine.processor and self.engine.processor.supported_languages:
            langs = ["自動偵測"] + self.engine.processor.supported_languages
            self._lang_list = self.engine.processor.supported_languages
            self.lang_combo.configure(values=langs, state="readonly")
            self.lang_var.set("自動偵測")
        elif backend == "chatllm":
            # chatllm 模型支援所有語系，提供常用語系清單
            common_langs = [
                "Chinese", "English", "Japanese", "Korean",
                "Cantonese", "French", "German", "Spanish",
                "Portuguese", "Russian", "Arabic", "Thai",
                "Vietnamese", "Indonesian", "Malay",
            ]
            self.lang_combo.configure(
                values=["自動偵測"] + common_langs, state="readonly"
            )
            self.lang_var.set("自動偵測")
        # 說話者分離 checkbox
        if self.engine.diar_engine and self.engine.diar_engine.ready:
            self.diarize_chk.configure(state="normal")
        else:
            # 模型未就緒：背景確認是否需要下載
            threading.Thread(
                target=self._check_diarization_models, daemon=True
            ).start()

    # ── 說話者分離模型：啟動時檢查 + 按需下載 ─────────────────────────

    def _check_diarization_models(self):
        """背景執行緒：若說話者分離模型不存在，則在主執行緒詢問使用者。"""
        from downloader import quick_check_diarization
        if self._model_dir and not quick_check_diarization(self._model_dir):
            self.after(0, self._ask_download_diarization)

    def _ask_download_diarization(self):
        """主執行緒：詢問使用者是否下載說話者分離模型（約 32 MB）。"""
        answer = messagebox.askyesno(
            "說話者分離模型",
            "說話者分離功能需要額外下載模型（約 32 MB）：\n"
            "  • segmentation-community-1.onnx\n"
            "  • embedding_model.onnx\n\n"
            "是否立即下載？（選「否」可稍後透過重新載入模型觸發）",
        )
        if answer:
            threading.Thread(
                target=self._download_diarization_models, daemon=True
            ).start()

    def _download_diarization_models(self):
        """背景執行緒：下載說話者分離模型，完成後重新載入 DiarizationEngine。"""
        from downloader import download_diarization
        from diarize import DiarizationEngine

        diar_dir = self._model_dir / "diarization"
        self.after(0, self._show_dl_bar)
        self._set_status("⬇ 下載說話者分離模型…")
        try:
            download_diarization(diar_dir, progress_cb=self._on_dl_progress)
        except Exception as e:
            msg = str(e)
            self.after(0, self._hide_dl_bar)
            self.after(0, lambda: messagebox.showerror(
                "下載失敗",
                f"說話者分離模型下載失敗：\n{msg}\n\n請確認網路連線後重試。",
            ))
            self.after(0, lambda: self._set_status("❌ 下載失敗"))
            return

        self.after(0, self._hide_dl_bar)

        # 重新載入 DiarizationEngine
        try:
            eng = DiarizationEngine(diar_dir)
            if eng.ready:
                self.engine.diar_engine = eng
                self.after(0, lambda: self.diarize_chk.configure(state="normal"))
                device = self.device_var.get()
                self.after(0, lambda: self._set_status(f"✅ 就緒（{device}）"))
            else:
                self.after(0, lambda: messagebox.showerror(
                    "載入失敗", "說話者分離模型下載完成，但無法正常載入，請重新啟動程式。"
                ))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: messagebox.showerror(
                "載入失敗", f"說話者分離模型載入失敗：{err}"
            ))

    def _on_models_failed(self, device: str, reason: str):
        """模型載入失敗：若為 Vulkan（chatllm）後端，自動退回 CPU 重試；
        若本身已是 OpenVINO 路線，還原 UI 讓使用者手動選擇。
        """
        # ── 判斷是否為 Vulkan 後端失敗 ──────────────────────────────────
        failed_backend = (self._settings or {}).get("backend", "openvino")

        if failed_backend == "chatllm":
            # Vulkan 引擎（AMD / NVIDIA）失敗 → 自動 fallback 到 CPU
            # 1. 通知使用者（非阻塞式，因為要繼續觸發 fallback 載入）
            self.after(0, lambda: messagebox.showwarning(
                "GPU 引擎失敗，自動退回 CPU",
                f"Vulkan GPU（{device}）載入失敗：\n{reason}\n\n"
                "已自動切換為 CPU 模式重新載入，請稍候…",
            ))
            # 2. 更新設定與 UI 選單至 CPU
            fallback: dict = dict(self._settings) if self._settings else {}
            fallback["backend"] = "openvino"
            fallback["device"]  = "CPU"
            self._settings = fallback
            self._save_settings(fallback)
            self.device_var.set("CPU")
            # 3. 在背景執行緒重新以 CPU 載入（不阻塞 UI thread）
            self.engine.ready = False
            threading.Thread(target=self._load_models, daemon=True).start()
        else:
            # OpenVINO 路線失敗（GPU.0 Intel iGPU 等）→ 還原 UI 讓使用者重試
            self.device_combo.configure(state="readonly")
            self.reload_btn.configure(state="normal")
            self.status_dot.configure(
                text=f"❌ {device} 載入失敗，請切換裝置後點「重新載入」",
                text_color="#EF5350",
            )
            messagebox.showerror(
                "模型載入失敗",
                f"裝置「{device}」載入失敗：\n{reason}\n\n"
                "建議：將裝置切換為 CPU 後點「重新載入」。",
            )

    def _on_reload_models(self):
        if self._converting:
            messagebox.showwarning("提示", "轉換進行中，請等候完成後再重新載入")
            return
        if self._rt_mgr:
            self._on_rt_stop()

        # 從 UI 狀態同步設定（允許使用者在 dev_bar 手動切換裝置後重新載入）
        dev_label  = self.device_var.get()
        model_sel  = self.model_var.get()
        cur        = dict(self._settings) if self._settings else self._load_settings()

        if "Vulkan" in dev_label:
            cur["backend"] = "chatllm"
            cur["device"]  = dev_label
        else:
            cur["backend"] = "openvino"
            cur["device"]  = dev_label
            cur["cpu_model_size"] = "1.7B" if "1.7B" in model_sel else "0.6B"

        self._settings = cur

        if self.engine:
            self.engine.ready = False
        self.convert_btn.configure(state="disabled")
        self.rt_start_btn.configure(state="disabled")
        self.reload_btn.configure(state="disabled")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_dot.configure(text=msg))

    def _refresh_audio_devices(self):
        try:
            import sounddevice as sd
            devs    = sd.query_devices()
            choices = []
            self._dev_idx_map = {}
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    name = d["name"][:50]
                    choices.append(name)
                    self._dev_idx_map[name] = i
            if choices:
                self.rt_dev_combo.configure(values=choices)
                default = sd.default.device[0]
                default_name = next(
                    (k for k, v in self._dev_idx_map.items() if v == default), choices[0]
                )
                self.rt_dev_combo.set(default_name)
        except ImportError:
            self.rt_dev_combo.configure(values=["（需安裝 sounddevice）"])

    # ── 音檔轉字幕操作 ─────────────────────────────────

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="選擇音訊 / 影片檔案",
            filetypes=[
                ("音訊 / 影片檔案",
                 "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.opus *.wma "
                 "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.ts"),
                ("音訊檔案", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.opus *.wma"),
                ("影片檔案", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.ts *.m2ts"),
                ("所有檔案", "*.*"),
            ],
        )
        if path:
            self._audio_file = Path(path)
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, str(self._audio_file))
            if self.engine.ready:
                self.convert_btn.configure(state="normal")

    def _on_verify(self):
        """開啟字幕驗證編輯視窗。"""
        if not self._srt_output or not self._srt_output.exists():
            messagebox.showwarning("提示", "尚無可驗證的字幕，請先執行轉換。")
            return
        SubtitleEditorWindow(
            self,
            srt_path     = self._srt_output,
            audio_path   = self._audio_file,
            diarize_mode = getattr(self, "_file_diarize", False),
        )

    def _on_convert(self):
        with self._convert_lock:
            if self._converting:
                return
        path = Path(self.file_entry.get().strip())
        if not path.exists():
            messagebox.showwarning("提示", "找不到檔案，請重新選擇")
            return
        if not self.engine.ready:
            messagebox.showwarning("提示", "模型尚未載入完成")
            return

        self._audio_file = path
        # 讀取語系、hint 與說話者分離選項（在主執行緒讀取 UI 值，再傳給 worker）
        lang_sel = self.lang_var.get()
        self._selected_language = lang_sel if lang_sel != "自動偵測" else None
        hint_text = self.hint_box.get("1.0", "end").strip()
        self._file_hint = hint_text if hint_text else None
        self._file_diarize = self._diarize_var.get()
        n_spk_sel = self.n_spk_combo.get()
        self._file_n_speakers = (int(n_spk_sel)
                                  if n_spk_sel.isdigit() else None)

        # 所有檔案（包括音訊）皆需 ffmpeg 進行處理
        from ffmpeg_utils import ensure_ffmpeg
        
        def _on_ffmpeg_ready(ffmpeg_path):
            self._ffmpeg_exe = ffmpeg_path
            self._do_start_convert()
            
        ensure_ffmpeg(self, on_ready=_on_ffmpeg_ready)

    def _do_start_convert(self):
        """ffmpeg 確認後（或非影片檔案時）實際啟動轉換執行緒。"""
        with self._convert_lock:
            self._converting = True
        self.convert_btn.configure(state="disabled", text="轉換中…")
        self.prog_bar.set(0)
        self._file_log_clear()
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _convert_worker(self):
        path = self._audio_file

        # 擷取語系、hint 與說話者分離（在主執行緒已取好，直接帶入 worker）
        language   = self._selected_language
        context    = self._file_hint
        diarize    = getattr(self, "_file_diarize", False)
        n_speakers = getattr(self, "_file_n_speakers", None)
        ffmpeg_exe = getattr(self, "_ffmpeg_exe", None)

        def prog_cb(done, total, msg):
            pct = done / total if total > 0 else 0
            self.after(0, lambda: self.prog_bar.set(pct))
            self.after(0, lambda: self.prog_label.configure(text=msg))
            self._file_log(msg)

        tmp_wav: Path | None = None
        try:
            t0 = time.perf_counter()
            # 影片音軌提取
            from ffmpeg_utils import is_video, extract_audio_to_wav
            if is_video(path):
                if not ffmpeg_exe:
                    raise RuntimeError("找不到 ffmpeg，無法提取影片音軌。")
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(tmp_fd)
                tmp_wav = Path(tmp_path)
                self._file_log(f"🎬 提取音軌中：{path.name}")
                extract_audio_to_wav(path, tmp_wav, ffmpeg_exe)
                self._file_log(f"   音軌提取完成，開始辨識…")
                proc_path = tmp_wav
            else:
                proc_path = path

            lang_info  = f"  語系：{language or '自動'}"
            hint_info  = f"  提示：{context[:30]}…" if context and len(context) > 30 else (f"  提示：{context}" if context else "")
            if diarize:
                n_str = str(n_speakers) if n_speakers else "自動"
                diar_info = f"  [說話者分離，人數：{n_str}]"
            else:
                diar_info = ""
            self._file_log(f"開始處理：{path.name}{lang_info}{hint_info}{diar_info}")
            srt = self.engine.process_file(
                proc_path, progress_cb=prog_cb, language=language,
                context=context, diarize=diarize, n_speakers=n_speakers,
            )
            elapsed = time.perf_counter() - t0

            if srt:
                self._srt_output = srt
                self._file_log(f"\n✅ 完成！耗時 {elapsed:.1f}s")
                self._file_log(f"SRT 儲存至：{srt}")
                self.after(0, lambda: [
                    self.prog_bar.set(1.0),
                    self.open_dir_btn.configure(state="normal"),
                    self.verify_btn.configure(state="normal"),
                    self.prog_label.configure(text="完成"),
                ])
            else:
                self._file_log("⚠ 未偵測到人聲，未產生字幕")
                self.after(0, lambda: self.prog_bar.set(0))
        except Exception as e:
            self._file_log(f"❌ 錯誤：{e}")
            self.after(0, lambda: self.prog_bar.set(0))
        finally:
            # 清理臨時 WAV（影片音軌提取）
            if tmp_wav and tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except Exception:
                    pass
            with self._convert_lock:
                self._converting = False
            self.after(0, lambda: self.convert_btn.configure(
                state="normal", text="▶  開始轉換"
            ))

    def _file_log(self, msg: str):
        def _do():
            self.file_log.configure(state="normal")
            self.file_log.insert("end", msg + "\n")
            self.file_log.see("end")
            self.file_log.configure(state="disabled")
        self.after(0, _do)

    def _file_log_clear(self):
        self.file_log.configure(state="normal")
        self.file_log.delete("1.0", "end")
        self.file_log.configure(state="disabled")

    # ── 即時轉換操作 ───────────────────────────────────

    def _on_rt_start(self):
        name = self.rt_dev_combo.get()
        idx  = self._dev_idx_map.get(name)
        if idx is None:
            messagebox.showwarning("提示", "請選擇有效的音訊輸入裝置")
            return

        lang_sel = self.lang_var.get()
        rt_lang  = lang_sel if lang_sel != "自動偵測" else None
        rt_hint  = self.rt_hint_entry.get().strip() or None

        self._rt_mgr = RealtimeManager(
            asr=self.engine,
            device_idx=idx,
            on_text=self._on_rt_text,
            on_status=self._on_rt_status,
            language=rt_lang,
            context=rt_hint,
        )
        try:
            self._rt_mgr.start()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟音訊裝置：{e}")
            self._rt_mgr = None
            return

        self.rt_start_btn.configure(state="disabled")
        self.rt_stop_btn.configure(state="normal")

    def _on_rt_stop(self):
        if self._rt_mgr:
            self._rt_mgr.stop()
            self._rt_mgr = None
        self.rt_start_btn.configure(state="normal")
        self.rt_stop_btn.configure(state="disabled")

    def _on_rt_text(self, text: str, start_sec: float, end_sec: float):
        with self._rt_lock:
            self._rt_log.append((text, start_sec, end_sec))
        def _do():
            ts = datetime.now().strftime("%H:%M:%S")
            self.rt_textbox.configure(state="normal")
            self.rt_textbox.insert("end", f"[{ts}]  {text}\n")
            self.rt_textbox.see("end")
            self.rt_textbox.configure(state="disabled")
        self.after(0, _do)

    def _on_rt_status(self, msg: str):
        self.after(0, lambda: self.rt_status_lbl.configure(text=msg))

    def _on_rt_clear(self):
        with self._rt_lock:
            self._rt_log.clear()
        self.rt_textbox.configure(state="normal")
        self.rt_textbox.delete("1.0", "end")
        self.rt_textbox.configure(state="disabled")

    def _on_rt_save(self):
        with self._rt_lock:
            log_snapshot = list(self._rt_log)
        if not log_snapshot:
            messagebox.showinfo("提示", "目前沒有字幕內容可儲存")
            return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = SRT_DIR / f"realtime_{ts}.srt"
        with open(out, "w", encoding="utf-8") as f:
            for idx, (text, start, end) in enumerate(log_snapshot, 1):
                f.write(f"{idx}\n{srt_ts(start)} --> {srt_ts(end)}\n{text}\n\n")
        messagebox.showinfo("儲存完成", f"已儲存至：\n{out}")
        os.startfile(str(SRT_DIR))

    # ── 關閉處理 ───────────────────────────────────────

    def _on_close(self):
        # 轉換進行中：請使用者確認
        if self._converting:  # 讀取 bool 是原子操作，此處不需 lock
            if not messagebox.askyesno(
                "確認關閉",
                "音訊轉換正在進行中。\n確定要強制關閉嗎？（目前進度將遺失）",
                icon="warning",
                default="no",
            ):
                return

        # 停止 Streamlit 服務
        if hasattr(self, "_settings_tab"):
            self._settings_tab.stop_service()

        # 停止即時錄音（安靜地停，不需要確認）
        if self._rt_mgr:
            try:
                self._rt_mgr.stop()
            except Exception:
                pass

        # 銷毀視窗，再強制終止 process。
        # os._exit(0) 確保 OpenVINO / onnxruntime 的 C++ 背景執行緒
        # 不會讓程式殘留在工作管理員中。
        self.destroy()
        os._exit(0)


# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
