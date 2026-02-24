"""setting.py — 獨立設定 Tab

SettingsTab(CTkScrollableFrame) 整合：
  1. Streamlit 服務控制（show_service=True 時顯示）
  2. 外觀主題
  3. 中文輸出語言
  4. 模型路徑
  5. FFmpeg 路徑

使用方式（app.py）：
    from setting import SettingsTab
    self._settings_tab = SettingsTab(
        self.tabs.tab("  設定  "), self, show_service=True)
    self._settings_tab.pack(fill="both", expand=True)

使用方式（app-gpu.py）：
    self._settings_tab = SettingsTab(
        self.tabs.tab("  設定  "), self, show_service=False)
    self._settings_tab.pack(fill="both", expand=True)

對外 API：
    sync_prefs(settings: dict)  — 由 App._apply_ui_prefs 呼叫
    stop_service()              — 由 App._on_close 呼叫
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

# ── 字型常數（與 app.py / app-gpu.py 保持一致）────────────────────────
FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_SMALL = ("Microsoft JhengHei", 11)
FONT_MONO  = ("Consolas", 12)


# ── 模組函式：取得 Python 解譯器路徑（供服務啟動）─────────────────────

def _get_python_exe() -> Path:
    """取得可執行的 Python 解譯器路徑。
    EXE 模式下在 _python/ 子目錄尋找 python.exe（避免 DLL 衝突）。
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        for cand in [
            base / "_python" / "python.exe",
            base / "python.exe",
            base / "_internal" / "python.exe",
        ]:
            if cand.exists():
                return cand
        return Path(sys.executable)
    return Path(sys.executable)


def _hsep(parent):
    """水平分隔線。"""
    ctk.CTkFrame(
        parent, fg_color=("gray80", "gray25"), height=1, corner_radius=0,
    ).pack(fill="x", padx=0, pady=8)


# ══════════════════════════════════════════════════════════════════════
# SettingsTab
# ══════════════════════════════════════════════════════════════════════

class SettingsTab(ctk.CTkScrollableFrame):
    """設定頁籤：外觀、語言、模型路徑、FFmpeg，可選 Streamlit 服務控制。"""

    def __init__(self, parent, app, *, show_service: bool = False):
        super().__init__(parent, fg_color=("gray92", "gray17"))
        self._app           = app
        self._show_service  = show_service
        self._sl_process: subprocess.Popen | None = None
        self._sl_port: int  = 8501
        self._log_expanded  = False
        self._build()

    # ══ 建構 UI ══════════════════════════════════════════════════════

    def _build(self):
        if self._show_service:
            self._build_service_section()
            _hsep(self)

        self._build_appearance_section()
        _hsep(self)

        self._build_language_section()
        _hsep(self)

        self._build_model_path_section()
        _hsep(self)

        self._build_ffmpeg_section()

    # ── 1. Streamlit 服務 ─────────────────────────────────────────────

    def _build_service_section(self):
        ctk.CTkLabel(
            self, text="🌐 Streamlit 網頁服務",
            font=("Microsoft JhengHei", 14, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 4))

        ctk.CTkLabel(
            self,
            text="在本機啟動網頁版前端，啟動後點選按鈕開啟瀏覽器，不會自動彈出視窗。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 6))

        # 狀態 row
        status_row = ctk.CTkFrame(self, fg_color="transparent")
        status_row.pack(fill="x", padx=12, pady=(0, 2))

        self._sl_status_dot = ctk.CTkLabel(
            status_row, text="⚫", font=FONT_BODY, width=28, anchor="w",
        )
        self._sl_status_dot.pack(side="left")

        self._sl_status_lbl = ctk.CTkLabel(
            status_row, text="服務未啟動", font=FONT_BODY, anchor="w",
        )
        self._sl_status_lbl.pack(side="left")

        self._sl_url_lbl = ctk.CTkLabel(
            status_row, text="", font=FONT_BODY,
            text_color="#7dd3fc", cursor="hand2",
        )
        self._sl_url_lbl.pack(side="left", padx=(8, 0))
        self._sl_url_lbl.bind("<Button-1>", lambda _: self._on_sl_open())

        # 連接埠 + 控制按鈕 row
        ctrl_row = ctk.CTkFrame(self, fg_color="transparent")
        ctrl_row.pack(fill="x", padx=12, pady=(2, 6))

        ctk.CTkLabel(ctrl_row, text="連接埠：", font=FONT_BODY).pack(side="left")
        self._sl_port_var   = ctk.StringVar(value="8501")
        self._sl_port_entry = ctk.CTkEntry(
            ctrl_row, textvariable=self._sl_port_var,
            width=72, height=30, font=FONT_BODY,
        )
        self._sl_port_entry.pack(side="left", padx=(4, 12))

        self._sl_start_btn = ctk.CTkButton(
            ctrl_row, text="▶ 啟動", width=86, height=30, font=FONT_BODY,
            command=self._on_sl_start,
        )
        self._sl_start_btn.pack(side="left", padx=(0, 4))

        self._sl_stop_btn = ctk.CTkButton(
            ctrl_row, text="■ 停止", width=76, height=30, font=FONT_BODY,
            fg_color=("gray60", "gray35"), hover_color=("gray50", "gray25"),
            state="disabled",
            command=self._on_sl_stop,
        )
        self._sl_stop_btn.pack(side="left", padx=(0, 4))

        self._sl_open_btn = ctk.CTkButton(
            ctrl_row, text="🌐 開啟", width=80, height=30, font=FONT_BODY,
            state="disabled", command=self._on_sl_open,
        )
        self._sl_open_btn.pack(side="left", padx=(0, 4))

        self._sl_copy_btn = ctk.CTkButton(
            ctrl_row, text="📋 複製", width=76, height=30, font=FONT_BODY,
            state="disabled", command=self._on_sl_copy_url,
        )
        self._sl_copy_btn.pack(side="left")

        # 日誌展開/收合
        self._log_toggle_btn = ctk.CTkButton(
            self, text="⋯ 服務日誌", width=110, height=26,
            fg_color=("gray82", "gray28"), hover_color=("gray72", "gray35"),
            font=FONT_SMALL, anchor="w",
            command=self._toggle_log,
        )
        self._log_toggle_btn.pack(anchor="w", padx=12, pady=(0, 4))

        # 日誌框（預設收合）
        self._sl_log_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._sl_log_box = ctk.CTkTextbox(
            self._sl_log_frame, font=("Consolas", 11), state="disabled", height=140,
        )
        self._sl_log_box.pack(fill="x", padx=12, pady=(0, 6))

    def _toggle_log(self):
        if self._log_expanded:
            self._sl_log_frame.pack_forget()
            self._log_toggle_btn.configure(text="⋯ 服務日誌")
        else:
            self._sl_log_frame.pack(fill="x")
            self._log_toggle_btn.configure(text="▲ 收合日誌")
        self._log_expanded = not self._log_expanded

    # ── 2. 外觀主題 ───────────────────────────────────────────────────

    def _build_appearance_section(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(10, 8))

        ctk.CTkLabel(
            row, text="🎨 外觀主題", font=FONT_BODY, width=130, anchor="w",
        ).pack(side="left")

        self.appearance_seg = ctk.CTkSegmentedButton(
            row, values=["🌑 深色", "☀ 淺色"],
            width=160, height=30, font=FONT_BODY,
            command=self._on_appearance_seg,
        )
        self.appearance_seg.set("🌑 深色")
        self.appearance_seg.pack(side="left")

    def _on_appearance_seg(self, value: str):
        # 映射為 App._on_appearance_change 接受的值
        mapped = "☀" if "淺" in value else "🌑"
        self._app._on_appearance_change(mapped)

    # ── 3. 中文輸出語言 ───────────────────────────────────────────────

    def _build_language_section(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(10, 8))

        ctk.CTkLabel(
            row, text="🈶 中文輸出", font=FONT_BODY, width=130, anchor="w",
        ).pack(side="left")

        self.chinese_seg = ctk.CTkSegmentedButton(
            row, values=["繁體中文", "簡體中文"],
            width=160, height=30, font=FONT_BODY,
            command=self._on_chinese_seg,
        )
        self.chinese_seg.set("繁體中文")
        self.chinese_seg.pack(side="left")

    def _on_chinese_seg(self, value: str):
        # 映射為 App._on_chinese_mode_change 接受的值
        mapped = "簡體" if "簡" in value else "繁體"
        self._app._on_chinese_mode_change(mapped)

    # ── 4. 模型路徑 ───────────────────────────────────────────────────

    def _build_model_path_section(self):
        ctk.CTkLabel(
            self, text="📦 模型路徑", font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))

        self._model_path_lbl = ctk.CTkLabel(
            row, text=self._get_model_path_text(),
            font=FONT_SMALL, anchor="w",
            text_color=("gray30", "gray70"),
            wraplength=400, justify="left",
        )
        self._model_path_lbl.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            row, text="更改…", width=72, height=28, font=FONT_SMALL,
            command=self._on_change_model_dir,
        ).pack(side="right")

    def _get_model_path_text(self) -> str:
        """取得顯示用模型路徑文字（相容 app.py / app-gpu.py）。"""
        import sys
        app_module = sys.modules.get(type(self._app).__module__)
        # GPU 版（app-gpu.py）有 GPU_MODEL_DIR 模組全域
        gpu_dir  = getattr(app_module, "GPU_MODEL_DIR",  None)
        asr_name = getattr(app_module, "ASR_MODEL_NAME", None)
        if gpu_dir and asr_name:
            return str(gpu_dir / asr_name)
        # CPU 版（app.py）使用 self._settings
        if hasattr(self._app, "_settings"):
            md = self._app._settings.get("model_dir", "")  # type: ignore
            return str(md) if md else "（尚未設定）"
        return "（尚未設定）"

    def _on_change_model_dir(self):
        d = filedialog.askdirectory(parent=self, title="選擇模型目錄")
        if not d:
            return
        import sys
        app_module = sys.modules.get(type(self._app).__module__)
        # GPU 版儲存至 gpu_model_dir
        if getattr(app_module, "GPU_MODEL_DIR", None):
            self._app._patch_setting("gpu_model_dir", d)  # type: ignore
        else:
            self._app._patch_setting("model_dir", d)  # type: ignore
        self._model_path_lbl.configure(text=d)

    # ── 5. FFmpeg ─────────────────────────────────────────────────────

    def _build_ffmpeg_section(self):
        ctk.CTkLabel(
            self, text="🎞 FFmpeg", font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))

        self._ffmpeg_status_lbl = ctk.CTkLabel(
            row, text="（載入中…）",
            font=FONT_SMALL, anchor="w",
            text_color=("gray30", "gray70"),
        )
        self._ffmpeg_status_lbl.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            row, text="選擇 ffmpeg.exe", width=130, height=28, font=FONT_SMALL,
            command=self._on_choose_ffmpeg,
        ).pack(side="right")

    def _on_choose_ffmpeg(self):
        p = filedialog.askopenfilename(
            parent=self,
            title="選擇 ffmpeg.exe",
            filetypes=[("可執行檔", "ffmpeg.exe *.exe"), ("所有檔案", "*.*")],
        )
        if not p:
            return
        self._app._patch_setting("ffmpeg_path", p)  # type: ignore
        if hasattr(self._app, "_ffmpeg_exe"):
            self._app._ffmpeg_exe = Path(p)  # type: ignore
        self._ffmpeg_status_lbl.configure(
            text=f"✅ {p}", text_color=("green", "#88CC88"),
        )

    # ══ 對外 API ══════════════════════════════════════════════════════

    def sync_prefs(self, settings: dict):
        """由 App._apply_ui_prefs 呼叫，同步所有 UI 控件狀態。"""
        # 外觀主題
        mode = settings.get("appearance_mode", "dark")
        self.appearance_seg.set("☀ 淺色" if mode == "light" else "🌑 深色")

        # 中文輸出
        self.chinese_seg.set(
            "簡體中文" if settings.get("output_simplified") else "繁體中文"
        )

        # FFmpeg 狀態
        ffpath = settings.get("ffmpeg_path", "")
        if ffpath and Path(ffpath).exists():
            self._ffmpeg_status_lbl.configure(
                text=f"✅ {ffpath}", text_color=("green", "#88CC88"),
            )
        else:
            ffexe = getattr(self._app, "_ffmpeg_exe", None)
            if ffexe and Path(ffexe).exists():
                self._ffmpeg_status_lbl.configure(
                    text=f"✅ {ffexe}", text_color=("green", "#88CC88"),
                )
            else:
                self._ffmpeg_status_lbl.configure(
                    text="❌ 未配置", text_color=("red", "#CC6666"),
                )

        # 模型路徑（settings 已同步到 self._app._settings，可直接讀）
        self._model_path_lbl.configure(text=self._get_model_path_text())

    def stop_service(self):
        """由 App._on_close 呼叫，安靜地終止 Streamlit 子程序。"""
        if self._sl_process:
            try:
                self._sl_process.terminate()
            except Exception:
                pass

    # ══ Streamlit 服務方法（show_service=True 時有效）═════════════════

    def _on_sl_start(self):
        """啟動 Streamlit 服務（子程序）。"""
        import sys as _sys
        app_module = _sys.modules.get(type(self._app).__module__)
        base = getattr(app_module, "BASE_DIR", None) or Path(__file__).parent

        sl_script = base / "streamlit_vulkan.py"
        if not sl_script.exists():
            self._sl_append_log("❌ 找不到 streamlit_vulkan.py，無法啟動服務")
            return

        try:
            port = int(self._sl_port_var.get())
        except ValueError:
            port = 8501
            self._sl_port_var.set("8501")
        self._sl_port = port

        py_exe  = _get_python_exe()
        _NO_WIN = 0x08000000 if sys.platform == "win32" else 0
        cmd = [
            str(py_exe), "-m", "streamlit", "run",
            str(sl_script),
            "--server.port",              str(port),
            "--server.headless",          "true",
            "--browser.gatherUsageStats", "false",
        ]
        self._sl_append_log(
            f"▶ 啟動：streamlit run streamlit_vulkan.py --server.port {port}"
        )
        self._sl_append_log("⏳ 等待 Streamlit 初始化（通常需要 5–15 秒）…")

        try:
            self._sl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(base),
                creationflags=_NO_WIN,
            )
        except Exception as e:
            self._sl_append_log(f"❌ 啟動失敗：{e}")
            return

        self._sl_status_dot.configure(text="🟡")
        self._sl_status_lbl.configure(text="啟動中…")
        self._sl_start_btn.configure(state="disabled")
        self._sl_stop_btn.configure(state="normal")
        self._sl_port_entry.configure(state="disabled")

        threading.Thread(target=self._sl_log_reader, daemon=True).start()
        threading.Thread(target=self._sl_monitor,    daemon=True).start()

    def _on_sl_stop(self):
        """停止 Streamlit 服務。"""
        proc, self._sl_process = self._sl_process, None
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._sl_on_stopped()
        self._sl_append_log("■ 服務已手動停止")

    def _on_sl_open(self):
        """在預設瀏覽器中開啟 Streamlit 網頁。"""
        url = self._sl_url_lbl.cget("text")
        if url:
            webbrowser.open(url)
        else:
            webbrowser.open(f"http://localhost:{self._sl_port}")

    def _on_sl_copy_url(self):
        """複製 URL 到剪貼簿。"""
        url = self._sl_url_lbl.cget("text") or f"http://localhost:{self._sl_port}"
        self.clipboard_clear()
        self.clipboard_append(url)
        self._sl_copy_btn.configure(text="✅ 已複製")
        self.after(2000, lambda: self._sl_copy_btn.configure(text="📋 複製"))

    def _sl_append_log(self, text: str):
        """（可跨執行緒）在服務日誌框末尾追加一行。"""
        if not self._show_service:
            return

        def _do():
            ts = datetime.now().strftime("%H:%M:%S")
            self._sl_log_box.configure(state="normal")
            self._sl_log_box.insert("end", f"[{ts}] {text}\n")
            self._sl_log_box.see("end")
            self._sl_log_box.configure(state="disabled")

        self.after(0, _do)

    def _sl_log_reader(self):
        """背景：讀取 Streamlit stdout；解析 'Local URL:' 偵測就緒。"""
        _ANSI = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
        proc   = self._sl_process
        if not proc or not proc.stdout:
            return
        for raw in proc.stdout:
            line = _ANSI.sub("", raw).rstrip()
            if not line:
                continue
            self._sl_append_log(line)
            if "Local URL:" in line:
                url = line.split("Local URL:")[-1].strip()
                self.after(0, lambda u=url: self._sl_on_ready(u))
        if self._sl_process is not None:
            self.after(0, self._sl_on_stopped)

    def _sl_monitor(self):
        """背景：等待程序退出。"""
        proc = self._sl_process
        if proc:
            proc.wait()
        if self._sl_process is not None:
            self._sl_process = None
            self.after(0, self._sl_on_stopped)

    def _sl_on_ready(self, url: str):
        """Streamlit 已就緒（主執行緒）。"""
        self._sl_status_dot.configure(text="🟢")
        self._sl_status_lbl.configure(text="服務就緒")
        self._sl_url_lbl.configure(text=url)
        self._sl_open_btn.configure(state="normal")
        self._sl_copy_btn.configure(state="normal")
        self._sl_append_log(f"✅ 服務就緒：{url}")

    def _sl_on_stopped(self):
        """程序退出後重設 UI（主執行緒）。"""
        self._sl_status_dot.configure(text="⚫")
        self._sl_status_lbl.configure(text="服務未啟動")
        self._sl_url_lbl.configure(text="")
        self._sl_start_btn.configure(state="normal")
        self._sl_stop_btn.configure(state="disabled")
        self._sl_open_btn.configure(state="disabled")
        self._sl_copy_btn.configure(state="disabled")
        self._sl_port_entry.configure(state="normal")
