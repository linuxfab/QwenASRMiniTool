"""subtitle_detail_editor.py — 字幕詳細時間軸編輯視窗

從 subtitle_editor.py 拆分。
"""
from __future__ import annotations

import math
import re
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk

# ── 字型常數（與 app.py / app-gpu.py 保持一致）────────────────────────
FONT_MONO = ("Consolas", 12)


# ══════════════════════════════════════════════════════════════════════
# 字幕詳細時間軸編輯視窗
# ══════════════════════════════════════════════════════════════════════

class SubtitleDetailEditor(ctk.CTkToplevel):
    """字幕條目詳細時間軸視窗：前/中/後三段時間軸視覺化，含拖曳調整與播放。"""

    HANDLE_W = 16  # 拖曳 handle 的點擊容忍半寬（像素）

    def __init__(self, parent_editor, idx: int):
        super().__init__(parent_editor)
        self._editor = parent_editor
        self._rows   = parent_editor._rows
        self._idx    = idx
        self._dragging: "str | None" = None   # "start" | "end" | None
        self._tl_t_min = 0.0
        self._tl_t_max = 1.0
        self._is_playing = False

        self.title("字幕詳細編輯")
        self.geometry("860x540")
        self.resizable(True, True)
        self.minsize(640, 420)
        # 非 modal，可與主列表同時操作

        self._build_ui()
        self._refresh()
        # CTkToplevel 在 Windows 上有時會以最小化狀態出現，延遲後強制顯示
        self.after(120, self._bring_to_front)

    def _bring_to_front(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    # ── 時間轉換 ──────────────────────────────────────────────────────

    @staticmethod
    def _ts_to_sec(ts: str) -> float:
        try:
            h, m, rest = ts.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        except Exception:
            return 0.0

    @staticmethod
    def _sec_to_ts(sec: float) -> str:
        sec = max(0.0, sec)
        h   = int(sec // 3600)
        sec -= h * 3600
        m   = int(sec // 60)
        sec -= m * 60
        s   = int(sec)
        ms  = int(round((sec - s) * 1000))
        if ms >= 1000:
            s += 1; ms -= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ── 座標轉換 ──────────────────────────────────────────────────────

    def _t2x(self, t: float) -> int:
        w = self._tl_canvas.winfo_width() or 800
        span = max(0.001, self._tl_t_max - self._tl_t_min)
        return int((t - self._tl_t_min) / span * w)

    def _x2t(self, x: int) -> float:
        w = self._tl_canvas.winfo_width() or 800
        span = max(0.001, self._tl_t_max - self._tl_t_min)
        return self._tl_t_min + x / w * span

    # ── UI 建構 ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── 標題列
        self._title_lbl = ctk.CTkLabel(
            self, text="", font=("Microsoft JhengHei", 13, "bold"),
            text_color="#AAAACC",
        )
        self._title_lbl.pack(fill="x", padx=12, pady=(8, 2))

        # ── 時間軸 Canvas
        tl_frame = ctk.CTkFrame(self, fg_color=("gray92", "#1A1A24"), corner_radius=6, height=80)
        tl_frame.pack(fill="x", padx=8, pady=(2, 4))
        tl_frame.pack_propagate(False)

        self._tl_canvas = tk.Canvas(
            tl_frame, bg="#1A1A24", highlightthickness=0, height=80,
        )
        self._tl_canvas.pack(fill="both", expand=True)
        self._tl_canvas.bind("<ButtonPress-1>",   self._on_tl_press)
        self._tl_canvas.bind("<B1-Motion>",        self._on_tl_drag)
        self._tl_canvas.bind("<ButtonRelease-1>",  self._on_tl_release)
        self._tl_canvas.bind("<Configure>",        lambda e: self._draw_timeline())
        self._tl_canvas.bind("<Motion>",           self._on_tl_hover)

        # ── 三欄面板
        self._panel_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._panel_frame.pack(fill="x", padx=8, pady=(2, 4))
        self._panel_frame.columnconfigure(0, weight=1)
        self._panel_frame.columnconfigure(1, weight=0, minsize=8)
        self._panel_frame.columnconfigure(2, weight=2)
        self._panel_frame.columnconfigure(3, weight=0, minsize=8)
        self._panel_frame.columnconfigure(4, weight=1)

        self._prev_panel = ctk.CTkFrame(self._panel_frame, fg_color=("gray88", "#181820"), corner_radius=6)
        self._prev_panel.grid(row=0, column=0, sticky="nsew")

        self._curr_panel = ctk.CTkFrame(self._panel_frame, fg_color=("gray93", "#1E1E30"), corner_radius=6)
        self._curr_panel.grid(row=0, column=2, sticky="nsew")

        self._next_panel = ctk.CTkFrame(self._panel_frame, fg_color=("gray88", "#181820"), corner_radius=6)
        self._next_panel.grid(row=0, column=4, sticky="nsew")

        # ── 控制列
        ctrl = ctk.CTkFrame(self, fg_color=("gray85", "#14141E"), corner_radius=0, height=48)
        ctrl.pack(fill="x", side="bottom")
        ctrl.pack_propagate(False)

        ctk.CTkButton(
            ctrl, text="⏮", width=50, height=32,
            fg_color="#282838", hover_color="#383850",
            font=("Segoe UI Emoji", 15),
            command=lambda: self._navigate(-1),
        ).pack(side="left", padx=(10, 4), pady=8)

        self._play_btn = ctk.CTkButton(
            ctrl, text="▶", width=50, height=32,
            fg_color="#1A3A5C", hover_color="#265A8A",
            font=("Segoe UI Emoji", 15),
            command=self._play_current,
        )
        self._play_btn.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(
            ctrl, text="✕", width=44, height=32,
            fg_color="#38181A", hover_color="#552428",
            font=("Segoe UI Emoji", 15),
            command=self._close,
        ).pack(side="right", padx=(4, 10), pady=8)

        ctk.CTkButton(
            ctrl, text="⏭", width=50, height=32,
            fg_color="#282838", hover_color="#383850",
            font=("Segoe UI Emoji", 15),
            command=lambda: self._navigate(1),
        ).pack(side="right", padx=4, pady=8)

    # ── 重新整理內容 ──────────────────────────────────────────────────

    def _refresh(self):
        rows  = self._rows
        n     = len(rows)
        idx   = self._idx
        row   = rows[idx]

        # 標題
        spk_tag = ""
        if self._editor.has_speakers:
            sid = row["speaker"].get()
            name = self._editor._spk_name_vars.get(sid, ctk.StringVar()).get() or sid
            spk_tag = f"  【{name}】"
        self._title_lbl.configure(
            text=f"第 {idx+1} / {n} 條{spk_tag}  "
                 f"{row['start'].get()} → {row['end'].get()}"
        )

        # 三段面板
        self._build_side_panel(self._prev_panel, idx - 1 if idx > 0 else None, "前段")
        self._build_curr_panel(self._curr_panel, idx)
        self._build_side_panel(self._next_panel, idx + 1 if idx < n - 1 else None, "後段")

        # 時間軸
        self._calc_tl_range()
        self._draw_timeline()

    def _build_side_panel(self, panel: ctk.CTkFrame, idx: "int | None", label: str):
        for w in panel.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            panel, text=label,
            font=("Microsoft JhengHei", 10), text_color=("gray35", "#555566"),
        ).pack(anchor="nw", padx=8, pady=(6, 0))
        if idx is None:
            ctk.CTkLabel(
                panel, text="（無）",
                font=("Microsoft JhengHei", 11), text_color=("gray45", "#444455"),
            ).pack(padx=8, pady=8)
            return
        row = self._rows[idx]
        ctk.CTkLabel(
            panel,
            text=f"{row['start'].get()} → {row['end'].get()}",
            font=("Consolas", 10), text_color=("gray45", "#666680"),
        ).pack(anchor="nw", padx=8, pady=(2, 2))
        ctk.CTkLabel(
            panel, text=row["text"].get() or "（空白）",
            font=("Microsoft JhengHei", 11),
            text_color=("gray40", "#888898") if row["text"].get() else ("gray50", "#664466"),
            wraplength=160, justify="left",
        ).pack(anchor="nw", padx=8, pady=(0, 6))

    def _build_curr_panel(self, panel: ctk.CTkFrame, idx: int):
        for w in panel.winfo_children():
            w.destroy()
        row = self._rows[idx]
        ctk.CTkLabel(
            panel, text="本段（可編輯）",
            font=("Microsoft JhengHei", 10, "bold"), text_color=("gray35", "#8888CC"),
        ).pack(anchor="nw", padx=8, pady=(6, 0))

        # 時間行
        time_row = ctk.CTkFrame(panel, fg_color="transparent")
        time_row.pack(fill="x", padx=8, pady=(2, 0))
        ctk.CTkLabel(time_row, text="起:", font=("Microsoft JhengHei", 11),
                     text_color=("gray40", "#888899"), width=22).pack(side="left")
        ctk.CTkEntry(
            time_row, textvariable=row["start"], width=108, height=26,
            font=FONT_MONO, justify="center",
        ).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(time_row, text="→", font=("Microsoft JhengHei", 12),
                     text_color=("gray45", "#444455")).pack(side="left")
        ctk.CTkLabel(time_row, text="終:", font=("Microsoft JhengHei", 11),
                     text_color=("gray40", "#888899"), width=22).pack(side="left", padx=(6, 0))
        ctk.CTkEntry(
            time_row, textvariable=row["end"], width=108, height=26,
            font=FONT_MONO, justify="center",
        ).pack(side="left")

        # 文字輸入
        ctk.CTkEntry(
            panel, textvariable=row["text"], height=28,
            font=("Microsoft JhengHei", 12),
        ).pack(fill="x", padx=8, pady=(4, 4))

        # 說話者下拉（如有）
        if self._editor.has_speakers:
            spk_row = ctk.CTkFrame(panel, fg_color="transparent")
            spk_row.pack(fill="x", padx=8, pady=(0, 6))
            ctk.CTkLabel(spk_row, text="說話者:", font=("Microsoft JhengHei", 11),
                         text_color=("gray40", "#888899")).pack(side="left", padx=(0, 4))
            sid = row["speaker"].get()
            ci  = self._editor._all_spk_ids.index(sid) if sid in self._editor._all_spk_ids else -1
            accent = self._editor._SPK_ACCENT[ci % len(self._editor._SPK_ACCENT)] if ci >= 0 else "#666677"
            ctk.CTkComboBox(
                spk_row, variable=row["speaker"],
                values=list(self._editor._all_spk_ids),
                width=120, height=26, font=("Microsoft JhengHei", 11),
                button_color=accent, border_color=accent,
                command=lambda v: self._refresh(),
            ).pack(side="left")

    # ── 時間軸計算 ────────────────────────────────────────────────────

    def _calc_tl_range(self):
        rows = self._rows
        idx  = self._idx
        segs = [idx - 1, idx, idx + 1]
        times = []
        for i in segs:
            if 0 <= i < len(rows):
                times.append(self._ts_to_sec(rows[i]["start"].get()))
                times.append(self._ts_to_sec(rows[i]["end"].get()))
        if not times:
            self._tl_t_min = 0.0
            self._tl_t_max = 1.0
            return
        self._tl_t_min = max(0.0, min(times) - 0.5)
        self._tl_t_max = max(times) + 0.5

    def _draw_timeline(self):
        canvas = self._tl_canvas
        canvas.delete("all")
        w = canvas.winfo_width() or 800
        h = canvas.winfo_height() or 80
        RULER_H = 18   # 底部刻度區高度
        BAR_Y1  = 6
        BAR_Y2  = h - RULER_H - 4

        # 動態切換 canvas 背景（tk.Canvas 不支援 CTk 雙色元組）
        is_dark = ctk.get_appearance_mode() == "Dark"
        canvas_bg = "#1A1A24" if is_dark else "#EEF0F8"
        self._tl_canvas.configure(bg=canvas_bg)
        canvas.create_rectangle(0, 0, w, h, fill=canvas_bg, outline="")

        rows  = self._rows
        idx   = self._idx
        segs  = [(idx - 1, False), (idx, True), (idx + 1, False)]

        for seg_idx, is_curr in segs:
            if seg_idx < 0 or seg_idx >= len(rows):
                continue
            row   = rows[seg_idx]
            s     = self._ts_to_sec(row["start"].get())
            e     = self._ts_to_sec(row["end"].get())
            x1    = self._t2x(s)
            x2    = self._t2x(e)
            if x2 <= x1:
                x2 = x1 + 2

            is_blank = not row["text"].get().strip()
            sid  = row["speaker"].get()
            ci   = (self._editor._all_spk_ids.index(sid)
                    if sid in self._editor._all_spk_ids else -1)

            if is_blank:
                fill_c = "#2E1A28"
            elif is_curr:
                fill_c = (self._editor._SPK_ACCENT[ci % len(self._editor._SPK_ACCENT)]
                           if ci >= 0 else "#5577AA")
            else:
                fill_c = (self._editor._SPK_ROW_BG[ci % len(self._editor._SPK_ROW_BG)]
                           if ci >= 0 else "#222233")

            outline_c = "#FFFFFF" if is_curr else ""
            lw = 1 if is_curr else 0
            canvas.create_rectangle(
                x1, BAR_Y1, x2, BAR_Y2,
                fill=fill_c,
                outline=outline_c, width=lw,
            )

            # 拖曳 handle（current 段的左右邊緣）
            if is_curr:
                for hx in [x1, x2]:
                    canvas.create_line(hx, BAR_Y1, hx, BAR_Y2, fill="#FFFFFF", width=2)
                    # 三角指示
                    mid_y = (BAR_Y1 + BAR_Y2) // 2
                    if hx == x1:
                        canvas.create_polygon(
                            hx, mid_y - 6, hx + 7, mid_y, hx, mid_y + 6,
                            fill="#FFFFFF", outline="",
                        )
                    else:
                        canvas.create_polygon(
                            hx, mid_y - 6, hx - 7, mid_y, hx, mid_y + 6,
                            fill="#FFFFFF", outline="",
                        )

            # 標籤
            label_text = row["text"].get()[:12] + "…" if len(row["text"].get()) > 12 else row["text"].get()
            if label_text:
                lx = max(x1 + 3, min((x1 + x2) // 2, x2 - 3))
                canvas.create_text(
                    lx, (BAR_Y1 + BAR_Y2) // 2,
                    text=label_text,
                    fill="#FFFFFF" if is_curr else "#888898",
                    font=("Microsoft JhengHei", 9),
                    anchor="center",
                )

        # Ruler 刻度
        span = max(0.001, self._tl_t_max - self._tl_t_min)
        if span <= 10:
            tick_step = 1
        elif span <= 30:
            tick_step = 2
        else:
            tick_step = 5
        t_first = math.ceil(self._tl_t_min / tick_step) * tick_step
        t = t_first
        while t <= self._tl_t_max:
            rx = self._t2x(t)
            canvas.create_line(rx, BAR_Y2 + 2, rx, h - 2, fill="#445566", width=1)
            mm = int(t // 60)
            ss = int(t % 60)
            canvas.create_text(
                rx, h - 2,
                text=f"{mm:02d}:{ss:02d}",
                fill="#556677", font=("Consolas", 8), anchor="s",
            )
            t += tick_step
            t = round(t, 3)

    # ── 時間軸滑鼠事件 ────────────────────────────────────────────────

    def _on_tl_press(self, event):
        row  = self._rows[self._idx]
        s    = self._ts_to_sec(row["start"].get())
        e    = self._ts_to_sec(row["end"].get())
        x1   = self._t2x(s)
        x2   = self._t2x(e)
        HW   = self.HANDLE_W
        if abs(event.x - x1) <= HW:
            self._dragging = "start"
            self._tl_canvas.configure(cursor="sb_h_double_arrow")
        elif abs(event.x - x2) <= HW:
            self._dragging = "end"
            self._tl_canvas.configure(cursor="sb_h_double_arrow")
        elif x1 <= event.x <= x2:
            # 點在段落條內部：左半拖起點、右半拖終點
            mid = (x1 + x2) / 2
            self._dragging = "start" if event.x <= mid else "end"
            self._tl_canvas.configure(cursor="sb_h_double_arrow")
        else:
            self._dragging = None

    def _on_tl_hover(self, event):
        """滑鼠移動時提供游標回饋（↔）。"""
        if self._dragging is not None:
            return   # 拖曳中不需重新判斷
        row = self._rows[self._idx]
        s   = self._ts_to_sec(row["start"].get())
        e   = self._ts_to_sec(row["end"].get())
        x1  = self._t2x(s)
        x2  = self._t2x(e)
        HW  = self.HANDLE_W
        if (abs(event.x - x1) <= HW or abs(event.x - x2) <= HW
                or x1 <= event.x <= x2):
            self._tl_canvas.configure(cursor="sb_h_double_arrow")
        else:
            self._tl_canvas.configure(cursor="")

    def _on_tl_drag(self, event):
        if self._dragging is None:
            return
        row = self._rows[self._idx]
        s   = self._ts_to_sec(row["start"].get())
        e   = self._ts_to_sec(row["end"].get())
        t   = self._x2t(event.x)
        if self._dragging == "start":
            t = max(self._tl_t_min, min(t, e - 0.1))
            row["start"].set(self._sec_to_ts(t))
        else:
            t = max(s + 0.1, min(t, self._tl_t_max))
            row["end"].set(self._sec_to_ts(t))
        # 更新標題
        self._title_lbl.configure(
            text=self._title_lbl.cget("text").split("  ")[0]
            + f"  {row['start'].get()} → {row['end'].get()}"
        )
        self._draw_timeline()

    def _on_tl_release(self, event):
        self._dragging = None
        self._tl_canvas.configure(cursor="")

    # ── 播放控制 ─────────────────────────────────────────────────────

    def _play_current(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        row = self._rows[self._idx]
        s   = self._ts_to_sec(row["start"].get())
        e   = self._ts_to_sec(row["end"].get())
        if e <= s or self._editor._audio_data is None:
            return
        si  = max(0, int(s * self._editor._audio_sr))
        ei  = min(len(self._editor._audio_data), int(e * self._editor._audio_sr))
        seg = self._editor._audio_data[si:ei]
        if len(seg) == 0:
            return
        try:
            import sounddevice as sd
            sd.play(seg, self._editor._audio_sr)
        except Exception:
            return
        self._is_playing = True
        self._play_btn.configure(text="⏸", command=self._stop_playback)

        def _wait():
            try:
                import sounddevice as sd
                sd.wait()
            except Exception:
                pass
            self.after(0, self._on_play_done)

        threading.Thread(target=_wait, daemon=True).start()

    def _stop_playback(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._on_play_done()

    def _on_play_done(self):
        self._is_playing = False
        try:
            self._play_btn.configure(text="▶", command=self._play_current)
        except Exception:
            pass

    # ── 導航 ─────────────────────────────────────────────────────────

    def _navigate(self, delta: int):
        new_idx = self._idx + delta
        if new_idx < 0 or new_idx >= len(self._rows):
            return
        self._stop_playback()
        self._idx = new_idx
        self._calc_tl_range()
        self._refresh()

    # ── 關閉 ─────────────────────────────────────────────────────────

    def _close(self):
        self._stop_playback()
        self.destroy()
