"""subtitle_editor_window.py — 字幕驗證與編輯主視窗

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

from subtitle_detail_editor import SubtitleDetailEditor

# ── 字型常數（與 app.py / subtitle_detail_editor.py 保持一致）──────
FONT_MONO = ("Consolas", 12)


# ══════════════════════════════════════════════════════════════════════
# 字幕驗證 & 編輯視窗
# ══════════════════════════════════════════════════════════════════════

class SubtitleEditorWindow(ctk.CTkToplevel):
    """字幕逐條驗證、段落試聽與編輯的獨立子視窗。

    功能：
      - 逐條顯示 SRT 字幕（起迄時間可直接編輯）
      - ▶ 段落試聽：從音訊指定時間點播放到結束點後停止
      - (+) / (−)：在指定條目後新增 / 刪除條目
      - 多說話者模式：不同顏色區別說話者，可下拉切換，可命名
      - 確認儲存 → <原檔>_edited_<時間戳>.srt
    """

    # 每位說話者的行背景色（深色主題）
    _SPK_ROW_BG = [
        "#122030",  # 0 深藍
        "#102010",  # 1 深綠
        "#241508",  # 2 深橙棕
        "#1C1028",  # 3 深紫
        "#281010",  # 4 深紅
        "#0E2020",  # 5 深青
    ]
    # 無文字空白段的行背景（深紫紅 / 淺色模式用粉白）
    @property
    def _blank_bg(self) -> str:
        return "#241520" if ctk.get_appearance_mode() == "Dark" else "#F8F0F4"

    # 說話者強調色（文字 / 邊框 / 按鈕）
    _SPK_ACCENT = [
        "#5DADE2",  # 0 亮藍
        "#58D68D",  # 1 亮綠
        "#F0B27A",  # 2 橙
        "#C39BD3",  # 3 紫
        "#F1948A",  # 4 粉紅
        "#76D7C4",  # 5 青
    ]

    # 分頁常數：每頁顯示幾行字幕
    PAGE_SIZE = 20

    def __init__(
        self,
        parent,
        srt_path: Path,
        audio_path: "Path | None",
        diarize_mode: bool = False,
    ):
        super().__init__(parent)
        self.srt_path     = srt_path
        self.audio_path   = audio_path
        self.diarize_mode = diarize_mode

        self._audio_data: "np.ndarray | None" = None
        self._audio_sr   = 16000
        self._rows: list[dict] = []   # 每條 = {start, end, speaker, text} StringVar
        self._page: int = 0           # 目前分頁（0-based）

        raw = self._parse_srt(srt_path)
        self._all_spk_ids: list[str] = sorted({e["speaker"] for e in raw if e["speaker"]})
        self.has_speakers = bool(self._all_spk_ids) and diarize_mode

        # 說話者顯示名稱（使用者可修改，預設「說話者1」…）
        self._spk_name_vars: dict[str, ctk.StringVar] = {
            sid: ctk.StringVar(value=f"說話者{i + 1}")
            for i, sid in enumerate(self._all_spk_ids)
        }
        self._init_rows(raw)
        self._draft_status_var = ctk.StringVar(value="")  # 暫存狀態顯示
        self._build_ui()

        if audio_path and audio_path.exists():
            threading.Thread(target=self._load_audio, daemon=True).start()
        # 視窗渲染完成後再偵測草稿
        self.after(200, self._check_draft)

    # ── SRT 解析 ─────────────────────────────────────────────────────

    def _parse_srt(self, path: Path) -> list[dict]:
        text   = path.read_text(encoding="utf-8")
        blocks = re.split(r"\n\s*\n", text.strip())
        out: list[dict] = []
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            m = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                lines[1],
            )
            if not m:
                continue
            content = " ".join(l.strip() for l in lines[2:])
            speaker = ""
            sm = re.match(r"^(說話者\d+|Speaker\s*\d+)：(.+)$", content, re.DOTALL)
            if sm:
                speaker = sm.group(1)
                content = sm.group(2).strip()
            out.append({
                "start": m.group(1), "end": m.group(2),
                "speaker": speaker,  "text": content,
            })
        return out

    def _init_rows(self, entries: list[dict]):
        self._rows = [
            {
                "start":   ctk.StringVar(value=e["start"]),
                "end":     ctk.StringVar(value=e["end"]),
                "speaker": ctk.StringVar(value=e["speaker"]),
                "text":    ctk.StringVar(value=e["text"]),
            }
            for e in entries
        ]

    @staticmethod
    def _ts_to_sec(ts: str) -> float:
        try:
            h, m, rest = ts.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        except Exception:
            return 0.0

    # ── UI 建構 ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.title(f"字幕驗證編輯器 — {self.srt_path.name}")
        self.geometry("960x680")
        self.resizable(True, True)
        self.minsize(720, 420)
        self.grab_set()

        if self.has_speakers:
            self._build_spk_name_bar()
        self._build_header()

        # ── 分頁導覽列（置於清單上方）────────────────────────────────
        self._pager_bar = ctk.CTkFrame(self, fg_color=("gray88", "#18182A"),
                                       corner_radius=0, height=36)
        self._pager_bar.pack(fill="x", padx=6, pady=(0, 2))
        self._pager_bar.pack_propagate(False)
        self._build_pager()

        self._sf = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._sf.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        self._rebuild_rows()
        self._build_bottom()

    def _build_spk_name_bar(self):
        bar = ctk.CTkFrame(self, fg_color=("gray90", "#1A1A2E"), corner_radius=8)
        bar.pack(fill="x", padx=6, pady=(8, 2))
        ctk.CTkLabel(
            bar, text="說話者命名：",
            font=("Microsoft JhengHei", 12, "bold"), text_color=("gray40", "#888899"),
        ).pack(side="left", padx=(10, 8), pady=6)
        for i, sid in enumerate(self._all_spk_ids):
            accent = self._SPK_ACCENT[i % len(self._SPK_ACCENT)]
            ctk.CTkLabel(
                bar, text=f"{sid}：",
                font=("Microsoft JhengHei", 12), text_color=accent,
            ).pack(side="left", padx=(0, 2))
            ctk.CTkEntry(
                bar, textvariable=self._spk_name_vars[sid],
                width=80, height=28, font=("Microsoft JhengHei", 12),
            ).pack(side="left", padx=(0, 14))
        ctk.CTkButton(
            bar, text="確定", width=60, height=28,
            fg_color="#2A3A1A", hover_color="#3A5028",
            font=("Microsoft JhengHei", 12),
            command=self._apply_spk_names,
        ).pack(side="right", padx=(0, 10))

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=("gray85", "#1E1E32"), corner_radius=0, height=26)
        hdr.pack(fill="x", padx=6, pady=(2, 0))
        hdr.pack_propagate(False)
        cols = [("  #", 36), ("起始時間", 110), (" ", 22), ("結束時間", 110)]
        if self.has_speakers:
            cols.append(("說話者", 98))
        cols.append(("字幕文字", 0))
        cols.append(("操作", 138))
        for txt, w in cols:
            kw: dict = dict(
                text=txt, font=("Microsoft JhengHei", 11),
                text_color=("gray35", "#55556A"), anchor="w",
            )
            if w:
                kw["width"] = w
            ctk.CTkLabel(hdr, **kw).pack(side="left", padx=(4, 0))

    @property
    def _total_pages(self) -> int:
        """總頁數（至少 1 頁）。"""
        return max(1, math.ceil(len(self._rows) / self.PAGE_SIZE))

    def _page_slice(self) -> tuple[int, int]:
        """回傳目前頁對應的 self._rows 切片範圍 (start, end)。"""
        s = self._page * self.PAGE_SIZE
        e = min(s + self.PAGE_SIZE, len(self._rows))
        return s, e

    def _build_pager(self):
        """第一次建立分頁 bar 內容（初始化時呼叫一次）。"""
        bar = self._pager_bar
        for w in bar.winfo_children():
            w.destroy()

        # 上頁
        self._btn_prev = ctk.CTkButton(
            bar, text="◀ 上頁", width=72, height=26,
            fg_color="#1A2040", hover_color="#263060",
            font=("Microsoft JhengHei", 11),
            command=self._prev_page,
        )
        self._btn_prev.pack(side="left", padx=(8, 4), pady=5)

        # 頁碼顯示 + 跳頁輸入
        self._page_info_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            bar, textvariable=self._page_info_var,
            font=("Microsoft JhengHei", 11),
            text_color=("gray25", "#8888AA"),
            width=120, anchor="center",
        ).pack(side="left", padx=4)

        ctk.CTkLabel(
            bar, text="跳至：",
            font=("Microsoft JhengHei", 11),
            text_color=("gray40", "#555577"),
        ).pack(side="left", padx=(8, 2))

        self._jump_var = ctk.StringVar(value="")
        jump_entry = ctk.CTkEntry(
            bar, textvariable=self._jump_var,
            width=52, height=26, font=("Consolas", 11), justify="center",
        )
        jump_entry.pack(side="left", padx=(0, 4))
        jump_entry.bind("<Return>", lambda e: self._goto_page())

        ctk.CTkButton(
            bar, text="Go", width=44, height=26,
            fg_color="#243030", hover_color="#35484A",
            font=("Microsoft JhengHei", 11),
            command=self._goto_page,
        ).pack(side="left", padx=(0, 12))

        # 下頁
        self._btn_next = ctk.CTkButton(
            bar, text="下頁 ▶", width=72, height=26,
            fg_color="#1A2040", hover_color="#263060",
            font=("Microsoft JhengHei", 11),
            command=self._next_page,
        )
        self._btn_next.pack(side="left", padx=(0, 8))

        # 行數統計（右側）
        self._row_count_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            bar, textvariable=self._row_count_var,
            font=("Microsoft JhengHei", 10),
            text_color=("gray50", "#555566"),
        ).pack(side="right", padx=(0, 12))

        self._refresh_pager()

    def _refresh_pager(self):
        """更新分頁 bar 的頁碼文字與按鈕狀態（每次換頁後呼叫）。"""
        total = self._total_pages
        # 確保 _page 在合法範圍
        self._page = max(0, min(self._page, total - 1))
        s, e = self._page_slice()
        self._page_info_var.set(f"第 {self._page + 1} / {total} 頁")
        self._jump_var.set(str(self._page + 1))
        self._row_count_var.set(
            f"顯示第 {s + 1}–{e} 行  （共 {len(self._rows)} 行）"
        )
        # 上/下頁按鈕啟停
        self._btn_prev.configure(
            state="normal" if self._page > 0 else "disabled"
        )
        self._btn_next.configure(
            state="normal" if self._page < total - 1 else "disabled"
        )

    def _prev_page(self):
        if self._page > 0:
            self._page -= 1
            self._rebuild_rows()

    def _next_page(self):
        if self._page < self._total_pages - 1:
            self._page += 1
            self._rebuild_rows()

    def _goto_page(self):
        """依跳頁輸入框的值切換頁碼。"""
        try:
            target = int(self._jump_var.get()) - 1  # 轉為 0-based
        except ValueError:
            return
        target = max(0, min(target, self._total_pages - 1))
        if target != self._page:
            self._page = target
            self._rebuild_rows()

    def _rebuild_rows(self):
        """重建目前分頁的 widget，並更新分頁導覽列。"""
        for w in self._sf.winfo_children():
            w.destroy()

        # 只在第一頁頂端顯示「在最前插入」
        if self._page == 0:
            top_bar = ctk.CTkFrame(self._sf, fg_color=("gray88", "#181828"), corner_radius=4)
            top_bar.pack(fill="x", padx=2, pady=(0, 2))
            ctk.CTkButton(
                top_bar, text="⊕  在最前面插入空白段", width=200, height=24,
                fg_color="#1B2A1B", hover_color="#253825",
                font=("Microsoft JhengHei", 11),
                command=self._insert_at_top,
            ).pack(side="left", padx=6, pady=3)

        # 只建立當頁的 widget（降低 widget 數量，避免卡頓）
        s, e = self._page_slice()
        for i in range(s, e):
            self._build_one_row(i, self._rows[i])

        # 更新分頁導覽列
        self._refresh_pager()

    def _build_one_row(self, idx: int, row: dict):
        spk_id = row["speaker"].get()
        ci = self._all_spk_ids.index(spk_id) if spk_id in self._all_spk_ids else -1

        # 空白段使用不同背景色
        is_blank = not row["text"].get().strip()
        if is_blank:
            bg = self._blank_bg
        elif self.has_speakers and ci >= 0:
            bg = self._SPK_ROW_BG[ci % len(self._SPK_ROW_BG)]
        else:
            bg = ("gray95", "#1C1C1C") if idx % 2 == 0 else ("gray91", "#222228")

        # 行 frame（pack 到 scroll frame）
        fr = ctk.CTkFrame(self._sf, fg_color=bg, corner_radius=4)
        fr.pack(fill="x", padx=2, pady=1)
        row["_frame_ref"] = fr  # 儲存 frame 參照，供 _on_spk_change 直接更新

        # 文字欄使用 grid weight 佔滿剩餘寬度
        if self.has_speakers:
            text_col = 6
        else:
            text_col = 4
        fr.columnconfigure(text_col, weight=1)

        col = 0
        # 序號
        ctk.CTkLabel(
            fr, text=str(idx + 1), width=32, anchor="e",
            font=("Consolas", 11), text_color=("gray35", "#555566"),
        ).grid(row=0, column=col, padx=(6, 2), pady=5)
        col += 1

        # 起始時間
        ctk.CTkEntry(
            fr, textvariable=row["start"], width=108, height=28,
            font=FONT_MONO, justify="center",
        ).grid(row=0, column=col, padx=(2, 0), pady=4)
        col += 1

        # 箭頭
        ctk.CTkLabel(
            fr, text="→", width=22, font=("Microsoft JhengHei", 12),
            text_color=("gray45", "#444455"),
        ).grid(row=0, column=col)
        col += 1

        # 結束時間
        ctk.CTkEntry(
            fr, textvariable=row["end"], width=108, height=28,
            font=FONT_MONO, justify="center",
        ).grid(row=0, column=col, padx=(0, 4), pady=4)
        col += 1

        # 說話者下拉（多說話者模式）
        if self.has_speakers:
            accent = self._SPK_ACCENT[ci % len(self._SPK_ACCENT)] if ci >= 0 else "#666677"
            combo = ctk.CTkComboBox(
                fr, variable=row["speaker"], values=list(self._all_spk_ids),
                width=94, height=28, font=("Microsoft JhengHei", 11),
                button_color=accent, border_color=accent,
                command=lambda v, i=idx: self._on_spk_change(i),
            )
            combo.grid(row=0, column=col, padx=(0, 4), pady=4)
            row["_combo_ref"] = combo  # 儲存 combo 參照
            col += 1

            # 說話者顯示名稱 label
            if "_name_var" not in row:
                init_name = self._spk_name_vars[spk_id].get() if spk_id in self._spk_name_vars else ""
                row["_name_var"] = ctk.StringVar(value=init_name)
            ctk.CTkLabel(
                fr, textvariable=row["_name_var"],
                font=("Microsoft JhengHei", 10), text_color=("gray40", "#888899"),
                width=70, anchor="w",
            ).grid(row=0, column=col, padx=(0, 4))
            col += 1

        # 字幕文字（sticky="ew" 填滿剩餘寬度）
        ctk.CTkEntry(
            fr, textvariable=row["text"], height=28,
            font=("Microsoft JhengHei", 12),
        ).grid(row=0, column=col, sticky="ew", padx=(0, 4), pady=4)
        col += 1

        # 操作按鈕組
        btn_fr = ctk.CTkFrame(fr, fg_color="transparent")
        btn_fr.grid(row=0, column=col, padx=(0, 6), pady=4)

        ctk.CTkButton(
            btn_fr, text="+", width=26, height=26,
            fg_color="#1B4A1B", hover_color="#28602A",
            font=("Consolas", 13, "bold"),
            command=lambda i=idx: self._add_after(i),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_fr, text="−", width=26, height=26,
            fg_color="#4A1B1B", hover_color="#602828",
            font=("Consolas", 13, "bold"),
            command=lambda i=idx: self._delete(i),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_fr, text="▶", width=34, height=26,
            fg_color="#1A3A5C", hover_color="#265A8A",
            font=("Microsoft JhengHei", 11),
            command=lambda r=row: self._play(r),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_fr, text="⋯", width=34, height=26,
            fg_color="#2A1A4A", hover_color="#3D2870",
            font=("Segoe UI Emoji", 13),
            command=lambda i=idx: self._open_detail(i),
        ).pack(side="left", padx=(2, 0))

    def _apply_spk_names(self):
        """說話者命名確定：即時更新所有行的顯示名稱 StringVar。"""
        for row in self._rows:
            sid = row["speaker"].get()
            if sid in self._spk_name_vars and "_name_var" in row:
                row["_name_var"].set(self._spk_name_vars[sid].get())

    def _open_detail(self, idx: int):
        """開啟字幕詳細時間軸編輯視窗。"""
        SubtitleDetailEditor(self, idx)

    # ── 行操作 ───────────────────────────────────────────────────────

    def _on_spk_change(self, idx: int):
        """說話者切換後只更新該行的顏色，避免全部重建閃爍。"""
        row = self._rows[idx]
        spk_id = row["speaker"].get()
        ci = self._all_spk_ids.index(spk_id) if spk_id in self._all_spk_ids else -1
        is_blank = not row["text"].get().strip()
        if is_blank:
            new_bg = self._blank_bg
        elif self.has_speakers and ci >= 0:
            new_bg = self._SPK_ROW_BG[ci % len(self._SPK_ROW_BG)]
        else:
            new_bg = ("gray95", "#1C1C1C") if idx % 2 == 0 else ("gray91", "#222228")
        new_accent = self._SPK_ACCENT[ci % len(self._SPK_ACCENT)] if ci >= 0 else "#666677"
        if "_frame_ref" in row:
            row["_frame_ref"].configure(fg_color=new_bg)
        if "_combo_ref" in row:
            row["_combo_ref"].configure(button_color=new_accent, border_color=new_accent)
        if "_name_var" in row and spk_id in self._spk_name_vars:
            row["_name_var"].set(self._spk_name_vars[spk_id].get())

    def _add_after(self, idx: int):
        """在 idx 後插入空白行，起迄時間繼承當前行的結束點。"""
        cur_end = self._rows[idx]["end"].get()
        self._rows.insert(idx + 1, {
            "start":   ctk.StringVar(value=cur_end),
            "end":     ctk.StringVar(value=cur_end),
            "speaker": ctk.StringVar(value=self._rows[idx]["speaker"].get()),
            "text":    ctk.StringVar(value=""),
        })
        # 插入後跳到新行所在的頁
        self._page = (idx + 1) // self.PAGE_SIZE
        self._rebuild_rows()

    def _delete(self, idx: int):
        if len(self._rows) <= 1:
            return
        del self._rows[idx]
        # 刪除後確保頁碼不超出範圍（_refresh_pager 內已做 clamp，但需先更新）
        self._page = min(self._page, max(0, self._total_pages - 1))
        self._rebuild_rows()

    def _insert_at_top(self):
        """在最前面插入空白段（起迄時間設為 00:00:00,000）。"""
        first_start = self._rows[0]["start"].get() if self._rows else "00:00:00,000"
        self._rows.insert(0, {
            "start":   ctk.StringVar(value="00:00:00,000"),
            "end":     ctk.StringVar(value=first_start),
            "speaker": ctk.StringVar(value=self._rows[0]["speaker"].get() if self._rows else ""),
            "text":    ctk.StringVar(value=""),
        })
        self._page = 0  # 回到第一頁，讓使用者看到剛插入的行
        self._rebuild_rows()

    def _reorder_and_fix(self):
        """依起始時間排序所有行，並消除時間重疊（前一行的結束截到下一行的起始）。"""
        if not self._rows:
            return
        self._rows.sort(key=lambda r: self._ts_to_sec(r["start"].get()))
        for i in range(len(self._rows) - 1):
            e_i  = self._ts_to_sec(self._rows[i]["end"].get())
            s_n  = self._ts_to_sec(self._rows[i + 1]["start"].get())
            if e_i > s_n:
                self._rows[i]["end"].set(self._sec_to_ts(s_n))
        self._page = 0  # 排序後回到第一頁
        self._rebuild_rows()

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

    # ── 音訊播放 ─────────────────────────────────────────────────────

    def _play(self, row: dict):
        """段落試聽：從起始時間播放到結束時間後自動停止。"""
        try:
            import sounddevice as sd
            sd.stop()
            if self._audio_data is None:
                return
            s  = self._ts_to_sec(row["start"].get())
            e  = self._ts_to_sec(row["end"].get())
            if e <= s:
                return
            si  = max(0, int(s * self._audio_sr))
            ei  = min(len(self._audio_data), int(e * self._audio_sr))
            seg = self._audio_data[si:ei]
            if len(seg) > 0:
                sd.play(seg, self._audio_sr)
        except Exception:
            pass

    def _load_audio(self):
        """背景執行緒載入音訊（soundfile 優先，librosa 備用）。"""
        try:
            import soundfile as sf
            data, sr = sf.read(str(self.audio_path), always_2d=False, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                # 讓下方 catch block 用 ffmpeg_utils 處理重採樣
                raise ValueError("Requires 16000Hz")
            self._audio_data = data
            self._audio_sr   = 16000
        except Exception:
            try:
                from ffmpeg_utils import decode_audio_to_numpy, find_ffmpeg
                ffmpeg_exe = find_ffmpeg()
                if ffmpeg_exe:
                    data = decode_audio_to_numpy(self.audio_path, ffmpeg_exe, sr=16000)
                    self._audio_data = data
                    self._audio_sr   = 16000
                else:
                    self._audio_data = None
            except Exception:
                self._audio_data = None

    # ── 底部操作列 ───────────────────────────────────────────────────

    def _build_bottom(self):
        bot = ctk.CTkFrame(self, fg_color=("gray85", "#14141E"), corner_radius=0, height=54)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)

        # ── 左側：暫存 + 狀態 + 工具
        ctk.CTkButton(
            bot, text="💾 暫存", width=84, height=36,
            fg_color="#1A2A40", hover_color="#243652",
            font=("Microsoft JhengHei", 12),
            command=self._save_draft,
        ).pack(side="left", padx=(10, 4), pady=9)

        ctk.CTkLabel(
            bot, textvariable=self._draft_status_var,
            font=("Microsoft JhengHei", 11), text_color=("gray40", "#4A7A8A"),
            width=150, anchor="w",
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            bot, text="↕ 重整", width=80, height=36,
            fg_color="#1A2A3A", hover_color="#263850",
            font=("Microsoft JhengHei", 12),
            command=self._reorder_and_fix,
        ).pack(side="left", padx=(0, 4), pady=9)

        ctk.CTkButton(
            bot, text="📂 載入字幕", width=100, height=36,
            fg_color="#241A30", hover_color="#362448",
            font=("Microsoft JhengHei", 12),
            command=self._load_srt_dialog,
        ).pack(side="left", padx=(0, 4), pady=9)

        # ── 新增：純文字輸出 + 分段音訊
        ctk.CTkButton(
            bot, text="📄 純文字", width=88, height=36,
            fg_color="#1A3A2A", hover_color="#245538",
            font=("Microsoft JhengHei", 12),
            command=self._export_plain_text,
        ).pack(side="left", padx=(0, 4), pady=9)

        ctk.CTkButton(
            bot, text="✂ 分段音訊", width=96, height=36,
            fg_color="#2A2A1A", hover_color="#404028",
            font=("Microsoft JhengHei", 12),
            command=self._export_audio_segments,
        ).pack(side="left", padx=(0, 4), pady=9)

        # ── 右側：取消 + 完成關閉
        ctk.CTkButton(
            bot, text="✖  取消", width=88, height=36,
            fg_color="#38181A", hover_color="#552428",
            font=("Microsoft JhengHei", 13),
            command=self._cancel,
        ).pack(side="right", padx=(4, 10), pady=9)

        ctk.CTkButton(
            bot, text="✔  完成關閉", width=120, height=36,
            fg_color="#183A1A", hover_color="#245528",
            font=("Microsoft JhengHei", 13, "bold"),
            command=self._save,
        ).pack(side="right", padx=(0, 4), pady=9)

    # ── 輸出功能 ───────────────────────────────────────────

    def _export_plain_text(self):
        """???字幕所有行的文字，不含時間軸，存為 .txt。"""
        out_path = filedialog.asksaveasfilename(
            parent=self,
            title="儲存純文字",
            defaultextension=".txt",
            initialfile=self.srt_path.stem + "_text.txt",
            filetypes=[("Text Files", "*.txt"), ("所有檔案", "*.*")],
            initialdir=str(self.srt_path.parent),
        )
        if not out_path:
            return
        lines: list[str] = []
        for row in self._rows:
            text = row["text"].get().strip()
            if not text:
                continue
            # 將每行加回「。」（使繼徜帶標點的语感）
            spk = row["speaker"].get()
            if self.has_speakers and spk and spk in self._spk_name_vars:
                display = self._spk_name_vars[spk].get() or spk
                text = f"{display}：{text}"
            lines.append(text)
        try:
            Path(out_path).write_text("\n".join(lines), encoding="utf-8")
            from tkinter import messagebox
            messagebox.showinfo("已完成", f"純文字已儲存至：\n{out_path}", parent=self)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("儲存失敗", str(e), parent=self)

    def _export_audio_segments(self):
        """???音訊檔依字幕時間軸切残，每段存為獨立 wav。
        檔名格式： 00001-10000.wav（起始毫秒 - 結束毫秒）。
        """
        if self._audio_data is None:
            from tkinter import messagebox
            messagebox.showwarning("無音訊", "請先載入音訊檔才能分段輸出。", parent=self)
            return

        out_dir = filedialog.askdirectory(
            parent=self, title="選擇分段音訊儲存目錄"
        )
        if not out_dir:
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ok = 0
        errors: list[str] = []
        for row in self._rows:
            text = row["text"].get().strip()
            if not text:
                continue
            try:
                s = self._ts_to_sec(row["start"].get())
                e = self._ts_to_sec(row["end"].get())
                if e <= s:
                    continue
                si  = max(0, int(s * self._audio_sr))
                ei  = min(len(self._audio_data), int(e * self._audio_sr))
                seg = self._audio_data[si:ei]
                if len(seg) == 0:
                    continue
                # 檔名： 00001-10000.wav（20位落小毫秒）
                s_ms = int(round(s * 1000))
                e_ms = int(round(e * 1000))
                fname = f"{s_ms:08d}-{e_ms:08d}.wav"

                try:
                    import soundfile as sf
                    sf.write(str(out_path / fname), seg, self._audio_sr, subtype="PCM_16")
                except ImportError:
                    import wave, struct
                    import numpy as np
                    seg16 = (seg * 32767).clip(-32768, 32767).astype(np.int16)
                    with wave.open(str(out_path / fname), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self._audio_sr)
                        wf.writeframes(seg16.tobytes())
                ok += 1
            except Exception as ex:
                errors.append(f"{row['start'].get()}: {ex}")

        from tkinter import messagebox
        msg = f"分段完成：{ok} 個音訊檔 → {out_path}"
        if errors:
            msg += f"\n失敗 {len(errors)} 個：\n" + "\n".join(errors[:5])
        messagebox.showinfo("區段音訊已輸出", msg, parent=self)

    # ── 草稿路徑 ─────────────────────────────────────────────────────

    @property
    def _draft_path(self) -> Path:
        return self.srt_path.parent / f"{self.srt_path.stem}_draft.srt"

    # ── SRT 寫入（暫存與最終共用）────────────────────────────────────

    def _write_srt(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for i, row in enumerate(self._rows, 1):
                start = row["start"].get()
                end   = row["end"].get()
                text  = row["text"].get().strip()
                spk   = row["speaker"].get()
                if self.has_speakers and spk and spk in self._spk_name_vars:
                    display = self._spk_name_vars[spk].get() or spk
                    prefix  = f"{display}："
                else:
                    prefix = ""
                f.write(f"{i}\n{start} --> {end}\n{prefix}{text}\n\n")

    # ── 暫存 ─────────────────────────────────────────────────────────

    def _save_draft(self):
        """將目前狀態暫存至 _draft.srt，不關閉視窗。"""
        try:
            self._write_srt(self._draft_path)
            ts = datetime.now().strftime("%H:%M:%S")
            self._draft_status_var.set(f"暫存於 {ts}")
        except Exception as e:
            messagebox.showerror("暫存失敗", str(e), parent=self)

    def _check_draft(self):
        """啟動時若發現草稿，詢問是否繼續上次的編輯。"""
        dp = self._draft_path
        if not dp.exists():
            return
        if messagebox.askyesno(
            "發現暫存草稿",
            f"找到上次的暫存草稿：\n{dp.name}\n\n是否從草稿繼續編輯？\n（選「否」則從原始字幕重新開始）",
            parent=self,
        ):
            self._load_srt_file(dp)
            self._draft_status_var.set(f"已載入草稿 {dp.name}")

    # ── 載入字幕 ─────────────────────────────────────────────────────

    def _load_srt_file(self, path: Path):
        """解析並載入任意 SRT 檔案，保留說話者命名對應。"""
        try:
            raw = self._parse_srt(path)
        except Exception as e:
            messagebox.showerror("載入失敗", str(e), parent=self)
            return
        new_spk_ids = sorted({e["speaker"] for e in raw if e["speaker"]})
        for i, sid in enumerate(new_spk_ids):
            if sid not in self._spk_name_vars:
                self._spk_name_vars[sid] = ctk.StringVar(value=f"說話者{i + 1}")
        self._all_spk_ids = new_spk_ids
        self.has_speakers = bool(self._all_spk_ids) and self.diarize_mode
        self._init_rows(raw)
        self._page = 0  # 載入新字幕時回到第一頁
        self._rebuild_rows()

    def _load_srt_dialog(self):
        """開啟檔案對話框，選擇要載入的 SRT。"""
        path = filedialog.askopenfilename(
            parent=self,
            title="選擇要載入的字幕檔",
            filetypes=[("SRT 字幕", "*.srt"), ("所有檔案", "*.*")],
            initialdir=str(self.srt_path.parent),
        )
        if not path:
            return
        self._load_srt_file(Path(path))
        self._draft_status_var.set(f"已載入 {Path(path).name}")

    # ── 音訊控制與關閉 ───────────────────────────────────────────────

    def _stop_audio(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def _cancel(self):
        """取消並關閉；若有暫存草稿，詢問是否保留以備下次繼續。"""
        dp = self._draft_path
        if dp.exists():
            keep = messagebox.askyesnocancel(
                "保留草稿？",
                f"尚有暫存草稿 {dp.name}。\n\n"
                "是  → 保留草稿，下次開啟可繼續編輯\n"
                "否  → 刪除草稿並關閉\n"
                "取消 → 回到編輯",
                parent=self,
            )
            if keep is None:   # 取消 → 回到編輯
                return
            if not keep:       # 否 → 刪除草稿
                try:
                    dp.unlink()
                except Exception:
                    pass
        self._stop_audio()
        self.destroy()

    def _save(self):
        """完成確認，儲存為最終 _edited_<時間戳>.srt，刪除草稿，關閉視窗。"""
        self._stop_audio()
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.srt_path.parent / f"{self.srt_path.stem}_edited_{ts}.srt"
        try:
            self._write_srt(out_path)
        except Exception as e:
            messagebox.showerror("儲存失敗", str(e), parent=self)
            return
        try:
            self._draft_path.unlink(missing_ok=True)
        except Exception:
            pass
        messagebox.showinfo("已完成", f"字幕已儲存至：\n{out_path}", parent=self)
        self.destroy()
