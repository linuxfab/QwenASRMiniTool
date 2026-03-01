"""
onboarding.py — 初始設定引導畫面

首次啟動時顯示，引導使用者選擇推理後端（OpenVINO / Vulkan）、
選定模型大小並下載。完成後回傳 settings dict。

主要入口：
  run_onboarding(parent_app, chosen: list, evt: threading.Event)
"""
from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

FONT_BODY = ("Microsoft JhengHei", 13)


def run_onboarding(
    parent_app,
    chosen: list,
    evt: threading.Event,
    *,
    all_devices: dict,
    default_model_dir: Path,
    bin_path: Path,
    chatllm_dir: Path,
    set_status,
    load_settings,
):
    """主執行緒：顯示初始設定引導畫面（modal）。

    Parameters
    ----------
    parent_app : ctk.CTk
        父視窗（用於定位與 after() 排程）
    chosen : list
        chosen[0] 用於回傳結果 dict（或 None 表示取消）
    evt : threading.Event
        下載完成或取消後 set，解除 _startup_check 的 wait
    all_devices : dict
        偵測到的裝置資訊 {"igpu": [...], "nvidia_amd": [...]}
    default_model_dir : Path
        預設模型路徑（如 ov_models/）
    bin_path : Path
        chatllm .bin 模型預設路徑
    chatllm_dir : Path
        chatllm DLL 目錄路徑
    set_status : callable
        更新主視窗狀態列的 callback
    load_settings : callable
        載入現有 settings 的函式
    """
    dlg = ctk.CTkToplevel(parent_app)
    dlg.title("QwenASR 初始設定")
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.focus_set()

    parent_app.update_idletasks()
    scr_h  = dlg.winfo_screenheight()
    dlg_w  = 640
    dlg_h  = min(scr_h - 120, 660)
    x = parent_app.winfo_x() + (parent_app.winfo_width()  - dlg_w) // 2
    y = max(40, parent_app.winfo_y() + (parent_app.winfo_height() - dlg_h) // 2)
    dlg.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")

    # ══ 底部按鈕列（先 pack → 永遠可見，不被內容擠走）══════════════
    bottom_bar = ctk.CTkFrame(dlg, fg_color="#252525", height=72)
    bottom_bar.pack(side="bottom", fill="x")
    bottom_bar.pack_propagate(False)

    # 分隔線
    ctk.CTkFrame(dlg, fg_color="#3A3A3A", height=1).pack(
        side="bottom", fill="x"
    )

    confirm_btn = ctk.CTkButton(
        bottom_bar,
        text="✔  確認並開始下載",
        width=200, height=44,
        font=("Microsoft JhengHei", 14, "bold"),
        corner_radius=8,
    )
    confirm_btn.pack(side="left", padx=(24, 10), pady=14)

    ctk.CTkButton(
        bottom_bar,
        text="取消",
        width=110, height=44,
        font=("Microsoft JhengHei", 14),
        fg_color="gray35", hover_color="gray25",
        corner_radius=8,
        command=lambda: _cancel_onboarding(),
    ).pack(side="left", padx=0, pady=14)

    # ══ 可捲動內容區（低解析度也能捲動到底）═════════════════════════
    scroll = ctk.CTkScrollableFrame(dlg, fg_color="transparent")
    scroll.pack(fill="both", expand=True)

    # ── 標題 ──────────────────────────────────────────────────────
    ctk.CTkLabel(
        scroll, text="🎙  QwenASR 初始設定",
        font=("Microsoft JhengHei", 18, "bold"), anchor="w",
    ).pack(fill="x", padx=24, pady=(20, 4))

    ctk.CTkLabel(
        scroll, text="首次啟動需要選擇推理方式並下載對應模型。",
        font=FONT_BODY, text_color="#AAAAAA", anchor="w",
    ).pack(fill="x", padx=24, pady=(0, 12))

    # ── 偵測到的裝置 ──────────────────────────────────────────────
    dev_frame = ctk.CTkFrame(scroll, fg_color="#1E1E1E", corner_radius=8)
    dev_frame.pack(fill="x", padx=24, pady=(0, 14))

    ctk.CTkLabel(
        dev_frame, text="偵測到的裝置", font=FONT_BODY,
        text_color="#AAAAAA", anchor="w",
    ).pack(anchor="w", padx=12, pady=(8, 2))

    ctk.CTkLabel(dev_frame, text="✅ CPU（可用）", font=FONT_BODY, anchor="w").pack(
        anchor="w", padx=20, pady=2
    )
    igpu_list   = all_devices.get("igpu", [])
    nvidia_list = all_devices.get("nvidia_amd", [])
    for g in igpu_list:
        ctk.CTkLabel(
            dev_frame, text=f"✅ Intel GPU：{g['name']}", font=FONT_BODY, anchor="w",
        ).pack(anchor="w", padx=20, pady=2)
    for g in nvidia_list:
        vram_gb = g['vram_free'] / 1_073_741_824
        ctk.CTkLabel(
            dev_frame,
            text=f"✅ GPU：{g['name']}（可用 VRAM {vram_gb:.1f} GB，Vulkan）",
            font=FONT_BODY, anchor="w",
        ).pack(anchor="w", padx=20, pady=2)
    if not igpu_list and not nvidia_list:
        ctk.CTkLabel(
            dev_frame, text="ℹ 未偵測到獨立 GPU，僅 CPU 推理可用",
            font=FONT_BODY, text_color="#888888", anchor="w",
        ).pack(anchor="w", padx=20, pady=2)
    ctk.CTkLabel(dev_frame, text="").pack(pady=2)

    # ── 後端選擇 ──────────────────────────────────────────────────
    ctk.CTkLabel(
        scroll, text="選擇推理方式：", font=FONT_BODY, anchor="w",
    ).pack(fill="x", padx=24, pady=(0, 6))

    backend_var = ctk.StringVar(value="openvino_cpu")
    opt_frame   = ctk.CTkFrame(scroll, fg_color="transparent")
    opt_frame.pack(fill="x", padx=24, pady=(0, 10))

    # CPU 選項框
    cpu_box = ctk.CTkFrame(opt_frame, fg_color="#1E1E1E", corner_radius=8)
    cpu_box.pack(fill="x", pady=(0, 6))

    ctk.CTkRadioButton(
        cpu_box, text="CPU 推理（OpenVINO）",
        variable=backend_var, value="openvino_cpu",
        font=FONT_BODY,
    ).pack(anchor="w", padx=12, pady=(10, 4))

    size_frame = ctk.CTkFrame(cpu_box, fg_color="transparent")
    size_frame.pack(fill="x", padx=32, pady=(0, 10))
    size_var = ctk.StringVar(value="0.6B")
    ctk.CTkRadioButton(
        size_frame, text="0.6B 輕量（~1.2 GB，速度快）",
        variable=size_var, value="0.6B", font=FONT_BODY,
        command=lambda: backend_var.set("openvino_cpu"),
    ).pack(side="left", padx=(0, 20))
    ctk.CTkRadioButton(
        size_frame, text="1.7B 高精度（~4.3 GB）",
        variable=size_var, value="1.7B", font=FONT_BODY,
        command=lambda: backend_var.set("openvino_cpu"),
    ).pack(side="left")

    # GPU 選項框（有 NVIDIA/AMD 才顯示）
    if nvidia_list:
        gpu_options = [f"GPU:{g['id']} ({g['name']}) [Vulkan]" for g in nvidia_list]
        gpu_box = ctk.CTkFrame(opt_frame, fg_color="#1E1E1E", corner_radius=8)
        gpu_box.pack(fill="x", pady=(0, 6))
        gpu_var = ctk.StringVar(value=gpu_options[0] if gpu_options else "")
        ctk.CTkRadioButton(
            gpu_box, text="GPU 推理（Vulkan，速度最快）",
            variable=backend_var, value="chatllm",
            font=FONT_BODY,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        for opt in gpu_options:
            ctk.CTkRadioButton(
                gpu_box, text=f"  {opt}",
                variable=gpu_var, value=opt, font=FONT_BODY,
                command=lambda: backend_var.set("chatllm"),
            ).pack(anchor="w", padx=32, pady=2)
        ctk.CTkLabel(
            gpu_box,
            text="  1.7B .bin 格式（~2.3 GB），需先下載",
            font=("Microsoft JhengHei", 11), text_color="#888888",
        ).pack(anchor="w", padx=32, pady=(0, 10))
    else:
        gpu_var = ctk.StringVar(value="")

    # ── 路徑設定（模型存放位置）────────────────────────────────────
    path_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    path_frame.pack(fill="x", padx=24, pady=(0, 8))
    ctk.CTkLabel(path_frame, text="模型存放位置：", font=FONT_BODY).pack(
        side="left", padx=(0, 6)
    )
    saved_dir = load_settings().get("model_dir", str(default_model_dir))
    path_var = ctk.StringVar(value=saved_dir)
    ctk.CTkEntry(path_frame, textvariable=path_var, width=280, font=FONT_BODY).pack(
        side="left"
    )
    def _browse_dir():
        d = filedialog.askdirectory(title="選擇模型存放資料夾", parent=dlg)
        if d:
            path_var.set(d)
    ctk.CTkButton(
        path_frame, text="瀏覽…", width=70, font=FONT_BODY,
        command=_browse_dir,
    ).pack(side="left", padx=(6, 0))

    # ── 下載進度條（平時隱藏）──────────────────────────────────────
    prog_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    prog_frame.pack(fill="x", padx=24, pady=(0, 8))
    onb_prog_lbl = ctk.CTkLabel(
        prog_frame, text="", font=("Microsoft JhengHei", 11),
        text_color="#AAAAAA", anchor="w",
    )
    onb_prog_lbl.pack(fill="x")
    onb_bar = ctk.CTkProgressBar(prog_frame, height=10)
    onb_bar.set(0)
    onb_bar.pack(fill="x")
    onb_bar.pack_forget()
    onb_prog_lbl.pack_forget()

    def _onb_progress(pct: float, msg: str):
        def _do():
            onb_bar.set(pct)
            onb_prog_lbl.configure(text=msg)
        dlg.after(0, _do)
        set_status(f"⬇ {msg}")

    def _show_onb_prog():
        onb_prog_lbl.pack(fill="x")
        onb_bar.pack(fill="x")

    def _hide_onb_prog():
        onb_bar.pack_forget()
        onb_prog_lbl.pack_forget()

    def _cancel_onboarding():
        chosen[0] = None
        dlg.destroy()
        evt.set()

    def _do_download():
        """背景執行緒：執行下載動作，完成後關閉引導畫面。"""
        from downloader import (quick_check, download_all,
                                quick_check_1p7b, download_1p7b)

        backend    = backend_var.get()
        model_path = Path(path_var.get().strip())
        model_path.mkdir(parents=True, exist_ok=True)

        # 禁用按鈕
        dlg.after(0, lambda: confirm_btn.configure(state="disabled", text="⏳  下載中…"))
        dlg.after(0, _show_onb_prog)

        try:
            if backend == "chatllm":
                # 確保 VAD 存在
                vad_dest = default_model_dir / "silero_vad_v4.onnx"
                if not vad_dest.exists():
                    set_status("⬇ 下載 VAD 模型…")
                    from downloader import _download_file, _VAD_URL
                    default_model_dir.mkdir(parents=True, exist_ok=True)
                    _download_file(_VAD_URL, vad_dest)

                # 下載 chatllm .bin 模型
                bin_dest = bin_path
                bin_dest.parent.mkdir(parents=True, exist_ok=True)
                if not bin_dest.exists():
                    set_status("⬇ 下載 chatllm 模型（~2.3 GB）…")
                    url = ("https://huggingface.co/dseditor/Collection"
                           "/resolve/main/qwen3-asr-1.7b.bin")

                    def _dl_bin():
                        import urllib.request
                        from downloader import _ssl_ctx
                        req = urllib.request.Request(
                            url,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; QwenASR)"}
                        )
                        with urllib.request.urlopen(req, context=_ssl_ctx()) as resp, \
                             open(str(bin_dest) + ".tmp", "wb") as out:
                            total = int(resp.headers.get("Content-Length", 0))
                            done  = 0
                            while True:
                                block = resp.read(65536)
                                if not block:
                                    break
                                out.write(block)
                                done += len(block)
                                if total > 0:
                                    pct = done / total
                                    mb  = done / 1_048_576
                                    tmb = total / 1_048_576
                                    dlg.after(0, lambda p=pct, m=mb, t=tmb:
                                        _onb_progress(p, f"下載模型 {m:.0f} / {t:.0f} MB"))
                        import os
                        os.replace(str(bin_dest) + ".tmp", str(bin_dest))
                    _dl_bin()

                # chatllm_dir：優先 chatllm/，fallback chatllmtest
                cl_dir = chatllm_dir if chatllm_dir.exists() else \
                         default_model_dir.parent / "chatllmtest" / "chatllm_win_x64" / "bin"

                # 選取的 GPU device
                gpu_label = gpu_var.get()

                final_settings = {
                    "backend":      "chatllm",
                    "device":       gpu_label,
                    "model_dir":    str(model_path),
                    "model_path":   str(bin_path),
                    "chatllm_dir":  str(cl_dir),
                }

            else:  # openvino_cpu
                sz = size_var.get()   # "0.6B" | "1.7B"
                # 下載 0.6B（必要）
                if not quick_check(model_path):
                    set_status("⬇ 下載 0.6B 模型…")
                    download_all(model_path, progress_cb=_onb_progress)

                # 下載 1.7B（若選擇）
                if sz == "1.7B" and not quick_check_1p7b(model_path):
                    set_status("⬇ 下載 1.7B 模型（~4.3 GB）…")
                    download_1p7b(model_path, progress_cb=_onb_progress)

                final_settings = {
                    "backend":        "openvino",
                    "device":         "CPU",
                    "cpu_model_size": sz,
                    "model_dir":      str(model_path),
                }

            dlg.after(0, lambda: _onb_progress(1.0, "下載完成！"))
            dlg.after(0, _hide_onb_prog)
            chosen[0] = final_settings
            dlg.after(0, dlg.destroy)
            evt.set()

        except Exception as e:
            err = str(e)
            dlg.after(0, _hide_onb_prog)
            dlg.after(0, lambda: confirm_btn.configure(
                state="normal", text="✔  確認並開始下載"
            ))
            dlg.after(0, lambda: messagebox.showerror(
                "下載失敗", f"下載失敗：\n{err}\n\n請確認網路連線後重試。", parent=dlg
            ))

    confirm_btn.configure(command=lambda: threading.Thread(
        target=_do_download, daemon=True,
    ).start())

    dlg.protocol("WM_DELETE_WINDOW", _cancel_onboarding)
