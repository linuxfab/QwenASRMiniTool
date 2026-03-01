# Agent 專案修改日誌

* 更新日期時間：2026-03-01 12:02
* 重點：修復 4 項 bug 與改進 — _srt_ts NameError、硬編碼路徑、Thread Safety、即時字幕時間軸
* 影響：
  * 修復 `app.py` 與 `app-gpu.py` 的 `_on_rt_save` 中使用未定義的 `_srt_ts` 函式（應為 `srt_ts`），此 bug 會在儲存即時字幕時直接 crash
  * 修復 `generate_prompt_template.py` 硬編碼絕對路徑，改為相對路徑並支援 CLI 引數覆寫
  * `app.py` 新增 `threading.Lock`（`_convert_lock` / `_rt_lock`）保護 `_converting` 和 `_rt_log` 的多線程存取
  * `asr_utils.py` 中 `RealtimeManager` 新增 chunk 計數器追蹤錄音累計時間，`on_text` 回呼簽名變更為 `(text, start_sec, end_sec)`
  * `app.py` / `app-gpu.py` 的 `_rt_log` 型別從 `list[str]` 改為 `list[tuple[str, float, float]]`，`_on_rt_save` 使用真實時間軸寫入 SRT
* 結果：修復 2 個會 crash 的 bug，消除 3 處 race condition 風險，即時字幕 SRT 時間軸從假的固定 5 秒間隔改為基於 VAD 的真實錄音位置。
* 更新者：antigravity agent

---
* 更新日期時間：2026-02-28 14:06
* 重點：將 engine-decoupling 功能合併回主分支
* 影響：合併 app.py, engine 等多個模組重構程式碼，統一 API 介面。
* 結果：整合完成，未來有統一入口。
* 更新者：antigravity agent

---
* 更新日期時間：2026-02-28 09:35
* 重點：架構重構 — 將 ASR 引擎抽離為獨立 `engine/` 模組（Core Engine API）
* 影響：
  * 新增 `engine/__init__.py`, `engine/base.py`, `engine/factory.py` — 統一介面與工廠函式
  * 新增 `engine/openvino_backend.py` — 從 `app.py` 提取 `ASREngine` + `ASREngine1p7B`
  * 新增 `engine/vulkan_backend.py` — 從 `chatllm_engine.py` 提取 `ChatLLMASREngine`
  * 新增 `engine/cuda_backend.py` — 從 `app-gpu.py` 提取 `GPUASREngine`
  * 修改 `app.py`：移除 ~370 行內嵌引擎類別，改用 `create_engine()` 工廠函式；消除 `_g_output_simplified` 全域變數
* 結果：`process_file()` 統一在 `ASREngineBase` 中實作，消除 3 份重複（~180 行 × 3）。未來新增後端（如 ONNX Runtime, TensorRT）或新前端（如 FastAPI, PyQt）只需實作 `load()` + `transcribe()` 兩個方法。
* 更新者：antigravity agent

---

* 更新日期時間：2026-02-27 15:00
* 重點：啟用 OpenVINO 模型編譯快取 (Model Caching)。
* 影響：
  * 於 `app.py` 中的 `ASREngine` 與 `ChatLLMASREngine` 初始化時，加入 `core.set_property({'CACHE_DIR': str(cache_dir)})` 設定。
  * 快取檔案將存放於 `models/ov_cache` 目錄下。
* 結果：大幅縮短後續啟動與切換模型時的 `core.compile_model()` 編譯耗時（可相差 10 秒以上）。
* 更新者：antigravity agent

---

* 更新日期時間：2026-02-27 14:55
* 重點：移除 `librosa` 依賴，改用 `ffmpeg` pipe 加速載入長音檔。
* 影響：
  * 在 `ffmpeg_utils.py` 中新增 `decode_audio_to_numpy()` 和 `get_audio_duration()`，直接將音訊解碼至 `float32 numpy array`，省去 Python 端 resampling 負載。
  * 替換 `app.py`、`app-gpu.py`、`chatllm_engine.py`、`streamlit_vulkan.py`、`subtitle_editor.py` 及 `batch_tab.py` 中共 8 處的 `librosa.load` 與 `librosa.get_duration`。
  * 移除 `pyproject.toml`、`requirements.txt` 及 `requirements-gpu.txt` 中的 `librosa` 依賴並透過 `uv sync` 更新。
* 結果：1 小時以上的長音檔載入極快（且記憶體佔用極低），並解決了先前因 Python 資源耗盡而閃退的問題，同時安裝環境更輕量化。
* 更新者：antigravity agent

---Logs

## 2026-02-27 12:55

**重點**: 遷移開發環境至 `uv` 管理

**影響**:
- 初始化 `pyproject.toml`，整合 CPU 與 GPU 兩版依賴
- 設定 `requires-python = ">=3.10, <3.13"` 以確保 PyTorch CUDA (cu121) 相容性
- 釘選 Python 3.12 版本於 `.python-version`
- 設定 PyTorch 為專屬 Index (`https://download.pytorch.org/whl/cu121`)
- 更新 `README.md` 指引開發者使用 `uv sync` 與 `uv run`

**結果**: 統一開發環境配置，解決 Python 3.14 系統不相容 PyTorch 的問題，簡化環境建立流程。

**更新者**: antigravity agent

## 2026-02-27 12:45

**重點**: 重構重複程式碼，建立 `asr_utils.py` 共用模組

**影響**:
- 新建 `asr_utils.py`，集中 `detect_speech_groups`、`split_to_lines`、`srt_ts`、`assign_ts`、`RealtimeManager` 及相關常數
- 修改 `app.py`：移除 ~250 行重複碼，改用 `from asr_utils import ...`
- 修改 `app-gpu.py`：移除 ~230 行重複碼（保留 ForcedAligner 專用函式），改用 import
- 修改 `chatllm_engine.py`：移除 ~100 行重複碼，改用 import
- 修改 `streamlit_vulkan.py`：移除 ~95 行重複碼，改用 import
- `_split_to_lines` 統一為精細版（逐字元解析、英文整字保留詞界空格），改善 `chatllm_engine` 和 `streamlit_vulkan` 的斷句品質
- `RealtimeManager` 統一為 `app.py` 的動態 buffer 版本

**結果**: 消除約 675 行重複程式碼，降低維護成本。未來修改斷句邏輯或 VAD 參數只需改一處。

**更新者**: antigravity agent
