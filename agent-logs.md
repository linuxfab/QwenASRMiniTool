# Agent Logs

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
