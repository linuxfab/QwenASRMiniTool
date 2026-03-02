"""subtitle_editor.py — 向後相容 re-export

原始內容已拆分為：
  - subtitle_detail_editor.py  （SubtitleDetailEditor，時間軸視覺化）
  - subtitle_editor_window.py  （SubtitleEditorWindow，字幕驗證/編輯主視窗）

既有的 import 仍可照常使用：
    from subtitle_editor import SubtitleDetailEditor, SubtitleEditorWindow
"""

from subtitle_detail_editor import SubtitleDetailEditor     # noqa: F401
from subtitle_editor_window import SubtitleEditorWindow     # noqa: F401
