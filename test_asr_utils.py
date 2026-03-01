"""
test_asr_utils.py — asr_utils 純函式的單元測試

測試項目：
  - srt_ts()         : 秒數 → SRT 時間戳
  - split_to_lines() : 文字 → 斷句清單
  - assign_ts()      : 行列 + 時間範圍 → (start, end, text) 清單
"""
import pytest
from asr_utils import srt_ts, split_to_lines, assign_ts, MIN_SUB_SEC, GAP_SEC


# ══════════════════════════════════════════════════════════════════════
# srt_ts
# ══════════════════════════════════════════════════════════════════════

class TestSrtTs:
    """srt_ts: 秒 → HH:MM:SS,mmm 格式"""

    def test_zero(self):
        assert srt_ts(0) == "00:00:00,000"

    def test_small(self):
        assert srt_ts(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert srt_ts(61.123) == "00:01:01,123"

    def test_hours(self):
        assert srt_ts(3661.999) == "01:01:01,999"

    def test_rounding(self):
        # 0.9999 秒 → 應四捨五入到 1000ms = 00:00:01,000
        assert srt_ts(0.9999) == "00:00:01,000"

    def test_large(self):
        # 2 小時 30 分 45.678 秒
        assert srt_ts(9045.678) == "02:30:45,678"


# ══════════════════════════════════════════════════════════════════════
# split_to_lines
# ══════════════════════════════════════════════════════════════════════

class TestSplitToLines:
    """split_to_lines: 文字斷句"""

    def test_empty(self):
        assert split_to_lines("") == []

    def test_whitespace_only(self):
        assert split_to_lines("   ") == []

    def test_chinese_simple(self):
        lines = split_to_lines("你好世界")
        assert lines == ["你好世界"]

    def test_chinese_with_punctuation(self):
        lines = split_to_lines("你好，世界。")
        assert lines == ["你好", "世界"]

    def test_chinese_multiple_sentences(self):
        lines = split_to_lines("早安！今天天氣真好。我們出去走走吧？")
        assert "早安" in lines
        assert "今天天氣真好" in lines
        assert "我們出去走走吧" in lines

    def test_english_simple(self):
        lines = split_to_lines("Hello world")
        assert lines == ["Hello world"]

    def test_english_with_punctuation(self):
        lines = split_to_lines("Hello, world. How are you?")
        assert "Hello" in lines
        assert "world" in lines
        assert "How are you" in lines

    def test_asr_text_prefix(self):
        """<asr_text> 標記應被移除"""
        lines = split_to_lines("language zh<asr_text>你好世界")
        assert lines == ["你好世界"]

    def test_mixed_cjk_english(self):
        """中英混合"""
        lines = split_to_lines("今天是Monday，天氣很好。")
        assert any("Monday" in l for l in lines)

    def test_long_chinese_wraps(self):
        """超過 MAX_CHARS=20 的中文字串應強制換行"""
        long_text = "我" * 30
        lines = split_to_lines(long_text)
        assert len(lines) >= 2
        for line in lines:
            assert len(line) <= 20

    def test_no_empty_lines(self):
        """不應產生空行"""
        lines = split_to_lines("，，，你好，，")
        assert all(l.strip() for l in lines)

    def test_punctuation_removed(self):
        """標點符號不在輸出中"""
        lines = split_to_lines("你好！世界？再見。")
        for line in lines:
            for p in "，。？！；：…—、.,!?;:":
                assert p not in line


# ══════════════════════════════════════════════════════════════════════
# assign_ts
# ══════════════════════════════════════════════════════════════════════

class TestAssignTs:
    """assign_ts: 時間軸分配"""

    def test_empty(self):
        assert assign_ts([], 0.0, 10.0) == []

    def test_single_line(self):
        result = assign_ts(["你好"], 0.0, 5.0)
        assert len(result) == 1
        start, end, text = result[0]
        assert text == "你好"
        assert start == pytest.approx(0.0)
        assert end >= 5.0  # 最後一行 end >= g1

    def test_two_lines_coverage(self):
        """兩行應覆蓋整個時間範圍"""
        result = assign_ts(["你好", "世界"], 10.0, 20.0)
        assert len(result) == 2
        # 第一行從 g0 開始
        assert result[0][0] == pytest.approx(10.0)
        # 最後一行結束 >= g1
        assert result[-1][1] >= 20.0

    def test_proportional(self):
        """字數多的行分到更長的時間"""
        result = assign_ts(["短", "這是一段比較長的文字"], 0.0, 10.0)
        dur_short = result[0][1] - result[0][0]
        dur_long  = result[1][1] - result[1][0]
        assert dur_long > dur_short

    def test_min_duration(self):
        """每段至少 MIN_SUB_SEC"""
        result = assign_ts(["a"], 0.0, 100.0)
        dur = result[0][1] - result[0][0]
        assert dur >= MIN_SUB_SEC

    def test_gap_between_lines(self):
        """相鄰行之間有 GAP_SEC 間隔"""
        result = assign_ts(["你好", "世界", "再見"], 0.0, 30.0)
        for i in range(len(result) - 1):
            gap = result[i + 1][0] - result[i][1]
            assert gap == pytest.approx(GAP_SEC)
