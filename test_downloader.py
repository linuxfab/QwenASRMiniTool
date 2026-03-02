"""
test_downloader.py — downloader.py 單元測試

覆蓋：
  - _file_is_real（LFS pointer 偵測）
  - _get_paths
  - quick_check / quick_check_1p7b / quick_check_diarization
  - _sha256
"""
import hashlib
import textwrap
from pathlib import Path

import pytest

from downloader import (
    DIAR_FILES,
    REQUIRED_BIN,
    REQUIRED_OTHER,
    _1P7B_REQUIRED_BIN,
    _1P7B_REQUIRED_OTHER,
    _file_is_real,
    _get_paths,
    _sha256,
    quick_check,
    quick_check_1p7b,
    quick_check_diarization,
)


# ══════════════════════════════════════════════════════════════════════
# _file_is_real
# ══════════════════════════════════════════════════════════════════════

class TestFileIsReal:
    def test_nonexistent_file(self, tmp_path: Path):
        assert _file_is_real(tmp_path / "no_such_file.bin") is False

    def test_real_binary_file(self, tmp_path: Path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"\x00\x01\x02" * 100)
        assert _file_is_real(f) is True

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert _file_is_real(f) is True

    def test_lfs_pointer_rejected(self, tmp_path: Path):
        """Git LFS pointer 檔應被拒絕。"""
        lfs_content = textwrap.dedent("""\
            version https://git-lfs.github.com/spec/v1
            oid sha256:abc123
            size 12345
        """).encode("utf-8")
        f = tmp_path / "model.bin"
        f.write_bytes(lfs_content)
        assert _file_is_real(f) is False

    def test_lfs_exact_magic_only(self, tmp_path: Path):
        """只有前 43 bytes 完全匹配 LFS magic 才被拒絕。"""
        f = tmp_path / "looks_similar.bin"
        f.write_bytes(b"version https://git-lfs.github.com/spec/v2_EXTRA")
        assert _file_is_real(f) is True

    def test_short_file_not_rejected(self, tmp_path: Path):
        """比 LFS magic 短的檔案不應被拒絕。"""
        f = tmp_path / "tiny.bin"
        f.write_bytes(b"version https://")
        assert _file_is_real(f) is True

    def test_text_file_not_rejected(self, tmp_path: Path):
        """普通 JSON 設定檔不應被拒絕。"""
        f = tmp_path / "config.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        assert _file_is_real(f) is True


# ══════════════════════════════════════════════════════════════════════
# _get_paths
# ══════════════════════════════════════════════════════════════════════

class TestGetPaths:
    def test_returns_correct_subdirs(self, tmp_path: Path):
        ov_dir, vad_path = _get_paths(tmp_path)
        assert ov_dir == tmp_path / "qwen3_asr_int8"
        assert vad_path == tmp_path / "silero_vad_v4.onnx"

    def test_paths_are_path_objects(self, tmp_path: Path):
        ov_dir, vad_path = _get_paths(tmp_path)
        assert isinstance(ov_dir, Path)
        assert isinstance(vad_path, Path)


# ══════════════════════════════════════════════════════════════════════
# quick_check (0.6B)
# ══════════════════════════════════════════════════════════════════════

def _populate_model_dir(model_dir: Path):
    """在 tmp_path 建立完整的 0.6B 模型目錄結構（假檔案）。"""
    ov_dir, vad_path = _get_paths(model_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)
    vad_path.write_bytes(b"\x00" * 100)
    for fname in list(REQUIRED_BIN.keys()) + REQUIRED_OTHER:
        (ov_dir / fname).write_bytes(b"\x00" * 100)


class TestQuickCheck:
    def test_empty_dir_fails(self, tmp_path: Path):
        assert quick_check(tmp_path) is False

    def test_complete_dir_passes(self, tmp_path: Path):
        _populate_model_dir(tmp_path)
        assert quick_check(tmp_path) is True

    def test_missing_vad_fails(self, tmp_path: Path):
        _populate_model_dir(tmp_path)
        (tmp_path / "silero_vad_v4.onnx").unlink()
        assert quick_check(tmp_path) is False

    def test_missing_bin_fails(self, tmp_path: Path):
        _populate_model_dir(tmp_path)
        first_bin = list(REQUIRED_BIN.keys())[0]
        (tmp_path / "qwen3_asr_int8" / first_bin).unlink()
        assert quick_check(tmp_path) is False

    def test_lfs_pointer_bin_fails(self, tmp_path: Path):
        """即使檔案存在但為 LFS pointer 也應失敗。"""
        _populate_model_dir(tmp_path)
        first_bin = list(REQUIRED_BIN.keys())[0]
        lfs = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 99\n"
        (tmp_path / "qwen3_asr_int8" / first_bin).write_bytes(lfs)
        assert quick_check(tmp_path) is False


# ══════════════════════════════════════════════════════════════════════
# quick_check_1p7b
# ══════════════════════════════════════════════════════════════════════

class TestQuickCheck1p7b:
    def test_empty_dir_fails(self, tmp_path: Path):
        assert quick_check_1p7b(tmp_path) is False

    def test_complete_1p7b_passes(self, tmp_path: Path):
        kv_dir = tmp_path / "qwen3_asr_1p7b_kv_int8"
        kv_dir.mkdir(parents=True)
        for fname in _1P7B_REQUIRED_BIN + _1P7B_REQUIRED_OTHER:
            (kv_dir / fname).write_bytes(b"\x00" * 100)
        assert quick_check_1p7b(tmp_path) is True


# ══════════════════════════════════════════════════════════════════════
# quick_check_diarization
# ══════════════════════════════════════════════════════════════════════

class TestQuickCheckDiarization:
    def test_empty_dir_fails(self, tmp_path: Path):
        assert quick_check_diarization(tmp_path) is False

    def test_complete_diar_passes(self, tmp_path: Path):
        diar_dir = tmp_path / "diarization"
        diar_dir.mkdir(parents=True)
        for fname in DIAR_FILES:
            (diar_dir / fname).write_bytes(b"\x00" * 100)
        assert quick_check_diarization(tmp_path) is True


# ══════════════════════════════════════════════════════════════════════
# _sha256
# ══════════════════════════════════════════════════════════════════════

class TestSha256:
    def test_known_hash(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        data = b"hello world"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(f) == expected

    def test_empty_file_hash(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256(f) == expected

    def test_progress_callback(self, tmp_path: Path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * 2048)
        calls = []
        _sha256(f, progress_cb=lambda d, t: calls.append((d, t)))
        assert len(calls) > 0
        assert calls[-1][0] == calls[-1][1]  # 最後一次 done == total
