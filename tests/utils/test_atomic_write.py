"""Tests for atomic write helpers."""

from pathlib import Path
from unittest.mock import patch

from nanobot.utils.atomic_write import replace_file_with_retry


def test_replace_file_with_retry_retries_transient_permission_error(tmp_path: Path) -> None:
    src = tmp_path / "state.tmp"
    dst = tmp_path / "state.json"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")

    import os

    original_replace = os.replace
    attempts = 0

    def flaky_replace(src_path, dst_path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        original_replace(src_path, dst_path)

    with patch("nanobot.utils.atomic_write.os.replace", side_effect=flaky_replace), \
         patch("nanobot.utils.atomic_write.time.sleep") as sleep:
        replace_file_with_retry(src, dst)

    assert attempts == 3
    assert sleep.call_count == 2
    assert dst.read_text(encoding="utf-8") == "new"


def test_replace_file_with_retry_reraises_after_retries(tmp_path: Path) -> None:
    src = tmp_path / "state.tmp"
    dst = tmp_path / "state.json"
    src.write_text("new", encoding="utf-8")

    with patch("nanobot.utils.atomic_write.os.replace", side_effect=PermissionError("locked")), \
         patch("nanobot.utils.atomic_write.time.sleep") as sleep:
        try:
            replace_file_with_retry(src, dst, retries=2)
        except PermissionError:
            pass
        else:  # pragma: no cover
            raise AssertionError("PermissionError was not raised")

    assert sleep.call_count == 2
    assert src.exists()
