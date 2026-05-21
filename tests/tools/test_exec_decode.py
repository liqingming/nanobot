"""Tests for ExecTool's robust console-output decoding."""
from __future__ import annotations

import sys

import pytest

from nanobot.agent.tools.shell import _decode_console_bytes


def test_decodes_utf8_bytes() -> None:
    assert _decode_console_bytes("hello 世界".encode("utf-8")) == "hello 世界"


def test_decodes_ascii() -> None:
    assert _decode_console_bytes(b"plain ascii") == "plain ascii"


def test_empty_bytes_returns_empty_string() -> None:
    assert _decode_console_bytes(b"") == ""


def test_falls_back_to_gbk_for_gbk_encoded_bytes() -> None:
    """The reported regression — chinese Windows console emits cp936 bytes."""
    original = "[INFO] 正在从 Yahoo Finance 获取数据"
    gbk_bytes = original.encode("cp936")
    decoded = _decode_console_bytes(gbk_bytes)
    assert decoded == original
    # Must not contain mojibake replacement chars
    assert "�" not in decoded


def test_falls_back_to_latin1_for_high_bytes() -> None:
    """Bytes that aren't valid UTF-8 and aren't well-formed CJK should still
    decode to *something* without crashing — last resort uses errors='replace'."""
    # 0xFF is not valid UTF-8 start byte, not a meaningful CJK pair either.
    out = _decode_console_bytes(b"\xff\xfe")
    assert isinstance(out, str)


def test_uses_pythonioencoding_hint_when_present(monkeypatch) -> None:
    """If PYTHONIOENCODING is set, give it a chance before locale guess."""
    if sys.platform != "win32":
        pytest.skip("env-var hint path is Windows-only")
    monkeypatch.setenv("PYTHONIOENCODING", "cp936:replace")
    original = "中文测试"
    out = _decode_console_bytes(original.encode("cp936"))
    assert out == original


def test_pure_utf8_chinese_stays_utf8() -> None:
    """UTF-8 path should win for Linux/modern Windows output."""
    original = "纯 UTF-8 输出"
    out = _decode_console_bytes(original.encode("utf-8"))
    assert out == original
