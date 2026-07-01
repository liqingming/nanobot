"""Tests for Textual output text selection."""

from __future__ import annotations

import pytest
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE, _OutputLog

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE, reason="textual library is not installed"
)


def _line(text: str) -> Strip:
    return Strip([Segment(text)], len(text))


def test_output_log_extracts_column_selection_across_lines() -> None:
    output = _OutputLog()
    output.lines = [_line("abcdef"), _line("ghijkl")]
    output._sel_start = (0, 2)
    output._sel_end = (1, 3)

    assert output._extract_selected_text() == "cdef\nghij"


def test_output_log_extracts_reversed_column_selection() -> None:
    output = _OutputLog()
    output.lines = [_line("abcdef")]
    output._sel_start = (0, 4)
    output._sel_end = (0, 1)

    assert output._extract_selected_text() == "bcde"


def test_output_log_preserves_selected_trailing_spaces() -> None:
    output = _OutputLog()
    output.lines = [_line("abc  ")]
    output._sel_start = (0, 2)
    output._sel_end = (0, 4)

    assert output._extract_selected_text() == "c  "


def test_output_log_highlights_only_selected_columns() -> None:
    strip = Strip([Segment("abcdef", Style(color="red"))], 6)

    highlighted = _OutputLog._force_color_range(
        strip,
        start=2,
        end=4,
        bgcolor="white",
        color="black",
    )

    assert highlighted.text == "abcdef"
    selected = highlighted.crop(2, 4)
    assert selected.text == "cd"
    expected = Style(color="black", bgcolor="white")
    assert all(
        segment.style
        and segment.style.color == expected.color
        and segment.style.bgcolor == expected.bgcolor
        for segment in selected._segments
    )
