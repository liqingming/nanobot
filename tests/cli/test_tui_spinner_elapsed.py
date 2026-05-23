"""Tests for the spinner elapsed-time suffix helper."""
from __future__ import annotations

import time

import pytest

from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE, reason="textual not installed"
)


def _suffix_at(elapsed_seconds: float) -> str:
    """Construct a fake start_time so the helper sees a known elapsed value."""
    from nanobot.fork.cli.tui_textual import _NanobotApp
    fake_start = time.monotonic() - elapsed_seconds
    return _NanobotApp._elapsed_suffix(fake_start)


# ── Below 1s threshold: suppressed ─────────────────────────────────────────


def test_suffix_empty_below_one_second() -> None:
    """Suppress flicker for very quick operations."""
    assert _suffix_at(0.0) == ""
    assert _suffix_at(0.5) == ""
    assert _suffix_at(0.99) == ""


# ── Sub-minute: integer seconds ────────────────────────────────────────────


def test_suffix_integer_seconds() -> None:
    assert _suffix_at(1.0) == " (1s)"
    assert _suffix_at(5.7) == " (5s)"     # truncated, not rounded
    assert _suffix_at(9.9) == " (9s)"
    assert _suffix_at(10.0) == " (10s)"
    assert _suffix_at(59.4) == " (59s)"


# ── 60s+: minutes + seconds ────────────────────────────────────────────────


def test_suffix_at_60_seconds_uses_minutes() -> None:
    assert _suffix_at(60.0) == " (1m0s)"


def test_suffix_minutes_format() -> None:
    assert _suffix_at(75.0) == " (1m15s)"
    assert _suffix_at(125.5) == " (2m5s)"
    assert _suffix_at(605.0) == " (10m5s)"


def test_suffix_transition_at_minute_boundary() -> None:
    """Below 60s shows seconds; at/above 60s switches to NmNs format."""
    sub = _suffix_at(58.0)
    over = _suffix_at(60.5)
    assert sub == " (58s)"
    assert "m" in over and "1m" in over


# ── Linearity property: real time advance matches display advance ──────────


def test_suffix_progresses_linearly_with_real_time() -> None:
    """Critical sanity: the displayed seconds advance at real-time rate."""
    from nanobot.fork.cli.tui_textual import _NanobotApp
    import re

    # Pretend we started 3 seconds ago, then sleep 1.1 real seconds and
    # check that the displayed seconds advance by ~1.
    start = time.monotonic() - 3.0
    s1 = _NanobotApp._elapsed_suffix(start)
    time.sleep(1.1)
    s2 = _NanobotApp._elapsed_suffix(start)
    f1 = int(re.search(r"\((\d+)s\)", s1).group(1))
    f2 = int(re.search(r"\((\d+)s\)", s2).group(1))
    delta = f2 - f1
    assert delta == 1, (
        f"Displayed delta {delta}s should be 1 after sleeping 1.1s real time"
    )
