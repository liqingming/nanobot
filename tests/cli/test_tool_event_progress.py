from __future__ import annotations

from nanobot.cli.commands import _tool_result_summary_from_events


def test_tool_event_end_petrifies_textual_spinner_without_visible_summary() -> None:
    events = [{
        "phase": "end",
        "name": "read_file",
        "result": "1|{}",
        "error": None,
    }]

    assert _tool_result_summary_from_events(events) == ""


def test_tool_event_error_returns_error_summary() -> None:
    events = [{
        "phase": "error",
        "name": "read_file",
        "result": None,
        "error": "permission denied",
    }]

    assert _tool_result_summary_from_events(events) == "Error: permission denied"


def test_tool_event_start_does_not_petrify_textual_spinner() -> None:
    events = [{
        "phase": "start",
        "name": "read_file",
        "result": None,
        "error": None,
    }]

    assert _tool_result_summary_from_events(events) is None
