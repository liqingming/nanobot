"""Readable fork tool hints and structured batch item formatting."""
from __future__ import annotations

from nanobot.fork.utils.tool_hints import format_tool_event_items, format_tool_hint
from nanobot.providers.base import ToolCallRequest


def _tc(name: str, arguments: dict) -> ToolCallRequest:
    return ToolCallRequest(id=name, name=name, arguments=arguments)


def test_powershell_here_string_python_is_semantic_single_line() -> None:
    command = "@'\nimport inspect, textual\nprint(inspect.getsource(textual))\n'@ | python -X utf8 -"

    hint = format_tool_hint([_tc("exec", {"command": command})])

    assert hint == "运行 Python 检查脚本"
    assert "@'" not in hint
    assert "\n" not in hint


def test_short_exec_command_remains_recognizable() -> None:
    assert format_tool_hint([_tc("exec", {"command": "git status --short"})]) == (
        'exec("git status --short")'
    )


def test_items_aggregate_same_targets_even_when_not_adjacent() -> None:
    calls = [
        _tc("read_file", {"path": "a.py"}),
        _tc("grep", {"pattern": "TODO"}),
        _tc("read_file", {"path": "a.py"}),
    ]

    items = format_tool_hint(calls).split(", ")

    assert items == ['read_file("a.py") × 2', 'grep("TODO")']


def test_structured_event_items_keep_error_separate_from_success() -> None:
    events = [
        {"phase": "end", "name": "read_file", "arguments": {"path": "a.py"}},
        {
            "phase": "error",
            "name": "read_file",
            "arguments": {"path": "a.py"},
            "error": "not found",
        },
    ]

    items = format_tool_event_items(events)

    assert [(item["label"], item["status"]) for item in items] == [
        ('read_file("a.py")', "ok"),
        ('read_file("a.py")', "error"),
    ]
