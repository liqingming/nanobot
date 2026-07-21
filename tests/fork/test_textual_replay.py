"""Regression: TextualTUI.load_session_history must replay tool traces.

This broke when _replay_tool_trace imported ``format_tool_hint`` from
``nanobot.agent.loop`` (which only imports it function-locally as ``_fork_fmt``),
raising ImportError that the surrounding ``except`` silently swallowed — so
every restored tool trace vanished on restart while live runs looked fine.
"""
from __future__ import annotations

import pytest

from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE, TextualTUI

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE, reason="textual library is not installed"
)


@pytest.mark.asyncio
async def test_load_session_history_replays_tool_trace():
    tui = TextualTUI()
    messages = [
        {"role": "user", "content": "问我几个问题"},
        {
            "role": "assistant",
            "content": "好的，我先确认几个选择：",
            "tool_calls": [
                {"id": "c1", "function": {"name": "ask_user", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "用户已回复"},
    ]

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.load_session_history(messages)
        await pilot.pause()
        out = app.query_one("#output")
        rendered = "\n".join(strip.text for strip in out.lines)

    # The assistant text is restored under one turn header …
    assert "确认几个选择" in rendered
    assistant_headers = [
        line for line in rendered.splitlines() if line.startswith("🐈 nanobot 20")
    ]
    assert len(assistant_headers) == 1
    assert rendered.index(assistant_headers[0]) < rendered.index("确认几个选择")
    assert rendered.index("确认几个选择") < rendered.index("ask_user")
    # … and crucially the tool trace is replayed (before the fix the swallowed
    # ImportError left no trace at all).
    assert "ask_user" in rendered


@pytest.mark.asyncio
async def test_load_session_history_replays_readable_multitool_batch() -> None:
    tui = TextualTUI(workspace=".")
    here_python = "@'\nimport inspect\nprint(inspect.signature(len))\n'@ | python -X utf8 -"
    messages = [
        {"role": "user", "content": "检查实现"},
        {
            "role": "assistant",
            "content": "我先检查。",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "exec",
                        "arguments": __import__("json").dumps({"command": here_python}),
                    },
                },
                {
                    "id": "c2",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"nanobot/fork/cli/tui_textual.py"}',
                    },
                },
                {
                    "id": "c3",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"nanobot/fork/cli/tui_textual.py"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "tool", "tool_call_id": "c3", "content": "ok"},
    ]

    async with tui._app.run_test(size=(48, 20)) as pilot:
        tui.load_session_history(messages, workspace=".")
        await pilot.pause()
        out = tui._app.query_one("#output")
        rendered = "\n".join(strip.text for strip in out.lines)

    assert "工具 · 3 项" in rendered
    assert "运行 Python 检查脚本" in rendered
    assert 'read_file("nanobot/fork/cli/tui_textual.py") × 2' in rendered
    assert "@'" not in rendered
    assert "├─" in rendered
    assert "└─" in rendered


@pytest.mark.asyncio
async def test_history_tool_only_assistant_writes_header_before_tools_and_final_reply() -> None:
    tui = TextualTUI(render_markdown=False)
    messages = [
        {
            "role": "user",
            "content": "不需要了，提交",
            "timestamp": "2026-07-21T14:08:42",
        },
        {
            "role": "assistant",
            "content": "",
            "timestamp": "2026-07-21T14:08:43",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"git status --short"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "clean"},
        {
            "role": "assistant",
            "content": "已提交。",
            "timestamp": "2026-07-21T14:09:37",
        },
    ]

    async with tui._app.run_test() as pilot:
        tui.load_session_history(messages)
        await pilot.pause()
        rendered = "\n".join(strip.text for strip in tui._app.query_one("#output").lines)

    assert rendered.count("nanobot 2026-") == 1
    header_index = rendered.index("nanobot 2026-")
    assert header_index < rendered.index("exec") < rendered.index("已提交。")
    assert "2026-07-21 14:08:43" in rendered


@pytest.mark.asyncio
async def test_history_resets_assistant_header_once_per_user_turn() -> None:
    tui = TextualTUI(render_markdown=False)
    messages = [
        {"role": "user", "content": "第一问", "timestamp": "2026-07-21T10:00:00"},
        {"role": "assistant", "content": "第一答", "timestamp": "2026-07-21T10:00:01"},
        {"role": "user", "content": "第二问", "timestamp": "2026-07-21T10:01:00"},
        {
            "role": "assistant",
            "content": "",
            "timestamp": "2026-07-21T10:01:01",
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "第二答", "timestamp": "2026-07-21T10:01:02"},
    ]

    async with tui._app.run_test() as pilot:
        tui.load_session_history(messages)
        await pilot.pause()
        rendered = "\n".join(strip.text for strip in tui._app.query_one("#output").lines)

    assert rendered.count("nanobot 2026-") == 2
    assert rendered.index("第二问") < rendered.index("read_file") < rendered.index("第二答")
