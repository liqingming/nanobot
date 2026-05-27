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

    # The assistant text is restored …
    assert "确认几个选择" in rendered
    # … and crucially the tool trace is replayed (before the fix the swallowed
    # ImportError left no trace at all).
    assert "ask_user" in rendered
