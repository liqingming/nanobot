"""Regression: TextualTUI.stop_thinking cancels the idle spinner on turn end.

Bug: the idle "思考中..." spinner scheduled after a tool call (in
_petrify_tool_placeholder → _schedule_idle_thinking) was only stopped by
pop_stream. The non-streaming reply path (add_response + _turn_complete) had no
such hook, so the spinner kept spinning AND counting forever after the turn
finished. _turn_complete now calls tui.stop_thinking() on every path.
"""
from __future__ import annotations

import pytest

from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE, TextualTUI

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE, reason="textual library is not installed"
)


@pytest.mark.asyncio
async def test_stop_thinking_cancels_idle_spinner():
    tui = TextualTUI()
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate the post-tool "still thinking" idle spinner being scheduled
        # (default 0.5s delay → the task stays pending).
        tui._schedule_idle_thinking()
        assert tui._idle_thinking_task is not None

        # Turn completes via the non-streaming path → _turn_complete → stop_thinking.
        tui.stop_thinking()
        await pilot.pause()

        # The idle spinner task must be cancelled so it doesn't spin forever.
        assert tui._idle_thinking_task is None


@pytest.mark.asyncio
async def test_stop_thinking_is_safe_when_no_spinner_active():
    tui = TextualTUI()
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        # No idle spinner scheduled — stop_thinking must be a harmless no-op.
        tui.stop_thinking()
        await pilot.pause()
        assert tui._idle_thinking_task is None
