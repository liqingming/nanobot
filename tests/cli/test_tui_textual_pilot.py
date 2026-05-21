"""Pilot-based integration tests for TextualTUI Enter routing.

Covers the /new topic-creation flow end-to-end: typing /new, pressing Enter,
typing the topic name, pressing Enter again. The topic name must reach the
new_topic callback and NOT be submitted as a chat message — that was the
original bug we fixed by unifying Enter routing through decide_enter_action.
"""
from __future__ import annotations

import pytest

from nanobot.cli.tui_textual import _TEXTUAL_AVAILABLE, TextualTUI

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE, reason="textual library is not installed"
)


async def _press_text(pilot, text: str) -> None:
    """Type each character via Pilot.press, mapping '/' to its key name."""
    keys: list[str] = []
    for ch in text:
        if ch == "/":
            keys.append("slash")
        elif ch == " ":
            keys.append("space")
        elif ch == "_":
            keys.append("underscore")
        else:
            keys.append(ch)
    await pilot.press(*keys)


@pytest.mark.asyncio
async def test_new_topic_full_flow_routes_to_topic_callback() -> None:
    """End-to-end: /new + Enter + name + Enter → topic callback, not on_submit."""
    tui = TextualTUI()
    submitted: list[str] = []
    topic_callbacks: list[str] = []

    async def topic_cb(name: str) -> None:
        topic_callbacks.append(name)

    async def on_submit(text: str) -> None:
        # The real commands.py would inspect text.startswith("/new") and call
        # tui.enter_new_topic_mode(topic_cb). We mirror that here.
        if text == "/new":
            tui.enter_new_topic_mode(topic_cb)
            return
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([("/new", "新建话题")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()

        # Type "/new" — the command popup will appear
        await _press_text(pilot, "/new")
        await pilot.pause()
        assert tui._popup_mode == "command"
        assert tui._popup_items

        # Enter: command popup is highlighted with /new, should submit it.
        # decide_enter_action returns COMMAND_SUBMIT; on_submit triggers
        # enter_new_topic_mode which clears popup and switches input_mode.
        await pilot.press("enter")
        await pilot.pause()
        assert tui._input_mode == "new_topic"
        assert tui._popup_mode == "hidden"
        assert submitted == []  # /new should NOT have been treated as chat

        # Type the topic name in new_topic mode
        await _press_text(pilot, "mytopic")
        await pilot.pause()
        # Popup must NOT reappear in new_topic mode
        assert tui._popup_mode == "hidden"

        # Enter: should route to topic_cb, NOT to on_submit
        await pilot.press("enter")
        await pilot.pause()
        assert topic_callbacks == ["mytopic"]
        # Critical: the topic name must not have been sent as a chat message
        assert "mytopic" not in submitted
        # Mode returns to chat after exit_new_topic_mode
        assert tui._input_mode == "chat"


@pytest.mark.asyncio
async def test_new_topic_via_explicit_api_routes_correctly() -> None:
    """When commands.py calls enter_new_topic_mode directly, Enter must route
    to the topic callback even if a stale popup state existed before."""
    tui = TextualTUI()
    submitted: list[str] = []
    topic_callbacks: list[str] = []

    async def topic_cb(name: str) -> None:
        topic_callbacks.append(name)

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([("/new", "新建话题")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()

        # Programmatically enter new_topic mode (simulates commands.py path
        # where /new is detected and enter_new_topic_mode is called).
        tui.enter_new_topic_mode(topic_cb)
        await pilot.pause()
        assert tui._input_mode == "new_topic"

        await _press_text(pilot, "test_topic")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert topic_callbacks == ["test_topic"]
        assert submitted == []
        assert tui._input_mode == "chat"


@pytest.mark.asyncio
async def test_normal_chat_message_routes_to_on_submit() -> None:
    """Sanity check: normal typing + Enter still reaches on_submit."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([("/new", "新建话题")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["hello"]


@pytest.mark.asyncio
async def test_command_popup_enter_submits_selected_command() -> None:
    """Typing /n and pressing Enter must submit the highlighted /new command."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([("/new", "新建话题"), ("/resume", "切换话题")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        # Type only /n — popup matches /new (and /resume? /resume doesn't start with n)
        await _press_text(pilot, "/n")
        await pilot.pause()
        assert tui._popup_mode == "command"
        assert tui._popup_items
        # The highlighted (first) item should be /new
        selected_value = tui._popup_items[tui._popup_idx][0]
        assert selected_value == "/new"

        await pilot.press("enter")
        await pilot.pause()
        # The selected /new (not the typed /n) should have been submitted
        assert submitted == ["/new"]


@pytest.mark.asyncio
async def test_command_submit_does_not_echo_command_to_output() -> None:
    """Commands (e.g. /new) must NOT trigger pre_submit / add_user_echo.

    Regression test: previously COMMAND_SUBMIT called _on_pre_submit, which
    in turn called add_user_echo, making /new appear in the output as if it
    were sent as a chat message.
    """
    tui = TextualTUI()
    submitted: list[str] = []
    pre_submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    def on_pre_submit(text: str) -> None:
        pre_submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_on_pre_submit(on_pre_submit)
    tui.set_commands([("/new", "新建话题")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "/new")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["/new"]
        # Critical: pre_submit (which would echo the command) must NOT fire
        assert pre_submitted == []


@pytest.mark.asyncio
async def test_command_submit_does_not_pollute_input_history() -> None:
    """Commands should not be remembered in the up-arrow history."""
    tui = TextualTUI()

    async def on_submit(text: str) -> None:
        pass

    tui.set_on_submit(on_submit)
    tui.set_commands([("/new", "新建话题")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        history_before = list(tui._history)
        await _press_text(pilot, "/new")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # /new should NOT have been added to history
        assert tui._history == history_before


@pytest.mark.asyncio
async def test_show_question_popup_collects_answer() -> None:
    """End-to-end: question popup, user selects option, callback receives answer."""
    tui = TextualTUI()
    tui.set_commands([])
    completed: list[dict | None] = []

    async def on_complete(answers: dict | None) -> None:
        completed.append(answers)

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        questions = [{
            "question": "Pick a color?",
            "options": [
                {"label": "red", "description": "warm"},
                {"label": "blue", "description": "cool"},
            ],
        }]
        tui.show_question_popup(questions, on_complete)
        await pilot.pause()
        # popup should be visible with two items, idx=0 (red)
        assert tui._popup_items
        assert tui._popup_items[0][0] == "red"
        # Enter to select the highlighted option
        await pilot.press("enter")
        await pilot.pause()
        assert completed == [{"Pick a color?": "red"}]


@pytest.mark.asyncio
async def test_show_question_popup_cancellation_via_escape() -> None:
    tui = TextualTUI()
    tui.set_commands([])
    completed: list[dict | None] = []

    async def on_complete(answers: dict | None) -> None:
        completed.append(answers)

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        questions = [{
            "question": "Pick?",
            "options": [
                {"label": "a", "description": ""},
                {"label": "b", "description": ""},
            ],
        }]
        tui.show_question_popup(questions, on_complete)
        await pilot.pause()
        # User presses ESC instead of selecting
        await pilot.press("escape")
        await pilot.pause()
        # on_complete must have been called with None (cancellation signal)
        assert completed == [None]


@pytest.mark.asyncio
async def test_show_question_popup_multiple_questions_sequential() -> None:
    """Two questions: answer first, popup auto-shows second, then on_complete."""
    tui = TextualTUI()
    tui.set_commands([])
    completed: list[dict | None] = []

    async def on_complete(answers: dict | None) -> None:
        completed.append(answers)

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        questions = [
            {"question": "Q1?", "options": [{"label": "a1", "description": ""}, {"label": "b1", "description": ""}]},
            {"question": "Q2?", "options": [{"label": "a2", "description": ""}, {"label": "b2", "description": ""}]},
        ]
        tui.show_question_popup(questions, on_complete)
        await pilot.pause()
        # First popup — pick first option ("a1")
        await pilot.press("enter")
        await pilot.pause()
        # Second popup should appear automatically
        assert tui._popup_items
        assert tui._popup_items[0][0] == "a2"
        await pilot.press("enter")
        await pilot.pause()
        # on_complete called with both answers
        assert completed == [{"Q1?": "a1", "Q2?": "a2"}]


@pytest.mark.asyncio
async def test_idle_thinking_spinner_starts_after_stream_delta() -> None:
    """After a stream_delta, if no further delta arrives within ~500ms,
    the idle thinking spinner should start in #live without crashing on
    shutdown (regression test for the active_app LookupError).
    """
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.stream_start()
        await pilot.pause()
        tui.stream_delta("hello")
        # Wait long enough for _schedule_idle_thinking's 500ms timer to fire
        # and start_spinner's set_interval to create a Timer.
        await pilot.pause(0.7)
        # No assertion on visual state — the regression is "shutdown crashes
        # because active_app is missing on the Timer's context". If app exits
        # cleanly past this block, the bug is fixed.


@pytest.mark.asyncio
async def test_idle_thinking_after_tool_completes_does_not_crash() -> None:
    """add_tool_result also schedules idle thinking — same regression path
    must not surface LookupError on shutdown.
    """
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.stream_start()
        await pilot.pause()
        tui.tool_phase_start()
        await pilot.pause(0.1)
        tui.add_progress("read_file(\"x\")")
        await pilot.pause(0.1)
        tui.add_tool_result("3 lines, 50 chars")
        # The add_tool_result path schedules another idle_thinking — let it fire
        await pilot.pause(0.7)


@pytest.mark.asyncio
async def test_normal_submit_does_not_pollute_history_for_slash_text() -> None:
    """Typing /xyz that doesn't match a command shouldn't go to history either."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])  # no commands → no popup match for /xyz

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        history_before = list(tui._history)
        await _press_text(pilot, "/xyz")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # SUBMIT was the action (no popup match), but /xyz starts with / → skip history
        assert tui._history == history_before
        assert submitted == ["/xyz"]
