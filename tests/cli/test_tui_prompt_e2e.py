"""Semi-e2e tests for PromptTUI key handlers.

We can't easily spin up a real prompt_toolkit Application in tests (it needs a
terminal). Instead, the ``prompt_tui`` fixture in conftest.py initializes just
the input buffer and we exercise the extracted key-handler methods directly.
This covers the same ground as the Textual Pilot tests: widget state changes
+ callback routing follow the unified decide_* decisions.
"""
from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.document import Document

from nanobot.fork.cli.tui import PromptTUI


async def _drain() -> None:
    """Yield so any asyncio.ensure_future scheduled work can run."""
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_normal_chat_message_routes_to_on_submit(prompt_tui: PromptTUI) -> None:
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    prompt_tui.set_on_submit(on_submit)
    prompt_tui._input_buffer.set_document(Document("hello", 5))
    prompt_tui._handle_enter_key()
    await _drain()

    assert submitted == ["hello"]
    assert prompt_tui._input_buffer.text == ""


@pytest.mark.asyncio
async def test_command_popup_enter_submits_selected_command(prompt_tui: PromptTUI) -> None:
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    prompt_tui.set_on_submit(on_submit)
    # Typing "/n" filters popup down to /new
    prompt_tui._input_buffer.set_document(Document("/n", 2))
    assert prompt_tui._popup_mode == "command"
    assert prompt_tui._popup_items
    assert prompt_tui._popup_items[prompt_tui._popup_idx][0] == "/new"

    prompt_tui._handle_enter_key()
    await _drain()

    # The selected /new (not the typed /n) is what should be submitted
    assert submitted == ["/new"]
    # Buffer is cleared after submit; popup may transiently re-render via
    # on_text_changed but that's a render concern, not a correctness one.
    assert prompt_tui._input_buffer.text == ""


@pytest.mark.asyncio
async def test_skills_command_enter_does_not_call_pre_submit(prompt_tui: PromptTUI) -> None:
    submitted: list[str] = []
    pre_submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    def on_pre_submit(text: str) -> None:
        pre_submitted.append(text)

    prompt_tui.set_on_submit(on_submit)
    prompt_tui.set_on_pre_submit(on_pre_submit)
    prompt_tui.set_commands([("/skills", "List available skills")])
    prompt_tui._input_buffer.set_document(Document("/skills", 7))

    prompt_tui._handle_enter_key()
    await _drain()

    assert submitted == ["/skills"]
    assert pre_submitted == []


@pytest.mark.asyncio
async def test_new_topic_flow_routes_to_topic_callback(prompt_tui: PromptTUI) -> None:
    """End-to-end: /new triggers enter_new_topic_mode, typing a name + Enter
    must route to the topic callback, NOT submit as a chat message."""
    submitted: list[str] = []
    topic_callbacks: list[str] = []

    async def topic_cb(name: str) -> None:
        topic_callbacks.append(name)

    async def on_submit(text: str) -> None:
        if text == "/new":
            prompt_tui.enter_new_topic_mode(topic_cb)
            return
        submitted.append(text)

    prompt_tui.set_on_submit(on_submit)
    # Step 1: type /n, popup shows /new; press Enter → /new submitted → enter_new_topic_mode
    prompt_tui._input_buffer.set_document(Document("/n", 2))
    prompt_tui._handle_enter_key()
    await _drain()
    assert prompt_tui._input_mode == "new_topic"
    assert submitted == []  # /new was a command, not a chat message

    # Step 2: type topic name, press Enter → topic_cb, not on_submit
    prompt_tui._input_buffer.set_document(Document("mytopic", 7))
    prompt_tui._handle_enter_key()
    await _drain()

    assert topic_callbacks == ["mytopic"]
    assert "mytopic" not in submitted
    assert prompt_tui._input_mode == "chat"  # exits new_topic mode after submission


@pytest.mark.asyncio
async def test_new_topic_empty_name_still_calls_callback_with_empty(prompt_tui: PromptTUI) -> None:
    """Empty topic name routes to callback with empty string; caller decides default."""
    received: list[str] = []

    async def cb(name: str) -> None:
        received.append(name)

    prompt_tui.enter_new_topic_mode(cb)
    prompt_tui._input_buffer.set_document(Document("", 0))
    prompt_tui._handle_enter_key()
    await _drain()

    assert received == [""]


def test_tab_completes_command_from_popup(prompt_tui: PromptTUI) -> None:
    prompt_tui._input_buffer.set_document(Document("/n", 2))
    assert prompt_tui._popup_mode == "command"
    prompt_tui._handle_tab_key()
    assert prompt_tui._input_buffer.text == "/new"


def test_tab_does_nothing_when_no_popup(prompt_tui: PromptTUI) -> None:
    prompt_tui._input_buffer.set_document(Document("hello", 5))
    prompt_tui._handle_tab_key()
    assert prompt_tui._input_buffer.text == "hello"  # unchanged


def test_up_cycles_popup_when_visible(prompt_tui: PromptTUI) -> None:
    prompt_tui._input_buffer.set_document(Document("/", 1))
    # popup_idx starts at 0, both /new and /resume present
    assert prompt_tui._popup_items
    prompt_tui._popup_idx = 1
    prompt_tui._handle_popup_key("up")
    assert prompt_tui._popup_idx == 0


def test_down_cycles_popup_when_visible(prompt_tui: PromptTUI) -> None:
    prompt_tui._input_buffer.set_document(Document("/", 1))
    assert len(prompt_tui._popup_items) >= 2
    prompt_tui._popup_idx = 0
    prompt_tui._handle_popup_key("down")
    assert prompt_tui._popup_idx == 1


def test_up_walks_history_when_popup_hidden(prompt_tui: PromptTUI) -> None:
    """When popup is hidden, up should walk input history (no crash)."""
    # Empty input → no popup. history_backward on empty history is a no-op.
    prompt_tui._input_buffer.set_document(Document("", 0))
    assert prompt_tui._popup_mode == "hidden"
    prompt_tui._handle_popup_key("up")  # should not raise
    assert prompt_tui._input_buffer.text == ""


# ── extracted key handlers (escape / ctrl-d / pageup / pagedown) ──────────


def test_escape_in_new_topic_mode_exits_mode(prompt_tui: PromptTUI) -> None:
    async def cb(name: str) -> None:
        pass
    prompt_tui.enter_new_topic_mode(cb)
    assert prompt_tui._input_mode == "new_topic"
    prompt_tui._handle_escape_key()
    assert prompt_tui._input_mode == "chat"
    assert prompt_tui._new_topic_cb is None


def test_escape_with_popup_hides_popup(prompt_tui: PromptTUI) -> None:
    prompt_tui._input_buffer.set_document(Document("/n", 2))
    assert prompt_tui._popup_mode == "command"
    prompt_tui._handle_escape_key()
    assert prompt_tui._popup_mode == "hidden"


def test_escape_otherwise_calls_on_cancel(prompt_tui: PromptTUI) -> None:
    cancelled: list[bool] = []
    prompt_tui.set_on_cancel(lambda: cancelled.append(True))
    prompt_tui._handle_escape_key()
    assert cancelled == [True]


def test_ctrl_d_returns_true_when_buffer_empty(prompt_tui: PromptTUI) -> None:
    assert prompt_tui._handle_ctrl_d_key() is True


def test_ctrl_d_returns_false_when_buffer_has_text(prompt_tui: PromptTUI) -> None:
    prompt_tui._input_buffer.set_document(Document("hello", 5))
    assert prompt_tui._handle_ctrl_d_key() is False


def test_pageup_increments_scroll_offset(prompt_tui: PromptTUI) -> None:
    prompt_tui._scroll_offset = 0
    prompt_tui._handle_pageup_key()
    assert prompt_tui._scroll_offset == 10


def test_pagedown_decrements_scroll_offset(prompt_tui: PromptTUI) -> None:
    prompt_tui._scroll_offset = 25
    prompt_tui._handle_pagedown_key()
    assert prompt_tui._scroll_offset == 15


def test_pagedown_clamps_at_zero(prompt_tui: PromptTUI) -> None:
    prompt_tui._scroll_offset = 5
    prompt_tui._handle_pagedown_key()
    assert prompt_tui._scroll_offset == 0
