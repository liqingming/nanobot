"""Shared key-routing decision functions for nanobot TUI backends.

Both TextualTUI (``tui_textual.py``) and PromptTUI (``tui.py``) need
identical decision logic for:

  * Enter  — submit text, select popup item, or trigger new-topic callback
  * Up / Down — cycle popup selection vs walk input history
  * Tab    — autocomplete from command popup

Each backend constructs a :class:`TUIState` snapshot and calls one of the
``decide_*`` functions; the returned dataclass tells it what to do. Widget
mutation (writing buffer text, hiding popup, etc.) stays in the backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── Shared state snapshot ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TUIState:
    """Snapshot of the TUI input/popup state.

    Both backends build this immediately before delegating to a decide_*
    function, so a single object beats threading 5+ kwargs through.
    """
    input_mode: str               # "chat" | "new_topic"
    popup_mode: str               # "hidden" | "command" | "topic"
    popup_has_items: bool
    popup_selected_value: str | None
    input_text: str


# ── Enter decision tree ───────────────────────────────────────────────────


class EnterAction(Enum):
    NOOP = "noop"
    NEW_TOPIC = "new_topic"
    TOPIC_SELECT = "topic_select"
    COMMAND_SUBMIT = "command_submit"
    SUBMIT = "submit"


@dataclass(frozen=True)
class EnterDecision:
    """Outcome of the Enter-key decision tree.

    Attributes:
        action: which branch was selected.
        value: the text/value the backend should act on (already stripped
            where appropriate). For NOOP, the value is the empty string.
    """
    action: EnterAction
    value: str = ""


def decide_enter_action(state: TUIState) -> EnterDecision:
    """Return the action the TUI should take when Enter is pressed.

    Priority:
      1. new_topic mode wins over any popup
      2. topic popup with items → select highlighted
      3. command popup with items → submit highlighted command
      4. non-empty input → normal submit
      5. otherwise → no-op
    """
    if state.input_mode == "new_topic":
        return EnterDecision(EnterAction.NEW_TOPIC, state.input_text.strip())

    if state.popup_has_items and state.popup_selected_value is not None:
        if state.popup_mode == "topic":
            return EnterDecision(EnterAction.TOPIC_SELECT, state.popup_selected_value)
        if state.popup_mode == "command":
            return EnterDecision(EnterAction.COMMAND_SUBMIT, state.popup_selected_value)

    if state.input_text.strip():
        return EnterDecision(EnterAction.SUBMIT, state.input_text)

    return EnterDecision(EnterAction.NOOP, "")


# ── Popup navigation keys (up/down/tab) ───────────────────────────────────


class PopupAction(Enum):
    IGNORE = "ignore"            # backend should do nothing extra (default key behavior may run)
    CYCLE_UP = "cycle_up"        # decrement popup selection
    CYCLE_DOWN = "cycle_down"    # increment popup selection
    COMPLETE = "complete"        # autocomplete the popup-selected value into input
    HISTORY_BACK = "history_back"
    HISTORY_FORWARD = "history_forward"


@dataclass(frozen=True)
class PopupDecision:
    action: PopupAction
    value: str = ""              # populated for COMPLETE (the text to insert)


def decide_popup_key(key: str, state: TUIState) -> PopupDecision:
    """Decide what up/down/tab should do given the current TUI state.

    * up    — popup visible: cycle up; else history_back
    * down  — popup visible: cycle down; else history_forward
    * tab   — only meaningful in command popup with a selected item → complete
    * any other key → IGNORE
    """
    if key == "up":
        if state.popup_has_items:
            return PopupDecision(PopupAction.CYCLE_UP)
        return PopupDecision(PopupAction.HISTORY_BACK)
    if key == "down":
        if state.popup_has_items:
            return PopupDecision(PopupAction.CYCLE_DOWN)
        return PopupDecision(PopupAction.HISTORY_FORWARD)
    if key == "tab":
        if (
            state.popup_mode == "command"
            and state.popup_has_items
            and state.popup_selected_value is not None
        ):
            return PopupDecision(PopupAction.COMPLETE, state.popup_selected_value)
        return PopupDecision(PopupAction.IGNORE)
    return PopupDecision(PopupAction.IGNORE)
