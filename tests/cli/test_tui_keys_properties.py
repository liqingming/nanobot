"""Property-based tests for tui_keys decision functions.

These complement the example-based tests in test_tui_keys.py by exhaustively
exercising decide_enter_action and decide_popup_key across the cross-product
of state values, asserting invariants that must hold regardless of inputs.
"""
from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from nanobot.fork.cli.tui_keys import (
    EnterAction,
    EnterDecision,
    PopupAction,
    PopupDecision,
    TUIState,
    decide_enter_action,
    decide_popup_key,
)


# ── Strategies ────────────────────────────────────────────────────────────


input_modes = st.sampled_from(["chat", "new_topic"])
popup_modes = st.sampled_from(["hidden", "command", "topic"])
optional_values = st.one_of(st.none(), st.text(max_size=50))
free_text = st.text(max_size=200)


@st.composite
def tui_states(draw) -> TUIState:
    return TUIState(
        input_mode=draw(input_modes),
        popup_mode=draw(popup_modes),
        popup_has_items=draw(st.booleans()),
        popup_selected_value=draw(optional_values),
        input_text=draw(free_text),
    )


popup_keys = st.sampled_from(["up", "down", "tab"])
any_key = st.one_of(
    popup_keys,
    st.sampled_from(["enter", "escape", "space", "a", "1", "f1", ""]),
)


# ── decide_enter_action invariants ────────────────────────────────────────


@given(tui_states())
def test_enter_always_returns_decision(state: TUIState) -> None:
    decision = decide_enter_action(state)
    assert isinstance(decision, EnterDecision)
    assert isinstance(decision.action, EnterAction)
    assert isinstance(decision.value, str)


@given(tui_states())
def test_new_topic_mode_always_routes_to_new_topic(state: TUIState) -> None:
    """input_mode='new_topic' must always win over any popup state."""
    assume(state.input_mode == "new_topic")
    decision = decide_enter_action(state)
    assert decision.action == EnterAction.NEW_TOPIC
    assert decision.value == state.input_text.strip()


@given(tui_states())
def test_popup_branches_require_items_and_selected_value(state: TUIState) -> None:
    """TOPIC_SELECT and COMMAND_SUBMIT must never fire when popup is hidden,
    has no items, or has no selected value."""
    decision = decide_enter_action(state)
    if decision.action in (EnterAction.TOPIC_SELECT, EnterAction.COMMAND_SUBMIT):
        assert state.popup_has_items
        assert state.popup_selected_value is not None
        assert state.popup_mode in ("topic", "command")
        assert state.input_mode != "new_topic"


@given(tui_states())
def test_command_submit_returns_selected_value(state: TUIState) -> None:
    decision = decide_enter_action(state)
    if decision.action == EnterAction.COMMAND_SUBMIT:
        assert state.popup_mode == "command"
        assert decision.value == state.popup_selected_value


@given(tui_states())
def test_topic_select_returns_selected_value(state: TUIState) -> None:
    decision = decide_enter_action(state)
    if decision.action == EnterAction.TOPIC_SELECT:
        assert state.popup_mode == "topic"
        assert decision.value == state.popup_selected_value


@given(tui_states())
def test_submit_requires_non_empty_text(state: TUIState) -> None:
    """SUBMIT must only fire when input_text has non-whitespace content."""
    decision = decide_enter_action(state)
    if decision.action == EnterAction.SUBMIT:
        assert state.input_text.strip() != ""
        assert decision.value == state.input_text


@given(tui_states())
def test_noop_value_is_empty(state: TUIState) -> None:
    decision = decide_enter_action(state)
    if decision.action == EnterAction.NOOP:
        assert decision.value == ""


@given(tui_states())
def test_noop_implies_no_other_action_was_applicable(state: TUIState) -> None:
    """If we returned NOOP, none of the higher-priority branches could fire."""
    decision = decide_enter_action(state)
    if decision.action == EnterAction.NOOP:
        assert state.input_mode != "new_topic"
        # Either no popup match or text is empty
        popup_match = (
            state.popup_has_items
            and state.popup_selected_value is not None
            and state.popup_mode in ("command", "topic")
        )
        assert not popup_match
        assert state.input_text.strip() == ""


@given(tui_states())
def test_new_topic_priority_unaffected_by_popup_state(state: TUIState) -> None:
    """Setting/clearing popup never changes the new_topic outcome."""
    assume(state.input_mode == "new_topic")
    flipped = TUIState(
        input_mode=state.input_mode,
        popup_mode="hidden" if state.popup_mode != "hidden" else "command",
        popup_has_items=not state.popup_has_items,
        popup_selected_value=None if state.popup_selected_value else "/x",
        input_text=state.input_text,
    )
    assert decide_enter_action(state).action == decide_enter_action(flipped).action == EnterAction.NEW_TOPIC


# ── decide_popup_key invariants ───────────────────────────────────────────


@given(any_key, tui_states())
def test_popup_key_always_returns_decision(key: str, state: TUIState) -> None:
    decision = decide_popup_key(key, state)
    assert isinstance(decision, PopupDecision)
    assert isinstance(decision.action, PopupAction)
    assert isinstance(decision.value, str)


@given(tui_states())
def test_unknown_key_is_ignored(state: TUIState) -> None:
    """Any key that isn't up/down/tab must yield IGNORE."""
    for key in ("enter", "escape", "space", "a", "1", "f1", ""):
        assert decide_popup_key(key, state).action == PopupAction.IGNORE


@given(tui_states())
def test_up_routing(state: TUIState) -> None:
    decision = decide_popup_key("up", state)
    if state.popup_has_items:
        assert decision.action == PopupAction.CYCLE_UP
    else:
        assert decision.action == PopupAction.HISTORY_BACK


@given(tui_states())
def test_down_routing(state: TUIState) -> None:
    decision = decide_popup_key("down", state)
    if state.popup_has_items:
        assert decision.action == PopupAction.CYCLE_DOWN
    else:
        assert decision.action == PopupAction.HISTORY_FORWARD


@given(tui_states())
def test_tab_only_completes_in_command_popup(state: TUIState) -> None:
    """Tab must only trigger COMPLETE in command popup with a selected value;
    otherwise IGNORE."""
    decision = decide_popup_key("tab", state)
    if decision.action == PopupAction.COMPLETE:
        assert state.popup_mode == "command"
        assert state.popup_has_items
        assert state.popup_selected_value is not None
        assert decision.value == state.popup_selected_value
    else:
        assert decision.action == PopupAction.IGNORE


@given(tui_states())
def test_tab_in_topic_popup_is_ignored(state: TUIState) -> None:
    assume(state.popup_mode == "topic")
    assert decide_popup_key("tab", state).action == PopupAction.IGNORE


@given(tui_states())
def test_complete_value_only_set_when_action_is_complete(state: TUIState) -> None:
    """COMPLETE always has a non-empty source value; other actions have empty value."""
    for key in ("up", "down", "tab"):
        decision = decide_popup_key(key, state)
        if decision.action != PopupAction.COMPLETE:
            assert decision.value == ""


# ── Cross-function invariants ─────────────────────────────────────────────


@given(tui_states())
def test_decisions_are_deterministic(state: TUIState) -> None:
    """Same state → same decision (no hidden randomness or mutation)."""
    assert decide_enter_action(state) == decide_enter_action(state)
    for key in ("up", "down", "tab"):
        assert decide_popup_key(key, state) == decide_popup_key(key, state)
