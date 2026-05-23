from nanobot.fork.cli.tui_keys import (
    EnterAction,
    PopupAction,
    TUIState,
    decide_enter_action,
    decide_popup_key,
)


def _state(**kwargs) -> TUIState:
    base = dict(
        input_mode="chat",
        popup_mode="hidden",
        popup_has_items=False,
        popup_selected_value=None,
        input_text="",
    )
    base.update(kwargs)
    return TUIState(**base)


def _decide(**kwargs):
    return decide_enter_action(_state(**kwargs))


def test_new_topic_mode_wins_over_popup() -> None:
    """new_topic must take priority even if a popup is somehow still showing."""
    decision = _decide(
        input_mode="new_topic",
        popup_mode="command",
        popup_has_items=True,
        popup_selected_value="/new",
        input_text="mytopic",
    )
    assert decision.action == EnterAction.NEW_TOPIC
    assert decision.value == "mytopic"


def test_new_topic_strips_whitespace() -> None:
    decision = _decide(input_mode="new_topic", input_text="  test name  ")
    assert decision.action == EnterAction.NEW_TOPIC
    assert decision.value == "test name"


def test_new_topic_empty_text_allowed() -> None:
    """Empty topic name still routes to NEW_TOPIC; caller decides default name."""
    decision = _decide(input_mode="new_topic", input_text="")
    assert decision.action == EnterAction.NEW_TOPIC
    assert decision.value == ""


def test_topic_popup_select() -> None:
    decision = _decide(
        popup_mode="topic",
        popup_has_items=True,
        popup_selected_value="existing_topic",
        input_text="exi",
    )
    assert decision.action == EnterAction.TOPIC_SELECT
    assert decision.value == "existing_topic"


def test_command_popup_submit() -> None:
    decision = _decide(
        popup_mode="command",
        popup_has_items=True,
        popup_selected_value="/new",
        input_text="/n",
    )
    assert decision.action == EnterAction.COMMAND_SUBMIT
    assert decision.value == "/new"


def test_popup_without_items_falls_through() -> None:
    """If popup mode is set but items are empty, fall through to normal submit."""
    decision = _decide(
        popup_mode="command",
        popup_has_items=False,
        popup_selected_value=None,
        input_text="hello",
    )
    assert decision.action == EnterAction.SUBMIT
    assert decision.value == "hello"


def test_normal_submit() -> None:
    decision = _decide(input_text="hello world")
    assert decision.action == EnterAction.SUBMIT
    assert decision.value == "hello world"


def test_noop_when_input_empty_and_no_popup() -> None:
    decision = _decide(input_text="")
    assert decision.action == EnterAction.NOOP
    assert decision.value == ""


def test_noop_when_input_whitespace_only() -> None:
    decision = _decide(input_text="   ")
    assert decision.action == EnterAction.NOOP


# ── popup key decisions ────────────────────────────────────────────────────


def test_up_cycles_popup_when_items_present() -> None:
    decision = decide_popup_key(
        "up", _state(popup_mode="command", popup_has_items=True, popup_selected_value="/new"),
    )
    assert decision.action == PopupAction.CYCLE_UP


def test_up_walks_history_when_popup_empty() -> None:
    decision = decide_popup_key("up", _state())
    assert decision.action == PopupAction.HISTORY_BACK


def test_down_cycles_popup_when_items_present() -> None:
    decision = decide_popup_key(
        "down", _state(popup_mode="topic", popup_has_items=True, popup_selected_value="t"),
    )
    assert decision.action == PopupAction.CYCLE_DOWN


def test_down_walks_history_when_popup_empty() -> None:
    decision = decide_popup_key("down", _state())
    assert decision.action == PopupAction.HISTORY_FORWARD


def test_tab_completes_in_command_popup() -> None:
    decision = decide_popup_key(
        "tab",
        _state(popup_mode="command", popup_has_items=True, popup_selected_value="/new"),
    )
    assert decision.action == PopupAction.COMPLETE
    assert decision.value == "/new"


def test_tab_ignored_in_topic_popup() -> None:
    decision = decide_popup_key(
        "tab",
        _state(popup_mode="topic", popup_has_items=True, popup_selected_value="t"),
    )
    assert decision.action == PopupAction.IGNORE


def test_tab_ignored_when_popup_hidden() -> None:
    decision = decide_popup_key("tab", _state())
    assert decision.action == PopupAction.IGNORE


def test_unknown_key_is_ignored() -> None:
    decision = decide_popup_key("space", _state())
    assert decision.action == PopupAction.IGNORE
