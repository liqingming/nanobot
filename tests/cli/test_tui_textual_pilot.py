"""Pilot-based integration tests for TextualTUI Enter routing.

Covers the /new topic-creation flow end-to-end: typing /new, pressing Enter,
typing the topic name, pressing Enter again. The topic name must reach the
new_topic callback and NOT be submitted as a chat message — that was the
original bug we fixed by unifying Enter routing through decide_enter_action.
"""
from __future__ import annotations

import pytest
from textual.events import Paste

from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE, TextualTUI, _compact_path_label

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
async def test_skin_is_disabled_by_default_and_keeps_opaque_background() -> None:
    tui = TextualTUI()

    async with tui._app.run_test() as pilot:
        await pilot.pause()
        assert "glass-skin" not in tui._app.screen.classes
        output = tui._app.query_one("#output")
        assert output.styles.background.hex == "#0C0C0C"
        assert output._user_background == "#2d2d2d"
        assert tui._app.query_one("#popup").styles.background.hex == "#0C0C0C"


@pytest.mark.asyncio
async def test_enabled_skin_uses_terminal_default_canvas_but_opaque_popup() -> None:
    tui = TextualTUI(skin_enabled=True)

    async with tui._app.run_test() as pilot:
        await pilot.pause()
        assert "glass-skin" in tui._app.screen.classes
        assert tui._app.query_one("#output")._user_background == "default"
        for selector in ("#output", "#input", "#status"):
            background = tui._app.query_one(selector).styles.background
            assert background.ansi == -1
            assert background.rich_color.is_default
        output = tui._app.query_one("#output")
        assert output.styles.overflow_x == "hidden"
        assert output.styles.scrollbar_size_horizontal == 0
        scrollbar_background = output.styles.scrollbar_background
        assert scrollbar_background.ansi == -1
        assert scrollbar_background.rich_color.is_default
        assert tui._app.query_one("#popup").styles.background.hex == "#0C0C0C"
        input_widget = tui._app.query_one("#input")
        cursor_line = input_widget.get_component_rich_style("text-area--cursor-line")
        assert cursor_line.bgcolor.is_default
        cursor = input_widget.get_component_rich_style("text-area--cursor")
        assert cursor.bgcolor.name == "#ffffff"
        assert cursor.color.name == "#000000"

        # Verify the final compositor output, after Textual line filters. The
        # terminal-default background must survive rather than becoming #0c0c0c.
        strips = tui._app.screen._compositor.render_strips()
        ordinary_backgrounds = {
            style.bgcolor
            for strip in strips
            for _text, style, _control in strip._segments
            if style is not None and style.bgcolor is not None
        }
        assert any(background.is_default for background in ordinary_backgrounds)
        assert not any(str(background) == "Color('#0c0c0c', ColorType.TRUECOLOR, triplet=ColorTriplet(red=12, green=12, blue=12))" for background in ordinary_backgrounds)


@pytest.mark.asyncio
async def test_argument_command_selection_enters_edit_mode_without_submit() -> None:
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([("/model", "Switch model preset", "edit")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "/mo")
        await pilot.pause()
        assert tui._popup_mode == "command"

        await pilot.press("enter")
        await pilot.pause()

        inp = app.query_one("#input")
        assert inp.value == "/model "
        assert submitted == []
        assert tui._popup_mode == "hidden"


@pytest.mark.asyncio
async def test_submit_command_selection_still_submits_immediately() -> None:
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([("/status", "Show status", "submit")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "/sta")
        await pilot.pause()
        assert tui._popup_mode == "command"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["/status"]
        assert tui._popup_mode == "hidden"


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
async def test_startup_new_topic_selection_routes_name_to_topic_callback() -> None:
    """Startup picker new-topic selection must arm new_topic before typing."""
    tui = TextualTUI()
    submitted: list[str] = []
    topic_callbacks: list[str] = []

    async def confirm_topic(name: str) -> None:
        topic_callbacks.append(name)

    def on_startup_select(name: str) -> None:
        if name == "[ 新建话题 ]":
            tui.enter_new_topic_mode(confirm_topic)

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.show_topic_popup([("[ 新建话题 ]", "[ 新建话题 ]")], on_startup_select)
        await pilot.pause()

        await pilot.press("enter")
        await _press_text(pilot, "startup_topic")
        await pilot.press("enter")
        await pilot.pause()

        assert topic_callbacks == ["startup_topic"]
        assert submitted == []

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


def test_compact_path_label_preserves_short_paths() -> None:
    assert _compact_path_label(r"E:\learn\nanobot") == r"E:\learn\nanobot"


def test_compact_path_label_shortens_long_paths() -> None:
    label = _compact_path_label(
        r"E:\very\long\workspace\path\with\many\segments\nanobot",
        max_len=24,
    )
    assert label.startswith("E:")
    assert "..." in label
    assert label.endswith(r"segments\nanobot")


@pytest.mark.asyncio
async def test_topic_bar_shows_workspace_without_topic() -> None:
    tui = TextualTUI(workspace=r"E:\learn\nanobot")
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#topic-bar")
        renderable = bar.render()
        text = getattr(renderable, "plain", str(renderable))
        assert r"E:\learn\nanobot" in text
        assert "·" not in text


@pytest.mark.asyncio
async def test_topic_bar_shows_workspace_and_topic() -> None:
    tui = TextualTUI(workspace=r"E:\learn\nanobot")
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.set_topic("feature-diff")
        await pilot.pause()
        assert app.title == "nanobot — feature-diff"
        bar = app.query_one("#topic-bar")
        renderable = bar.render()
        text = getattr(renderable, "plain", str(renderable))
        assert r"E:\learn\nanobot" in text
        assert "feature-diff" in text
        assert "·" in text


@pytest.mark.asyncio
async def test_command_popup_scrolls_with_selection_and_shows_remaining_count() -> None:
    tui = TextualTUI()
    tui.set_commands([(f"/command-{i}", f"Command {i}") for i in range(8)])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "/")
        await pilot.pause()

        assert tui._popup_idx == 0
        assert tui._popup_visible_range() == (0, 6)
        popup = app.query_one("#popup")
        text = getattr(popup.render(), "plain", str(popup.render()))
        assert "command-0" in text
        assert "command-5" in text
        assert "command-6" not in text
        assert "↓ 还有 2 项" in text

        await pilot.press(*(["down"] * 6))
        await pilot.pause()

        assert tui._popup_idx == 6
        assert tui._popup_visible_range() == (1, 7)
        text = getattr(popup.render(), "plain", str(popup.render()))
        assert "↑ 还有 1 项" in text
        assert "command-0" not in text
        assert "command-6" in text
        assert "↓ 还有 1 项" in text


@pytest.mark.asyncio
async def test_topic_popup_can_show_cache_label_but_select_topic_value() -> None:
    tui = TextualTUI()
    selected: list[str] = []

    async def on_select(value: str) -> None:
        selected.append(value)

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.show_topic_popup([("topic-a", "topic-a  [1.2 KB]")], on_select)
        await pilot.pause()
        assert tui._popup_items == [("topic-a", "topic-a  [1.2 KB]")]
        popup = app.query_one("#popup")
        renderable = popup.render()
        text = getattr(renderable, "plain", str(renderable))
        assert "topic-a  [1.2 KB]" in text

        await pilot.press("enter")
        await pilot.pause()

        assert selected == ["topic-a"]


@pytest.mark.asyncio
async def test_multiline_paste_is_submitted_as_one_message() -> None:
    """Bracketed paste should display a token and submit the full payload."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "before ")
        await pilot.pause()
        inp = app.query_one("#input")
        inp._on_paste(Paste("alpha\r\nbeta\ngamma"))
        await pilot.pause()
        await _press_text(pilot, " after")
        await pilot.pause()

        assert inp.value == "before [pasted 3 lines] after"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["before alpha\nbeta\ngamma after"]


@pytest.mark.asyncio
async def test_single_line_paste_stays_visible_in_input() -> None:
    """Single-line paste should behave like ordinary input text."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#input")
        inp._on_paste(Paste("hello from clipboard"))
        await pilot.pause()

        assert inp.value == "hello from clipboard"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["hello from clipboard"]


@pytest.mark.asyncio
async def test_large_single_line_paste_uses_hidden_payload() -> None:
    """Large pastes should not flood the input widget but must submit fully."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    payload = "x" * (5 * 1024 + 1)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#input")
        inp._on_paste(Paste(payload))
        await pilot.pause()

        assert inp.value == f"[pasted {len(payload)} chars]"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == [payload]


@pytest.mark.asyncio
async def test_app_level_large_paste_routes_to_input() -> None:
    """Large paste events received by the app should still populate the input."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    payload = "y" * (5 * 1024 + 1)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#input")
        app.on_paste(Paste(payload))
        await pilot.pause()

        assert inp.value == f"[pasted {len(payload)} chars]"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == [payload]

def test_file_edit_diff_block_numbers_lines_and_folds_long_diff(monkeypatch) -> None:
    tui = TextualTUI()
    written: list[str] = []

    def capture(*items) -> None:
        for item in items:
            written.append(getattr(item, "plain", str(item)))

    monkeypatch.setattr(tui, "_log_write", capture)
    diff_lines = ["--- a/app.py", "+++ b/app.py", "@@ -10,65 +10,65 @@"]
    diff_lines.extend(f"-old {idx}" for idx in range(65))
    diff_lines.extend(f"+new {idx}" for idx in range(65))

    block = tui._format_file_edit_event({
        "phase": "end",
        "status": "done",
        "path": "app.py",
        "added": 65,
        "deleted": 65,
        "diff": "\n".join(diff_lines),
        "diff_total_lines": len(diff_lines),
    })

    assert block is not None
    tui._write_file_edit_block(block)

    text = "\n".join(written)
    assert "app.py (+65 -65)" in text
    assert "10 -old 0" in text
    assert "10 +new 0" in text
    assert "--- a/app.py" not in text
    assert "+++ b/app.py" not in text
    assert "@@ -10,65 +10,65 @@" not in text
    assert "已折叠 10 行" in text


def test_file_edit_diff_single_line_number_uses_old_for_delete_new_for_add() -> None:
    lines = TextualTUI._number_file_diff_lines(
        "@@ -8,3 +8,4 @@\n context\n-old\n+new\n+extra\n tail"
    )

    assert lines == [
        (8, " context"),
        (9, "-old"),
        (9, "+new"),
        (10, "+extra"),
        (11, " tail"),
    ]


def test_file_edit_diff_new_file_has_sequential_new_line_numbers() -> None:
    lines = TextualTUI._number_file_diff_lines(
        "--- /dev/null\n+++ b/plan.md\n@@ -0,0 +1,3 @@\n+# title\n+\n+body"
    )

    assert lines == [(1, "+# title"), (2, "+"), (3, "+body")]


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_enter_submits() -> None:
    """Shift+Enter should add a line break; Enter should submit the composer."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "alpha")
        await pilot.pause()
        await pilot.press("shift+enter")
        await pilot.pause()
        await _press_text(pilot, "beta")
        await pilot.pause()

        inp = app.query_one("#input")
        assert inp.value == "alpha\nbeta"
        assert submitted == []

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["alpha\nbeta"]


@pytest.mark.asyncio
async def test_editing_multiline_paste_placeholder_drops_hidden_payload() -> None:
    """Editing the paste token should submit the edited visible text instead."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#input")
        inp._on_paste(Paste("alpha\nbeta"))
        await pilot.pause()

        inp.value = "manual edit"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["manual edit"]


@pytest.mark.asyncio
async def test_multiline_paste_can_be_inserted_at_cursor() -> None:
    """A multiline paste token should preserve text before and after it."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "hello world")
        await pilot.pause()
        inp = app.query_one("#input")
        inp.cursor_position = len("hello ")
        inp._on_paste(Paste("one\ntwo"))
        await pilot.pause()

        assert inp.value == "hello [pasted 2 lines]world"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["hello one\ntwoworld"]


@pytest.mark.asyncio
async def test_multiple_multiline_pastes_restore_in_display_order() -> None:
    """Multiple paste tokens should restore to their own payloads."""
    tui = TextualTUI()
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#input")
        inp._on_paste(Paste("a\nb"))
        await pilot.pause()
        await _press_text(pilot, " + ")
        await pilot.pause()
        inp._on_paste(Paste("c\nd\ne"))
        await pilot.pause()

        assert inp.value == "[pasted 2 lines] + [pasted 3 lines]"

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["a\nb + c\nd\ne"]


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
async def test_skills_command_submit_does_not_start_pre_submit_spinner() -> None:
    """The /skills command is local UI output, so Enter must not start a turn."""
    tui = TextualTUI()
    submitted: list[str] = []
    pre_submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    def on_pre_submit(text: str) -> None:
        pre_submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_on_pre_submit(on_pre_submit)
    tui.set_commands([("/skills", "List available skills")])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        await _press_text(pilot, "/skills")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["/skills"]
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
async def test_show_question_popup_appends_custom_input_row() -> None:
    """popup_items must always have one extra entry (sentinel value, custom
    display label) appended after the LLM-provided options."""
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
        # 2 options + 1 custom row
        assert len(tui._popup_items) == 3
        assert tui._popup_items[-1][0] == tui._CUSTOM_INPUT_SENTINEL
        assert "自定义" in tui._popup_items[-1][1]


@pytest.mark.asyncio
async def test_show_question_popup_custom_input_routes_typed_text_as_answer() -> None:
    """User picks the custom row → input mode → typed text becomes answer."""
    tui = TextualTUI()
    tui.set_commands([])
    completed: list[dict | None] = []

    async def on_complete(answers: dict | None) -> None:
        completed.append(answers)

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        questions = [{
            "question": "What approach?",
            "options": [
                {"label": "A", "description": ""},
                {"label": "B", "description": ""},
            ],
        }]
        tui.show_question_popup(questions, on_complete)
        await pilot.pause()
        # Navigate down past A, B to the custom row (idx 2)
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        assert tui._popup_items[tui._popup_idx][0] == tui._CUSTOM_INPUT_SENTINEL
        # Enter selects → opens input mode for free text
        await pilot.press("enter")
        await pilot.pause()
        assert tui._input_mode == "new_topic"
        assert tui._popup_mode == "hidden"
        # Type a custom answer + Enter
        for ch in "C":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()
        assert completed == [{"What approach?": "C"}]


@pytest.mark.asyncio
async def test_show_question_popup_empty_custom_input_reshows_popup() -> None:
    """Empty custom answer shouldn't kill the question — it re-prompts."""
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
                {"label": "x", "description": ""},
                {"label": "y", "description": ""},
            ],
        }]
        tui.show_question_popup(questions, on_complete)
        await pilot.pause()
        # Move to custom row, Enter
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        # Just Enter (empty) — should not complete; popup should re-show
        await pilot.press("enter")
        await pilot.pause()
        assert completed == []
        assert tui._popup_mode == "topic"
        # Now pick x
        await pilot.press("enter")
        await pilot.pause()
        assert completed == [{"Pick?": "x"}]


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
async def test_stream_delta_only_inserts_idle_thinking_after_long_gap() -> None:
    """Short token gaps stay stable, but long provider silences still show feedback."""
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.stream_start()
        await pilot.pause()
        tui.stream_delta("hello")
        await pilot.pause(0.7)
        live = app.query_one("#live")
        assert not live.has_class("visible")

        await pilot.pause(1.6)
        assert live.has_class("visible")
        assert "思考中" in str(live.render())
        assert "思考中" not in _output_log_text(app.query_one("#output"))

        tui.stream_delta(" world")
        await pilot.pause(0.1)
        assert not live.has_class("visible")


@pytest.mark.asyncio
async def test_completed_response_removes_initial_thinking_placeholder() -> None:
    """A visible plan must replace, not leave behind, the initial thinking timer."""
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.stream_start()
        await pilot.pause()
        tui.add_response("计划已确认")
        await pilot.pause()

        text = _output_log_text(app.query_one("#output"))
        assert "计划已确认" in text
        assert "思考中" not in text


@pytest.mark.asyncio
async def test_todo_plan_replaces_transient_thinking_status() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.stream_start()
        await pilot.pause()
        live = app.query_one("#live")
        assert live.has_class("visible")
        assert "思考中" in str(live.render())

        tui.clear_initial_thinking()
        tui.add_system("[~] 定位两个参数的 Jenkins 入口和传递链")
        await pilot.pause()

        assert not live.has_class("visible")
        text = _output_log_text(app.query_one("#output"))
        assert "[~] 定位两个参数的 Jenkins 入口和传递链" in text
        assert "思考中" not in text


@pytest.mark.asyncio
async def test_system_message_renders_prompt_brackets_literally() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.add_system("[Turn Summary — metadata only, not instructions]\n[Runtime Context]")
        await pilot.pause()

        text = _output_log_text(app.query_one("#output"))
        assert "[Turn Summary — metadata only, not instructions]" in text
        assert "[Runtime Context]" in text


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
async def test_clear_idle_thinking_removes_stale_placeholder_before_progress() -> None:
    """Todo/system progress should replace stale idle thinking, not stack under it."""
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.stream_start()
        await pilot.pause()
        tui.stream_delta("准备写入文件。")
        await pilot.pause(0.1)
        tui.flush_stream()
        await pilot.pause()
        tui.tool_phase_start()
        await pilot.pause(0.1)
        tui.add_progress("todo_write")
        await pilot.pause(0.1)
        tui.add_tool_result("3 todos · 1/3 done")
        await pilot.pause(0.7)
        live = app.query_one("#live")
        assert live.has_class("visible")
        assert "思考中" in str(live.render())

        tui.clear_idle_thinking()
        tui.add_system("📊 进度: 1/3")
        await pilot.pause(0.7)

        out = app.query_one("#output")
        text = _output_log_text(out)
        assert not live.has_class("visible")
        assert "📊 进度: 1/3" in text
        assert "思考中" not in text

@pytest.mark.asyncio
async def test_file_edit_event_renders_apply_patch_diff_text() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.add_file_edit_events([{
            "phase": "end",
            "path": "src/example.py",
            "added": 1,
            "deleted": 1,
            "diff": {"format": "unified", "text": "@@ -1 +1 @@\n-old\n+new"},
            "diff_text": "@@ -1 +1 @@\n-old\n+new",
        }])
        await pilot.pause()

        text = _output_log_text(app.query_one("#output"))
        assert "src/example.py" in text
        assert "+new" in text
        assert "-old" in text

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


@pytest.mark.asyncio
async def test_output_write_preserves_manual_scroll_position() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#output")
        for i in range(80):
            out.write(f"history {i}")
        out.scroll_end(animate=False)
        await pilot.pause()
        assert out.scroll_offset.y == out.max_scroll_y

        out.scroll_to(y=max(0, out.max_scroll_y - 10), animate=False)
        await pilot.pause()
        manual_y = out.scroll_offset.y
        assert manual_y < out.max_scroll_y

        out.write("new streamed line")
        await pilot.pause()

        assert out.scroll_offset.y == manual_y


@pytest.mark.asyncio
async def test_output_write_follows_when_already_at_bottom() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#output")
        for i in range(80):
            out.write(f"history {i}")
        out.scroll_end(animate=False)
        await pilot.pause()

        out.write("new streamed line")
        await pilot.pause()

        assert out.scroll_offset.y == out.max_scroll_y


def _output_log_text(out) -> str:
    parts: list[str] = []
    for line in out.lines:
        for segment in getattr(line, "_segments", []):
            parts.append(segment[0])
        parts.append("\n")
    return "".join(parts)


@pytest.mark.asyncio
async def test_welcome_shows_reasoning_effort() -> None:
    tui = TextualTUI(model="gpt-5.5", reasoning_effort="high")
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#output")

        text = _output_log_text(out)
        assert "gpt-5.5" in text
        assert "reasoning: high" in text


@pytest.mark.asyncio
async def test_stream_delta_renders_after_debounce() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#output")
        tui.stream_start()
        await pilot.pause()

        tui.stream_delta("debounced hello")
        await pilot.pause(0.02)
        assert "debounced hello" not in _output_log_text(out)

        await pilot.pause(0.12)
        assert "debounced hello" in _output_log_text(out)


@pytest.mark.asyncio
async def test_stream_start_follows_latest_when_user_sends_from_history_position() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#output")
        for i in range(80):
            out.write(f"history {i}")
        tui.stream_start()
        await pilot.pause()
        out.scroll_to(y=max(0, out.max_scroll_y - 10), animate=False)
        out.mark_user_scroll()
        await pilot.pause()
        manual_y = out.scroll_offset.y
        assert manual_y < out.max_scroll_y

        tui.stream_start()
        await pilot.pause()
        assert out.scroll_offset.y == out.max_scroll_y

        tui.stream_delta("fresh reply")
        await pilot.pause(0.12)

        assert "fresh reply" in _output_log_text(out)
        assert out.scroll_offset.y == out.max_scroll_y


@pytest.mark.asyncio
async def test_stream_delta_debounce_respects_manual_scroll_window() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#output")
        for i in range(80):
            out.write(f"history {i}")
        tui.stream_start()
        await pilot.pause()
        out.scroll_to(y=max(0, out.max_scroll_y - 10), animate=False)
        out.mark_user_scroll()
        await pilot.pause()
        manual_y = out.scroll_offset.y

        tui.stream_delta("hidden while scrolling")
        await pilot.pause(0.12)

        assert out.scroll_offset.y == manual_y
        assert "hidden while scrolling" not in _output_log_text(out)


@pytest.mark.asyncio
async def test_output_history_reflows_when_terminal_width_changes() -> None:
    tui = TextualTUI()
    tui.set_commands([])
    long_text = "中文历史消息需要根据窗口宽度自动换行 " * 8
    messages = [
        {"role": "user", "content": long_text, "timestamp": "2026-07-19T12:00:00"},
        {
            "role": "assistant",
            "content": "## 标题\n\n" + long_text + "\n\n```python\nprint('resize')\n```",
            "timestamp": "2026-07-19T12:00:01",
        },
    ]

    app = tui._app
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.clear_output()
        tui.load_session_history(messages)
        await pilot.pause()
        out = app.query_one("#output")
        wide_lines = len(out.lines)
        logical_records = len(out._logical_records)
        assert out.scroll_offset.y == out.max_scroll_y
        assert any(record.get("user") for record in out._logical_records)

        await pilot.resize_terminal(52, 30)
        await pilot.pause()
        narrow_lines = len(out.lines)
        assert narrow_lines > wide_lines
        assert len(out._logical_records) == logical_records
        assert out.scroll_offset.y == out.max_scroll_y
        assert "中文历史消息" in _output_log_text(out)
        assert out._user_ranges

        await pilot.resize_terminal(120, 30)
        await pilot.pause()
        assert len(out.lines) < narrow_lines
        assert len(out._logical_records) == logical_records
        assert out.scroll_offset.y == out.max_scroll_y


@pytest.mark.asyncio
async def test_output_resize_preserves_manual_reading_record() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app.clear_output()
        out = app.query_one("#output")
        for i in range(40):
            out.write(f"record {i}: " + ("自适应内容 " * 10))
        out.scroll_end(animate=False)
        await pilot.pause()
        target_record = 18
        old_start = out._record_spans[target_record][0]
        out.scroll_to(y=old_start, animate=False)
        await pilot.pause()
        assert out.scroll_offset.y < out.max_scroll_y

        await pilot.resize_terminal(58, 24)
        await pilot.pause()
        new_start, new_end = out._record_spans[target_record]
        assert new_start <= out.scroll_offset.y < new_end


@pytest.mark.asyncio
async def test_stream_anchor_survives_resize_and_final_markdown_replaces_snapshot() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test(size=(96, 24)) as pilot:
        await pilot.pause()
        tui.stream_start()
        tui.stream_delta("流式响应 " * 20)
        await pilot.pause()
        out = app.query_one("#output")
        assert tui._stream_buf.startswith("流式响应")

        await pilot.resize_terminal(50, 24)
        await pilot.pause()
        tui.flush_stream()
        await pilot.pause()

        text = _output_log_text(out)
        assert "流式响应" in text
        assert len(out._logical_records) == len(out._record_spans)
        assert 0 <= tui._tool_placeholder_line <= len(out.lines)


@pytest.mark.asyncio
async def test_escape_cancels_active_turn_even_when_popup_is_visible() -> None:
    tui = TextualTUI()
    tui.set_commands([("/status", "status")])
    cancelled: list[bool] = []

    async def cancel() -> None:
        cancelled.append(True)

    tui.set_on_cancel(cancel)
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.set_is_processing(True)
        app.set_input_value("/s")
        tui._on_input_changed("/s")
        assert tui._popup_mode != "hidden"

        await pilot.press("escape")
        await pilot.pause()

        assert cancelled == [True]
        assert tui._popup_mode == "hidden"


@pytest.mark.asyncio
async def test_empty_enter_jumps_output_to_bottom_without_submitting() -> None:
    tui = TextualTUI(render_markdown=False)
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    async with tui._app.run_test(size=(60, 12)) as pilot:
        output = tui._app.query_one("#output")
        for index in range(80):
            output.write(f"history-{index}")
        output.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()
        output.scroll_to(y=0, animate=False, immediate=True, force=True)
        await pilot.pause()
        assert not output.is_at_bottom()

        await pilot.press("enter")
        await pilot.pause()

        assert output.is_at_bottom()
        assert submitted == []
        assert tui._app.query_one("#input").value == ""


@pytest.mark.asyncio
async def test_ctrl_arrows_navigate_only_loaded_user_messages_without_editing_input() -> None:
    tui = TextualTUI(render_markdown=False)
    messages: list[dict] = []
    for index in range(1, 5):
        messages.extend([
            {
                "role": "user",
                "content": f"user message {index}",
                "timestamp": f"2026-07-23T10:0{index}:00",
                "_transcript_id": f"user-{index}",
            },
            {
                "role": "assistant",
                "content": (f"assistant response {index} " * 12).strip(),
                "timestamp": f"2026-07-23T10:0{index}:01",
                "_transcript_id": f"assistant-{index}",
            },
        ])

    async with tui._app.run_test(size=(60, 14)) as pilot:
        tui._app.clear_output()
        tui.load_session_history(messages)
        tui._app.set_input_value("draft stays here")
        output = tui._app.query_one("#output")
        output.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()

        assert [message_id for message_id, _line in output.user_message_targets()] == [
            "user-1", "user-2", "user-3", "user-4",
        ]

        # At the first loaded message, previous is a no-op rather than wrapping.
        output.scroll_to(y=0, animate=False, immediate=True, force=True)
        output.reset_user_navigation()
        await pilot.press("ctrl+up")
        await pilot.pause()
        assert output._user_navigation_id is None

        output.scroll_end(animate=False, immediate=True, force=True)
        await pilot.press("ctrl+up")
        await pilot.pause()
        assert output._user_navigation_id == "user-4"
        first_position = output.scroll_offset.y

        await pilot.press("ctrl+up")
        await pilot.pause()
        assert output._user_navigation_id == "user-3"
        assert output.scroll_offset.y < first_position

        await pilot.press("ctrl+down")
        await pilot.pause()
        assert output._user_navigation_id == "user-4"

        # The last boundary does not wrap, and navigation never edits input text.
        await pilot.press("ctrl+down")
        await pilot.pause()
        assert output._user_navigation_id == "user-4"
        assert tui._app.query_one("#input").value == "draft stays here"


@pytest.mark.asyncio
async def test_user_message_navigation_stops_at_first_message_and_survives_reflow() -> None:
    tui = TextualTUI(render_markdown=False)
    messages = [
        {
            "role": role,
            "content": (f"{role} {index} 中文长内容 " * 8).strip(),
            "timestamp": f"2026-07-23T11:0{index}:00",
            "_transcript_id": f"{role}-{index}",
        }
        for index in range(1, 4)
        for role in ("user", "assistant")
    ]

    async with tui._app.run_test(size=(80, 14)) as pilot:
        tui._app.clear_output()
        tui.load_session_history(messages)
        output = tui._app.query_one("#output")
        output.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()

        for _ in range(5):
            await pilot.press("ctrl+up")
            await pilot.pause()
        assert output._user_navigation_id == "user-1"

        await pilot.resize_terminal(48, 14)
        await pilot.pause()
        targets = dict(output.user_message_targets())
        assert output._user_navigation_id == "user-1"
        assert output.scroll_offset.y <= targets["user-1"]

        await pilot.press("ctrl+down")
        await pilot.pause()
        assert output._user_navigation_id == "user-2"


@pytest.mark.asyncio
async def test_user_message_navigation_includes_prepended_history_page() -> None:
    tui = TextualTUI(render_markdown=False)
    recent = [
        {
            "role": "user",
            "content": "recent user",
            "timestamp": "2026-07-23T12:02:00",
            "_transcript_id": "user-recent",
        },
        {
            "role": "assistant",
            "content": "recent assistant " * 30,
            "timestamp": "2026-07-23T12:02:01",
            "_transcript_id": "assistant-recent",
        },
    ]
    older = [
        {
            "role": "user",
            "content": "older user",
            "timestamp": "2026-07-23T12:01:00",
            "_transcript_id": "user-older",
        },
        {
            "role": "assistant",
            "content": "older assistant",
            "timestamp": "2026-07-23T12:01:01",
            "_transcript_id": "assistant-older",
        },
    ]

    async def load_older(_before_offset):
        return older, None, False

    async with tui._app.run_test(size=(60, 12)) as pilot:
        tui._app.clear_output()
        tui.load_session_history(recent)
        tui.set_history_page_loader(load_older, before_offset=2, has_older=True)
        output = tui._app.query_one("#output")
        output.scroll_to(y=0, animate=False, immediate=True, force=True)
        tui._request_older_history()
        await pilot.pause()
        await pilot.pause()

        assert [message_id for message_id, _line in output.user_message_targets()] == [
            "user-older", "user-recent",
        ]
        output.scroll_end(animate=False, immediate=True, force=True)
        output.reset_user_navigation()
        await pilot.press("ctrl+up")
        await pilot.press("ctrl+up")
        await pilot.pause()
        assert output._user_navigation_id == "user-older"


@pytest.mark.asyncio
async def test_welcome_page_lists_bound_shortcuts_and_common_commands(tmp_path):
    tui = TextualTUI(history_file=tmp_path / "history")
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        output = app.query_one("#output")
        welcome = _output_log_text(output)

    for shortcut in (
        "PageUp / PageDown",
        "Ctrl+↑ / Ctrl+↓",
        "↑ / ↓",
        "ESC",
        "Ctrl+B",
        "F6",
        "Ctrl+C / Ctrl+D",
        "鼠标拖选",
    ):
        assert shortcut in welcome

    for command in (
        "/rename",
        "/resume",
        "/clear",
        "/todos",
        "/continue",
        "/bookmarks",
        "/bookmarks-clear",
        "/model",
        "/status",
        "/system-prompt",
        "/skin",
        "/exit",
    ):
        assert command in welcome

    for unavailable_command in (
        "/new",
        "/topics",
        "/clear-bookmarks",
        "/preset",
        "/ctx",
        "/copy",
    ):
        assert unavailable_command not in welcome
    assert "Ctrl+C / Ctrl+D" in welcome
    assert "复制选区或退出" in welcome
    assert "选择要复制的文本" in welcome
    assert "自动复制到剪贴板" not in welcome
    assert "/skin [参数]" in welcome
    assert "无参数：打开背景图选择列表" in welcome
    assert "list：列出背景图；next / prev：上一张 / 下一张" in welcome
    assert "random：随机切换；编号 / 文件名：切换到指定背景图" in welcome
    assert "输入 / 可查看完整命令列表" in welcome
    assert "Ctrl+Shift+B" not in welcome
    assert "Alt+B" not in welcome
    assert "Ctrl+Alt+B" not in welcome


@pytest.mark.asyncio
async def test_todo_bar_visibility_preserves_output_bottom_anchor() -> None:
    tui = TextualTUI()
    tui.set_commands([])

    app = tui._app
    async with app.run_test(size=(80, 14)) as pilot:
        await pilot.pause()
        out = app.query_one("#output")
        for i in range(80):
            out.write(f"history {i}")
        out.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()
        assert out.is_at_bottom()

        app.update_todo_bar("⚡ 正在执行 (0/3)")
        # Visibility changes take effect after Textual's next layout refresh.
        await pilot.pause(0.05)

        assert out.is_at_bottom()
        assert out.scroll_offset.y == out.max_scroll_y

        app.update_todo_bar("")
        await pilot.pause(0.05)

        assert out.is_at_bottom()
        assert out.scroll_offset.y == out.max_scroll_y
