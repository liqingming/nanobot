import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from nanobot.fork.cli.tui import PromptTUI
from nanobot.fork.cli.tui_base import input_history_path
from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE, TextualTUI


def _write_history(path, *entries: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for entry in entries:
            for line in entry.splitlines():
                file.write(f"+{line}\n")
            file.write("\n")


def test_input_history_path_is_stable_and_separates_topics(tmp_path) -> None:
    base = tmp_path / "cli_history"

    first = input_history_path(base, "cli:topic-a")
    same = input_history_path(base, "cli:topic-a")
    second = input_history_path(base, "cli:topic-b")

    assert first == same
    assert first != second
    assert first.parent == tmp_path / "topics"
    assert "topic-a" not in first.name


def test_textual_history_isolated_and_restored_per_topic(tmp_path) -> None:
    base = tmp_path / "cli_history"
    topic_a = "cli:topic-a"
    topic_b = "cli:topic-b"
    _write_history(input_history_path(base, topic_a), "a-old", "a-new")
    _write_history(input_history_path(base, topic_b), "b-only")
    tui = TextualTUI(history_file=str(base))

    tui.set_input_history_topic(topic_a)
    assert tui._history_backward() == "a-new"
    assert tui._history_backward() == "a-old"

    tui.set_input_history_topic(topic_b)
    assert tui._history_pos == -1
    assert tui._history_backward() == "b-only"

    tui.set_input_history_topic(topic_a)
    assert tui._history_pos == -1
    assert tui._history_backward() == "a-new"


def test_textual_new_topic_history_starts_empty(tmp_path) -> None:
    tui = TextualTUI(history_file=str(tmp_path / "cli_history"))

    tui.set_input_history_topic("cli:new-topic")

    assert tui._history == []
    assert tui._history_backward() is None


@pytest.mark.skipif(not _TEXTUAL_AVAILABLE, reason="textual library is not installed")
@pytest.mark.asyncio
async def test_textual_submissions_persist_only_in_active_topic(tmp_path) -> None:
    base = tmp_path / "cli_history"
    tui = TextualTUI(history_file=str(base))
    submitted: list[str] = []

    async def on_submit(text: str) -> None:
        submitted.append(text)

    tui.set_on_submit(on_submit)
    tui.set_commands([])
    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.set_input_history_topic("cli:topic-a")
        await pilot.press("a", "enter")
        await pilot.pause()

        tui.set_input_history_topic("cli:topic-b")
        await pilot.press("b", "enter")
        await pilot.pause()

    assert submitted == ["a", "b"]
    assert TextualTUI(history_file=str(base))._history == []

    topic_a = TextualTUI(history_file=str(base))
    topic_a.set_input_history_topic("cli:topic-a")
    assert topic_a._history == ["a"]

    topic_b = TextualTUI(history_file=str(base))
    topic_b.set_input_history_topic("cli:topic-b")
    assert topic_b._history == ["b"]


def test_prompt_history_switches_storage_per_topic(tmp_path) -> None:
    base = tmp_path / "cli_history"
    topic_a = "cli:topic-a"
    topic_b = "cli:topic-b"
    topic_a_path = input_history_path(base, topic_a)
    topic_b_path = input_history_path(base, topic_b)
    _write_history(topic_a_path, "a-only")
    _write_history(topic_b_path, "b-only")

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            tui = PromptTUI(history_file=str(base))
            tui.set_input_history_topic(topic_a)
            assert list(tui._input_buffer.history.load_history_strings()) == ["a-only"]

            tui.set_input_history_topic(topic_b)
            assert list(tui._input_buffer.history.load_history_strings()) == ["b-only"]

            tui.set_input_history_topic("cli:new-topic")
            assert list(tui._input_buffer.history.load_history_strings()) == []
