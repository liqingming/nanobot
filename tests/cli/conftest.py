"""Shared fixtures for nanobot CLI/TUI tests."""
from __future__ import annotations

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from nanobot.cli.tui import PromptTUI


@pytest.fixture
def prompt_tui():
    """A PromptTUI with just the input buffer initialized, no Application.

    prompt_toolkit's Buffer touches the active AppSession on construction
    and on Windows tries to create a Win32Output (which raises outside a
    real console). Wrapping in ``create_app_session`` with a pipe input +
    ``DummyOutput`` avoids that; the buffer + popup wiring is the bit
    tests actually exercise.
    """
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            t = PromptTUI()
            t._input_buffer = Buffer(name="nanobot_input", multiline=False)
            t._input_buffer.on_text_changed += lambda _: t._update_popup()
            t.set_commands([("/new", "新建话题"), ("/resume", "切换话题")])
            yield t
