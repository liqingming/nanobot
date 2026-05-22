"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from importlib.resources import files as pkg_files
from pathlib import Path
import datetime as datetime_module

from nanobot.agent.context import ContextBuilder


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_bootstrap_files_are_backed_by_templates() -> None:
    template_dir = pkg_files("nanobot") / "templates"

    for filename in ContextBuilder.BOOTSTRAP_FILES:
        assert (template_dir / filename).is_file(), f"missing bootstrap template: {filename}"


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be merged with the user message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    assert messages[0]["role"] == "system"
    assert "## Current Session" not in messages[0]["content"]

    # Runtime context is now merged with user message into a single message
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, str)
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in user_content
    assert "Current Time:" in user_content
    assert "Channel: cli" in user_content
    assert "Chat ID: direct" in user_content
    assert "Return exactly: OK" in user_content


def test_pending_summary_injected_as_system_reminder(tmp_path) -> None:
    """When a pending consolidation summary is buffered, build_messages
    injects it as a <system-reminder> inside the user message — without
    touching system prompt (so the prompt cache stays warm)."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    msgs_with = builder.build_messages(
        history=[],
        current_message="hi",
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        pending_summary="### Progress\n- Step 1 done\n- Step 2 in progress",
    )
    msgs_without = builder.build_messages(
        history=[],
        current_message="hi",
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        pending_summary=None,
    )

    # System prompts are byte-identical (the summary did NOT enter it).
    assert msgs_with[0]["content"] == msgs_without[0]["content"]

    # The summary ended up wrapped in <system-reminder> inside the user msg.
    user_content_with = msgs_with[-1]["content"]
    assert "<system-reminder>" in user_content_with
    assert "Step 1 done" in user_content_with
    assert "Step 2 in progress" in user_content_with


def test_empty_pending_summary_does_not_inject_reminder(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    msgs = builder.build_messages(
        history=[],
        current_message="hi",
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        pending_summary="   ",  # whitespace only
    )
    user_content = msgs[-1]["content"]
    assert "<system-reminder>" not in str(user_content)
