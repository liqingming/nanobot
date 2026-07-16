"""Abstract base class for nanobot TUI backends.

``commands.py`` talks exclusively to this interface; the concrete
implementation (prompt_toolkit or Textual) is selected at runtime by
``tui_factory.create_tui()``.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


def input_history_path(history_file: str | Path | None, topic_key: str) -> Path | None:
    """Return a stable per-session history path without exposing the session key."""
    if not history_file or not topic_key:
        return None
    base = Path(history_file)
    digest = hashlib.sha256(topic_key.encode("utf-8")).hexdigest()[:16]
    return base.parent / "topics" / f"{digest}.history"


class TUIBase(ABC):
    """Interface contract every TUI backend must satisfy."""

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @abstractmethod
    async def run_async(self) -> None: ...

    @abstractmethod
    def exit(self) -> None: ...

    # ── Callback registration ──────────────────────────────────────────────

    @abstractmethod
    def set_on_submit(self, callback: Callable[[str], Awaitable[None]]) -> None: ...

    @abstractmethod
    def set_on_pre_submit(self, callback: Callable[[str], None]) -> None: ...

    @abstractmethod
    def set_on_cancel(self, callback: Callable[[], None]) -> None: ...

    # ── Content write API ──────────────────────────────────────────────────

    @abstractmethod
    def load_session_history(
        self,
        messages: list[dict],
        max_messages: int = 200,
        tool_registry: Any = None,
        workspace: Any = None,
    ) -> None: ...

    @abstractmethod
    def add_user_echo(self, text: str) -> None: ...

    @abstractmethod
    def add_response(
        self,
        content: str,
        metadata: dict | None = None,
        ts: str | None = None,
    ) -> None: ...

    @abstractmethod
    def add_progress(self, text: str) -> None: ...

    @abstractmethod
    def add_system(self, text: str) -> None: ...

    # ── Streaming ──────────────────────────────────────────────────────────

    @abstractmethod
    def stream_start(self) -> None: ...

    @abstractmethod
    def tool_phase_start(self) -> None: ...

    @abstractmethod
    def stream_delta(self, delta: str) -> None: ...

    @abstractmethod
    def flush_stream(self, metadata: dict | None = None) -> None: ...

    @abstractmethod
    def pop_stream(self) -> str: ...

    def flush_accumulator(self) -> str:
        """Return and clear intermediate LLM text flushed between tool calls.

        Default returns empty string; TextualTUI overrides with real accumulation.
        """
        return ""

    def set_todos(self, todos: list[dict]) -> None:
        """Update the active todo display (in_progress item + progress count).

        Default is a no-op so backends that don't render todos can ignore it.
        TextualTUI overrides to update the #todo-bar widget.
        """
        return None

    def add_tool_result(self, summary: str) -> None:
        """Append a short result summary to the most-recent tool placeholder
        (e.g. turning "⠋ exec(cmd)" into "→ exec(cmd) → exit 0, 12 lines").

        Default is a no-op so backends without per-tool tracing can ignore it.
        """
        return None

    def add_file_edit_events(self, events: list[dict[str, Any]]) -> None:
        """Render structured file edit progress events."""
        return None

    def stop_thinking(self) -> None:
        """Stop any in-flight thinking / idle spinner. Called on turn completion
        so the spinner never outlives the turn — the streaming path stops it via
        pop_stream, but the non-streaming reply path has no such hook. Default
        no-op for backends without a spinner.
        """
        return None

    # ── State updates ──────────────────────────────────────────────────────

    @abstractmethod
    def set_topic(self, name: str) -> None: ...

    @abstractmethod
    def set_input_history_topic(self, topic_key: str) -> None: ...

    @abstractmethod
    def set_is_processing(self, value: bool) -> None: ...

    @abstractmethod
    def update_context_usage(self, used: int, total: int) -> None: ...

    @abstractmethod
    def reset_history(self) -> None: ...

    # ── Interactive modes ──────────────────────────────────────────────────

    @abstractmethod
    def enter_new_topic_mode(
        self, callback: Callable[[str], Awaitable[None]]
    ) -> None: ...

    @abstractmethod
    def set_commands(self, commands: list[tuple[str, str]]) -> None: ...

    @abstractmethod
    def show_topic_popup(
        self,
        topics: list[str | tuple[str, str]],
        on_select: Callable[[str], Awaitable[None]],
    ) -> None: ...

    def show_question_popup(
        self,
        questions: list[dict],
        on_complete: Callable[[dict[str, str] | None], Awaitable[None]],
    ) -> None:
        """Ask the user a series of multiple-choice questions and report all
        answers (or ``None`` on cancellation) via ``on_complete``.

        Default falls back to immediate cancellation so backends that don't
        implement interactive popups still satisfy the contract.
        """
        import asyncio as _asyncio
        _asyncio.ensure_future(on_complete(None))

    @abstractmethod
    def hide_popup(self) -> None: ...
