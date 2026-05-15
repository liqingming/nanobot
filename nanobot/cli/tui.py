"""Split-pane TUI for interactive nanobot chat.

Layout:
  ┌──────────────────────────────────┐
  │  Output area (scrollable)        │
  │  History + streaming content     │
  ├──────────────────────────────────┤
  │  You: [input field]              │
  └──────────────────────────────────┘

prompt_toolkit Application owns the terminal. Streaming deltas are rendered
inline; the input line is always visible and accepts keystrokes at any time.
"""
from __future__ import annotations

import shutil
from collections.abc import Awaitable, Callable
from datetime import datetime
from io import StringIO
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI, HTML, AnyFormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import D
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from nanobot import __logo__


def _ansi_width() -> int:
    return max(40, shutil.get_terminal_size((80, 24)).columns - 2)


def _rich_to_ansi(render_fn: Callable[[Console], Any]) -> str:
    """Render Rich content to an ANSI escape string."""
    buf = StringIO()
    c = Console(
        file=buf,
        width=_ansi_width(),
        force_terminal=True,
        highlight=False,
        color_system="256",
    )
    render_fn(c)
    return buf.getvalue()


class SplitTUI:
    """Full-screen split TUI: scrollable output pane + always-visible input line.

    Usage::

        tui = SplitTUI(render_markdown=True, history_file="~/.nanobot/history")
        tui.set_on_submit(async_callback)
        await tui.run_async()
    """

    def __init__(
        self,
        render_markdown: bool = True,
        history_file: str | None = None,
    ) -> None:
        self._render_md = render_markdown
        self._output_lines: list[str] = []   # completed ANSI blocks
        self._stream_buf: str = ""            # raw text accumulating during streaming
        self._stream_ts: str = ""            # timestamp captured at stream start
        self._stream_cache: str = ""         # cached ANSI render of stream_buf
        self._stream_cache_key: int = 0      # len(stream_buf) when cache was built
        self._on_submit: Callable[[str], Awaitable[None]] | None = None
        self._output_window: Window | None = None
        self._app: Application | None = None
        self._setup(history_file)

    # ── Rendering helpers ──────────────────────────────────────────────────

    def _render_response(self, content: str, metadata: dict | None = None, ts: str | None = None) -> str:
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        render_as_text = (metadata or {}).get("render_as") == "text"

        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            if self._render_md and not render_as_text and content.strip():
                c.print(Markdown(content))
            else:
                c.print(Text(content))
            c.print()

        return _rich_to_ansi(_fn)

    def _render_stream_snapshot(self) -> str:
        """Re-render stream_buf to ANSI (called only when buffer changed)."""
        ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            if self._render_md and self._stream_buf.strip():
                c.print(Markdown(self._stream_buf))
            else:
                c.print(self._stream_buf or "")

        return _rich_to_ansi(_fn)

    def _render_progress(self, text: str) -> str:
        return _rich_to_ansi(lambda c: c.print(f"  [dim]↳ {text}[/dim]"))

    def _render_system(self, text: str) -> str:
        return _rich_to_ansi(lambda c: c.print(f"[dim]{text}[/dim]"))

    def _render_user_echo(self, text: str) -> str:
        return _rich_to_ansi(lambda c: (c.print(), c.print(f"[bold blue]You:[/bold blue] {text}")))

    def _render_thinking(self) -> str:
        ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            c.print("[dim]thinking...[/dim]")
        return _rich_to_ansi(_fn)

    # ── FormattedTextControl callback ──────────────────────────────────────

    def _get_output_text(self) -> AnyFormattedText:
        parts = "".join(self._output_lines)
        if self._stream_buf:
            parts += self._stream_cache      # real streaming content
        elif self._stream_ts:
            parts += self._stream_cache      # thinking... placeholder
        return ANSI(parts) if parts else HTML("")

    # ── Layout setup ───────────────────────────────────────────────────────

    def _setup(self, history_file: str | None) -> None:
        output_ctrl = FormattedTextControl(text=self._get_output_text, focusable=False)
        self._output_window = Window(
            content=output_ctrl,
            wrap_lines=True,
            always_hide_cursor=True,
        )

        separator = Window(height=1, char="─", style="class:separator")

        buf_kwargs: dict[str, Any] = {"name": "nanobot_input", "multiline": False}
        if history_file:
            buf_kwargs["history"] = FileHistory(history_file)
        self._input_buffer = Buffer(**buf_kwargs)

        input_window = Window(
            content=BufferControl(buffer=self._input_buffer, focus_on_click=True),
            height=D(min=1, max=4),
            wrap_lines=True,
            get_line_prefix=lambda lineno, wrap_count: (
                HTML("<b fg='ansiblue'>You: </b>") if lineno == 0 and wrap_count == 0 else HTML("      ")
            ),
        )

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            text = self._input_buffer.text
            self._input_buffer.reset()
            if text.strip() and self._on_submit:
                import asyncio
                asyncio.ensure_future(self._on_submit(text))

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def _ctrl_d(event: Any) -> None:
            if not self._input_buffer.text:
                event.app.exit(exception=EOFError())

        @kb.add("up")
        def _up(event: Any) -> None:
            buf = self._input_buffer
            if not buf.text or buf.cursor_position == 0:
                buf.history_backward()

        @kb.add("down")
        def _down(event: Any) -> None:
            self._input_buffer.history_forward()

        @kb.add("pageup")
        def _page_up(event: Any) -> None:
            if self._output_window is not None:
                self._output_window.vertical_scroll = max(
                    0, self._output_window.vertical_scroll - 10
                )

        @kb.add("pagedown")
        def _page_down(event: Any) -> None:
            if self._output_window is not None:
                self._output_window.vertical_scroll += 10

        layout = Layout(
            HSplit([self._output_window, separator, input_window]),
            focused_element=input_window,
        )

        self._app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
        )

    # ── Scroll helper ──────────────────────────────────────────────────────

    def _pin_to_bottom(self) -> None:
        if self._output_window is not None:
            self._output_window.vertical_scroll = 999999

    def _invalidate(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    # ── Public write API ───────────────────────────────────────────────────

    def set_on_submit(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._on_submit = callback

    def add_user_echo(self, text: str) -> None:
        """Echo the user's submitted message to the output pane."""
        self._output_lines.append(self._render_user_echo(text))
        self._pin_to_bottom()
        self._invalidate()

    def add_response(self, content: str, metadata: dict | None = None, ts: str | None = None) -> None:
        """Add a completed nanobot response block."""
        self._output_lines.append(self._render_response(content, metadata, ts))
        self._pin_to_bottom()
        self._invalidate()

    def add_progress(self, text: str) -> None:
        """Add a progress / tool-hint line."""
        self._output_lines.append(self._render_progress(text))
        self._pin_to_bottom()
        self._invalidate()

    def add_system(self, text: str) -> None:
        """Add a system notification line (e.g., 'queued')."""
        self._output_lines.append(self._render_system(text))
        self._pin_to_bottom()
        self._invalidate()

    def stream_start(self) -> None:
        """Mark the beginning of a new streaming response (captures timestamp)."""
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._stream_cache = self._render_thinking()  # shown until first delta arrives
        self._stream_cache_key = 0
        self._pin_to_bottom()
        self._invalidate()

    def stream_delta(self, delta: str) -> None:
        """Append a streaming delta and refresh the output pane."""
        self._stream_buf += delta
        key = len(self._stream_buf)
        if key != self._stream_cache_key:
            self._stream_cache = self._render_stream_snapshot()
            self._stream_cache_key = key
        self._pin_to_bottom()
        self._invalidate()

    def flush_stream(self, metadata: dict | None = None) -> None:
        """Finalize the current stream: render to history and clear buffer."""
        if self._stream_buf.strip():
            self._output_lines.append(
                self._render_response(self._stream_buf, metadata, ts=self._stream_ts)
            )
        self._stream_buf = ""
        self._stream_cache = ""
        self._stream_cache_key = 0
        self._stream_ts = ""
        self._pin_to_bottom()
        self._invalidate()

    def pop_stream(self) -> str:
        """Return accumulated stream text and clear without adding to history."""
        buf = self._stream_buf
        self._stream_buf = ""
        self._stream_cache = ""
        self._stream_cache_key = 0
        return buf

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def run_async(self) -> None:
        if self._app is not None:
            await self._app.run_async()

    def exit(self) -> None:
        if self._app is not None:
            self._app.exit()
