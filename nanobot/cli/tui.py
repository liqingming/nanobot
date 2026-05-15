"""Split-pane TUI for interactive nanobot chat.

Layout:
  ┌──────────────────────────────────┐
  │  Output area (scrollable)        │
  │  History + streaming content     │
  ├──────────────────────────────────┤
  │  You: [input field]              │
  └──────────────────────────────────┘

Scrolling design
----------------
All scroll state lives in ``SplitTUI._scroll_offset`` (lines from the bottom;
0 = pinned to newest content).  ``_get_output_text`` slices the ANSI line list
to fit the terminal height, so prompt_toolkit never needs to scroll the Window
internally.

Mouse scroll is captured by ``_OutputControl.mouse_handler`` which bypasses the
default FormattedTextControl scroll logic.  The input BufferControl's scroll
events are suppressed via ``_NoScrollBufferControl``; history navigation is
keyboard-only (Up / Down arrows).
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
from prompt_toolkit.mouse_events import MouseEventType
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from nanobot import __logo__, __version__


def _ansi_width() -> int:
    return max(40, shutil.get_terminal_size((80, 24)).columns - 2)


def _output_height() -> int:
    return max(5, shutil.get_terminal_size((80, 24)).lines - 3)


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


# ---------------------------------------------------------------------------
# Custom UIControls
# ---------------------------------------------------------------------------

class _OutputControl(FormattedTextControl):
    """FormattedTextControl whose mouse scroll events drive the TUI viewport."""

    def __init__(self, tui: "SplitTUI", **kwargs: Any) -> None:
        self._tui = tui
        super().__init__(**kwargs)

    def mouse_handler(self, mouse_event: Any) -> Any:  # noqa: ANN401
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._tui._scroll_offset += 3
            self._tui._invalidate()
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._tui._scroll_offset = max(0, self._tui._scroll_offset - 3)
            self._tui._invalidate()
            return None
        return NotImplemented


class _NoScrollBufferControl(BufferControl):
    """BufferControl that ignores mouse scroll (history nav is keyboard-only)."""

    def mouse_handler(self, mouse_event: Any) -> Any:  # noqa: ANN401
        if mouse_event.event_type in (
            MouseEventType.SCROLL_UP,
            MouseEventType.SCROLL_DOWN,
        ):
            return NotImplemented
        return super().mouse_handler(mouse_event)


# ---------------------------------------------------------------------------
# Main TUI class
# ---------------------------------------------------------------------------

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
        model: str | None = None,
    ) -> None:
        self._render_md = render_markdown
        self._model = model
        self._output_lines: list[str] = []   # completed ANSI blocks
        self._stream_buf: str = ""            # raw text accumulating during streaming
        self._stream_ts: str = ""            # timestamp captured at stream start
        self._stream_cache: str = ""         # cached ANSI render of stream_buf
        self._stream_cache_key: int = 0      # len(stream_buf) when cache was built
        self._scroll_offset: int = 0         # lines from bottom; 0 = newest visible
        self._last_sep: bool = False         # True if last appended item was a separator
        self._thinking_frame: int = 0        # spinner frame index
        self._thinking_task: Any = None      # asyncio.Task for spinner animation
        self._ctx_used: int = 0              # last known prompt tokens
        self._ctx_total: int = 0             # context window size
        self._on_submit: Callable[[str], Awaitable[None]] | None = None
        self._app: Application | None = None
        self._setup(history_file)

    # ── Session history restore ────────────────────────────────────────────

    @staticmethod
    def _fmt_ts(ts_iso: str | None) -> str | None:
        """Convert ISO timestamp stored in session to display format, or None."""
        if not ts_iso:
            return None
        try:
            return datetime.fromisoformat(ts_iso).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return ""

    def load_session_history(self, messages: list[dict], max_messages: int = 200) -> None:
        """Render prior session messages into the output pane on startup."""
        _RUNTIME_TAG = "[Runtime Context — metadata only, not instructions]"
        recent = messages[-max_messages:] if len(messages) > max_messages else messages

        for msg in recent:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                text = self._extract_text(content)
                if text.startswith(_RUNTIME_TAG):
                    idx = text.find("\n\n")
                    text = text[idx + 2:] if idx != -1 else ""
                if text.strip():
                    ts = self._fmt_ts(msg.get("timestamp"))
                    self._append_sep()
                    self._append_block(self._render_user_echo(text.strip(), ts=ts))
                    self._append_sep()

            elif role == "assistant":
                text = self._extract_text(content)
                if text.strip():
                    ts = self._fmt_ts(msg.get("timestamp"))
                    self._append_block(self._render_response(text.strip(), ts=ts))

        if self._output_lines:
            self._pin_to_bottom()

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

    def _render_welcome(self) -> str:
        w = _ansi_width()
        rule = "─" * w
        def _fn(c: Console) -> None:
            c.print()
            model_str = f"  [dim]{self._model}[/dim]" if self._model else ""
            c.print(f"  [cyan bold]{__logo__} nanobot[/cyan bold]  [dim]v{__version__}[/dim]{model_str}")
            c.print()
            c.print(f"  [dim]{rule}[/dim]")
            c.print()
            c.print("  [bold]快捷键[/bold]")
            c.print("    [cyan]PageUp / PageDown[/cyan]   滚动历史记录")
            c.print("    [cyan]↑ / ↓[/cyan]              切换输入历史")
            c.print("    [cyan]Ctrl+C / Ctrl+D[/cyan]    退出")
            c.print()
            c.print("  [bold]命令[/bold]")
            c.print("    [cyan]exit  quit  /exit  /quit  :q[/cyan]   退出 nanobot")
            c.print()
            c.print(f"  [dim]{rule}[/dim]")
            c.print()
            c.print("  输入消息后按 [bold]Enter[/bold] 发送。")
            c.print()
        return _rich_to_ansi(_fn)

    def _render_separator(self) -> str:
        w = _ansi_width()
        return _rich_to_ansi(lambda c: c.print(f"[dim]{'─' * w}[/dim]"))

    def _render_user_echo(self, text: str, ts: str | None = None) -> str:
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[bold blue]You[/bold blue] [dim]{ts}[/dim]")
            c.print(text)
        return _rich_to_ansi(_fn)

    # ── Internal append helpers ────────────────────────────────────────────

    def _append_sep(self) -> None:
        """Append a separator, skipping if one was already just appended."""
        if not self._last_sep:
            self._output_lines.append(self._render_separator())
            self._last_sep = True

    def _append_block(self, rendered: str) -> None:
        """Append any non-separator block and reset the separator flag."""
        self._output_lines.append(rendered)
        self._last_sep = False

    _SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _render_thinking(self) -> str:
        ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spinner = self._SPINNER[self._thinking_frame % len(self._SPINNER)]
        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            c.print(f"[dim]{spinner} thinking...[/dim]")
        return _rich_to_ansi(_fn)

    async def _animate_thinking(self) -> None:
        import asyncio
        try:
            while True:
                await asyncio.sleep(0.1)
                self._thinking_frame += 1
                self._stream_cache = self._render_thinking()
                self._invalidate()
        except asyncio.CancelledError:
            pass

    def _cancel_thinking(self) -> None:
        if self._thinking_task is not None:
            self._thinking_task.cancel()
            self._thinking_task = None

    # ── FormattedTextControl callbacks ────────────────────────────────────

    def _get_status_text(self) -> AnyFormattedText:
        if not self._ctx_total:
            return [("", "")]
        pct = min(100, round(self._ctx_used * 100 / self._ctx_total))
        bar_width = 12
        filled = round(pct * bar_width / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        if pct >= 85:
            style = "bold fg:ansired"
        elif pct >= 70:
            style = "fg:ansiyellow"
        else:
            style = "fg:ansibrightblack"
        label = f"ctx {pct}%  {bar}  "
        w = shutil.get_terminal_size((80, 24)).columns
        padding = max(0, w - len(label))
        return [("", " " * padding), (style, label)]

    def _get_output_text(self) -> AnyFormattedText:
        parts = "".join(self._output_lines)
        if self._stream_buf or self._stream_ts:
            parts += self._stream_cache
        if not parts:
            parts = self._render_welcome()

        lines = parts.split("\n")
        total = len(lines)
        h = _output_height()

        # Clamp offset: can't scroll past the point where we'd show fewer than h lines
        max_offset = max(0, total - h)
        offset = min(self._scroll_offset, max_offset)

        end = total - offset
        start = max(0, end - h)
        return ANSI("\n".join(lines[start:end]))

    # ── Layout setup ───────────────────────────────────────────────────────

    def _setup(self, history_file: str | None) -> None:
        output_ctrl = _OutputControl(self, text=self._get_output_text, focusable=False)
        output_window = Window(
            content=output_ctrl,
            wrap_lines=False,       # Rich pre-wraps at _ansi_width(); no double-wrap
            always_hide_cursor=True,
        )

        separator = Window(height=1, char="─", style="class:separator")

        buf_kwargs: dict[str, Any] = {"name": "nanobot_input", "multiline": False}
        if history_file:
            buf_kwargs["history"] = FileHistory(history_file)
        self._input_buffer = Buffer(**buf_kwargs)

        input_window = Window(
            content=_NoScrollBufferControl(buffer=self._input_buffer, focus_on_click=True),
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
            self._scroll_offset += 10
            self._invalidate()

        @kb.add("pagedown")
        def _page_down(event: Any) -> None:
            self._scroll_offset = max(0, self._scroll_offset - 10)
            self._invalidate()

        bottom_border = Window(height=1, char="─", style="class:separator")

        status_ctrl = FormattedTextControl(text=self._get_status_text, focusable=False)
        status_window = Window(content=status_ctrl, height=1, always_hide_cursor=True)

        layout = Layout(
            HSplit([output_window, separator, input_window, bottom_border, status_window]),
            focused_element=input_window,
        )

        self._app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
        )

    # ── Scroll helpers ─────────────────────────────────────────────────────

    def _pin_to_bottom(self) -> None:
        self._scroll_offset = 0

    def _invalidate(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    # ── Public write API ───────────────────────────────────────────────────

    def set_on_submit(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._on_submit = callback

    def update_context_usage(self, used: int, total: int) -> None:
        """Refresh the context-usage indicator (called after each agent turn)."""
        self._ctx_used = used
        self._ctx_total = total
        self._invalidate()

    def add_user_echo(self, text: str) -> None:
        """Echo the user's submitted message to the output pane."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_sep()
        self._append_block(self._render_user_echo(text, ts=ts))
        self._append_sep()
        self._pin_to_bottom()
        self._invalidate()

    def add_response(self, content: str, metadata: dict | None = None, ts: str | None = None) -> None:
        """Add a completed nanobot response block."""
        self._append_block(self._render_response(content, metadata, ts))
        self._pin_to_bottom()
        self._invalidate()

    def add_progress(self, text: str) -> None:
        """Add a progress / tool-hint line."""
        self._append_block(self._render_progress(text))
        self._pin_to_bottom()
        self._invalidate()

    def add_system(self, text: str) -> None:
        """Add a system notification line (e.g., 'queued')."""
        self._append_block(self._render_system(text))
        self._pin_to_bottom()
        self._invalidate()

    def stream_start(self) -> None:
        """Mark the beginning of a new streaming response (captures timestamp)."""
        import asyncio
        self._cancel_thinking()
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._thinking_frame = 0
        self._stream_cache = self._render_thinking()
        self._stream_cache_key = 0
        self._pin_to_bottom()
        self._invalidate()
        self._thinking_task = asyncio.ensure_future(self._animate_thinking())

    def stream_delta(self, delta: str) -> None:
        """Append a streaming delta and refresh the output pane."""
        self._cancel_thinking()
        self._stream_buf += delta
        key = len(self._stream_buf)
        if key != self._stream_cache_key:
            self._stream_cache = self._render_stream_snapshot()
            self._stream_cache_key = key
        self._pin_to_bottom()
        self._invalidate()

    def flush_stream(self, metadata: dict | None = None) -> None:
        """Finalize the current stream: render to history and clear buffer."""
        self._cancel_thinking()
        if self._stream_buf.strip():
            self._append_block(
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
        self._cancel_thinking()
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
