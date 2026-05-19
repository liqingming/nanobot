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

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from datetime import datetime
from io import StringIO
from typing import Any

from wcwidth import wcswidth as _wcswidth

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, HTML, AnyFormattedText
from prompt_toolkit.history import FileHistory as _FileHistory
from prompt_toolkit.key_binding import KeyBindings

_HISTORY_SKIP = {"exit", "quit", "/exit", "/quit", ":q"}


class _FilteredFileHistory(_FileHistory):
    """FileHistory that silently drops exit-style commands."""

    def store_string(self, string: str) -> None:
        if string.strip().lower() not in _HISTORY_SKIP:
            super().store_string(string)
from prompt_toolkit.layout import Layout
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.mouse_events import MouseEventType
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from nanobot import __logo__, __version__


def _ansi_width() -> int:
    return max(40, shutil.get_terminal_size((80, 24)).columns - 3)


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

    def create_content(self, width: int, height: int | None) -> Any:
        if height is not None and height > 0:
            self._tui._actual_output_height = height
        return super().create_content(width, height)

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


class _ScrollbarControl(UIControl):
    """1-column vertical scrollbar for the output pane."""

    def __init__(self, tui: "SplitTUI") -> None:
        self._tui = tui

    def create_content(self, width: int, height: int) -> UIContent:
        total = self._tui._total_output_lines
        viewport = self._tui._actual_output_height or max(1, height)
        offset = self._tui._scroll_offset
        max_offset = max(0, total - viewport)

        if max_offset <= 0 or height <= 0:
            # All content visible — blank scrollbar
            def get_line_blank(i: int) -> list[tuple[str, str]]:
                return [("fg:ansibrightblack", " ")]
            return UIContent(get_line=get_line_blank, line_count=height, show_cursor=False)

        # Thumb size proportional to viewport / total, minimum 1 row
        thumb_size = max(1, round(height * viewport / total))
        thumb_size = min(thumb_size, height)

        # scroll_from_top: 0 = pinned to top, max_offset = pinned to bottom
        scroll_from_top = max_offset - offset
        track = height - thumb_size
        thumb_top = round(scroll_from_top / max_offset * track) if max_offset > 0 else 0
        thumb_top = max(0, min(thumb_top, height - thumb_size))

        def get_line(i: int) -> list[tuple[str, str]]:
            if thumb_top <= i < thumb_top + thumb_size:
                return [("fg:ansiwhite", "█")]
            return [("fg:ansibrightblack", "│")]

        return UIContent(get_line=get_line, line_count=height, show_cursor=False)

    def is_focusable(self) -> bool:
        return False


class _RightAlignedLabelControl(UIControl):
    """UIControl that right-aligns a text label using the actual render width
    passed by prompt_toolkit — no terminal-size guessing required."""

    def __init__(self, get_text: Callable[[], str]) -> None:
        self._get_text = get_text

    def create_content(self, width: int, height: int) -> UIContent:
        label = self._get_text()
        # wcswidth gives display columns (CJK chars = 2), len() gives char count
        display_w = max(0, _wcswidth(label))
        # Windows console: last column is not writable, use width-1 as effective
        effective = max(0, width - 1)
        padding = max(0, effective - display_w)

        def get_line(i: int) -> list[tuple[str, str]]:
            return [("", " " * padding), ("bold fg:ansicyan", label)]

        return UIContent(get_line=get_line, line_count=1, show_cursor=False)

    def is_focusable(self) -> bool:
        return False


class _CommandLexer(Lexer):
    """Colors the input text indigo when it exactly matches a known command."""

    def __init__(self, tui: "SplitTUI") -> None:
        self._tui = tui

    def lex_document(self, document: Document):  # type: ignore[override]
        text = document.text
        cmds = {cmd for cmd, _ in self._tui._all_commands}

        def get_line(lineno: int) -> list[tuple[str, str]]:
            if lineno != 0:
                return [("", "")]
            if text in cmds:
                return [("fg:#5c6bc0 bold", text)]
            for cmd in cmds:
                if text.startswith(cmd + " "):
                    return [("fg:#5c6bc0 bold", cmd), ("fg:#7986cb", text[len(cmd):])]
            return [("", text)]

        return get_line


class _PopupMenuControl(UIControl):
    """Floating popup for command completion and topic selection."""

    MAX_VISIBLE = 6

    def __init__(self, tui: "SplitTUI") -> None:
        self._tui = tui

    def create_content(self, width: int, height: int) -> UIContent:
        all_items = self._tui._popup_items
        idx = self._tui._popup_idx
        mode = self._tui._popup_mode
        n = len(all_items)

        # Scroll popup window to keep selected item visible
        start = max(0, idx - self.MAX_VISIBLE + 1)
        if start + self.MAX_VISIBLE > n:
            start = max(0, n - self.MAX_VISIBLE)
        end = min(n, start + self.MAX_VISIBLE)
        visible = all_items[start:end]
        has_above = start > 0
        has_below = end < n

        # Build line descriptors with optional scroll hints
        line_descs: list[tuple[str, Any]] = []
        if has_above:
            line_descs.append(("up", start))
        for i, (value, label) in enumerate(visible):
            line_descs.append(("item", (start + i, value, label)))
        if has_below:
            line_descs.append(("down", n - end))

        def get_line(i: int) -> list[tuple[str, str]]:
            if i >= len(line_descs):
                return [("", "")]
            kind, data = line_descs[i]
            if kind == "up":
                hint = f"  ↑ 还有 {data} 项"
                return [("fg:ansibrightblack", hint + " " * max(0, width - _wcswidth(hint)))]
            if kind == "down":
                hint = f"  ↓ 还有 {data} 项"
                return [("fg:ansibrightblack", hint + " " * max(0, width - _wcswidth(hint)))]
            actual_idx, value, label = data
            selected = actual_idx == idx
            prefix = " ▶ " if selected else "   "
            sel_style = "bg:#1e3a5f bold fg:ansiwhite"
            nrm_style = "fg:ansibrightblack"
            style = sel_style if selected else nrm_style
            if mode == "command":
                dw = max(0, _wcswidth(value))
                pad = max(0, 12 - dw)
                body = f"{prefix}{value}{' ' * pad}  {label}"
            else:
                body = f"{prefix}{value}"
            tail = " " * max(0, width - 1 - _wcswidth(body))
            return [(style, body + tail)]

        return UIContent(get_line=get_line, line_count=len(line_descs), show_cursor=False)

    def is_focusable(self) -> bool:
        return False


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
        self._actual_output_height: int = 0  # captured by _OutputControl.create_content()
        self._total_output_lines: int = 0    # total logical lines, for scrollbar
        self._last_sep: bool = False         # True if last appended item was a separator
        self._thinking_frame: int = 0        # spinner frame index
        self._thinking_task: Any = None      # asyncio.Task for spinner animation
        self._tool_frame: int = 0            # spinner frame index for tool execution
        self._tool_task: Any = None          # asyncio.Task for tool animation
        self._tool_hint: str = ""            # current tool hint label
        self._ctx_used: int = 0              # last known prompt tokens
        self._ctx_total: int = 0             # context window size
        self._is_processing: bool = False    # True while agent is running
        self._topic: str = ""               # current topic name (chat_id)
        self._live_progress: str = ""        # tool-hint lines shown in live area
        self._on_submit: Callable[[str], Awaitable[None]] | None = None
        self._on_pre_submit: Callable[[str], None] | None = None
        self._on_cancel: Callable[[], None] | None = None
        self._app: Application | None = None
        # New-topic name input mode
        self._input_mode: str = "chat"       # "chat" | "new_topic"
        self._new_topic_cb: Callable[[str], Awaitable[None]] | None = None
        # Command / topic popup state
        self._all_commands: list[tuple[str, str]] = []
        self._popup_mode: str = "hidden"    # "hidden" | "command" | "topic"
        self._popup_items: list[tuple[str, str]] = []   # (value, label) visible items
        self._popup_idx: int = 0
        self._popup_on_select: Callable[[str], Awaitable[None]] | None = None
        self._popup_all_topics: list[str] = []
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
            c.print("    [cyan]ESC[/cyan]                取消当前请求")
            c.print("    [cyan]Ctrl+C / Ctrl+D[/cyan]    退出")
            c.print()
            c.print("  [bold]命令[/bold]")
            c.print("    [cyan]exit  quit  /exit  /quit  :q[/cyan]   退出 nanobot")
            c.print("    [cyan]/new [名称][/cyan]                   新建话题")
            c.print("    [cyan]/resume [名称][/cyan]                切换/恢复话题（无参数时交互选择）")
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
    _TOOL_SPINNER = ["◐", "◓", "◑", "◒"]

    def _render_thinking(self) -> str:
        ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spinner = self._SPINNER[self._thinking_frame % len(self._SPINNER)]
        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            c.print(f"[dim]{spinner} thinking...[/dim]")
        return _rich_to_ansi(_fn)

    def _render_tool_executing(self) -> str:
        ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = self._tool_hint or "executing..."
        spinner = self._TOOL_SPINNER[self._tool_frame % len(self._TOOL_SPINNER)]
        def _fn(c: Console) -> None:
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            c.print(f"[dim]{spinner} {label}[/dim]")
        return _rich_to_ansi(_fn)

    async def _animate_thinking(self) -> None:
        import asyncio
        try:
            while True:
                self._thinking_frame += 1
                self._stream_cache = self._render_thinking()
                self._invalidate()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _cancel_thinking(self) -> None:
        if self._thinking_task is not None:
            self._thinking_task.cancel()
            self._thinking_task = None

    def _cancel_tool_task(self) -> None:
        if self._tool_task is not None:
            self._tool_task.cancel()
            self._tool_task = None

    async def _animate_tool_executing(self) -> None:
        import asyncio
        try:
            while True:
                self._tool_frame += 1
                self._stream_cache = self._render_tool_executing()
                self._invalidate()
                await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            pass

    # ── FormattedTextControl callbacks ────────────────────────────────────


    def _get_status_text(self) -> AnyFormattedText:
        w = shutil.get_terminal_size((80, 24)).columns

        if self._is_processing:
            esc_text = " ESC 打断 "
            esc_part: list[tuple[str, str]] = [("fg:ansibrightblack", esc_text)]
            esc_w = max(0, _wcswidth(esc_text))
        else:
            esc_text = ""
            esc_part = []
            esc_w = 0

        if self._ctx_total:
            pct = min(100, round(self._ctx_used * 100 / self._ctx_total))
            bar_width = 12
            filled = round(pct * bar_width / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            ctx_style = (
                "bold fg:ansired" if pct >= 85
                else "fg:ansiyellow" if pct >= 70
                else "fg:ansibrightblack"
            )
            ctx_text = f"ctx {pct}%  {bar}  "
            ctx_part: list[tuple[str, str]] = [(ctx_style, ctx_text)]
        else:
            ctx_text = ""
            ctx_part = []

        ctx_w = max(0, _wcswidth(ctx_text))
        padding = max(0, w - esc_w - ctx_w)
        return esc_part + [("", " " * padding)] + ctx_part

    def _get_output_text(self) -> AnyFormattedText:
        parts = "".join(self._output_lines)
        if self._stream_buf or self._stream_ts:
            parts += self._stream_cache
            if self._live_progress:
                parts += self._live_progress
        if not parts:
            parts = self._render_welcome()

        lines = parts.split("\n")
        total = len(lines)
        self._total_output_lines = total
        h = self._actual_output_height or _output_height()

        # Clamp offset: can't scroll past the point where we'd show fewer than h lines
        max_offset = max(0, total - h)
        offset = min(self._scroll_offset, max_offset)

        end = total - offset
        start = max(0, end - h)
        return ANSI("\n".join(lines[start:end]))

    # ── Layout setup ───────────────────────────────────────────────────────

    def _setup(self, history_file: str | None) -> None:
        output_ctrl = _OutputControl(self, text=self._get_output_text, focusable=False)
        scrollbar_ctrl = _ScrollbarControl(self)
        output_window = VSplit([
            Window(
                content=output_ctrl,
                wrap_lines=False,       # Rich pre-wraps at _ansi_width(); no double-wrap
                always_hide_cursor=True,
            ),
            Window(
                content=scrollbar_ctrl,
                width=1,
                always_hide_cursor=True,
            ),
        ])

        topic_line = ConditionalContainer(
            Window(
                content=_RightAlignedLabelControl(lambda: f" {self._topic} "),
                height=1,
                always_hide_cursor=True,
            ),
            filter=Condition(lambda: bool(self._topic)),
        )
        separator = Window(height=1, char="─", style="class:separator")

        buf_kwargs: dict[str, Any] = {"name": "nanobot_input", "multiline": False}
        if history_file:
            buf_kwargs["history"] = _FilteredFileHistory(history_file)
        self._input_buffer = Buffer(**buf_kwargs)
        self._input_buffer.on_text_changed += lambda _: self._update_popup()

        input_window = Window(
            content=_NoScrollBufferControl(
                buffer=self._input_buffer,
                focus_on_click=True,
                lexer=_CommandLexer(self),
            ),
            height=D(min=1, max=4),
            wrap_lines=True,
            get_line_prefix=lambda lineno, wrap_count: (
                (
                    HTML("<b fg='ansiyellow'>话题名: </b>")
                    if self._input_mode == "new_topic"
                    else HTML("<b fg='ansiblue'>You: </b>")
                )
                if lineno == 0 and wrap_count == 0
                else HTML("        " if self._input_mode == "new_topic" else "      ")
            ),
        )

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            if self._input_mode == "new_topic":
                name = self._input_buffer.text.strip()
                cb = self._new_topic_cb
                self._exit_new_topic_mode()
                if cb:
                    asyncio.ensure_future(cb(name))
                return
            if self._popup_mode == "topic" and self._popup_items:
                value, _ = self._popup_items[self._popup_idx]
                cb = self._popup_on_select
                self.hide_popup()
                self._input_buffer.reset()          # topic selected, not typed — skip history
                if cb:
                    asyncio.ensure_future(cb(value))
                return
            if self._popup_mode == "command" and self._popup_items:
                value, _ = self._popup_items[self._popup_idx]
                self.hide_popup()
                # Save the full command (not the partial typed text) to history
                self._input_buffer.set_document(Document(value, len(value)))
                self._input_buffer.reset(append_to_history=False)
                if value.strip() and self._on_submit:
                    asyncio.ensure_future(self._on_submit(value))
                return
            text = self._input_buffer.text
            if text.strip() and self._on_submit:
                if self._on_pre_submit:
                    self._on_pre_submit(text)
                self._input_buffer.reset(append_to_history=not text.strip().startswith("/"))
                asyncio.ensure_future(self._on_submit(text))
            else:
                self._input_buffer.reset(append_to_history=False)

        @kb.add("tab")
        def _tab(event: Any) -> None:
            if self._popup_mode == "command" and self._popup_items:
                value, _ = self._popup_items[self._popup_idx]
                self._input_buffer.set_document(Document(value, len(value)))

        @kb.add("escape")
        def _escape(event: Any) -> None:
            if self._input_mode == "new_topic":
                self._exit_new_topic_mode()
                return
            if self._popup_items:
                self.hide_popup()
            elif self._on_cancel is not None:
                self._on_cancel()

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def _ctrl_d(event: Any) -> None:
            if not self._input_buffer.text:
                event.app.exit(exception=EOFError())

        @kb.add("up")
        def _up(event: Any) -> None:
            if self._popup_items:
                self._popup_idx = max(0, self._popup_idx - 1)
                self._invalidate()
                return
            self._input_buffer.history_backward()

        @kb.add("down")
        def _down(event: Any) -> None:
            if self._popup_items:
                self._popup_idx = min(len(self._popup_items) - 1, self._popup_idx + 1)
                self._invalidate()
                return
            self._input_buffer.history_forward()

        @kb.add("pageup")
        def _page_up(event: Any) -> None:
            self._scroll_offset += 10
            self._invalidate()

        @kb.add("pagedown")
        def _page_down(event: Any) -> None:
            self._scroll_offset = max(0, self._scroll_offset - 10)
            self._invalidate()

        popup_ctrl = _PopupMenuControl(self)
        popup_menu = ConditionalContainer(
            Window(
                content=popup_ctrl,
                height=lambda: D.exact(self._popup_height()),
                always_hide_cursor=True,
            ),
            filter=Condition(lambda: bool(self._popup_items)),
        )

        bottom_border = Window(height=1, char="─", style="class:separator")

        status_ctrl = FormattedTextControl(text=self._get_status_text, focusable=False)
        status_window = Window(content=status_ctrl, height=1, always_hide_cursor=True)

        layout = Layout(
            HSplit([
                output_window, topic_line, popup_menu,
                separator, input_window, bottom_border, status_window,
            ]),
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

    def set_on_pre_submit(self, callback: Callable[[str], None]) -> None:
        self._on_pre_submit = callback

    def set_on_cancel(self, callback: Callable[[], None]) -> None:
        self._on_cancel = callback

    def enter_new_topic_mode(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Switch input box to topic-name entry mode."""
        self._input_mode = "new_topic"
        self._new_topic_cb = callback
        self._input_buffer.reset()
        self._invalidate()

    def _exit_new_topic_mode(self) -> None:
        self._input_mode = "chat"
        self._new_topic_cb = None
        self._input_buffer.reset()
        self._invalidate()

    def set_commands(self, commands: list[tuple[str, str]]) -> None:
        """Register available commands for the popup completion menu."""
        self._all_commands = commands

    def show_topic_popup(
        self,
        topics: list[str],
        on_select: Callable[[str], Awaitable[None]],
    ) -> None:
        """Show an interactive topic picker above the input field."""
        self._popup_all_topics = list(topics)
        self._popup_mode = "topic"
        self._popup_items = [(t, t) for t in topics]
        self._popup_idx = 0
        self._popup_on_select = on_select
        self._input_buffer.reset()
        self._invalidate()

    def hide_popup(self) -> None:
        """Hide the command/topic popup."""
        self._popup_mode = "hidden"
        self._popup_items = []
        self._popup_idx = 0
        self._popup_on_select = None
        self._popup_all_topics = []
        self._invalidate()

    def _popup_height(self) -> int:
        """Total rendered lines for the popup (items + scroll hint rows)."""
        n = len(self._popup_items)
        if n == 0:
            return 0
        MAX = _PopupMenuControl.MAX_VISIBLE
        idx = self._popup_idx
        start = max(0, idx - MAX + 1)
        if start + MAX > n:
            start = max(0, n - MAX)
        end = min(n, start + MAX)
        return (end - start) + (1 if start > 0 else 0) + (1 if end < n else 0)

    def _update_popup(self) -> None:
        """Recompute popup items based on current buffer text (called on text change)."""
        if self._input_mode == "new_topic":
            return
        text = self._input_buffer.text
        if self._popup_mode == "topic":
            query = text.lower()
            filtered = [(t, t) for t in self._popup_all_topics if query in t.lower()]
            self._popup_items = filtered
            self._popup_idx = min(self._popup_idx, max(0, len(filtered) - 1))
            self._invalidate()
            return
        if not text.startswith("/"):
            if self._popup_items:
                self._popup_mode = "hidden"
                self._popup_items = []
                self._popup_idx = 0
                self._invalidate()
            return
        query = text[1:].lower()
        ranked: list[tuple[int, str, str]] = []
        for cmd, desc in self._all_commands:
            cmd_name = cmd[1:].lower()
            if not query:
                ranked.append((1, cmd, desc))
            elif cmd_name == query:
                ranked.append((0, cmd, desc))
            elif cmd_name.startswith(query):
                ranked.append((1, cmd, desc))
        ranked.sort(key=lambda x: x[0])
        new_items = [(cmd, desc) for _, cmd, desc in ranked]
        self._popup_items = new_items
        self._popup_idx = 0
        self._popup_mode = "command" if new_items else "hidden"
        self._invalidate()

    def set_is_processing(self, value: bool) -> None:
        """Toggle the ESC-to-cancel hint in the status bar."""
        self._is_processing = value
        self._invalidate()

    def update_context_usage(self, used: int, total: int) -> None:
        """Refresh the context-usage indicator (called after each agent turn)."""
        self._ctx_used = used
        self._ctx_total = total
        self._invalidate()

    def set_topic(self, name: str) -> None:
        """Update the displayed topic name in the status bar and terminal title."""
        self._topic = name
        self._invalidate()
        print(f"\033]0;nanobot — {name}\007", end="", flush=True)

    def reset_history(self) -> None:
        """Clear all output history (used when switching topics)."""
        self._cancel_thinking()
        self._cancel_tool_task()
        self._output_lines.clear()
        self._stream_buf = ""
        self._stream_cache = ""
        self._stream_cache_key = 0
        self._stream_ts = ""
        self._last_sep = False
        self._scroll_offset = 0
        self._ctx_used = 0
        self._ctx_total = 0
        self._live_progress = ""
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
        """Show the current tool being executed (with rotation animation)."""
        self._tool_hint = text
        self._stream_cache = self._render_tool_executing()
        if self._tool_task is None:
            self._tool_frame = 0
            self._tool_task = asyncio.ensure_future(self._animate_tool_executing())
        self._pin_to_bottom()
        self._invalidate()

    def add_system(self, text: str) -> None:
        """Add a system notification line (e.g., 'queued')."""
        self._append_block(self._render_system(text))
        self._pin_to_bottom()
        self._invalidate()

    def stream_start(self) -> None:
        """Mark the beginning of a new streaming response (captures timestamp)."""
        self._cancel_thinking()
        self._cancel_tool_task()
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._live_progress = ""
        self._thinking_frame = 0
        self._stream_cache = self._render_thinking()
        self._stream_cache_key = 0
        self._pin_to_bottom()
        self._invalidate()
        self._thinking_task = asyncio.ensure_future(self._animate_thinking())

    def tool_phase_start(self) -> None:
        """Switch to tool-execution phase: stop thinking animation, start tool rotation."""
        self._cancel_thinking()
        self._cancel_tool_task()
        self._stream_buf = ""
        self._live_progress = ""
        self._thinking_frame = 0
        self._tool_frame = 0
        self._tool_hint = ""
        if not self._stream_ts:
            self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_cache = self._render_tool_executing()
        self._stream_cache_key = 0
        self._pin_to_bottom()
        self._invalidate()
        self._tool_task = asyncio.ensure_future(self._animate_tool_executing())

    def stream_delta(self, delta: str) -> None:
        """Append a streaming delta and refresh the output pane."""
        self._cancel_thinking()
        self._cancel_tool_task()
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
        self._cancel_tool_task()
        if self._stream_buf.strip():
            self._append_block(
                self._render_response(self._stream_buf, metadata, ts=self._stream_ts)
            )
        self._stream_buf = ""
        self._stream_cache = ""
        self._stream_cache_key = 0
        # _stream_ts intentionally kept so live area stays visible for tool phase
        self._pin_to_bottom()
        self._invalidate()

    def pop_stream(self) -> str:
        """Return accumulated stream text and clear without adding to history."""
        self._cancel_thinking()
        self._cancel_tool_task()
        buf = self._stream_buf
        self._stream_buf = ""
        self._stream_cache = ""
        self._stream_cache_key = 0
        self._stream_ts = ""
        self._live_progress = ""
        return buf

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def run_async(self) -> None:
        if self._app is not None:
            # Pre-load FileHistory before first draw so the first Enter isn't slow.
            # history_backward() triggers the lazy synchronous file read; reset()
            # immediately restores the empty state (no visual artifact).
            try:
                self._input_buffer.history_backward()
                self._input_buffer.reset()
            except Exception:
                pass
            await self._app.run_async()

    def exit(self) -> None:
        if self._app is not None:
            self._app.exit()
