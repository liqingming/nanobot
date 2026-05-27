"""Textual-based TUI backend for nanobot.

Drop-in replacement for PromptTUI.  Key advantages over the prompt_toolkit
backend:
  - Native text selection + scroll while selecting (via Textual / terminal)
  - RichLog handles all scrolling; no manual line-slicing needed
  - Proper mouse scroll over the full output area without stealing native
    terminal selection

Activate by setting the environment variable::

    NANOBOT_TUI=textual nanobot agent

or adding ``tui_backend: "textual"`` to your config (when supported).

──────────────────────────────────────────────────────────────────────────
Cross-context calling convention
──────────────────────────────────────────────────────────────────────────

Textual maintains the running app via a ``ContextVar`` named ``active_app``.
Any task that internally creates a ``Timer`` (``set_interval`` /
``set_timer`` / many widget methods) expects to read that ContextVar at
tick time. If the task was started **outside** Textual (e.g. via
``asyncio.create_task`` from our bus consumer or any tool callback),
``active_app.get()`` raises ``LookupError`` later — typically surfacing
as a noisy traceback on app shutdown.

**Rule**: whenever a method on ``_NanobotApp`` that creates Textual Timers
is called from non-Textual code, route the call through
``_NanobotApp._safe_call(fn, *args)``. It schedules the call via
``call_later`` (which enters Textual's context) and swallows exceptions so
background tasks can't crash the loop.

Affected APIs (must use _safe_call when called from non-Textual paths):
  - ``start_thinking_spinner``
  - ``start_spinner``
  - ``_update_progress_line``

Safe to call directly (already inside Textual's event handlers):
  - ``stop_*`` / ``update_*`` / ``query_one`` / etc. — these don't create
    Timers; they only read or mutate widget state.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.fork.cli.tui_base import TUIBase
from nanobot.fork.cli.tui_keys import (
    EnterAction,
    PopupAction,
    TUIState,
    decide_enter_action,
    decide_popup_key,
)

# ---------------------------------------------------------------------------
# Textual imports — lazy-guarded so the module can be imported even when
# textual is not installed (the factory will raise a clear error then).
# ---------------------------------------------------------------------------
try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal
    from textual.events import Key
    from textual.widgets import Input, RichLog, Static
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False
    App = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


if _TEXTUAL_AVAILABLE:
    import subprocess
    import sys as _sys
    from rich.style import Style as _Style
    from textual.events import MouseDown, MouseMove, MouseUp, MouseScrollUp, MouseScrollDown
    from textual.geometry import Size
    from textual.strip import Strip

    class _OutputLog(RichLog):
        """RichLog with line-range text selection and clipboard copy.

        Mouse drag selects rows; releasing the mouse copies the selection to
        the system clipboard via Windows ``clip`` (primary) or Textual's OSC-52
        API (fallback).  Selected rows are highlighted with a dark-blue tint.
        """
        can_focus = False

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._sel_start: int | None = None  # content-space row index
            self._sel_end: int | None = None
            self._selecting: bool = False
            self._sel_moved: bool = False  # True once mouse moves during drag
            self._user_ranges: list[tuple[int, int]] = []  # gray-bg row ranges
            self._collapsed: dict[int, list[Any]] = {}  # summary_line → hidden strips
            self._expanded: dict[int, list[Any]] = {}   # expand_start → original strips (for re-collapse)

        # ── selection helpers ──────────────────────────────────────────────

        def _sel_rows(self) -> tuple[int, int] | None:
            if self._sel_start is None or self._sel_end is None:
                return None
            return min(self._sel_start, self._sel_end), max(self._sel_start, self._sel_end)

        def _clear_selection(self) -> None:
            self._sel_start = None
            self._sel_end = None
            self.refresh()

        def _extract_selected_text(self) -> str:
            rows = self._sel_rows()
            if rows is None:
                return ""
            start, end = rows
            parts: list[str] = []
            for row in range(start, end + 1):
                if row < len(self.lines):
                    parts.append(self.lines[row].text.rstrip())
            return "\n".join(parts)

        def _copy_to_clipboard(self, text: str) -> None:
            copied = False
            if _sys.platform == "win32":
                try:
                    import ctypes
                    CF_UNICODETEXT = 13
                    GMEM_MOVEABLE = 0x0002
                    k32 = ctypes.windll.kernel32
                    u32 = ctypes.windll.user32
                    # Must declare restype=c_void_p — default c_long truncates
                    # 64-bit handles on 64-bit Python, causing memmove to crash.
                    k32.GlobalAlloc.restype = ctypes.c_void_p
                    k32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
                    k32.GlobalLock.restype = ctypes.c_void_p
                    k32.GlobalLock.argtypes = [ctypes.c_void_p]
                    k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
                    u32.OpenClipboard.restype = ctypes.c_bool
                    u32.OpenClipboard.argtypes = [ctypes.c_void_p]
                    u32.SetClipboardData.restype = ctypes.c_void_p
                    u32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
                    encoded = text.encode("utf-16-le") + b"\x00\x00"
                    handle = k32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
                    if not handle:
                        raise OSError("GlobalAlloc failed")
                    ptr = k32.GlobalLock(handle)
                    if not ptr:
                        raise OSError("GlobalLock failed")
                    ctypes.memmove(ptr, encoded, len(encoded))
                    k32.GlobalUnlock(handle)
                    if not u32.OpenClipboard(None):
                        raise OSError("OpenClipboard failed")
                    try:
                        u32.EmptyClipboard()
                        u32.SetClipboardData(CF_UNICODETEXT, handle)
                    finally:
                        u32.CloseClipboard()
                    copied = True
                except Exception:
                    pass
            if not copied:
                try:
                    self.app.copy_to_clipboard(text)
                    copied = True
                except Exception:
                    pass
            try:
                if copied:
                    self.app.notify("已复制到剪贴板", timeout=1.5)
            except Exception:
                pass

        # ── rendering ──────────────────────────────────────────────────────

        def add_user_range(self, start: int, end: int) -> None:
            self._user_ranges.append((start, end))
            self._line_cache.clear()
            self.refresh()

        @staticmethod
        def _force_bgcolor(strip: Strip, bgcolor: str) -> Strip:
            from rich.segment import Segment as _Seg
            new_segs = []
            for t, s, c in strip._segments:
                new_style = _Style(
                    color=s.color if s else None,
                    bgcolor=bgcolor,
                    bold=s.bold if s else None,
                    italic=s.italic if s else None,
                    underline=s.underline if s else None,
                    dim=s.dim if s else None,
                    strike=s.strike if s else None,
                    reverse=s.reverse if s else None,
                    overline=s.overline if s else None,
                )
                new_segs.append(_Seg(t, new_style, c))
            return Strip(new_segs, strip.cell_length)

        @staticmethod
        def _force_colors(strip: Strip, bgcolor: str, color: str) -> Strip:
            """Rebuild a Strip forcing specific bgcolor and color on every segment.

            ``apply_style`` won't work here because existing segment bgcolors
            (e.g. #0c0c0c from rich_style) take priority in Rich's style merge.
            """
            from rich.segment import Segment as _Seg
            target = _Style(bgcolor=bgcolor, color=color)
            new_segs = []
            for text, style, ctrl in strip._segments:
                new_style = _Style(
                    color=target.color,
                    bgcolor=target.bgcolor,
                    bold=style.bold if style else None,
                    italic=style.italic if style else None,
                    underline=style.underline if style else None,
                    dim=style.dim if style else None,
                    strike=style.strike if style else None,
                )
                new_segs.append(_Seg(text, new_style, ctrl))
            return Strip(new_segs, strip.cell_length)

        def render_line(self, y: int) -> Strip:
            scroll_x, scroll_y = self.scroll_offset
            n = len(self.lines)
            width = self.scrollable_content_region.width
            # Clamp scroll_y when truncate_to() reduced virtual_size but scroll
            # position hasn't been updated yet — prevents blank lines during repaint.
            if n > 0:
                max_scroll = max(0, n - self.scrollable_content_region.height)
                if scroll_y > max_scroll:
                    scroll_y = max_scroll
            content_row = scroll_y + y
            if scroll_y != self.scroll_offset.y:
                if content_row < n:
                    strip = self._render_line(content_row, scroll_x, width)
                    strip = strip.apply_style(self.rich_style)
                else:
                    strip = Strip.blank(width, self.rich_style)
            else:
                strip = super().render_line(y)
            if self._user_ranges and any(s <= content_row <= e for s, e in self._user_ranges):
                strip = self._force_bgcolor(strip, "#2d2d2d")
            rows = self._sel_rows()
            if rows is not None and rows[0] <= content_row <= rows[1]:
                strip = self._force_colors(strip, bgcolor="white", color="black")
            return strip

        # ── mouse events ───────────────────────────────────────────────────

        def on_mouse_down(self, event: MouseDown) -> None:
            if event.button != 1:  # left button only
                return
            try:
                n = len(self.lines)
                if n == 0:
                    return
                _, scroll_y = self.scroll_offset
                row = max(0, min(scroll_y + event.y, n - 1))
                self._sel_start = row
                self._sel_end = row
                self._selecting = True
                self._sel_moved = False
                self.capture_mouse()
            except Exception:
                pass
            event.stop()

        def on_mouse_move(self, event: MouseMove) -> None:
            if not self._selecting:
                return
            try:
                _, scroll_y = self.scroll_offset
                row = max(0, min(scroll_y + event.y, len(self.lines) - 1))
                self._sel_moved = True
                if row != self._sel_end:
                    self._sel_end = row
                    self.refresh()
            except Exception:
                pass
            event.stop()

        def on_mouse_up(self, event: MouseUp) -> None:
            if not self._selecting:
                return
            try:
                _, scroll_y = self.scroll_offset
                row = max(0, min(scroll_y + event.y, len(self.lines) - 1))
                self._sel_end = row
                self._selecting = False
                self.release_mouse()
                if self._sel_moved:
                    # Drag → copy selected text
                    text = self._extract_selected_text()
                    if text.strip():
                        self._copy_to_clipboard(text)
                        self.set_timer(1.5, self._clear_selection)
                    else:
                        self._clear_selection()
                else:
                    # Bare click → toggle collapsed/expanded block
                    self.toggle_block(row)
                    self._clear_selection()
            except Exception:
                self._selecting = False
                try:
                    self.release_mouse()
                except Exception:
                    pass
                self._clear_selection()
            event.stop()

        def truncate_to(self, n: int) -> None:
            """Remove all lines after index n, used to overwrite streaming content."""
            if n >= len(self.lines):
                return
            self._collapsed = {k: v for k, v in self._collapsed.items() if k < n}
            self._expanded = {k: v for k, v in self._expanded.items() if k < n}
            self.lines = self.lines[:n]
            self._widest_line_width = (
                max(line.cell_length for line in self.lines) if self.lines else 0
            )
            self._line_cache.clear()
            self.virtual_size = Size(self._widest_line_width, len(self.lines))
            # Note: scroll_offset.y may briefly exceed new virtual height here.
            # render_line() clamps it so no blank lines appear during the repaint
            # triggered by virtual_size change (a reactive that fires immediately).
            self.refresh()

        def _shift_block_keys(self, after: int, delta: int) -> None:
            """Shift all block-tracking keys > after by delta (used after insert/remove)."""
            self._collapsed = {(k + delta if k > after else k): v for k, v in self._collapsed.items()}
            self._expanded = {(k + delta if k > after else k): v for k, v in self._expanded.items()}

        def collapse_lines(self, start: int, end: int) -> None:
            """Collapse lines [start, end) to a single clickable summary line."""
            if end <= start + 1 or end > len(self.lines):
                return
            hidden = list(self.lines[start:end])
            n = len(hidden)
            preview = hidden[0].text.strip()[:60] if hidden else ""
            suffix = "…" if len(preview) == 60 else ""
            label = f"▶ {n} 行  (点击展开)  {preview}{suffix}"
            rt = Text(label, style="dim italic")
            width = max(self.scrollable_content_region.width, self.min_width)
            console = self.app.console
            from rich.segment import Segment as _Seg
            segs = list(console.render(rt, console.options.update_width(width)))
            line_segs = list(_Seg.split_lines(segs))
            summary_strip = Strip.from_lines(line_segs)[0].adjust_cell_length(width) if line_segs else Strip.blank(width)
            # Fold any _expanded records that fall entirely within [start, end)
            self._expanded = {k: v for k, v in self._expanded.items() if not (start <= k < end)}
            self.lines[start] = summary_strip
            del self.lines[start + 1:end]
            self._collapsed[start] = hidden
            removed = (end - start) - 1  # net lines removed
            self._shift_block_keys(start, -removed)
            self._widest_line_width = max(line.cell_length for line in self.lines) if self.lines else 0
            self._line_cache.clear()
            self.virtual_size = Size(self._widest_line_width, len(self.lines))
            self.refresh()

        def expand_line(self, idx: int) -> None:
            """Replace a collapsed summary line at idx with its full hidden content."""
            if idx not in self._collapsed:
                return
            hidden = self._collapsed.pop(idx)
            self._expanded[idx] = hidden
            self.lines[idx:idx + 1] = hidden
            shift = len(hidden) - 1
            self._shift_block_keys(idx, shift)
            self._widest_line_width = max(line.cell_length for line in self.lines) if self.lines else 0
            self._line_cache.clear()
            self.virtual_size = Size(self._widest_line_width, len(self.lines))
            self.refresh()

        def toggle_block(self, row: int) -> None:
            """Expand a collapsed summary line, or re-collapse an expanded block."""
            if row in self._collapsed:
                self.expand_line(row)
                return
            for start, strips in list(self._expanded.items()):
                if start <= row < start + len(strips):
                    del self._expanded[start]
                    self.collapse_lines(start, start + len(strips))
                    return

        def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
            event.stop()
            self.scroll_relative(y=-3)

        def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
            event.stop()
            self.scroll_relative(y=3)
_TOOL_SPINNER = ["◐", "◓", "◑", "◒"]


def _rich_to_ansi(render_fn: Callable[[Console], Any], width: int = 100) -> str:
    buf = StringIO()
    c = Console(file=buf, width=width, force_terminal=True, highlight=False, color_system="256")
    render_fn(c)
    return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# Custom Input with history navigation + popup awareness
# ---------------------------------------------------------------------------

if _TEXTUAL_AVAILABLE:
    class _HistoryInput(Input):
        """Input that delegates ↑/↓ to the TUI's history / popup state."""

        def __init__(self, tui: "TextualTUI", **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._tui_ref = tui

        async def _on_key(self, event: Key) -> None:
            tui = self._tui_ref
            if event.key not in ("up", "down", "tab"):
                # Enter falls through to Input.Submitted; other keys use defaults.
                return

            selected = (
                tui._popup_items[tui._popup_idx][0]
                if tui._popup_items and 0 <= tui._popup_idx < len(tui._popup_items)
                else None
            )
            state = TUIState(
                input_mode=tui._input_mode,
                popup_mode=tui._popup_mode,
                popup_has_items=bool(tui._popup_items),
                popup_selected_value=selected,
                input_text=self.value,
            )
            decision = decide_popup_key(event.key, state)
            action = decision.action

            if action == PopupAction.CYCLE_UP:
                tui._popup_idx = max(0, tui._popup_idx - 1)
                tui._refresh_popup()
                event.prevent_default()
            elif action == PopupAction.CYCLE_DOWN:
                tui._popup_idx = min(len(tui._popup_items) - 1, tui._popup_idx + 1)
                tui._refresh_popup()
                event.prevent_default()
            elif action == PopupAction.COMPLETE:
                self.value = decision.value
                self.cursor_position = len(decision.value)
                event.prevent_default()
            elif action == PopupAction.HISTORY_BACK:
                text = tui._history_backward()
                if text is not None:
                    self.value = text
                    self.cursor_position = len(text)
                event.prevent_default()
            elif action == PopupAction.HISTORY_FORWARD:
                text = tui._history_forward()
                self.value = text if text is not None else ""
                if text is not None:
                    self.cursor_position = len(text)
                event.prevent_default()
            # PopupAction.IGNORE → don't call prevent_default, let the key flow normally

    # -----------------------------------------------------------------------
    # Textual App
    # -----------------------------------------------------------------------

    class _NanobotApp(App):  # type: ignore[misc]
        ENABLE_MOUSE_SUPPORT = True  # needed for on_mouse_down/move/up selection events

        CSS = """
        Screen {
            layout: vertical;
            background: #0c0c0c;
        }
        #output {
            height: 1fr;
            scrollbar-gutter: stable;
            scrollbar-size-vertical: 1;
            scrollbar-color: grey;
            scrollbar-color-hover: grey;
            scrollbar-color-active: grey;
            scrollbar-background: #0c0c0c;
            border: none;
            background: #0c0c0c;
        }
        #sep-row {
            height: 1;
            background: #0c0c0c;
        }
        #sep {
            width: 1fr;
            background: #0c0c0c;
        }
        #topic-bar {
            width: auto;
            background: #0c0c0c;
            padding: 0 1;
            display: none;
        }
        #topic-bar.visible {
            display: block;
        }
        #live {
            height: auto;
            min-height: 1;
            padding: 0 1;
            background: #0c0c0c;
            color: $text-muted;
        }
        #popup {
            height: auto;
            max-height: 10;
            display: none;
            background: #0c0c0c;
            border: round $primary;
            padding: 0 1;
        }
        #popup.visible {
            display: block;
        }
        #input {
            height: auto;
            border: none;
            padding: 0 1;
            background: #0c0c0c;
        }
        Input {
            background: #0c0c0c;
        }
        #status {
            height: 1;
            background: #0c0c0c;
            color: $text-muted;
            padding: 0 1;
        }
        #todo-bar {
            height: 1;
            background: #0c0c0c;
            color: $text-muted;
            padding: 0 1;
            display: none;
        }
        #todo-bar.visible {
            display: block;
        }
        """

        BINDINGS = [
            Binding("ctrl+c", "quit_app", show=False, priority=True),
            Binding("ctrl+d", "eof_app", show=False, priority=True),
            Binding("escape", "escape_app", show=False, priority=True),
            Binding("pageup", "page_up", show=False),
            Binding("pagedown", "page_down", show=False),
        ]

        def __init__(self, tui: "TextualTUI") -> None:
            super().__init__()
            self._tui = tui
            self._spinner_timer: Any = None
            self._spinner_frame = 0
            self._spinner_start_time: float = 0.0
            self._thinking_timer: Any = None
            self._thinking_frame = 0
            self._thinking_start_time: float = 0.0

        def _safe_call(self, fn: Any, *args: Any) -> None:
            """Schedule ``fn(*args)`` via ``call_later`` so it runs inside
            Textual's active_app context, and swallow any exception so
            background-task callers can't surface noisy tracebacks.

            Use this whenever an asyncio Task / Thread / external callback
            needs to invoke a method that internally creates Textual Timers
            (e.g. ``start_spinner``, ``start_thinking_spinner``). Without
            the context wrap, Timer's ``_tick`` will raise ``LookupError`` on
            ``active_app.get()`` and pollute shutdown with tracebacks.
            """
            try:
                self.call_later(lambda: fn(*args))
            except Exception:
                pass

        @staticmethod
        def _elapsed_suffix(start_time: float) -> str:
            """Render elapsed time since ``start_time`` as a spinner suffix.

            Format:
              * < 1s    → ""              (avoid flicker on quick ops)
              * < 60s   → " (5s)"         (integer seconds)
              * >= 60s  → " (1m30s)"      (compact minutes+seconds)

            Uses ``time.monotonic`` which is immune to wall-clock adjustments,
            so the suffix faithfully reflects real elapsed time.
            """
            import time
            elapsed = time.monotonic() - start_time
            if elapsed < 1.0:
                return ""
            if elapsed < 60:
                return f" ({int(elapsed)}s)"
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f" ({minutes}m{seconds}s)"

        def compose(self) -> ComposeResult:
            yield _OutputLog(id="output", markup=True, highlight=False, wrap=True)
            yield Static("", id="live")
            yield Static("", id="popup")
            yield Static("", id="todo-bar")
            with Horizontal(id="sep-row"):
                yield Static("[dim cyan]" + "─" * 80 + "[/dim cyan]", id="sep", markup=True)
                yield Static("", id="topic-bar")
            yield _HistoryInput(self._tui, placeholder="You: ", id="input")
            yield Static("", id="status")

        def on_mount(self) -> None:
            self.query_one("#input").focus()
            self._write_welcome()
            # After the first full render, re-focus + full layout refresh so
            # Windows Terminal updates its IME candidate window position to the
            # actual input cursor location instead of defaulting to top-left.
            self.call_after_refresh(self._refocus_input)

        def _refocus_input(self) -> None:
            inp = self.query_one("#input", Input)
            inp.focus()
            self.refresh(layout=True)

        def _write_welcome(self) -> None:
            out = self.query_one("#output", _OutputLog)
            tui = self._tui
            out.write(f"[cyan bold]{__logo__} nanobot[/cyan bold]  [dim]v{__version__}[/dim]"
                      + (f"  [dim]{tui._model}[/dim]" if tui._model else ""))
            out.write("")
            out.write("[bold]快捷键[/bold]")
            out.write("  [cyan]PageUp / PageDown[/cyan]   滚动历史记录")
            out.write("  [cyan]↑ / ↓[/cyan]              切换输入历史")
            out.write("  [cyan]ESC[/cyan]                取消当前请求")
            out.write("  [cyan]Ctrl+C / Ctrl+D[/cyan]    退出")
            out.write("  [cyan]鼠标拖选[/cyan]            选中行后自动复制到剪贴板")
            out.write("")

        # ── input callbacks ────────────────────────────────────────────────

        def on_input_submitted(self, event: Input.Submitted) -> None:
            tui = self._tui
            selected = (
                tui._popup_items[tui._popup_idx][0]
                if tui._popup_items and 0 <= tui._popup_idx < len(tui._popup_items)
                else None
            )
            decision = decide_enter_action(TUIState(
                input_mode=tui._input_mode,
                popup_mode=tui._popup_mode,
                popup_has_items=bool(tui._popup_items),
                popup_selected_value=selected,
                input_text=event.value,
            ))
            action, value = decision.action, decision.value
            event.input.clear()
            tui._history_pos = -1

            if action == EnterAction.NEW_TOPIC:
                cb = tui._new_topic_cb
                tui._exit_new_topic_mode()
                if cb:
                    asyncio.ensure_future(cb(value))
                return
            if action == EnterAction.TOPIC_SELECT:
                cb = tui._popup_on_select
                tui.hide_popup()
                if cb:
                    asyncio.ensure_future(cb(value))
                return
            if action == EnterAction.COMMAND_SUBMIT:
                tui.hide_popup()
                # Commands must NOT echo to output (no add_user_echo / pre_submit)
                # and must NOT be saved to input history.
                if tui._on_submit:
                    asyncio.ensure_future(tui._on_submit(value))
                return
            if action == EnterAction.SUBMIT:
                # Slash-prefixed typed text is also a command — skip history.
                if not value.strip().startswith("/"):
                    tui._add_to_history(value)
                if tui._on_pre_submit and not tui._is_processing:
                    tui._on_pre_submit(value)
                if tui._on_submit:
                    asyncio.ensure_future(tui._on_submit(value))
                return
            # NOOP: nothing to do

        def on_input_changed(self, event: Input.Changed) -> None:
            self._tui._on_input_changed(event.value)

        # ── actions ────────────────────────────────────────────────────────

        def action_quit_app(self) -> None:
            try:
                out = self.query_one("#output", _OutputLog)
                if out._sel_moved and out._sel_rows() is not None:
                    text = out._extract_selected_text()
                    if text.strip():
                        out._copy_to_clipboard(text)
                        out.set_timer(1.5, out._clear_selection)
                        return
            except Exception:
                pass
            self.exit()

        def action_eof_app(self) -> None:
            if not self.query_one("#input", Input).value:
                self.exit()

        def action_escape_app(self) -> None:
            tui = self._tui
            if tui._popup_mode != "hidden":
                # If a question-popup sequence is in flight, ESC cancels the
                # whole sequence (not just this one question).
                tui._cancel_question_flow()
                tui.hide_popup()
            elif tui._input_mode == "new_topic":
                tui._exit_new_topic_mode()
            elif tui._on_cancel:
                tui._on_cancel()

        def action_page_up(self) -> None:
            self.query_one("#output", _OutputLog).scroll_relative(y=-10)

        def action_page_down(self) -> None:
            self.query_one("#output", _OutputLog).scroll_relative(y=10)

        def on_mouse_scroll_up(self) -> None:
            self.query_one("#output", _OutputLog).scroll_relative(y=-3)

        def on_mouse_scroll_down(self) -> None:
            self.query_one("#output", _OutputLog).scroll_relative(y=3)

        # ── spinner ────────────────────────────────────────────────────────

        def start_spinner(self, tool: bool = False) -> None:
            import time
            self._stop_spinner()
            self._spinner_frame = 0
            self._spinner_tool = tool
            # tool spinner: per-tool clock (each tool starts fresh).
            # thinking spinner: anchored to the turn start, so the elapsed
            # counter doesn't reset between LLM segments / idle gaps.
            self._spinner_start_time = time.monotonic()

            def _tick() -> None:
                self._spinner_frame += 1
                tui = self._tui
                frames = _TOOL_SPINNER if self._spinner_tool else _SPINNER
                icon = frames[self._spinner_frame % len(frames)]
                if self._spinner_tool:
                    base = self._spinner_start_time
                else:
                    base = tui._turn_start_time or self._spinner_start_time
                suffix = self._elapsed_suffix(base)
                # #live shows just the spinner line — the "🐈 nanobot ts"
                # header already lives in #output, so duplicating it here
                # is visual noise (especially right under the popup area).
                # Use grey50 instead of plain "dim" so the live strip reads
                # as clearly secondary to the main output.
                if self._spinner_tool:
                    label = tui._tool_hint or "executing..."
                    text = f"[grey50]{icon} {label}{suffix}[/grey50]"
                else:
                    text = f"[grey50]{icon} thinking{suffix}...[/grey50]"
                try:
                    self.query_one("#live", Static).update(text)
                except Exception:
                    pass

            interval = 0.15 if tool else 0.10
            self._spinner_timer = self.set_interval(interval, _tick)

        def _stop_spinner(self) -> None:
            if self._spinner_timer is not None:
                self._spinner_timer.stop()
                self._spinner_timer = None
            # Explicitly clear #live so the last spinner frame doesn't
            # linger as a stale-looking static glyph after stop.
            try:
                self.query_one("#live", Static).update("")
            except Exception:
                pass

        def stop_spinner(self) -> None:
            self._stop_spinner()

        # ── thinking spinner (animates inside #output) ─────────────────────

        def start_thinking_spinner(self) -> None:
            import time
            self._stop_thinking_spinner()
            self._thinking_frame = 0
            self._thinking_start_time = time.monotonic()

            def _tick() -> None:
                from rich.segment import Segment as _Seg
                self._thinking_frame += 1
                icon = _SPINNER[self._thinking_frame % len(_SPINNER)]
                # Use the turn-level anchor when present so the elapsed counter
                # tracks the whole "thinking" duration, not just this spinner.
                base = self._tui._turn_start_time or self._thinking_start_time
                suffix = self._elapsed_suffix(base)
                try:
                    out = self.query_one("#output", _OutputLog)
                    idx = self._tui._tool_placeholder_line
                    if 0 <= idx < len(out.lines):
                        hint = self._tui._tool_hint
                        # Tool execution → 4-space indent + cyan (matches the
                        # eventual "↳ tool" trace). Idle thinking → 2-space
                        # indent + grey50 (subtler, marks waiting state).
                        if hint:
                            rt = Text(f"    {icon} {hint}{suffix}", style="cyan")
                        else:
                            rt = Text(f"  {icon} 思考中...{suffix}", style="grey50")
                        width = max(out.scrollable_content_region.width, out.min_width)
                        console = out.app.console
                        segs = list(console.render(rt, console.options.update_width(width)))
                        line_segs = list(_Seg.split_lines(segs))
                        if line_segs:
                            new_strip = Strip.from_lines(line_segs)[0].adjust_cell_length(width)
                        else:
                            new_strip = Strip.blank(width)
                        out.lines[idx] = new_strip
                        out._line_cache.clear()
                        out.refresh()
                except Exception:
                    pass

            self._thinking_timer = self.set_interval(0.1, _tick)

        def _stop_thinking_spinner(self) -> None:
            if self._thinking_timer is not None:
                self._thinking_timer.stop()
                self._thinking_timer = None

        def stop_thinking_spinner(self) -> None:
            self._stop_thinking_spinner()

        # ── live area helpers ──────────────────────────────────────────────

        def update_live(self, text: str) -> None:
            try:
                self.query_one("#live", Static).update(text)
            except Exception:
                pass

        def clear_live(self) -> None:
            self.update_live("")

        # ── output helpers ─────────────────────────────────────────────────

        def write_output(self, text: str) -> None:
            try:
                out = self.query_one("#output", _OutputLog)
                out.write(text)
            except Exception:
                pass

        def write_separator(self) -> None:
            self.write_output("[dim]" + "─" * 80 + "[/dim]")

        def clear_output(self) -> None:
            try:
                self.query_one("#output", _OutputLog).clear()
            except Exception:
                pass

        # ── status / topic helpers ─────────────────────────────────────────

        def update_status(self, text: str) -> None:
            try:
                self.query_one("#status", Static).update(text)
            except Exception:
                pass

        def update_topic_bar(self, name: str) -> None:
            try:
                bar = self.query_one("#topic-bar", Static)
                bar.update(f"[dim cyan] {name} [/dim cyan]")
                if name:
                    bar.add_class("visible")
                else:
                    bar.remove_class("visible")
            except Exception:
                pass

        def update_todo_bar(self, text: str) -> None:
            try:
                bar = self.query_one("#todo-bar", Static)
                bar.update(text)
                if text:
                    bar.add_class("visible")
                else:
                    bar.remove_class("visible")
            except Exception:
                pass

        # ── popup helpers ──────────────────────────────────────────────────

        def update_popup(self, lines: list[str], visible: bool) -> None:
            try:
                popup = self.query_one("#popup", Static)
                if visible and lines:
                    popup.update("\n".join(lines))
                    popup.add_class("visible")
                else:
                    popup.update("")
                    popup.remove_class("visible")
            except Exception:
                pass

        # ── input helpers ──────────────────────────────────────────────────

        def set_input_placeholder(self, text: str) -> None:
            try:
                self.query_one("#input", Input).placeholder = text
            except Exception:
                pass

        def set_input_value(self, text: str) -> None:
            try:
                inp = self.query_one("#input", Input)
                inp.value = text
                inp.cursor_position = len(text)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TextualTUI — the TUIBase implementation
# ---------------------------------------------------------------------------

class TextualTUI(TUIBase):
    """Textual-based TUI backend.

    Set ``NANOBOT_TUI=textual`` to activate via the factory.
    """

    # ── Theme: shared color tokens across tool traces, todo bar, system msgs ──
    #
    # Keep these in sync conceptually so users see consistent semantics:
    #   ACTIVE  = a task is currently running / in flight
    #   SUCCESS = a task finished successfully
    #   ERROR   = a task failed
    #   MARKER  = static visual markers like →, ↳, ☐, • (low-emphasis structure)
    #   HINT    = primary text inside a trace (tool name, todo content)
    #   MUTED   = tertiary annotations (result summary, progress count)
    THEME_ACTIVE = "yellow"
    THEME_SUCCESS = "green"
    THEME_ERROR = "red"
    THEME_MARKER = "dim cyan"
    THEME_HINT = "cyan"
    THEME_MUTED = "dim"

    def __init__(
        self,
        render_markdown: bool = True,
        history_file: str | None = None,
        model: str | None = None,
    ) -> None:
        if not _TEXTUAL_AVAILABLE:
            raise ImportError(
                "textual is required for the Textual TUI backend.\n"
                "Install it with:  pip install 'nanobot-ai[textual]'"
            )
        self._render_md = render_markdown
        self._history_file = Path(history_file) if history_file else None
        self._model = model

        # Input history
        self._history: list[str] = self._load_history()
        self._history_pos: int = -1  # -1 = not navigating

        # Streaming state
        self._stream_buf: str = ""
        self._stream_ts: str = ""
        self._stream_header_line: int = 0  # output-log line index where stream header was written
        self._tool_placeholder_line: int = 0  # output-log line index of the current thinking/executing placeholder
        self._last_content_start: int = 0  # where the most recent intermediate content block started
        self._flushed_parts: list[str] = []  # intermediate LLM text flushed between tool calls
        self._tool_hint: str = ""
        # Accumulates reasoning_content chunks (LLM thinking trace) so we can
        # flush them as a dim italic history block on _reasoning_end. Mirrors
        # PromptTUI's behavior; required because reasoning models like
        # DeepSeek-v4-pro otherwise have no place to land their trace.
        self._reasoning_buf: str = ""
        self._last_sep: bool = False
        self._header_already_rendered: bool = False  # set by pop_stream so add_response skips a second header
        # Fork: when a turn's header is on screen but NOTHING visible followed
        # it (reasoning suppressed + no tool trace + no mid-turn flush), the
        # continuation "─ts─" separator in _write_response would dangle with
        # nothing to separate. pop_stream sets this so _write_response skips it.
        self._suppress_segment_sep: bool = False
        self._idle_thinking_task: Any = None  # asyncio.Task scheduling the "still thinking" spinner
        self._idle_placeholder_visible: bool = False  # whether the idle thinking line is in #output
        # When idle thinking is shown, _tool_placeholder_line is moved to its
        # line so the spinner updates that line. This backup preserves the
        # original (stream_delta / tool) anchor so cancel_idle can restore it
        # — without restoration, pop_stream would truncate the wrong line and
        # add_response would duplicate the streamed content.
        self._tool_placeholder_line_backup: int | None = None
        self._turn_start_time: float = 0.0  # monotonic timestamp when the current LLM turn started (stream_start)
        self._tool_start_time: float = 0.0  # monotonic timestamp when the current tool started (add_progress)
        # State for show_question_popup (sequential multi-question prompt)
        self._question_queue: list[dict] = []
        self._question_answers: dict[str, str] = {}
        self._question_on_complete: Callable[[dict[str, str] | None], Awaitable[None]] | None = None

        # Callbacks
        self._on_submit: Callable[[str], Awaitable[None]] | None = None
        self._on_pre_submit: Callable[[str], None] | None = None
        self._on_cancel: Callable[[], None] | None = None

        # State
        self._topic: str = ""
        self._is_processing: bool = False
        self._ctx_used: int = 0
        self._ctx_total: int = 0
        self._input_mode: str = "chat"  # "chat" | "new_topic"
        self._new_topic_cb: Callable[[str], Awaitable[None]] | None = None

        # Popup state
        self._all_commands: list[tuple[str, str]] = []
        self._popup_mode: str = "hidden"
        self._popup_items: list[tuple[str, str]] = []
        self._popup_idx: int = 0
        self._popup_on_select: Callable[[str], Awaitable[None]] | None = None
        self._popup_all_topics: list[str] = []

        self._app = _NanobotApp(self)

    # ── History file ───────────────────────────────────────────────────────

    def _load_history(self) -> list[str]:
        if not self._history_file or not self._history_file.exists():
            return []
        try:
            lines = self._history_file.read_text(encoding="utf-8").splitlines()
            # prompt_toolkit FileHistory format: entries separated by blank lines,
            # each line prefixed with "+". Read all non-empty, non-comment lines.
            entries: list[str] = []
            current: list[str] = []
            for line in lines:
                if line.startswith("+"):
                    current.append(line[1:])
                elif not line.strip() and current:
                    entries.append("\n".join(current))
                    current = []
            if current:
                entries.append("\n".join(current))
            return list(reversed(entries))  # most-recent first for backward()
        except Exception:
            return []

    def _add_to_history(self, text: str) -> None:
        # Avoid duplicates at the front
        if self._history and self._history[0] == text:
            return
        self._history.insert(0, text)
        if self._history_file:
            try:
                # Append in prompt_toolkit FileHistory format
                self._history_file.parent.mkdir(parents=True, exist_ok=True)
                with self._history_file.open("a", encoding="utf-8") as f:
                    for line in text.splitlines():
                        f.write(f"+{line}\n")
                    f.write("\n")
            except Exception:
                pass

    def _history_backward(self) -> str | None:
        if not self._history:
            return None
        self._history_pos = min(self._history_pos + 1, len(self._history) - 1)
        return self._history[self._history_pos]

    def _history_forward(self) -> str | None:
        if self._history_pos <= 0:
            self._history_pos = -1
            return None
        self._history_pos -= 1
        return self._history[self._history_pos]

    # ── Log write helpers ──────────────────────────────────────────────────

    def _log_write(self, *items: Any) -> None:
        """Write Rich renderables or markup strings directly to the output log."""
        try:
            out = self._app.query_one("#output", _OutputLog)
            for item in items:
                out.write(item)
        except Exception:
            pass

    def _write_response(self, content: str, ts: str, metadata: dict | None = None) -> None:
        """Write a completed response block as Rich objects (no ANSI conversion).

        When the response follows a streaming session that already rendered
        the "🐈 nanobot timestamp" header (and possibly tool traces below it),
        the header is omitted so we don't get duplicate headers — instead a
        short "─ HH:MM:SS ─" separator marks the new segment.
        """
        render_as_text = (metadata or {}).get("render_as") == "text"
        if self._header_already_rendered:
            # Continuation of an already-headed turn — mark the new segment
            # with a lightweight timestamp so it isn't glued to the previous one.
            # Strip date portion if present ("2026-05-20 22:45:30" → "22:45:30").
            # Fork: skip the separator entirely when nothing visible followed the
            # header (suppressed reasoning, no tool trace) — see pop_stream.
            if not self._suppress_segment_sep:
                short_ts = ts.split(" ", 1)[1] if " " in ts else ts
                self._log_write(f"[{self.THEME_MUTED}]─ {short_ts} ─[/{self.THEME_MUTED}]")
                self._log_write("")
        else:
            self._log_write(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
            self._log_write("")
        self._header_already_rendered = False  # consume flag
        self._suppress_segment_sep = False  # consume flag
        if self._render_md and not render_as_text and content.strip():
            self._log_write(Markdown(content))
        else:
            self._log_write(Text(content))
        self._log_write("")
        self._last_sep = True
        # Reset streaming header anchor so the next turn starts fresh.
        self._stream_header_line = 0

    def _write_user(self, text: str, ts: str) -> None:
        """Write a user message block; records line range for gray background."""
        try:
            out = self._app.query_one("#output", _OutputLog)
            start = len(out.lines)
        except Exception:
            start = None
        self._log_write(f"[bold blue]You[/bold blue] [dim]{ts}[/dim]")
        self._log_write("")
        self._log_write(text)
        if start is not None:
            try:
                out = self._app.query_one("#output", _OutputLog)
                out.add_user_range(start, len(out.lines) - 1)
            except Exception:
                pass
        self._log_write("")  # trailing blank (outside gray bg)
        # Fork: a full-width rule between the user block and the upcoming
        # response header — turns are separated more clearly than by a bare
        # blank line. Written outside the gray user-range recorded above.
        self._log_write(f"[{self.THEME_MUTED}]{'─' * 80}[/{self.THEME_MUTED}]")
        self._log_write("")
        self._last_sep = True

    # ── Popup helpers ──────────────────────────────────────────────────────

    def _refresh_popup(self) -> None:
        items = self._popup_items
        idx = self._popup_idx
        mode = self._popup_mode
        if not items or mode == "hidden":
            self._app.update_popup([], False)
            return
        lines: list[str] = []
        for i, (value, label) in enumerate(items):
            selected = i == idx
            prefix = " ▶ " if selected else "   "
            if selected:
                if mode == "command":
                    lines.append(f"[reverse]{prefix}{value:<12}  {label}[/reverse]")
                else:
                    lines.append(f"[reverse]{prefix}{value}[/reverse]")
            else:
                if mode == "command":
                    lines.append(f"[dim]{prefix}{value:<12}  {label}[/dim]")
                else:
                    lines.append(f"[dim]{prefix}{value}[/dim]")
        self._app.update_popup(lines, True)

    def _on_input_changed(self, text: str) -> None:
        """Called on every keystroke — update popup."""
        if self._input_mode == "new_topic":
            return
        if self._popup_mode == "topic":
            query = text.lower()
            filtered = [(t, t) for t in self._popup_all_topics if query in t.lower()]
            self._popup_items = filtered
            self._popup_idx = min(self._popup_idx, max(0, len(filtered) - 1))
            self._refresh_popup()
            return
        if not text.startswith("/"):
            if self._popup_mode != "hidden":
                self.hide_popup()
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
        self._refresh_popup()

    def _update_status(self) -> None:
        text = "[dim] ESC 打断 [/dim]" if self._is_processing else ""
        self._app.update_status(text)

    def _exit_new_topic_mode(self) -> None:
        self._input_mode = "chat"
        self._new_topic_cb = None
        self._app.set_input_placeholder("You: ")
        self._app.set_input_value("")

    def _append_sep(self) -> None:
        if not self._last_sep:
            self._log_write("")
            self._last_sep = True

    # ── TUIBase: lifecycle ─────────────────────────────────────────────────

    async def run_async(self) -> None:
        await self._app.run_async()

    def exit(self) -> None:
        self._app.exit()

    # ── TUIBase: callbacks ─────────────────────────────────────────────────

    def set_on_submit(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._on_submit = callback

    def set_on_pre_submit(self, callback: Callable[[str], None]) -> None:
        self._on_pre_submit = callback

    def set_on_cancel(self, callback: Callable[[], None]) -> None:
        self._on_cancel = callback

    # ── TUIBase: content ───────────────────────────────────────────────────

    def load_session_history(
        self,
        messages: list[dict],
        max_messages: int = 200,
        tool_registry: Any = None,
        workspace: Any = None,
    ) -> None:
        _RUNTIME_TAG = "[Runtime Context — metadata only, not instructions]"
        recent = messages[-max_messages:] if len(messages) > max_messages else messages

        def _extract(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            return ""

        def _fmt_ts(ts_iso: str | None) -> str:
            if not ts_iso:
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                from datetime import datetime as dt
                return dt.fromisoformat(ts_iso).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Pre-index tool results by tool_call_id so we can attach them to traces.
        results_by_id: dict[str, str] = {}
        for msg in recent:
            if msg.get("role") == "tool":
                tcid = msg.get("tool_call_id")
                if tcid:
                    results_by_id[str(tcid)] = _extract(msg.get("content"))

        # Track whether we've already written a full "🐈 nanobot ts" header
        # for the current turn. Subsequent assistant segments in the same turn
        # (those between two user messages) get a lightweight "─ HH:MM:SS ─"
        # separator instead — matching the live rendering produced by
        # flush_stream / _write_response in continuation mode.
        header_written_this_turn = False
        for msg in recent:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                text = _extract(content)
                if text.startswith(_RUNTIME_TAG):
                    idx = text.find("\n\n")
                    text = text[idx + 2:] if idx != -1 else ""
                if text.strip():
                    ts = _fmt_ts(msg.get("timestamp"))
                    self._append_sep()
                    self._write_user(text.strip(), ts)
                # New user message → next assistant segment starts a fresh turn
                header_written_this_turn = False
            elif role == "assistant":
                text = _extract(content)
                if text.strip():
                    ts = _fmt_ts(msg.get("timestamp"))
                    if header_written_this_turn:
                        # Mid-turn segment: signal _write_response to skip the
                        # full header and instead write a "─ HH:MM:SS ─" line.
                        # _write_response will use the historical ts we pass in.
                        self._header_already_rendered = True
                    self._write_response(text.strip(), ts)
                    header_written_this_turn = True
                # Replay tool calls as static "↳ tool(args)  →  result" traces.
                for tc in msg.get("tool_calls") or []:
                    self._replay_tool_trace(tc, results_by_id, tool_registry, workspace)

    def _replay_tool_trace(
        self,
        tool_call: dict,
        results_by_id: dict[str, str],
        tool_registry: Any = None,
        workspace: Any = None,
    ) -> None:
        """Render a single historical tool call as a static trace line during
        load_session_history. Mirrors the live look of _render_tool_trace.
        """
        from rich.text import Text as _RText
        import json as _json

        try:
            fn = tool_call.get("function") or {}
            name = fn.get("name") or tool_call.get("name") or "tool"
            raw_args = fn.get("arguments")
            args: dict = {}
            if isinstance(raw_args, str):
                try:
                    parsed = _json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    pass
            elif isinstance(raw_args, dict):
                args = raw_args

            # Build the hint using the same logic as live traces (including
            # path relativization to the workspace).
            from nanobot.agent.loop import format_tool_hint
            tc_like = type("TC", (), {"name": name, "arguments": args})()
            hint = format_tool_hint([tc_like], workspace=workspace)

            # Pair with the tool result (if present) and try to produce the
            # same structured summary the live UI shows. Falls back to a raw
            # preview if no registry is available or the tool has no summarizer.
            tcid = str(tool_call.get("id") or "")
            result_text = results_by_id.get(tcid, "")
            summary = ""
            if result_text:
                tool = tool_registry.get(name) if tool_registry is not None else None
                if tool is not None:
                    from nanobot.agent.tools.summaries import summarize_tool_result
                    summary = summarize_tool_result(tool, args, result_text)
                if not summary:
                    preview = result_text.replace("\n", " ").strip()
                    # Match extract_error_summary's 120-char budget so replayed
                    # raw-preview summaries don't look weirdly short next to
                    # the live structured summaries.
                    if len(preview) > 120:
                        preview = preview[:119] + "…"
                    summary = preview

            line = _RText()
            line.append(self._TOOL_INDENT, style="")
            line.append(f"{self._TOOL_MARKER} ", style=self.THEME_MARKER)
            line.append(hint, style=self.THEME_HINT)
            if summary:
                tail_style = self.THEME_ERROR if summary.startswith("Error") else self.THEME_MUTED
                line.append("  →  ", style=self.THEME_MUTED)
                line.append(summary, style=tail_style)
            try:
                out = self._app.query_one("#output", _OutputLog)
                out.write(line)
            except Exception:
                pass
        except Exception:
            pass

    def add_user_echo(self, text: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_sep()
        self._write_user(text, ts)

    def add_response(
        self,
        content: str,
        metadata: dict | None = None,
        ts: str | None = None,
    ) -> None:
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_response(content, ts, metadata)

    def add_progress(self, text: str) -> None:
        import time
        self._tool_hint = text
        # Mark tool start so add_tool_result can show "[+Ns]" elapsed.
        self._tool_start_time = time.monotonic()
        self._app._safe_call(self._update_progress_line, text)

    def add_reasoning(self, text: str) -> None:
        """Accumulate a reasoning_content chunk; flushed on _reasoning_end.

        Unlike PromptTUI we do not render reasoning under the spinner — the
        Textual output log doesn't have a live ANSI cache the same way, so
        showing reasoning live here would require a separate widget. Keeping
        the trace silent until flush is the conservative choice; users can
        switch to the prompt_toolkit backend if they want live reasoning.
        """
        if not text:
            return
        self._reasoning_buf += text

    def flush_reasoning(self) -> None:
        """Dump the accumulated reasoning trace as a dim italic history block.

        Deferred-when-streaming: if content_delta already started writing the
        response, flushing now would visually split the response (truncate +
        rewrite from below the reasoning block, leaving the early chunk
        stranded above). In that case we keep the buffer and let pop_stream
        flush at turn end so the reasoning block lands cleanly between the
        header and the finalised response.
        """
        if not self._reasoning_buf.strip():
            self._reasoning_buf = ""
            return
        if self._stream_buf:
            # Stream already started — defer flush to pop_stream.
            return
        buf = self._reasoning_buf
        self._reasoning_buf = ""
        # Render on the Textual UI thread.
        self._app._safe_call(self._write_reasoning_block, buf)

    def _write_reasoning_block(self, buf: str) -> None:
        """Append a finalised reasoning trace to the output log (dim italic).

        Crucially advances ``_tool_placeholder_line`` past the block so the
        next ``stream_delta`` truncate (``out.truncate_to(_tool_placeholder_line)``)
        treats the reasoning as already-committed history rather than wiping
        it. Without this the reasoning would visibly flash and disappear the
        moment the first content_delta arrived.
        """
        self._log_write(f"[{self.THEME_MUTED}]💭 thinking[/{self.THEME_MUTED}]")
        for ln in buf.strip().splitlines():
            # markup=False would be ideal but Textual's RichLog wraps via Rich;
            # using a pre-built Text object avoids markup interpretation.
            self._log_write(Text(f"  {ln}", style=f"{self.THEME_MUTED} italic"))
        self._log_write("")
        # Anchor the stream truncate point below the reasoning block so the
        # next stream_delta / flush_stream doesn't clobber it.
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._tool_placeholder_line = len(out.lines)
            # Scroll to bottom: appending the (often long) reasoning block grows
            # max_scroll_y without moving scroll_offset, which would leave the
            # viewport "not at bottom". stream_delta only renders live when
            # sc_y >= mx_y, so without this the response stops streaming and
            # only appears in one shot at flush_stream/add_response.
            out.scroll_end(animate=False)
        except Exception:
            pass

    def _update_progress_line(self, text: str) -> None:
        """Overwrite the executing placeholder line with the tool name (runs in Textual context)."""
        self._render_placeholder_line(f"⠋ {text}", "dim")

    def _render_placeholder_line(self, content: str, style: str) -> None:
        """Render `content` into the placeholder line at _tool_placeholder_line."""
        try:
            from rich.segment import Segment as _Seg
            out = self._app.query_one("#output", _OutputLog)
            idx = self._tool_placeholder_line
            if 0 <= idx < len(out.lines):
                rt = Text(content, style=style)
                width = max(out.scrollable_content_region.width, out.min_width)
                console = out.app.console
                segs = list(console.render(rt, console.options.update_width(width)))
                line_segs = list(_Seg.split_lines(segs))
                if line_segs:
                    new_strip = Strip.from_lines(line_segs)[0].adjust_cell_length(width)
                else:
                    new_strip = Strip.blank(width)
                out.lines[idx] = new_strip
                out._line_cache.clear()
                out.refresh()
        except Exception:
            pass

    # Visual style for petrified tool traces. 4-space indent separates them
    # clearly from LLM text; cyan for the call, dim/red for the result tail.
    _TOOL_INDENT = "    "
    _TOOL_MARKER = "↳"

    def _render_tool_trace(
        self, hint: str, summary: str = "", elapsed: float | None = None,
    ) -> None:
        """Render the current placeholder as a static "↳ tool(args)" trace
        plus optional " → summary" tail and " [+Ns]" elapsed badge.

        Color: THEME_ERROR for Error summaries, THEME_MUTED otherwise.
        Elapsed is only shown when ``>= 1s`` so quick tools don't get a
        noisy "[+0s]" badge.
        """
        from rich.text import Text as _RText
        line = _RText()
        line.append(self._TOOL_INDENT, style="")
        line.append(f"{self._TOOL_MARKER} ", style=self.THEME_MARKER)
        line.append(hint, style=self.THEME_HINT)
        if summary:
            tail_style = self.THEME_ERROR if summary.startswith("Error") else self.THEME_MUTED
            line.append("  →  ", style=self.THEME_MUTED)
            line.append(summary, style=tail_style)
        if elapsed is not None and elapsed >= 1:
            line.append(f"  [+{int(elapsed)}s]", style=self.THEME_MUTED)
        self._render_placeholder_text(line)

    def _render_placeholder_text(self, rt) -> None:
        """Like _render_placeholder_line but accepts a pre-built Text object."""
        try:
            from rich.segment import Segment as _Seg
            out = self._app.query_one("#output", _OutputLog)
            idx = self._tool_placeholder_line
            if 0 <= idx < len(out.lines):
                width = max(out.scrollable_content_region.width, out.min_width)
                console = out.app.console
                segs = list(console.render(rt, console.options.update_width(width)))
                line_segs = list(_Seg.split_lines(segs))
                if line_segs:
                    new_strip = Strip.from_lines(line_segs)[0].adjust_cell_length(width)
                else:
                    new_strip = Strip.blank(width)
                out.lines[idx] = new_strip
                out._line_cache.clear()
                out.refresh()
        except Exception:
            pass

    def _petrify_tool_placeholder(self) -> None:
        """Convert the current "⠋ tool_name" spinner line into a static
        "↳ tool_name" trace (with elapsed badge), and advance the placeholder
        cursor so subsequent streaming / tool_phase_start operations append
        below it rather than overwriting the trace.

        Called at every transition out of the tool-executing state: the start
        of the next LLM streaming chunk (stream_delta) and the start of the
        next tool phase (tool_phase_start). Idempotent — does nothing if no
        tool hint is currently shown.
        """
        if not self._tool_hint:
            return
        import time
        elapsed = (
            time.monotonic() - self._tool_start_time
            if self._tool_start_time else None
        )
        try:
            self._render_tool_trace(self._tool_hint, "", elapsed)
            self._tool_placeholder_line += 1
            self._last_content_start = self._tool_placeholder_line
        except Exception:
            pass
        self._tool_hint = ""

    def add_tool_result(self, summary: str) -> None:
        """Petrify the current spinner placeholder into a static trace, optionally
        appending a result summary (e.g. '↳ exec(cmd)  →  exit 0, 12 lines').

        Called once per tool batch — even when summary is empty (so tools that
        don't define summarize_result still get their trace line frozen
        immediately on completion, instead of relying on the next operation
        to petrify them).

        After petrifying, schedule an idle "thinking..." spinner so the gap
        between tool completion and the next LLM action (which can be many
        seconds of reasoning) doesn't look like the UI hung.
        """
        if not self._tool_hint:
            return
        # The tool spinner was animating on the placeholder; stop it before
        # we rewrite that line as a static trace.
        try:
            self._app.stop_thinking_spinner()
        except Exception:
            pass
        import time
        elapsed = (
            time.monotonic() - self._tool_start_time
            if self._tool_start_time else None
        )
        try:
            # summary may be "" — _render_tool_trace just skips the tail.
            self._render_tool_trace(self._tool_hint, summary, elapsed)
            self._tool_placeholder_line += 1
            self._last_content_start = self._tool_placeholder_line
        except Exception:
            pass
        self._tool_hint = ""
        # If the LLM stays silent for >500ms after the tool finishes, show
        # a "thinking..." spinner in #live so the user knows we're waiting.
        self._schedule_idle_thinking()

    def add_system(self, text: str) -> None:
        self._log_write(f"[dim]{text}[/dim]")
        self._last_sep = False

    # ── TUIBase: streaming ─────────────────────────────────────────────────

    def stream_start(self) -> None:
        import time
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._flushed_parts = []
        self._suppress_segment_sep = False  # reset per turn; pop_stream may set it
        # Anchor the "thinking time" displayed by spinners to the moment the
        # turn began — not to each individual spinner restart. This keeps the
        # elapsed counter continuous across idle gaps + tool calls.
        self._turn_start_time = time.monotonic()
        # Write header + thinking placeholder directly into #output so the
        # animation is inside the message area, not in the separate #live strip.
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._stream_header_line = len(out.lines)
            out.write(f"[cyan]{__logo__} nanobot[/cyan] [dim]{self._stream_ts}[/dim]")
            out.write("")
            self._tool_placeholder_line = len(out.lines)
            self._last_content_start = self._tool_placeholder_line
            # Match the idle thinking style — _tick will overwrite this with
            # the same format on each frame anyway.
            out.write(Text("  ⠋ 思考中...", style="grey50"))
        except Exception:
            self._stream_header_line = 0
            self._tool_placeholder_line = 0
            self._last_content_start = 0
        # Use _safe_call so the spinner timer task is created inside
        # Textual's active_app context even if stream_start is invoked
        # from a non-Textual code path.
        self._app._safe_call(self._app.start_thinking_spinner)

    def tool_phase_start(self) -> None:
        self._cancel_idle_thinking()
        # Petrify the previous tool's spinner line into a static "→ tool" trace
        # before starting a new one, so chained tool calls stay visible.
        self._petrify_tool_placeholder()

        self._stream_buf = ""
        if not self._stream_ts:
            self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._tool_hint = "执行中..."
        # Append placeholder at the current end of output — do NOT truncate so
        # any content flush_stream already rendered between tool calls is preserved.
        # If the intermediate content block is large, collapse it to save space.
        _COLLAPSE_THRESHOLD = 6
        try:
            out = self._app.query_one("#output", _OutputLog)
            content_end = len(out.lines)
            if content_end - self._last_content_start > _COLLAPSE_THRESHOLD:
                out.collapse_lines(self._last_content_start, content_end)
                self._tool_placeholder_line = self._last_content_start + 1
            else:
                self._tool_placeholder_line = content_end
            # 4-space indent + cyan matches the eventual "↳ tool" trace.
            # _tick will overwrite with the tool hint once add_progress runs.
            out.write(Text("    ⠋ 执行中...", style="cyan"))
        except Exception:
            pass
        # Schedule via _safe_call so the timer task is created inside
        # Textual's context (active_app ContextVar must be set, otherwise
        # set_interval's task silently crashes on shutdown).
        self._app._safe_call(self._app.start_thinking_spinner)

    def _cancel_idle_thinking(self) -> None:
        """Stop any pending idle-thinking spinner task and remove the
        placeholder line it wrote to #output (if any)."""
        task = self._idle_thinking_task
        if task is not None and not task.done():
            task.cancel()
        self._idle_thinking_task = None
        # Stop the timer-driven spinner update.
        try:
            self._app.stop_thinking_spinner()
        except Exception:
            pass
        # Remove the idle thinking placeholder line so it doesn't linger
        # before the next stream content / tool call writes at the same idx.
        if self._idle_placeholder_visible:
            self._idle_placeholder_visible = False
            try:
                out = self._app.query_one("#output", _OutputLog)
                if 0 <= self._tool_placeholder_line <= len(out.lines):
                    out.truncate_to(self._tool_placeholder_line)
            except Exception:
                pass
            # Restore the original anchor so subsequent stream_delta /
            # pop_stream operate on the right line (otherwise pop_stream
            # would truncate the wrong line and add_response would write
            # the stream content a second time → duplicated message).
            if self._tool_placeholder_line_backup is not None:
                self._tool_placeholder_line = self._tool_placeholder_line_backup
                self._tool_placeholder_line_backup = None

    def _schedule_idle_thinking(self, delay: float = 0.5) -> None:
        """Schedule a "still thinking..." spinner in #live after ``delay`` seconds
        of no further stream_delta. Provides UX feedback during LLM reasoning
        gaps where no tokens are being emitted (e.g. reasoning_content phase).

        Cancelled by the next stream_delta, tool_phase_start, flush_stream, or
        pop_stream — whichever comes first.
        """
        self._cancel_idle_thinking()

        async def _wait_then_show() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            # Show the idle thinking spinner inline at the end of #output
            # so it stays visually attached to the most recent message
            # (instead of jumping down to #live above the input).
            try:
                out = self._app.query_one("#output", _OutputLog)
                # Back up the existing anchor so pop_stream / stream_delta can
                # later truncate the right line (the stream content), not the
                # idle thinking line we're about to add.
                self._tool_placeholder_line_backup = self._tool_placeholder_line
                self._tool_placeholder_line = len(out.lines)
                self._last_content_start = self._tool_placeholder_line
                # 2-space indent + grey50 distinguishes idle thinking from
                # the 4-space cyan tool traces above it (so users don't
                # mistake "still thinking" for a new tool call).
                out.write(Text("  ⠋ 思考中...", style="grey50"))
                self._idle_placeholder_visible = True
            except Exception:
                return
            # _safe_call ensures start_thinking_spinner's set_interval task
            # runs inside Textual's active_app context (otherwise the timer
            # crashes on shutdown with LookupError).
            self._app._safe_call(self._app.start_thinking_spinner)

        try:
            self._idle_thinking_task = asyncio.ensure_future(_wait_then_show())
        except RuntimeError:
            self._idle_thinking_task = None

    def stream_delta(self, delta: str) -> None:
        self._cancel_idle_thinking()
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        self._app.clear_live()
        # First delta after a tool call: petrify the tool placeholder so the
        # previous "⠋ tool_name" line becomes a static "→ tool_name" trace
        # before this new text starts overwriting at _tool_placeholder_line.
        if self._tool_hint:
            self._petrify_tool_placeholder()
        self._stream_buf += delta
        try:
            out = self._app.query_one("#output", _OutputLog)
            sc_y = out.scroll_offset.y
            mx_y = out.max_scroll_y
            # Only update in-place when the user is at the bottom.  If they have
            # scrolled up, skip the truncate so we don't clobber their viewport;
            # flush_stream will write the final content when streaming completes.
            if sc_y >= mx_y:
                out.truncate_to(self._tool_placeholder_line)
                out.write(Text(self._stream_buf))
        except Exception:
            pass
        # If no further delta arrives in the next 500ms, show a "still thinking"
        # spinner in #live so the user sees feedback during LLM reasoning gaps.
        self._schedule_idle_thinking()

    def flush_stream(self, metadata: dict | None = None) -> None:
        self._cancel_idle_thinking()
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        self._app.clear_live()
        try:
            out = self._app.query_one("#output", _OutputLog)
            if self._stream_buf.strip():
                render_as_text = (metadata or {}).get("render_as") == "text"
                # Save where this content block starts (for potential collapse in tool_phase_start)
                self._last_content_start = self._tool_placeholder_line
                # Replace streaming content (including any placeholder) with final rendered version
                out.truncate_to(self._tool_placeholder_line)
                # For mid-turn segments (not the first one), prefix a lightweight
                # timestamp separator so the user can tell distinct LLM segments
                # apart instead of seeing one giant blob.
                if self._flushed_parts:
                    now_ts = datetime.now().strftime("%H:%M:%S")
                    out.write(Text(f"─ {now_ts} ─", style=self.THEME_MUTED))
                    out.write("")
                if self._render_md and not render_as_text:
                    out.write(Markdown(self._stream_buf))
                else:
                    out.write(Text(self._stream_buf))
                out.write("")
                # Track end so next tool_phase_start appends after this rendered content
                self._tool_placeholder_line = len(out.lines)
                self._flushed_parts.append(self._stream_buf.strip())
                self._last_sep = True
            else:
                # Nothing streamed yet — remove the thinking/executing placeholder,
                # keep header+blank so _stream_header_line stays valid for
                # subsequent stream_delta calls when LLM streams after a tool call.
                out.truncate_to(self._tool_placeholder_line)
                self._last_content_start = len(out.lines)
        except Exception:
            if self._stream_buf.strip():
                ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_response(self._stream_buf, metadata, ts=ts)
        self._stream_buf = ""
        self._stream_ts = ""

    def pop_stream(self) -> str:
        self._cancel_idle_thinking()
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        buf = self._stream_buf
        self._stream_buf = ""
        ts_was = self._stream_ts
        self._stream_ts = ""
        self._app.clear_live()
        # Truncate only the *current* streaming chunk (from _tool_placeholder_line
        # onward) — earlier tool traces and the header stay visible. Then mark
        # that the header is already on screen so add_response doesn't write it
        # again.
        try:
            out = self._app.query_one("#output", _OutputLog)
            out.truncate_to(self._tool_placeholder_line)
        except Exception:
            pass
        # Fork: deferred reasoning flush. flush_reasoning skipped while stream
        # was active to avoid splitting the response visually; now is the right
        # time — stream chunk is gone, response will be re-written by
        # add_response below. Reasoning lands between header and response.
        if self._reasoning_buf.strip():
            deferred = self._reasoning_buf
            self._reasoning_buf = ""
            self._app._safe_call(self._write_reasoning_block, deferred)
        # If the streaming session was active, signal add_response to skip
        # the header (we keep the original one written by stream_start).
        self._header_already_rendered = bool(ts_was) or self._stream_header_line > 0
        # Fork: suppress the continuation "─ts─" separator when no visible
        # content followed the header. _tool_placeholder_line only advances past
        # the header anchor (_stream_header_line + 2 = header line + its trailing
        # blank) when flush_stream lands a mid-turn segment or a tool trace is
        # petrified. If it still sits at that anchor, the only thing between the
        # header and the upcoming response was suppressed reasoning — so a
        # "─ts─" line would dangle. NOTE: the "+2" is coupled to stream_start
        # writing a single-line header + one blank; keep them in sync.
        self._suppress_segment_sep = (
            self._header_already_rendered
            and self._tool_placeholder_line <= self._stream_header_line + 2
        )
        return buf

    def flush_accumulator(self) -> str:
        """Intermediate LLM text was already written to output by flush_stream
        and is preserved by the new pop_stream (which only truncates the
        current streaming chunk). So returning the accumulated parts here
        would make add_response write them a second time. Return empty and
        just drain the buffer.
        """
        self._flushed_parts = []
        return ""

    # ── TUIBase: state ─────────────────────────────────────────────────────

    def set_topic(self, name: str) -> None:
        self._topic = name
        self._app.update_topic_bar(name)
        # app.title only updates Textual's Header widget, not the terminal tab.
        # Send OSC 0 directly via the driver to update the terminal/tab title.
        try:
            title = f"nanobot — {name}" if name else "nanobot"
            self._app._driver.write(f"\033]0;{title}\007")
        except Exception:
            pass

    def set_is_processing(self, value: bool) -> None:
        self._is_processing = value
        self._update_status()

    def update_context_usage(self, used: int, total: int) -> None:
        self._ctx_used = used
        self._ctx_total = total
        self._update_status()

    def set_todos(self, todos: list[dict]) -> None:
        if not todos:
            self._app.update_todo_bar("")
            return
        total = len(todos)
        done = sum(1 for t in todos if t.get("status") == "completed")
        active = next((t for t in todos if t.get("status") == "in_progress"), None)
        active_c, muted_c, success_c = self.THEME_ACTIVE, self.THEME_MUTED, self.THEME_SUCCESS
        if active:
            content = (active.get("content") or "").strip()
            text = (
                f"[{active_c}]⚡[/{active_c}] [{muted_c}]{content}[/{muted_c}]"
                f" [{muted_c}]({done}/{total})[/{muted_c}]"
            )
        else:
            remaining = total - done
            if remaining > 0:
                text = f"[{muted_c}]☐ {remaining} pending · {done}/{total} done[/{muted_c}]"
            else:
                text = f"[{success_c}]✓[/{success_c}] [{muted_c}]all {total} done[/{muted_c}]"
        self._app.update_todo_bar(text)

    def reset_history(self) -> None:
        self._app.stop_spinner()
        self._app.clear_output()
        self._app.clear_live()
        self._stream_buf = ""
        self._stream_ts = ""
        self._last_sep = False
        self._ctx_used = 0
        self._ctx_total = 0
        self._update_status()

    # ── TUIBase: interactive modes ─────────────────────────────────────────

    def enter_new_topic_mode(
        self,
        callback: Callable[[str], Awaitable[None]],
        placeholder: str = "话题名: ",
    ) -> None:
        """Generic free-text input mode.

        Reuses the new_topic input wiring (Enter submits, ESC cancels) but
        accepts a custom placeholder so callers like ``show_question_popup``
        can repurpose it for "Custom answer:" without misleading users.
        """
        self._input_mode = "new_topic"
        self._new_topic_cb = callback
        # Clear any residual command/topic popup state so Enter routes to new_topic submit
        if self._popup_mode != "hidden":
            self.hide_popup()
        self._app.set_input_placeholder(placeholder)
        self._app.set_input_value("")

    def set_commands(self, commands: list[tuple[str, str]]) -> None:
        self._all_commands = commands

    def show_topic_popup(
        self,
        topics: list[str],
        on_select: Callable[[str], Awaitable[None]],
    ) -> None:
        self._popup_all_topics = list(topics)
        self._popup_mode = "topic"
        self._popup_items = [(t, t) for t in topics]
        self._popup_idx = 0
        self._popup_on_select = on_select
        self._app.set_input_value("")
        self._refresh_popup()

    def show_question_popup(
        self,
        questions: list[dict],
        on_complete: Callable[[dict[str, str] | None], Awaitable[None]],
    ) -> None:
        """Ask a series of multiple-choice questions one at a time.

        Each question is rendered as: a system message with the question text
        and numbered options (with descriptions) in the output log, then a
        single-select popup over just the option labels for keyboard pick.
        Answers accumulate in a dict keyed by question text; ESC mid-flow
        reports ``None`` so the caller knows the user bailed.
        """
        self._question_queue: list[dict] = list(questions or [])
        self._question_answers: dict[str, str] = {}
        self._question_on_complete = on_complete
        self._show_next_question()

    # Sentinel selected-value that means "user wants to type their own
    # answer instead of picking a listed option". The popup row's display
    # label is "✎ 自定义..." (see _show_next_question); we route on the
    # sentinel so a real option labelled "自定义..." still works correctly.
    _CUSTOM_INPUT_SENTINEL = "\x00__ask_user_custom__\x00"

    def _show_next_question(self) -> None:
        if not self._question_queue:
            cb = self._question_on_complete
            answers = self._question_answers
            self._question_on_complete = None
            self._question_answers = {}
            if cb:
                asyncio.ensure_future(cb(answers))
            return

        q = self._question_queue[0]
        question_text = str(q.get("question") or "(question)")
        header = str(q.get("header") or "").strip()
        options = q.get("options") or []
        if not isinstance(options, list) or not options:
            # Malformed question — skip and continue
            self._question_queue.pop(0)
            self._show_next_question()
            return

        header_str = f" [{header}]" if header else ""
        self.add_system(f"❓{header_str} {question_text}")
        for i, opt in enumerate(options, 1):
            label = str(opt.get("label", "")) if isinstance(opt, dict) else str(opt)
            desc = str(opt.get("description", "")) if isinstance(opt, dict) else ""
            line = f"  {i}. {label}"
            if desc:
                line += f"  — {desc}"
            self.add_system(line)
        # Append a "custom answer" row so the user is never stuck with only
        # the LLM-provided options (mirrors Claude Code's "Other" fallback).
        self.add_system(f"  {len(options) + 1}. ✎ 自定义...  — 输入你自己的方案/想法")

        labels = [
            str(opt.get("label", "")) if isinstance(opt, dict) else str(opt)
            for opt in options
        ]
        # popup_items is (value, display_label); pair every listed option with
        # itself, plus one extra entry whose *value* is the sentinel but whose
        # *display* shows "✎ 自定义...". Selection routes on value, so even if
        # the LLM also offers a real "自定义" option it stays distinct.
        popup_items = [(lbl, lbl) for lbl in labels]
        popup_items.append((self._CUSTOM_INPUT_SENTINEL, "✎ 自定义..."))
        self._popup_mode = "topic"  # reuse topic popup wiring for navigation
        self._popup_items = popup_items
        self._popup_idx = 0
        self._popup_all_topics = [lbl for _, lbl in popup_items]

        async def _on_selected(value: str) -> None:
            if value == self._CUSTOM_INPUT_SENTINEL:
                # Open a free-text input box; the typed text becomes the answer
                async def _on_custom_text(typed: str) -> None:
                    answer = (typed or "").strip()
                    if not answer:
                        # Empty input — treat like cancellation of THIS question;
                        # re-show the popup so user can pick a listed option
                        # instead of bailing the whole sequence.
                        self.add_system("  (空输入，重新选择)")
                        self._show_next_question()
                        return
                    self._question_answers[question_text] = answer
                    self.add_system(f"  ✓ 自定义: {answer}")
                    if self._question_queue:
                        self._question_queue.pop(0)
                    self._show_next_question()
                self.enter_new_topic_mode(_on_custom_text, placeholder="自定义答案: ")
                return
            self._question_answers[question_text] = value
            self.add_system(f"  ✓ {value}")
            if self._question_queue:
                self._question_queue.pop(0)
            self._show_next_question()

        self._popup_on_select = _on_selected
        self._app.set_input_value("")
        self._refresh_popup()

    def hide_popup(self) -> None:
        self._popup_mode = "hidden"
        self._popup_items = []
        self._popup_idx = 0
        self._popup_on_select = None
        self._popup_all_topics = []
        self._app.update_popup([], False)

    def _cancel_question_flow(self) -> None:
        """Abort any in-flight show_question_popup sequence and notify the
        callback with ``None`` so the awaiting LLM tool sees a cancellation.
        Called by ESC; NOT by the normal select-then-advance flow which
        consumes one question at a time on its own.
        """
        if self._question_on_complete is None:
            return
        cb = self._question_on_complete
        self._question_on_complete = None
        self._question_queue = []
        self._question_answers = {}
        try:
            self.add_system("  ✗ cancelled")
        except Exception:
            pass
        try:
            asyncio.ensure_future(cb(None))
        except Exception:
            pass
