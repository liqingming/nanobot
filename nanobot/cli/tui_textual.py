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
from nanobot.cli.tui_base import TUIBase

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
            if event.key == "up":
                if tui._popup_mode != "hidden" and tui._popup_items:
                    tui._popup_idx = max(0, tui._popup_idx - 1)
                    tui._refresh_popup()
                else:
                    text = tui._history_backward()
                    if text is not None:
                        self.value = text
                        self.cursor_position = len(text)
                event.prevent_default()

            elif event.key == "down":
                if tui._popup_mode != "hidden" and tui._popup_items:
                    tui._popup_idx = min(len(tui._popup_items) - 1, tui._popup_idx + 1)
                    tui._refresh_popup()
                else:
                    text = tui._history_forward()
                    self.value = text if text is not None else ""
                    if text is not None:
                        self.cursor_position = len(text)
                event.prevent_default()

            elif event.key == "tab":
                if tui._popup_mode == "command" and tui._popup_items:
                    value, _ = tui._popup_items[tui._popup_idx]
                    self.value = value
                    self.cursor_position = len(value)
                    event.prevent_default()

            elif event.key == "enter":
                if tui._popup_mode != "hidden" and tui._popup_items:
                    tui._popup_enter_handled = True
                    value, _ = tui._popup_items[tui._popup_idx]
                    cb = tui._popup_on_select  # save before hide_popup() clears it
                    tui.hide_popup()
                    self.value = ""
                    if cb:
                        await cb(value)
                    event.prevent_default()

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
            self._thinking_timer: Any = None
            self._thinking_frame = 0

        def compose(self) -> ComposeResult:
            yield _OutputLog(id="output", markup=True, highlight=False, wrap=True)
            yield Static("", id="live")
            yield Static("", id="popup")
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
            # Enter was consumed by popup selection in _HistoryInput._on_key
            if tui._popup_enter_handled:
                tui._popup_enter_handled = False
                return
            text = event.value
            event.input.clear()
            tui._history_pos = -1
            if tui._input_mode == "new_topic":
                cb = tui._new_topic_cb
                tui._exit_new_topic_mode()
                if cb:
                    asyncio.ensure_future(cb(text.strip()))
                return
            if text.strip():
                tui._add_to_history(text)
            if tui._on_pre_submit and not tui._is_processing:
                tui._on_pre_submit(text)
            if tui._on_submit:
                asyncio.ensure_future(tui._on_submit(text))

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
            self._stop_spinner()
            self._spinner_frame = 0
            self._spinner_tool = tool

            def _tick() -> None:
                self._spinner_frame += 1
                tui = self._tui
                frames = _TOOL_SPINNER if self._spinner_tool else _SPINNER
                icon = frames[self._spinner_frame % len(frames)]
                if self._spinner_tool:
                    label = tui._tool_hint or "executing..."
                    ts = tui._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    text = f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]\n[dim]{icon} {label}[/dim]"
                else:
                    ts = tui._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    text = f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]\n[dim]{icon} thinking...[/dim]"
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

        def stop_spinner(self) -> None:
            self._stop_spinner()

        # ── thinking spinner (animates inside #output) ─────────────────────

        def start_thinking_spinner(self) -> None:
            self._stop_thinking_spinner()
            self._thinking_frame = 0

            def _tick() -> None:
                from rich.segment import Segment as _Seg
                self._thinking_frame += 1
                icon = _SPINNER[self._thinking_frame % len(_SPINNER)]
                try:
                    out = self.query_one("#output", _OutputLog)
                    idx = self._tui._tool_placeholder_line
                    if 0 <= idx < len(out.lines):
                        hint = self._tui._tool_hint
                        label = hint if hint else "思考中..."
                        rt = Text(f"{icon} {label}", style="dim")
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
        self._last_sep: bool = False

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
        self._popup_enter_handled: bool = False  # set before hide_popup so on_input_submitted can skip

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
        """Write a completed response block as Rich objects (no ANSI conversion)."""
        render_as_text = (metadata or {}).get("render_as") == "text"
        self._log_write(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
        self._log_write("")
        if self._render_md and not render_as_text and content.strip():
            self._log_write(Markdown(content))
        else:
            self._log_write(Text(content))
        self._log_write("")
        self._last_sep = True

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

    def load_session_history(self, messages: list[dict], max_messages: int = 200) -> None:
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
            elif role == "assistant":
                text = _extract(content)
                if text.strip():
                    ts = _fmt_ts(msg.get("timestamp"))
                    self._write_response(text.strip(), ts)

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
        self._tool_hint = text
        self._app.call_later(self._update_progress_line, text)

    def _update_progress_line(self, text: str) -> None:
        """Overwrite the executing placeholder line with the tool name (runs in Textual context)."""
        try:
            from rich.segment import Segment as _Seg
            out = self._app.query_one("#output", _OutputLog)
            idx = self._tool_placeholder_line
            if 0 <= idx < len(out.lines):
                rt = Text(f"⠋ {text}", style="dim")
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

    def add_system(self, text: str) -> None:
        self._log_write(f"[dim]{text}[/dim]")
        self._last_sep = False

    # ── TUIBase: streaming ─────────────────────────────────────────────────

    def stream_start(self) -> None:
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._flushed_parts = []
        # Write header + thinking placeholder directly into #output so the
        # animation is inside the message area, not in the separate #live strip.
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._stream_header_line = len(out.lines)
            out.write(f"[cyan]{__logo__} nanobot[/cyan] [dim]{self._stream_ts}[/dim]")
            out.write("")
            self._tool_placeholder_line = len(out.lines)
            self._last_content_start = self._tool_placeholder_line
            out.write(Text("⠋ 思考中...", style="dim"))
        except Exception:
            self._stream_header_line = 0
            self._tool_placeholder_line = 0
            self._last_content_start = 0
        self._app.start_thinking_spinner()

    def tool_phase_start(self) -> None:
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
            out.write(Text("⠋ 执行中...", style="dim"))
        except Exception:
            pass
        # schedule via call_later so the timer task is created inside Textual's context
        # (active_app ContextVar must be set, otherwise set_interval's task silently crashes)
        self._app.call_later(self._app.start_thinking_spinner)

    def stream_delta(self, delta: str) -> None:
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        self._app.clear_live()
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

    def flush_stream(self, metadata: dict | None = None) -> None:
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
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        buf = self._stream_buf
        self._stream_buf = ""
        self._stream_ts = ""
        self._app.clear_live()
        # Remove the streaming block (header + content) so add_response can
        # write a clean final version without duplicating content.
        try:
            out = self._app.query_one("#output", _OutputLog)
            out.truncate_to(self._stream_header_line)
        except Exception:
            pass
        return buf

    def flush_accumulator(self) -> str:
        """Return and clear all intermediate LLM text flushed between tool calls."""
        parts = [p for p in self._flushed_parts if p]
        self._flushed_parts = []
        return "\n\n".join(parts)

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

    def enter_new_topic_mode(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._input_mode = "new_topic"
        self._new_topic_cb = callback
        self._app.set_input_placeholder("话题名: ")
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

    def hide_popup(self) -> None:
        self._popup_mode = "hidden"
        self._popup_items = []
        self._popup_idx = 0
        self._popup_on_select = None
        self._popup_all_topics = []
        self._app.update_popup([], False)
