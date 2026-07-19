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
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.cli.markdown import terminal_markdown
from nanobot.fork.cli.tui_base import TUIBase, input_history_path
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
    from textual.events import Key, Paste
    from textual.widgets import RichLog, Static, TextArea
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False
    App = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


_TUI_LAG_LOG = Path(__file__).resolve().parents[2] / "tui_lag.log"


def _compact_path_label(path: str, max_len: int = 48) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    normalized = text.replace("/", "\\")
    if len(normalized) <= max_len:
        return normalized
    p = Path(text)
    parts = p.parts
    if len(parts) >= 3:
        prefix = parts[0]
        tail = "\\".join(parts[-2:])
        compact = f"{prefix}...\\{tail}"
        if len(compact) <= max_len:
            return compact
    return "..." + normalized[-max(1, max_len - 3):]


def _normalize_topic_items(topics: list[str | tuple[str, str]]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for item in topics:
        if isinstance(item, tuple) and len(item) == 2:
            value, label = item
            items.append((str(value), str(label)))
        else:
            text = str(item)
            items.append((text, text))
    return items


def _append_tui_lag_log(message: str) -> None:
    try:
        _TUI_LAG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _TUI_LAG_LOG.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


if _TEXTUAL_AVAILABLE:
    import sys as _sys

    from rich.style import Style as _Style
    from textual.events import MouseDown, MouseMove, MouseScrollDown, MouseScrollUp, MouseUp
    from textual.geometry import Size
    from textual.strip import Strip

    class _OutputLog(RichLog):
        """RichLog with cell-range text selection and clipboard copy.

        Mouse drag selects text; releasing the mouse copies the selection to
        the system clipboard via Windows ``clip`` (primary) or Textual's OSC-52
        API (fallback).  Selected text is highlighted with a dark-blue tint.
        """
        can_focus = False

        def __init__(self, *, user_background: str = "#2d2d2d", **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._user_background = user_background
            if hasattr(self, "auto_scroll"):
                self.auto_scroll = False
            self._sel_start: tuple[int, int] | None = None  # content-space row, cell column
            self._sel_end: tuple[int, int] | None = None
            self._selecting: bool = False
            self._sel_moved: bool = False  # True once mouse moves during drag
            self._user_ranges: list[tuple[int, int]] = []  # gray-bg row ranges

            # RichLog stores only already-wrapped physical lines. Keep the original
            # renderables as logical records as well, so a terminal resize can
            # render them again at the new content width instead of retaining the
            # width that happened to be active when they were written.
            self._logical_records: list[dict[str, Any]] = []
            self._record_spans: list[tuple[int, int]] = []
            self._record_writes = True
            self._render_width = 0
            self._reflow_scheduled = False
            self._on_reflow: Callable[[list[int], list[int]], None] | None = None

        def is_at_bottom(self, threshold: int = 0) -> bool:
            """Return True when the viewport is at, or very near, the newest line."""
            try:
                return self.scroll_offset.y >= max(0, self.max_scroll_y - threshold)
            except Exception:
                return True

        def mark_user_scroll(self) -> None:
            """Compatibility hook for tests and callers that track manual scrolling."""
            return None

        def user_is_scrolling(self) -> bool:
            """Return True when the viewport is intentionally away from the bottom."""
            return not self.is_at_bottom()

        # ── selection helpers ──────────────────────────────────────────────
        def _selection_points(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
            if self._sel_start is None or self._sel_end is None:
                return None
            return (
                (self._sel_start, self._sel_end)
                if self._sel_start <= self._sel_end
                else (self._sel_end, self._sel_start)
            )

        def _sel_rows(self) -> tuple[int, int] | None:
            points = self._selection_points()
            if points is None:
                return None
            start, end = points
            return start[0], end[0]

        @staticmethod
        def _selection_cols_for_row(
            row: int,
            start: tuple[int, int],
            end: tuple[int, int],
            line_width: int,
        ) -> tuple[int, int] | None:
            start_row, start_col = start
            end_row, end_col = end
            if row < start_row or row > end_row:
                return None
            if start_row == end_row:
                col_start = min(start_col, line_width)
                col_end = min(end_col + 1, line_width)
            elif row == start_row:
                col_start = min(start_col, line_width)
                col_end = line_width
            elif row == end_row:
                col_start = 0
                col_end = min(end_col + 1, line_width)
            else:
                col_start = 0
                col_end = line_width
            if col_end <= col_start:
                return None
            return col_start, col_end

        def _clear_selection(self) -> None:
            self._sel_start = None
            self._sel_end = None
            self.refresh()

        def _extract_selected_text(self) -> str:
            points = self._selection_points()
            if points is None:
                return ""
            row_start, row_end = self._sel_rows() or (0, -1)
            parts: list[str] = []
            for row in range(row_start, row_end + 1):
                if row < len(self.lines):
                    line = self.lines[row]
                    cols = self._selection_cols_for_row(
                        row,
                        points[0],
                        points[1],
                        line.cell_length,
                    )
                    if cols is not None:
                        col_start, col_end = cols
                        parts.append(line.crop(col_start, col_end).text)
            return "\n".join(parts)

        def _copy_to_clipboard(self, text: str) -> None:
            copied = False
            if _sys.platform == "win32":
                try:
                    import ctypes
                    CF_UNICODETEXT = 13  # noqa: N806
                    GMEM_MOVEABLE = 0x0002  # noqa: N806
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
            for index, (record_start, record_end) in enumerate(self._record_spans):
                if record_start <= end and record_end > start:
                    self._logical_records[index]["user"] = True
            self._line_cache.clear()
            self.refresh()

        def write(
            self,
            content: Any,
            width: int | None = None,
            expand: bool = False,
            shrink: bool = True,
            scroll_end: bool | None = None,
            animate: bool = False,
            follow: bool | None = None,
        ) -> Any:
            """Write one logical renderable and retain it for resize reflow."""
            if follow is not None:
                scroll_end = follow
            should_follow = self.is_at_bottom() if scroll_end is None else scroll_end
            before = len(self.lines)
            if self._record_writes:
                saved = content.copy() if isinstance(content, Text) else content
                self._logical_records.append(
                    {
                        "content": saved,
                        "width": width,
                        "expand": expand,
                        "shrink": shrink,
                        "user": False,
                    }
                )
            result = super().write(
                content,
                width=width,
                expand=expand,
                shrink=shrink,
                scroll_end=False,
                animate=animate,
            )
            if self._record_writes and self._size_known:
                self._record_spans.append((before, len(self.lines)))
                self._render_width = self.scrollable_content_region.width
            if should_follow:
                self.scroll_end(animate=False, immediate=True, force=True)
            return result

        def clear(self) -> Any:
            if self._record_writes:
                self._logical_records.clear()
                self._record_spans.clear()
                self._user_ranges.clear()
            return super().clear()

        def on_resize(self, event: Any) -> None:
            # RichLog replays deferred writes the first time its size is known.
            # Those writes are already in _logical_records, so suppress recording.
            was_known = self._size_known
            self._record_writes = False
            try:
                super().on_resize(event)
            finally:
                self._record_writes = True
            if not was_known:
                self._rebuild_spans_from_current_lines()
                self._render_width = self.scrollable_content_region.width
                return
            if not self._reflow_scheduled:
                self._reflow_scheduled = True
                self.call_later(self._reflow_after_resize)

        def _reflow_after_resize(self) -> None:
            self._reflow_scheduled = False
            width = self.scrollable_content_region.width
            if width > 0 and width != self._render_width:
                self._reflow(width)

        def _rebuild_spans_from_current_lines(self) -> None:
            """Re-render records after deferred startup writes to recover spans."""
            if not self._logical_records:
                self._record_spans = []
                return
            self._reflow(max(1, self.scrollable_content_region.width), preserve_view=False)

        @staticmethod
        def _boundary_map(spans: list[tuple[int, int]], total: int) -> list[int]:
            return [start for start, _end in spans] + [total]

        def _reflow(self, width: int, *, preserve_view: bool = True) -> None:
            old_spans = list(self._record_spans)
            old_total = len(self.lines)
            old_boundaries = self._boundary_map(old_spans, old_total)
            was_at_bottom = self.is_at_bottom()
            old_top = int(self.scroll_offset.y)
            top_record = 0
            top_offset = 0
            if old_spans:
                top_record = len(old_spans) - 1
                for index, (start, end) in enumerate(old_spans):
                    if old_top < end:
                        top_record = index
                        top_offset = max(0, old_top - start)
                        break

            self._record_writes = False
            try:
                super().clear()
                self._record_spans = []
                self._user_ranges = []
                for record in self._logical_records:
                    start = len(self.lines)
                    # Explicitly use the current output width. This also makes
                    # Markdown/code blocks and wide CJK text reflow consistently.
                    super().write(
                        record["content"],
                        width=width,
                        expand=record["expand"],
                        shrink=record["shrink"],
                        scroll_end=False,
                    )
                    end = len(self.lines)
                    self._record_spans.append((start, end))
                    if record.get("user") and end > start:
                        self._user_ranges.append((start, end - 1))
            finally:
                self._record_writes = True
            self._render_width = width
            new_boundaries = self._boundary_map(self._record_spans, len(self.lines))
            if self._on_reflow is not None:
                self._on_reflow(old_boundaries, new_boundaries)
            self._clear_selection()
            if not preserve_view:
                return
            if was_at_bottom:
                self.scroll_end(animate=False, immediate=True, force=True)
            elif self._record_spans:
                start, end = self._record_spans[min(top_record, len(self._record_spans) - 1)]
                new_top = start + min(top_offset, max(0, end - start - 1))
                self.scroll_to(y=new_top, animate=False, immediate=True, force=True)

        def _record_index_at_line(self, line: int) -> int | None:
            for index, (start, end) in enumerate(self._record_spans):
                if start <= line < end:
                    return index
            return None

        def replace_line(self, line: int, content: Any) -> bool:
            """Replace a placeholder record while avoiding a full-history redraw."""
            from rich.segment import Segment

            index = self._record_index_at_line(line)
            if index is None:
                return False
            saved = content.copy() if isinstance(content, Text) else content
            self._logical_records[index]["content"] = saved
            start, end = self._record_spans[index]
            width = max(1, self.scrollable_content_region.width)
            renderable = self._make_renderable(content)
            segments = self.app.console.render(
                renderable,
                self.app.console.options.update_width(width),
            )
            rendered = Strip.from_lines(Segment.split_lines(segments))
            if len(rendered) == end - start:
                self.lines[start:end] = [strip.adjust_cell_length(width) for strip in rendered]
                self._line_cache.clear()
                self.refresh()
            else:
                self._reflow(width)
            return True

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

        @classmethod
        def _force_color_range(
            cls,
            strip: Strip,
            start: int,
            end: int,
            bgcolor: str,
            color: str,
        ) -> Strip:
            if end <= start:
                return strip
            before = strip.crop(0, start)
            selected = cls._force_colors(strip.crop(start, end), bgcolor, color)
            after = strip.crop(end, strip.cell_length)
            return Strip(
                [
                    *before._segments,
                    *selected._segments,
                    *after._segments,
                ],
                strip.cell_length,
            )

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
                strip = self._force_bgcolor(strip, self._user_background)
            points = self._selection_points()
            if points is not None:
                cols = self._selection_cols_for_row(
                    content_row,
                    points[0],
                    points[1],
                    self.lines[content_row].cell_length if content_row < n else 0,
                )
                if cols is not None:
                    scroll_x, _ = self.scroll_offset
                    col_start = max(0, cols[0] - scroll_x)
                    col_end = max(0, cols[1] - scroll_x)
                    strip = self._force_color_range(
                        strip,
                        col_start,
                        col_end,
                        bgcolor="white",
                        color="black",
                    )
            return strip

        # ── mouse events ───────────────────────────────────────────────────

        def on_mouse_down(self, event: MouseDown) -> None:
            if event.button != 1:  # left button only
                return
            try:
                n = len(self.lines)
                if n == 0:
                    return
                scroll_x, scroll_y = self.scroll_offset
                row = max(0, min(scroll_y + event.y, n - 1))
                col = max(0, scroll_x + event.x)
                self._sel_start = (row, col)
                self._sel_end = (row, col)
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
                scroll_x, scroll_y = self.scroll_offset
                row = max(0, min(scroll_y + event.y, len(self.lines) - 1))
                col = max(0, scroll_x + event.x)
                point = (row, col)
                if point != self._sel_end:
                    self._sel_moved = True
                    self._sel_end = point
                    self.refresh()
            except Exception:
                pass
            event.stop()

        def on_mouse_up(self, event: MouseUp) -> None:
            if not self._selecting:
                return
            try:
                scroll_x, scroll_y = self.scroll_offset
                row = max(0, min(scroll_y + event.y, len(self.lines) - 1))
                col = max(0, scroll_x + event.x)
                if (row, col) != self._sel_end:
                    self._sel_moved = True
                self._sel_end = (row, col)
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
            """Remove records at/after physical line boundary ``n``."""
            if n >= len(self.lines):
                return
            keep = 0
            line_boundary = 0
            for _start, end in self._record_spans:
                if end <= n:
                    keep += 1
                    line_boundary = end
                else:
                    break
            del self._logical_records[keep:]
            del self._record_spans[keep:]
            self.lines = self.lines[:line_boundary]
            self._user_ranges = [
                (start, min(end, line_boundary - 1))
                for start, end in self._user_ranges
                if start < line_boundary
            ]
            self._refresh_line_metrics()

        def remove_line(self, index: int) -> bool:
            """Remove the logical record containing a rendered placeholder line."""
            record_index = self._record_index_at_line(index)
            if record_index is None:
                return False
            del self._logical_records[record_index]
            self._reflow(max(1, self.scrollable_content_region.width))
            return True

        def _refresh_line_metrics(self) -> None:
            self._widest_line_width = (
                max(line.cell_length for line in self.lines) if self.lines else 0
            )
            self._line_cache.clear()
            self.virtual_size = Size(self._widest_line_width, len(self.lines))
            # Note: scroll_offset.y may briefly exceed new virtual height here.
            # render_line() clamps it so no blank lines appear during the repaint
            # triggered by virtual_size change (a reactive that fires immediately).
            self.refresh()

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
    class _ComposerInput(TextArea):
        """Input that delegates ↑/↓ to the TUI's history / popup state."""

        _LARGE_PASTE_CHARS = 5 * 1024

        def __init__(self, tui: "TextualTUI", **kwargs: Any) -> None:
            super().__init__("", soft_wrap=True, tab_behavior="focus", compact=True, **kwargs)
            self._tui_ref = tui
            self._multiline_paste_tokens: list[dict[str, str]] = []

        @property
        def value(self) -> str:
            return self.text

        @value.setter
        def value(self, text: str) -> None:
            self.text = text

        @property
        def cursor_position(self) -> int:
            return self._location_to_offset(self.cursor_location)

        @cursor_position.setter
        def cursor_position(self, offset: int) -> None:
            self.move_cursor(self._offset_to_location(offset))

        @staticmethod
        def _normalize_paste_text(text: str) -> str:
            return text.replace("\r\n", "\n").replace("\r", "\n")

        @staticmethod
        def _paste_line_count(text: str) -> int:
            return text.count("\n") + 1

        def _paste_display_token(self, text: str) -> str:
            lines = self._paste_line_count(text)
            if lines > 1:
                return f"[pasted {lines} lines]"
            return f"[pasted {len(text)} chars]"

        def _location_to_offset(self, location: tuple[int, int]) -> int:
            row, column = location
            lines = self.value.split("\n")
            row = max(0, min(row, len(lines) - 1))
            column = max(0, min(column, len(lines[row])))
            return sum(len(line) + 1 for line in lines[:row]) + column

        def _offset_to_location(self, offset: int) -> tuple[int, int]:
            text = self.value
            offset = max(0, min(offset, len(text)))
            row = 0
            remaining = offset
            for line in text.split("\n"):
                if remaining <= len(line):
                    return (row, remaining)
                remaining -= len(line) + 1
                row += 1
            lines = text.split("\n")
            return (len(lines) - 1, len(lines[-1]))

        def _clear_multiline_paste_tokens(self) -> None:
            self._multiline_paste_tokens = []

        def _token_positions(self) -> list[tuple[int, dict[str, str]]]:
            positions: list[tuple[int, dict[str, str]]] = []
            start = 0
            for token in self._multiline_paste_tokens:
                index = self.value.find(token["display"], start)
                if index < 0:
                    return []
                positions.append((index, token))
                start = index + len(token["display"])
            return positions

        def _insert_multiline_token(self, text: str) -> None:
            display = self._paste_display_token(text)
            selection = self.selection
            start_offset = self._location_to_offset(selection.start)
            end_offset = self._location_to_offset(selection.end)
            selection_start = min(start_offset, end_offset)
            selection_end = max(start_offset, end_offset)
            insert_at = selection_start
            before_positions = self._token_positions()
            result = self.replace(display, *selection)
            self.move_cursor(result.end_location)
            self.cursor_position = insert_at + len(display)

            new_token = {"display": display, "text": text}
            inserted = False
            tokens: list[dict[str, str]] = []
            for position, token in before_positions:
                token_end = position + len(token["display"])
                if token_end > selection_start and position < selection_end:
                    continue
                if not inserted and position >= insert_at:
                    tokens.append(new_token)
                    inserted = True
                tokens.append(token)
            if not inserted:
                tokens.append(new_token)
            self._multiline_paste_tokens = tokens

        def _insert_paste_text(self, text: str) -> None:
            text = self._normalize_paste_text(text)
            if not text:
                return
            if "\n" in text or len(text) > self._LARGE_PASTE_CHARS:
                self._insert_multiline_token(text)
                return
            selection = self.selection
            if selection.start == selection.end:
                result = self.insert(text)
            else:
                result = self.replace(text, *selection)
            self.move_cursor(result.end_location)

        def insert_paste_text(self, text: str) -> None:
            self._insert_paste_text(text)

        def submit_value(self) -> str:
            if not self._multiline_paste_tokens:
                return self.value
            positions = self._token_positions()
            if len(positions) != len(self._multiline_paste_tokens):
                return self.value
            restored: list[str] = []
            cursor = 0
            for index, token in positions:
                display = token["display"]
                restored.append(self.value[cursor:index])
                restored.append(token["text"])
                cursor = index + len(display)
            restored.append(self.value[cursor:])
            return "".join(restored)

        def sync_multiline_paste_tokens(self) -> None:
            if (
                self._multiline_paste_tokens
                and len(self._token_positions()) != len(self._multiline_paste_tokens)
            ):
                self._clear_multiline_paste_tokens()

        def _on_paste(self, event: Paste) -> None:
            self.insert_paste_text(event.text)
            event.prevent_default()
            event.stop()

        def action_paste(self) -> None:
            self.insert_paste_text(self.app.clipboard)

        async def _on_key(self, event: Key) -> None:
            tui = self._tui_ref
            if event.key in ("shift+enter", "shift_enter"):
                result = self.replace("\n", *self.selection)
                self.move_cursor(result.end_location)
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter":
                event.prevent_default()
                event.stop()
                self._tui_ref._app.submit_input(self)
                return
            if event.key not in ("up", "down", "tab"):
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
                event.stop()
            elif action == PopupAction.CYCLE_DOWN:
                tui._popup_idx = min(len(tui._popup_items) - 1, tui._popup_idx + 1)
                tui._refresh_popup()
                event.prevent_default()
                event.stop()
            elif action == PopupAction.COMPLETE:
                self._clear_multiline_paste_tokens()
                self.value = decision.value
                self.cursor_position = len(decision.value)
                event.prevent_default()
                event.stop()
            elif action == PopupAction.HISTORY_BACK:
                text = tui._history_backward()
                if text is not None:
                    self._clear_multiline_paste_tokens()
                    self.value = text
                    self.cursor_position = len(text)
                event.prevent_default()
                event.stop()
            elif action == PopupAction.HISTORY_FORWARD:
                text = tui._history_forward()
                self._clear_multiline_paste_tokens()
                self.value = text if text is not None else ""
                if text is not None:
                    self.cursor_position = len(text)
                event.prevent_default()
                event.stop()
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
            max-height: 6;
            min-height: 1;
            border: none;
            padding: 0 1;
            background: #0c0c0c;
        }
        TextArea {
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

        /* Optional glass skin. Use the terminal's default background rather than
           Textual's transparent color: a transparent root defers to App.render(),
           which paints the Textual theme background and hides terminal acrylic. */
        Screen.glass-skin,
        Screen.glass-skin #output,
        Screen.glass-skin #sep-row,
        Screen.glass-skin #sep,
        Screen.glass-skin #topic-bar,
        Screen.glass-skin #live,
        Screen.glass-skin #input,
        Screen.glass-skin TextArea,
        Screen.glass-skin #status,
        Screen.glass-skin #todo-bar {
            background: ansi_default;
        }
        Screen.glass-skin #output {
            overflow-x: hidden;
            scrollbar-size-horizontal: 0;
            scrollbar-background: ansi_default;
        }
        Screen.glass-skin TextArea .text-area--cursor-line {
            background: ansi_default;
        }
        Screen.glass-skin TextArea .text-area--cursor {
            color: black;
            background: white;
            text-style: none;
        }
        /* Keep transient overlays opaque so commands remain readable over
           arbitrary user-selected terminal background images. */
        Screen.glass-skin #popup {
            background: #0c0c0c;
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
            # Preserve ``ansi_default`` in glass mode. Textual normally converts
            # ANSI colors to truecolor, which turns the terminal-default canvas
            # back into the opaque theme background before terminal output.
            super().__init__(ansi_color=True if tui._skin_enabled else None)
            self._tui = tui
            self._spinner_timer: Any = None
            self._spinner_frame = 0
            self._spinner_start_time: float = 0.0
            self._thinking_timer: Any = None
            self._thinking_frame = 0
            self._thinking_start_time: float = 0.0
            self._lag_timer: Any = None
            self._lag_last: float = 0.0
            self._lag_warn_threshold_s = 0.5

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
            user_background = "default" if self._tui._skin_enabled else "#2d2d2d"
            output = _OutputLog(
                id="output",
                markup=True,
                highlight=False,
                wrap=True,
                user_background=user_background,
            )
            output._on_reflow = self._tui._remap_output_anchors
            yield output
            yield Static("", id="live")
            yield Static("", id="popup")
            yield Static("", id="todo-bar")
            with Horizontal(id="sep-row"):
                yield Static("[dim cyan]" + "─" * 80 + "[/dim cyan]", id="sep", markup=True)
                yield Static("", id="topic-bar")
            yield _ComposerInput(self._tui, placeholder="You: ", id="input")
            yield Static("", id="status")

        def on_mount(self) -> None:
            if self._tui._skin_enabled:
                self.screen.add_class("glass-skin")
            self.query_one("#input").focus()
            self._write_welcome()
            self.update_topic_bar(self._tui._workspace_label, self._tui._topic)
            self._start_lag_watchdog()
            # After the first full render, re-focus + full layout refresh so
            # Windows Terminal updates its IME candidate window position to the
            # actual input cursor location instead of defaulting to top-left.
            self.call_after_refresh(self._refocus_input)

        def _start_lag_watchdog(self) -> None:
            import time

            self._lag_last = time.monotonic()

            def _tick() -> None:
                now = time.monotonic()
                elapsed = now - self._lag_last
                self._lag_last = now
                lag = elapsed - 0.25
                if lag >= self._lag_warn_threshold_s:
                    message = (
                        f"{datetime.now().isoformat(timespec='seconds')} "
                        f"Textual event loop lag detected: {lag * 1000:.0f}ms; "
                        f"phase={self._tui._activity_phase}"
                    )
                    _append_tui_lag_log(message)
                    try:
                        logger.warning(
                            "Textual event loop lag detected: {:.0f}ms; phase={}",
                            lag * 1000,
                            self._tui._activity_phase,
                        )
                    except Exception:
                        pass

            self._lag_timer = self.set_interval(0.25, _tick)

        def _refocus_input(self) -> None:
            inp = self.query_one("#input", _ComposerInput)
            inp.focus()
            self.refresh(layout=True)

        def _write_welcome(self) -> None:
            out = self.query_one("#output", _OutputLog)
            tui = self._tui
            out.write(f"[cyan bold]{__logo__} nanobot[/cyan bold]  [dim]v{__version__}[/dim]"
                      + (f"  [dim]{tui._model}[/dim]" if tui._model else "")
                      + (f"  [dim]reasoning: {tui._reasoning_effort}[/dim]" if tui._reasoning_effort else ""))
            out.write("")
            out.write("[bold]快捷键[/bold]")
            out.write("  [cyan]PageUp / PageDown[/cyan]   滚动历史记录")
            out.write("  [cyan]↑ / ↓[/cyan]              切换输入历史")
            out.write("  [cyan]ESC[/cyan]                取消当前请求")
            out.write("  [cyan]Ctrl+C / Ctrl+D[/cyan]    退出")
            out.write("  [cyan]鼠标拖选[/cyan]            选中行后自动复制到剪贴板")
            out.write("")

        # ── input callbacks ────────────────────────────────────────────────

        def submit_input(self, input_widget: _ComposerInput) -> None:
            tui = self._tui
            input_text = input_widget.submit_value()
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
                input_text=input_text,
            ))
            action, value = decision.action, decision.value
            input_widget.sync_multiline_paste_tokens()
            input_widget._clear_multiline_paste_tokens()
            input_widget.value = ""
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
                    result = cb(value)
                    if isinstance(result, Awaitable):
                        asyncio.ensure_future(result)
                return
            if action == EnterAction.COMMAND_SUBMIT:
                tui.hide_popup()
                if value in tui._command_edit_values:
                    input_widget.value = value + " "
                    input_widget.move_cursor(input_widget._offset_to_location(len(input_widget.value)))
                    tui._on_input_changed(input_widget.value)
                    return
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

        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            input_widget = event.text_area
            if isinstance(input_widget, _ComposerInput):
                input_widget.sync_multiline_paste_tokens()
                self._tui._on_input_changed(input_widget.value)

        def on_paste(self, event: Paste) -> None:
            try:
                input_widget = self.query_one("#input", _ComposerInput)
            except Exception:
                return
            input_widget.insert_paste_text(event.text)
            input_widget.focus()
            event.prevent_default()
            event.stop()

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
            if not self.query_one("#input", _ComposerInput).value:
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
                        out.replace_line(idx, rt)
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

        def update_topic_bar(self, workspace: str, name: str) -> None:
            try:
                bar = self.query_one("#topic-bar", Static)
                label = Text()
                if workspace:
                    label.append(" ")
                    label.append(workspace, style="dim")
                if name:
                    if workspace:
                        label.append("  ·  ", style="dim")
                    else:
                        label.append(" ")
                    label.append(name, style="cyan")
                if workspace or name:
                    label.append(" ")
                bar.update(label)
                if workspace or name:
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
                self.query_one("#input", _ComposerInput).placeholder = text
            except Exception:
                pass

        def set_input_value(self, text: str) -> None:
            try:
                inp = self.query_one("#input", _ComposerInput)
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
        reasoning_effort: str | None = None,
        skin_enabled: bool = False,
        workspace: str | Path | None = None,
    ) -> None:
        if not _TEXTUAL_AVAILABLE:
            raise ImportError(
                "textual is required for the Textual TUI backend.\n"
                "Install it with:  pip install 'nanobot-ai[textual]'"
            )
        self._render_md = render_markdown
        self._history_base_file = Path(history_file) if history_file else None
        self._history_file: Path | None = None
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._skin_enabled = skin_enabled
        self._workspace_label = _compact_path_label(str(workspace or Path.cwd()))

        # Input history is bound to an internal session key by
        # set_input_history_topic(); no topic means no browsable history.
        self._history: list[str] = []
        self._history_pos: int = -1  # -1 = not navigating

        # Streaming state
        self._stream_buf: str = ""
        self._stream_ts: str = ""
        self._stream_header_line: int = 0  # output-log line index where stream header was written
        self._initial_thinking_placeholder_line: int | None = None
        self._tool_placeholder_line: int = 0  # output-log line index of the current thinking/executing placeholder
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
        self._initial_thinking_placeholder_visible: bool = False  # stream_start placeholder before visible output
        # When idle thinking is shown, _tool_placeholder_line is moved to its
        # line so the spinner updates that line. This backup preserves the
        # original (stream_delta / tool) anchor so cancel_idle can restore it
        # — without restoration, pop_stream would truncate the wrong line and
        # add_response would duplicate the streamed content.
        self._tool_placeholder_line_backup: int | None = None
        self._turn_start_time: float = 0.0  # monotonic timestamp when the current LLM turn started (stream_start)
        self._tool_start_time: float = 0.0  # monotonic timestamp when the current tool started (add_progress)
        self._stream_render_task: Any = None  # debounce task for live stream rendering
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
        self._activity_phase: str = "idle"
        self._ctx_used: int = 0
        self._ctx_total: int = 0
        self._input_mode: str = "chat"  # "chat" | "new_topic"
        self._new_topic_cb: Callable[[str], Awaitable[None]] | None = None

        # Popup state
        self._popup_max_visible = 6
        self._all_commands: list[tuple[str, str]] = []
        self._command_edit_values: set[str] = set()
        self._popup_mode: str = "hidden"
        self._popup_items: list[tuple[str, str]] = []
        self._popup_idx: int = 0
        self._popup_on_select: Callable[[str], Awaitable[None]] | None = None
        self._popup_all_topics: list[str] = []
        self._popup_all_items: list[tuple[str, str]] = []

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

    def _remap_output_anchors(self, old: list[int], new: list[int]) -> None:
        """Keep streaming placeholder/header anchors valid after output reflow."""
        if not old or len(old) != len(new):
            return

        def remap(value: int | None) -> int | None:
            if value is None:
                return None
            try:
                return new[old.index(value)]
            except ValueError:
                # Anchors should be logical-record boundaries. For defensive
                # compatibility, preserve their offset within the nearest record.
                record = max(0, min(len(old) - 2, next(
                    (i - 1 for i, boundary in enumerate(old) if boundary > value),
                    len(old) - 2,
                )))
                offset = max(0, value - old[record])
                return min(new[record] + offset, new[record + 1])

        self._stream_header_line = remap(self._stream_header_line) or 0
        self._tool_placeholder_line = remap(self._tool_placeholder_line) or 0
        self._initial_thinking_placeholder_line = remap(
            self._initial_thinking_placeholder_line
        )
        self._tool_placeholder_line_backup = remap(self._tool_placeholder_line_backup)

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
        self._activity_phase = "write_response"
        render_as = (metadata or {}).get("render_as")
        render_as_text = render_as == "text"
        render_as_error = render_as == "error"
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
            header_style = "red bold" if render_as_error else "cyan"
            self._log_write(f"[{header_style}]{__logo__} nanobot[/] [dim]{ts}[/dim]")
            self._log_write("")
        self._header_already_rendered = False  # consume flag
        self._suppress_segment_sep = False  # consume flag
        if render_as_error:
            self._log_write(Text(content, style=self.THEME_ERROR))
        elif self._render_md and not render_as_text and content.strip():
            self._log_write(terminal_markdown(content))
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

    def _popup_visible_range(self) -> tuple[int, int]:
        """Return the slice that keeps the selected popup item on screen."""
        count = len(self._popup_items)
        if not count:
            return 0, 0
        visible = self._popup_max_visible
        start = max(0, self._popup_idx - visible + 1)
        if start + visible > count:
            start = max(0, count - visible)
        return start, min(count, start + visible)

    def _refresh_popup(self) -> None:
        items = self._popup_items
        idx = self._popup_idx
        mode = self._popup_mode
        if not items or mode == "hidden":
            self._app.update_popup([], False)
            return
        start, end = self._popup_visible_range()
        lines: list[str] = []
        if start:
            lines.append(f"[dim]  ↑ 还有 {start} 项[/dim]")
        for i in range(start, end):
            value, label = items[i]
            selected = i == idx
            prefix = " ▶ " if selected else "   "
            if selected:
                if mode == "command":
                    lines.append(f"[reverse]{prefix}{value:<12}  {label}[/reverse]")
                else:
                    lines.append(f"[reverse]{prefix}{label or value}[/reverse]")
            else:
                if mode == "command":
                    lines.append(f"[dim]{prefix}{value:<12}  {label}[/dim]")
                else:
                    lines.append(f"[dim]{prefix}{label or value}[/dim]")
        if end < len(items):
            lines.append(f"[dim]  ↓ 还有 {len(items) - end} 项[/dim]")
        self._app.update_popup(lines, True)

    def _on_input_changed(self, text: str) -> None:
        """Called on every keystroke — update popup."""
        if self._input_mode == "new_topic":
            return
        if self._popup_mode == "topic":
            query = text.lower()
            if query:
                source = self._popup_all_items or [(t, t) for t in self._popup_all_topics]
                filtered = [
                    (value, label)
                    for value, label in source
                    if query in label.lower()
                ]
            else:
                filtered = list(self._popup_all_items) or [(t, t) for t in self._popup_all_topics]
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
        _RUNTIME_TAG = "[Runtime Context — metadata only, not instructions]"  # noqa: N806
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
        import json as _json

        from rich.text import Text as _RText

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
            # path relativization to the workspace). Import from the real module
            # — loop.py only imports it function-locally as _fork_fmt, so
            # `from nanobot.agent.loop import format_tool_hint` raised ImportError
            # and (silently swallowed below) wiped every replayed tool trace.
            from nanobot.fork.utils.tool_hints import format_tool_hint
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
                logger.debug("replay tool trace: write to #output failed", exc_info=True)
        except Exception:
            # Log instead of silently swallowing — a swallowed ImportError here
            # is exactly what hid the lost-tool-trace bug for so long.
            logger.debug("replay tool trace: render failed", exc_info=True)

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
        self._clear_initial_thinking_placeholder()
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
            if out.is_at_bottom():
                out.scroll_end(animate=False)
        except Exception:
            pass

    def _update_progress_line(self, text: str) -> None:
        """Overwrite the executing placeholder line with the tool name (runs in Textual context)."""
        self._render_placeholder_line(f"⠋ {text}", "dim")

    def _render_placeholder_line(self, content: str, style: str) -> None:
        """Render `content` into the placeholder record at _tool_placeholder_line."""
        try:
            out = self._app.query_one("#output", _OutputLog)
            out.replace_line(self._tool_placeholder_line, Text(content, style=style))
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

    def _render_placeholder_text(self, rt: Text) -> None:
        """Like _render_placeholder_line but accepts a pre-built Text object."""
        try:
            out = self._app.query_one("#output", _OutputLog)
            out.replace_line(self._tool_placeholder_line, rt)
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
        except Exception:
            pass
        self._tool_hint = ""
        # If the LLM stays silent for >500ms after the tool finishes, show
        # a "thinking..." spinner in #live so the user knows we're waiting.
        self._schedule_idle_thinking()

    _FILE_DIFF_VISIBLE_LINES = 120
    _FILE_DIFF_HUNK_RE = re.compile(
        r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
    )

    def add_file_edit_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        rendered = [self._format_file_edit_event(event) for event in events]
        rendered = [block for block in rendered if block is not None]
        if not rendered:
            return
        self._petrify_tool_placeholder()
        for block in rendered:
            self._write_file_edit_block(block)
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._tool_placeholder_line = len(out.lines)
        except Exception:
            pass
        self._schedule_idle_thinking()

    def _format_file_edit_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        phase = event.get("phase")
        status = event.get("status")
        if phase not in {"end", "error"} and status not in {"done", "error"}:
            return None
        path = str(event.get("path") or event.get("absolute_path") or "").strip()
        if not path and event.get("pending"):
            path = "(pending file)"
        if not path:
            return None
        added = int(event.get("added") or 0)
        deleted = int(event.get("deleted") or 0)
        raw_diff = event.get("diff")
        if isinstance(event.get("diff_text"), str):
            diff = event["diff_text"]
        elif isinstance(raw_diff, dict):
            diff = raw_diff.get("text") if isinstance(raw_diff.get("text"), str) else ""
        else:
            diff = raw_diff if isinstance(raw_diff, str) else ""
        total_lines = int(event.get("diff_total_lines") or 0)
        if total_lines <= 0 and diff:
            total_lines = len(diff.splitlines())
        return {
            "path": path,
            "added": max(0, added),
            "deleted": max(0, deleted),
            "status": "error" if status == "error" or phase == "error" else "done",
            "binary": bool(event.get("binary")),
            "error": str(event.get("error") or ""),
            "diff": diff,
            "diff_total_lines": max(0, total_lines),
            "diff_truncated": bool(event.get("diff_truncated")),
        }

    def _write_file_edit_block(self, block: dict[str, Any]) -> None:
        from rich.text import Text as _RText

        added = block["added"]
        deleted = block["deleted"]
        status = block["status"]
        path = block["path"]
        header = _RText()
        header.append(self._TOOL_INDENT, style="")
        if status == "error":
            header.append("✗ ", style=self.THEME_ERROR)
        header.append(path, style=self.THEME_HINT)
        if block["binary"]:
            header.append("  binary", style=self.THEME_MUTED)
        else:
            header.append(" (", style=self.THEME_MUTED)
            header.append(f"+{added}", style="green")
            header.append(" ", style="")
            header.append(f"-{deleted}", style="red")
            header.append(")", style=self.THEME_MUTED)
        if block["error"]:
            header.append(f"  {block['error']}", style=self.THEME_ERROR)
        self._log_write(header)

        diff = block["diff"]
        if not diff:
            return
        lines = self._number_file_diff_lines(diff)
        visible = lines[: self._FILE_DIFF_VISIBLE_LINES]
        width = max((len(str(number)) for number, _line in visible), default=1)
        for number, line in visible:
            text = _RText()
            text.append(self._TOOL_INDENT + "  ", style="")
            text.append(f"{number:>{width}} ", style=self.THEME_MUTED)
            if line.startswith("+"):
                text.append(line, style="green")
            elif line.startswith("-"):
                text.append(line, style="red")
            else:
                text.append(line, style=self.THEME_MUTED)
            self._log_write(text)
        hidden = max(0, len(lines) - len(visible))
        if hidden or block["diff_truncated"]:
            suffix = "，diff 已截断" if block["diff_truncated"] else ""
            self._log_write(
                Text(
                    f"{self._TOOL_INDENT}  ... 已折叠 {hidden} 行{suffix}",
                    style=self.THEME_MUTED,
                )
            )

    @classmethod
    def _number_file_diff_lines(cls, diff: str) -> list[tuple[int, str]]:
        """Convert unified diff hunks to one readable file-line-number column."""
        numbered: list[tuple[int, str]] = []
        old_line = 0
        new_line = 0
        in_hunk = False
        for line in diff.splitlines():
            match = cls._FILE_DIFF_HUNK_RE.match(line)
            if match is not None:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
                in_hunk = True
                continue
            if not in_hunk or line.startswith("\\ No newline at end of file"):
                continue
            if line.startswith("-"):
                numbered.append((old_line, line))
                old_line += 1
            elif line.startswith("+"):
                numbered.append((new_line, line))
                new_line += 1
            else:
                numbered.append((new_line, line))
                old_line += 1
                new_line += 1
        return numbered

    def add_system(self, text: str) -> None:
        self._log_write(Text(text, style="dim"))
        self._last_sep = False

    # ── TUIBase: streaming ─────────────────────────────────────────────────

    def stream_start(self) -> None:
        import time
        self._activity_phase = "stream_start"
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
            self._initial_thinking_placeholder_line = self._tool_placeholder_line
            # Match the idle thinking style — _tick will overwrite this with
            # the same format on each frame anyway.
            out.write(Text("  ⠋ 思考中...", style="grey50"))
            self._initial_thinking_placeholder_visible = True
            out.scroll_end(animate=False)
        except Exception:
            self._stream_header_line = 0
            self._initial_thinking_placeholder_line = None
            self._tool_placeholder_line = 0
        # Use _safe_call so the spinner timer task is created inside
        # Textual's active_app context even if stream_start is invoked
        # from a non-Textual code path.
        self._app._safe_call(self._app.start_thinking_spinner)

    def tool_phase_start(self) -> None:
        self._clear_initial_thinking_placeholder()
        self._activity_phase = "tool_phase"
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
        # Fork: intermediate-content folding removed — mid-turn text now stays
        # fully expanded (the collapse/expand machinery in _OutputLog was deleted).
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._tool_placeholder_line = len(out.lines)
            # 4-space indent + cyan matches the eventual "↳ tool" trace.
            # _tick will overwrite with the tool hint once add_progress runs.
            out.write(Text("    ⠋ 执行中...", style="cyan"))
        except Exception:
            pass
        # Schedule via _safe_call so the timer task is created inside
        # Textual's context (active_app ContextVar must be set, otherwise
        # set_interval's task silently crashes on shutdown).
        self._app._safe_call(self._app.start_thinking_spinner)

    def _clear_initial_thinking_placeholder(self) -> None:
        """Remove the stream-start thinking line once visible activity replaces it."""
        if not self._initial_thinking_placeholder_visible:
            return
        self._initial_thinking_placeholder_visible = False
        initial_line = self._initial_thinking_placeholder_line
        self._initial_thinking_placeholder_line = None
        try:
            # A later idle/tool spinner may be actively animating at another
            # line. Stop the timer only while it still owns the initial line.
            if initial_line == self._tool_placeholder_line:
                self._app.stop_thinking_spinner()
            out = self._app.query_one("#output", _OutputLog)
            if initial_line is not None and out.remove_line(initial_line):
                if self._tool_placeholder_line > initial_line:
                    self._tool_placeholder_line -= 1
                if (
                    self._tool_placeholder_line_backup is not None
                    and self._tool_placeholder_line_backup > initial_line
                ):
                    self._tool_placeholder_line_backup -= 1
        except Exception:
            pass

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

    def clear_initial_thinking(self) -> None:
        """Remove only the stream-start thinking placeholder.

        Todo plans are visible progress, so they replace the initial waiting
        state. Do not cancel a later idle-thinking spinner: it represents the
        model's active post-tool deliberation and should remain visible.
        """
        self._clear_initial_thinking_placeholder()

    def clear_idle_thinking(self) -> None:
        """Remove the stale idle-thinking placeholder before visible progress.

        System/todo progress messages are already user-visible activity. Keeping
        the previous "思考中" line below them makes the UI look stuck, and the
        elapsed counter can restart on a second placeholder.
        """
        self._cancel_idle_thinking()
    def stop_thinking(self) -> None:
        """TUIBase hook: stop the idle/thinking spinner on turn completion so it
        never outlives the turn. The non-streaming reply path has no pop_stream
        to stop it, so the idle spinner scheduled after the last tool call would
        otherwise spin (and keep counting) forever.
        """
        self._cancel_idle_thinking()
        try:
            self._app.stop_spinner()
        except Exception:
            pass

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
                if self._tool_placeholder_line_backup is None:
                    self._tool_placeholder_line_backup = self._tool_placeholder_line
                self._tool_placeholder_line = len(out.lines)
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

    def _cancel_stream_render(self) -> None:
        task = self._stream_render_task
        if task is not None and not task.done():
            task.cancel()
        self._stream_render_task = None

    def _render_stream_live(self) -> None:
        try:
            out = self._app.query_one("#output", _OutputLog)
            if out.user_is_scrolling() and not out.is_at_bottom():
                return
            if not out.is_at_bottom():
                return
            out.truncate_to(self._tool_placeholder_line)
            out.write(Text(self._stream_buf))
        except Exception:
            pass

    def _schedule_stream_render(self, delay: float = 0.075) -> None:
        task = self._stream_render_task
        if task is not None and not task.done():
            return

        async def _wait_then_render() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            self._stream_render_task = None
            self._app._safe_call(self._render_stream_live)

        try:
            self._stream_render_task = asyncio.ensure_future(_wait_then_render())
        except RuntimeError:
            self._stream_render_task = None

    def stream_delta(self, delta: str) -> None:
        self._clear_initial_thinking_placeholder()
        self._activity_phase = "stream_delta"
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
        self._schedule_stream_render()
        # Keep normal token streaming stable, but still surface long provider
        # silences such as "generating a large tool_call argument". The delay is
        # intentionally much longer than the render debounce so small chunks do
        # not get separated by a thinking line.
        self._schedule_idle_thinking(delay=2.0)

    def flush_stream(self, metadata: dict | None = None) -> None:
        self._clear_initial_thinking_placeholder()
        self._activity_phase = "flush_stream"
        self._cancel_idle_thinking()
        self._cancel_stream_render()
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        self._app.clear_live()
        try:
            out = self._app.query_one("#output", _OutputLog)
            if self._stream_buf.strip():
                render_as_text = (metadata or {}).get("render_as") == "text"
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
                    out.write(terminal_markdown(self._stream_buf))
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
        except Exception:
            if self._stream_buf.strip():
                ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_response(self._stream_buf, metadata, ts=ts)
        self._stream_buf = ""
        self._stream_ts = ""

    def pop_stream(self) -> str:
        self._cancel_idle_thinking()
        self._cancel_stream_render()
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
        self._app.update_topic_bar(self._workspace_label, name)
        title = f"nanobot — {name}" if name else "nanobot"
        # Keep Textual's public title state in sync. Its driver owns the
        # terminal tab title lifecycle, so writing only a raw OSC sequence is
        # not sufficient on every backend.
        self._app.title = title
        # Some terminals do not refresh their tab from Textual's title change
        # while the application is already mounted. Keep the OSC write as a
        # compatibility fallback.
        try:
            self._app._driver.write(f"\033]0;{title}\007")
        except Exception:
            pass

    def set_input_history_topic(self, topic_key: str) -> None:
        self._history_file = input_history_path(self._history_base_file, topic_key)
        self._history = self._load_history()
        self._history_pos = -1
        try:
            input_widget = self._app.query_one("#input", _ComposerInput)
            input_widget._clear_multiline_paste_tokens()
            input_widget.value = ""
        except Exception:
            pass

    def set_is_processing(self, value: bool) -> None:
        self._is_processing = value
        if not value:
            self._activity_phase = "idle"
        self._update_status()

    def set_activity_phase(self, phase: str) -> None:
        self._activity_phase = phase or "idle"

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
        placeholder: str = "会话名: ",
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

    def set_commands(self, commands: list[tuple[str, str] | tuple[str, str, str]]) -> None:
        normalized: list[tuple[str, str]] = []
        edit_values: set[str] = set()
        for item in commands:
            command, description = item[0], item[1]
            action = item[2] if len(item) > 2 else "submit"
            normalized.append((command, description))
            if action == "edit":
                edit_values.add(command)
        self._all_commands = normalized
        self._command_edit_values = edit_values

    def show_topic_popup(
        self,
        topics: list[str | tuple[str, str]],
        on_select: Callable[[str], Awaitable[None]],
    ) -> None:
        items = _normalize_topic_items(topics)
        self._popup_all_topics = [value for value, _ in items]
        self._popup_all_items = items
        self._popup_mode = "topic"
        self._popup_items = list(self._popup_all_items)
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
        self._popup_all_items = list(popup_items)

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
        self._popup_all_items = []
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
