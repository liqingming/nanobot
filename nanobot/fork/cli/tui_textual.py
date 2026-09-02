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
import hashlib
import json
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
from nanobot.fork.cli.tui_base import TUIBase, input_history_path, recent_complete_turns
from nanobot.fork.cli.tui_keys import (
    EnterAction,
    PopupAction,
    TUIState,
    decide_enter_action,
    decide_popup_key,
)
from nanobot.utils.session_runtime_log import append_session_runtime_log

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
    from textual.geometry import Region, Size
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
            # Message blocks retain logical-record references, so bookmark
            # targets remain valid when resize reflows physical output lines.
            self._message_blocks: list[dict[str, Any]] = []
            self._active_message_block: dict[str, Any] | None = None
            self._bookmark_highlight: tuple[int, int] | None = None
            self._bookmark_entries: list[dict[str, Any]] = []
            self._bookmark_lines: set[int] = set()
            self._user_navigation_id: str | None = None
            self._record_writes = True
            self._render_width = 0
            self._reflow_scheduled = False
            self._on_reflow: Callable[[list[int], list[int]], None] | None = None
            self._on_top_reached: Callable[[], None] | None = None

        def is_at_bottom(self, threshold: int = 0) -> bool:
            """Return True when the viewport is at, or very near, the newest line."""
            try:
                return self.scroll_offset.y >= max(0, self.max_scroll_y - threshold)
            except Exception:
                return True

        def watch_scroll_y(self, old_value: float, new_value: float) -> None:
            super().watch_scroll_y(old_value, new_value)
            if new_value <= 0 < old_value:
                self.call_later(self._notify_top_if_needed)

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
            self._sel_moved = False
            self.refresh()

        def selected_bookmark_line(self) -> int | None:
            """返回拖选终点行；单击或空选区不作为书签目标。"""
            if not self._sel_moved or self._sel_end is None:
                return None
            return self._sel_end[0] if self._extract_selected_text().strip() else None

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
                if self._active_message_block is not None:
                    self._active_message_block["records"].append(self._logical_records[-1])
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
                self._message_blocks.clear()
                self._active_message_block = None
                self._bookmark_highlight = None
                self._user_navigation_id = None
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

        def _reflow(
            self,
            width: int,
            *,
            preserve_view: bool = True,
            notify_reflow: bool = True,
        ) -> None:
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
            self._rebuild_bookmark_lines()
            new_boundaries = self._boundary_map(self._record_spans, len(self.lines))
            if notify_reflow and self._on_reflow is not None:
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
            if (
                self._bookmark_highlight is not None
                and self._bookmark_highlight[0] <= content_row <= self._bookmark_highlight[1]
            ):
                strip = self._force_bgcolor(strip, "#665500")
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

        _BOOKMARK_MARKER = "◆"

        @classmethod
        def _draw_bookmark_marker(cls, strip: Strip, column: int) -> Strip:
            """Draw a marker over one padding cell without touching content."""
            from rich.segment import Segment

            if not 0 <= column < strip.cell_length:
                return strip
            cell = strip.crop(column, column + 1)
            background = (
                cell._segments[0].style.bgcolor
                if cell._segments and cell._segments[0].style
                else None
            )
            marker = Segment(
                cls._BOOKMARK_MARKER,
                _Style(color="#ffd75f", bgcolor=background, bold=True),
            )
            return Strip(
                [
                    *strip.crop(0, column)._segments,
                    marker,
                    *strip.crop(column + 1, strip.cell_length)._segments,
                ],
                strip.cell_length,
            )

        def render_lines(self, crop: Region) -> list[Strip]:
            """Render bookmark markers in the dedicated left padding column."""
            strips = super().render_lines(crop)
            marker_x = self.styles.gutter.left - self.styles.padding.left
            if not crop.x <= marker_x < crop.right:
                return strips
            marker_column = marker_x - crop.x
            scroll_y = int(self.scroll_offset.y)
            content_top = self.styles.gutter.top
            for index, strip in enumerate(strips):
                content_y = crop.y + index - content_top
                if content_y < 0:
                    continue
                if self._line_has_bookmark_marker(scroll_y + content_y):
                    strips[index] = self._draw_bookmark_marker(strip, marker_column)
            return strips

        def _mouse_content_point(
            self, event: MouseDown | MouseMove | MouseUp
        ) -> tuple[int, int]:
            """Map widget-relative mouse coordinates to RichLog content coordinates."""
            offset = event.get_content_offset_capture(self)
            scroll_x, scroll_y = self.scroll_offset
            return int(scroll_y + offset.y), int(scroll_x + offset.x)

        # ── mouse events ───────────────────────────────────────────────────

        def on_mouse_down(self, event: MouseDown) -> None:
            if event.button != 1:  # left button only
                return
            try:
                n = len(self.lines)
                if n == 0:
                    return
                content_row, content_col = self._mouse_content_point(event)
                row = max(0, min(content_row, n - 1))
                col = max(0, content_col)
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
                content_row, content_col = self._mouse_content_point(event)
                row = max(0, min(content_row, len(self.lines) - 1))
                col = max(0, content_col)
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
                content_row, content_col = self._mouse_content_point(event)
                row = max(0, min(content_row, len(self.lines) - 1))
                col = max(0, content_col)
                if (row, col) != self._sel_end:
                    self._sel_moved = True
                self._sel_end = (row, col)
                self._selecting = False
                self.release_mouse()
                # 拖选只保留高亮；复制统一由 Ctrl+C 触发。
                if self._sel_moved and self._extract_selected_text().strip():
                    self.refresh()
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

        def begin_message(self, message_id: str, role: str, summary: str) -> None:
            block = {
                "id": message_id,
                "role": role,
                "summary": summary,
                "records": [],
            }
            self._message_blocks.append(block)
            self._active_message_block = block

        def end_message(self) -> None:
            self._active_message_block = None

        def message_blocks(self) -> list[dict[str, Any]]:
            return list(self._message_blocks)

        def reset_user_navigation(self) -> None:
            self._user_navigation_id = None

        def user_message_targets(self) -> list[tuple[str, int]]:
            """Return loaded user messages in display order with current start lines."""
            targets: list[tuple[str, int]] = []
            for block in self._message_blocks:
                if block.get("role") != "user":
                    continue
                message_id = str(block.get("id", ""))
                line = self.message_start_line(message_id)
                if message_id and line is not None:
                    targets.append((message_id, line))
            return targets

        def jump_user_message(self, direction: int) -> int | None:
            """Jump to the previous/next loaded user message without wrapping."""
            targets = self.user_message_targets()
            if not targets or direction == 0:
                return None
            ids = [message_id for message_id, _line in targets]
            if self._user_navigation_id in ids:
                current = ids.index(self._user_navigation_id)
                target_index = current - 1 if direction < 0 else current + 1
            else:
                viewport_top = int(self.scroll_offset.y)
                if direction < 0:
                    candidates = [
                        index for index, (_message_id, line) in enumerate(targets)
                        if line < viewport_top
                    ]
                    if not candidates:
                        return None
                    target_index = candidates[-1]
                else:
                    candidates = [
                        index for index, (_message_id, line) in enumerate(targets)
                        if line > viewport_top
                    ]
                    if not candidates:
                        return None
                    target_index = candidates[0]
            if not 0 <= target_index < len(targets):
                return None
            message_id, line = targets[target_index]
            self._user_navigation_id = message_id
            self.scroll_to(y=line, animate=False, immediate=True, force=True)
            return line

        def message_start_line(self, message_id: str) -> int | None:
            for block in self._message_blocks:
                if block["id"] != message_id:
                    continue
                record_ids = {id(record) for record in block["records"]}
                for index, record in enumerate(self._logical_records):
                    if id(record) in record_ids and index < len(self._record_spans):
                        return self._record_spans[index][0]
                return None
            return None

        @staticmethod
        def _anchor_text_length(text: str) -> int:
            """使用非空白字符计数，使锚点不受终端自动换行影响。"""
            return sum(not char.isspace() for char in text)

        def bookmark_anchor_at_line(self, line: int) -> dict[str, Any] | None:
            """把物理行转换成消息内稳定锚点。"""
            for block in self._message_blocks:
                record_ids = {id(record): index for index, record in enumerate(block["records"])}
                for global_index, record in enumerate(self._logical_records):
                    record_index = record_ids.get(id(record))
                    if record_index is None or global_index >= len(self._record_spans):
                        continue
                    start, end = self._record_spans[global_index]
                    if not start <= line < end:
                        continue
                    char_offset = sum(
                        self._anchor_text_length(self.lines[row].text)
                        for row in range(start, line)
                    )
                    line_text = " ".join(self.lines[line].text.split())
                    return {
                        "message_id": block["id"],
                        "record_index": record_index,
                        "char_offset": char_offset,
                        "role": block["role"],
                        "summary": line_text or block["summary"],
                    }
            return None

        def _bookmark_record_span(
            self, bookmark: dict[str, Any]
        ) -> tuple[int, int] | None:
            message_id = str(bookmark.get("message_id", ""))
            try:
                record_index = int(bookmark.get("record_index", 0))
            except (TypeError, ValueError):
                return None
            for block in self._message_blocks:
                if block["id"] != message_id or not 0 <= record_index < len(block["records"]):
                    continue
                target_record = block["records"][record_index]
                for global_index, record in enumerate(self._logical_records):
                    if record is target_record and global_index < len(self._record_spans):
                        return self._record_spans[global_index]
            return None

        def bookmark_line(self, bookmark: dict[str, Any]) -> int | None:
            """在当前排版中解析消息内书签锚点。"""
            try:
                target_offset = max(0, int(bookmark.get("char_offset", 0)))
            except (TypeError, ValueError):
                return None
            span = self._bookmark_record_span(bookmark)
            if span is None:
                return None
            start, end = span
            consumed = 0
            for row in range(start, end):
                row_length = self._anchor_text_length(self.lines[row].text)
                if row_length and target_offset < consumed + row_length:
                    return row
                consumed += row_length
            return max(start, end - 1) if end > start else None

        def bookmark_context_summary(
            self, bookmark: dict[str, Any], *, limit: int = 120
        ) -> str | None:
            """动态生成带相邻行语境的书签摘要，兼容旧书签。"""
            line = self.bookmark_line(bookmark)
            span = self._bookmark_record_span(bookmark)
            if line is None or span is None:
                return None
            start, end = span

            def normalized(row: int) -> str:
                return " ".join(self.lines[row].text.split())

            before = [
                text
                for row in range(max(start, line - 4), line)
                if (text := normalized(row))
            ][-1:]
            target = normalized(line) or "书签位置"
            after = [
                text
                for row in range(line + 1, min(end, line + 5))
                if (text := normalized(row))
            ][:1]

            def clipped(text: str, size: int) -> str:
                return text if len(text) <= size else text[: size - 1].rstrip() + "…"

            parts = [f"【{clipped(target, 48)}】"]
            if before:
                parts.append(f"前：{clipped(before[0], 30)}")
            if after:
                parts.append(f"后：{clipped(after[0], 30)}")
            summary = "  ·  ".join(parts)
            return summary if len(summary) <= limit else summary[: limit - 1].rstrip() + "…"

        def set_bookmarks(self, bookmarks: list[dict[str, Any]]) -> None:
            self._bookmark_entries = list(bookmarks)
            self._rebuild_bookmark_lines()
            self.refresh()

        def _rebuild_bookmark_lines(self) -> None:
            self._bookmark_lines = {
                line
                for bookmark in self._bookmark_entries
                if (line := self.bookmark_line(bookmark)) is not None
            }

        def _line_has_bookmark_marker(self, line: int) -> bool:
            return line in self._bookmark_lines

        def message_line_range(self, message_id: str) -> tuple[int, int] | None:
            for block in self._message_blocks:
                if block["id"] != message_id:
                    continue
                record_ids = {id(record) for record in block["records"]}
                spans = [
                    self._record_spans[index]
                    for index, record in enumerate(self._logical_records)
                    if id(record) in record_ids and index < len(self._record_spans)
                ]
                if spans:
                    return spans[0][0], max(end - 1 for _start, end in spans)
                return None
            return None

        def flash_bookmark(self, bookmark: dict[str, Any]) -> None:
            line = self.bookmark_line(bookmark)
            self._bookmark_highlight = (line, line) if line is not None else None
            self.refresh()
            if self._bookmark_highlight is not None:
                self.set_timer(1.2, self.clear_bookmark_highlight)

        def clear_bookmark_highlight(self) -> None:
            self._bookmark_highlight = None
            self.refresh()

        def message_at_line(self, line: int) -> dict[str, Any] | None:
            candidate: dict[str, Any] | None = None
            candidate_start = -1
            for block in self._message_blocks:
                start = self.message_start_line(block["id"])
                if start is not None and start <= line and start >= candidate_start:
                    candidate = block
                    candidate_start = start
            return candidate

        def record_marker(self) -> int:
            """Return an opaque marker for records written after this point."""
            return len(self._logical_records)

        def records_since(self, marker: int) -> list[dict[str, Any]]:
            """Return stable references to logical records written after ``marker``."""
            return list(self._logical_records[max(0, marker):])

        def prepend_recent_records(
            self,
            record_marker: int,
            block_marker: int,
            *,
            previous_top: int | None = None,
        ) -> int:
            """Move records just appended by replay to the front and preserve the viewport."""
            new_records = self._logical_records[record_marker:]
            new_blocks = self._message_blocks[block_marker:]
            if not new_records:
                return 0
            old_records = self._logical_records[:record_marker]
            old_blocks = self._message_blocks[:block_marker]
            old_top = int(self.scroll_offset.y) if previous_top is None else previous_top
            self._logical_records = new_records + old_records
            self._message_blocks = new_blocks + old_blocks
            self._reflow(
                max(1, self.scrollable_content_region.width),
                preserve_view=False,
                notify_reflow=False,
            )
            inserted_lines = (
                self._record_spans[len(new_records)][0]
                if len(self._record_spans) > len(new_records)
                else len(self.lines)
            )
            self.scroll_to(
                y=old_top + inserted_lines,
                animate=False,
                immediate=True,
                force=True,
            )
            return inserted_lines

        def _notify_top_if_needed(self) -> None:
            if self.scroll_offset.y <= 0 and self._on_top_reached is not None:
                self._on_top_reached()

        def remove_records(self, records: list[dict[str, Any]]) -> None:
            """Remove selected logical records while preserving later records."""
            record_ids = {id(record) for record in records}
            if not record_ids:
                return
            kept = [record for record in self._logical_records if id(record) not in record_ids]
            if len(kept) == len(self._logical_records):
                return
            self._logical_records = kept
            self._message_blocks = [
                block
                for block in self._message_blocks
                if any(id(record) not in record_ids for record in block["records"])
            ]
            # The caller resets the active stream anchor after removal, so avoid
            # remapping anchors against a deliberately non-contiguous deletion.
            self._reflow(
                max(1, self.scrollable_content_region.width),
                notify_reflow=False,
            )

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
            self.scroll_relative(y=-3, animate=False)
            self.call_later(self._notify_top_if_needed)

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
            padding: 0 0 0 1;
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
            height: 1;
            padding: 0 1;
            background: #0c0c0c;
            color: $text-muted;
            display: none;
        }
        #live.visible {
            display: block;
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
            Binding("ctrl+up", "previous_user_message", show=False, priority=True),
            Binding("ctrl+down", "next_user_message", show=False, priority=True),
            Binding("ctrl+b", "toggle_bookmark", show=False, priority=True),
            Binding("f6", "previous_bookmark", show=False, priority=True),
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
            self._live_layout_generation = 0
            self._lag_timer: Any = None
            self._lag_last: float = 0.0
            self._lag_warn_threshold_s = 0.5

        async def _check_bindings(self, key: str, priority: bool = False) -> bool:
            """在 Textual 查找绑定前记录书签快捷键。"""
            normalized = key.lower()
            if normalized == "f6" or (
                "b" in normalized and any(modifier in normalized for modifier in ("ctrl", "alt"))
            ):
                self._tui._bookmark_runtime_log(
                    "tui.bookmark.key",
                    key=key,
                    priority=priority,
                    focused=type(self.focused).__name__ if self.focused is not None else None,
                )
            return await super()._check_bindings(key, priority)

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
            out.write("  [cyan]Ctrl+↑ / Ctrl+↓[/cyan]    跳转上一条/下一条用户消息")
            out.write("  [cyan]↑ / ↓[/cyan]              切换输入历史")
            out.write("  [cyan]ESC[/cyan]                取消当前请求")
            out.write("  [cyan]Ctrl+B[/cyan]             添加/删除当前位置书签")
            out.write("  [cyan]F6[/cyan]                 跳转上一个书签")
            out.write("  [cyan]Ctrl+C / Ctrl+D[/cyan]    复制选区或退出")
            out.write("  [cyan]鼠标拖选[/cyan]            选择要复制的文本")
            out.write("")
            out.write("[bold]常用命令[/bold]")
            out.write("  [cyan]/rename · /resume · /clear[/cyan]    重命名、恢复、清空话题")
            out.write("  [cyan]/todos · /continue[/cyan]             查看待办、继续当前计划")
            out.write("  [cyan]/bookmarks · /bookmarks-clear[/cyan]  查看、清空书签")
            out.write("  [cyan]/model · /status[/cyan]                模型与运行状态")
            out.write("  [cyan]/system-prompt[/cyan]                  查看当前话题的系统提示词")
            out.write("  [cyan]/skin [参数][/cyan]                    切换 Windows Terminal 背景图")
            out.write("    [dim]无参数：打开背景图选择列表[/dim]")
            out.write("    [dim]list：列出背景图；next / prev：上一张 / 下一张[/dim]")
            out.write("    [dim]random：随机切换；编号 / 文件名：切换到指定背景图[/dim]")
            out.write("  [cyan]/exit[/cyan]                           退出")
            out.write("  [dim]输入 / 可查看完整命令列表[/dim]")
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
            if action == EnterAction.NOOP:
                # Empty Enter is a navigation shortcut: jump back to the newest
                # message without submitting or touching input history.
                self.query_one("#output", _OutputLog).scroll_end(
                    animate=False,
                    immediate=True,
                    force=True,
                )

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
                        out._clear_selection()
                        return
            except Exception:
                pass
            self.exit()

        def action_toggle_bookmark(self) -> None:
            self._tui._bookmark_runtime_log("tui.bookmark.action", action="toggle_bookmark")
            self._tui.toggle_bookmark_at_view()

        def action_previous_bookmark(self) -> None:
            self._tui._bookmark_runtime_log("tui.bookmark.action", action="previous_bookmark")
            self._tui.jump_to_previous_bookmark()

        def action_eof_app(self) -> None:
            if not self.query_one("#input", _ComposerInput).value:
                self.exit()

        def action_escape_app(self) -> None:
            tui = self._tui
            if tui._is_processing and tui._on_cancel:
                # During an active turn ESC always means interrupt, even if a command,
                # topic, bookmark, or ask-user popup currently owns the visual focus.
                if tui._popup_mode != "hidden":
                    tui._cancel_question_flow()
                    tui.hide_popup()
                asyncio.create_task(tui._on_cancel())
            elif tui._popup_mode != "hidden":
                tui._cancel_question_flow()
                tui.hide_popup()
            elif tui._input_mode == "new_topic":
                tui._exit_new_topic_mode()

        def action_page_up(self) -> None:
            out = self.query_one("#output", _OutputLog)
            out.reset_user_navigation()
            out.scroll_relative(y=-10, animate=False)
            out.call_later(out._notify_top_if_needed)

        def action_page_down(self) -> None:
            out = self.query_one("#output", _OutputLog)
            out.reset_user_navigation()
            out.scroll_relative(y=10)

        def action_previous_user_message(self) -> None:
            self.query_one("#output", _OutputLog).jump_user_message(-1)

        def action_next_user_message(self) -> None:
            self.query_one("#output", _OutputLog).jump_user_message(1)

        def on_mouse_scroll_up(self) -> None:
            out = self.query_one("#output", _OutputLog)
            out.reset_user_navigation()
            out.scroll_relative(y=-3)

        def on_mouse_scroll_down(self) -> None:
            out = self.query_one("#output", _OutputLog)
            out.reset_user_navigation()
            out.scroll_relative(y=3)

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

        # ── thinking/tool spinner (single transient #live line) ────────────

        def start_thinking_spinner(self) -> None:
            """Animate the single transient status line outside output history."""
            import time
            self._stop_thinking_spinner()
            # A queued start may run after its phase was already cancelled.
            if not self._tui._live_activity_visible:
                return
            self._thinking_frame = 0
            self._thinking_start_time = time.monotonic()

            def _tick() -> None:
                if not self._tui._live_activity_visible:
                    self._stop_thinking_spinner()
                    return
                self._thinking_frame += 1
                icon = _SPINNER[self._thinking_frame % len(_SPINNER)]
                hint = self._tui._tool_hint
                if hint:
                    base = self._tui._tool_start_time or self._thinking_start_time
                    suffix = self._elapsed_suffix(base)
                    rt = Text(f"    {icon} {hint}{suffix}", style="cyan")
                else:
                    base = self._tui._model_wait_start_time or self._thinking_start_time
                    suffix = self._elapsed_suffix(base)
                    rt = Text(f"  {icon} 模型处理中 · 已等待{suffix or ' (0s)'}", style="grey50")
                self.update_live(rt)

            self._thinking_timer = self.set_interval(0.1, _tick)

        def _stop_thinking_spinner(self) -> None:
            if self._thinking_timer is not None:
                self._thinking_timer.stop()
                self._thinking_timer = None

        def stop_thinking_spinner(self) -> None:
            self._stop_thinking_spinner()

        # ── live area helpers ──────────────────────────────────────────────

        def _change_live_visibility(self, live: Static, visible: bool) -> None:
            """Change #live layout while preserving an existing bottom anchor."""
            currently_visible = live.has_class("visible")
            if currently_visible == visible:
                return
            try:
                output = self.query_one("#output", _OutputLog)
                was_at_bottom = output.is_at_bottom()
            except Exception:
                output = None
                was_at_bottom = False
            self._live_layout_generation += 1
            generation = self._live_layout_generation
            live.set_class(visible, "visible")
            if not was_at_bottom or output is None:
                return

            def _restore_bottom_anchor() -> None:
                if generation != self._live_layout_generation:
                    return
                output.scroll_end(animate=False, immediate=True, force=True)

            # display:none/block changes #output's height only after layout.
            self.call_after_refresh(_restore_bottom_anchor)

        def update_live(self, content: Any) -> None:
            try:
                live = self.query_one("#live", Static)
                live.update(content)
                self._change_live_visibility(live, True)
            except Exception:
                pass

        def clear_live(self) -> None:
            try:
                live = self.query_one("#live", Static)
                live.update("")
                self._change_live_visibility(live, False)
            except Exception:
                pass

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
                # Like #live, showing or hiding the todo row changes #output's
                # height. Preserve the bottom anchor unless the user has
                # intentionally scrolled into history.
                self._change_live_visibility(bar, bool(text))
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
        show_tool_preface: bool = True,
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
        self._show_tool_preface = show_tool_preface
        self._workspace = Path(workspace or Path.cwd())
        self._workspace_label = _compact_path_label(str(self._workspace))

        # Input history is bound to an internal session key by
        # set_input_history_topic(); no topic means no browsable history.
        self._history: list[str] = []
        self._history_pos: int = -1  # -1 = not navigating

        # Topic-local persistent bookmarks live beside the existing input
        # history file, keeping this fork feature independent of session schema.
        bookmark_base = self._history_base_file or (Path.home() / ".nanobot" / "history")
        self._bookmark_dir = bookmark_base.parent / f"{bookmark_base.name}.bookmarks"
        self._bookmark_file: Path | None = None
        self._bookmarks: dict[str, dict[str, Any]] = {}
        self._message_occurrences: dict[tuple[str, str], int] = {}
        self._bookmark_popup_ids: list[str] = []
        self._bookmark_notice_task: Any = None
        self._runtime_log_path: Path | None = None

        # Streaming state
        self._stream_buf: str = ""
        self._stream_ts: str = ""
        self._stream_header_line: int = 0  # output-log line index where stream header was written
        self._tool_placeholder_line: int = 0  # output-log boundary of the current live stream chunk
        self._flushed_parts: list[str] = []  # intermediate LLM text flushed between tool calls
        # A flushed segment becomes temporary only after actual tool progress
        # confirms that it accompanied a tool call. Length continuations and
        # retries therefore remain permanent.
        self._pending_tool_text_records: list[dict[str, Any]] = []
        self._temporary_tool_text_records: list[dict[str, Any]] = []
        self._tool_hint: str = ""
        self._active_tool_events: list[dict[str, Any]] = []
        # Accumulates reasoning_content chunks (LLM thinking trace) so we can
        # flush them as a dim italic history block on _reasoning_end. Mirrors
        # PromptTUI's behavior; required because reasoning models like
        # DeepSeek-v4-pro otherwise have no place to land their trace.
        self._reasoning_buf: str = ""
        self._last_sep: bool = False
        # One persistent state owns the assistant turn header for both live output
        # and history replay. Every assistant artifact calls
        # _ensure_assistant_turn_header() before writing; a user message resets it.
        self._assistant_turn_header_rendered: bool = False
        # Fork: when a turn's header is on screen but NOTHING visible followed
        # it (reasoning suppressed + no tool trace + no mid-turn flush), the
        # continuation "─ts─" separator in _write_response would dangle with
        # nothing to separate. pop_stream sets this so _write_response skips it.
        self._suppress_segment_sep: bool = False
        self._idle_thinking_task: Any = None  # delayed post-tool thinking task
        self._live_activity_visible: bool = False  # whether #live owns the current transient phase
        self._turn_start_time: float = 0.0  # monotonic timestamp when the current LLM turn started (stream_start)
        self._tool_start_time: float = 0.0  # monotonic timestamp when the current tool started (add_progress)
        self._model_wait_start_time: float = 0.0  # current provider request/wait segment
        self._stream_render_task: Any = None  # debounce task for live stream rendering
        self._stream_render_follow: bool = False  # preserve bottom-follow across #live collapse
        # State for show_question_popup (sequential multi-question prompt)
        self._question_queue: list[dict] = []
        self._question_answers: dict[str, str] = {}
        self._question_on_complete: Callable[[dict[str, str] | None], Awaitable[None]] | None = None

        # Callbacks
        self._on_submit: Callable[[str], Awaitable[None]] | None = None
        self._on_pre_submit: Callable[[str], None] | None = None
        self._on_cancel: Callable[[], Awaitable[None]] | None = None
        self._history_page_loader: (
            Callable[[int | None], Awaitable[tuple[list[dict], int | None, bool]]] | None
        ) = None
        self._history_before_offset: int | None = None
        self._history_has_older: bool = False
        self._history_loading: bool = False
        self._history_generation: int = 0
        self._history_tool_registry: Any = None
        self._history_workspace: Any = None

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

    # ── History file / topic bookmarks ─────────────────────────────────────

    @staticmethod
    def _topic_file_key(topic: str) -> str:
        return hashlib.sha256(topic.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _message_fingerprint(role: str, content: str) -> str:
        normalized = " ".join(content.split())
        return hashlib.sha256(f"{role}\0{normalized}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _message_summary(content: str, limit: int = 56) -> str:
        summary = " ".join(content.split()) or "（空消息）"
        return summary if len(summary) <= limit else summary[: limit - 1] + "…"

    def _next_message_id(self, role: str, content: str, timestamp: str = "") -> str:
        fingerprint = self._message_fingerprint(role, f"{timestamp}\0{content}")
        key = (role, fingerprint)
        occurrence = self._message_occurrences.get(key, 0) + 1
        self._message_occurrences[key] = occurrence
        return f"{role}:{fingerprint}:{occurrence}"

    def _bookmark_runtime_log(self, event_name: str, **fields: Any) -> None:
        append_session_runtime_log(
            self._runtime_log_path,
            event_name,
            topic=self._topic,
            bookmark_file=str(self._bookmark_file) if self._bookmark_file else None,
            bookmark_count=len(self._bookmarks),
            **fields,
        )

    def set_session_runtime_log_path(self, path: str | Path | None) -> None:
        self._runtime_log_path = Path(path) if path is not None else None
        self._bookmark_runtime_log("tui.bookmark.runtime_attached")

    def _load_bookmarks(self) -> None:
        self._bookmarks = {}
        if self._bookmark_file is None or not self._bookmark_file.exists():
            self._bookmark_runtime_log("tui.bookmark.loaded", file_exists=False)
            return
        try:
            payload = json.loads(self._bookmark_file.read_text(encoding="utf-8"))
            entries = payload.get("bookmarks", []) if isinstance(payload, dict) else []
            for entry in entries:
                message_id = str(entry.get("message_id", ""))
                if not message_id:
                    continue
                try:
                    record_index = max(0, int(entry.get("record_index", 0)))
                    char_offset = max(0, int(entry.get("char_offset", 0)))
                except (TypeError, ValueError):
                    continue
                bookmark_id = str(entry.get("bookmark_id", "")) or self._bookmark_id(
                    message_id,
                    record_index,
                    char_offset,
                )
                self._bookmarks[bookmark_id] = {
                    "bookmark_id": bookmark_id,
                    "message_id": message_id,
                    "record_index": record_index,
                    "char_offset": char_offset,
                    "role": str(entry.get("role", "")),
                    "summary": str(entry.get("summary", "")),
                    "created_at": str(entry.get("created_at", "")),
                }
            self._bookmark_runtime_log("tui.bookmark.loaded", file_exists=True)
        except Exception as exc:
            logger.warning("Failed to load topic bookmarks {}: {}", self._bookmark_file, exc)
            self._bookmark_runtime_log(
                "tui.bookmark.load_failed",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )

    def _save_bookmarks(self) -> None:
        if self._bookmark_file is None:
            return
        payload = {
            "schema": 2,
            "topic": self._topic,
            "bookmarks": list(self._bookmarks.values()),
        }
        try:
            self._bookmark_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._bookmark_file.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._bookmark_file)
        except Exception as exc:
            logger.warning("Failed to save topic bookmarks {}: {}", self._bookmark_file, exc)

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

    def _log_write(self, *items: Any) -> None:
        """Write Rich renderables or markup strings directly to the output log."""
        try:
            out = self._app.query_one("#output", _OutputLog)
            for item in items:
                out.write(item)
        except Exception:
            pass

    def _start_assistant_turn(self) -> None:
        """Reset header ownership when a new user turn begins."""
        self._assistant_turn_header_rendered = False
        self._suppress_segment_sep = False

    def _ensure_assistant_turn_header(
        self,
        ts: str | None = None,
        *,
        error: bool = False,
    ) -> bool:
        """Write one full nanobot header before the first assistant artifact.

        Returns True only when this call created the header. Tool traces, file
        edits, streamed text, completed replies, and history replay all share
        this gate so their ordering cannot diverge.
        """
        if self._assistant_turn_header_rendered:
            return False
        timestamp = ts or self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_style = "red bold" if error else "cyan"
        self._log_write(f"[{header_style}]{__logo__} nanobot[/] [dim]{timestamp}[/dim]")
        self._log_write("")
        self._assistant_turn_header_rendered = True
        return True

    def _write_response(
        self,
        content: str,
        ts: str,
        metadata: dict | None = None,
        *,
        message_id: str | None = None,
    ) -> None:
        """Write a completed response block as Rich objects (no ANSI conversion).

        When the response follows a streaming session that already rendered
        the "🐈 nanobot timestamp" header (and possibly tool traces below it),
        the header is omitted so we don't get duplicate headers — instead a
        short "─ HH:MM:SS ─" separator marks the new segment.
        """
        self._activity_phase = "write_response"
        out: _OutputLog | None = None
        if (metadata or {}).get("render_as") not in {"error", "system"}:
            try:
                out = self._app.query_one("#output", _OutputLog)
                out.begin_message(
                    message_id or self._next_message_id("assistant", content, ts),
                    "assistant",
                    self._message_summary(content),
                )
            except Exception:
                out = None
        render_as = (metadata or {}).get("render_as")
        render_as_text = render_as == "text"
        render_as_error = render_as == "error"
        header_created = self._ensure_assistant_turn_header(ts, error=render_as_error)
        if not header_created:
            # Continuation of an already-headed turn — mark the new segment
            # with a lightweight timestamp so it isn't glued to the previous one.
            # Strip date portion if present ("2026-05-20 22:45:30" → "22:45:30").
            # Fork: skip the separator entirely when nothing visible followed the
            # header (suppressed reasoning, no tool trace) — see pop_stream.
            if not self._suppress_segment_sep:
                short_ts = ts.split(" ", 1)[1] if " " in ts else ts
                self._log_write(f"[{self.THEME_MUTED}]─ {short_ts} ─[/{self.THEME_MUTED}]")
                self._log_write("")
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
        if out is not None:
            out.end_message()

    def _write_user(self, text: str, ts: str, *, message_id: str | None = None) -> None:
        """Write a user message block; records line range for gray background."""
        try:
            out = self._app.query_one("#output", _OutputLog)
            start = len(out.lines)
            out.begin_message(
                message_id or self._next_message_id("user", text, ts),
                "user",
                self._message_summary(text),
            )
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
        self._start_assistant_turn()
        # Fork: a full-width rule between the user block and the upcoming
        # response header — turns are separated more clearly than by a bare
        # blank line. Written outside the gray user-range recorded above.
        self._log_write(f"[{self.THEME_MUTED}]{'─' * 80}[/{self.THEME_MUTED}]")
        self._log_write("")
        self._last_sep = True
        try:
            out.end_message()
        except Exception:
            pass

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

    def set_on_cancel(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._on_cancel = callback

    def set_history_page_loader(
        self,
        callback: Callable[[int | None], Awaitable[tuple[list[dict], int | None, bool]]] | None,
        *,
        before_offset: int | None = None,
        has_older: bool = False,
    ) -> None:
        self._history_generation += 1
        self._history_page_loader = callback
        self._history_before_offset = before_offset
        self._history_has_older = has_older
        self._history_loading = False
        try:
            out = self._app.query_one("#output", _OutputLog)
            out._on_top_reached = self._request_older_history if callback is not None else None
        except Exception:
            pass

    def _request_older_history(self) -> None:
        if (
            self._history_loading
            or not self._history_has_older
            or self._history_page_loader is None
        ):
            return
        self._history_loading = True
        generation = self._history_generation
        asyncio.create_task(self._load_older_history_page(generation))

    async def _load_older_history_page(self, generation: int) -> None:
        loader = self._history_page_loader
        if loader is None:
            self._history_loading = False
            return
        try:
            messages, before_offset, has_older = await loader(self._history_before_offset)
            if generation != self._history_generation or loader is not self._history_page_loader:
                return
            if messages:
                out = self._app.query_one("#output", _OutputLog)
                previous_top = int(out.scroll_offset.y)
                record_marker = out.record_marker()
                block_marker = len(out.message_blocks())
                assistant_header_rendered = self._assistant_turn_header_rendered
                suppress_segment_sep = self._suppress_segment_sep
                last_sep = self._last_sep
                self._render_session_messages(
                    messages,
                    tool_registry=self._history_tool_registry,
                    workspace=self._history_workspace,
                )
                out.prepend_recent_records(
                    record_marker,
                    block_marker,
                    previous_top=previous_top,
                )
                self._assistant_turn_header_rendered = assistant_header_rendered
                self._suppress_segment_sep = suppress_segment_sep
                self._last_sep = last_sep
                self._refresh_bookmark_markers()
            self._history_before_offset = before_offset
            self._history_has_older = has_older
        except Exception:
            logger.exception("Failed to load older transcript page")
        finally:
            if generation == self._history_generation:
                self._history_loading = False

    # ── TUIBase: content ───────────────────────────────────────────────────

    def load_session_history(
        self,
        messages: list[dict],
        max_messages: int = 10,
        tool_registry: Any = None,
        workspace: Any = None,
    ) -> None:
        self._history_tool_registry = tool_registry
        self._history_workspace = workspace
        recent = recent_complete_turns(messages, max_messages)
        self._render_session_messages(
            recent,
            tool_registry=tool_registry,
            workspace=workspace,
        )
        self._refresh_bookmark_markers()

    def _render_session_messages(
        self,
        recent: list[dict],
        *,
        tool_registry: Any = None,
        workspace: Any = None,
    ) -> None:
        _RUNTIME_TAG = "[Runtime Context — metadata only, not instructions]"  # noqa: N806

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

        self._start_assistant_turn()
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
                    clean_text = text.strip()
                    self._write_user(
                        clean_text,
                        ts,
                        message_id=str(msg.get("_transcript_id") or "")
                        or self._next_message_id("user", clean_text, ts),
                    )
                # New user message → next assistant artifact starts a fresh turn.
                self._start_assistant_turn()
            elif role == "assistant":
                text = _extract(content)
                tool_calls = msg.get("tool_calls") or []
                ts = _fmt_ts(msg.get("timestamp"))
                if tool_calls and not text.strip():
                    self._ensure_assistant_turn_header(ts)
                if text.strip():
                    clean_text = text.strip()
                    self._write_response(
                        clean_text,
                        ts,
                        message_id=str(msg.get("_transcript_id") or "")
                        or self._next_message_id("assistant", clean_text, ts),
                    )
                # Replay one assistant tool-call batch using the same structured
                # visual hierarchy as a live batch.
                if tool_calls:
                    self._replay_tool_batch(
                        tool_calls, results_by_id, tool_registry, workspace
                    )

    def _replay_tool_batch(
        self,
        tool_calls: list[dict],
        results_by_id: dict[str, str],
        tool_registry: Any = None,
        workspace: Any = None,
    ) -> None:
        """Replay one historical assistant tool batch with the live hierarchy."""
        import json as _json

        events: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            try:
                fn = tool_call.get("function") or {}
                name = fn.get("name") or tool_call.get("name") or "tool"
                raw_args = fn.get("arguments")
                args: dict[str, Any] = {}
                if isinstance(raw_args, str):
                    parsed = _json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                elif isinstance(raw_args, dict):
                    args = raw_args
                result_text = results_by_id.get(str(tool_call.get("id") or ""), "")
                error = result_text.strip() if result_text.lstrip().startswith("Error") else ""
                events.append({
                    "phase": "error" if error else "end",
                    "name": name,
                    "arguments": args,
                    "error": error,
                })
            except Exception:
                logger.debug("replay tool batch: decode failed", exc_info=True)
        if events:
            self._render_tool_batch(events, elapsed=None)

    def add_user_echo(self, text: str, *, message_id: str | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._app.query_one("#output", _OutputLog).reset_user_navigation()
        except Exception:
            pass
        self._append_sep()
        self._write_user(text, ts, message_id=message_id)

    def add_response(
        self,
        content: str,
        metadata: dict | None = None,
        ts: str | None = None,
        *,
        message_id: str | None = None,
    ) -> None:
        self._clear_initial_thinking_placeholder()
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_response(content, ts, metadata, message_id=message_id)

    def add_progress(
        self, text: str, tool_events: list[dict[str, Any]] | None = None
    ) -> None:
        import time
        # Progress is the first definitive signal that the preceding resuming
        # stream ended for a tool call rather than a continuation or retry.
        if self._pending_tool_text_records:
            self._temporary_tool_text_records.extend(self._pending_tool_text_records)
            self._pending_tool_text_records = []
        self._tool_hint = text
        self._active_tool_events = [
            event for event in (tool_events or []) if isinstance(event, dict)
        ]
        # Mark tool start so add_tool_result can show the batch elapsed time.
        self._tool_start_time = time.monotonic()
        live_text = text
        if len(self._active_tool_events) > 1:
            live_text = f"执行中 · {len(self._active_tool_events)} 项"
        self._app._safe_call(self._update_progress_line, live_text)

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
        """Show the current tool in the transient single-line status area."""
        if not self._live_activity_visible:
            return
        self._app.update_live(Text(f"    ⠋ {text}", style="cyan"))

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
        self._ensure_assistant_turn_header()
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
        try:
            out = self._app.query_one("#output", _OutputLog)
            # Completed tool traces are permanent history. Append rather than
            # truncating by a physical line anchor, which may move after wrapping.
            out.write(line)
            self._tool_placeholder_line = len(out.lines)
        except Exception:
            pass

    def _petrify_tool_placeholder(self) -> None:
        """Commit an unfinished tool batch before another output phase replaces it."""
        if not self._tool_hint and not self._active_tool_events:
            return
        import time
        elapsed = (
            time.monotonic() - self._tool_start_time
            if self._tool_start_time else None
        )
        if self._active_tool_events:
            self._render_tool_batch(self._active_tool_events, elapsed=elapsed)
        else:
            self._render_tool_trace(self._tool_hint, "", elapsed)
        self._tool_hint = ""
        self._active_tool_events = []

    def _render_tool_batch(
        self,
        tool_events: list[dict[str, Any]],
        *,
        elapsed: float | None,
    ) -> None:
        """Append a readable batch title and one line per distinct tool target."""
        from rich.text import Text as _RText

        from nanobot.fork.utils.tool_hints import format_tool_event_items

        items = format_tool_event_items(tool_events, workspace=self._workspace)
        if not items:
            return
        self._ensure_assistant_turn_header()
        out = self._app.query_one("#output", _OutputLog)
        if len(tool_events) == 1 and len(items) == 1:
            item = items[0]
            line = _RText()
            line.append(self._TOOL_INDENT, style="")
            line.append(f"{self._TOOL_MARKER} ", style=self.THEME_MARKER)
            line.append(str(item["label"]), style=self.THEME_HINT)
            if item["status"] == "error":
                error = " ".join(str(item["error"] or "执行失败").split())
                if len(error) > 80:
                    error = error[:79] + "…"
                line.append(f"  · 失败：{error}", style=self.THEME_ERROR)
            if elapsed is not None and elapsed >= 1:
                line.append(f" · {int(elapsed)}s", style=self.THEME_MUTED)
            out.write(line)
            self._tool_placeholder_line = len(out.lines)
            return

        title = _RText()
        title.append(self._TOOL_INDENT, style="")
        title.append(f"{self._TOOL_MARKER} ", style=self.THEME_MARKER)
        title.append(f"工具 · {len(tool_events)} 项", style=self.THEME_HINT)
        if elapsed is not None and elapsed >= 1:
            title.append(f" · {int(elapsed)}s", style=self.THEME_MUTED)
        out.write(title)
        for index, item in enumerate(items):
            line = _RText()
            branch = "└─ " if index == len(items) - 1 else "├─ "
            line.append(f"{self._TOOL_INDENT}  {branch}", style=self.THEME_MARKER)
            line.append(str(item["label"]), style=self.THEME_HINT)
            if item["status"] == "error":
                error = " ".join(str(item["error"] or "执行失败").split())
                if len(error) > 80:
                    error = error[:79] + "…"
                line.append(f"  · 失败：{error}", style=self.THEME_ERROR)
            out.write(line)
        self._tool_placeholder_line = len(out.lines)

    def add_tool_result(
        self,
        summary: str,
        tool_events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Commit the current structured tool batch, then resume idle thinking."""
        if not self._tool_hint and not self._active_tool_events:
            return
        self._hide_live_activity()
        import time
        elapsed = (
            time.monotonic() - self._tool_start_time
            if self._tool_start_time else None
        )
        completed = [
            event for event in (tool_events or []) if isinstance(event, dict)
        ]
        if completed or self._active_tool_events:
            self._render_tool_batch(completed or self._active_tool_events, elapsed=elapsed)
        else:
            self._render_tool_trace(self._tool_hint, summary, elapsed)
        self._tool_hint = ""
        self._active_tool_events = []
        self._start_model_wait()

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
        self._ensure_assistant_turn_header()
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
        self._start_assistant_turn()
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._flushed_parts = []
        self._pending_tool_text_records = []
        self._temporary_tool_text_records = []
        self._suppress_segment_sep = False  # reset per turn; pop_stream may set it
        # Anchor the "thinking time" displayed by spinners to the moment the
        # turn began — not to each individual spinner restart. This keeps the
        # elapsed counter continuous across idle gaps + tool calls.
        self._turn_start_time = time.monotonic()
        self._model_wait_start_time = self._turn_start_time
        # The response header is permanent history; activity remains in #live.
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._stream_header_line = len(out.lines)
            self._ensure_assistant_turn_header(self._stream_ts)
            self._tool_placeholder_line = len(out.lines)
            out.scroll_end(animate=False)
        except Exception:
            self._stream_header_line = 0
            self._tool_placeholder_line = 0
        self._show_live_activity()

    def tool_phase_start(self) -> None:
        self._activity_phase = "tool_phase"
        self._cancel_idle_thinking()
        self._petrify_tool_placeholder()
        self._stream_buf = ""
        if not self._stream_ts:
            self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._tool_hint = "执行中..."
        self._show_live_activity()

    def _show_live_activity(self) -> None:
        """Show exactly one transient activity line below the output history."""
        self._live_activity_visible = True
        if self._tool_hint:
            content = Text(f"    ⠋ {self._tool_hint}", style="cyan")
        else:
            content = Text("  ⠋ 模型处理中 · 已等待 (0s)", style="grey50")
        self._app.update_live(content)
        self._app._safe_call(self._app.start_thinking_spinner)

    def _start_model_wait(self) -> None:
        """Replace a completed tool hint with an explicit provider wait state."""
        import time
        self._activity_phase = "model_wait"
        self._cancel_idle_thinking()
        self._tool_hint = ""
        self._model_wait_start_time = time.monotonic()
        self._show_live_activity()

    def _hide_live_activity(self) -> None:
        """Stop and collapse the transient activity line without touching history."""
        self._live_activity_visible = False
        try:
            self._app.stop_thinking_spinner()
        except Exception:
            pass
        self._app.clear_live()

    def _clear_initial_thinking_placeholder(self) -> None:
        """Compatibility hook: visible progress replaces the transient status."""
        self._hide_live_activity()

    def _cancel_idle_thinking(self) -> None:
        task = self._idle_thinking_task
        if task is not None and not task.done():
            task.cancel()
        self._idle_thinking_task = None
        self._hide_live_activity()

    def clear_initial_thinking(self) -> None:
        self._cancel_idle_thinking()

    def clear_idle_thinking(self) -> None:
        self._cancel_idle_thinking()

    def stop_thinking(self) -> None:
        self._cancel_idle_thinking()
        try:
            self._app.stop_spinner()
        except Exception:
            pass

    def _schedule_idle_thinking(self, delay: float = 0.5) -> None:
        """Show post-tool/provider waiting state after a quiet delay."""
        self._cancel_idle_thinking()

        async def _wait_then_show() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            self._idle_thinking_task = None
            self._show_live_activity()

        try:
            self._idle_thinking_task = asyncio.ensure_future(_wait_then_show())
        except RuntimeError:
            self._idle_thinking_task = None

    def _cancel_stream_render(self) -> None:
        task = self._stream_render_task
        if task is not None and not task.done():
            task.cancel()
        self._stream_render_task = None
        self._stream_render_follow = False

    def _render_stream_live(self) -> None:
        if not self._show_tool_preface:
            return
        try:
            out = self._app.query_one("#output", _OutputLog)
            follow = self._stream_render_follow
            self._stream_render_follow = False
            if not follow and out.user_is_scrolling() and not out.is_at_bottom():
                return
            if not follow and not out.is_at_bottom():
                return
            out.truncate_to(self._tool_placeholder_line)
            out.write(Text(self._stream_buf), scroll_end=follow)
        except Exception:
            pass

    def _schedule_stream_render(self, delay: float = 0.075) -> None:
        if not self._show_tool_preface:
            return
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
        # Capture bottom-follow before hiding #live changes the output viewport.
        try:
            out = self._app.query_one("#output", _OutputLog)
            self._stream_render_follow = self._stream_render_follow or out.is_at_bottom()
        except Exception:
            pass
        # No tool progress before the next segment means the previous flush was
        # a continuation/retry and must not be removed with tool-preface text.
        self._pending_tool_text_records = []
        self._clear_initial_thinking_placeholder()
        self._activity_phase = "stream_delta"
        self._cancel_idle_thinking()
        self._app.stop_thinking_spinner()
        self._app.stop_spinner()
        self._app.clear_live()
        # If a result event was omitted, commit the active tool before streamed
        # response text starts replacing the current live stream chunk.
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
            if self._stream_buf.strip() and self._show_tool_preface:
                render_as_text = (metadata or {}).get("render_as") == "text"
                # Replace the current live-stream text with its final rendered version.
                out.truncate_to(self._tool_placeholder_line)
                record_marker = out.record_marker()
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
                self._pending_tool_text_records = out.records_since(record_marker)
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
            # Tool-preface text was useful during execution, but the final reply
            # replaces it. Remove only those records; later tool traces survive.
            out.remove_records(self._temporary_tool_text_records)
            self._tool_placeholder_line = len(out.lines)
        except Exception:
            pass
        self._pending_tool_text_records = []
        self._temporary_tool_text_records = []
        # Fork: deferred reasoning flush. flush_reasoning skipped while stream
        # was active to avoid splitting the response visually; now is the right
        # time — stream chunk is gone, response will be re-written by
        # add_response below. Reasoning lands between header and response.
        if self._reasoning_buf.strip():
            deferred = self._reasoning_buf
            self._reasoning_buf = ""
            self._app._safe_call(self._write_reasoning_block, deferred)
        # stream_start and every assistant artifact share the persistent header
        # state; pop_stream no longer needs a one-shot skip flag.
        if bool(ts_was) or self._stream_header_line > 0:
            self._assistant_turn_header_rendered = True
        # Fork: suppress the continuation "─ts─" separator when no visible
        # content followed the header. _tool_placeholder_line only advances past
        # the header anchor (_stream_header_line + 2 = header line + its trailing
        # blank) when flush_stream lands a mid-turn segment or a tool trace is
        # petrified. If it still sits at that anchor, the only thing between the
        # header and the upcoming response was suppressed reasoning — so a
        # "─ts─" line would dangle. NOTE: the "+2" is coupled to stream_start
        # writing a single-line header + one blank; keep them in sync.
        self._suppress_segment_sep = (
            self._assistant_turn_header_rendered
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
        self._bookmark_file = self._bookmark_dir / f"{self._topic_file_key(topic_key)}.json"
        self._load_bookmarks()
        self._refresh_bookmark_markers()
        self._message_occurrences = {}
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
        self._message_occurrences = {}
        self._app.clear_live()
        self._stream_buf = ""
        self._stream_ts = ""
        self._assistant_turn_header_rendered = False
        self._suppress_segment_sep = False
        self._last_sep = False
        self._ctx_used = 0
        self._ctx_total = 0
        self.set_history_page_loader(None)
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

    def _notify_bookmark(self, text: str) -> None:
        try:
            self._app.notify(text, timeout=1.5)
        except Exception:
            pass

    @staticmethod
    def _bookmark_id(message_id: str, record_index: int, char_offset: int) -> str:
        raw = f"{message_id}\0{record_index}\0{char_offset}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _visible_bookmark_targets(self) -> list[tuple[int, dict[str, Any]]]:
        try:
            out = self._app.query_one("#output", _OutputLog)
        except Exception:
            return []
        targets = []
        for bookmark in self._bookmarks.values():
            line = out.bookmark_line(bookmark)
            if line is not None:
                targets.append((line, bookmark))
        return sorted(targets, key=lambda item: item[0])

    def _refresh_bookmark_markers(self) -> None:
        try:
            out = self._app.query_one("#output", _OutputLog)
        except Exception:
            return
        out.set_bookmarks(list(self._bookmarks.values()))

    def toggle_bookmark_at_view(self) -> None:
        diagnostic: dict[str, Any] = {}
        try:
            out = self._app.query_one("#output", _OutputLog)
            scroll_line = int(out.scroll_offset.y)
            selected_line = out.selected_bookmark_line()
            target_line = selected_line if selected_line is not None else scroll_line
            anchor = out.bookmark_anchor_at_line(target_line)
            diagnostic = {
                "scroll_line": scroll_line,
                "selected_line": selected_line,
                "target_line": target_line,
                "target_source": "selection" if selected_line is not None else "viewport_top",
                "line_count": len(out.lines),
                "message_block_count": len(out.message_blocks()),
                "record_span_count": len(out._record_spans),
                "anchor_found": anchor is not None,
                "anchor_role": anchor.get("role") if anchor else None,
                "anchor_record_index": anchor.get("record_index") if anchor else None,
                "anchor_char_offset": anchor.get("char_offset") if anchor else None,
                "anchor_message_id": anchor.get("message_id") if anchor else None,
            }
        except Exception as exc:
            anchor = None
            diagnostic = {
                "anchor_found": False,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        self._bookmark_runtime_log("tui.bookmark.toggle_resolved", **diagnostic)
        if anchor is None:
            self._notify_bookmark("当前位置没有可添加书签的消息")
            return
        bookmark_id = self._bookmark_id(
            anchor["message_id"],
            anchor["record_index"],
            anchor["char_offset"],
        )
        if bookmark_id in self._bookmarks:
            del self._bookmarks[bookmark_id]
            self._save_bookmarks()
            self._refresh_bookmark_markers()
            out._clear_selection()
            self._notify_bookmark("已删除书签")
            return
        self._bookmarks[bookmark_id] = {
            "bookmark_id": bookmark_id,
            **anchor,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_bookmarks()
        self._refresh_bookmark_markers()
        out._clear_selection()
        self._notify_bookmark("已添加书签")

    def jump_to_bookmark(self, bookmark_id: str) -> bool:
        bookmark = self._bookmarks.get(bookmark_id)
        if bookmark is None:
            return False
        try:
            out = self._app.query_one("#output", _OutputLog)
            line = out.bookmark_line(bookmark)
            if line is None:
                self._notify_bookmark("该书签对应消息已失效")
                return False
            out.scroll_to(y=line, animate=False, immediate=True, force=True)
            out.flash_bookmark(bookmark)
            self._notify_bookmark("已定位到书签")
            return True
        except Exception:
            return False

    def jump_to_previous_bookmark(self) -> bool:
        targets = self._visible_bookmark_targets()
        self._bookmark_runtime_log(
            "tui.bookmark.jump_resolved",
            visible_target_count=len(targets),
            target_lines=[line for line, _bookmark in targets],
        )
        if not targets:
            self._notify_bookmark("本话题没有可用书签")
            return False
        try:
            out = self._app.query_one("#output", _OutputLog)
            current = int(out.scroll_offset.y)
        except Exception:
            return False
        if out.is_at_bottom():
            target = targets[-1]
            result = self.jump_to_bookmark(target[1]["bookmark_id"])
            target_line: int | str = target[0]
        else:
            earlier = [item for item in targets if item[0] < current]
            if earlier:
                target = earlier[-1]
                result = self.jump_to_bookmark(target[1]["bookmark_id"])
                target_line = target[0]
            else:
                out.clear_bookmark_highlight()
                out.scroll_end(animate=False, immediate=True, force=True)
                self._notify_bookmark("已定位到底部")
                result = True
                target_line = "bottom"
        self._bookmark_runtime_log(
            "tui.bookmark.jump_completed",
            current_line=current,
            target_line=target_line,
            result=result,
        )
        return result

    def delete_bookmark(self, bookmark_id: str) -> bool:
        if bookmark_id not in self._bookmarks:
            return False
        del self._bookmarks[bookmark_id]
        self._save_bookmarks()
        self._refresh_bookmark_markers()
        return True

    def clear_topic_bookmarks(self) -> int:
        count = len(self._bookmarks)
        self._bookmarks = {}
        self._save_bookmarks()
        self._refresh_bookmark_markers()
        self.hide_popup()
        self._notify_bookmark(f"已清理本话题 {count} 个书签")
        return count

    def _bookmark_popup_items(self) -> list[tuple[str, str]]:
        entries = list(self._bookmarks.values())
        valid = self._visible_bookmark_targets()
        valid_ids = {bookmark["bookmark_id"] for _line, bookmark in valid}
        order = {bookmark["bookmark_id"]: line for line, bookmark in valid}
        entries.sort(key=lambda entry: order.get(entry["bookmark_id"], 10**18))
        try:
            out = self._app.query_one("#output", _OutputLog)
        except Exception:
            out = None
        items: list[tuple[str, str]] = []
        for entry in entries:
            role = "你" if entry["role"] == "user" else "nanobot"
            is_valid = entry["bookmark_id"] in valid_ids
            context = out.bookmark_context_summary(entry) if out is not None and is_valid else None
            summary = context or str(entry.get("summary", "")) or "（无摘要）"
            marker = "" if is_valid else " [失效]"
            items.append((entry["bookmark_id"], f"{role}: {summary}{marker}"))
        return items

    def show_bookmark_popup(self) -> None:
        self._bookmark_runtime_log("tui.bookmark.popup_requested")
        items = self._bookmark_popup_items()
        if not items:
            self._notify_bookmark("本话题没有书签")
            return

        async def _jump(bookmark_id: str) -> None:
            self.jump_to_bookmark(bookmark_id)

        self._bookmark_popup_ids = [value for value, _label in items]
        self.show_topic_popup(items, _jump)

    def show_bookmark_delete_popup(self) -> None:
        self._bookmark_runtime_log("tui.bookmark.delete_popup_requested")
        items = self._bookmark_popup_items()
        if not items:
            self._notify_bookmark("本话题没有书签")
            return

        async def _delete(bookmark_id: str) -> None:
            if self.delete_bookmark(bookmark_id):
                self._notify_bookmark("已删除书签")

        self._bookmark_popup_ids = [value for value, _label in items]
        self.show_topic_popup(items, _delete)

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
