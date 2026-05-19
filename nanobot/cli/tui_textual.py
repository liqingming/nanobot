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
    from textual.events import MouseScrollUp, MouseScrollDown

    class _OutputLog(RichLog):
        """RichLog that never takes keyboard focus and owns its scroll events exclusively."""
        can_focus = False

        def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
            event.stop()  # prevent App-level handler from double-firing
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
        ENABLE_MOUSE_SUPPORT = False  # let terminal handle native selection/scroll

        CSS = """
        Screen {
            layout: vertical;
            background: #000000;
        }
        #output {
            height: 1fr;
            scrollbar-gutter: stable;
            border: none;
            background: #000000;
        }
        #topic-bar {
            height: 1;
            background: $panel;
            text-align: right;
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
            color: $text-muted;
        }
        #popup {
            height: auto;
            max-height: 10;
            display: none;
            background: $panel;
            border: round $primary;
            padding: 0 1;
        }
        #popup.visible {
            display: block;
        }
        #sep {
            height: 1;
            color: $text-muted;
        }
        #input {
            height: auto;
            border: none;
            padding: 0 1;
        }
        #status {
            height: 1;
            background: $panel-darken-1;
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

        def compose(self) -> ComposeResult:
            yield _OutputLog(id="output", markup=True, highlight=False, wrap=True)
            yield Static("", id="topic-bar")
            yield Static("", id="live")
            yield Static("", id="popup")
            yield Static("─" * 80, id="sep")
            yield _HistoryInput(self._tui, placeholder="You: ", id="input")
            yield Static("", id="status")

        def on_mount(self) -> None:
            self.query_one("#input").focus()
            self._write_welcome()

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
            out.write("  [cyan]拖动选中 + 滚轮[/cyan]    复制多屏内容")
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
        if self._render_md and not render_as_text and content.strip():
            self._log_write(Markdown(content))
        else:
            self._log_write(Text(content))
        self._log_write("")
        self._last_sep = False

    def _write_user(self, text: str, ts: str) -> None:
        """Write a user message block as Rich markup strings (no ANSI conversion)."""
        self._log_write(
            f"[bold blue]You[/bold blue] [dim]{ts}[/dim]",
            text,
        )
        self._last_sep = False

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
            self._log_write("[dim]" + "─" * 80 + "[/dim]")
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
                    self._append_sep()
            elif role == "assistant":
                text = _extract(content)
                if text.strip():
                    ts = _fmt_ts(msg.get("timestamp"))
                    self._write_response(text.strip(), ts)

    def add_user_echo(self, text: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_sep()
        self._write_user(text, ts)
        self._append_sep()

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
        self._app.start_spinner(tool=True)

    def add_system(self, text: str) -> None:
        self._log_write(f"[dim]{text}[/dim]")
        self._last_sep = False

    # ── TUIBase: streaming ─────────────────────────────────────────────────

    def stream_start(self) -> None:
        self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_buf = ""
        self._app.start_spinner(tool=False)

    def tool_phase_start(self) -> None:
        self._stream_buf = ""
        if not self._stream_ts:
            self._stream_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._tool_hint = ""
        self._app.start_spinner(tool=True)

    def stream_delta(self, delta: str) -> None:
        self._app.stop_spinner()
        self._stream_buf += delta
        ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Use plain text during streaming; Markdown is rendered on flush_stream
        live = f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]\n{self._stream_buf}"
        self._app.update_live(live)

    def flush_stream(self, metadata: dict | None = None) -> None:
        self._app.stop_spinner()
        if self._stream_buf.strip():
            ts = self._stream_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.add_response(self._stream_buf, metadata, ts=ts)
        self._stream_buf = ""
        self._app.clear_live()

    def pop_stream(self) -> str:
        self._app.stop_spinner()
        buf = self._stream_buf
        self._stream_buf = ""
        self._stream_ts = ""
        self._app.clear_live()
        return buf

    # ── TUIBase: state ─────────────────────────────────────────────────────

    def set_topic(self, name: str) -> None:
        self._topic = name
        self._app.update_topic_bar(name)
        self._app.title = f"nanobot — {name}"  # Textual writes OSC title sequence

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
