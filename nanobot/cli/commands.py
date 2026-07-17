"""CLI commands for nanobot."""

import asyncio
import json
import os
import select
import signal
import sys
import uuid
from collections.abc import Callable, Iterable
from contextlib import nullcontext, suppress
from datetime import datetime
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        with suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Keep console encoding setup before importing CLI UI/logging libraries.
import typer  # noqa: E402
from loguru import logger  # noqa: E402

# Remove default handler and re-add with unified nanobot format
logger.remove()
_log_handler_id = logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<cyan>{extra[channel]}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=None,
    filter=lambda record: record["extra"].setdefault("channel", "-") or True,
)


def _set_nanobot_logs(enabled: bool) -> None:
    if enabled:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")


from prompt_toolkit import PromptSession, print_formatted_text  # noqa: E402
from prompt_toolkit.application import run_in_terminal  # noqa: E402
from prompt_toolkit.formatted_text import ANSI, HTML  # noqa: E402
from prompt_toolkit.history import FileHistory  # noqa: E402
from prompt_toolkit.key_binding import KeyBindings  # noqa: E402
from prompt_toolkit.keys import Keys  # noqa: E402
from prompt_toolkit.patch_stdout import patch_stdout  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from nanobot import __logo__, __version__  # noqa: E402
from nanobot import optional_features as feature_support  # noqa: E402
from nanobot.agent.hooks import create_file_edit_activity_hook  # noqa: E402
from nanobot.agent.loop import AgentLoop  # noqa: E402
from nanobot.bus.outbound_events import (  # noqa: E402
    ProgressEvent,
    RetryWaitEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    outbound_event_from_message,
)
from nanobot.cli.gateway import create_gateway_app  # noqa: E402
from nanobot.cli.markdown import terminal_markdown  # noqa: E402
from nanobot.cli.stream import StreamRenderer, ThinkingSpinner  # noqa: E402
from nanobot.command.builtin import BUILTIN_COMMAND_SPECS  # noqa: E402
from nanobot.config.paths import get_workspace_cache_dir, get_workspace_path, is_default_workspace  # noqa: E402
from nanobot.config.schema import Config  # noqa: E402
from nanobot.utils.evaluator import evaluate_response  # noqa: E402
from nanobot.utils.helpers import safe_filename, sync_workspace_templates  # noqa: E402
from nanobot.utils.oauth_compat import call_with_optional_proxy  # noqa: E402
from nanobot.utils.restart import (  # noqa: E402
    consume_restart_notice_from_env,
    format_restart_completed_message,
    should_show_cli_restart_notice,
)
from nanobot.webui.sidebar_state import read_webui_sidebar_state  # noqa: E402


def _sanitize_surrogates(text: str) -> str:
    """Reconstruct surrogate pairs into real characters; replace lone surrogates.

    On Windows, console input may produce lone surrogate code points (e.g.
    ``\\ud83d\\udc08`` for U+1F408).  Round-tripping through UTF-16 reconstructs
    paired surrogates into their actual characters and replaces unpaired ones
    with U+FFFD.
    """
    return text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")


def _signal_name(signum: int) -> str:
    with suppress(ValueError):
        return signal.Signals(signum).name
    return f"signal {signum}"


def _ensure_gateway_tty_signal_mode() -> None:
    """Keep foreground gateway Ctrl+C usable even after a raw-mode TTY leak."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    with suppress(Exception):
        import termios

        attrs = termios.tcgetattr(fd)
        lflag = attrs[3]
        required = termios.ISIG | termios.ICANON | termios.ECHO
        if (lflag & required) == required:
            return
        attrs[3] = lflag | required
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
        logger.debug("Restored foreground gateway TTY signal mode")


def _install_gateway_shutdown_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    tasks: list[asyncio.Task],
    print_status: Callable[[str], None],
) -> Callable[[], None]:
    """Install foreground gateway signal handlers and return a restore callback."""
    loop_signals: list[int] = []
    previous_handlers: list[tuple[int, Any]] = []
    shutdown_requested = False

    def request_shutdown(signum: int) -> None:
        nonlocal shutdown_requested
        sig_name = _signal_name(signum)
        if shutdown_requested:
            logger.warning("Forcing gateway shutdown after repeated {}", sig_name)
            for task in tasks:
                if not task.done():
                    task.cancel()
            return
        shutdown_requested = True
        logger.info("Gateway shutdown requested by {}", sig_name)
        print_status("\nShutting down... Press Ctrl+C again to force.")
        shutdown_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_shutdown, signum)
        except (NotImplementedError, RuntimeError, ValueError):
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, lambda sig, _frame: request_shutdown(sig))
            except (RuntimeError, ValueError):
                logger.debug("Could not install gateway handler for {}", _signal_name(signum))
                continue
            previous_handlers.append((signum, previous))
        else:
            loop_signals.append(signum)

    def restore() -> None:
        for signum in loop_signals:
            with suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(signum)
        for signum, handler in previous_handlers:
            with suppress(RuntimeError, ValueError):
                signal.signal(signum, handler)

    return restore


def _advance_dream_cursor_if_behind(memory: Any) -> None:
    latest = memory.get_latest_cursor()
    if memory.get_last_dream_cursor() < latest:
        memory.set_last_dream_cursor(latest)


class SafeFileHistory(FileHistory):
    """FileHistory subclass that sanitizes surrogate characters on write.

    On Windows, special Unicode input (emoji, mixed-script) can produce
    surrogate characters that crash prompt_toolkit's file write.
    See issue #2846.
    """

    def store_string(self, string: str) -> None:
        super().store_string(_sanitize_surrogates(string))



app = typer.Typer(
    name="nanobot",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=False,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
_REASONING_SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")
_REASONING_FLUSH_CHARS = 60


def _tool_result_summary_from_events(tool_events: object) -> str | None:
    """Return a TUI completion summary for structured tool finish events."""
    if not isinstance(tool_events, list):
        return None
    completed = [
        event for event in tool_events
        if isinstance(event, dict) and event.get("phase") in {"end", "error"}
    ]
    if not completed:
        return None
    errors = [
        str(event.get("error") or "").strip()
        for event in completed
        if event.get("phase") == "error"
    ]
    errors = [err for err in errors if err]
    if errors:
        return f"Error: {errors[0]}"
    return ""


def _format_skills_command(skills_loader: Any) -> str:
    return skills_loader.format_listing()


def _format_prompt_inspection(messages: list[dict[str, Any]]) -> str:
    """Render the inspectable parts of an exact message list without dumping history."""
    system = messages[0].get("content", "") if messages else ""
    current = messages[-1].get("content", "") if len(messages) > 1 else ""
    if not isinstance(system, str):
        system = json.dumps(system, ensure_ascii=False, indent=2)
    if not isinstance(current, str):
        current = json.dumps(current, ensure_ascii=False, indent=2)

    history = messages[1:-1] if len(messages) > 2 else []
    counts: dict[str, int] = {}
    chars = 0
    for message in history:
        role = str(message.get("role", "unknown"))
        counts[role] = counts.get(role, 0) + 1
        content = message.get("content", "")
        chars += len(content) if isinstance(content, str) else len(json.dumps(content, ensure_ascii=False))
    history_summary = "、".join(f"{role} {count}" for role, count in counts.items()) or "无"

    return "\n\n---\n\n".join([
        "# 当前请求上下文（只读检查）",
        f"## 历史摘要\n\n未展开 {len(history)} 条历史消息（{history_summary}，约 {chars:,} 字符）。",
        f"## System Prompt\n\n{system}",
        f"## 当前请求上下文（检查占位消息）\n\n{current}",
    ])


def _format_topic_cache_size(size_bytes: int | None) -> str:
    if size_bytes is None or size_bytes < 0:
        return "? B"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = size_bytes / 1024
    if value < 1024:
        return f"{value:.1f} KB" if value < 10 else f"{value:.0f} KB"
    value /= 1024
    if value < 1024:
        return f"{value:.1f} MB" if value < 10 else f"{value:.0f} MB"
    value /= 1024
    return f"{value:.1f} GB" if value < 10 else f"{value:.0f} GB"


def _topic_memory_dir(data_dir: Path, session_key: str) -> Path:
    safe_key = safe_filename(session_key.replace(":", "_"))
    return data_dir / "memory" / "topics" / safe_key


def _path_tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _topic_cache_size_bytes(
    *,
    data_dir: Path,
    session_key: str,
    session_path: str | None = None,
    transcript_path: str | None = None,
) -> int:
    total = _path_tree_size(Path(session_path)) if session_path else 0
    total += _path_tree_size(Path(transcript_path)) if transcript_path else 0
    total += _path_tree_size(_topic_memory_dir(data_dir, session_key))
    return total


def _format_topic_popup_label(topic: str, size_bytes: int | None) -> str:
    return f"{topic}  [{_format_topic_cache_size(size_bytes)}]"


_HEARTBEAT_PREAMBLE = (
    "[Your response will be delivered directly to the user's messaging app. "
    "Output ONLY the final user-facing message. Never reference internal "
    "files (HEARTBEAT.md, AWARENESS.md, etc.), your instructions, or your "
    "decision process. If nothing needs reporting, respond with just "
    "'All clear.' and nothing else.]\n\n"
)


def _heartbeat_has_active_tasks(content: str) -> bool:
    """True if HEARTBEAT.md has task lines, ignoring headers, blanks and comments."""
    in_comment = False
    in_active_section: bool = False
    for line in content.splitlines():
        stripped = line.strip()
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("##") and not stripped.startswith("###"):
                heading = stripped.lstrip("#").strip().lower()
                in_active_section = heading.startswith("active tasks")
            continue
        if stripped.startswith("<!--"):
            if "-->" not in stripped[4:]:
                in_comment = True
            continue
        if in_active_section is False:
            continue
        return True
    return False


def _pick_heartbeat_target_from_sessions(
    *,
    enabled_channels: Iterable[str],
    sessions: Iterable[dict[str, Any]],
    archived_keys: Iterable[str],
) -> tuple[str, str]:
    enabled = set(enabled_channels)
    archived = set(archived_keys)
    for item in sessions:
        key = item.get("key") or ""
        if key in archived:
            continue
        if ":" not in key:
            continue
        channel, chat_id = key.split(":", 1)
        if channel in {"cli", "system"}:
            continue
        if channel in enabled and chat_id:
            return channel, chat_id
    return "cli", "direct"


# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    with suppress(Exception):
        import termios

        termios.tcflush(fd, termios.TCIFLUSH)
        return

    with suppress(Exception):
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    with suppress(Exception):
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)


def _build_cli_key_bindings() -> KeyBindings:
    """Key bindings for the interactive prompt.

    Behaviour:
      * Enter       -> submit the current input (keeps the familiar
                       single-line Enter-to-send feel even though the buffer
                       is multiline-capable).
      * Alt+Enter   -> insert a newline for multi-line input.
    """
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter / Meta+Enter (ESC + CR, "\x1b\r")
    def _(event):
        event.current_buffer.insert_text("\n")

    # LF-as-Enter terminals send Alt+Enter as ESC + LF rather than ESC + CR.
    @kb.add("escape", Keys.ControlJ)  # Alt+Enter on LF-as-Enter terminals
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    with suppress(Exception):
        import termios

        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())

    from nanobot.config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=SafeFileHistory(str(history_file)),
        enable_open_in_editor=False,
        # Multiline-capable buffer; Enter still submits via the custom key
        # bindings, while Alt+Enter adds a newline.
        multiline=True,
        key_bindings=_build_cli_key_bindings(),
    )


def _make_console() -> Console:
    return Console(file=sys.stdout)


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=sys.stdout.isatty(),
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


def _print_agent_response(
    response: str,
    render_markdown: bool,
    metadata: dict | None = None,
    show_header: bool = True,
) -> None:
    """Render assistant response with consistent terminal styling."""
    console = _make_console()
    content = response or ""
    body = _response_renderable(content, render_markdown, metadata)
    if show_header:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print()
        console.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]")
    console.print(body)
    console.print()


def _response_renderable(content: str, render_markdown: bool, metadata: dict | None = None):
    """Render plain-text command output without markdown collapsing newlines."""
    if not render_markdown:
        return Text(content)
    if (metadata or {}).get("render_as") == "text":
        return Text(content)
    return terminal_markdown(content)


async def _print_interactive_line(text: str) -> None:
    """Print async interactive updates with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        ansi = _render_interactive_ansi(
            lambda c: c.print(f"  [dim]↳ {text}[/dim]")
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _print_interactive_response(
    response: str,
    render_markdown: bool,
    metadata: dict | None = None,
) -> None:
    """Print async interactive replies with prompt_toolkit-safe Rich styling."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def _write() -> None:
        content = response or ""
        ansi = _render_interactive_ansi(
            lambda c: (
                c.print(),
                c.print(f"[cyan]{__logo__} nanobot[/cyan] [dim]{ts}[/dim]"),
                c.print(_response_renderable(content, render_markdown, metadata)),
                c.print(),
            )
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


def _print_cli_progress_line(text: str, thinking: ThinkingSpinner | None, renderer: StreamRenderer | None = None) -> None:
    """Print a CLI progress line, pausing the spinner if needed."""
    if not text.strip():
        return
    target = renderer.console if renderer else console
    pause = renderer.pause_spinner() if renderer else (thinking.pause() if thinking else nullcontext())
    with pause:
        if renderer:
            renderer.ensure_header()
        target.print(f"  [dim]↳ {text}[/dim]")


class _ReasoningBuffer:
    def __init__(self) -> None:
        self._text = ""

    def add(self, text: str) -> str | None:
        if not text:
            return None
        self._text += text
        if self._should_flush(text):
            return self.flush()
        return None

    def flush(self) -> str | None:
        text = self._text.strip()
        self._text = ""
        return text or None

    def clear(self) -> None:
        self._text = ""

    def _should_flush(self, text: str) -> bool:
        stripped = text.rstrip()
        return (
            "\n" in text
            or stripped.endswith(_REASONING_SENTENCE_ENDINGS)
            or len(self._text) >= _REASONING_FLUSH_CHARS
        )


def _print_cli_reasoning(text: str, thinking: ThinkingSpinner | None, renderer: StreamRenderer | None = None) -> None:
    """Print reasoning/thinking content in a distinct style."""
    if not text.strip():
        return
    target = renderer.console if renderer else console
    pause = renderer.pause_spinner() if renderer else (thinking.pause() if thinking else nullcontext())
    with pause:
        if renderer:
            renderer.ensure_header()
        target.print(f"[dim italic]✻ {text}[/dim italic]")


def _flush_cli_reasoning(
    reasoning_buffer: _ReasoningBuffer,
    thinking: ThinkingSpinner | None,
    renderer: StreamRenderer | None = None,
) -> None:
    text = reasoning_buffer.flush()
    if text:
        _print_cli_reasoning(text, thinking, renderer)


async def _print_interactive_progress_line(text: str, thinking: ThinkingSpinner | None, renderer: StreamRenderer | None = None) -> None:
    """Print an interactive progress line, pausing the spinner if needed."""
    if not text.strip():
        return
    if renderer:
        with renderer.pause_spinner():
            renderer.ensure_header()
            renderer.console.print(f"  [dim]↳ {text}[/dim]")
    else:
        with thinking.pause() if thinking else nullcontext():
            await _print_interactive_line(text)


async def _maybe_print_interactive_progress(
    msg: Any,
    thinking: ThinkingSpinner | None,
    channels_config: Any,
    renderer: StreamRenderer | None = None,
    reasoning_buffer: _ReasoningBuffer | None = None,
) -> bool:
    event = outbound_event_from_message(msg)
    if isinstance(event, RetryWaitEvent):
        await _print_interactive_progress_line(msg.content, thinking, renderer)
        return True

    if not isinstance(event, ProgressEvent):
        return False

    reasoning_buffer = reasoning_buffer or _ReasoningBuffer()

    if event.reasoning_end:
        if channels_config and not channels_config.show_reasoning:
            reasoning_buffer.clear()
        else:
            _flush_cli_reasoning(reasoning_buffer, thinking, renderer)
        return True

    is_tool_hint = event.tool_hint
    is_reasoning = event.reasoning or event.reasoning_delta
    if is_reasoning:
        if channels_config and not channels_config.show_reasoning:
            reasoning_buffer.clear()
            return True
        text = reasoning_buffer.add(msg.content)
        if text:
            _print_cli_reasoning(text, thinking, renderer)
        return True
    if channels_config and is_tool_hint and not channels_config.send_tool_hints:
        return True
    if channels_config and not is_tool_hint and not channels_config.send_progress:
        return True

    await _print_interactive_progress_line(msg.content, thinking, renderer)
    return True


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


def _is_cli_local_command(text: str) -> bool:
    """Return whether interactive CLI handles *text* without an agent turn.

    Both TUI backends invoke pre-submit before the CLI callback for slash text
    typed with arguments, because such text is no longer an exact palette item.
    Keep this classifier beside the CLI dispatcher so local commands never get
    rendered as user messages or start a speculative thinking animation.
    """
    command = text.strip()
    return (
        _is_exit_command(command)
        or command == "/skills"
        or command == "/system-prompt"
        or command == "/skin"
        or command.startswith("/skin ")
        or command == "/clear"
        or command == "/rename"
        or command.startswith("/rename ")
        or command == "/commit_memory"
        or command.startswith("/commit_memory ")
        or command == "/todos"
        or command.startswith("/todos ")
        or command == "/resume"
        or command.startswith("/resume ")
    )


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    if ctx.invoked_subcommand is not None:
        return
    agent(
        message=None,
        session_id="cli:direct",
        workspace=None,
        config=None,
        markdown=True,
        logs=False,
    )
    raise typer.Exit()

# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    wizard: bool = typer.Option(False, "--wizard", help="Use interactive wizard"),
    non_interactive_refresh: bool = typer.Option(False, "--refresh", help="Refresh config, preserving existing settings without prompting"),
):
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path
    from nanobot.config.schema import Config

    if config:
        config_path = Path(config).expanduser().resolve()
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        config_path = get_config_path()

    def _apply_workspace_override(loaded: Config) -> Config:
        if workspace:
            loaded.agents.defaults.workspace = workspace
        return loaded

    # Create or update config
    if config_path.exists():
        if wizard:
            config = _apply_workspace_override(load_config(config_path))
        else:
            should_refresh = non_interactive_refresh
            if not non_interactive_refresh:
                console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
                console.print(
                    "  [bold]y[/bold] = overwrite with defaults (existing values will be lost)"
                )
                console.print(
                    "  [bold]N[/bold] = refresh config, keeping existing values and adding new fields"
                )
                if typer.confirm("Overwrite?"):
                    config = _apply_workspace_override(Config())
                    save_config(config, config_path)
                    console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
                else:
                    should_refresh = True

            if should_refresh:
                config = _apply_workspace_override(load_config(config_path))
                save_config(config, config_path)
                console.print(
                    f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)"
                )
    else:
        config = _apply_workspace_override(Config())
        # In wizard mode, don't save yet - the wizard will handle saving if should_save=True
        if not wizard:
            save_config(config, config_path)
            console.print(f"[green]✓[/green] Created config at {config_path}")

    # Run interactive wizard if enabled
    if wizard:
        from nanobot.cli.onboard import run_onboard

        try:
            result = run_onboard(initial_config=config)
            if not result.should_save:
                console.print("[yellow]Configuration discarded. No changes were saved.[/yellow]")
                return

            config = result.config
            save_config(config, config_path)
            console.print(f"[green]✓[/green] Config saved at {config_path}")
        except Exception as e:
            console.print(f"[red]✗[/red] Error during configuration: {e}")
            console.print("[yellow]Please run 'nanobot onboard' again to complete setup.[/yellow]")
            raise typer.Exit(1)
    _onboard_plugins(config_path)

    # Create workspace, preferring the configured workspace path.
    workspace_path = get_workspace_path(config.workspace_path)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace_path}")

    sync_workspace_templates(workspace_path)

    agent_cmd = 'nanobot agent -m "Hello!"'
    gateway_cmd = "nanobot gateway"
    if config:
        agent_cmd += f" --config {config_path}"
        gateway_cmd += f" --config {config_path}"

    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    if wizard:
        console.print(f"  1. Chat: [cyan]{agent_cmd}[/cyan]")
        console.print(f"  2. Start gateway: [cyan]{gateway_cmd}[/cyan]")
    else:
        console.print(f"  1. Add your API key to [cyan]{config_path}[/cyan]")
        console.print("     Get one at: https://openrouter.ai/keys")
        console.print(f"  2. Chat: [cyan]{agent_cmd}[/cyan]")
    console.print(
        "\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]"
    )


def _merge_missing_defaults(existing: Any, defaults: Any) -> Any:
    """Recursively fill in missing values from defaults without overwriting user config."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing

    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = _merge_missing_defaults(merged[key], value)
    return merged


def _onboard_plugins(config_path: Path) -> None:
    """Inject default config for all discovered channels (built-in + plugins)."""
    import json

    from nanobot.channels.registry import discover_all

    all_channels = discover_all()
    if not all_channels:
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    channels = data.setdefault("channels", {})
    for name, cls in all_channels.items():
        if name not in channels:
            channels[name] = cls.default_config()
        else:
            channels[name] = _merge_missing_defaults(channels[name], cls.default_config())

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_enable_options(
    extras: dict[str, list[str] | None],
    builtin_channels: set[str],
    plugin_channels: dict[str, Any] | None = None,
    config: Config | None = None,
) -> None:
    plugin_channels = plugin_channels or {}
    if config is None:
        if not extras:
            return
        for section, values in extras.items():
            if not values:
                continue
            available = ", ".join(values)
            console.print(f"[dim]{section}: {available}[/dim]")
        return

    table = Table(title="Available Features")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Enabled")

    for item in sorted(builtin_channels | set(plugin_channels) | set(extras)):
        is_channel = item in builtin_channels or item in plugin_channels
        enabled = (
            feature_support.channel_enabled(config, item)
            if is_channel
            else feature_support.extra_installed(item, extras[item])
        )
        table.add_row(
            item,
            "channel" if is_channel else "feature",
            "[green]yes[/green]" if enabled else "[dim]no[/dim]",
        )

    console.print(table)


def _model_display(config: Config) -> tuple[str, str, str | None]:
    """Return (resolved_model_name, preset_tag, reasoning_effort) for display strings."""
    resolved = config.resolve_preset()
    name = config.agents.defaults.model_preset
    tag = f" (preset: {name})" if name else ""
    return resolved.model, tag, resolved.reasoning_effort


def _todos_all_completed(todos: list[dict[str, Any]]) -> bool:
    return bool(todos) and all(t.get("status") == "completed" for t in todos)


def _should_hide_stale_todos_on_new_turn(todos: list[dict[str, Any]]) -> bool:
    return bool(todos) and not _todos_all_completed(todos)


_CLI_UNNAMED_SESSION_LABEL = "未命名会话"


def _mark_cli_session_unnamed(session: Any) -> None:
    """Mark a CLI session as unnamed without deciding whether to persist it."""
    session.metadata["cli_unnamed"] = True
    session.metadata.pop("cli_title", None)


def _cli_session_display_name(session_info: dict[str, Any], cli_channel: str) -> str | None:
    """Return a human-facing CLI session name, or None for another channel."""
    key = str(session_info.get("key") or "")
    prefix = f"{cli_channel}:"
    if not key.startswith(prefix):
        return None
    metadata = session_info.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("cli_title")
        if isinstance(title, str) and title.strip():
            return title.strip()
        if metadata.get("cli_unnamed") is True:
            return _CLI_UNNAMED_SESSION_LABEL
    # SessionManager also exposes its generic title as a flat list field. It is
    # the fallback for older list entries that predate CLI-specific metadata.
    title = session_info.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return key[len(prefix):]


def _resolve_cli_session_key(
    session_infos: list[dict[str, Any]], cli_channel: str, query: str
) -> str | None:
    """Resolve a /resume argument against a visible name or legacy session ID."""
    normalized = query.strip()
    if not normalized:
        return None
    matches = [
        str(info["key"])
        for info in session_infos
        if _cli_session_display_name(info, cli_channel) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    direct_key = f"{cli_channel}:{normalized}"
    return direct_key if any(str(info.get("key")) == direct_key for info in session_infos) else None


def _tui_command_palette() -> list[tuple[str, str, str]]:
    items = [
        (spec.command, spec.description, "edit" if spec.arg_hint else "submit")
        for spec in BUILTIN_COMMAND_SPECS
        if spec.command != "/new"
    ]
    items.extend([
        ("/rename", "Rename the current CLI session.", "edit"),
        ("/system-prompt", "Show the current topic's rendered system rules.", "submit"),
        ("/skin", "Switch the Windows Terminal background image.", "edit"),
        ("/clear", "Clear context and start an unnamed empty session.", "submit"),
        ("/resume", "Switch to a saved CLI session.", "edit"),
        ("/todos", "Show or clear the current topic todo list.", "submit"),
        ("/continue", "Continue the last interrupted task.", "submit"),
        ("/commit_memory", "Promote or preview pending memory consolidation.", "submit"),
        ("/exit", "Exit nanobot.", "submit"),
    ])
    seen: set[str] = set()
    deduped: list[tuple[str, str, str]] = []
    for command, description, action in items:
        if command in seen:
            continue
        seen.add(command)
        deduped.append((command, description, action))
    return sorted(deduped, key=lambda item: item[0].lower())


def _runtime_data_dir_for_workspace(workspace_path: Path) -> Path:
    """Return where nanobot runtime metadata should live for a workspace."""
    if is_default_workspace(workspace_path):
        return workspace_path
    return get_workspace_cache_dir(workspace_path)


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from nanobot.config.loader import load_config, resolve_config_env_vars, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    try:
        loaded = resolve_config_env_vars(load_config(config_path))
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    _warn_deprecated_config_keys(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


def _read_trigger_cli_message(message: str | None) -> str:
    """Read a trigger message from an argument or stdin."""
    if message and message.strip():
        return message
    try:
        if not sys.stdin.isatty():
            content = sys.stdin.read()
            if content.strip():
                return content
    except Exception:
        pass
    console.print("[red]Error: trigger message is required[/red]")
    raise typer.Exit(1)


def _warn_deprecated_config_keys(config_path: Path | None) -> None:
    """Hint users to remove obsolete keys from their config file."""
    import json

    from nanobot.config.loader import get_config_path

    path = config_path or get_config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "memoryWindow" in raw.get("agents", {}).get("defaults", {}):
        console.print(
            "[dim]Hint: `memoryWindow` in your config is no longer used "
            "and can be safely removed.[/dim]"
        )


def _load_inspection_config(
    config: str | None = None,
    workspace: str | None = None,
) -> tuple[Path, Config]:
    """Load config for diagnostic commands without resolving secret env refs."""
    from nanobot.config.loader import get_config_path, load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve(strict=False)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    display_path = config_path or get_config_path()
    try:
        loaded = load_config(config_path)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    _warn_deprecated_config_keys(display_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return display_path, loaded


def _confirm_webui_action(message: str, *, yes: bool) -> None:
    """Confirm a WebUI first-run mutation or fail clearly in non-interactive shells."""
    if yes:
        return
    try:
        interactive = sys.stdin.isatty()
    except Exception:
        interactive = False
    if not interactive:
        console.print(
            "[red]Error: WebUI setup needs confirmation. Re-run with --yes or use "
            "`nanobot onboard --wizard`.[/red]"
        )
        raise typer.Exit(1)
    if not typer.confirm(message, default=True):
        console.print("[yellow]WebUI setup cancelled.[/yellow]")
        raise typer.Exit(1)


def _resolve_webui_config_path(config: str | None) -> Path:
    """Resolve the config path used by ``nanobot webui`` and bind loader state."""
    from nanobot.config.loader import get_config_path, set_config_path

    if not config:
        return get_config_path()
    config_path = Path(config).expanduser().resolve(strict=False)
    set_config_path(config_path)
    console.print(f"[dim]Using config: {config_path}[/dim]")
    return config_path


def _load_webui_setup_config(config_path: Path) -> Config:
    """Load config for first-run mutation without resolving env-var placeholders."""
    from nanobot.config.loader import load_config

    try:
        return load_config(config_path)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


def _provider_setup_error(config: Config) -> str | None:
    """Return the provider setup error, or None when the current model can start."""
    from nanobot.config.loader import resolve_config_env_vars
    from nanobot.providers.factory import build_provider_snapshot

    try:
        build_provider_snapshot(resolve_config_env_vars(config.model_copy(deep=True)))
    except ValueError as exc:
        return str(exc)
    return None


def _webui_config_dict(config: Config) -> dict[str, Any]:
    """Return the current WebSocket config as a mutable alias-key dictionary."""
    from nanobot.channels.websocket import WebSocketConfig

    current = getattr(config.channels, "websocket", None) or {}
    model = WebSocketConfig.model_validate(current)
    return model.model_dump(by_alias=True, exclude_none=True)


def _host_for_local_browser(host: str) -> str:
    """Map bind hosts to a browser-openable local host."""
    if host in {"0.0.0.0", ""}:
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _webui_bootstrap_secret(config: Config) -> str:
    ws_cfg = _webui_config_dict(config)
    return str(ws_cfg.get("tokenIssueSecret") or ws_cfg.get("token") or "").strip()


def _webui_browser_url(config: Config) -> str:
    from urllib.parse import quote

    ws_cfg = _webui_config_dict(config)
    host = _host_for_local_browser(str(ws_cfg.get("host") or "127.0.0.1"))
    port = int(ws_cfg.get("port") or 8765)
    base_url = f"http://{host}:{port}"
    secret = _webui_bootstrap_secret(config)
    if not secret:
        return base_url
    return f"{base_url}/#/?bootstrapSecret={quote(secret, safe='')}"


def _webui_display_url(url: str) -> str:
    marker = "bootstrapSecret="
    if marker not in url:
        return url
    prefix, _ = url.split(marker, 1)
    return f"{prefix}{marker}<redacted>"


def _ensure_local_webui_channel(config: Config, *, port: int | None, yes: bool) -> tuple[bool, bool]:
    """Enable the local WebUI channel with safe localhost defaults."""
    from nanobot.channels.websocket import WebSocketConfig

    current = getattr(config.channels, "websocket", None) or {}
    model = WebSocketConfig.model_validate(current)
    changed = False
    generated_secret = False

    needs_enable = not model.enabled
    needs_port = port is not None and model.port != port
    needs_secret = not model.token_issue_secret.strip() and not model.token.strip()
    if not needs_enable and not needs_port and not needs_secret:
        return False, False

    target_port = port if port is not None else model.port
    console.print()
    console.print("[bold]Local WebUI setup[/bold]")
    console.print(f"  URL: [cyan]http://127.0.0.1:{target_port}[/cyan]")
    console.print("  Bind: [cyan]127.0.0.1 only[/cyan] (not exposed to your LAN)")
    console.print("  Auth: generated WebUI bootstrap secret stored in config")
    console.print(
        "  LAN access requires an explicit host change plus a WebUI password in config."
    )
    _confirm_webui_action("Update the local WebUI channel in this config?", yes=yes)

    if not model.enabled:
        model.enabled = True
        changed = True
    if model.host != "127.0.0.1":
        model.host = "127.0.0.1"
        changed = True
    if port is not None and model.port != port:
        model.port = port
        changed = True
    if not model.websocket_requires_token:
        model.websocket_requires_token = True
        changed = True
    if needs_secret:
        import secrets

        model.token_issue_secret = secrets.token_urlsafe(32)
        changed = True
        generated_secret = True

    setattr(config.channels, "websocket", model.model_dump(by_alias=True, exclude_none=True))
    return changed, generated_secret


def _warn_webui_bind_scope(config: Config) -> None:
    ws_cfg = _webui_config_dict(config)
    host = str(ws_cfg.get("host") or "127.0.0.1")
    if host in {"127.0.0.1", "localhost", "::1"}:
        return
    console.print(
        "[yellow]Warning: WebUI is configured to bind outside localhost. "
        "Keep tokenIssueSecret set and use this only on trusted networks.[/yellow]"
    )


def _wait_for_webui(url: str, *, timeout_s: float = 5.0) -> None:
    """Best-effort wait for the WebUI listener before opening a browser."""
    import socket
    import time
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)


def _open_webui_browser(url: str, *, wait: bool = True) -> None:
    """Open the WebUI in the user's default browser, with a copyable fallback."""
    import webbrowser

    if wait:
        _wait_for_webui(url)
    try:
        webbrowser.open(url)
        console.print(f"[green]✓[/green] Opened WebUI: [cyan]{url}[/cyan]")
    except Exception as exc:
        console.print(f"[yellow]Could not open browser ({exc}); visit {url}[/yellow]")


def _gateway_instance_command(
    subcommand: str,
    *,
    config_path: Path,
    workspace: str | None,
) -> str:
    """Return a copyable gateway command for the same config/workspace instance."""
    import shlex

    parts = ["nanobot", "gateway", subcommand, "--config", str(config_path)]
    if workspace:
        workspace_path = str(Path(workspace).expanduser().resolve(strict=False))
        parts.extend(["--workspace", workspace_path])
    return " ".join(shlex.quote(part) for part in parts)


def _run_quick_start_for_webui(config: Config, *, yes: bool) -> Config:
    """Offer the existing Quick Start flow when provider setup is missing."""
    if yes:
        console.print(
            "[red]Error: provider/model setup is incomplete, and --yes cannot answer "
            "provider credentials. Run `nanobot webui` interactively or "
            "`nanobot onboard --wizard`.[/red]"
        )
        raise typer.Exit(1)

    console.print()
    console.print("[yellow]Model provider setup is not ready.[/yellow]")
    console.print("Quick Start will ask for provider, API key/base URL, model, and WebUI password.")
    _confirm_webui_action("Run Quick Start now?", yes=False)

    from nanobot.cli.onboard import run_quick_start_onboard

    try:
        result = run_quick_start_onboard(config)
    except RuntimeError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        console.print("[yellow]Run `nanobot onboard --wizard` after installing wizard dependencies.[/yellow]")
        raise typer.Exit(1) from exc
    if not result.should_save:
        console.print("[yellow]Quick Start cancelled. No changes were saved.[/yellow]")
        raise typer.Exit(1)
    return result.config


def _migrate_cron_store(config: "Config") -> None:
    """One-time migration: move legacy global cron store into the workspace."""
    from nanobot.config.paths import get_cron_dir

    legacy_path = get_cron_dir() / "jobs.json"
    new_path = config.workspace_path / "cron" / "jobs.json"
    if legacy_path.is_file() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(str(legacy_path), str(new_path))


@app.command()
def trigger(
    trigger_id: str = typer.Argument(..., help="Trigger ID returned by /trigger"),
    message: str | None = typer.Argument(None, help="Message to deliver; stdin is used when omitted"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Deliver a local trigger message to its bound chat session."""
    from nanobot.triggers.local_store import (
        LocalTriggerStore,
        TriggerDisabledError,
        TriggerNotFoundError,
        TriggerStoreError,
    )

    runtime_config = _load_runtime_config(config, workspace)
    content = _read_trigger_cli_message(message)
    store = LocalTriggerStore(runtime_config.workspace_path)
    try:
        delivery = store.enqueue(trigger_id, content)
    except (TriggerNotFoundError, TriggerDisabledError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    except (TriggerStoreError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Queued[/green] {delivery.trigger_id} ({delivery.id})")


# ============================================================================
# OpenAI-Compatible API Server
# ============================================================================


@app.command()
def serve(
    port: int | None = typer.Option(None, "--port", "-p", help="API server port"),
    host: str | None = typer.Option(None, "--host", "-H", help="Bind address"),
    timeout: float | None = typer.Option(None, "--timeout", "-t", help="Per-request timeout (seconds)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show nanobot runtime logs"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the OpenAI-compatible API server (/v1/chat/completions)."""
    try:
        from aiohttp import web  # noqa: F401
    except ImportError:
        console.print("[red]aiohttp is required. Install with: nanobot plugins enable api[/red]")
        raise typer.Exit(1)

    from nanobot.api.server import create_app
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.image_generation import image_gen_provider_configs
    from nanobot.session.manager import SessionManager

    _set_nanobot_logs(verbose)

    runtime_config = _load_runtime_config(config, workspace)
    api_cfg = runtime_config.api
    host = host if host is not None else api_cfg.host
    port = port if port is not None else api_cfg.port
    timeout = timeout if timeout is not None else api_cfg.timeout
    api_key = api_cfg.api_key.strip() if api_cfg.api_key else ""
    if host in {"0.0.0.0", "::"} and not api_key:
        console.print(
            "[red]Error: host is 0.0.0.0 (all interfaces) but api_key is not set. "
            "Set api.api_key in config to prevent unauthenticated access.[/red]"
        )
        raise typer.Exit(1)
    # API sessions are often short-lived. Keep their cache directory persistent,
    # but do not pre-populate it with bootstrap templates or initialize GitStore.
    # SessionManager and agent subsystems create runtime directories lazily.
    data_dir = _runtime_data_dir_for_workspace(runtime_config.workspace_path)
    bus = MessageBus()
    session_manager = SessionManager(data_dir)
    try:
        agent_loop = AgentLoop.from_config(
            runtime_config, bus,
            data_dir=data_dir,
            session_manager=session_manager,
            image_generation_provider_configs=image_gen_provider_configs(runtime_config),
            hook_factories=[create_file_edit_activity_hook],
        )
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    model_name, preset_tag, reasoning_effort = _model_display(runtime_config)
    console.print(f"{__logo__} Starting OpenAI-compatible API server")
    console.print(f"  [cyan]Endpoint[/cyan] : http://{host}:{port}/v1/chat/completions")
    console.print(f"  [cyan]Model[/cyan]    : {model_name}{preset_tag}")
    if reasoning_effort:
        console.print(f"  [cyan]Reasoning[/cyan]: {reasoning_effort}")
    console.print("  [cyan]Session[/cyan]  : api:default")
    console.print(f"  [cyan]Timeout[/cyan]  : {timeout}s")
    if host in {"0.0.0.0", "::"}:
        console.print(
            "[yellow]API is bound to all interfaces "
            "(authentication required).[/yellow]"
        )
    console.print()

    api_app = create_app(
        agent_loop, model_name=model_name, request_timeout=timeout,
        api_key=api_key,
    )

    async def on_startup(_app):
        await agent_loop._connect_mcp()

    async def on_cleanup(_app):
        await agent_loop.close_mcp()

    api_app.on_startup.append(on_startup)
    api_app.on_cleanup.append(on_cleanup)

    web.run_app(api_app, host=host, port=port, print=lambda msg: logger.info(msg))


# ============================================================================
# WebUI Launcher
# ============================================================================


@app.command()
def webui(
    port: int | None = typer.Option(None, "--port", "-p", help="WebUI port"),
    gateway_port: int | None = typer.Option(
        None,
        "--gateway-port",
        help="Gateway health port",
    ),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    background: bool = typer.Option(False, "--background", help="Start gateway in the background"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Apply safe local WebUI defaults without prompting",
    ),
) -> None:
    """Prepare the local WebUI, start the gateway, and open the browser workbench."""
    from nanobot.config.loader import save_config
    from nanobot.gateway import GatewayRuntime, GatewayRuntimePaths, GatewayStartOptions

    config_path = _resolve_webui_config_path(config)
    created_config = not config_path.exists()
    if created_config:
        console.print(f"[yellow]No config found at {config_path}.[/yellow]")
        _confirm_webui_action("Create a nanobot config and workspace now?", yes=yes)

    setup_config = _load_webui_setup_config(config_path)
    if workspace:
        setup_config.agents.defaults.workspace = workspace

    provider_error = _provider_setup_error(setup_config)
    if provider_error:
        console.print(f"[dim]Provider check: {provider_error}[/dim]")
        setup_config = _run_quick_start_for_webui(setup_config, yes=yes)
        if workspace:
            setup_config.agents.defaults.workspace = workspace

    try:
        changed_webui, generated_bootstrap_secret = _ensure_local_webui_channel(
            setup_config,
            port=port,
            yes=yes,
        )
        _warn_webui_bind_scope(setup_config)
        webui_url = _webui_browser_url(setup_config)
    except ValueError as exc:
        console.print(f"[red]Error: invalid WebUI channel config: {exc}[/red]")
        raise typer.Exit(1) from exc

    if created_config or provider_error or changed_webui or workspace:
        save_config(setup_config, config_path)
        console.print(f"[green]✓[/green] Saved config: {config_path}")

    workspace_path = get_workspace_path(setup_config.workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace_path)

    runtime_config = _load_runtime_config(str(config_path), workspace)
    effective_gateway_port = gateway_port if gateway_port is not None else runtime_config.gateway.port

    console.print()
    console.print(f"WebUI: [cyan]{_webui_display_url(webui_url)}[/cyan]")
    console.print(f"Gateway health: [cyan]http://{runtime_config.gateway.host}:{effective_gateway_port}/health[/cyan]")
    if no_open:
        console.print("[dim]Browser opening disabled by --no-open.[/dim]")
        if generated_bootstrap_secret:
            console.print(
                "[yellow]A WebUI bootstrap secret was generated and saved in this config.[/yellow]"
            )
            console.print(
                "[dim]Open the WebUI and enter channels.websocket.tokenIssueSecret from "
                f"{config_path}, or rerun without --no-open to open the authenticated URL.[/dim]"
            )

    if background:
        config_arg = str(config_path)
        workspace_arg = str(Path(workspace).expanduser().resolve(strict=False)) if workspace else None
        runtime = GatewayRuntime(
            paths=GatewayRuntimePaths.for_instance(
                data_dir=config_path.parent,
                workspace=workspace_arg,
                config_path=config_arg,
            )
        )
        start_options = GatewayStartOptions(
            port=effective_gateway_port,
            workspace=workspace_arg,
            config_path=config_arg,
        )
        result = runtime.start_background(start_options)
        restarted = False
        restart_attempted = False
        if not result.ok and result.message == "gateway_already_running" and changed_webui:
            restart_attempted = True
            console.print("[yellow]WebUI config changed; restarting the background gateway.[/yellow]")
            result = runtime.restart(start_options, timeout_s=20)
            restarted = result.ok
        if not result.ok and (restart_attempted or result.message != "gateway_already_running"):
            action = "restarted" if restart_attempted else "started"
            console.print(f"[yellow]Gateway was not {action}: {result.message}[/yellow]")
            console.print(f"Logs: {result.status.log_path}")
            raise typer.Exit(1)
        if restarted:
            console.print("[green]Gateway restarted in the background.[/green]")
        elif result.ok:
            console.print("[green]Gateway started in the background.[/green]")
        else:
            console.print("[yellow]Gateway is already running in the background.[/yellow]")
        console.print(
            "Manage this instance: "
            f"[cyan]{_gateway_instance_command('status', config_path=config_path, workspace=workspace)}[/cyan]"
        )
        console.print(
            "View logs: "
            f"[cyan]{_gateway_instance_command('logs', config_path=config_path, workspace=workspace)}[/cyan]"
        )
        if not no_open:
            _open_webui_browser(webui_url)
        return

    _run_gateway(
        runtime_config,
        port=effective_gateway_port,
        open_browser_url=None if no_open else webui_url,
    )


# ============================================================================
# Gateway / Server
# ============================================================================


def _run_gateway(
    config: Config,
    *,
    port: int | None = None,
    open_browser_url: str | None = None,
    webui_static_dist: bool = True,
    webui_runtime_surface: str = "browser",
    webui_runtime_capabilities: dict[str, Any] | None = None,
    health_server_enabled: bool = True,
) -> None:
    """Shared gateway runtime; ``open_browser_url`` opens a tab once channels are up."""
    from nanobot.agent.tools.message import MessageTool
    from nanobot.bus.queue import MessageBus
    from nanobot.bus.runtime_events import RuntimeEventBus
    from nanobot.channels.manager import ChannelManager
    from nanobot.cron.bound_runner import run_bound_cron_job
    from nanobot.cron.service import CronJobSkippedError, CronService
    from nanobot.cron.session_turns import is_bound_cron_job
    from nanobot.cron.types import CronJob
    from nanobot.providers.factory import build_provider_snapshot, load_provider_snapshot
    from nanobot.providers.image_generation import image_gen_provider_configs
    from nanobot.session.manager import SessionManager
    from nanobot.session.webui_turns import WebuiTurnCoordinator
    from nanobot.triggers.local_runner import run_local_trigger_queue
    from nanobot.triggers.local_store import LocalTriggerStore
    from nanobot.webui.token_usage import TokenUsageHook

    port = port if port is not None else config.gateway.port

    console.print(f"{__logo__} Starting nanobot gateway version {__version__} on port {port}...")
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    runtime_events = RuntimeEventBus()
    try:
        provider_snapshot = build_provider_snapshot(config)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    session_manager = SessionManager(config.workspace_path)

    # Self-heal the gateway state file with the current PID after any restart.
    from nanobot.config.loader import get_config_path
    from nanobot.gateway.runtime import GatewayRuntime, GatewayRuntimePaths

    config_path = str(get_config_path().resolve(strict=False))
    GatewayRuntime.refresh_state_pid(
        paths=GatewayRuntimePaths.for_instance(
            workspace=str(config.workspace_path)
            if not is_default_workspace(config.workspace_path)
            else None,
            config_path=config_path,
        )
    )

    # Preserve existing single-workspace installs, but keep custom workspaces clean.
    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with workspace-scoped store
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron = CronService(cron_store_path)
    trigger_store = LocalTriggerStore(config.workspace_path)

    # Create agent with cron service
    agent = AgentLoop.from_config(
        config, bus,
        provider=provider_snapshot.provider,
        model=provider_snapshot.model,
        context_window_tokens=provider_snapshot.context_window_tokens,
        cron_service=cron,
        session_manager=session_manager,
        image_generation_provider_configs=image_gen_provider_configs(config),
        provider_snapshot_loader=load_provider_snapshot,
        runtime_events=runtime_events,
        provider_signature=provider_snapshot.signature,
        hooks=[TokenUsageHook(timezone_name=config.agents.defaults.timezone)],
        local_trigger_store=trigger_store,
        hook_factories=[create_file_edit_activity_hook],
    )
    WebuiTurnCoordinator(
        bus=bus,
        sessions=session_manager,
        schedule_background=lambda coro: agent._schedule_background(coro),
    ).subscribe(runtime_events)
    from nanobot.bus.events import OutboundMessage
    from nanobot.session.keys import session_key_for_channel

    def _channel_session_key(channel: str, chat_id: str) -> str:
        return session_key_for_channel(
            channel,
            chat_id,
            unified_session=config.agents.defaults.unified_session,
        )

    async def _deliver_to_channel(
        msg: OutboundMessage, *, record: bool = False, session_key: str | None = None,
    ) -> None:
        """Publish a user-visible message and mirror it into that channel's session."""
        metadata = dict(msg.metadata or {})
        record = record or bool(metadata.pop("_record_channel_delivery", False))
        if metadata != (msg.metadata or {}):
            msg = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=msg.content,
                reply_to=msg.reply_to,
                media=msg.media,
                metadata=metadata,
                buttons=msg.buttons,
            )
        if (
            record
            and msg.channel != "cli"
            and msg.content.strip()
            and hasattr(session_manager, "get_or_create")
            and hasattr(session_manager, "save")
        ):
            key = session_key or _channel_session_key(msg.channel, msg.chat_id)
            session = session_manager.get_or_create(key)
            extra: dict[str, Any] = {"_channel_delivery": True}
            if msg.media:
                extra["media"] = list(msg.media)
            session.add_message("assistant", msg.content, **extra)
            session_manager.save(session)
        await bus.publish_outbound(msg)

    message_tool = getattr(agent, "tools", {}).get("message")
    if isinstance(message_tool, MessageTool):
        message_tool.set_send_callback(_deliver_to_channel)

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        async def _silent(*_args, **_kwargs):
            pass

        # Dream is an internal job — run directly, not through the agent loop.
        if job.name == "dream":
            from nanobot.agent.memory import MemoryStore

            dream_session_key = MemoryStore.dream_session_key
            build_dream_commit_message = MemoryStore.build_dream_commit_message
            prune_dream_sessions = MemoryStore.prune_dream_sessions

            store = agent.context.memory
            resp = None
            diff_body = ""
            try:
                result = store.build_dream_prompt()
                if result is None:
                    logger.info("Dream: nothing to process")
                    return None
                prompt, last_cursor = result
                key = dream_session_key()
                resp = await agent.process_direct(
                    prompt,
                    session_key=key,
                    ephemeral=True,
                    tools=store.build_dream_tools(),
                    on_progress=_silent,
                )
                # Ground truth: the real file delta, not the LLM's self-report.
                diff_body = store.dream_content_diff()
                productive = bool(diff_body) or (
                    not store.git.is_initialized()
                    and MemoryStore.dream_run_completed(resp)
                )
                if productive:
                    store.set_last_dream_cursor(last_cursor)
                    logger.info("Dream cron job completed, cursor advanced to {}", last_cursor)
                elif MemoryStore.dream_run_completed(resp):
                    logger.info(
                        "Dream cron job completed with no memory changes; "
                        "cursor not advanced",
                    )
                else:
                    logger.warning(
                        "Dream cron job did not complete; cursor remains at {}",
                        store.get_last_dream_cursor(),
                    )
            except Exception:
                logger.exception("Dream cron job failed")
            finally:
                from nanobot.webui.token_usage import record_response_token_usage

                record_response_token_usage(
                    resp,
                    source="dream",
                    timezone_name=config.agents.defaults.timezone,
                )
                if store.git.is_initialized():
                    msg = build_dream_commit_message(
                        "dream: periodic memory consolidation", diff_body,
                    )
                    sha = store.git.auto_commit(msg)
                    if sha:
                        logger.info("Dream commit: {}", sha)
                store.compact_history()
                prune_dream_sessions(agent.sessions.sessions_dir)
            return None

        # Heartbeat is a system job that checks HEARTBEAT.md for active tasks.
        if job.name == "heartbeat":
            heartbeat_file = config.workspace_path / "HEARTBEAT.md"
            try:
                content = heartbeat_file.read_text(encoding="utf-8")
            except OSError:
                logger.debug("Heartbeat: HEARTBEAT.md missing")
                return None
            if not _heartbeat_has_active_tasks(content):
                logger.debug("Heartbeat: HEARTBEAT.md has no active tasks")
                return None

            channel, chat_id = _pick_heartbeat_target()
            if channel == "cli":
                return None

            prompt = (
                _HEARTBEAT_PREAMBLE
                + f"Review the following HEARTBEAT.md and report any active tasks:\n\n{content}"
            )

            # Internal check: funnel all output through the post-run gate so the
            # turn can't deliver directly via the message tool and skip it.
            suppress_token = None
            if isinstance(message_tool, MessageTool):
                suppress_token = message_tool.set_suppress_delivery(True)
            try:
                resp = await agent.process_direct(
                    prompt,
                    session_key="heartbeat",
                    channel=channel,
                    chat_id=chat_id,
                    on_progress=_silent,
                )
            finally:
                if isinstance(message_tool, MessageTool) and suppress_token is not None:
                    message_tool.reset_suppress_delivery(suppress_token)
            response = resp.content if resp else ""

            # Keep a small tail of heartbeat history so the loop stays bounded.
            session = agent.sessions.get_or_create("heartbeat")
            session.retain_recent_legal_suffix(hb_cfg.keep_recent_messages)
            agent.sessions.save(session)

            if not response:
                return None

            # Fail closed: stay silent on evaluator failure instead of notifying.
            should_notify = await evaluate_response(
                response, prompt, agent.provider, agent.model,
                default_notify=False,
            )
            if should_notify:
                logger.info("Heartbeat: completed, delivering response")
                await _deliver_to_channel(
                    OutboundMessage(channel=channel, chat_id=chat_id, content=response),
                    record=True,
                )
            else:
                logger.info("Heartbeat: silenced by post-run evaluation")
            return response

        if is_bound_cron_job(job):
            return await run_bound_cron_job(job, agent=agent, cron=cron)

        reason = "unbound agent cron job must be recreated from a chat session"
        logger.warning(
            "Cron: skipped unbound agent job '{}' ({}): {}",
            job.name,
            job.id,
            reason,
        )
        raise CronJobSkippedError(reason)

    cron.on_job = on_cron_job

    def _webui_runtime_model_name() -> str | None:
        model = getattr(agent, "model", None)
        if isinstance(model, str):
            stripped = model.strip()
            return stripped or None
        return None

    # Create channel manager (forwards SessionManager so the WebSocket channel
    # can serve the embedded webui's REST surface).
    channels = ChannelManager(
        config,
        bus,
        session_manager=session_manager,
        cron_service=cron,
        local_trigger_store=trigger_store,
        webui_runtime_model_name=_webui_runtime_model_name,
        webui_cron_pending_job_ids=getattr(agent, "pending_cron_job_ids_for_session", None),
        webui_local_trigger_pending_ids=getattr(
            agent,
            "pending_local_trigger_ids_for_session",
            None,
        ),
        webui_static_dist=webui_static_dist,
        webui_runtime_surface=webui_runtime_surface,
        webui_runtime_capabilities=webui_runtime_capabilities,
    )

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        sidebar_state = read_webui_sidebar_state()
        return _pick_heartbeat_target_from_sessions(
            enabled_channels=channels.enabled_channels,
            sessions=session_manager.list_sessions(),
            archived_keys=sidebar_state.get("archived_keys", []),
        )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    hb_cfg = config.gateway.heartbeat
    if hb_cfg.enabled:
        console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")
    else:
        console.print("[yellow]✗[/yellow] Heartbeat: disabled")

    async def _health_server(host: str, health_port: int):
        """Lightweight HTTP health endpoint on the gateway port."""
        import json as _json

        async def handle(reader, writer):
            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=5)
            except (asyncio.TimeoutError, ConnectionError):
                writer.close()
                return

            request_line = data.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
            method, path = "", ""
            parts = request_line.split(" ")
            if len(parts) >= 2:
                method, path = parts[0], parts[1]

            if method == "GET" and path == "/health":
                body = _json.dumps({"status": "ok"})
                resp = (
                    f"HTTP/1.0 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n{body}"
                )
            else:
                body = "Not Found"
                resp = (
                    f"HTTP/1.0 404 Not Found\r\n"
                    f"Content-Type: text/plain\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n{body}"
                )

            writer.write(resp.encode())
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, host, health_port)
        console.print(f"[green]✓[/green] Health endpoint: http://{host}:{health_port}/health")
        async with server:
            await server.serve_forever()
    # Register Dream system job (idempotent on restart)
    from nanobot.cron.types import CronJob, CronPayload, CronSchedule
    dream_cfg = config.agents.defaults.dream
    if dream_cfg.enabled:
        cron.register_system_job(CronJob(
            id="dream",
            name="dream",
            schedule=dream_cfg.build_schedule(config.agents.defaults.timezone),
            payload=CronPayload(kind="system_event"),
        ))
        console.print(f"[green]✓[/green] Dream: {dream_cfg.describe_schedule()}")
    else:
        console.print("[yellow]○[/yellow] Dream: disabled")
        _advance_dream_cursor_if_behind(agent.context.memory)

    # Register Heartbeat system job (idempotent on restart)
    if hb_cfg.enabled:
        cron.register_system_job(CronJob(
            id="heartbeat",
            name="heartbeat",
            schedule=CronSchedule(
                kind="every",
                every_ms=hb_cfg.interval_s * 1000,
                tz=config.agents.defaults.timezone,
            ),
            payload=CronPayload(kind="system_event"),
        ))

    async def _open_browser_when_ready() -> None:
        """Wait for the gateway to bind, then point the user's browser at the webui."""
        if not open_browser_url:
            return
        import webbrowser
        from urllib.parse import urlparse

        parsed = urlparse(open_browser_url)
        target_host = parsed.hostname or config.gateway.host or "127.0.0.1"
        target_port = parsed.port or port
        # Channels start asynchronously; a short poll lets us avoid racing the bind.
        for _ in range(40):  # ~4s max
            try:
                reader, writer = await asyncio.open_connection(
                    target_host,
                    target_port,
                )
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(0.1)
        try:
            webbrowser.open(open_browser_url)
            console.print(f"[green]✓[/green] Opened browser at {open_browser_url}")
        except Exception as e:
            console.print(f"[yellow]Could not open browser ({e}); visit {open_browser_url}[/yellow]")

    async def run():
        tasks: list[asyncio.Task] = []
        shutdown_task: asyncio.Task | None = None
        runtime_tasks: asyncio.Future | None = None
        runtime_tasks_drained = False
        shutdown_event = asyncio.Event()
        _ensure_gateway_tty_signal_mode()
        restore_shutdown_handlers = _install_gateway_shutdown_handlers(
            asyncio.get_running_loop(),
            shutdown_event,
            tasks,
            console.print,
        )
        try:
            await cron.start()
            tasks = [
                asyncio.create_task(agent.run(), name="nanobot-agent-loop"),
                asyncio.create_task(channels.start_all(), name="nanobot-channels"),
                asyncio.create_task(
                    run_local_trigger_queue(
                        store=trigger_store,
                        submit_turn=getattr(agent, "submit_local_trigger_turn", None),
                    ),
                    name="nanobot-local-triggers",
                ),
            ]
            if health_server_enabled:
                tasks.append(asyncio.create_task(
                    _health_server(config.gateway.host, port),
                    name="nanobot-health-server",
                ))
            if open_browser_url:
                tasks.append(asyncio.create_task(
                    _open_browser_when_ready(),
                    name="nanobot-open-browser",
                ))
            runtime_tasks = asyncio.gather(*tasks)
            shutdown_task = asyncio.create_task(
                shutdown_event.wait(),
                name="nanobot-gateway-shutdown",
            )
            done, _pending = await asyncio.wait(
                {runtime_tasks, shutdown_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if runtime_tasks in done:
                runtime_tasks_drained = True
                await runtime_tasks
            elif runtime_tasks is not None:
                runtime_tasks.cancel()
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        except Exception:
            import traceback

            console.print("\n[red]Error: Gateway crashed unexpectedly[/red]")
            console.print(traceback.format_exc())
        finally:
            try:
                if shutdown_task and not shutdown_task.done():
                    shutdown_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shutdown_task
                cron.stop()
                agent.stop()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if runtime_tasks is not None and not runtime_tasks_drained:
                    with suppress(asyncio.CancelledError, Exception):
                        await runtime_tasks
                await channels.stop_all()
                # Flush all cached sessions to durable storage before exit.
                # This prevents data loss on filesystems with write-back
                # caching (rclone VFS, NFS, FUSE mounts, etc.).
                flushed = agent.sessions.flush_all()
                if flushed:
                    logger.info("Shutdown: flushed {} session(s) to disk", flushed)
            finally:
                restore_shutdown_handlers()

    asyncio.run(run())


app.add_typer(
    create_gateway_app(
        console=console,
        log_handler_id=_log_handler_id,
        load_runtime_config=_load_runtime_config,
        run_gateway=_run_gateway,
    ),
    name="gateway",
)


# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"),
):
    """Interact with the agent directly."""
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.service import CronService
    from nanobot.providers.image_generation import image_gen_provider_configs

    if workspace is None:
        workspace = str(Path.cwd())
    config = _load_runtime_config(config, workspace)

    # For non-default workspaces, store nanobot metadata in the central cache dir
    # (~/.nanobot/caches/<name>_<hash>/) so project directories stay clean.
    data_dir = _runtime_data_dir_for_workspace(config.workspace_path)
    sync_workspace_templates(data_dir)

    bus = MessageBus()

    # Preserve existing single-workspace installs, but keep custom workspaces clean.
    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with data_dir-scoped store
    cron_store_path = data_dir / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    _set_nanobot_logs(logs)

    try:
        agent_loop = AgentLoop.from_config(
            config, bus,
            data_dir=data_dir,
            cron_service=cron,
            image_generation_provider_configs=image_gen_provider_configs(config),
            hook_factories=[create_file_edit_activity_hook],
        )
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    restart_notice = consume_restart_notice_from_env()
    if restart_notice and should_show_cli_restart_notice(restart_notice, session_id):
        _print_agent_response(
            format_restart_completed_message(restart_notice.started_at_raw),
            render_markdown=False,
        )

    # Shared reference for progress callbacks
    _thinking: ThinkingSpinner | None = None

    def _make_progress(renderer: StreamRenderer | None = None):
        reasoning_buffer = _ReasoningBuffer()

        async def _cli_progress(content: str, *, tool_hint: bool = False, reasoning: bool = False, **_kwargs: Any) -> None:
            ch = agent_loop.channels_config

            if _kwargs.get("reasoning_end"):
                if ch and not ch.show_reasoning:
                    reasoning_buffer.clear()
                else:
                    _flush_cli_reasoning(reasoning_buffer, _thinking, renderer)
                return

            if reasoning:
                if ch and not ch.show_reasoning:
                    reasoning_buffer.clear()
                    return
                text = reasoning_buffer.add(content)
                if text:
                    _print_cli_reasoning(text, _thinking, renderer)
                return
            if ch and tool_hint and not ch.send_tool_hints:
                return
            if ch and not tool_hint and not ch.send_progress:
                return
            _print_cli_progress_line(content, _thinking, renderer)
        return _cli_progress

    if message:
        # Single message mode — direct call, no bus needed
        async def run_once():
            renderer = StreamRenderer(
                render_markdown=markdown,
                bot_name=config.agents.defaults.bot_name,
                bot_icon=config.agents.defaults.bot_icon,
            )
            response = await agent_loop.process_direct(
                message, session_id,
                on_progress=_make_progress(renderer),
                on_stream=renderer.on_delta,
                on_stream_end=renderer.on_end,
            )
            if not renderer.streamed:
                await renderer.close()
                print_kwargs: dict[str, Any] = {}
                if renderer.header_printed:
                    print_kwargs["show_header"] = False
                _print_agent_response(
                    response.content if response else "",
                    render_markdown=markdown,
                    metadata=response.metadata if response else None,
                    **print_kwargs,
                )
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — split-pane TUI (output pane + persistent input line)
        from nanobot.bus.events import InboundMessage
        from nanobot.config.paths import get_cli_history_path
        from nanobot.fork.cli.tui_factory import create_tui

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        history_file = str(get_cli_history_path())

        async def run_interactive():
            tui = create_tui(
                render_markdown=markdown,
                history_file=history_file,
                model=agent_loop.model,
                reasoning_effort=getattr(agent_loop.provider.generation, "reasoning_effort", None),
                backend=config.agents.defaults.tui_backend,
                skin_enabled=config.agents.defaults.tui_skin_enabled,
                workspace=agent_loop.workspace,
            )

            # A fresh CLI start is intentionally unnamed. Mark it in memory so
            # the first persisted user turn has a stable display name, but do
            # not save it yet or unused launches would clutter /resume.
            fresh_chat_id = f"session_{uuid.uuid4().hex}"
            fresh_session = agent_loop.sessions.get_or_create(f"{cli_channel}:{fresh_chat_id}")
            _mark_cli_session_unnamed(fresh_session)
            topic_state: dict[str, str] = {"chat_id": fresh_chat_id}
            tui.set_input_history_topic(fresh_session.key)

            def _load_topic(name: str) -> None:
                """Load session history and context estimate for the given topic."""
                s = agent_loop.sessions.get_or_create(f"{cli_channel}:{name}")
                # Honor config flag: optionally flush any leftover pending
                # consolidation summary into MEMORY.md on (re)load. Default
                # off keeps the prompt cache warm across restarts.
                if (
                    config.agents.defaults.promote_pending_on_restart
                    and s.pending_consolidation_summary
                ):
                    chars = len(s.pending_consolidation_summary)
                    try:
                        agent_loop.memory_consolidator.promote_pending_summary(s.key)
                        tui.add_system(
                            f"已自动 promote 上次的 pending consolidation "
                            f"({chars} chars) 到 MEMORY.md"
                        )
                    except Exception:
                        pass
                display_name = _cli_session_display_name(
                    {"key": s.key, "metadata": s.metadata}, cli_channel
                )
                tui.set_input_history_topic(s.key)
                tui.set_topic("" if display_name == _CLI_UNNAMED_SESSION_LABEL else display_name or name)
                tui.load_session_history(
                    agent_loop.sessions.display_history(s.key, s.messages),
                    tool_registry=agent_loop.tools,
                    workspace=agent_loop.workspace,
                )
                tui.set_todos(s.todos)
                if agent_loop.context_window_tokens:
                    async def _estimate_context_usage(session=s) -> None:
                        try:
                            ctx_est, _ = await asyncio.to_thread(
                                agent_loop.memory_consolidator.estimate_session_prompt_tokens,
                                session,
                            )
                            tui.update_context_usage(ctx_est, agent_loop.context_window_tokens)
                        except Exception:
                            pass

                    agent_loop._schedule_background(_estimate_context_usage())
                # Fork(perf): warm this topic's lazy caches (skills/memory/BM25)
                # in the background so the first turn doesn't pay cold-start cost.
                # _load_topic runs inside the async run_interactive loop, so a
                # running event loop is guaranteed here.
                agent_loop._schedule_background(
                    agent_loop.warmup_caches(f"{cli_channel}:{name}")
                )

            tui.set_commands(_tui_command_palette())

            def _topic_popup_items(session_infos: list[dict[str, Any]]) -> list[tuple[str, str]]:
                items: list[tuple[str, str]] = []
                for info in session_infos:
                    key = str(info.get("key") or "")
                    display_name = _cli_session_display_name(info, cli_channel)
                    if display_name is None:
                        continue
                    size = _topic_cache_size_bytes(
                        data_dir=agent_loop.context.data_dir,
                        session_key=key,
                        session_path=str(info.get("path") or "") or None,
                        transcript_path=str(info.get("transcript_path") or "") or None,
                    )
                    items.append((key, _format_topic_popup_label(display_name, size)))
                return items

            # Show startup topic picker if existing sessions are available
            async def _startup_picker() -> None:
                await asyncio.sleep(0.05)
                sessions_list = agent_loop.sessions.list_sessions()
                topics = [
                    _cli_session_display_name(s, cli_channel)
                    for s in sessions_list
                    if _cli_session_display_name(s, cli_channel) is not None
                ]
                if topics:
                    tui.show_topic_popup(_topic_popup_items(sessions_list), _switch_topic)

            asyncio.create_task(_startup_picker())

            bus_task = asyncio.create_task(agent_loop.run())
            is_processing = False
            pending_queue: list[str] = []
            _pre_submitted: list[bool] = [False]
            _turn_cancelled: list[bool] = [False]
            _todo_bar_waiting_for_new_plan: list[bool] = [False]

            # Override signals so Ctrl+C / SIGTERM cleanly exit the TUI
            def _tui_signal(signum, frame):  # noqa: ARG001
                tui.exit()

            signal.signal(signal.SIGINT, _tui_signal)
            signal.signal(signal.SIGTERM, _tui_signal)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, _tui_signal)
            if hasattr(signal, "SIGPIPE"):
                signal.signal(signal.SIGPIPE, signal.SIG_IGN)

            def _cancel_current() -> None:
                nonlocal is_processing
                if not is_processing:
                    return
                _turn_cancelled[0] = True
                session_key = f"{cli_channel}:{topic_state['chat_id']}"
                tasks = agent_loop._active_tasks.get(session_key, [])
                for task in list(tasks):
                    task.cancel()
                tui.flush_stream()
                is_processing = False
                tui.set_is_processing(False)
                pending_queue.clear()
                tui.add_system("已取消当前请求。")

            tui.set_on_cancel(_cancel_current)

            def _set_activity_phase(phase: str) -> None:
                setter = getattr(tui, "set_activity_phase", None)
                if callable(setter):
                    setter(phase)

            def _pre_submit(text: str) -> None:
                if not is_processing and not _is_cli_local_command(text):
                    tui.add_user_echo(text)
                    tui.stream_start()
                    _pre_submitted[0] = True

            async def _send_message(text: str) -> None:
                nonlocal is_processing
                _turn_cancelled[0] = False
                is_processing = True
                tui.set_is_processing(True)
                _set_activity_phase("prepare_turn")
                # Hide stale todos at the start of a fresh user turn. Keep the
                # persisted list intact so the model still sees prior context;
                # TodoWriteTool will make the bar visible again when it publishes
                # a new plan for this turn.
                _todo_bar_waiting_for_new_plan[0] = False
                try:
                    s = agent_loop.sessions.get_or_create(
                        f"{cli_channel}:{topic_state['chat_id']}"
                    )
                    if _todos_all_completed(s.todos):
                        s.todos = []
                        agent_loop.sessions.save(s)
                        tui.set_todos([])
                    elif _should_hide_stale_todos_on_new_turn(s.todos):
                        tui.set_todos([])
                        _todo_bar_waiting_for_new_plan[0] = True
                except Exception:
                    pass
                if _pre_submitted[0]:
                    _pre_submitted[0] = False
                else:
                    tui.add_user_echo(text)
                    tui.stream_start()
                # Exceed prompt_toolkit's max_render_postpone_time (0.01s) so the
                # thinking animation is guaranteed to reach the screen before the
                # bus receives the message and the LLM starts responding.
                await asyncio.sleep(0.015)
                _set_activity_phase("agent_processing")
                await bus.publish_inbound(InboundMessage(
                    channel=cli_channel,
                    sender_id="user",
                    chat_id=topic_state["chat_id"],
                    content=text,
                    metadata={"_wants_stream": True},
                ))

            async def _turn_complete() -> None:
                nonlocal is_processing
                is_processing = False
                tui.set_is_processing(False)
                _set_activity_phase("idle")
                # Fork: stop any thinking/idle spinner on turn completion. The
                # streaming path stops it via pop_stream, but the non-streaming
                # reply path (add_response) had no such hook — an idle spinner
                # scheduled after the last tool call would otherwise spin forever.
                tui.stop_thinking()
                usage = agent_loop._last_usage
                if usage and agent_loop.context_window_tokens:
                    # Providers normalize prompt_tokens as the complete request
                    # input. Cache fields are a subset/breakdown, never an
                    # additional amount to add to context usage.
                    tui.update_context_usage(
                        usage.get("prompt_tokens", 0),
                        agent_loop.context_window_tokens,
                    )
                # Refresh the todo bar unless this turn never produced a fresh
                # TodoWrite update. In that case keep the old plan hidden so a
                # stale in-progress badge does not reappear above the input.
                try:
                    s = agent_loop.sessions.get_or_create(f"{cli_channel}:{topic_state['chat_id']}")
                    if not _todo_bar_waiting_for_new_plan[0]:
                        tui.set_todos(s.todos)
                except Exception:
                    pass
                if pending_queue:
                    await _send_message(pending_queue.pop(0))

            async def _switch_topic(session_key: str) -> None:
                """Switch to a persisted session key, keeping its display name separate."""
                prefix = f"{cli_channel}:"
                if not session_key.startswith(prefix):
                    return
                # Fork: drop the previous topic's per-session learning state so
                # the loop's learning dicts don't grow unbounded as topics pile up.
                old_key = f"{cli_channel}:{topic_state['chat_id']}"
                if old_key != session_key:
                    agent_loop.clear_session_learning(old_key)
                _todo_bar_waiting_for_new_plan[0] = False
                topic_state["chat_id"] = session_key[len(prefix):]
                tui.reset_history()
                _load_topic(topic_state["chat_id"])

            async def _clear_context() -> None:
                """Leave the old session intact and enter a persisted unnamed one."""
                fresh_id = f"session_{uuid.uuid4().hex}"
                fresh = agent_loop.sessions.get_or_create(f"{cli_channel}:{fresh_id}")
                _mark_cli_session_unnamed(fresh)
                agent_loop.sessions.save(fresh)
                await _switch_topic(fresh.key)
                tui.set_topic("")
                tui.add_system("已清空上下文，当前为空白未命名会话。使用 /rename 命名，或 /resume 切换会话。")

            async def _rename_current(title: str) -> None:
                name = title.strip()
                if not name:
                    tui.add_system("会话名称不能为空。")
                    return
                current_key = f"{cli_channel}:{topic_state['chat_id']}"
                for info in agent_loop.sessions.list_sessions():
                    if str(info.get("key")) == current_key:
                        continue
                    if _cli_session_display_name(info, cli_channel) == name:
                        tui.add_system(f"会话名称已存在: {name}")
                        return
                session = agent_loop.sessions.get_or_create(current_key)
                session.metadata["cli_title"] = name
                session.metadata.pop("cli_unnamed", None)
                agent_loop.sessions.save(session)
                tui.set_topic(name)
                tui.add_system(f"已重命名当前会话: {name}")

            async def _on_submit(user_input: str) -> None:
                text = user_input.strip()
                if not text:
                    return
                if _is_exit_command(text):
                    tui.exit()
                    return

                # ── 话题管理命令 ──────────────────────────────────────────────
                if text == "/skills":
                    tui.add_system(_format_skills_command(agent_loop.context.skills))
                    return

                if text == "/system-prompt":
                    session = agent_loop.sessions.get_or_create(
                        f"{cli_channel}:{topic_state['chat_id']}"
                    )
                    inspection_msg = InboundMessage(
                        channel=cli_channel,
                        sender_id="user",
                        chat_id=topic_state["chat_id"],
                        content="",
                        metadata={"_prompt_inspection": True},
                    )
                    scope = agent_loop.workspace_scopes.for_message(
                        inspection_msg, session.metadata
                    )
                    history = session.get_history(
                        max_messages=agent_loop._max_messages,
                        max_tokens=agent_loop._replay_token_budget(),
                        extend_to_user=False,
                    )
                    pending_summary = None
                    summary_meta = session.metadata.get("_last_summary")
                    if isinstance(summary_meta, dict):
                        from nanobot.agent.autocompact import AutoCompact

                        pending_summary = AutoCompact._format_summary(
                            str(summary_meta["text"]),
                            datetime.fromisoformat(str(summary_meta["last_active"])),
                        )
                    messages = agent_loop.context.build_messages(
                        history=history,
                        current_message="",
                        channel=cli_channel,
                        chat_id=topic_state["chat_id"],
                        sender_id="user",
                        session_summary=pending_summary,
                        session_metadata=session.metadata,
                        learning_ctx=agent_loop._build_learning_ctx(session.key),
                        todos=session.todos,
                        workspace=scope.project_path,
                        runtime_state=agent_loop,
                        inbound_message=inspection_msg,
                        session_key=session.key,
                        unified_session=agent_loop._unified_session,
                    )
                    tui.add_system(_format_prompt_inspection(messages))
                    return

                if text == "/skin" or text.startswith("/skin "):
                    from nanobot.cli.terminal_skin import (
                        SkinError,
                        current_background,
                        find_terminal_settings,
                        format_skin_list,
                        list_skin_images,
                        switch_skin,
                    )

                    def _switch_terminal_skin(selector: str) -> None:
                        try:
                            selected, settings, _backup = switch_skin(selector)
                        except (SkinError, OSError, ValueError) as exc:
                            tui.add_system(f"切换终端背景失败: {exc}")
                            return
                        tui.add_system(
                            f"已切换 Windows Terminal 背景图: {selected.name}\n"
                            f"配置: {settings}"
                        )

                    arg = text[len("/skin"):].strip()
                    try:
                        images = list_skin_images()
                        current = current_background(find_terminal_settings())
                    except (SkinError, OSError, ValueError) as exc:
                        tui.add_system(f"读取终端背景失败: {exc}")
                        return
                    if arg.casefold() == "list":
                        tui.add_system(format_skin_list(images, current))
                        return
                    if arg:
                        _switch_terminal_skin(arg)
                        return

                    async def _on_skin_select(selector: str) -> None:
                        _switch_terminal_skin(selector)

                    items = [
                        (
                            image.name,
                            f"{'* ' if image == current else '  '}{index}. {image.name}",
                        )
                        for index, image in enumerate(images, 1)
                    ]
                    tui.show_topic_popup(items, _on_skin_select)
                    return

                if text == "/clear":
                    if is_processing:
                        tui.add_system("请等待当前响应完成后再清空上下文。")
                        return
                    await _clear_context()
                    return

                if text == "/rename" or text.startswith("/rename "):
                    if is_processing:
                        tui.add_system("请等待当前响应完成后再重命名会话。")
                        return
                    name = text[len("/rename"):].strip()
                    if name:
                        await _rename_current(name)
                    else:
                        tui.enter_new_topic_mode(_rename_current)
                    return

                if text == "/continue":
                    # Resume work after the previous turn hit the iteration
                    # ceiling. Sends a synthetic "继续" user message so the
                    # LLM picks up with full prior context (tools + history
                    # already in session.messages).
                    if is_processing:
                        tui.add_system("当前还在处理消息，请稍后再 /continue。")
                        return
                    await asyncio.sleep(0)
                    await _send_message("请继续上次中断的任务。")
                    return

                if text == "/commit_memory" or text.startswith("/commit_memory "):
                    # /commit_memory       → promote pending → MEMORY.md
                    # /commit_memory show  → preview pending content (no write)
                    arg = text[len("/commit_memory"):].strip()
                    key = f"{cli_channel}:{topic_state['chat_id']}"
                    s = agent_loop.sessions.get_or_create(key)
                    if not s.pending_consolidation_summary:
                        tui.add_system("当前话题暂无 pending consolidation summary。")
                        return
                    chars = len(s.pending_consolidation_summary)
                    if arg == "show":
                        tui.add_system(
                            f"Pending consolidation summary ({chars} chars):\n\n"
                            f"{s.pending_consolidation_summary}"
                        )
                        return
                    agent_loop.memory_consolidator.promote_pending_summary(key)
                    tui.add_system(
                        f"已将 pending summary ({chars} chars) 写入 MEMORY.md "
                        "(下次 LLM 调用 system prompt 会重建，cache 会失效一次)。"
                    )
                    return

                if text == "/todos" or text.startswith("/todos "):
                    from nanobot.fork.agent.tools.todo import format_todos
                    arg = text[len("/todos"):].strip()
                    s = agent_loop.sessions.get_or_create(
                        f"{cli_channel}:{topic_state['chat_id']}"
                    )
                    if arg == "clear":
                        s.todos = []
                        agent_loop.sessions.save(s)
                        _todo_bar_waiting_for_new_plan[0] = False
                        tui.set_todos([])
                        tui.add_system("已清空当前话题的 todo 列表。")
                        return
                    if not s.todos:
                        tui.add_system("当前话题暂无 todo。")
                        return
                    total = len(s.todos)
                    done = sum(1 for t in s.todos if t.get("status") == "completed")
                    header = f"Todos ({done}/{total}):"
                    tui.add_system(f"{header}\n{format_todos(s.todos)}")
                    return

                if text == "/resume" or text.startswith("/resume "):
                    if is_processing:
                        tui.add_system("请等待当前响应完成后再切换话题。")
                        return
                    arg = text[7:].strip()
                    sessions_list = agent_loop.sessions.list_sessions()
                    if arg:
                        session_key = _resolve_cli_session_key(sessions_list, cli_channel, arg)
                        if session_key is None:
                            tui.add_system(f"未找到会话: {arg}")
                            return
                        await _switch_topic(session_key)
                        tui.add_system(f"已切换到会话: {arg}")
                    else:
                        # /resume — interactive picker
                        items = _topic_popup_items(sessions_list)
                        if not items:
                            tui.add_system("当前没有已保存的会话。")
                            return

                        async def _on_topic_select(session_key: str) -> None:
                            await _switch_topic(session_key)
                            tui.add_system("已切换会话。")

                        tui.show_topic_popup(items, _on_topic_select)
                    return
                # ─────────────────────────────────────────────────────────────

                if is_processing:
                    pending_queue.append(user_input)
                    tui.add_system("Message queued — will send when nanobot finishes.")
                else:
                    await asyncio.sleep(0)
                    await _send_message(user_input)

            tui.set_on_submit(_on_submit)
            tui.set_on_pre_submit(_pre_submit)

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        event = outbound_event_from_message(msg)

                        if isinstance(event, StreamDeltaEvent) or msg.metadata.get("_stream_delta"):
                            if not _turn_cancelled[0]:
                                tui.stream_delta(msg.content)
                            continue

                        if isinstance(event, StreamEndEvent) or msg.metadata.get("_stream_end"):
                            if not _turn_cancelled[0] and ((isinstance(event, StreamEndEvent) and event.resuming) or msg.metadata.get("_resuming")):
                                # Flush LLM chunk, then keep animation alive during tool exec
                                tui.flush_stream()
                                tui.tool_phase_start()
                            continue

                        if isinstance(event, StreamedResponseEvent) or msg.metadata.get("_streamed"):
                            if _turn_cancelled[0]:
                                _turn_cancelled[0] = False
                                continue
                            streamed = tui.pop_stream()
                            intermediate = tui.flush_accumulator()
                            if msg.metadata.get("_no_content"):
                                final = streamed or ""
                            else:
                                final = streamed or msg.content or ""
                            content = "\n\n".join(p for p in [intermediate, final] if p.strip())
                            if content.strip():
                                tui.add_response(content, dict(msg.metadata or {}))
                            await _turn_complete()
                            continue

                        if msg.metadata.get("_progress"):
                            if not _turn_cancelled[0]:
                                # Reasoning trace (LLM thinking content). Render
                                # below the thinking spinner, then flush to
                                # history on _reasoning_end. Must branch
                                # *before* add_progress: otherwise reasoning
                                # would race with stream_delta over _stream_cache
                                # and the UI flickers visibly.
                                # hasattr guard: not every TUI backend ships
                                # reasoning hooks; missing methods used to
                                # raise AttributeError, killing _consume_outbound
                                # entirely and freezing the UI at the spinner.
                                if msg.metadata.get("_reasoning_end") or msg.metadata.get("_reasoning_delta"):
                                    # Fork: reasoning (思考链) 显示已屏蔽 — 直接丢弃,
                                    # 不分发到 TUI。必须在 add_progress 之前显式拦截:
                                    # 否则带 _reasoning_delta 的消息(content 为思考链
                                    # 文本)会掉到下面的 add_progress 分支被当成普通进度
                                    # 渲染出来。TUI 侧 add_reasoning/flush_reasoning
                                    # 方法仍保留,日后想恢复只需还原本分支。
                                    pass
                                elif msg.metadata.get("_file_edit_events"):
                                    tui.add_file_edit_events(
                                        msg.metadata.get("_file_edit_events") or []
                                    )
                                else:
                                    summary = _tool_result_summary_from_events(
                                        msg.metadata.get("_tool_events")
                                    )
                                    if summary is not None:
                                        tui.add_tool_result(summary)
                                    elif msg.content:
                                        if msg.metadata.get("_tool_result"):
                                            tui.add_tool_result(msg.content)
                                        else:
                                            tui.add_progress(msg.content)
                            continue

                        # Interactive question popup pushed by AskUserTool.
                        # We dispatch the user's selection back to the tool's
                        # awaiting Future via the module-level deliver_reply.
                        if msg.metadata.get("_ask_user"):
                            from nanobot.fork.agent.tools.ask_user import deliver_reply
                            correlation_id = str(msg.metadata.get("_ask_user_id") or "")
                            questions = msg.metadata.get("_ask_user_questions") or []

                            async def _on_question_complete(
                                answers: dict[str, str] | None,
                                _cid: str = correlation_id,
                            ) -> None:
                                if answers is None:
                                    deliver_reply(_cid, None, cancelled=True)
                                else:
                                    deliver_reply(_cid, answers)

                            try:
                                tui.show_question_popup(questions, _on_question_complete)
                            except Exception:
                                # Don't leave the tool hanging — report failure
                                deliver_reply(correlation_id, None, cancelled=True)
                            continue

                        # Live system message (e.g. todo diff pushed by TodoWriteTool).
                        # Must be displayed mid-turn, not aggregated into the final reply.
                        if msg.metadata.get("_system_message"):
                            clear_initial = getattr(tui, "clear_initial_thinking", None)
                            if callable(clear_initial):
                                clear_initial()
                            if msg.content:
                                tui.add_system(msg.content)
                            # Also refresh the bar in case todos changed.
                            try:
                                s = agent_loop.sessions.get_or_create(
                                    f"{cli_channel}:{topic_state['chat_id']}"
                                )
                                _todo_bar_waiting_for_new_plan[0] = False
                                tui.set_todos(s.todos)
                            except Exception:
                                pass
                            continue

                        if msg.metadata.get("_error"):
                            try:
                                tui.pop_stream()
                                tui.flush_accumulator()
                            except Exception:
                                pass
                            content = msg.content or "nanobot task failed. See this topic's runtime.log for details."
                            tui.add_response(content, dict(msg.metadata or {}))
                            if is_processing:
                                await _turn_complete()
                            continue

                        # Non-streaming response (or unsolicited push from cron etc.)
                        if msg.content:
                            tui.add_response(msg.content, dict(msg.metadata or {}))
                        if is_processing:
                            await _turn_complete()

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        # Never let an unhandled exception kill the consumer:
                        # if the task dies, _stream_delta / _streamed messages
                        # stop being consumed and the UI freezes at the
                        # spinner.  Log and continue so the user always gets
                        # the response (lost the offending message at worst).
                        logger.exception("Outbound consumer recovered from error")
                        continue

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                await tui.run_async()
            except (KeyboardInterrupt, EOFError):
                pass
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()
                print("\033]0;nanobot\007", end="", flush=True)

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show channel status."""
    from nanobot.channels.registry import discover_all

    _, loaded = _load_inspection_config(config=config)

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled")

    for name, cls in sorted(discover_all().items()):
        section = getattr(loaded.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            "[green]\u2713[/green]" if enabled else "[dim]\u2717[/dim]",
        )

    console.print(table)


@channels_app.command("login")
def channels_login(
    channel_name: str = typer.Argument(..., help="Channel name (e.g. weixin, whatsapp)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Authenticate with a channel via QR code or other interactive login."""
    from nanobot.channels.registry import discover_all

    _, loaded = _load_inspection_config(config=config)
    channel_cfg = getattr(loaded.channels, channel_name, None) or {}

    # Validate channel exists
    all_channels = discover_all()
    if channel_name not in all_channels:
        available = ", ".join(all_channels.keys())
        console.print(f"[red]Unknown channel: {channel_name}[/red]  Available: {available}")
        raise typer.Exit(1)

    console.print(f"{__logo__} {all_channels[channel_name].display_name} Login\n")

    channel_cls = all_channels[channel_name]
    channel = channel_cls(channel_cfg, bus=None)

    success = asyncio.run(channel.login(force=force))

    if not success:
        raise typer.Exit(1)


# ============================================================================
# Plugin Commands
# ============================================================================

plugins_app = typer.Typer(help="Manage optional nanobot features")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """List optional nanobot features."""
    from nanobot.channels.registry import discover_channel_names, discover_plugins
    from nanobot.config.loader import load_config, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)

    _print_enable_options(
        feature_support.optional_dependency_groups(),
        set(discover_channel_names()),
        discover_plugins(),
        load_config(resolved_config_path),
    )


@plugins_app.command("enable")
def plugins_enable(
    name: str = typer.Argument(..., help="Feature name (e.g. weixin, matrix, pdf)"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show optional package install logs"),
):
    """Enable a nanobot feature."""
    from nanobot.config.loader import get_config_path, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)
    resolved_config_path = resolved_config_path or get_config_path()
    _set_nanobot_logs(logs)

    try:
        payload = feature_support.enable_optional_feature(
            name,
            config_path=resolved_config_path,
            runner=feature_support.run_install_command,
        )
    except feature_support.OptionalFeatureError as exc:
        console.print(f"[red]{escape(exc.message)}[/red]")
        raise typer.Exit(1) from exc

    message = payload.get("last_action", {}).get("message") or f"Enabled feature '{name}'"
    console.print(f"[green]{escape(message)}[/green]")


@plugins_app.command("disable")
def plugins_disable(
    name: str = typer.Argument(..., help="Channel name (e.g. telegram, matrix, slack)"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Disable a nanobot channel feature."""
    from nanobot.config.loader import get_config_path, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)
    resolved_config_path = resolved_config_path or get_config_path()

    try:
        payload = feature_support.disable_optional_feature(name, config_path=resolved_config_path)
    except feature_support.OptionalFeatureError as exc:
        console.print(f"[red]{escape(exc.message)}[/red]")
        raise typer.Exit(1) from exc

    message = payload.get("last_action", {}).get("message") or f"Disabled channel '{name}'"
    console.print(f"[green]{escape(message)}[/green] in {resolved_config_path}")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
):
    """Show nanobot status."""
    config_path, loaded = _load_inspection_config(config=config, workspace=workspace)
    workspace_path = loaded.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(
        f"Workspace: {workspace_path} "
        f"{'[green]✓[/green]' if workspace_path.exists() else '[red]✗[/red]'}"
    )

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        _model, _preset_tag, _reasoning_effort = _model_display(loaded)
        console.print(f"Model: {_model}{_preset_tag}")
        if _reasoning_effort:
            console.print(f"Reasoning: {_reasoning_effort}")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(loaded.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, Callable[[], None]] = {}
_LOGOUT_HANDLERS: dict[str, Callable[[], None]] = {}

_PROVIDER_DISPLAY: dict[str, str] = {
    "openai_codex": "OpenAI Codex",
    "github_copilot": "GitHub Copilot",
}

_OAUTH_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai_codex": "openai-codex/gpt-5.4-mini",
    "github_copilot": "github-copilot/gpt-5.4-mini",
}


def _register_login(name: str):
    """Register an OAuth login handler."""
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn

    return decorator


def _register_logout(name: str):
    """Register an OAuth logout handler."""
    def decorator(fn):
        _LOGOUT_HANDLERS[name] = fn
        return fn
    return decorator


def _resolve_oauth_provider(provider: str):
    """Resolve and validate an OAuth provider configuration."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)
    return spec


def _set_oauth_provider_as_main(
    provider_name: str,
    *,
    model: str | None = None,
    config_path: str | None = None,
) -> None:
    """Persist an OAuth provider as the active agent provider."""
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)
        console.print(f"[dim]Using config: {resolved_config_path}[/dim]")

    config = load_config(resolved_config_path)
    selected_model = (model or "").strip() or _OAUTH_PROVIDER_DEFAULT_MODELS[provider_name]
    config.agents.defaults.model_preset = None
    config.agents.defaults.provider = provider_name
    config.agents.defaults.model = selected_model
    save_config(config, resolved_config_path)

    saved_path = resolved_config_path or get_config_path()
    console.print(
        f"[green]✓ Set {provider_name.replace('_', '-')} as the main provider[/green]  "
        f"[dim]{selected_model}[/dim]"
    )
    console.print(f"[dim]Saved: {saved_path}[/dim]")


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
    set_main: bool = typer.Option(
        False,
        "--set-main",
        "--main",
        help="Set this OAuth provider as the active agent provider after login",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use when setting this provider as the active provider",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Authenticate with an OAuth provider."""
    spec = _resolve_oauth_provider(provider)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()
    if set_main or model:
        _set_oauth_provider_as_main(spec.name, model=model, config_path=config)


@provider_app.command("logout")
def provider_logout(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Log out from an OAuth provider."""
    spec = _resolve_oauth_provider(provider)

    handler = _LOGOUT_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Logout not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Logout - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive

        from nanobot.config.loader import load_config, resolve_config_env_vars

        proxy = None
        try:
            proxy = resolve_config_env_vars(load_config()).providers.openai_codex.proxy or None
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        token = None
        with suppress(Exception):
            token = call_with_optional_proxy(get_token, proxy=proxy)
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = call_with_optional_proxy(
                login_oauth_interactive,
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
                proxy=proxy,
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_logout("openai_codex")
def _logout_openai_codex() -> None:
    """Clear local OAuth credentials for OpenAI Codex."""
    try:
        from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER
        from oauth_cli_kit.storage import FileTokenStorage
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)

    storage = FileTokenStorage(token_filename=OPENAI_CODEX_PROVIDER.token_filename)
    _delete_oauth_files(storage.get_token_path(), _PROVIDER_DISPLAY["openai_codex"])


@_register_logout("github_copilot")
def _logout_github_copilot() -> None:
    """Clear local OAuth credentials for GitHub Copilot."""
    try:
        from nanobot.providers.github_copilot_provider import get_storage
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)

    storage = get_storage()
    _delete_oauth_files(storage.get_token_path(), _PROVIDER_DISPLAY["github_copilot"])


def _delete_oauth_files(token_path: Path, provider_label: str) -> None:
    """Delete OAuth token and lock files, reporting the result."""
    removed_paths: list[Path] = []
    skipped: list[tuple[Path, OSError]] = []
    for path in (token_path, token_path.with_suffix(".lock")):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            skipped.append((path, exc))
            continue
        removed_paths.append(path)

    if not removed_paths and not skipped:
        console.print(f"[yellow]! No local OAuth credentials found for {provider_label}[/yellow]")
        return

    if removed_paths:
        console.print(f"[green]✓ Logged out from {provider_label}[/green]")
        for path in removed_paths:
            console.print(f"[dim]Removed: {path}[/dim]")
    for path, exc in skipped:
        console.print(f"[yellow]! Could not remove {path}: {exc}[/yellow]")


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    try:
        from nanobot.providers.github_copilot_provider import login_github_copilot

        console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")
        token = login_github_copilot(
            print_fn=lambda s: console.print(s),
            prompt_fn=lambda s: typer.prompt(s),
        )
        account = token.account_id or "GitHub"
        console.print(f"[green]✓ Authenticated with GitHub Copilot[/green]  [dim]{account}[/dim]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)


@_register_login("claude_ai")
def _login_claude_ai() -> None:
    from nanobot.fork.providers.claude_ai_oauth_provider import (
        CLAUDE_CODE_CRED_FILE,
        save_oauth_token,
    )

    # Try to import from Claude Code credentials silently
    if CLAUDE_CODE_CRED_FILE.exists():
        try:
            import json as _json
            data = _json.loads(CLAUDE_CODE_CRED_FILE.read_text(encoding="utf-8"))
            token = (data.get("claudeAiOauth") or {}).get("accessToken", "")
            if token:
                save_oauth_token(token)
                console.print("[green]✓ OAuth token imported from Claude Code credentials[/green]")
                return
        except Exception:
            pass

    # Fall back to manual token entry
    console.print("[cyan]Paste your Claude.ai OAuth access token below.[/cyan]")
    console.print("[dim]Tip: run 'claude login' first, then the token is in ~/.claude/.credentials.json[/dim]\n")
    token = typer.prompt("Access token", hide_input=True)
    if not token.strip():
        console.print("[red]✗ No token provided[/red]")
        raise typer.Exit(1)
    save_oauth_token(token.strip())
    console.print("[green]✓ Claude.ai OAuth token saved[/green]")


# ============================================================================
# Cache Management
# ============================================================================

cache_app = typer.Typer(help="Manage nanobot workspace caches")
app.add_typer(cache_app, name="cache")


@cache_app.command("migrate")
def cache_migrate(
    old_path: str = typer.Argument(..., help="Old workspace directory path"),
    new_path: str = typer.Argument(..., help="New workspace directory path"),
):
    """Migrate workspace cache when a project directory is moved or renamed.

    Computes the cache directory names for both paths and renames the old
    cache to the new one, preserving all sessions, memory, and history.
    """
    import shutil

    from nanobot.config.paths import get_workspace_cache_dir

    old = Path(old_path).expanduser().resolve()
    new = Path(new_path).expanduser().resolve()

    old_cache = get_workspace_cache_dir(old)
    new_cache = get_workspace_cache_dir(new)

    if old_cache == new_cache:
        console.print("[yellow]Both paths resolve to the same cache directory — nothing to do.[/yellow]")
        raise typer.Exit(0)

    if not old_cache.exists():
        console.print(f"[red]No cache found for:[/red] {old}")
        console.print(f"[dim]Expected: {old_cache}[/dim]")
        raise typer.Exit(1)

    if new_cache.exists():
        console.print(f"[red]Cache already exists for new path:[/red] {new}")
        console.print(f"[dim]{new_cache}[/dim]")
        console.print("[dim]Remove it first if you want to overwrite.[/dim]")
        raise typer.Exit(1)

    shutil.move(str(old_cache), str(new_cache))
    console.print(f"[green]✓ Cache migrated[/green]")
    console.print(f"  [dim]{old_cache}[/dim]")
    console.print(f"  [dim]→ {new_cache}[/dim]")


@cache_app.command("list")
def cache_list():
    """List all workspace caches and their associated paths."""
    caches_root = Path.home() / ".nanobot" / "caches"
    if not caches_root.exists() or not any(caches_root.iterdir()):
        console.print("[dim]No workspace caches found.[/dim]")
        return

    console.print(f"[bold]Workspace caches[/bold] [dim]({caches_root})[/dim]\n")
    for d in sorted(caches_root.iterdir()):
        if not d.is_dir():
            continue
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1_048_576
        console.print(f"  [cyan]{d.name}[/cyan]  [dim]{size_mb:.1f} MB[/dim]")


if __name__ == "__main__":
    app()
