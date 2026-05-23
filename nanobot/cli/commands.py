"""CLI commands for nanobot."""

import asyncio
from contextlib import contextmanager, nullcontext
from datetime import datetime

import os
import select
import signal
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import typer
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.cli.stream import StreamRenderer, ThinkingSpinner
from nanobot.config.paths import get_workspace_path, is_default_workspace
from nanobot.config.schema import Config
from nanobot.utils.helpers import sync_workspace_templates

app = typer.Typer(
    name="nanobot",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

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

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    from nanobot.config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _make_console() -> Console:
    return Console(file=sys.stdout)


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=True,
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
) -> None:
    """Render assistant response with consistent terminal styling."""
    from datetime import datetime
    console = _make_console()
    content = response or ""
    body = _response_renderable(content, render_markdown, metadata)
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
    return Markdown(content)


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


def _print_cli_progress_line(text: str, thinking: ThinkingSpinner | None) -> None:
    """Print a CLI progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        console.print(f"  [dim]↳ {text}[/dim]")


async def _print_interactive_progress_line(text: str, thinking: ThinkingSpinner | None) -> None:
    """Print an interactive progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        await _print_interactive_line(text)


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


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


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    wizard: bool = typer.Option(False, "--wizard", help="Use interactive wizard"),
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
            console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
            console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
            console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
            if typer.confirm("Overwrite?"):
                config = _apply_workspace_override(Config())
                save_config(config, config_path)
                console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
            else:
                config = _apply_workspace_override(load_config(config_path))
                save_config(config, config_path)
                console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
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
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")


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


def _make_provider(config: Config):
    """Create the appropriate LLM provider from config.

    Routing is driven by ``ProviderSpec.backend`` in the registry.
    """
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    # --- validation ---
    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            console.print("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            console.print("Set them in ~/.nanobot/config.json under providers.azure_openai section")
            console.print("Use the model field to specify the deployment name.")
            raise typer.Exit(1)
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            console.print("[red]Error: No API key configured.[/red]")
            console.print("Set one in ~/.nanobot/config.json under providers section")
            raise typer.Exit(1)

    # --- instantiation by backend ---
    if backend == "claude_ai_oauth":
        from nanobot.fork.providers.claude_ai_oauth_provider import ClaudeAIOAuthProvider
        provider = ClaudeAIOAuthProvider(
            default_model=model,
            api_base=config.get_api_base(model),
            extra_headers=p.extra_headers if p else None,
        )
    elif backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider
        provider = OpenAICodexProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider
        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from nanobot.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    _warn_deprecated_config_keys(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


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


def _migrate_cron_store(config: "Config") -> None:
    """One-time migration: move legacy global cron store into the workspace."""
    from nanobot.config.paths import get_cron_dir

    legacy_path = get_cron_dir() / "jobs.json"
    new_path = config.workspace_path / "cron" / "jobs.json"
    if legacy_path.is_file() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(legacy_path), str(new_path))


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
        console.print("[red]aiohttp is required. Install with: pip install 'nanobot-ai[api]'[/red]")
        raise typer.Exit(1)

    from loguru import logger
    from nanobot.agent.loop import AgentLoop
    from nanobot.api.server import create_app
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager

    if verbose:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    runtime_config = _load_runtime_config(config, workspace)
    api_cfg = runtime_config.api
    host = host if host is not None else api_cfg.host
    port = port if port is not None else api_cfg.port
    timeout = timeout if timeout is not None else api_cfg.timeout
    sync_workspace_templates(runtime_config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(runtime_config)
    session_manager = SessionManager(runtime_config.workspace_path)
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=runtime_config.workspace_path,
        model=runtime_config.agents.defaults.model,
        max_iterations=runtime_config.agents.defaults.max_tool_iterations,
        context_window_tokens=runtime_config.agents.defaults.context_window_tokens,
        web_search_config=runtime_config.tools.web.search,
        web_proxy=runtime_config.tools.web.proxy or None,
        exec_config=runtime_config.tools.exec,
        restrict_to_workspace=runtime_config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=runtime_config.tools.mcp_servers,
        channels_config=runtime_config.channels,
        timezone=runtime_config.agents.defaults.timezone,
        enable_learning=runtime_config.agents.defaults.enable_learning,
    )

    model_name = runtime_config.agents.defaults.model
    console.print(f"{__logo__} Starting OpenAI-compatible API server")
    console.print(f"  [cyan]Endpoint[/cyan] : http://{host}:{port}/v1/chat/completions")
    console.print(f"  [cyan]Model[/cyan]    : {model_name}")
    console.print("  [cyan]Session[/cyan]  : api:default")
    console.print(f"  [cyan]Timeout[/cyan]  : {timeout}s")
    if host in {"0.0.0.0", "::"}:
        console.print(
            "[yellow]Warning:[/yellow] API is bound to all interfaces. "
            "Only do this behind a trusted network boundary, firewall, or reverse proxy."
        )
    console.print()

    api_app = create_app(agent_loop, model_name=model_name, request_timeout=timeout)

    async def on_startup(_app):
        await agent_loop._connect_mcp()

    async def on_cleanup(_app):
        await agent_loop.close_mcp()

    api_app.on_startup.append(on_startup)
    api_app.on_cleanup.append(on_cleanup)

    web.run_app(api_app, host=host, port=port, print=lambda msg: logger.info(msg))


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the nanobot gateway."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.manager import ChannelManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    from nanobot.session.manager import SessionManager

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    config = _load_runtime_config(config, workspace)
    port = port if port is not None else config.gateway.port

    console.print(f"{__logo__} Starting nanobot gateway version {__version__} on port {port}...")
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Preserve existing single-workspace installs, but keep custom workspaces clean.
    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with workspace-scoped store
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        enable_learning=config.agents.defaults.enable_learning,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        from nanobot.agent.tools.cron import CronTool
        from nanobot.agent.tools.message import MessageTool
        from nanobot.utils.evaluator import evaluate_response

        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        cron_tool = agent.tools.get("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)
        try:
            resp = await agent.process_direct(
                reminder_note,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
        finally:
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)

        response = resp.content if resp else ""

        message_tool = agent.tools.get("message")
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return response

        if job.payload.deliver and job.payload.to and response:
            should_notify = await evaluate_response(
                response, job.payload.message, provider, agent.model,
            )
            if should_notify:
                from nanobot.bus.events import OutboundMessage
                await bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=response,
                ))
        return response
    cron.on_job = on_cron_job

    # Create channel manager
    channels = ChannelManager(config, bus)

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(channels.enabled_channels)
        # Prefer the most recently updated non-internal session on an enabled channel.
        for item in session_manager.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        # Fallback keeps prior behavior but remains explicit.
        return "cli", "direct"

    # Create heartbeat service
    async def on_heartbeat_execute(tasks: str) -> str:
        """Phase 2: execute heartbeat tasks through the full agent loop."""
        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs):
            pass

        resp = await agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

        # Keep a small tail of heartbeat history so the loop stays bounded
        # without losing all short-term context between runs.
        session = agent.sessions.get_or_create("heartbeat")
        session.retain_recent_legal_suffix(hb_cfg.keep_recent_messages)
        agent.sessions.save(session)

        return resp.content if resp else ""

    async def on_heartbeat_notify(response: str) -> None:
        """Deliver a heartbeat response to the user's channel."""
        from nanobot.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return  # No external channel available to deliver to
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
        timezone=config.agents.defaults.timezone,
    )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        except Exception:
            import traceback
            console.print("\n[red]Error: Gateway crashed unexpectedly[/red]")
            console.print(traceback.format_exc())
        finally:
            await agent.close_mcp()
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())




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
    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.service import CronService

    if workspace is None:
        workspace = str(Path.cwd())
    config = _load_runtime_config(config, workspace)

    # For non-default workspaces, store nanobot metadata in the central cache dir
    # (~/.nanobot/caches/<name>_<hash>/) so project directories stay clean.
    if is_default_workspace(config.workspace_path):
        data_dir = config.workspace_path
    else:
        from nanobot.config.paths import get_workspace_cache_dir
        data_dir = get_workspace_cache_dir(config.workspace_path)
    sync_workspace_templates(data_dir)

    bus = MessageBus()
    provider = _make_provider(config)

    # Preserve existing single-workspace installs, but keep custom workspaces clean.
    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with data_dir-scoped store
    cron_store_path = data_dir / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        data_dir=data_dir,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        enable_learning=config.agents.defaults.enable_learning,
        pending_promote_threshold_chars=config.agents.defaults.pending_promote_threshold_chars,
    )

    # Shared reference for progress callbacks
    _thinking: ThinkingSpinner | None = None

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        _print_cli_progress_line(content, _thinking)

    if message:
        # Single message mode — direct call, no bus needed
        async def run_once():
            renderer = StreamRenderer(render_markdown=markdown)
            response = await agent_loop.process_direct(
                message, session_id,
                on_progress=_cli_progress,
                on_stream=renderer.on_delta,
                on_stream_end=renderer.on_end,
            )
            if not renderer.streamed:
                await renderer.close()
                _print_agent_response(
                    response.content if response else "",
                    render_markdown=markdown,
                    metadata=response.metadata if response else None,
                )
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — split-pane TUI (output pane + persistent input line)
        from nanobot.bus.events import InboundMessage
        from nanobot.fork.cli.tui_factory import create_tui
        from nanobot.config.paths import get_cli_history_path

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        history_file = str(get_cli_history_path())

        async def run_interactive():
            tui = create_tui(
                render_markdown=markdown,
                history_file=history_file,
                model=config.agents.defaults.model,
                backend=config.agents.defaults.tui_backend,
            )

            # Mutable topic state — fresh session by default; user picks via startup popup
            fresh_chat_id = datetime.now().strftime("topic_%Y%m%d_%H%M%S")
            topic_state: dict[str, str] = {"chat_id": fresh_chat_id}

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
                tui.set_topic(name)
                tui.load_session_history(
                    s.messages,
                    tool_registry=agent_loop.tools,
                    workspace=agent_loop.workspace,
                )
                tui.set_todos(s.todos)
                if agent_loop.context_window_tokens:
                    try:
                        ctx_est, _ = agent_loop.memory_consolidator.estimate_session_prompt_tokens(s)
                        tui.update_context_usage(ctx_est, agent_loop.context_window_tokens)
                    except Exception:
                        pass

            tui.set_commands([
                ("/new", "新建话题"),
                ("/resume", "切换/恢复话题"),
                ("/todos", "查看当前话题的 todo 列表"),
                ("/continue", "继续上次因达到 iteration 上限中断的任务"),
                ("/commit_memory", "把 pending consolidation 写入 MEMORY.md"),
                ("/exit", "退出 nanobot"),
            ])

            # Show startup topic picker if existing sessions are available
            async def _startup_picker() -> None:
                await asyncio.sleep(0.05)
                sessions_list = agent_loop.sessions.list_sessions()
                prefix = f"{cli_channel}:"
                topics = [
                    s["key"][len(prefix):]
                    for s in sessions_list
                    if s["key"].startswith(prefix)
                ]
                if topics:
                    options = ["[ 新建话题 ]"] + topics
                    async def _on_startup_select(name: str) -> None:
                        if name == "[ 新建话题 ]":
                            async def _confirm_new_topic(typed: str) -> None:
                                new_name = typed or datetime.now().strftime("topic_%Y%m%d_%H%M%S")
                                await _switch_topic(new_name)
                                tui.add_system(f"已创建话题: {new_name}")
                            tui.enter_new_topic_mode(_confirm_new_topic)
                        else:
                            await _switch_topic(name)
                    tui.show_topic_popup(options, _on_startup_select)

            asyncio.create_task(_startup_picker())

            bus_task = asyncio.create_task(agent_loop.run())
            is_processing = False
            pending_queue: list[str] = []
            _pre_submitted: list[bool] = [False]
            _turn_cancelled: list[bool] = [False]

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

            def _pre_submit(text: str) -> None:
                if not is_processing:
                    tui.add_user_echo(text)
                    tui.stream_start()
                    _pre_submitted[0] = True

            async def _send_message(text: str) -> None:
                nonlocal is_processing
                _turn_cancelled[0] = False
                is_processing = True
                tui.set_is_processing(True)
                # If the previous task finished and all todos are completed,
                # auto-clear them so the bar resets for the new task rather
                # than carrying a stale "✓ all N done" badge across turns.
                try:
                    s = agent_loop.sessions.get_or_create(
                        f"{cli_channel}:{topic_state['chat_id']}"
                    )
                    if s.todos and all(t.get("status") == "completed" for t in s.todos):
                        s.todos = []
                        agent_loop.sessions.save(s)
                        tui.set_todos([])
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
                usage = agent_loop._last_usage
                if usage and agent_loop.context_window_tokens:
                    ctx_used = (
                        usage.get("prompt_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0)
                    )
                    tui.update_context_usage(ctx_used, agent_loop.context_window_tokens)
                # Refresh the todo bar (diff messages are pushed live by TodoWriteTool
                # via the bus, not aggregated here).
                try:
                    s = agent_loop.sessions.get_or_create(f"{cli_channel}:{topic_state['chat_id']}")
                    tui.set_todos(s.todos)
                except Exception:
                    pass
                if pending_queue:
                    await _send_message(pending_queue.pop(0))

            async def _switch_topic(name: str) -> None:
                topic_state["chat_id"] = name
                tui.reset_history()
                _load_topic(name)

            async def _on_submit(user_input: str) -> None:
                text = user_input.strip()
                if not text:
                    return
                if _is_exit_command(text):
                    tui.exit()
                    return

                # ── 话题管理命令 ──────────────────────────────────────────────
                if text.startswith("/new"):
                    if is_processing:
                        tui.add_system("请等待当前响应完成后再新建话题。")
                        return
                    parts = text.split(None, 1)
                    if len(parts) > 1 and parts[1].strip():
                        # /new <name> — name provided directly, no need for input mode
                        name = parts[1].strip()
                        await _switch_topic(name)
                        tui.add_system(f"已创建并切换到话题: {name}")
                    else:
                        # /new with no args — enter topic-name input mode
                        async def _confirm_new_topic_cmd(typed: str) -> None:
                            name = typed or datetime.now().strftime("topic_%Y%m%d_%H%M%S")
                            await _switch_topic(name)
                            tui.add_system(f"已创建并切换到话题: {name}")
                        tui.enter_new_topic_mode(_confirm_new_topic_cmd)
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

                if text.startswith("/resume"):
                    if is_processing:
                        tui.add_system("请等待当前响应完成后再切换话题。")
                        return
                    arg = text[7:].strip()
                    if arg:
                        # /resume <name> — direct switch
                        await _switch_topic(arg)
                        tui.add_system(f"已切换到话题: {arg}")
                    else:
                        # /resume — interactive picker
                        sessions_list = agent_loop.sessions.list_sessions()
                        prefix = f"{cli_channel}:"
                        topics = [
                            s["key"][len(prefix):]
                            for s in sessions_list
                            if s["key"].startswith(prefix)
                        ]
                        if not topics:
                            tui.add_system("当前没有已保存的话题。")
                            return

                        async def _on_topic_select(name: str) -> None:
                            await _switch_topic(name)
                            tui.add_system(f"已切换到话题: {name}")

                        tui.show_topic_popup(topics, _on_topic_select)
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

                        if msg.metadata.get("_stream_delta"):
                            if not _turn_cancelled[0]:
                                tui.stream_delta(msg.content)
                            continue

                        if msg.metadata.get("_stream_end"):
                            if not _turn_cancelled[0] and msg.metadata.get("_resuming"):
                                # Flush LLM chunk, then keep animation alive during tool exec
                                tui.flush_stream()
                                tui.tool_phase_start()
                            continue

                        if msg.metadata.get("_streamed"):
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
                            if msg.content and not _turn_cancelled[0]:
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
                            if msg.content:
                                tui.add_system(msg.content)
                            # Also refresh the bar in case todos changed.
                            try:
                                s = agent_loop.sessions.get_or_create(
                                    f"{cli_channel}:{topic_state['chat_id']}"
                                )
                                tui.set_todos(s.todos)
                            except Exception:
                                pass
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
def channels_status():
    """Show channel status."""
    from nanobot.channels.registry import discover_all
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")

    for name, cls in sorted(discover_all().items()):
        section = getattr(config.channels, name, None)
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


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess

    # User's bridge location
    from nanobot.config.paths import get_bridge_install_dir

    user_bridge = get_bridge_install_dir()

    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    # Check for npm
    npm_path = shutil.which("npm")
    if not npm_path:
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)

    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)

    console.print(f"{__logo__} Setting up bridge...")

    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run([npm_path, "install"], cwd=user_bridge, check=True, capture_output=True)

        console.print("  Building...")
        subprocess.run([npm_path, "run", "build"], cwd=user_bridge, check=True, capture_output=True)

        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)

    return user_bridge


@channels_app.command("login")
def channels_login(
    channel_name: str = typer.Argument(..., help="Channel name (e.g. weixin, whatsapp)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
):
    """Authenticate with a channel via QR code or other interactive login."""
    from nanobot.channels.registry import discover_all
    from nanobot.config.loader import load_config

    config = load_config()
    channel_cfg = getattr(config.channels, channel_name, None) or {}

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

plugins_app = typer.Typer(help="Manage channel plugins")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list():
    """List all discovered channels (built-in and plugins)."""
    from nanobot.channels.registry import discover_all, discover_channel_names
    from nanobot.config.loader import load_config

    config = load_config()
    builtin_names = set(discover_channel_names())
    all_channels = discover_all()

    table = Table(title="Channel Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Enabled", style="green")

    for name in sorted(all_channels):
        cls = all_channels[name]
        source = "builtin" if name in builtin_names else "plugin"
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            source,
            "[green]yes[/green]" if enabled else "[dim]no[/dim]",
        )

    console.print(table)


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
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


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Authenticate with an OAuth provider."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
        token = None
        try:
            token = get_token()
        except Exception:
            pass
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    import asyncio

    from openai import AsyncOpenAI

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger():
        client = AsyncOpenAI(
            api_key="dummy",
            base_url="https://api.githubcopilot.com",
        )
        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
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
