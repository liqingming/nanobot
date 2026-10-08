"""Fork terminal controller, selected by the upstream CLI entry point."""
import asyncio
import json
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console

from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import (
    ContextCompactionEvent,
    ProgressEvent,
    RetryWaitEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
)
from nanobot.cli.log_control import _set_nanobot_logs
from nanobot.cli.runtime_config import _load_runtime_config, _migrate_cron_store
from nanobot.cli.stream import StreamRenderer, ThinkingSpinner
from nanobot.cli.terminal import (
    _flush_cli_reasoning,
    _print_agent_response,
    _print_cli_progress_line,
    _print_cli_reasoning,
    _ReasoningBuffer,
)
from nanobot.command.builtin import BUILTIN_COMMAND_SPECS
from nanobot.config.paths import get_workspace_cache_dir, is_default_workspace
from nanobot.fork.agent.context_usage import context_input_tokens
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import safe_filename, sync_workspace_templates
from nanobot.utils.restart import (
    consume_restart_notice_from_env,
    format_restart_completed_message,
    should_show_cli_restart_notice,
)
from nanobot.utils.session_runtime_log import append_session_runtime_log, exception_fields

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
_CLI_UNNAMED_SESSION_LABEL = "未命名会话"

_CLI_UNNAMED_SESSION_LABEL = "未命名会话"


def _cli_response_delivery_fields(msg: OutboundMessage, *, delivery_kind: str) -> dict[str, Any]:
    """Return privacy-safe correlation fields for CLI response delivery logs."""
    metadata = dict(msg.metadata or {})
    return {
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "turn_request_id": metadata.get("_turn_request_id"),
        "transcript_id": metadata.get("_transcript_id"),
        "delivery_kind": delivery_kind,
        "content_chars": len(msg.content or ""),
        "streamed": bool(metadata.get("_streamed")),
        "no_content": bool(metadata.get("_no_content")),
    }


def _render_cli_response(
    tui: Any,
    msg: OutboundMessage,
    content: str,
    *,
    runtime_log_path: Path,
    delivery_kind: str,
) -> None:
    """Render one final CLI response with durable start/done/exception diagnostics."""
    fields = _cli_response_delivery_fields(msg, delivery_kind=delivery_kind)
    fields["rendered_chars"] = len(content)
    append_session_runtime_log(runtime_log_path, "cli.response.render.start", **fields)
    try:
        tui.add_response(
            content,
            dict(msg.metadata or {}),
            message_id=(msg.metadata or {}).get("_transcript_id"),
        )
    except Exception as exc:
        append_session_runtime_log(
            runtime_log_path,
            "cli.response.render.exception",
            **fields,
            **exception_fields(exc),
        )
        raise
    append_session_runtime_log(runtime_log_path, "cli.response.render.done", **fields)


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


def _is_exit_command(command: str) -> bool:
    return command.strip().lower() in EXIT_COMMANDS


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
        or command == "/model"
        or command == "/skills"
        or command == "/system-prompt"
        or command == "/ide"
        or command.startswith("/ide ")
        or command == "/skin"
        or command.startswith("/skin ")
        or command == "/clear"
        or command == "/rename"
        or command.startswith("/rename ")
        or command == "/commit_memory"
        or command.startswith("/commit_memory ")
        or command == "/todos"
        or command.startswith("/todos ")
        or command in {"/bookmarks", "/bookmarks-delete", "/bookmarks-clear"}
        or command == "/resume"
        or command.startswith("/resume ")
    )


def _todos_all_completed(todos: list[dict[str, Any]]) -> bool:
    return bool(todos) and all(t.get("status") == "completed" for t in todos)


def _should_hide_stale_todos_on_new_turn(todos: list[dict[str, Any]]) -> bool:
    return bool(todos) and not _todos_all_completed(todos)


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
        (
            spec.command,
            spec.description,
            "submit" if spec.command == "/model" else ("edit" if spec.arg_hint else "submit"),
        )
        for spec in BUILTIN_COMMAND_SPECS
        if spec.command != "/new"
    ]
    items.extend([
        ("/rename", "Rename the current CLI session.", "edit"),
        ("/system-prompt", "Show the current topic's rendered system rules.", "submit"),
        ("/skin", "Switch the Windows Terminal background image.", "edit"),
        ("/ide", "管理 Rider 代码上下文：on/off/clear/show/remove。", "edit"),
        ("/clear", "Clear context and start an unnamed empty session.", "submit"),
        ("/resume", "Show saved CLI sessions and switch topics.", "submit"),
        ("/bookmarks", "Show bookmarks for the current topic.", "submit"),
        ("/bookmarks-delete", "Select and delete a bookmark.", "submit"),
        ("/bookmarks-clear", "Clear all bookmarks in the current topic.", "submit"),
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


def _model_picker_items(agent_loop: AgentLoop, session_key: str | None = None) -> list[tuple[str, str]]:
    """Build model preset picker rows, highlighting the active preset."""
    runtime = agent_loop.runtime_for_session(agent_loop.sessions.get_or_create(session_key)) if session_key else None
    active = (runtime.model_preset if runtime else agent_loop.model_preset) or "default"
    names = ["default", *sorted(name for name in agent_loop.model_presets if name != "default")]
    items: list[tuple[str, str]] = []
    for name in names:
        preset = agent_loop.model_presets.get(name)
        model = getattr(preset, "model", "")
        marker = "✓ " if name == active else "  "
        label = f"{marker}{name}"
        if model:
            label += f" — {model}"
        items.append((name, label))
    return items


def _switch_model_from_picker(agent_loop: AgentLoop, tui: Any, name: str, session_key: str | None = None) -> None:
    """Apply a model preset selected from the interactive CLI picker."""
    try:
        if session_key:
            runtime = agent_loop.set_session_model_preset(session_key, name)
            model = runtime.model
            name = runtime.model_preset or "default"
        else:
            agent_loop.set_model_preset(name)
            model = agent_loop.model
    except (KeyError, ValueError) as exc:
        tui.add_system(f"切换模型预设失败: {exc}")
        return
    try:
        from nanobot.config.loader import save_active_model_preset

        config_path = save_active_model_preset(name)
    except (OSError, ValueError) as exc:
        tui.add_system(
            f"已切换模型预设: {name}\n模型: {model}\n"
            f"但保存启动默认值失败: {exc}"
        )
        return
    tui.add_system(
        f"已切换模型预设: {name}\n模型: {model}\n"
        f"已保存为启动默认值: {config_path}"
    )


def _runtime_data_dir_for_workspace(workspace_path: Path) -> Path:
    """Return where nanobot runtime metadata should live for a workspace."""
    return get_workspace_cache_dir(workspace_path)


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
    tools = ToolRegistry()
    mcp_provider = MCPProvider.from_config(config, tools)

    try:
        agent_loop = AgentLoop.from_config(
            config, bus,
            data_dir=data_dir,
            session_manager=SessionManager(
                config.workspace_path,
                sessions_root=data_dir / "sessions",
                workspace_state_root=data_dir,
            ),
            tool_registry=tools,
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
            await mcp_provider.connect()
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
            await agent_loop.aclose()
            await mcp_provider.aclose()

        asyncio.run(run_once())
    else:
        # Interactive mode — split-pane TUI (output pane + persistent input line)
        from nanobot.bus.events import InboundMessage
        from nanobot.config.paths import get_cli_history_path
        from nanobot.fork.agent.input_evidence import ReceivedInput, capture_input
        from nanobot.fork.cli.tui_factory import create_tui

        if ":" in session_id:
            cli_channel, _cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, _cli_chat_id = "cli", session_id

        history_file = str(get_cli_history_path())

        async def run_interactive():
            await mcp_provider.connect()
            tui = create_tui(
                render_markdown=markdown,
                history_file=history_file,
                model=agent_loop.model,
                reasoning_effort=getattr(agent_loop.provider.generation, "reasoning_effort", None),
                backend=config.agents.defaults.tui_backend,
                skin_enabled=config.agents.defaults.tui_skin_enabled,
                show_tool_preface=config.agents.defaults.tui_show_tool_preface,
                workspace=agent_loop.workspace,
            )

            # A fresh CLI start is intentionally unnamed. Mark it in memory so
            # the first persisted user turn has a stable display name, but do
            # not save it yet or unused launches would clutter /resume.
            fresh_chat_id = f"session_{uuid.uuid4().hex}"
            fresh_session = agent_loop.sessions.get_or_create(f"{cli_channel}:{fresh_chat_id}")
            _mark_cli_session_unnamed(fresh_session)
            topic_state: dict[str, str] = {"chat_id": fresh_chat_id}
            from nanobot.fork.cli.ide_integration import IDEIntegration

            ide = IDEIntegration(agent_loop.workspace, tui)
            ide.pending.switch(fresh_session.key)
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
                set_runtime_log_path = getattr(tui, "set_session_runtime_log_path", None)
                if callable(set_runtime_log_path):
                    set_runtime_log_path(agent_loop.sessions.get_session_runtime_log_path(s.key))
                tui.set_input_history_topic(s.key)
                tui.set_topic("" if display_name == _CLI_UNNAMED_SESSION_LABEL else display_name or name)
                page = agent_loop.sessions.display_history_page(
                    s.key,
                    s.messages,
                    turn_limit=10,
                )
                tui.load_session_history(
                    page.messages,
                    max_messages=10,
                    tool_registry=agent_loop.tools,
                    workspace=agent_loop.workspace,
                )

                async def _load_older_history(
                    before_offset: int | None,
                    *,
                    session_key: str = s.key,
                ) -> tuple[list[dict], int | None, bool]:
                    older = await asyncio.to_thread(
                        agent_loop.sessions.display_history_page,
                        session_key,
                        [],
                        before_offset=before_offset,
                        turn_limit=10,
                    )
                    return older.messages, older.before_offset, older.has_older

                tui.set_history_page_loader(
                    _load_older_history,
                    before_offset=page.before_offset,
                    has_older=page.has_older,
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
            pending_queue: list[tuple[str, ReceivedInput]] = []
            _pre_submitted: list[bool] = [False]
            _pending_transcript_ids: dict[str, str] = {}
            _turn_cancelled: list[bool] = [False]
            _active_turn_request_id: list[str | None] = [None]
            _cancel_in_progress: list[bool] = [False]
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

            async def _cancel_current() -> None:
                nonlocal is_processing
                if not is_processing or _cancel_in_progress[0]:
                    return
                _cancel_in_progress[0] = True
                _turn_cancelled[0] = True
                _set_activity_phase("cancelling")
                try:
                    session_key = agent_loop._effective_session_key(InboundMessage(
                        channel=cli_channel,
                        sender_id="user",
                        chat_id=topic_state["chat_id"],
                        content="",
                    ))
                    await agent_loop.cancel_session_turn(
                        session_key,
                        request_id=_active_turn_request_id[0],
                    )
                finally:
                    tui.flush_stream()
                    is_processing = False
                    tui.set_is_processing(False)
                    pending_queue.clear()
                    _active_turn_request_id[0] = None
                    _cancel_in_progress[0] = False
                    _set_activity_phase("idle")
                    tui.add_system("已取消当前请求。")

            tui.set_on_cancel(_cancel_current)

            def _set_activity_phase(phase: str) -> None:
                setter = getattr(tui, "set_activity_phase", None)
                if callable(setter):
                    setter(phase)

            def _pre_submit(text: str) -> None:
                if not is_processing and not _is_cli_local_command(text):
                    _pending_transcript_ids["user"] = uuid.uuid4().hex
                    _pending_transcript_ids["assistant"] = uuid.uuid4().hex
                    tui.add_user_echo(
                        ide.pending.preview(text), message_id=_pending_transcript_ids["user"]
                    )
                    tui.stream_start()
                    _pre_submitted[0] = True

            def _capture_submission(text: str) -> ReceivedInput:
                return capture_input(InboundMessage(
                    channel=cli_channel, sender_id="user",
                    chat_id=topic_state["chat_id"], content=text,
                ), "cli_interactive")

            async def _send_message(text: str, original_input: ReceivedInput | None = None) -> None:
                nonlocal is_processing
                _turn_cancelled[0] = False
                request_id = uuid.uuid4().hex
                _active_turn_request_id[0] = request_id
                is_processing = True
                tui.set_is_processing(True)
                _set_activity_phase("prepare_turn")
                if _pre_submitted[0]:
                    _pre_submitted[0] = False
                else:
                    _pending_transcript_ids["user"] = uuid.uuid4().hex
                    _pending_transcript_ids["assistant"] = uuid.uuid4().hex
                    tui.add_user_echo(text, message_id=_pending_transcript_ids["user"])
                    tui.stream_start()
                # Exceed prompt_toolkit's max_render_postpone_time (0.01s) so the
                # thinking animation is guaranteed to reach the screen before the
                # bus receives the message and the LLM starts responding.
                await asyncio.sleep(0.015)
                _set_activity_phase("agent_processing")
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
                await bus.publish_inbound(capture_input(InboundMessage(
                    channel=cli_channel,
                    sender_id="user",
                    chat_id=topic_state["chat_id"],
                    content=text,
                    metadata={
                        "_wants_stream": True,
                        "_user_transcript_id": _pending_transcript_ids.get("user"),
                        "_assistant_transcript_id": _pending_transcript_ids.get("assistant"),
                        "_turn_request_id": request_id,
                    },
                ), original=original_input))

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
                    # 整轮消耗仍累计；ctx 仅使用最近一次上下文输入，不叠加缓存。
                    tui.update_context_usage(
                        context_input_tokens(usage),
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
                    await _send_message(*pending_queue.pop(0))

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
                ide.pending.switch(session_key)
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

                if await ide.command(text):
                    return

                # ── 话题管理命令 ──────────────────────────────────────────────
                if text == "/model":
                    async def _on_model_select(name: str) -> None:
                        _switch_model_from_picker(agent_loop, tui, name, f"{cli_channel}:{topic_state['chat_id']}")

                    tui.show_topic_popup(_model_picker_items(agent_loop, f"{cli_channel}:{topic_state['chat_id']}"), _on_model_select)
                    return

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

                    skin_dir = Path(config.agents.defaults.tui_skin_dir)

                    def _switch_terminal_skin(selector: str) -> None:
                        try:
                            selected, settings, _backup = switch_skin(selector, skin_dir)
                        except (SkinError, OSError, ValueError) as exc:
                            tui.add_system(f"切换终端背景失败: {exc}")
                            return
                        tui.add_system(
                            f"已切换 Windows Terminal 背景图: {selected.name}\n"
                            f"配置: {settings}"
                        )

                    arg = text[len("/skin"):].strip()
                    try:
                        images = list_skin_images(skin_dir)
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
                    await _send_message(
                        "请继续上次中断的任务。", original_input=_capture_submission(user_input),
                    )
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

                if text == "/bookmarks":
                    show_bookmarks = getattr(tui, "show_bookmark_popup", None)
                    if callable(show_bookmarks):
                        show_bookmarks()
                    else:
                        tui.add_system("当前 TUI 不支持书签列表。")
                    return

                if text == "/bookmarks-delete":
                    show_delete = getattr(tui, "show_bookmark_delete_popup", None)
                    if callable(show_delete):
                        show_delete()
                    else:
                        tui.add_system("当前 TUI 不支持删除书签。")
                    return

                if text == "/bookmarks-clear":
                    clear_bookmarks = getattr(tui, "clear_topic_bookmarks", None)
                    if callable(clear_bookmarks):
                        clear_bookmarks()
                    else:
                        tui.add_system("当前 TUI 不支持清理书签。")
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

                # 在提交时冻结附件，而非等排队发送时再取，避免下一条问题混入新选区。
                original_input = _capture_submission(user_input)
                user_input = ide.pending.consume(user_input)
                if is_processing:
                    pending_queue.append((user_input, original_input))
                    tui.add_system("Message queued — will send when nanobot finishes.")
                else:
                    await asyncio.sleep(0)
                    await _send_message(user_input, original_input=original_input)

            tui.set_on_submit(_on_submit)
            tui.set_on_pre_submit(_pre_submit)

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        event = msg.event
                        if isinstance(event, (RetryWaitEvent, ContextCompactionEvent)):
                            if msg.content:
                                tui.add_progress(msg.content)
                            continue
                        if event is not None and not isinstance(event, (StreamDeltaEvent, StreamEndEvent, StreamedResponseEvent, ProgressEvent)):
                            continue
                        runtime_log_path = agent_loop.sessions.get_session_runtime_log_path(
                            f"{cli_channel}:{msg.chat_id}"
                        )

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
                            delivery_fields = _cli_response_delivery_fields(
                                msg, delivery_kind="streamed_final"
                            )
                            append_session_runtime_log(
                                runtime_log_path, "cli.response.received", **delivery_fields
                            )
                            if _turn_cancelled[0]:
                                append_session_runtime_log(
                                    runtime_log_path,
                                    "cli.response.render.skipped",
                                    **delivery_fields,
                                    reason="turn_cancelled",
                                )
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
                                _render_cli_response(
                                    tui,
                                    msg,
                                    content,
                                    runtime_log_path=runtime_log_path,
                                    delivery_kind="streamed_final",
                                )
                            else:
                                append_session_runtime_log(
                                    runtime_log_path,
                                    "cli.response.render.skipped",
                                    **delivery_fields,
                                    reason="empty_rendered_content",
                                    rendered_chars=len(content),
                                )
                            await _turn_complete()
                            continue

                        if isinstance(event, ProgressEvent) or msg.metadata.get("_progress"):
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
                                    tool_events = msg.metadata.get("_tool_events")
                                    if summary is not None:
                                        tui.add_tool_result(summary, tool_events=tool_events)
                                    elif msg.content:
                                        if msg.metadata.get("_tool_result"):
                                            tui.add_tool_result(msg.content, tool_events=tool_events)
                                        else:
                                            tui.add_progress(msg.content, tool_events=tool_events)
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
                            delivery_fields = _cli_response_delivery_fields(
                                msg, delivery_kind="error_final"
                            )
                            append_session_runtime_log(
                                runtime_log_path, "cli.response.received", **delivery_fields
                            )
                            try:
                                tui.pop_stream()
                                tui.flush_accumulator()
                            except Exception:
                                pass
                            content = msg.content or "nanobot task failed. See this topic's runtime.log for details."
                            _render_cli_response(
                                tui,
                                msg,
                                content,
                                runtime_log_path=runtime_log_path,
                                delivery_kind="error_final",
                            )
                            if is_processing:
                                await _turn_complete()
                            continue

                        # Non-streaming response (or unsolicited push from cron etc.)
                        delivery_fields = _cli_response_delivery_fields(
                            msg, delivery_kind="non_streaming_final"
                        )
                        append_session_runtime_log(
                            runtime_log_path, "cli.response.received", **delivery_fields
                        )
                        if msg.content:
                            _render_cli_response(
                                tui,
                                msg,
                                msg.content,
                                runtime_log_path=runtime_log_path,
                                delivery_kind="non_streaming_final",
                            )
                        else:
                            append_session_runtime_log(
                                runtime_log_path,
                                "cli.response.render.skipped",
                                **delivery_fields,
                                reason="empty_message_content",
                                rendered_chars=0,
                            )
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
                await ide.bridge.close()
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.aclose()
                await mcp_provider.aclose()
                print("\033]0;nanobot\007", end="", flush=True)

        asyncio.run(run_interactive())
