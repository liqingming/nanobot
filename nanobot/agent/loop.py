"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.fork.agent.learning import (
    PATTERN_THRESHOLD,
    PatternStore,
    TurnSummary,
    _compress_tool_sequence,
    detect_user_delta,
)
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.runner import AgentRunResult, AgentRunSpec, AgentRunner
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry, iter_fork_tool_factories
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


_HINT_KEY_PRIORITY: tuple[str, ...] = (
    "path", "file_path", "filepath", "file",
    "url", "command", "cmd",
    "query", "q",
    "symbol", "name", "topic", "content",
)
_HINT_PATH_KEYS: tuple[str, ...] = ("path", "file_path", "filepath", "file")


def relativize_path(value: str, workspace: Any) -> str:
    """If ``value`` is an absolute path inside ``workspace``, return its
    workspace-relative form prefixed with ``./`` so users can tell at a
    glance that it's relative; otherwise return it unchanged.

    Best-effort — any exception falls back to the original string.
    """
    if not value or workspace is None:
        return value
    try:
        from pathlib import Path
        p = Path(value)
        if not p.is_absolute():
            return value
        ws = Path(workspace).resolve()
        try:
            rel = p.resolve().relative_to(ws)
        except ValueError:
            return value
        rel_str = str(rel)
        if rel_str == ".":
            return value
        # Normalize Windows backslashes to forward slashes for the trace
        # display so paths don't carry the noisy "./src\foo\bar.py" mix.
        rel_str = rel_str.replace("\\", "/")
        return f"./{rel_str}"
    except Exception:
        return value


def _smart_truncate(text: str, max_len: int = 40) -> str:
    """Shorten ``text`` to roughly ``max_len`` chars.

    For path-like strings (containing ``/`` or ``\\``), preserves both the
    leading segment and the trailing filename so users can still see what
    file is being touched:

        "./very/deep/nested/path/to/some_long_file.py"
        → "./very/deep/n…/some_long_file.py"

    For non-path strings, falls back to leading truncation.
    """
    if len(text) <= max_len:
        return text
    if "/" in text or "\\" in text:
        # Reserve 1 char for "…"; split remainder ~40% head / 60% tail so
        # the filename at the end stays visible.
        tail_len = max(int((max_len - 1) * 0.6), 1)
        head_len = max(max_len - 1 - tail_len, 1)
        return text[:head_len] + "…" + text[-tail_len:]
    return text[: max_len - 1] + "…"


def format_tool_hint(tool_calls: list, *, workspace: Any = None) -> str:
    """Build a concise "tool(arg)" hint string for a list of tool calls.

    Picks the most identifying argument (path/url/query/...) per call using
    ``_HINT_KEY_PRIORITY``. Path-like arguments are shortened to a
    workspace-relative form when applicable.
    """
    def _fmt(tc) -> str:
        args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
        if not isinstance(args, dict):
            return tc.name
        val: str | None = None
        chosen_key: str | None = None
        for key in _HINT_KEY_PRIORITY:
            candidate = args.get(key)
            if isinstance(candidate, str) and candidate.strip():
                val = candidate
                chosen_key = key
                break
        if val is None:
            val = next(
                (v for v in args.values() if isinstance(v, str) and v.strip()),
                None,
            )
        if not isinstance(val, str):
            return tc.name
        if chosen_key in _HINT_PATH_KEYS:
            val = relativize_path(val, workspace)
            # Normalize backslashes to forward slashes for visual consistency
            # in the trace (also covers absolute Windows paths outside the
            # workspace that relativize_path leaves untouched).
            val = val.replace("\\", "/")
        return f'{tc.name}("{_smart_truncate(val)}")'

    return ", ".join(_fmt(tc) for tc in tool_calls)


class _LoopHook(AgentHook):
    """Core lifecycle hook for the main agent loop.

    Handles streaming delta relay, progress reporting, tool-call logging,
    and think-tag stripping for the built-in agent path.
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean):]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    async def after_execute_tools(self, context: AgentHookContext) -> None:
        if not self._on_progress:
            return
        # Build per-call summaries paired with their tool calls so the UI can
        # turn the spinner placeholder into a "↳ tool(args) → result" trace.
        # We always emit (even when no tool produced a summary) so the TUI
        # petrifies the placeholder *now* — otherwise tools without a
        # summarize_result (e.g. todo_write) would only get petrified when
        # the next tool starts, and disappear entirely if the LLM ends
        # the turn right after them.
        from nanobot.agent.tools.summaries import summarize_tool_result
        parts: list[str] = []
        for tc, result in zip(context.tool_calls, context.tool_results):
            tool = self._loop.tools.get(tc.name)
            summary = summarize_tool_result(tool, tc.arguments, result)
            if summary:
                parts.append(summary)
        joined = " · ".join(parts)  # may be empty — that's fine
        try:
            await self._on_progress(joined, tool_hint=True, tool_result=True)
        except TypeError:
            await self._on_progress(joined, tool_hint=True)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class _LoopHookChain(AgentHook):
    """Run the core loop hook first, then best-effort extra hooks.

    This preserves the historical failure behavior of ``_LoopHook`` while still
    letting user-supplied hooks opt into ``CompositeHook`` isolation.
    """

    __slots__ = ("_primary", "_extras")

    def __init__(self, primary: AgentHook, extra_hooks: list[AgentHook]) -> None:
        self._primary = primary
        self._extras = CompositeHook(extra_hooks)

    def wants_streaming(self) -> bool:
        return self._primary.wants_streaming() or self._extras.wants_streaming()

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._primary.before_iteration(context)
        await self._extras.before_iteration(context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._primary.on_stream(context, delta)
        await self._extras.on_stream(context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._primary.on_stream_end(context, resuming=resuming)
        await self._extras.on_stream_end(context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._primary.before_execute_tools(context)
        await self._extras.before_execute_tools(context)

    async def after_execute_tools(self, context: AgentHookContext) -> None:
        await self._primary.after_execute_tools(context)
        await self._extras.after_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._primary.after_iteration(context)
        await self._extras.after_iteration(context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        content = self._primary.finalize_content(context, content)
        return self._extras.finalize_content(context, content)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        data_dir: Path | None = None,
        model: str | None = None,
        max_iterations: int = 1000,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        hooks: list[AgentHook] | None = None,
        enable_learning: bool = True,
        pending_promote_threshold_chars: int = 10_000,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.enable_learning = enable_learning
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        _data = data_dir or workspace
        self.context = ContextBuilder(_data, workspace=workspace, timezone=timezone)
        self.sessions = session_manager or SessionManager(_data)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        # Learning context: turn-level metadata injected before the next user message.
        self._last_turn_summary: dict[str, TurnSummary] = {}
        self._prev_consolidated: dict[str, int] = {}
        self._last_user_input: dict[str, str] = {}
        # Cross-session pattern store (persisted to data_dir/memory/patterns.json).
        self._pattern_store: PatternStore | None = (
            PatternStore(_data) if self.enable_learning else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=_data,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
            pending_promote_threshold_chars=pending_promote_threshold_chars,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))

        # Fork tools self-register via fork bootstrap. Iterate factories
        # here so fork tools see the loop's wiring (bus / sessions /
        # context) without import-time circular dependencies.
        for factory in iter_fork_tool_factories():
            try:
                tool = factory(self)
            except Exception:
                continue
            if tool is not None:
                self.tools.register(tool)

        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron", "todo_write", "ask_user"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _empty_after_tools(messages: list[dict]) -> bool:
        """True when the LLM returned an empty non-tool message right after tool results."""
        if len(messages) < 2:
            return False
        last = messages[-1]
        prev = messages[-2]
        return (
            last.get("role") == "assistant"
            and not last.get("content")
            and not last.get("tool_calls")
            and prev.get("role") == "tool"
        )

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think
        return strip_think(text) or None

    # Argument keys that identify *what* a tool is acting on. Listed in
    # priority order: when a tool call has multiple args, the hint shows the
    # one most useful for the user (e.g. for write_file we want the path,
    # not the file content). Fallback: first string value in the dict.
    _HINT_KEY_PRIORITY = (
        "path", "file_path", "filepath", "file",
        "url", "command", "cmd",
        "query", "q",
        "symbol", "name", "topic", "content",
    )
    _HINT_PATH_KEYS = ("path", "file_path", "filepath", "file")

    def _tool_hint(self, tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'.
        Paths inside the workspace are shown relative; otherwise absolute.
        """
        return format_tool_hint(tool_calls, workspace=self.workspace)

    # ── learning context helpers ────────────────────────────────────────

    def _build_learning_ctx(self, session_key: str) -> str | None:
        """Build TurnSummary block for injection before the next user message."""
        if not self.enable_learning:
            return None
        ts = self._last_turn_summary.get(session_key)
        if ts is None or not ts.is_significant:
            return None
        return ts.format_for_injection()

    def _capture_turn_summary(
        self, session_key: str, tools_used: list[str], session: Session,
        current_message: str, result: AgentRunResult,
    ) -> None:
        """Build and store a TurnSummary from the just-completed turn."""

        # Count tool errors.
        error_count = sum(
            1 for ev in (result.tool_events or [])
            if ev.get("status") == "error"
        )

        # Compute pressure.
        prompt_tokens = result.usage.get("prompt_tokens", 0)
        cw = self.context_window_tokens
        if cw > 0 and prompt_tokens > 0:
            pct = prompt_tokens * 100 // cw
            detail = f"{prompt_tokens // 1000}K/{cw // 1000}K"
        else:
            pct = 0
            detail = ""

        # Detect consolidation since last turn.
        prev = self._prev_consolidated.get(session_key, 0)
        curr = session.last_consolidated
        self._prev_consolidated[session_key] = curr
        consolidation_note = None
        if curr > prev:
            consolidation_note = f"merged {curr - prev} msgs into memory"

        # Count tools (deduplicated list + total calls).
        tools_list = list(dict.fromkeys(tools_used))  # preserve order, dedupe
        tools_count = len(tools_used)

        # Detect repeated tool pattern (order-preserving, cross-session).
        repeated_pattern: tuple[str, ...] | None = None
        repeated_count = 0
        if tools_used and self._pattern_store is not None:
            pattern_key = _compress_tool_sequence(tools_used)
            count = self._pattern_store.increment(pattern_key)
            if count >= PATTERN_THRESHOLD:
                repeated_pattern = pattern_key
                repeated_count = count

        # Detect user_delta.
        prev_input = self._last_user_input.get(session_key)
        self._last_user_input[session_key] = current_message
        user_delta = detect_user_delta(prev_input, current_message)

        # Turn index from session messages count.
        session_msgs = session.messages
        turn_index = sum(1 for m in session_msgs if m.get("role") == "user")

        self._last_turn_summary[session_key] = TurnSummary(
            turn_index=turn_index,
            pressure_pct=pct,
            pressure_detail=detail,
            tools_count=tools_count,
            tools_list=tools_list,
            error_count=error_count,
            stop_reason=result.stop_reason,
            consolidation_note=consolidation_note,
            user_delta=user_delta,
            repeated_pattern=repeated_pattern,
            repeated_count=repeated_count,
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> tuple[str | None, list[str], list[dict], AgentRunResult]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hook: AgentHook = (
            _LoopHookChain(loop_hook, self._extra_hooks)
            if self._extra_hooks
            else loop_hook
        )

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata={
                                "_stream_delta": True,
                                "_stream_id": _current_stream_id(),
                            },
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata={
                                "_stream_end": True,
                                "_resuming": resuming,
                                "_stream_id": _current_stream_id(),
                            },
                        ))
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            print(f"正在完成 {len(self._background_tasks)} 个后台任务（记忆整理等），请稍候…")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            print("正在断开 MCP 连接…")
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self.enable_learning:
                await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
                session_key=key,
                todos=session.todos,
                pending_summary=session.pending_consolidation_summary,
            )
            final_content, _, all_msgs, _ = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            if self.enable_learning:
                self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        if self.enable_learning:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        learning_ctx = self._build_learning_ctx(key)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            learning_ctx=learning_ctx,
            session_key=key,
            todos=session.todos,
            pending_summary=session.pending_consolidation_summary,
        )

        async def _bus_progress(
            content: str, *, tool_hint: bool = False, tool_result: bool = False,
        ) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            if tool_result:
                meta["_tool_result"] = True
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, tools_used, all_msgs, run_result = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
        )
        if self.enable_learning:
            self._capture_turn_summary(key, tools_used, session, msg.content, run_result)

        if final_content is None and self._empty_after_tools(all_msgs):
            logger.info("LLM returned empty after tool results for {}:{}, retrying with nudge", msg.channel, msg.chat_id)
            first_all_msgs = all_msgs
            nudge = {"role": "user", "content": "请将你对上述内容的分析和总结写在回复正文里。"}
            final_content, _, retry_all_msgs, _ = await self._run_agent_loop(
                first_all_msgs + [nudge],
                on_progress=on_progress or _bus_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                channel=msg.channel, chat_id=msg.chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            if final_content is not None:
                # Stitch: first run messages + retry new messages, skipping the nudge.
                # retry_all_msgs = first_all_msgs + [nudge] + retry_new_msgs
                # retry_all_msgs[len(first_all_msgs) + 1:] = retry_new_msgs (nudge excluded)
                all_msgs = first_all_msgs + retry_all_msgs[len(first_all_msgs) + 1:]

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        if self.enable_learning:
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
            if final_content == "I've completed processing but have no response to give.":
                meta["_no_content"] = True
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and ContextBuilder._RUNTIME_CONTEXT_TAG in block["text"]
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and ContextBuilder._RUNTIME_CONTEXT_TAG in content:
                    # Strip all metadata prefixes (TurnSummary + RuntimeContext).
                    # RuntimeContext tag is always present; split there, then skip its lines.
                    after_tag = content.split(ContextBuilder._RUNTIME_CONTEXT_TAG, 1)[1]
                    parts = after_tag.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
        )
