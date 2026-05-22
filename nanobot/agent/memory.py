"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain, safe_filename

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": (
                            "Full updated long-term memory as markdown. Include all existing facts plus new ones. "
                            "CRITICAL DISTINCTION from history_entry: history_entry records past events; "
                            "memory_update records CURRENT STATE — what an agent needs to know to continue "
                            "working correctly. If memory_update already tracks an ongoing multi-step process "
                            "(e.g. a learning sequence, a task with numbered steps), you MUST update its "
                            "current progress to reflect what was completed in this conversation. "
                            "Do NOT rely on history_entry alone to track current progress — "
                            "that information will not be visible to the agent in future turns. "
                            "Return unchanged only if no facts or current state has changed. "
                            "PLAN TREE FORMAT: Active plans are tracked as '## [计划] NAME `[STATUS]`' sections. "
                            "STATUS values: [⬜] pending, [🔄] in-progress, [✅ DATE] done, [❌] abandoned, [🔀 已替换] replaced. "
                            "If backed by an external file, add '*文件*：`path`' line under the header. "
                            "Tasks are indented bullet lists; sub-plans indent further as '子计划：NAME `[STATUS]`'. "
                            "RULES: (1) Only update status markers and add completed items — never restructure the parent hierarchy. "
                            "(2) Sub-plans must stay nested under their parent task, never promoted to top-level. "
                            "(3) If a file-backed plan exists, follow its structure; do not derive a new order."
                        ),
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path, session_key: str | None = None):
        if session_key is not None:
            safe_key = safe_filename(session_key.replace(":", "_"))
            self.memory_dir = ensure_dir(workspace / "memory" / "topics" / safe_key)
        else:
            self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        pending_callback: Callable[[str], None] | None = None,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md.

        ``pending_callback`` lets the caller intercept the ``memory_update``
        string and decide *not* to write it to MEMORY.md right now. This is
        used to keep the system prompt stable during a turn — the caller
        stores the update in ``session.pending_consolidation_summary`` and
        promotes it to MEMORY.md at the start of the next user-initiated
        turn (when a new system prompt is built anyway).

        When ``pending_callback`` is None (default), the legacy path is
        used: ``write_long_term(update)`` flushes immediately.
        """
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": (
                "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation. "
                "Key rule: history_entry records WHAT HAPPENED (events, decisions, topics covered). "
                "memory_update records CURRENT STATE (ongoing task progress, active context, what must be known to resume work). "
                "If the conversation shows progress on a multi-step task already tracked in long-term memory, "
                "update that progress in memory_update — history_entry alone is insufficient because it is not injected into future prompts. "
                "CRITICAL: If long-term memory references an external plan file (e.g. a .md guide or roadmap), "
                "do NOT replace that plan's structure with a derived or reordered version. "
                "Track progress using the original plan's layer/step structure. "
                "Only add new knowledge points; never restructure or rename the existing plan hierarchy. "
                "PLAN TREE FORMAT: Active plans are tracked as '## [计划] NAME `[STATUS]`' sections. "
                "STATUS values: [⬜] pending, [🔄] in-progress, [✅ DATE] done, [❌] abandoned, [🔀 已替换] replaced. "
                "If backed by an external file, add '*文件*：`path`' line under the header. "
                "Tasks are indented bullet lists; sub-plans indent further as '子计划：NAME `[STATUS]`'. "
                "When updating memory_update: (1) Only update status markers and add completed items — never restructure. "
                "(2) Sub-plans must stay nested under their parent task, never promoted to top-level. "
                "(3) If a file-backed plan exists, follow its structure exactly."
            )},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            self.append_history(entry)
            update = _ensure_text(update)
            if update != current_memory:
                if pending_callback is not None:
                    # Defer the write — caller will hold the update in pending
                    # state and promote it at a cache-friendly moment.
                    pending_callback(update)
                else:
                    self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    # Keep at most this many bytes of consolidated messages in the session file
    # so old history remains browsable without the file growing without bound.
    _MAX_HISTORY_BYTES = 1_000_000

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        pending_promote_threshold_chars: int = 10_000,
    ):
        self.workspace = workspace
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.pending_promote_threshold_chars = pending_promote_threshold_chars
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._topic_stores: dict[str, MemoryStore] = {}

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def _get_topic_store(self, session_key: str) -> MemoryStore:
        if session_key not in self._topic_stores:
            self._topic_stores[session_key] = MemoryStore(self.workspace, session_key)
        return self._topic_stores[session_key]

    async def consolidate_messages(self, messages: list[dict[str, object]], session_key: str | None = None) -> bool:
        """Archive a selected message chunk into persistent memory.

        When ``session_key`` is provided, the memory_update is buffered in
        ``session.pending_consolidation_summary`` instead of being written
        to MEMORY.md immediately. This keeps the LLM's system prompt stable
        for the rest of the current turn (prompt cache stays warm).

        Auto-promote safety valve: once buffered pending exceeds
        ``pending_promote_threshold_chars`` (default 10k ≈ 2.5k tokens),
        flush it to MEMORY.md immediately. Otherwise user messages would
        carry an unbounded <system-reminder> payload.
        """
        store = self._get_topic_store(session_key) if session_key else self.store

        if session_key is None:
            return await store.consolidate(messages, self.provider, self.model)

        session = self.sessions.get_or_create(session_key)

        def _buffer_to_session(update: str) -> None:
            session.pending_consolidation_summary = update

        ok = await store.consolidate(
            messages, self.provider, self.model,
            pending_callback=_buffer_to_session,
        )
        if ok and session.pending_consolidation_summary is not None:
            # Persist so reload/restart sees the pending state.
            self.sessions.save(session)
            # If pending grew past the threshold, force-promote now so the
            # next user message doesn't carry a huge system-reminder.
            if len(session.pending_consolidation_summary) > self.pending_promote_threshold_chars:
                logger.info(
                    "Pending consolidation summary exceeds {} chars, "
                    "auto-promoting to MEMORY.md for {}",
                    self.pending_promote_threshold_chars, session_key,
                )
                self.promote_pending_summary(session_key)
        return ok

    def promote_pending_summary(self, session_key: str) -> None:
        """Flush a session's pending consolidation summary to its topic
        MEMORY.md and clear the pending state. Idempotent — does nothing if
        there's no pending summary.

        Called by:
          * auto-promote (size threshold exceeded in consolidate_messages)
          * /commit_memory user command
          * AgentLoop startup when promote_pending_on_restart is enabled
        """
        session = self.sessions.get_or_create(session_key)
        pending = session.pending_consolidation_summary
        if not pending:
            return
        store = self._get_topic_store(session_key)
        chars = len(pending)
        try:
            store.write_long_term(pending)
            logger.info(
                "promoted pending consolidation summary for {}: "
                "{} chars written to MEMORY.md ({})",
                session_key, chars, store.memory_file,
            )
        finally:
            session.pending_consolidation_summary = None
            self.sessions.save(session)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            session_key=session.key,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]], session_key: str | None = None) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        topic_store = self._get_topic_store(session_key) if session_key else self.store
        for _ in range(topic_store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            # Pass session_key only when present; keeps 1-arg call compatible with mocks in tests.
            success = (
                await self.consolidate_messages(messages, session_key)
                if session_key
                else await self.consolidate_messages(messages)
            )
            if success:
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            # Trigger threshold is budget * 1.2 (not raw budget) — gives a 20%
            # headroom so we don't consolidate the moment we touch the limit.
            # Less frequent consolidations = fewer system-prompt churns, which
            # matters more once pending-summary message-injection is in place.
            trigger = int(budget * 1.2)
            if estimated < trigger:
                logger.debug(
                    "Token consolidation idle {}: {}/{} (trigger={}) via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    trigger,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk, session_key=session.key):
                    return
                session.last_consolidated = end_idx
                self._compact_history(session)
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return

    def _compact_history(self, session: Session) -> None:
        """Drop oldest *consolidated* messages so the session file stays under
        _MAX_HISTORY_BYTES.  Unconsolidated messages are never touched because
        they are still needed for the LLM prompt window."""
        import json as _json

        if session.last_consolidated == 0:
            return

        sizes = [
            len(_json.dumps(m, ensure_ascii=False).encode())
            for m in session.messages
        ]
        total = sum(sizes)
        if total <= self._MAX_HISTORY_BYTES:
            return

        drop = 0
        for i in range(session.last_consolidated):
            if total <= self._MAX_HISTORY_BYTES:
                break
            total -= sizes[i]
            drop = i + 1

        if drop:
            session.messages = session.messages[drop:]
            session.last_consolidated -= drop
            logger.debug(
                "Compacted session {}: dropped {} old messages, {} remain",
                session.key,
                drop,
                len(session.messages),
            )
