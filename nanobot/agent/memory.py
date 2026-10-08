"""Memory storage, transcript archiving, and session checkpoint consolidation."""

# Tool schemas are installed by the ``@tool_parameters`` class decorator at
# runtime; static analyzers cannot observe that it clears ``parameters`` from
# ``__abstractmethods__`` before these classes are instantiated.
# pyright: reportAbstractUsage=false, reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import weakref
from contextlib import suppress
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast
from uuid import uuid4

from loguru import logger

from nanobot.events import NO_EVENTS, ContextCompactionEvent, EventSink
from nanobot.fork.agent.context_budget import resolve_context_budget
from nanobot.fork.agent.native_context import uses_native_context
from nanobot.fork.agent.summary_diagnostics import (
    SummaryValidationError,
    record_summary_failure,
)
from nanobot.fork.agent.summary_transaction import (
    SUMMARY_MAX_CHARS,
    SummarySnapshot,
    SummaryTransactionError,
    commit_summary,
    valid_summary,
)
from nanobot.fork.agent.transactional_context import request_summary
from nanobot.providers.base import ProviderConversationState
from nanobot.runtime_context import public_history_messages
from nanobot.session.manager import Session, SessionManager
from nanobot.session.summary import is_summary_checkpoint, session_summary_from_metadata
from nanobot.utils.atomic_write import replace_file_with_retry
from nanobot.utils.gitstore import GitStore
from nanobot.utils.helpers import (
    content_with_media_breadcrumbs,
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    recent_message_start_index,
    strip_think,
    truncate_text,
    truncate_text_to_tokens,
)
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.workspace_prompts import (
    WORKSPACE_PROMPT_MAX_CHARS,
    has_workspace_prompt_override,
    load_workspace_prompt_override,
    workspace_prompt_file,
)

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMProvider
    from nanobot.utils.llm_runtime import LLMRuntime

# ---------------------------------------------------------------------------
# MemoryStore — pure file I/O layer
# ---------------------------------------------------------------------------


class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history.jsonl, SOUL.md, USER.md."""

    _DEFAULT_MAX_HISTORY = 1000
    _DREAM_FILE_EMBED_CAP = 8000
    _INTERNAL_HISTORY_SESSION_PREFIXES = ("cron:", "dream:")
    _INTERNAL_HISTORY_SESSION_KEYS = {"heartbeat"}
    # Durable files whose real working-tree delta grounds Dream commit messages.
    # Deliberately excludes memory/.dream_cursor so progress bookkeeping never
    # appears as a durable-memory edit in the audit record.
    _DREAM_CONTENT_PATHS = ("SOUL.md", "USER.md", "memory/MEMORY.md")
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(
        self,
        workspace: Path,
        max_history_entries: int = _DEFAULT_MAX_HISTORY,
        *,
        memory_dir_override: Path | None = None,
    ):
        # fork addition: ``memory_dir_override`` lets a caller mount this store
        # onto an arbitrary memory directory (e.g. per-topic subdirectory).
        # When None, falls back to the upstream default ``workspace/memory``.
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(
            memory_dir_override if memory_dir_override is not None else workspace / "memory"
        )
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._corruption_logged = False  # rate-limit invalid cursor warning
        self._malformed_entry_logged = False  # rate-limit bad history shape warning
        self._oversize_logged = False  # rate-limit oversized-entry warning
        # Fork(perf): cache MEMORY.md content keyed by mtime. build_system_prompt
        # reads it every turn (global store + per-topic store = two reads/turn);
        # invalidated by mtime change and by write_memory.
        self._memory_cache: tuple[float, str] | None = None
        self._dream_prompt_oversize_logged = False
        self._append_lock = threading.Lock()  # serialize cursor allocation + append
        self._git = GitStore(workspace, tracked_files=[
            "SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor",
        ])
        self._maybe_migrate_legacy_history()

    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers -----------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl.

        The migration is best-effort and prioritizes preserving as much content
        as possible over perfect parsing.
        """
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                # Default to "already processed" so upgrades do not replay the
                # user's entire historical archive into Dream on first start.
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- MEMORY.md (long-term facts) -----------------------------------------

    def read_memory(self) -> str:
        # Fork(perf): serve from mtime-keyed cache when MEMORY.md is unchanged.
        try:
            mtime = self.memory_file.stat().st_mtime
        except OSError:
            self._memory_cache = None
            return self.read_file(self.memory_file)
        cached = self._memory_cache
        if cached is not None and cached[0] == mtime:
            return cached[1]
        content = self.read_file(self.memory_file)
        self._memory_cache = (mtime, content)
        return content

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")
        self._memory_cache = None  # invalidate read_memory cache

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # -- context injection (used by context.py) ------------------------------

    @staticmethod
    def format_memory_context(long_term: str) -> str:
        """Remove untouched template scaffolding from long-term memory text."""
        if not long_term:
            return ""
        for section in (
            "## User Information\n\n(Important facts about the user)\n\n",
            "## Preferences\n\n(User preferences learned over time)\n\n",
            "## Project Context\n\n(Information about ongoing projects)\n\n",
            "## Important Notes\n\n(Things to remember)\n\n",
        ):
            long_term = long_term.replace(section, "")
        footer = "\n---\n\n*This file is automatically updated by nanobot when important information should be remembered.*"
        return long_term.replace(footer, "").strip()

    def get_memory_context(self) -> str:
        """Return meaningful long-term memory without its untouched template scaffolding."""
        return self.format_memory_context(self.read_memory())

    # -- history.jsonl — append-only, JSONL format ---------------------------

    def _normalize_history_entry(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
    ) -> str:
        """Return the exact bounded, model-safe text accepted by the journal."""
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        raw = entry.rstrip()
        content = strip_think(raw)
        if len(content) > limit:
            if not self._oversize_logged:
                self._oversize_logged = True
                logger.warning(
                    "history entry exceeds {} chars ({}); truncating. "
                    "Usually means a caller forgot its own cap; "
                    "further occurrences suppressed.",
                    limit,
                    len(content),
                )
            content = truncate_text(content, limit)
        return content

    def append_history(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> int:
        """Append *entry* to history.jsonl and return its auto-incrementing cursor.

        Entries are passed through `strip_think` to drop template-level leaks
        (e.g. unclosed `<think` prefixes, `<channel|>` markers) before being
        persisted. If the cleaned content is empty but the raw entry wasn't,
        the record is persisted with an empty string rather than falling back
        to the raw leak — otherwise `strip_think`'s guarantees would be
        undone when Dream consumes the journal entry.

        A defensive cap (*max_chars*, default ``_HISTORY_ENTRY_HARD_CAP``) is
        applied as a final safety net: individual callers should cap their own
        content more tightly; this default only exists to catch unintentional
        large writes (e.g. an LLM echoing its input back as a "summary").
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw = entry.rstrip()
        content = self._normalize_history_entry(entry, max_chars=max_chars)
        # Cursor allocation and the append must be atomic: concurrent writers
        # could otherwise read the same current cursor and emit duplicates.
        with self._append_lock:
            cursor = self._next_cursor()
            if raw and not content:
                logger.debug(
                    "history entry {} stripped to empty (likely template leak); "
                    "persisting empty content to avoid re-polluting Dream input",
                    cursor,
                )
            record = {"cursor": cursor, "timestamp": ts, "content": content}
            if session_key:
                record["session_key"] = session_key
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        """Non-negative int cursors only; reject bool (``isinstance(True, int)`` is True)."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        """Yield ``(entry, cursor)`` for well-formed entries; warn once on corruption."""
        poisoned: Any = None
        malformed_cursor: int | None = None
        for entry in self._read_entries():
            raw = entry.get("cursor")
            if raw is None:
                continue
            cursor = self._valid_cursor(raw)
            if cursor is None:
                poisoned = raw
                continue
            if not self._valid_history_payload(entry):
                malformed_cursor = cursor
                continue
            yield entry, cursor
        if poisoned is not None and not self._corruption_logged:
            self._corruption_logged = True
            logger.warning(
                "history.jsonl contains an invalid cursor ({!r}); dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                poisoned,
            )
        if malformed_cursor is not None and not self._malformed_entry_logged:
            self._malformed_entry_logged = True
            logger.warning(
                "history.jsonl contains a malformed entry at cursor {}; dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                malformed_cursor,
            )

    @staticmethod
    def _valid_history_payload(entry: dict[str, Any]) -> bool:
        if not isinstance(entry.get("timestamp"), str):
            return False
        if not isinstance(entry.get("content"), str):
            return False
        session_key = entry.get("session_key")
        return session_key is None or isinstance(session_key, str)

    def _read_cursor_counter(self) -> int | None:
        """Return the persisted cursor counter when it is usable."""
        if not self._cursor_file.exists():
            return None
        with suppress(ValueError, OSError):
            cursor = int(self._cursor_file.read_text(encoding="utf-8").strip())
            if cursor >= 0:
                return cursor
        return None

    def _next_cursor(self) -> int:
        """Read the current cursor counter and return the next value."""
        cursor_counter = self._read_cursor_counter()
        last = self._read_last_entry() or {}
        last_cursor = self._valid_cursor(last.get("cursor"))
        if cursor_counter is not None:
            if last_cursor is not None:
                return max(cursor_counter, last_cursor) + 1
            max_history_cursor = max((c for _, c in self._iter_valid_entries()), default=0)
            return max(cursor_counter, max_history_cursor) + 1

        # Fast path: trust the tail when intact.  Otherwise scan the whole
        # file and take ``max`` — that stays correct even if the monotonic
        # invariant was broken by external writes.
        if last_cursor is not None:
            return last_cursor + 1
        return max((c for _, c in self._iter_valid_entries()), default=0) + 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with a valid cursor > *since_cursor*."""
        return [e for e, c in self._iter_valid_entries() if c > since_cursor]

    def compact_history(self) -> None:
        """Drop oldest processed entries without discarding pending Dream input."""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        last_dream_cursor = self.get_last_dream_cursor()
        first_unprocessed = next(
            (
                index
                for index, entry in enumerate(entries)
                if (
                    (cursor := self._valid_cursor(entry.get("cursor"))) is not None
                    and cursor > last_dream_cursor
                )
            ),
            len(entries),
        )
        keep_from = min(len(entries) - self.max_history_entries, first_unprocessed)
        kept = entries[keep_from:]
        if len(kept) > self.max_history_entries:
            logger.warning(
                "History compaction retained {} unprocessed entries beyond the configured "
                "limit of {}",
                len(kept),
                self.max_history_entries,
            )
        self._write_entries(kept)

    # -- JSONL helpers -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history.jsonl."""
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            parsed: object = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict):
                            entries.append(cast(dict[str, Any], parsed))

        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """Read the last entry from the JSONL file efficiently."""
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [line for line in data.split("\n") if line.strip()]
                if not lines:
                    return None
                parsed: object = json.loads(lines[-1])
                return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history.jsonl with the given entries (atomic write)."""
        tmp_path = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            replace_file_with_retry(tmp_path, self.history_file)

            # fsync the directory so the rename is durable.
            # On Windows, opening a directory with O_RDONLY raises
            # PermissionError — skip the dir sync there (NTFS
            # journals metadata synchronously).
            with suppress(PermissionError):
                fd = os.open(str(self.history_file.parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @classmethod
    def _is_internal_history_session(cls, session_key: str | None) -> bool:
        if not session_key:
            return False
        return (
            session_key in cls._INTERNAL_HISTORY_SESSION_KEYS
            or session_key.startswith(cls._INTERNAL_HISTORY_SESSION_PREFIXES)
        )


    def read_recent_history_for_prompt(
        self,
        since_cursor: int,
        *,
        session_key: str | None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """Return unprocessed history entries safe to inject into a turn prompt."""
        entries = self.read_unprocessed_history(since_cursor=since_cursor)
        if session_key is None:
            return entries
        if not unified_session:
            return [e for e in entries if e.get("session_key") == session_key]

        return [
            entry
            for entry in entries
            if (entry_session := entry.get("session_key")) == session_key
            or not self._is_internal_history_session(entry_session)
        ]


    def _render_current_memory_files(self) -> str:
        """Render the durable memory files' current contents for the Dream prompt.

        Missing files render as ``(empty)``; oversized files are capped. The
        section is the ground truth the model must edit against.
        """
        files = [
            ("SOUL.md", self.soul_file),
            ("USER.md", self.user_file),
            ("memory/MEMORY.md", self.memory_file),
        ]
        blocks = []
        for label, path in files:
            try:
                content = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError:
                content = ""
            if len(content) > self._DREAM_FILE_EMBED_CAP:
                content = truncate_text(content, self._DREAM_FILE_EMBED_CAP) + "\n...[truncated]"
            blocks.append(f"### {label}\n{content}" if content.strip() else f"### {label}\n(empty)")
        return "## Current Memory Files\n" + "\n\n".join(blocks)


    # -- dream cursor --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    def get_latest_cursor(self) -> int:
        return max(self._next_cursor() - 1, 0)

    @property
    def dream_prompt_file(self) -> Path:
        return workspace_prompt_file(self.workspace, "dream")

    def has_dream_prompt_override(self) -> bool:
        return has_workspace_prompt_override(self.dream_prompt_file)

    @staticmethod
    def default_dream_prompt() -> str:
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        return render_template(
            "agent/dream.md",
            strip=True,
            skill_creator_path=str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"),
        )

    def _dream_template(self) -> str:
        text, original_chars = load_workspace_prompt_override(self.dream_prompt_file)
        if text is not None:
            if (
                original_chars > WORKSPACE_PROMPT_MAX_CHARS
                and not self._dream_prompt_oversize_logged
            ):
                self._dream_prompt_oversize_logged = True
                logger.warning(
                    "workspace Dream prompt exceeds {} chars ({}); truncating. "
                    "Further occurrences suppressed.",
                    WORKSPACE_PROMPT_MAX_CHARS, original_chars,
                )
            return text
        return self.default_dream_prompt()

    def build_dream_prompt(self, *, max_entries: int = 20) -> tuple[str, int] | None:
        """Build the Dream prompt with unprocessed history context.

        Returns ``(prompt, last_cursor)`` or ``None`` if nothing to process.

        The current contents of the durable memory files (SOUL.md, USER.md,
        memory/MEMORY.md) reach Dream through the normal agent system context.
        """
        last_cursor = self.get_last_dream_cursor()
        entries = self.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return None

        batch = entries[:max_entries]
        history_text = "\n".join(
            f"[{e['timestamp']}] {truncate_text(e['content'], 1000)}"
            for e in batch
        )
        template = self._dream_template()
        prompt = f"{template}\n\n## Conversation History\n{history_text}"
        return (prompt, batch[-1]["cursor"])

    def dream_content_diff(self) -> str:
        """Structured summary of uncommitted changes to the durable memory files.

        Returns "" when git is unavailable or no content file changed. This is
        the ground-truth input for diff-grounded Dream commit messages.
        """
        if not self._git.is_initialized():
            return ""
        return self._git.summarize_working_tree(list(self._DREAM_CONTENT_PATHS))

    def build_dream_tools(self) -> ToolRegistry:
        """Build the restricted tool registry used by Dream runs."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.apply_patch import ApplyPatchTool
        from nanobot.agent.tools.file_state import FileStates
        from nanobot.agent.tools.filesystem import (
            EditFileTool,
            ReadFileTool,
            WriteFileTool,
        )
        from nanobot.agent.tools.registry import ToolRegistry

        tools = ToolRegistry()
        file_states = FileStates()
        workspace = self.workspace
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        editable_files = [self.memory_file, self.soul_file, self.user_file]

        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_read_allowed_dirs=extra_read,
            file_states=file_states,
        ))
        tools.register(EditFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        tools.register(ApplyPatchTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        tools.register(WriteFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        return tools

    @staticmethod
    def dream_run_completed(
        resp: object | None,
    ) -> bool:
        """Return True when the Dream agent reached a normal terminal response."""
        metadata = getattr(resp, "metadata", None)
        if not isinstance(metadata, dict):
            return False
        return cast(dict[str, Any], metadata).get("_stop_reason") == "completed"

    @staticmethod
    def dream_incompletion_reason(
        resp: object | None,
    ) -> str:
        """Human-readable explanation of why a Dream run cannot advance."""
        metadata = getattr(resp, "metadata", None)
        if isinstance(metadata, dict):
            stop_reason = cast(dict[str, Any], metadata).get("_stop_reason", "unknown")
        else:
            stop_reason = "missing response metadata"
        return f"stop_reason: {stop_reason}"

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            if message.get("tool_calls"):
                lines.append("TOOL_CALLS: " + json.dumps(message["tool_calls"], ensure_ascii=False))
            content = content_with_media_breadcrumbs(
                message.get("role"),
                message.get("content", ""),
                message.get("media"),
            )
            if not content:
                continue
            tools_used = message.get("tools_used")
            tools = (
                f" [tools: {', '.join(cast(list[str], tools_used))}]"
                if tools_used
                else ""
            )
            raw_timestamp = message.get("timestamp")
            timestamp = str(raw_timestamp) if raw_timestamp is not None else "?"
            role = str(message.get("role") or "unknown")
            if role == "tool" and message.get("name"):
                role += " " + str(message["name"])
            label = "TOOL " + role[5:] if role.startswith("tool ") else role.upper()
            lines.append(f"[{timestamp[:16]}] {label}{tools}: {content}")
        return "\n".join(lines)

    def raw_archive(
        self,
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> str:
        """Persist and return a bounded raw checkpoint when summarization degrades."""
        checkpoint = self._build_raw_checkpoint(messages, max_chars=max_chars)
        self.append_history(checkpoint, session_key=session_key)
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )
        return checkpoint

    def _build_raw_checkpoint(
        self,
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
    ) -> str:
        """Build the same bounded checkpoint as :meth:`raw_archive` without writing it."""
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        checkpoint = (
            f"[RAW] {len(messages)} messages\n"
            f"{self._format_messages(public_history_messages(messages))}"
        )
        return self._normalize_history_entry(checkpoint, max_chars=limit)

    # ------------------------------------------------------------------
    # Dream helpers
    # ------------------------------------------------------------------

    @staticmethod
    def dream_session_key() -> str:
        """Return a unique session key for a Dream run, e.g. ``dream:20260528-100000``."""
        return f"dream:{datetime.now():%Y%m%d-%H%M%S}"

    @staticmethod
    def build_dream_commit_message(prefix: str, diff_body: str) -> str:
        """Build a Dream commit message grounded in the real working-tree diff.

        *diff_body* is a structured, machine-derived summary of the actual file
        changes (see :meth:`dream_content_diff` /
        :meth:`GitStore.summarize_working_tree`). The LLM narrative is
        deliberately excluded so the audit record (``/dream-log``) reflects the
        filesystem's truth, not the model's self-report.

        An empty *diff_body* yields the bare *prefix*, which ``auto_commit``
        turns into a no-op when there is nothing to stage.
        """
        diff_body = (diff_body or "").strip()
        if not diff_body:
            return prefix
        return f"{prefix}\n\n{diff_body}"

    @staticmethod
    def prune_dream_sessions(sessions: SessionManager, *, keep: int = 10) -> None:
        """Remove the oldest Dream session files, keeping only the N most recent.

        Only current base64url-encoded Dream session keys are considered.
        Non-dream session files are never touched.
        """
        with sessions.locked_session_files() as sessions_dir:
            dream_files: list[tuple[Path, str]] = []
            for path in sessions_dir.glob("*.jsonl"):
                decoded_key = SessionManager.decode_storage_key(path.stem)
                if decoded_key is not None and decoded_key.startswith("dream:"):
                    dream_files.append((path, decoded_key))
            dream_files.sort(key=lambda item: item[0].stat().st_mtime)

            for path, key in dream_files[: max(0, len(dream_files) - keep)]:
                if sessions.delete_session(key):
                    logger.debug("Pruned old dream session: {}", path.stem)
                else:
                    logger.warning("Failed to prune dream session {}", path)


# ---------------------------------------------------------------------------
# Memory ingestion and context-pressure coordination
# ---------------------------------------------------------------------------

# Raw fallbacks use a tighter cap. Completed model summaries may scale with the
# configured generation budget, while append_history() still enforces the
# emergency hard cap against pathological provider output.
_RAW_ARCHIVE_MAX_CHARS = 16_000   # fallback dump (LLM failed)
_HISTORY_ENTRY_HARD_CAP = 64_000  # emergency cap in append_history


_ARCHIVE_SUMMARY_MAX_CHARS = 8_000    # LLM-produced long-term candidates
_CONTINUATION_SUMMARY_MAX_CHARS = SUMMARY_MAX_CHARS
_CONTINUATION_OPEN = "<continuation>"
_CONTINUATION_CLOSE = "</continuation>"
_CANDIDATES_OPEN = "<memory-candidates>"
_CANDIDATES_CLOSE = "</memory-candidates>"
_DREAM_PROMPT_MAX_CHARS = 32_000      # workspace-local Dream prompt override


class MemoryArchiver:
    @staticmethod
    def _extract_section(text: str, opening: str, closing: str) -> str:
        start = text.find(opening)
        if start < 0:
            return ""
        end = text.find(closing, start + len(opening))
        if end < 0:
            return ""
        return text[start + len(opening):end].strip()


    """Write durable transcript batches to the Memory ingestion journal.

    The archiver deliberately has no SessionManager dependency: it may read a
    captured transcript batch and append to history.jsonl, but it cannot mutate
    provider continuation state or advance a session watermark.
    """

    def __init__(
        self,
        store: MemoryStore,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        resolve_prompt_context: Callable[[Session], tuple[str | None, Path | None]] | None = None,
    ) -> None:
        self.store = store
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._resolve_prompt_context = resolve_prompt_context

    def _raw_checkpoint(
        self,
        messages: list[dict[str, Any]],
        *,
        session_key: str,
        previous_summary: str | None,
        max_tokens: int,
    ) -> None:
        """Persist failed input for audit without issuing a replacement checkpoint."""
        self.store.raw_archive(messages, session_key=session_key)
        return None

    @staticmethod
    def _combine_raw_checkpoint(
        raw: str,
        *,
        previous_summary: str | None,
        max_tokens: int,
    ) -> str:
        """Return a bounded checkpoint that preserves prior and newly archived context."""
        token_limit = max(1, max_tokens)
        if not previous_summary:
            return truncate_text_to_tokens(raw, token_limit)

        combined = (
            "[Previous archived context]\n"
            f"{previous_summary}\n\n"
            "[Newly archived raw context]\n"
            f"{raw}"
        )
        bounded = truncate_text_to_tokens(combined, token_limit)
        if bounded == combined:
            return combined

        # Keep evidence from both sides when their full concatenation cannot fit.
        section_limit = max(1, (token_limit - 32) // 2)
        return truncate_text_to_tokens(
            "[Previous archived context]\n"
            f"{truncate_text_to_tokens(previous_summary, section_limit)}\n\n"
            "[Newly archived raw context]\n"
            f"{truncate_text_to_tokens(raw, section_limit)}",
            token_limit,
        )

    async def archive(
        self,
        source_messages: list[dict[str, Any]],
        *,
        runtime: LLMRuntime,
        session_key: str,
        history: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        previous_summary: str | None = None,
        input_token_budget: int | None = None,
        fallback_max_tokens: int | None = None,
        provider_state: ProviderConversationState | None = None,
    ) -> str | None:
        """Summarize an isolated full batch; failed summaries never advance coverage."""
        if not source_messages or uses_native_context(runtime.provider):
            return None
        stage = "prepare"
        estimated = budget = response = summary_chars = None
        try:
            formatted = MemoryStore._format_messages(source_messages)
            if previous_summary:
                formatted = (
                    "<existing-continuation>\n"
                    f"{previous_summary}\n"
                    "</existing-continuation>\n\n"
                    "<archived-messages>\n"
                    f"{formatted}\n"
                    "</archived-messages>"
                )
            request = [
                {"role": "system", "content": render_template(
                    "agent/consolidator_archive.md", strip=True,
                )},
                {"role": "user", "content": formatted},
            ]
            stage = "input_budget"
            estimated, _ = estimate_prompt_tokens_chain(runtime.provider, runtime.model, request, None)
            budget = (input_token_budget if input_token_budget is not None else runtime.context_window_tokens - runtime.generation.max_tokens - 1024)
            # 覆盖事务不能用截断后的输入代表完整原文；连系统指令也计入预算。
            if not 0 < estimated <= budget:
                raise SummaryValidationError(
                    "input_budget_exceeded", "完整摘要输入无法容纳，保留原文等待更小的安全分段。",
                )
            stage = "request"
            response = await request_summary(
                runtime.provider, model=runtime.model, messages=request,
                max_tokens=runtime.generation.max_tokens,
                generation=runtime.generation,
            )
            stage = "response_validation"
            if response.finish_reason not in ("stop", "end_turn") or (
                isinstance(response.tool_calls, list) and response.tool_calls
            ):
                # 工具输出不执行，也不能当成有效摘要。
                raise SummaryValidationError("response_incomplete", "摘要请求未完整完成。")
            stage = "summary_validation"
            response_text = strip_think(response.content or "")
            continuation = self._extract_section(
                response_text, _CONTINUATION_OPEN, _CONTINUATION_CLOSE
            )
            candidates = self._extract_section(
                response_text, _CANDIDATES_OPEN, _CANDIDATES_CLOSE
            )
            # Accept old summarizers during a rolling upgrade, but never inject
            # their unstructured output into the long-term history candidate stream.
            if not continuation and response_text.strip():
                # 仅对完全非结构化旧输出兼容；残缺标签不能冒充续接摘要。
                if any(tag in response_text for tag in (
                    _CONTINUATION_OPEN, _CONTINUATION_CLOSE, _CANDIDATES_OPEN, _CANDIDATES_CLOSE,
                )):
                    raise SummaryValidationError("continuation_missing", "摘要缺少完整续接部分。")
                continuation = response_text.strip()
            summary_chars = len(continuation)
            if not valid_summary(continuation):
                raise SummaryValidationError(
                    "summary_invalid", "摘要为空、无恢复内容或超出恢复预算。",
                )
            stage = "candidate_archive"
            if candidates and candidates != "(nothing)":
                self.store.append_history(
                    candidates,
                    max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
                    session_key=session_key,
                )
            return continuation
        except Exception as exc:
            record_summary_failure(
                getattr(self, "sessions", None), session_key=session_key, provider=runtime.provider, model=runtime.model,
                stage=stage, error=exc, estimated_tokens=estimated, input_budget=budget,
                response=response, summary_chars=summary_chars,
            )
            try:
                self.store.raw_archive(source_messages, session_key=session_key)
            except Exception as archive_exc:
                record_summary_failure(
                    getattr(self, "sessions", None), session_key=session_key, provider=runtime.provider, model=runtime.model,
                    stage="raw_archive", error=archive_exc, estimated_tokens=estimated,
                    input_budget=budget,
                )
                raise
            return None

    async def archive_session(
        self,
        session: Session,
        *,
        archive_end: int,
        runtime: LLMRuntime,
        input_token_budget: int,
    ) -> str | None:
        """Archive a captured session prefix without mutating the session."""
        messages = [
            message for message in session.messages[session.last_archived:archive_end]
            if not message.get("_command") and not is_summary_checkpoint(message)
        ]
        if not messages:
            return None
        session_summary = session_summary_from_metadata(
            session.metadata,
            fallback_last_active=session.updated_at,
        )
        previous_summary = session_summary["text"] if session_summary else None

        if input_token_budget <= 0:
            logger.debug(
                "Memory archive has no safe input budget for {}; raw-dumping",
                session.key,
            )
            return self._raw_checkpoint(
                messages,
                session_key=session.key,
                previous_summary=previous_summary,
                max_tokens=runtime.generation.max_tokens,
            )
        prefix = Session(
            key=session.key,
            messages=list(session.messages[:archive_end]),
            last_consolidated=session.last_archived,
        )
        history = prefix.get_history(max_tokens=input_token_budget)
        archive_history = Session(
            key=session.key,
            messages=messages,
        ).get_history()
        if not archive_history or history[-len(archive_history):] != archive_history:
            logger.debug(
                "Memory archive cannot replay the full chunk for {}; raw-dumping",
                session.key,
            )
            return self._raw_checkpoint(
                messages,
                session_key=session.key,
                previous_summary=previous_summary,
                max_tokens=runtime.generation.max_tokens,
            )
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        workspace: Path | None = None
        if self._resolve_prompt_context is not None:
            channel, workspace = self._resolve_prompt_context(session)
        history_messages = self._build_messages(
            history=history,
            current_message=None,
            channel=channel,
            session_summary=session_summary,
            workspace=workspace,
        )
        tools = self._get_tool_definitions()
        return await self.archive(
            messages,
            runtime=runtime,
            session_key=session.key,
            history=history_messages,
            request_tools=tools,
            previous_summary=previous_summary,
            input_token_budget=input_token_budget,
        )


class Consolidator:
    _MAX_CONSOLIDATION_ROUNDS = 5
    """Coordinate session Memory checkpoints through ``MemoryArchiver``."""

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        store: MemoryStore,
        sessions: SessionManager,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        resolve_prompt_context: Callable[[Session], tuple[str | None, Path | None]] | None = None,
        *,
        provider: LLMProvider | None = None,
        model: str = "",
        context_window_tokens: int = 0,
        max_completion_tokens: int = 4096,
        consolidation_ratio: float = 0.25,
        consolidation_trigger_ratio: float = 0.7,
        unified_session: bool = False,
        context_block_limit: int | None = None,
    ):
        self.store = store
        self.sessions = sessions
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self.archiver = MemoryArchiver(
            store=store,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            resolve_prompt_context=resolve_prompt_context,
        )
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

        if provider is not None:
            self._legacy___init__(store, provider, model, sessions, context_window_tokens,
                                  build_messages, get_tool_definitions, max_completion_tokens,
                                  consolidation_ratio, consolidation_trigger_ratio,
                                  unified_session, context_block_limit)

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def summarize_transcript(
        self,
        accepted_messages: list[dict[str, Any]],
        previous_summary: str | None,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]],
        provider_state: ProviderConversationState | None = None,
    ) -> str | None:
        """Summarize the exact transcript prefix already accepted by the model."""
        source_messages = [
            dict(message)
            for message in accepted_messages
            if message.get("role") != "system"
        ]
        if not source_messages:
            return None

        max_output_tokens = max(0, runtime.generation.max_tokens)
        input_token_budget = runtime.context_window_tokens - max_output_tokens
        checkpoint_tokens = min(
            max_output_tokens,
            max(1, (input_token_budget - self._SAFETY_BUFFER) // 2),
        )

        summary = await self.archiver.archive(
            source_messages,
            runtime=runtime,
            session_key=session_key,
            history=accepted_messages,
            request_tools=tools,
            previous_summary=previous_summary,
            input_token_budget=input_token_budget,
            fallback_max_tokens=max(1, checkpoint_tokens),
            provider_state=provider_state,
        )
        if summary is None:
            return None
        if truncate_text_to_tokens(summary, max(1, max_output_tokens)) != summary:
            return None
        return summary

    async def summarize_provider_compaction(
        self,
        state: ProviderConversationState,
        fallback_messages: list[dict[str, Any]],
        previous_summary: str | None,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]],
    ) -> str | None:
        """Prompt a native compacted state without replaying its raw history."""
        return await self.summarize_transcript(
            fallback_messages,
            previous_summary,
            runtime=runtime,
            session_key=session_key,
            tools=tools,
            provider_state=state,
        )

    @staticmethod
    def _full_replay_history(
        session: Session,
    ) -> list[dict[str, Any]]:
        """Return all messages that can reach the next model prompt."""
        if not session.messages:
            return []
        return session.get_history()

    def estimate_session_prompt_tokens(
        self,
        session: Session,
        *,
        runtime: LLMRuntime | None = None,
        session_summary: str | None = None,
    ) -> tuple[int, str]:
        """Estimate the full replay; support the previous explicit-provider SDK."""
        if runtime is None:
            return self._legacy_estimate_session_prompt_tokens(session)
        history = self._full_replay_history(session)
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        summary = session_summary_from_metadata(
            session.metadata,
            fallback_last_active=session.updated_at,
        )
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            session_summary=summary,
        )
        return estimate_prompt_tokens_chain(
            runtime.provider,
            runtime.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    def _input_token_budget(self, runtime: LLMRuntime) -> int:
        """Available input token budget for consolidation LLM."""
        return (
            runtime.context_window_tokens
            - runtime.generation.max_tokens
            - self._SAFETY_BUFFER
        )

    async def archive_session(
        self,
        session: Session,
        *,
        archive_end: int,
        runtime: LLMRuntime,
    ) -> str | None:
        """Archive one captured session range through the shared Memory path."""
        return await self.archiver.archive_session(
            session,
            archive_end=archive_end,
            runtime=runtime,
            input_token_budget=self._input_token_budget(runtime),
        )

    async def compact_idle_session(
        self,
        session_key: str,
        *,
        runtime: LLMRuntime | None = None,
        max_suffix: int = 0,
        events: EventSink = NO_EVENTS,
    ) -> str | None:
        """Replace archived history with a summary checkpoint.

        ``max_suffix`` is accepted for SDK compatibility and no longer retains
        archived messages. All compaction triggers share checkpoint replay.
        """
        if runtime is None:
            return await self._legacy_compact_idle_session(session_key, max_suffix=max_suffix)
        if uses_native_context(runtime.provider):
            return None
        lock = self.get_lock(session_key)
        async with lock:
            self.sessions.invalidate(session_key)
            session = self.sessions.get_or_create(session_key)

            snapshot = deepcopy(session.__dict__)
            archive_start = session.last_archived
            messages_to_archive = list(session.messages[archive_start:])
            has_new_messages = any(
                not message.get("_command") and not is_summary_checkpoint(message)
                for message in messages_to_archive
            )
            if not has_new_messages:
                return ""

            compaction_id = uuid4().hex
            await events.emit(
                ContextCompactionEvent(compaction_id=compaction_id, phase="started"),
            )
            last_active = session.updated_at
            archive_end = archive_start + len(messages_to_archive)
            try:
                summary = await self.archive_session(
                    session, archive_end=archive_end, runtime=runtime,
                )
                if summary:
                    if session.__dict__ != snapshot:
                        raise SummaryTransactionError("摘要期间会话发生变化，保留原文等待重试。")
                    # Concurrent appends remain after the captured boundary.
                    session.commit_summary_checkpoint(
                        summary, insert_at=archive_end, last_active=last_active,
                    )
                    # Resume from the summary and retained transcript, not the old provider history.
                    session.provider_state = None
                    try:
                        self.sessions.save(session)
                    except BaseException:
                        session.__dict__.clear()
                        session.__dict__.update(snapshot)
                        raise
            except (Exception, asyncio.CancelledError) as exc:
                await events.emit(
                    ContextCompactionEvent(
                        compaction_id=compaction_id,
                        phase="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    ),
                )
                raise
            if not summary:
                await events.emit(
                    ContextCompactionEvent(compaction_id=compaction_id, phase="failed"),
                )
                return None

            await events.emit(
                ContextCompactionEvent(
                    compaction_id=compaction_id,
                    phase="succeeded",
                ),
            )

            logger.info(
                "Idle-session compact for {}: archived={}, visible={}, retained={}, summary={}",
                session_key,
                len(messages_to_archive),
                len(session.get_history()),
                len(session.messages),
                bool(summary),
            )

            return summary

    def _legacy___init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        consolidation_ratio: float = 0.25,
        consolidation_trigger_ratio: float = 0.7,
        unified_session: bool = False,
        context_block_limit: int | None = None,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.context_block_limit = context_block_limit
        self.max_completion_tokens = max_completion_tokens
        self.consolidation_ratio = consolidation_ratio
        self.consolidation_trigger_ratio = max(consolidation_ratio, consolidation_trigger_ratio)
        self.unified_session = unified_session
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks = (
            weakref.WeakValueDictionary()
        )
        # Exact preflight probes are authoritative. Background post-save probes
        # may reuse a deliberately conservative upper bound; the next foreground
        # turn always rebuilds the full prompt before calling the model.
        self._background_probe_cache: dict[str, tuple[int, int, int]] = {}


    def set_provider(
        self,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = provider.generation.max_tokens
        self._background_probe_cache.clear()


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


    @staticmethod
    def _full_unconsolidated_history(
        session: Session,
    ) -> list[dict[str, Any]]:
        """Return the whole unconsolidated tail for consolidation decisions."""
        unconsolidated_count = len(session.messages) - session.last_consolidated
        if unconsolidated_count <= 0:
            return []
        return session.get_history(max_messages=unconsolidated_count)


    @staticmethod
    def _replay_overflow_boundary(
        session: Session,
        replay_max_messages: int | None,
    ) -> int | None:
        if not replay_max_messages or replay_max_messages <= 0:
            return None
        tail = list(enumerate(session.messages[session.last_consolidated:], session.last_consolidated))
        if len(tail) <= replay_max_messages:
            return None

        tail_messages = [message for _idx, message in tail]
        start_idx = recent_message_start_index(
            tail_messages,
            replay_max_messages,
            extend_to_user=True,
        )
        sliced = tail[start_idx:]
        for i, (_idx, message) in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1][1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        legal_start = find_legal_message_start([message for _idx, message in sliced])
        if legal_start:
            sliced = sliced[legal_start:]
        if not sliced:
            return len(session.messages)

        first_visible_idx = sliced[0][0]
        if first_visible_idx <= session.last_consolidated:
            return None
        return first_visible_idx


    async def _consolidate_replay_overflow(
        self,
        session: Session,
        replay_max_messages: int | None,
    ) -> str | None:
        """Archive messages that would be hidden by the replay message window."""
        end_idx = self._replay_overflow_boundary(session, replay_max_messages)
        if end_idx is None:
            return None
        chunk = session.messages[session.last_consolidated:end_idx]
        if not chunk:
            return None
        logger.info(
            "Replay-window consolidation for {}: chunk={} msgs, replay_max={}",
            session.key,
            len(chunk),
            replay_max_messages,
        )
        snapshot = SummarySnapshot.capture(session)
        summary = await self.archive(
            chunk,
            session_key=session.key,
            prior_continuation=self._summary_text(snapshot.metadata),
        )
        if not isinstance(summary, str) or not valid_summary(summary):
            return None
        commit_summary(
            self.sessions, session, snapshot, summary, covered_messages=chunk,
            end_cursor=end_idx, reason="replay_overflow",
        )
        return summary


    @staticmethod
    def _summary_text(metadata: dict[str, Any]) -> str | None:
        """Read the new continuation summary, with a one-release legacy fallback."""
        for key in ("_continuation_summary", "_last_summary"):
            value = metadata.get(key)
            text = value.get("text") if isinstance(value, dict) else value
            if isinstance(text, str) and text.strip():
                return text.strip()
        return None


    def _legacy_estimate_session_prompt_tokens(
        self,
        session: Session,
    ) -> tuple[int, str]:
        """Estimate prompt size from the full unconsolidated session tail."""
        history = self._full_unconsolidated_history(session)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        # Include archived summary in estimation so the budget accounts for it.
        summary = self._summary_text(session.metadata)
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            sender_id=None,
            session_summary=summary,
            session_metadata=session.metadata,
            session_key=session.key,
            unified_session=self.unified_session,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )


    @property
    def _legacy__input_token_budget(self) -> int:
        """Available input token budget for consolidation LLM."""
        return resolve_context_budget(
            self.provider, self.context_window_tokens, self.max_completion_tokens,
            self.context_block_limit, self._SAFETY_BUFFER,
        ).input_tokens or 0


    @staticmethod
    def _extract_section(text: str, opening: str, closing: str) -> str:
        start = text.find(opening)
        if start < 0:
            return ""
        end = text.find(closing, start + len(opening))
        if end < 0:
            return ""
        return text[start + len(opening):end].strip()


    async def archive(
        self,
        messages: list[dict],
        *,
        session_key: str | None = None,
        summary_messages: list[dict] | None = None,
        prior_continuation: str | None = None,
    ) -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        ``messages`` are the messages being archived (removed from the live
        session); they are what gets raw-dumped if the LLM call fails.
        ``summary_messages``, when given, lets callers include retained
        messages in the summary without archiving them.

        Returns the summary text on success, None if nothing to archive.
        """
        if not messages:
            return None
        messages_to_summarize = summary_messages if summary_messages is not None else messages
        stage = "prepare"
        estimated = budget = response = summary_chars = None
        try:
            formatted = MemoryStore._format_messages(messages_to_summarize)
            if prior_continuation:
                formatted = (
                    "<existing-continuation>\n"
                    f"{prior_continuation}\n"
                    "</existing-continuation>\n\n"
                    "<archived-messages>\n"
                    f"{formatted}\n"
                    "</archived-messages>"
                )
            request = [
                {"role": "system", "content": render_template(
                    "agent/consolidator_archive.md", strip=True,
                )},
                {"role": "user", "content": formatted},
            ]
            stage = "input_budget"
            estimated, _ = estimate_prompt_tokens_chain(self.provider, self.model, request, None)
            budget = self._legacy__input_token_budget
            # 覆盖事务不能用截断后的输入代表完整原文；连系统指令也计入预算。
            if not 0 < estimated <= budget:
                raise SummaryValidationError(
                    "input_budget_exceeded", "完整摘要输入无法容纳，保留原文等待更小的安全分段。",
                )
            stage = "request"
            response = await request_summary(
                self.provider, model=self.model, messages=request,
                max_tokens=self.max_completion_tokens,
            )
            stage = "response_validation"
            if response.finish_reason not in ("stop", "end_turn") or (
                isinstance(response.tool_calls, list) and response.tool_calls
            ):
                # 工具输出不执行，也不能当成有效摘要。
                raise SummaryValidationError("response_incomplete", "摘要请求未完整完成。")
            stage = "summary_validation"
            response_text = strip_think(response.content or "")
            continuation = self._extract_section(
                response_text, _CONTINUATION_OPEN, _CONTINUATION_CLOSE
            )
            candidates = self._extract_section(
                response_text, _CANDIDATES_OPEN, _CANDIDATES_CLOSE
            )
            # Accept old summarizers during a rolling upgrade, but never inject
            # their unstructured output into the long-term history candidate stream.
            if not continuation and response_text.strip():
                # 仅对完全非结构化旧输出兼容；残缺标签不能冒充续接摘要。
                if any(tag in response_text for tag in (
                    _CONTINUATION_OPEN, _CONTINUATION_CLOSE, _CANDIDATES_OPEN, _CANDIDATES_CLOSE,
                )):
                    raise SummaryValidationError("continuation_missing", "摘要缺少完整续接部分。")
                continuation = response_text.strip()
            summary_chars = len(continuation)
            if not valid_summary(continuation):
                raise SummaryValidationError(
                    "summary_invalid", "摘要为空、无恢复内容或超出恢复预算。",
                )
            stage = "candidate_archive"
            if candidates and candidates != "(nothing)":
                self.store.append_history(
                    candidates,
                    max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
                    session_key=session_key,
                )
            return continuation
        except Exception as exc:
            record_summary_failure(
                self.sessions, session_key=session_key, provider=self.provider, model=self.model,
                stage=stage, error=exc, estimated_tokens=estimated, input_budget=budget,
                response=response, summary_chars=summary_chars,
            )
            try:
                self.store.raw_archive(messages, session_key=session_key)
            except Exception as archive_exc:
                record_summary_failure(
                    self.sessions, session_key=session_key, provider=self.provider, model=self.model,
                    stage="raw_archive", error=archive_exc, estimated_tokens=estimated,
                    input_budget=budget,
                )
                raise
            return None


    def _background_probe_upper_bound(self, session: Session, budget: int) -> int | None:
        """Return a safe cached estimate for a cheap post-save check, if possible."""
        cached = self._background_probe_cache.get(session.key)
        if cached is None:
            return None
        estimated, message_count, last_consolidated = cached
        if last_consolidated != session.last_consolidated or message_count > len(session.messages):
            return None
        # Reuse only when the prior exact probe was far below the limit and the
        # newly appended tail is small. This is an upper bound for persisted
        # messages; foreground preflight still catches dynamic prompt changes.
        added = sum(estimate_message_tokens(message) for message in session.messages[message_count:])
        if estimated > budget // 2 or added > max(1024, budget // 10):
            return None
        return estimated + added


    def _remember_exact_probe(self, session: Session, estimated: int) -> None:
        self._background_probe_cache[session.key] = (
            estimated,
            len(session.messages),
            session.last_consolidated,
        )


    async def maybe_consolidate_by_tokens(
        self,
        session: Session,
        *,
        replay_max_messages: int | None = None,
        background: bool = False,
        completed_goal: bool = False,
    ) -> None:
        """Archive old messages; stop explicitly when a required summary cannot commit.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        # 原生线程自行压缩历史，且没有隔离的无工具摘要协议。
        # 前台、后台和消息数溢出均保留原文，不进入普通摘要事务。
        if uses_native_context(self.provider) or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            # Refresh session reference: AutoCompact may have replaced it.
            fresh = self.sessions.get_or_create(session.key)
            if fresh is not session:
                session = fresh
            if not session.messages:
                return

            budget = self._legacy__input_token_budget
            if budget <= 0:
                # 没有空间时不归档；实际请求由 runner 的预算门禁明确停止。
                return
            target = int(budget * self.consolidation_ratio)
            trigger = int(budget * self.consolidation_trigger_ratio)
            replay_boundary = self._replay_overflow_boundary(session, replay_max_messages)
            last_summary = await self._consolidate_replay_overflow(
                session, replay_max_messages,
            )
            if replay_boundary is not None and not valid_summary(last_summary):
                raise SummaryTransactionError(
                    "上下文摘要失败，已保留原文；本次请求停止，未裁剪重试。"
                )
            last_summary = last_summary or self._summary_text(session.metadata)
            estimated: int | None = (
                self._background_probe_upper_bound(session, budget) if background else None
            )
            source = "background_upper_bound" if estimated is not None else ""
            if estimated is not None:
                logger.debug(
                    "Token consolidation background fast path {}: estimate={}/{}",
                    session.key,
                    estimated,
                    budget,
                )
            else:
                probe_started_at = asyncio.get_running_loop().time()
                try:
                    estimated, source = await asyncio.to_thread(
                        self.estimate_session_prompt_tokens,
                        session,
                    )
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                logger.debug(
                    "Token consolidation probe {}: estimate={}, source={}, duration_ms={:.1f}",
                    session.key,
                    estimated,
                    source,
                    (asyncio.get_running_loop().time() - probe_started_at) * 1000,
                )
                self._remember_exact_probe(session, estimated)
            if estimated <= 0:
                return
            if estimated < trigger and not completed_goal:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} trigger={} via {}, msgs={}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    trigger,
                    source,
                    unconsolidated_count,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                # A completed goal may archive one finished task segment even
                # below the normal high-water mark, but never cascades through
                # multiple older tasks in the same pass.
                if estimated <= target or (completed_goal and round_num > 0):
                    break

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                end_idx = boundary[0]

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    break

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                snapshot = SummarySnapshot.capture(session)
                summary = await self.archive(
                    chunk,
                    session_key=session.key,
                    prior_continuation=last_summary,
                )
                if not isinstance(summary, str) or not valid_summary(summary):
                    # 原文归档只是审计线索，不推进覆盖，也不在本轮反复重试。
                    raise SummaryTransactionError(
                        "上下文摘要失败，已保留原文；本次请求停止，未裁剪重试。"
                    )
                commit_summary(
                    self.sessions, session, snapshot, summary, covered_messages=chunk,
                    end_cursor=end_idx, reason="token_budget",
                )
                last_summary = summary

                probe_started_at = asyncio.get_running_loop().time()
                try:
                    estimated, source = await asyncio.to_thread(
                        self.estimate_session_prompt_tokens,
                        session,
                    )
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                logger.debug(
                    "Token consolidation probe {}: estimate={}, source={}, duration_ms={:.1f}",
                    session.key,
                    estimated,
                    source,
                    (asyncio.get_running_loop().time() - probe_started_at) * 1000,
                )
                self._remember_exact_probe(session, estimated)
                if estimated <= 0:
                    break

            if completed_goal:
                session.metadata.pop("_completed_goal_needs_compaction", None)
                self.sessions.save(session)


    async def _legacy_compact_idle_session(
        self,
        session_key: str,
        max_suffix: int = 8,
    ) -> str | None:
        """Hard-truncate an idle session under the consolidation lock.

        Used by AutoCompact so all session mutation goes through a single
        lock-protected path.  Returns the summary text on success, ``None``
        if the LLM failed (raw_archive fallback), or ``""`` if there was
        nothing to archive.
        """
        if uses_native_context(self.provider):
            return None
        lock = self.get_lock(session_key)
        async with lock:
            self.sessions.invalidate(session_key)
            session = self.sessions.get_or_create(session_key)

            messages_to_summarize = list(session.messages[session.last_consolidated:])
            if not messages_to_summarize:
                self.sessions.save(session)
                return ""

            start_index = (recent_message_start_index(
                messages_to_summarize, max_suffix, extend_to_user=True,
            ) if max_suffix > 0 else len(messages_to_summarize))
            messages_to_keep = messages_to_summarize[start_index:]
            messages_to_remove = messages_to_summarize[:start_index]

            if not messages_to_remove and not messages_to_keep:
                self.sessions.save(session)
                return ""

            snapshot = SummarySnapshot.capture(session)
            summary: str | None = ""
            if messages_to_remove:
                # Summarize the retained suffix too, but only remove/raw-dump
                # the messages that are no longer kept in the live session.
                summary = await self.archive(
                    messages_to_remove,
                    session_key=session_key,
                    summary_messages=messages_to_summarize,
                    prior_continuation=self._summary_text(session.metadata),
                )

            if not messages_to_remove:
                return ""
            if not isinstance(summary, str) or not valid_summary(summary):
                return None
            commit_summary(
                self.sessions, session, snapshot, summary,
                covered_messages=messages_to_remove, end_cursor=len(snapshot.messages),
                retained_messages=messages_to_keep, reason="idle",
            )

            if messages_to_remove:
                logger.info(
                    "Idle-session compact for {}: archived={}, kept={}, summary={}",
                    session_key,
                    len(messages_to_remove),
                    len(messages_to_keep),
                    bool(summary),
                )

            return summary
