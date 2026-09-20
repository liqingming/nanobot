"""Search tools: file discovery and grep."""

# pyright: reportIncompatibleMethodOverride=false, reportPrivateUsage=false

from __future__ import annotations

import asyncio
import fnmatch
import heapq
import os
import re
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, TypeVar

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.filesystem import ListDirTool, _FsTool
from nanobot.utils.document import (
    LocatedDocumentLine,
    PdfPageRangeError,
    open_document_line_source,
)

_DEFAULT_HEAD_LIMIT = 100
_DEFAULT_FILE_HEAD_LIMIT = 100
_TOOL_POLICY_METADATA_KEY = "tool_policy"
_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".pptx"})
T = TypeVar("T")
_TYPE_GLOB_MAP = {
    "py": ("*.py", "*.pyi"),
    "python": ("*.py", "*.pyi"),
    "js": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "ts": ("*.ts", "*.tsx", "*.mts", "*.cts"),
    "tsx": ("*.tsx",),
    "jsx": ("*.jsx",),
    "json": ("*.json",),
    "md": ("*.md", "*.mdx"),
    "markdown": ("*.md", "*.mdx"),
    "go": ("*.go",),
    "rs": ("*.rs",),
    "rust": ("*.rs",),
    "java": ("*.java",),
    "sh": ("*.sh", "*.bash"),
    "yaml": ("*.yaml", "*.yml"),
    "yml": ("*.yaml", "*.yml"),
    "toml": ("*.toml",),
    "sql": ("*.sql",),
    "html": ("*.html", "*.htm"),
    "css": ("*.css", "*.scss", "*.sass"),
}


@dataclass(slots=True)
class _PendingContextMatch:
    lines: list[LocatedDocumentLine]
    match_index: int
    match_start: int
    remaining_after: int


@dataclass(slots=True)
class _FindFilesEntry:
    path: Path
    rel_path: str
    display_path: str
    name: str
    is_dir: bool


class _FindFilesCancelledError(Exception):
    """Stop a worker scan after its owning async task was cancelled."""


class _FindFilesBudgetExceededError(Exception):
    """Stop an unbounded filesystem scan at its configured budget."""


@dataclass(slots=True)
class _FindFilesBudget:
    cancelled: threading.Event
    deadline: float
    max_paths: int
    scanned_paths: int = 0

    def checkpoint(self) -> None:
        if self.cancelled.is_set():
            raise _FindFilesCancelledError
        if time.monotonic() >= self.deadline:
            raise _FindFilesBudgetExceededError("time")

    def visit_path(self) -> None:
        self.checkpoint()
        self.scanned_paths += 1
        if self.scanned_paths > self.max_paths:
            raise _FindFilesBudgetExceededError("paths")


def _normalize_pattern(pattern: str) -> str:
    return pattern.strip().replace("\\", "/")


def _match_glob(rel_path: str, name: str, pattern: str) -> bool:
    normalized = _normalize_pattern(pattern)
    if not normalized:
        return False
    if "/" in normalized or normalized.startswith("**"):
        pattern_parts = PurePosixPath(normalized).parts
        if "**" in pattern_parts:
            path_parts = PurePosixPath(rel_path).parts
            # Keep relative patterns suffix-matched, as with PurePath.match.
            matched = [True] * (len(path_parts) + 1)
            for part in pattern_parts:
                if part == "**":
                    # A globstar consumes zero or more complete path segments.
                    for index in range(1, len(matched)):
                        matched[index] = matched[index] or matched[index - 1]
                else:
                    matched = [False] + [
                        matched[index] and fnmatch.fnmatchcase(path_part, part)
                        for index, path_part in enumerate(path_parts)
                    ]
            return matched[-1]
        return PurePosixPath(rel_path).match(normalized)
    return fnmatch.fnmatch(name, normalized)


def _is_binary(raw: bytes) -> bool:
    if b"\x00" in raw:
        return True
    sample = raw[:4096]
    if not sample:
        return False
    non_text = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return (non_text / len(sample)) > 0.2


def _excel_column(index: int) -> str:
    """Return a 1-indexed spreadsheet column label without importing openpyxl."""
    label = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _paginate(items: list[T], limit: int | None, offset: int) -> tuple[list[T], bool]:
    if limit is None:
        return items[offset:], False
    sliced = items[offset : offset + limit]
    truncated = len(items) > offset + limit
    return sliced, truncated


def _pagination_note(limit: int | None, offset: int, truncated: bool) -> str | None:
    if truncated:
        if limit is None:
            return f"(pagination: offset={offset})"
        return f"(pagination: limit={limit}, offset={offset})"
    if offset > 0:
        return f"(pagination: offset={offset})"
    return None


def _matches_type(name: str, file_type: str | None) -> bool:
    if not file_type:
        return True
    lowered = file_type.strip().lower()
    if not lowered:
        return True
    patterns = _TYPE_GLOB_MAP.get(lowered, (f"*.{lowered}",))
    return any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in patterns)


def _matches_query(rel_path: str, query: str | None, mode: str = "all") -> bool:
    if not query:
        return True
    haystack = rel_path.lower()
    normalized = query.lower().strip()
    terms = [part for part in normalized.split() if part]
    if mode == "phrase":
        return normalized in haystack
    if mode == "any":
        return any(term in haystack for term in terms)
    return all(term in haystack for term in terms)


class _SearchTool(_FsTool):
    """Shared bounded tree-walk behavior for search tools."""

    _SCAN_BUDGET_SECONDS = 30.0
    _SCAN_BUDGET_FILES = 20_000
    _SCAN_BUDGET_BYTES = 512 * 1024 * 1024

    @property
    def execution_timeout_s(self) -> float | None:
        # The internal 30s checkpoint normally wins; this guard covers a stuck filesystem call.
        return 45.0

    def is_concurrency_safe_call(self, params: Any) -> bool:
        path = str(params.get("path", ".")) if isinstance(params, dict) else "."
        try:
            return self._resolve(path or ".").is_file()
        except Exception:
            return False

    def _is_broad_unfiltered(self, target: Path, *, glob: str | None, file_type: str | None) -> bool:
        if target.is_file():
            return False
        workspace = self._display_workspace()
        if workspace is None:
            return False
        with suppress(OSError, RuntimeError, ValueError):
            resolved_target = target.resolve(strict=False)
            resolved_workspace = workspace.resolve(strict=False)
            # Searching above the workspace is always an explicit scope expansion. At the
            # workspace root, only reject an effectively unfiltered recursive wildcard;
            # ordinary project-root grep/find calls remain ergonomic and are budgeted.
            if resolved_target in resolved_workspace.parents:
                return True
            broad_glob = _normalize_pattern(glob or "") in {"*", "**", "**/*"}
            return resolved_target == resolved_workspace and broad_glob and not file_type
        return False

    @staticmethod
    def _checkpoint_note(
        *,
        scanned: int,
        scanned_bytes: int,
        elapsed: float,
        next_cursor: int,
        reason: str,
    ) -> str:
        return (
            "[Search budget checkpoint — review before continuing]\n"
            f"Paused after this segment scanned {scanned} files "
            f"({scanned_bytes} bytes) in {elapsed:.1f}s; reason: {reason}.\n"
            "Decide whether this search is still justified. You may stop, narrow path/glob/type, "
            "or continue the same search by setting "
            f"scan_cursor={next_cursor} and confirm_broad_search=true. "
            "Continuation receives a fresh bounded segment; do not remove filters when retrying."
        )

    @staticmethod
    def _broad_confirmation(path: str) -> str:
        return (
            "[Broad search confirmation required — model decision]\n"
            f"The requested directory '{path or '.'}' is the workspace root (or its parent) and "
            "has no glob/type filter. Decide whether this scope is necessary. Prefer narrowing the "
            "path or adding glob/type; if the full scan is genuinely required, repeat the call with "
            "confirm_broad_search=true. The confirmed search will still pause at bounded checkpoints."
        )
    _IGNORE_DIRS = set(ListDirTool._IGNORE_DIRS)

    @staticmethod
    def _blocked_request_paths(policy_key: str) -> set[str]:
        request_ctx = current_request_context()
        policy = request_ctx.metadata.get(_TOOL_POLICY_METADATA_KEY) if request_ctx else None
        raw_paths = policy.get(policy_key) if isinstance(policy, dict) else None
        if not isinstance(raw_paths, list):
            return set()
        return {str(path).strip().replace("\\", "/").rstrip("/").lower() for path in raw_paths}

    @classmethod
    def _reject_policy_path(cls, path: str, policy_key: str, tool_name: str) -> str | None:
        normalized = path.strip().replace("\\", "/").rstrip("/").lower()
        if policy_key == "blocked_grep_paths" and any(
            normalized == allowed or normalized.startswith(allowed + "/")
            for allowed in cls._blocked_request_paths("allowed_grep_paths")
        ):
            return None
        if any(normalized == blocked or normalized.startswith(blocked + "/") for blocked in cls._blocked_request_paths(policy_key)):
            return (
                f"Error: path '{path or '.'}' is blocked by the request tool_policy for {tool_name}. "
                "Use a changed file path or a specific module/subdirectory."
            )
        return None

    def _display_path(self, target: Path, root: Path) -> str:
        workspace = self._display_workspace()
        if workspace:
            with suppress(ValueError):
                return target.relative_to(workspace).as_posix()
        return target.relative_to(root).as_posix()

    def _iter_files(self, root: Path) -> Iterable[Path]:
        if root.is_file():
            yield root
            return

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in self._IGNORE_DIRS)
            current = Path(dirpath)
            for filename in sorted(filenames):
                yield current / filename


class FindFilesTool(_SearchTool):
    """Find files by path fragment, glob, or type."""
    _scopes = {"core", "subagent"}
    _MAX_SCAN_PATHS = 500_000
    _MAX_SCAN_SECONDS = 30.0

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return (
            "Find workspace paths by name, glob, or file type. "
            "Returns relative paths and skips dependency/build directories."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Search root (default '.')",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive path fragment search. "
                        "Whitespace-separated terms follow query_mode (default: all)."
                    ),
                },
                "query_mode": {
                    "type": "string",
                    "enum": ["all", "any", "phrase"],
                    "description": "How query terms match: all terms, any term, or the exact phrase.",
                },
                "glob": {
                    "type": "string",
                    "description": "Path filter, e.g. '*.py' or 'tests/**/test_*.py'",
                },
                "type": {
                    "type": "string",
                    "description": "File type, e.g. 'py', 'ts', 'md', or 'json'",
                },
                "include_dirs": {
                    "type": "boolean",
                    "description": "Include directories (default false)",
                },
                "sort": {
                    "type": "string",
                    "enum": ["path", "modified"],
                    "description": "Sort order (default path)",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Maximum paths (default 200; 0 for all)",
                    "minimum": 0,
                    "maximum": 1000,
                },
                "offset": {
                    "type": "integer",
                    "description": "Paths to skip before head_limit",
                    "minimum": 0,
                    "maximum": 100000,
                },
                "scan_cursor": {
                    "type": "integer",
                    "description": "Opaque file-walk cursor from a prior search budget checkpoint.",
                    "minimum": 0,
                },
                "confirm_broad_search": {
                    "type": "boolean",
                    "description": "Confirm after the tool asks the model to justify a broad search.",
                },
            },
        }

    def _entry(self, path: Path, root: Path, *, is_dir: bool) -> _FindFilesEntry:
        display_path = self._display_path(path, root)
        return _FindFilesEntry(
            path=path,
            rel_path=path.relative_to(root).as_posix(),
            display_path=display_path,
            name=path.name,
            is_dir=is_dir,
        )

    def _push_directory_entries(
        self,
        directory: Path,
        root: Path,
        frontier: list[tuple[str, int, _FindFilesEntry]],
        sequence: int,
        budget: _FindFilesBudget,
    ) -> int:
        budget.checkpoint()
        try:
            with os.scandir(directory) as entries:
                for raw_entry in entries:
                    budget.visit_path()
                    try:
                        is_dir = raw_entry.is_dir(follow_symlinks=False)
                        # os.walk yields special files and broken file symlinks,
                        # but does not descend into directory symlinks by default.
                        if not is_dir and raw_entry.is_symlink() and raw_entry.is_dir():
                            continue
                    except OSError:
                        continue
                    if is_dir and raw_entry.name in self._IGNORE_DIRS:
                        continue

                    entry = self._entry(Path(raw_entry.path), root, is_dir=is_dir)
                    sort_path = entry.display_path + ("/" if is_dir else "")
                    heapq.heappush(frontier, (sort_path, sequence, entry))
                    sequence += 1
        except OSError:
            # os.walk silently skips directories that cannot be listed. Preserve
            # that behavior while still allowing cancellation and budget errors
            # to propagate from the explicit checkpoints above.
            pass
        return sequence

    def _iter_paths(
        self,
        root: Path,
        *,
        include_dirs: bool,
        budget: _FindFilesBudget,
    ) -> Iterable[_FindFilesEntry]:
        budget.checkpoint()
        if root.is_file():
            budget.visit_path()
            yield self._entry(root, root.parent, is_dir=False)
            return

        if include_dirs:
            yield self._entry(root, root, is_dir=True)

        frontier: list[tuple[str, int, _FindFilesEntry]] = []
        sequence = self._push_directory_entries(root, root, frontier, 0, budget)
        while frontier:
            budget.checkpoint()
            _, _, entry = heapq.heappop(frontier)
            if entry.is_dir:
                if include_dirs:
                    yield entry
                sequence = self._push_directory_entries(
                    entry.path,
                    root,
                    frontier,
                    sequence,
                    budget,
                )
            else:
                yield entry

    @staticmethod
    def _matches_entry(
        entry: _FindFilesEntry,
        *,
        query: str | None,
        glob: str | None,
        file_type: str | None,
        query_mode: str = "all",
    ) -> bool:
        if glob and not _match_glob(entry.rel_path, entry.name, glob):
            return False
        if entry.is_dir:
            if file_type:
                return False
        elif not _matches_type(entry.name, file_type):
            return False
        return _matches_query(entry.display_path, query, query_mode)

    async def execute(
        self,
        path: str = ".",
        query: str | None = None,
        query_mode: str = "all",
        glob: str | None = None,
        type: str | None = None,
        include_dirs: bool = False,
        sort: str = "path",
        head_limit: int | None = None,
        offset: int = 0,
        scan_cursor: int = 0,
        confirm_broad_search: bool = False,
        **kwargs: Any,
    ) -> str:
        cancelled = threading.Event()
        try:
            if rejection := self._reject_policy_path(path, "blocked_find_files_paths", "find_files"):
                return ToolResult.error(rejection)
            return await asyncio.to_thread(
                self._execute_sync,
                path=path,
                query=query,
                query_mode=query_mode,
                scan_cursor=scan_cursor,
                confirm_broad_search=confirm_broad_search,
                glob=glob,
                file_type=type,
                include_dirs=include_dirs,
                sort=sort,
                head_limit=head_limit,
                offset=offset,
                cancelled=cancelled,
            )
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error finding files: {e}")

    def _execute_sync(
        self,
        *,
        path: str,
        query: str | None,
        glob: str | None,
        file_type: str | None,
        include_dirs: bool,
        sort: str,
        head_limit: int | None,
        offset: int,
        cancelled: threading.Event,
        query_mode: str = "all",
        scan_cursor: int = 0,
        confirm_broad_search: bool = False,
    ) -> str:
        started_at = time.monotonic()
        if cancelled.is_set():
            raise _FindFilesCancelledError
        target = self._resolve(path or ".")
        if not target.exists():
            return ToolResult.error(f"Error: Path not found: {path}")
        if not (target.is_dir() or target.is_file()):
            return ToolResult.error(f"Error: Unsupported path: {path}")

        if sort not in {"path", "modified"}:
            return ToolResult.error("Error: sort must be 'path' or 'modified'")
        if query_mode not in {"all", "any", "phrase"}:
            return ToolResult.error("Error: query_mode must be 'all', 'any', or 'phrase'")
        if scan_cursor < 0:
            return ToolResult.error("Error: scan_cursor must be >= 0")
        if self._is_broad_unfiltered(target, glob=glob, file_type=file_type) and not confirm_broad_search:
            return self._broad_confirmation(path)

        limit = (
            _DEFAULT_FILE_HEAD_LIMIT
            if head_limit is None
            else None if head_limit == 0 else head_limit
        )
        budget = _FindFilesBudget(
            cancelled=cancelled,
            deadline=started_at + self._MAX_SCAN_SECONDS,
            max_paths=self._MAX_SCAN_PATHS,
        )

        checkpoint: str | None = None

        def matching_entries() -> Iterator[tuple[str, float]]:
            nonlocal checkpoint
            scanned = 0
            for walk_idx, entry in enumerate(self._iter_paths(
                target,
                include_dirs=include_dirs,
                budget=budget,
            ), start=1):
                if walk_idx <= scan_cursor:
                    continue
                elapsed = time.monotonic() - started_at
                if scanned >= self._SCAN_BUDGET_FILES or elapsed >= self._SCAN_BUDGET_SECONDS:
                    checkpoint = self._checkpoint_note(
                        scanned=scanned, scanned_bytes=0, elapsed=elapsed,
                        next_cursor=walk_idx - 1,
                        reason="file budget" if scanned >= self._SCAN_BUDGET_FILES else "time budget",
                    )
                    break
                scanned += 1
                if not self._matches_entry(
                    entry,
                    query=query,
                    query_mode=query_mode,
                    glob=glob,
                    file_type=file_type,
                ):
                    continue
                mtime = 0.0
                if sort == "modified":
                    try:
                        mtime = entry.path.stat().st_mtime
                    except OSError:
                        pass
                suffix = "/" if entry.is_dir else ""
                yield entry.display_path + suffix, mtime

        matches: list[tuple[str, float]]
        try:
            if sort == "modified":
                if limit is None:
                    matches = sorted(matching_entries(), key=lambda item: (-item[1], item[0]))
                else:
                    selection_size = offset + limit + 1
                    matches = heapq.nsmallest(
                        selection_size,
                        matching_entries(),
                        key=lambda item: (-item[1], item[0]),
                    )
            else:
                selection_size = None if limit is None else offset + limit + 1
                matches = []
                for match in matching_entries():
                    matches.append(match)
                    if selection_size is not None and len(matches) >= selection_size:
                        break
            budget.checkpoint()
        except _FindFilesBudgetExceededError as exc:
            if str(exc) == "paths":
                detail = f"{self._MAX_SCAN_PATHS} paths"
            else:
                detail = f"{self._MAX_SCAN_SECONDS:g} seconds"
            return ToolResult.error(
                f"Error: find_files scan exceeded {detail}; "
                "narrow path, query, glob, or type and retry."
            )

        paths = [item[0] for item in matches]
        paged, truncated = _paginate(paths, limit, offset)
        if not paged:
            result = "No files found"
            if query and query_mode == "all" and len(query.split()) > 1:
                result += "\n\nNote: query_mode='all' requires every term in one path; use query_mode='any' for alternatives."
        else:
            result = "\n".join(paged)
        if checkpoint:
            result += "\n\n" + checkpoint
        note = _pagination_note(limit, offset, truncated)
        if note:
            result += "\n\n" + note
        return result


class GrepTool(_SearchTool):
    """Search text and document contents using a regex-like pattern."""
    _scopes = {"core", "subagent"}

    _MAX_RESULT_CHARS = 128_000
    _MAX_RENDERED_LINE_CHARS = 2_000
    _MAX_FILE_BYTES = 2_000_000
    _MAX_EXPLICIT_FILE_BYTES = 100_000_000

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search text, PDF, DOCX, XLSX, and PPTX content. "
            "Returns matching file paths by default; content mode includes source locators."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex, or literal text when fixed_strings=true",
                    "minLength": 1,
                },
                "path": {
                    "type": "string",
                    "description": "Search root (default '.')",
                },
                "glob": {
                    "type": "string",
                    "description": "Path filter, e.g. '*.py' or 'tests/**/test_*.py'",
                },
                "type": {
                    "type": "string",
                    "description": "File type, e.g. 'py', 'ts', 'md', or 'json'",
                },
                "pages": {
                    "type": "string",
                    "description": "PDF page number or range, e.g. '7' or '101-200' (max 100 pages)",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Ignore case (default false)",
                },
                "fixed_strings": {
                    "type": "boolean",
                    "description": "Treat pattern literally (default false)",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "content: matches with optional context; "
                        "files_with_matches: paths (default); count: matches per file"
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["modified", "path"],
                    "description": (
                        "File result order (default modified). With files_with_matches and sort=path, "
                        "the scan can stop once the requested page is full."
                    ),
                },
                "context_before": {
                    "type": "integer",
                    "description": "Context lines before a match (default 0)",
                    "minimum": 0,
                    "maximum": 20,
                },
                "context_after": {
                    "type": "integer",
                    "description": "Context lines after a match (default 0)",
                    "minimum": 0,
                    "maximum": 20,
                },
                "head_limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to return. In content mode this limits "
                        "matching line blocks; in other modes it limits file entries. "
                        "Default 100"
                    ),
                    "minimum": 0,
                    "maximum": 1000,
                },
                "offset": {
                    "type": "integer",
                    "description": "Matches or file entries to skip before head_limit",
                    "minimum": 0,
                    "maximum": 100000,
                },
                "scan_cursor": {
                    "type": "integer",
                    "description": "Opaque file-walk cursor from a prior search budget checkpoint.",
                    "minimum": 0,
                },
                "confirm_broad_search": {
                    "type": "boolean",
                    "description": "Confirm after the tool asks the model to justify a broad search.",
                },
            },
            "required": ["pattern"],
        }

    @staticmethod
    def _clip_rendered_line(text: str, match_start: int | None = None) -> str:
        limit = GrepTool._MAX_RENDERED_LINE_CHARS
        if len(text) <= limit:
            return text

        marker = "..."
        available = limit - len(marker)
        if match_start is None:
            return text[:available] + marker

        start = max(0, match_start - available // 3)
        start = min(start, len(text) - available)
        end = start + available
        prefix = marker if start else ""
        suffix = marker if end < len(text) else ""
        visible = text[start:end]
        if prefix and suffix:
            visible = visible[: available - len(marker)]
        return prefix + visible + suffix

    @staticmethod
    def _matching_contexts(
        lines: Iterable[LocatedDocumentLine],
        regex: re.Pattern[str],
        before: int,
        after: int,
    ) -> Iterable[tuple[list[LocatedDocumentLine], int, int]]:
        history: deque[LocatedDocumentLine] = deque(maxlen=before)
        pending: list[_PendingContextMatch] = []

        for line in lines:
            if not line.searchable:
                continue

            still_pending: list[_PendingContextMatch] = []
            for item in pending:
                item.lines.append(line)
                item.remaining_after -= 1
                if item.remaining_after == 0:
                    yield item.lines, item.match_index, item.match_start
                else:
                    still_pending.append(item)
            pending = still_pending

            match = regex.search(line.text)
            if match is not None:
                context_lines = [*history, line]
                item = _PendingContextMatch(
                    lines=context_lines,
                    match_index=len(context_lines) - 1,
                    match_start=match.start(),
                    remaining_after=after,
                )
                if after == 0:
                    yield item.lines, item.match_index, item.match_start
                else:
                    pending.append(item)
            history.append(line)

        for item in pending:
            yield item.lines, item.match_index, item.match_start

    @staticmethod
    def _format_block(
        display_path: str,
        lines: list[LocatedDocumentLine],
        match_index: int,
        match_start: int = 0,
    ) -> str:
        match_line = lines[match_index]
        source_line = match_line.extracted_line
        match_locator = match_line.locator
        if match_locator.startswith("sheet="):
            column = _excel_column(match_line.text[:match_start].count("\t") + 1)
            row_match = re.search(r",row=(\d+)$", match_locator)
            if row_match:
                match_locator += f",cell={column}{row_match.group(1)}"
        suffix = f" [{match_locator}]" if match_locator else ""
        block = [f"{display_path}:{source_line}{suffix}"]
        for index, line in enumerate(lines):
            is_match = index == match_index
            marker = ">" if is_match else " "
            coordinate = str(line.extracted_line)
            if line.locator:
                coordinate += f" [{line.locator}]"
            rendered = GrepTool._clip_rendered_line(
                line.text,
                match_start if is_match else None,
            )
            block.append(f"{marker} {coordinate}| {rendered}")
        return "\n".join(block)

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        type: str | None = None,
        pages: str | None = None,
        case_insensitive: bool = False,
        fixed_strings: bool = False,
        output_mode: str = "files_with_matches",
        sort: str = "modified",
        context_before: int = 0,
        context_after: int = 0,
        max_matches: int | None = None,
        max_results: int | None = None,
        head_limit: int | None = None,
        offset: int = 0,
        scan_cursor: int = 0,
        confirm_broad_search: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            if rejection := self._reject_policy_path(path, "blocked_grep_paths", "grep"):
                return ToolResult.error(rejection)
            target = self._resolve(path or ".")
            if not target.exists():
                return ToolResult.error(f"Error: Path not found: {path}")
            if not (target.is_dir() or target.is_file()):
                return ToolResult.error(f"Error: Unsupported path: {path}")
            if scan_cursor < 0:
                return ToolResult.error("Error: scan_cursor must be >= 0")
            if sort not in {"modified", "path"}:
                return ToolResult.error("Error: sort must be 'modified' or 'path'")
            if self._is_broad_unfiltered(target, glob=glob, file_type=type) and not confirm_broad_search:
                return self._broad_confirmation(path)

            flags = re.IGNORECASE if case_insensitive else 0
            try:
                needle = re.escape(pattern) if fixed_strings else pattern
                regex = re.compile(needle, flags)
            except re.error as e:
                return ToolResult.error(f"Error: invalid regex pattern: {e}")

            if head_limit is not None:
                limit = None if head_limit == 0 else head_limit
            elif output_mode == "content" and max_matches is not None:
                limit = max_matches
            elif output_mode != "content" and max_results is not None:
                limit = max_results
            else:
                limit = _DEFAULT_HEAD_LIMIT
            blocks: list[str] = []
            result_chars = 0
            seen_content_matches = 0
            truncated = False
            size_truncated = False
            skipped_binary = 0
            skipped_large = 0
            document_errors: list[str] = []
            document_continuations: list[str] = []
            matching_files: list[str] = []
            counts: dict[str, int] = {}
            file_mtimes: dict[str, float] = {}
            root = target if target.is_dir() else target.parent
            started = time.monotonic()
            scanned = 0
            scanned_bytes = 0
            checkpoint: str | None = None
            max_file_bytes = (
                self._MAX_EXPLICIT_FILE_BYTES if target.is_file() else self._MAX_FILE_BYTES
            )

            for walk_idx, file_path in enumerate(self._iter_files(target), start=1):
                if walk_idx <= scan_cursor:
                    continue
                if walk_idx % 32 == 0:
                    await asyncio.sleep(0)
                elapsed = time.monotonic() - started
                if (
                    scanned >= self._SCAN_BUDGET_FILES
                    or scanned_bytes >= self._SCAN_BUDGET_BYTES
                    or elapsed >= self._SCAN_BUDGET_SECONDS
                ):
                    if scanned >= self._SCAN_BUDGET_FILES:
                        reason = "file budget"
                    elif scanned_bytes >= self._SCAN_BUDGET_BYTES:
                        reason = "byte budget"
                    else:
                        reason = "time budget"
                    checkpoint = self._checkpoint_note(
                        scanned=scanned,
                        scanned_bytes=scanned_bytes,
                        elapsed=elapsed,
                        next_cursor=walk_idx - 1,
                        reason=reason,
                    )
                    break
                scanned += 1
                rel_path = file_path.relative_to(root).as_posix()
                if glob and not _match_glob(rel_path, file_path.name, glob):
                    continue
                if not _matches_type(file_path.name, type):
                    continue
                display_path = self._display_path(file_path, root)

                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    skipped_binary += 1
                    continue
                scanned_bytes += file_size
                if file_size > max_file_bytes:
                    skipped_large += 1
                    continue
                try:
                    mtime = file_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                source_iterator: Iterator[LocatedDocumentLine] | None = None
                is_document = file_path.suffix.lower() in _DOCUMENT_EXTENSIONS
                try:
                    if is_document:
                        source = open_document_line_source(file_path, pages=pages)
                        if source is None:
                            skipped_binary += 1
                            continue
                        source_iterator = source.lines
                        source_lines: Iterable[LocatedDocumentLine] = source_iterator
                        if source.continuation:
                            document_continuations.append(
                                f"({display_path}: continue PDF search with "
                                f"{source.continuation})"
                            )
                    else:
                        with file_path.open("rb") as file:
                            raw = file.read(max_file_bytes + 1)
                        if _is_binary(raw):
                            skipped_binary += 1
                            continue
                        try:
                            content = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            skipped_binary += 1
                            continue
                        source_lines = (
                            LocatedDocumentLine(text, line_no, "")
                            for line_no, text in enumerate(content.splitlines(), 1)
                        )

                    file_had_match = False
                    if output_mode == "content":
                        contexts = self._matching_contexts(
                            source_lines,
                            regex,
                            context_before,
                            context_after,
                        )
                        for context_lines, match_index, match_start in contexts:
                            file_had_match = True
                            seen_content_matches += 1
                            if seen_content_matches <= offset:
                                continue
                            if limit is not None and len(blocks) >= limit:
                                truncated = True
                                break
                            block = self._format_block(
                                display_path,
                                context_lines,
                                match_index,
                                match_start,
                            )
                            extra_sep = 2 if blocks else 0
                            if result_chars + extra_sep + len(block) > self._MAX_RESULT_CHARS:
                                size_truncated = True
                                break
                            blocks.append(block)
                            result_chars += extra_sep + len(block)
                    else:
                        for line in source_lines:
                            if not line.searchable or regex.search(line.text) is None:
                                continue
                            file_had_match = True
                            if output_mode == "count":
                                counts[display_path] = counts.get(display_path, 0) + 1
                                continue
                            if display_path not in matching_files:
                                matching_files.append(display_path)
                                file_mtimes[display_path] = mtime
                            break
                except Exception as e:
                    if not is_document:
                        raise
                    if target.is_file():
                        if isinstance(e, PdfPageRangeError):
                            return ToolResult.error(
                                f"Error: Invalid PDF page range '{pages}': {e!s}."
                            )
                        return ToolResult.error(
                            f"Error searching document {display_path}: {e!s}"
                        )
                    skipped_binary += 1
                    document_errors.append(f"{display_path}: {e!s}")
                    continue
                finally:
                    close = getattr(source_iterator, "close", None)
                    if close is not None:
                        close()
                if output_mode == "count" and file_had_match:
                    if display_path not in matching_files:
                        matching_files.append(display_path)
                        file_mtimes[display_path] = mtime
                if (
                    output_mode == "files_with_matches"
                    and sort == "path"
                    and limit is not None
                    and len(matching_files) >= offset + limit + 1
                ):
                    truncated = True
                    break
                if output_mode in {"count", "files_with_matches"} and file_had_match:
                    continue
                if truncated or size_truncated:
                    break

            if output_mode == "files_with_matches":
                if not matching_files:
                    result = f"No matches found for pattern '{pattern}' in {path}"
                else:
                    ordered_files = sorted(
                        matching_files,
                        key=(
                            (lambda name: name)
                            if sort == "path"
                            else (lambda name: (-file_mtimes.get(name, 0.0), name))
                        ),
                    )
                    paged, truncated = _paginate(ordered_files, limit, offset)
                    result = "\n".join(paged)
            elif output_mode == "count":
                if not counts:
                    result = f"No matches found for pattern '{pattern}' in {path}"
                else:
                    ordered_files = sorted(
                        matching_files,
                        key=lambda name: (-file_mtimes.get(name, 0.0), name),
                    )
                    ordered, truncated = _paginate(ordered_files, limit, offset)
                    count_lines = [f"{name}: {counts[name]}" for name in ordered]
                    result = "\n".join(count_lines)
            else:
                if not blocks:
                    result = f"No matches found for pattern '{pattern}' in {path}"
                else:
                    result = "\n\n".join(blocks)

            notes: list[str] = []
            if output_mode == "content" and truncated:
                notes.append(
                    f"(pagination: limit={limit}, offset={offset}; "
                    f"use offset={offset + len(blocks)} to continue)"
                )
            elif output_mode == "content" and size_truncated:
                notes.append(
                    "(output truncated due to size; "
                    f"use offset={offset + len(blocks)} to continue)"
                )
            elif truncated and output_mode in {"count", "files_with_matches"}:
                notes.append(
                    f"(pagination: limit={limit}, offset={offset})"
                )
            elif output_mode in {"count", "files_with_matches"} and offset > 0:
                notes.append(f"(pagination: offset={offset})")
            elif output_mode == "content" and offset > 0 and blocks:
                notes.append(f"(pagination: offset={offset})")
            if skipped_binary:
                notes.append(f"(skipped {skipped_binary} binary/unreadable files)")
            if skipped_large:
                notes.append(f"(skipped {skipped_large} large files)")
            if document_errors:
                notes.append(f"(first document error: {document_errors[0]})")
            notes.extend(document_continuations[:10])
            if output_mode == "count" and counts:
                notes.append(
                    f"(total matches: {sum(counts.values())} in {len(counts)} files)"
                )
            if notes:
                result += "\n\n" + "\n".join(notes)
            if checkpoint:
                result += "\n\n" + checkpoint
            return result
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error searching files: {e}")
