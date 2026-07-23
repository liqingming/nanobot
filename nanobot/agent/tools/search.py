"""Search tools: file discovery and grep."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, TypeVar

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.filesystem import ListDirTool, _FsTool

_DEFAULT_HEAD_LIMIT = 100
_DEFAULT_FILE_HEAD_LIMIT = 100
_TOOL_POLICY_METADATA_KEY = "tool_policy"
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


def _normalize_pattern(pattern: str) -> str:
    return pattern.strip().replace("\\", "/")


def _match_glob(rel_path: str, name: str, pattern: str) -> bool:
    normalized = _normalize_pattern(pattern)
    if not normalized:
        return False
    if "/" in normalized or normalized.startswith("**"):
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

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return (
            "Find files by path fragment, glob, or file type. "
            "Use this before read_file when you need to locate files, and "
            "prefer it over shell find/ls for ordinary workspace discovery. "
            "Returns workspace-relative paths and skips common dependency/build "
            "directories."
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
                    "description": "Directory or file to search in (default '.')",
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
                    "description": "Optional file filter, e.g. '*.py' or 'tests/**/test_*.py'",
                },
                "type": {
                    "type": "string",
                    "description": "Optional file type shorthand, e.g. 'py', 'ts', 'md', 'json'",
                },
                "include_dirs": {
                    "type": "boolean",
                    "description": "Include matching directories as well as files (default false)",
                },
                "sort": {
                    "type": "string",
                    "enum": ["path", "modified"],
                    "description": "Sort by path or most recently modified first (default path)",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Maximum number of paths to return (default 200, 0 for all, max 1000)",
                    "minimum": 0,
                    "maximum": 1000,
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip the first N results before applying head_limit",
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

    def _iter_paths(self, root: Path, *, include_dirs: bool) -> Iterable[Path]:
        if root.is_file():
            yield root
            return
        if include_dirs:
            yield root
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in self._IGNORE_DIRS)
            current = Path(dirpath)
            if include_dirs and current != root:
                yield current
            for filename in sorted(filenames):
                yield current / filename

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
        try:
            if rejection := self._reject_policy_path(path, "blocked_find_files_paths", "find_files"):
                return ToolResult.error(rejection)
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
            if self._is_broad_unfiltered(target, glob=glob, file_type=type) and not confirm_broad_search:
                return self._broad_confirmation(path)

            limit = (
                _DEFAULT_FILE_HEAD_LIMIT
                if head_limit is None
                else None if head_limit == 0 else head_limit
            )
            root = target if target.is_dir() else target.parent
            matches: list[tuple[str, float]] = []
            started = time.monotonic()
            scanned = 0
            scanned_bytes = 0
            checkpoint: str | None = None

            for idx, candidate in enumerate(self._iter_paths(target, include_dirs=include_dirs), start=1):
                if idx <= scan_cursor:
                    continue
                if idx % 128 == 0:
                    await asyncio.sleep(0)
                elapsed = time.monotonic() - started
                if scanned >= self._SCAN_BUDGET_FILES or elapsed >= self._SCAN_BUDGET_SECONDS:
                    reason = "file budget" if scanned >= self._SCAN_BUDGET_FILES else "time budget"
                    checkpoint = self._checkpoint_note(
                        scanned=scanned,
                        scanned_bytes=scanned_bytes,
                        elapsed=elapsed,
                        next_cursor=idx - 1,
                        reason=reason,
                    )
                    break
                scanned += 1
                if candidate.is_dir() and not include_dirs:
                    continue
                rel_path = candidate.relative_to(root).as_posix()
                display_path = self._display_path(candidate, root)
                name = candidate.name

                if glob and not _match_glob(rel_path, name, glob):
                    continue
                if candidate.is_file() and not _matches_type(name, type):
                    continue
                if candidate.is_dir() and type:
                    continue
                if not _matches_query(display_path, query, query_mode):
                    continue
                try:
                    stat = candidate.stat()
                    mtime = stat.st_mtime
                    if candidate.is_file():
                        scanned_bytes += stat.st_size
                except OSError:
                    mtime = 0.0
                suffix = "/" if candidate.is_dir() else ""
                matches.append((display_path + suffix, mtime))

            if sort == "modified":
                matches.sort(key=lambda item: (-item[1], item[0]))
            else:
                matches.sort(key=lambda item: item[0])

            paths = [item[0] for item in matches]
            paged, truncated = _paginate(paths, limit, offset)
            if not paged:
                result = "No files found"
                if query and query_mode == "all" and len(query.split()) > 1:
                    result += (
                        "\n\nNote: query_mode='all' requires one path to contain every whitespace-separated "
                        "term. If the terms are alternatives, retry with query_mode='any' or separate calls."
                    )
            else:
                result = "\n".join(paged)
            note = _pagination_note(limit, offset, truncated)
            if note:
                result += "\n\n" + note
            if checkpoint:
                result += "\n\n" + checkpoint
            return result
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error finding files: {e}")


class GrepTool(_SearchTool):
    """Search file contents using a regex-like pattern."""
    _scopes = {"core", "subagent"}

    _MAX_RESULT_CHARS = 128_000
    _MAX_FILE_BYTES = 2_000_000

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with a regex pattern. "
            "Default output_mode is files_with_matches (file paths only); "
            "use content mode for matching lines with context. Prefer this "
            "over shell grep for ordinary workspace searches. "
            "Skips binary and files >2 MB. Supports glob/type filtering."
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
                    "description": "Regex or plain text pattern to search for",
                    "minLength": 1,
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default '.')",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional file filter, e.g. '*.py' or 'tests/**/test_*.py'",
                },
                "type": {
                    "type": "string",
                    "description": "Optional file type shorthand, e.g. 'py', 'ts', 'md', 'json'",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default false)",
                },
                "fixed_strings": {
                    "type": "boolean",
                    "description": "Treat pattern as plain text instead of regex (default false)",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "content: matching lines with optional context; "
                        "files_with_matches: only matching file paths; "
                        "count: matching line counts per file. "
                        "Default: files_with_matches"
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
                    "description": "Number of lines of context before each match",
                    "minimum": 0,
                    "maximum": 20,
                },
                "context_after": {
                    "type": "integer",
                    "description": "Number of lines of context after each match",
                    "minimum": 0,
                    "maximum": 20,
                },
                "max_matches": {
                    "type": "integer",
                    "description": (
                        "Legacy alias for head_limit in content mode"
                    ),
                    "minimum": 1,
                    "maximum": 1000,
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Legacy alias for head_limit in files_with_matches or count mode"
                    ),
                    "minimum": 1,
                    "maximum": 1000,
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
                    "description": "Skip the first N results before applying head_limit",
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
    def _format_block(
        display_path: str,
        lines: list[str],
        match_line: int,
        before: int,
        after: int,
    ) -> str:
        start = max(1, match_line - before)
        end = min(len(lines), match_line + after)
        block = [f"{display_path}:{match_line}"]
        for line_no in range(start, end + 1):
            marker = ">" if line_no == match_line else " "
            block.append(f"{marker} {line_no}| {lines[line_no - 1]}")
        return "\n".join(block)

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        type: str | None = None,
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
            matching_files: list[str] = []
            counts: dict[str, int] = {}
            file_mtimes: dict[str, float] = {}
            root = target if target.is_dir() else target.parent
            started = time.monotonic()
            scanned = 0
            scanned_bytes = 0
            checkpoint: str | None = None

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

                raw = file_path.read_bytes()
                scanned_bytes += len(raw)
                if len(raw) > self._MAX_FILE_BYTES:
                    skipped_large += 1
                    continue
                if _is_binary(raw):
                    skipped_binary += 1
                    continue
                try:
                    mtime = file_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    skipped_binary += 1
                    continue

                lines = content.splitlines()
                display_path = self._display_path(file_path, root)
                file_had_match = False
                for idx, line in enumerate(lines, start=1):
                    if not regex.search(line):
                        continue
                    file_had_match = True

                    if output_mode == "count":
                        counts[display_path] = counts.get(display_path, 0) + 1
                        continue
                    if output_mode == "files_with_matches":
                        if display_path not in matching_files:
                            matching_files.append(display_path)
                            file_mtimes[display_path] = mtime
                        break

                    seen_content_matches += 1
                    if seen_content_matches <= offset:
                        continue
                    if limit is not None and len(blocks) >= limit:
                        truncated = True
                        break
                    block = self._format_block(
                        display_path,
                        lines,
                        idx,
                        context_before,
                        context_after,
                    )
                    extra_sep = 2 if blocks else 0
                    if result_chars + extra_sep + len(block) > self._MAX_RESULT_CHARS:
                        size_truncated = True
                        break
                    blocks.append(block)
                    result_chars += extra_sep + len(block)
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
                    lines = [f"{name}: {counts[name]}" for name in ordered]
                    result = "\n".join(lines)
            else:
                if not blocks:
                    result = f"No matches found for pattern '{pattern}' in {path}"
                else:
                    result = "\n\n".join(blocks)

            notes: list[str] = []
            if output_mode == "content" and truncated:
                notes.append(
                    f"(pagination: limit={limit}, offset={offset})"
                )
            elif output_mode == "content" and size_truncated:
                notes.append("(output truncated due to size)")
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
