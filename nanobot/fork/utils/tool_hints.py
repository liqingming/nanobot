"""Fork tool hint formatter — workspace-relative path display.

Adds workspace-relative path display on top of upstream
:func:`nanobot.utils.tool_hints.format_tool_hints` so trace lines show
``write_file("./src/main.py")`` instead of the long absolute path.

Wires in via ``AgentProgressHook(formatter=...)`` from ``AgentLoop`` when
fork features are enabled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Argument keys that identify *what* a tool is acting on. Listed in priority
# order: when a tool call has multiple args, the hint shows the one most
# useful for the user (e.g. for write_file we want the path, not the file
# content). Fallback: first string value in the dict.
_HINT_KEY_PRIORITY: tuple[str, ...] = (
    "path", "file_path", "filepath", "file",
    "url", "command", "cmd",
    "query", "q",
    "symbol", "name", "topic", "content",
)
_HINT_PATH_KEYS: tuple[str, ...] = ("path", "file_path", "filepath", "file")


def relativize_path(value: str, workspace: Any) -> str:
    """If ``value`` is an absolute path inside ``workspace``, return its
    workspace-relative form prefixed with ``./``; otherwise return unchanged.

    Best-effort — any exception falls back to the original string.
    """
    if not value or workspace is None:
        return value
    try:
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
        # Normalize Windows backslashes to forward slashes so the trace
        # display stays clean across platforms.
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


def format_tool_hint(
    tool_calls: list,
    *,
    workspace: Any = None,
    max_length: int = 40,
) -> str:
    """Build a concise "tool(arg)" hint string for a list of tool calls.

    Picks the most identifying argument (path/url/query/...) per call using
    ``_HINT_KEY_PRIORITY``. Path-like arguments are shortened to a
    workspace-relative form when applicable.

    Signature compatible with :func:`nanobot.utils.tool_hints.format_tool_hints`
    so it can be injected as a drop-in via ``AgentProgressHook(formatter=...)``.
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
        return f'{tc.name}("{_smart_truncate(val, max_length)}")'

    return ", ".join(_fmt(tc) for tc in tool_calls)
