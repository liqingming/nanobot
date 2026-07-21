"""Fork tool hint formatter — workspace-relative path display.

Adds workspace-relative path display on top of upstream
:func:`nanobot.utils.tool_hints.format_tool_hints` so trace lines show
``write_file("./src/main.py")`` instead of the long absolute path.

Wires in via ``AgentProgressHook(formatter=...)`` from ``AgentLoop`` when
fork features are enabled.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Argument keys that identify *what* a tool is acting on. Listed in priority
# order: when a tool call has multiple args, the hint shows the one most
# useful for the user (e.g. for write_file we want the path, not the file
# content). Fallback: first string value in the dict.
_HINT_KEY_PRIORITY: tuple[str, ...] = (
    "path", "file_path", "filepath", "file",
    "old_text", "old_string",
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


_POWERSHELL_HERE_PYTHON_RE = re.compile(
    r"^\s*@(?P<quote>['\"])\s*\r?\n.*?\r?\n(?P=quote)@\s*\|\s*(?:py|python)(?:\.exe)?\b",
    re.IGNORECASE | re.DOTALL,
)


def _tool_name(tool_call: Any) -> str | None:
    name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
    return name if isinstance(name, str) and name else None


def _arguments(tool_call: Any) -> dict[str, Any]:
    args = (
        tool_call.get("arguments")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "arguments", None)
    )
    if isinstance(args, list):
        args = args[0] if args else {}
    return args if isinstance(args, dict) else {}


def _semantic_exec_hint(command: str, max_length: int) -> str:
    """Hide shell transport syntax while keeping ordinary commands recognizable."""
    command = command.strip()
    if _POWERSHELL_HERE_PYTHON_RE.match(command):
        return "运行 Python 检查脚本"
    return f'exec("{_smart_truncate(" ".join(command.split()), max_length)}")'


def format_tool_items(
    tool_calls: list,
    *,
    workspace: Any = None,
    max_length: int = 40,
) -> list[str]:
    """Return readable per-tool labels, aggregating identical targets globally."""
    labels: list[str] = []
    for tool_call in tool_calls:
        name = _tool_name(tool_call)
        if name is None:
            continue
        args = _arguments(tool_call)
        if name == "exec":
            command = args.get("command") or args.get("cmd")
            label = (
                _semantic_exec_hint(command, max_length)
                if isinstance(command, str) and command.strip()
                else "exec"
            )
        else:
            value: str | None = None
            chosen_key: str | None = None
            for key in _HINT_KEY_PRIORITY:
                candidate = args.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    value = candidate
                    chosen_key = key
                    break
            if value is None:
                value = next(
                    (
                        candidate
                        for candidate in args.values()
                        if isinstance(candidate, str) and candidate.strip()
                    ),
                    None,
                )
            if value is None:
                label = name
            else:
                if chosen_key in _HINT_PATH_KEYS:
                    value = relativize_path(value, workspace).replace("\\", "/")
                label = f'{name}("{_smart_truncate(value, max_length)}")'
        labels.append(label.replace("\r", " ").replace("\n", " "))

    counts: OrderedDict[str, int] = OrderedDict()
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return [f"{label} × {count}" if count > 1 else label for label, count in counts.items()]


def format_tool_hint(
    tool_calls: list,
    *,
    workspace: Any = None,
    max_length: int = 40,
) -> str:
    """Build the legacy one-line hint from the shared readable item formatter."""
    return ", ".join(
        format_tool_items(tool_calls, workspace=workspace, max_length=max_length)
    )


def format_tool_event_items(
    tool_events: list[dict[str, Any]],
    *,
    workspace: Any = None,
    max_length: int = 40,
) -> list[dict[str, Any]]:
    """Format structured progress events and aggregate equivalent outcomes."""
    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for event in tool_events:
        if not isinstance(event, dict):
            continue
        labels = format_tool_items([event], workspace=workspace, max_length=max_length)
        if not labels:
            continue
        label = labels[0]
        error = str(event.get("error") or "").strip()
        status = "error" if event.get("phase") == "error" or error else "ok"
        key = (label, status)
        if key not in grouped:
            grouped[key] = {"label": label, "status": status, "error": error, "count": 1}
        else:
            grouped[key]["count"] += 1
            if not grouped[key]["error"] and error:
                grouped[key]["error"] = error
    items = list(grouped.values())
    for item in items:
        if item["count"] > 1:
            item["label"] = f'{item["label"]} × {item["count"]}'
    return items
