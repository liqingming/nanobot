"""Per-tool result summarizers for the live tool-trace UI.

After each tool runs, the TUI displays a one-line trace like::

    ↳ exec("pip install …")  →  exit 0, 12 lines

The hint portion ("exec(…)") is built by ``AgentLoop._tool_hint`` from the
tool call. The "→ result" suffix is produced by each Tool's own
:meth:`Tool.summarize_result` method (default: empty / no summary).

This module exposes a thin dispatcher plus shared text-formatting helpers
that individual ``summarize_result`` implementations use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.agent.tools.base import Tool


# ── Shared helpers used by Tool.summarize_result implementations ──────────


def line_count(text: str) -> int:
    """Count lines in a text string (no trailing-blank fudge)."""
    if not text:
        return 0
    return text.count("\n") + (1 if not text.endswith("\n") else 0)


def truncate(text: str, width: int = 60) -> str:
    """Collapse newlines and truncate with ellipsis."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def extract_error_summary(result: str, width: int = 120) -> str:
    """Extract the first useful error line from an "Error: ..." result.

    Strips redundant "Error: " / "Error " / "error " prefixes so the user
    sees the actual cause, not the marker. Default width is generous (120)
    so errors stay readable — users can immediately see what failed without
    needing to dig into raw logs.
    """
    if not isinstance(result, str):
        return ""
    # First non-empty line
    first = next((ln for ln in result.splitlines() if ln.strip()), result)
    # Drop the redundant "Error:" prefix when we already render it in red
    for prefix in ("Error: ", "Error - ", "Error "):
        if first.startswith(prefix):
            first = first[len(prefix):]
            break
    return "Error: " + truncate(first, width)


def summarize_error_or(result: str, ok_summary: str) -> str:
    """Pattern: if result is an error, summarize it; otherwise return ok_summary."""
    if isinstance(result, str) and result.startswith("Error"):
        return extract_error_summary(result)
    return ok_summary


# ── Dispatcher used by the agent loop ─────────────────────────────────────


def summarize_tool_result(tool: "Tool | None", args: Any, result: Any) -> str:
    """Return tool's own short summary, or '' if it opts out / errors.

    Best-effort: never raises (UI shouldn't break because of a buggy
    summarizer). Tools opt in by overriding ``Tool.summarize_result``.
    """
    if tool is None:
        return ""
    if not isinstance(args, dict):
        args = {}
    try:
        return tool.summarize_result(args, result) or ""
    except Exception:
        return ""
