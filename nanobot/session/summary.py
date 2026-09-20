"""Helpers for validated session-summary metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict, cast

from nanobot.session.history_visibility import is_hidden_history_message

SUMMARY_CONTINUATION_TEXT = (
    "Continue the active task from the working-memory checkpoint above."
)

def is_summary_checkpoint(message: Mapping[str, Any]) -> bool:
    """Identify the durable boundary of a replacement summary."""
    return (
        is_hidden_history_message(message)
        and message.get("content") == SUMMARY_CONTINUATION_TEXT
    )


class SessionSummary(TypedDict):
    text: str
    last_active: str


@dataclass(frozen=True, slots=True)
class SessionSummaryCheckpoint:
    """A replacement summary and the raw transcript boundary it covers."""

    summary: str
    transcript_boundary: int


def session_summary_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    fallback_last_active: datetime,
) -> SessionSummary | None:
    summary_data: Mapping[str, object] | None = None
    text: str | None = None
    for key in ("_continuation_summary", "_last_summary"):
        raw = metadata.get(key) if metadata is not None else None
        if not isinstance(raw, Mapping):
            continue
        candidate = cast(Mapping[str, object], raw)
        candidate_text = candidate.get("text")
        if isinstance(candidate_text, str) and candidate_text:
            summary_data = candidate
            text = candidate_text
            break
    if summary_data is None or text is None:
        return None
    raw_last_active = summary_data.get("last_active")
    if isinstance(raw_last_active, str):
        try:
            datetime.fromisoformat(raw_last_active)
            last_active = raw_last_active
        except ValueError:
            last_active = fallback_last_active.isoformat()
    else:
        last_active = fallback_last_active.isoformat()
    return {"text": text, "last_active": last_active}
