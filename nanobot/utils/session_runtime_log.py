"""Small per-session runtime log writer."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


_MAX_FIELD_CHARS = 2000


def compact_value(value: Any, *, max_chars: int = _MAX_FIELD_CHARS) -> Any:
    if isinstance(value, str):
        text = value.replace("\r\n", "\n")
        if len(text) > max_chars:
            return text[:max_chars] + "...[truncated]"
        return text
    if isinstance(value, dict):
        return {
            str(k): compact_value(v, max_chars=max_chars)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [compact_value(item, max_chars=max_chars) for item in value[:20]]
    if isinstance(value, tuple):
        return [compact_value(item, max_chars=max_chars) for item in value[:20]]
    return value



_FULL_TEXT_FIELDS = frozenset({"final_error", "error_content", "error_detail", "traceback"})


def compact_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in fields.items():
        if key in _FULL_TEXT_FIELDS:
            compacted[key] = compact_value(value, max_chars=20000)
        else:
            compacted[key] = compact_value(value)
    return compacted


def exception_fields(exc: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def append_session_runtime_log(path: Path | None, event_name: str, **fields: Any) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event_name,
            **compact_log_fields(fields),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return
