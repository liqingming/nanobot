#!/usr/bin/env python3
"""Extract sanitized tool failures from all sessions in a nanobot work directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_ERROR_EVENTS = {"runner.tool.prepare_error", "runner.tool.error_result"}
_OUTPUT_ERROR = re.compile(
    r"(?:Exit code:\s*[1-9]\d*|SyntaxError:|Traceback \(most recent call last\)|timed out)",
    re.IGNORECASE,
)
_SECRET = re.compile(
    r"(?i)((?:api[-_]?key|authorization|token|password|secret|cookie)\s*[:=])\s*\S+"
)


def _session_directories(directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    if (directory / "runtime.log").is_file():
        return [directory]
    sessions_root = directory if directory.name == "sessions" else directory / "sessions"
    if not sessions_root.is_dir():
        return []
    return sorted(
        (
            candidate.resolve()
            for candidate in sessions_root.iterdir()
            if candidate.is_dir() and (candidate / "runtime.log").is_file()
        ),
        key=lambda path: path.name,
    )


def _safe_text(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else ""
    text = _SECRET.sub(r"\1 <redacted>", text)
    return text[:limit]


def _kind(record: dict[str, Any]) -> str | None:
    event = record.get("event")
    if event in _ERROR_EVENTS:
        return "tool_error"
    if event != "runner.tool.audit.end":
        return None
    detail = str(record.get("detail") or record.get("result_preview") or "")
    if re.search(r"Exit code:\s*[1-9]\d*", detail, re.IGNORECASE):
        return "nonzero_exit"
    if record.get("tool") == "exec" and _OUTPUT_ERROR.search(detail):
        return "diagnostic_error"
    return "tool_error" if record.get("status") == "error" else None


def scan(session_dir: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    with (session_dir / "runtime.log").open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict) or not (kind := _kind(record)):
                continue
            call_id = str(record.get("call_id") or "")
            dedupe_key = call_id or f"{line_number}:{record.get('event')}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            failures.append(
                {
                    "session": session_dir.name,
                    "timestamp": record.get("ts"),
                    "line": line_number,
                    "kind": kind,
                    "tool": record.get("tool"),
                    "event": record.get("event"),
                    "call_id": call_id or None,
                    "error_type": record.get("error_type"),
                    "summary": _safe_text(
                        record.get("error")
                        or record.get("result")
                        or record.get("detail")
                        or record.get("result_preview")
                    ),
                }
            )
    return failures


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    sessions = _session_directories(directory)
    if not sessions:
        print(
            json.dumps(
                {"error": "runtime_logs_not_found", "directory": str(directory)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    failures = [failure for session in sessions for failure in scan(session)]
    failures.sort(
        key=lambda item: (str(item.get("timestamp") or ""), item["session"], item["line"])
    )
    for sequence, failure in enumerate(failures, start=1):
        failure["sequence"] = sequence

    print(
        json.dumps(
            {
                "directory": str(directory),
                "sessions_scanned": len(sessions),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
