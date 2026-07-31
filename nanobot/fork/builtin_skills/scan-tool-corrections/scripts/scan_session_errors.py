#!/usr/bin/env python3
"""Extract sanitized tool failures from one nanobot runtime session."""

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


def _session_candidates(directory: Path, session: str) -> list[Path]:
    directory = directory.expanduser().resolve()
    if (directory / "runtime.log").is_file() and session in {".", directory.name}:
        return [directory]

    roots = [directory, directory / "sessions"]
    names = {session}
    if session.startswith("cli:"):
        names.add(session.replace(":", "_", 1))
    if session.startswith("session_"):
        names.add(f"cli_{session}")

    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidate = root / name
            if (candidate / "runtime.log").is_file():
                candidates.append(candidate.resolve())
        if root.is_dir():
            for candidate in root.iterdir():
                if (
                    candidate.is_dir()
                    and session.casefold() in candidate.name.casefold()
                    and (candidate / "runtime.log").is_file()
                ):
                    candidates.append(candidate.resolve())
    return list(dict.fromkeys(candidates))


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


def scan(log_path: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    with log_path.open(encoding="utf-8", errors="replace") as stream:
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
                    "sequence": len(failures) + 1,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    candidates = _session_candidates(Path(args.directory), args.session)
    if len(candidates) != 1:
        payload = {
            "error": "session_not_unique" if candidates else "session_not_found",
            "candidates": [str(path) for path in candidates],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    log_path = candidates[0] / "runtime.log"
    print(
        json.dumps(
            {"session_dir": str(candidates[0]), "failures": scan(log_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
