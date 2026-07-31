#!/usr/bin/env python3
"""Resolve one nanobot topic and extract sanitized tool failures."""

from __future__ import annotations

import argparse
import hashlib
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
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def _runtime_data_dir(work_directory: Path) -> Path:
    workspace = work_directory.expanduser().resolve()
    default_workspace = (Path.home() / ".nanobot" / "workspace").resolve(strict=False)
    if workspace == default_workspace:
        return workspace
    digest = hashlib.sha1(str(workspace).encode()).hexdigest()[:8]
    safe_name = re.sub(r"[^\w-]", "_", workspace.name) or "root"
    return Path.home() / ".nanobot" / "caches" / f"{safe_name}_{digest}"


def _safe_session_key(key: str) -> str:
    return _UNSAFE_CHARS.sub("_", key.replace(":", "_")).strip()


def _topic_sessions(data_dir: Path) -> list[dict[str, str]]:
    sessions_dir = data_dir / "sessions"
    rows: list[dict[str, str]] = []
    if not sessions_dir.is_dir():
        return rows
    for path in sorted(sessions_dir.glob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.loads(stream.readline())
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        metadata = record.get("metadata") if isinstance(record, dict) else None
        title = metadata.get("cli_title") if isinstance(metadata, dict) else None
        key = record.get("key") if isinstance(record, dict) else None
        if (
            record.get("_type") != "metadata"
            or not isinstance(title, str)
            or not isinstance(key, str)
        ):
            continue
        rows.append(
            {
                "title": title.strip(),
                "key": key,
                "updated_at": str(record.get("updated_at") or ""),
            }
        )
    return rows


def _resolve_topic(work_directory: Path, topic: str) -> tuple[Path, dict[str, str]]:
    data_dir = _runtime_data_dir(work_directory)
    sessions = _topic_sessions(data_dir)
    wanted = topic.strip()
    matches = [session for session in sessions if session["title"] == wanted]
    if len(matches) != 1:
        payload = {
            "error": "topic_not_found" if not matches else "topic_not_unique",
            "work_directory": str(work_directory.expanduser().resolve()),
            "data_directory": str(data_dir),
            "topic": wanted,
            "candidates": matches or sessions,
        }
        raise TopicResolutionError(payload)
    session = matches[0]
    runtime_log = data_dir / "sessions" / _safe_session_key(session["key"]) / "runtime.log"
    if not runtime_log.is_file():
        raise TopicResolutionError(
            {
                "error": "runtime_log_not_found",
                "work_directory": str(work_directory.expanduser().resolve()),
                "data_directory": str(data_dir),
                "topic": wanted,
                "session_key": session["key"],
                "runtime_log": str(runtime_log),
            }
        )
    return runtime_log, session


class TopicResolutionError(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload.get("error") or "topic_resolution_error"))
        self.payload = payload


def _safe_text(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else ""
    return _SECRET.sub(r"\1 <redacted>", text)[:limit]


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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--topic", required=True)
    args = parser.parse_args()

    work_directory = Path(args.directory)
    try:
        runtime_log, session = _resolve_topic(work_directory, args.topic)
    except TopicResolutionError as exc:
        print(json.dumps(exc.payload, ensure_ascii=False, indent=2))
        return 2

    print(
        json.dumps(
            {
                "work_directory": str(work_directory.expanduser().resolve()),
                "topic": session["title"],
                "session_key": session["key"],
                "runtime_log": str(runtime_log),
                "failures": scan(runtime_log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
