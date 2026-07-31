from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nanobot.agent.skills import SkillsLoader

SCRIPT = Path(
    "nanobot/fork/builtin_skills/scan-tool-corrections/scripts/scan_session_errors.py"
).resolve()


def test_fork_builtin_skill_is_registered(tmp_path) -> None:
    loader = SkillsLoader(tmp_path)

    entries = {entry["name"]: entry for entry in loader.list_skills(filter_unavailable=False)}

    assert entries["scan-tool-corrections"]["source"] == "fork-builtin"
    assert "Require exactly one input" in loader.load_skill("scan-tool-corrections")


def _write_log(session: Path, records: list[dict[str, object]]) -> None:
    session.mkdir(parents=True)
    (session / "runtime.log").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_scanner_scans_all_sessions_orders_deduplicates_and_redacts(tmp_path) -> None:
    _write_log(
        tmp_path / "sessions" / "cli_session_later",
        [
            {
                "ts": "2026-01-01T00:00:02",
                "event": "runner.tool.audit.end",
                "tool": "exec",
                "call_id": "call-2",
                "status": "ok",
                "detail": "Exit code: 1\nSyntaxError: bad quoting",
            }
        ],
    )
    _write_log(
        tmp_path / "sessions" / "cli_session_earlier",
        [
            {
                "ts": "2026-01-01T00:00:01",
                "event": "runner.tool.error_result",
                "tool": "exec",
                "call_id": "call-1",
                "result": "token=should-not-leak command timed out",
            },
            {
                "ts": "2026-01-01T00:00:01",
                "event": "runner.tool.audit.end",
                "tool": "exec",
                "call_id": "call-1",
                "status": "error",
            },
        ],
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--directory", str(tmp_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    failures = payload["failures"]
    assert payload["sessions_scanned"] == 2
    assert [failure["sequence"] for failure in failures] == [1, 2]
    assert [failure["session"] for failure in failures] == [
        "cli_session_earlier",
        "cli_session_later",
    ]
    assert [failure["kind"] for failure in failures] == ["tool_error", "nonzero_exit"]
    assert "should-not-leak" not in failures[0]["summary"]
    assert "<redacted>" in failures[0]["summary"]
