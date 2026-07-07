from __future__ import annotations

import json
from pathlib import Path

from nanobot.session.manager import SessionManager
from nanobot.utils.session_runtime_log import append_session_runtime_log


def test_session_runtime_log_path_is_topic_scoped(tmp_path: Path) -> None:
    manager = SessionManager(workspace=tmp_path)

    path = manager.get_session_runtime_log_path("cli:topic_20260707")

    assert path == tmp_path / "sessions" / "cli_topic_20260707" / "runtime.log"
    assert path.parent.exists()


def test_append_session_runtime_log_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "sessions" / "cli_topic" / "runtime.log"

    append_session_runtime_log(
        log_path,
        "turn.state.end",
        turn_id="cli:topic:1",
        transition_event="ok",
        long_text="x" * 3000,
    )

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["event"] == "turn.state.end"
    assert record["turn_id"] == "cli:topic:1"
    assert record["transition_event"] == "ok"
    assert record["long_text"].endswith("[truncated]")
