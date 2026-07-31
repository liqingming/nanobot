from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nanobot.agent.skills import SkillsLoader

SCRIPT = Path(
    "nanobot/fork/builtin_skills/scan-tool-corrections/scripts/scan_session_errors.py"
).resolve()


def _load_scanner():
    spec = importlib.util.spec_from_file_location("scan_tool_corrections", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fork_builtin_skill_is_registered(tmp_path) -> None:
    loader = SkillsLoader(tmp_path)

    entries = {entry["name"]: entry for entry in loader.list_skills(filter_unavailable=False)}

    assert entries["scan-tool-corrections"]["source"] == "fork-builtin"
    skill = loader.load_skill("scan-tool-corrections")
    assert skill is not None
    assert "Require exactly two user inputs" in skill
    assert "Never ask the user for a cache directory" in skill


def _write_topic(data_dir: Path, *, key: str, title: str, records: list[dict[str, object]]) -> None:
    sessions = data_dir / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    metadata = {
        "_type": "metadata",
        "key": key,
        "updated_at": "2026-01-01T00:00:03",
        "metadata": {"cli_title": title},
    }
    (sessions / f"topic-{len(list(sessions.glob('*.jsonl')))}.jsonl").write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    runtime_dir = sessions / key.replace(":", "_")
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "runtime.log").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_scanner_resolves_work_directory_and_topic_then_redacts(tmp_path, monkeypatch) -> None:
    scanner = _load_scanner()
    fake_home = tmp_path / "home"
    monkeypatch.setattr(scanner.Path, "home", lambda: fake_home)
    work_directory = tmp_path / "work" / "aiHome"
    work_directory.mkdir(parents=True)
    data_dir = scanner._runtime_data_dir(work_directory)
    _write_topic(data_dir, key="cli:session-other", title="其他话题", records=[])
    _write_topic(
        data_dir,
        key="cli:session-target",
        title="服务优化",
        records=[
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

    runtime_log, session = scanner._resolve_topic(work_directory, " 服务优化 ")
    failures = scanner.scan(runtime_log)

    assert session["key"] == "cli:session-target"
    assert runtime_log == data_dir / "sessions" / "cli_session-target" / "runtime.log"
    assert len(failures) == 1
    assert "should-not-leak" not in failures[0]["summary"]
    assert "<redacted>" in failures[0]["summary"]


def test_scanner_does_not_guess_unknown_or_duplicate_topic(tmp_path, monkeypatch) -> None:
    scanner = _load_scanner()
    fake_home = tmp_path / "home"
    monkeypatch.setattr(scanner.Path, "home", lambda: fake_home)
    work_directory = tmp_path / "workspace"
    work_directory.mkdir()
    data_dir = scanner._runtime_data_dir(work_directory)
    _write_topic(data_dir, key="cli:first", title="重名", records=[])
    _write_topic(data_dir, key="cli:second", title="重名", records=[])

    with pytest.raises(scanner.TopicResolutionError) as duplicate:
        scanner._resolve_topic(work_directory, "重名")
    with pytest.raises(scanner.TopicResolutionError) as missing:
        scanner._resolve_topic(work_directory, "不存在")

    assert duplicate.value.payload["error"] == "topic_not_unique"
    assert len(duplicate.value.payload["candidates"]) == 2
    assert missing.value.payload["error"] == "topic_not_found"
