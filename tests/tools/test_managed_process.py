from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.managed_process import (
    ManagedProcessManager,
    StartProcessTool,
)
from nanobot.agent.tools.shell import ExecTool


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class _FakePopen:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(70000 + len(self.processes))
        self.calls.append((args, kwargs))
        self.processes.append(process)
        return process


def _manager(
    tmp_path: Path, platform_name: str = "POSIX"
) -> tuple[ManagedProcessManager, _FakePopen]:
    popen = _FakePopen()
    manager = ManagedProcessManager(
        root=tmp_path,
        popen=popen,
        run=lambda *args, **kwargs: None,
        platform_name=platform_name,
        start_watcher=False,
    )
    return manager, popen


def _start(
    manager: ManagedProcessManager,
    tmp_path: Path,
    *,
    lifecycle: str = "service",
    restart_policy: str = "on-failure",
    owner: str | None = "session-a",
    health_url: str | None = None,
) -> dict[str, Any]:
    return manager.start(
        command="serve",
        cwd=str(tmp_path),
        env={"PATH": "/bin"},
        shell_program="/bin/bash",
        login=False,
        lifecycle=lifecycle,
        restart_policy=restart_policy,
        name=None,
        owner_session_key=owner,
        health_url=health_url,
        health_grace_s=0,
        max_restarts=3,
        max_log_bytes=1024 * 1024,
    )


def test_tools_are_auto_discovered() -> None:
    names = {tool.__name__ for tool in ToolLoader().discover()}

    assert {"StartProcessTool", "ProcessControlTool"} <= names


def test_service_is_detached_persisted_and_hides_environment(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path)

    record = _start(manager, tmp_path)
    listed = manager.list("session-a")

    assert record["pid"] == popen.processes[0].pid
    assert popen.calls[0][1]["start_new_session"] is True
    assert listed[0]["lifecycle"] == "service"
    assert "env" not in record
    assert "env" not in listed[0]
    assert manager.list("another-session") == []
    assert (tmp_path / "state.json").is_file()


def test_on_failure_does_not_restart_clean_exit(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path)
    _start(manager, tmp_path)
    popen.processes[0].returncode = 0

    first = manager.list()[0]
    second = manager.list()[0]

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert first["desired"] == "stopped"
    assert len(popen.processes) == 1


def test_on_failure_restarts_failed_service(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path)
    _start(manager, tmp_path)
    popen.processes[0].returncode = 7

    record = manager.list()[0]

    assert record["status"] == "running"
    assert record["restart_count"] == 1
    assert len(popen.processes) == 2


def test_always_restarts_clean_exit(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path)
    _start(manager, tmp_path, restart_policy="always")
    popen.processes[0].returncode = 0

    record = manager.list()[0]

    assert record["status"] == "running"
    assert len(popen.processes) == 2


def test_shutdown_stops_tasks_but_leaves_services_running(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path, platform_name="Windows")
    task = _start(
        manager,
        tmp_path,
        lifecycle="task",
        restart_policy="never",
    )
    service = _start(manager, tmp_path)

    manager.shutdown_tasks()
    records = {item["id"]: item for item in manager.list()}

    assert records[task["id"]]["status"] == "stopped"
    assert popen.processes[0].poll() == -15
    assert records[service["id"]]["status"] == "running"
    assert popen.processes[1].poll() is None


def test_shutdown_without_running_tasks_does_not_save(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    _start(manager, tmp_path)
    saves = 0

    def count_save() -> None:
        nonlocal saves
        saves += 1

    manager._save_locked = count_save

    manager.shutdown_tasks()

    assert saves == 0


def test_close_is_idempotent(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path, platform_name="Windows")
    _start(
        manager,
        tmp_path,
        lifecycle="task",
        restart_policy="never",
    )
    saves = 0
    original_save = manager._save_locked

    def count_save() -> None:
        nonlocal saves
        saves += 1
        original_save()

    manager._save_locked = count_save

    manager.close()
    manager.close()

    assert saves == 1
    assert popen.processes[0].poll() == -15


def test_save_uses_process_specific_temporary_file(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    (tmp_path / "state.json.tmp").mkdir()

    manager._save_locked()

    assert (tmp_path / "state.json").is_file()
    assert not (tmp_path / f"state.{os.getpid()}.json.tmp").exists()


def test_task_control_is_limited_to_owner_session(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    task = _start(
        manager,
        tmp_path,
        lifecycle="task",
        restart_policy="never",
    )

    with pytest.raises(PermissionError, match="another session"):
        manager.logs(task["id"], owner_session_key="session-b")

    assert manager.logs(task["id"], owner_session_key="session-a") == ""


def test_health_check_must_target_localhost(tmp_path: Path) -> None:
    manager, popen = _manager(tmp_path)

    with pytest.raises(ValueError, match="localhost"):
        _start(manager, tmp_path, health_url="http://example.com/health")

    assert not popen.processes

def test_start_failure_rolls_back_record(tmp_path: Path) -> None:
    def fail_to_start(*args: Any, **kwargs: Any) -> None:
        raise OSError("spawn failed")

    manager = ManagedProcessManager(
        root=tmp_path,
        popen=fail_to_start,
        platform_name="POSIX",
        start_watcher=False,
    )

    with pytest.raises(OSError, match="spawn failed"):
        _start(manager, tmp_path)

    assert manager.list() == []


def test_logs_returns_latest_lines_without_exposing_other_sessions(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    service = _start(manager, tmp_path)
    Path(service["log_path"]).write_text(
        "first\nsecond\nthird\n", encoding="utf-8"
    )

    assert manager.logs(
        service["id"], tail=2, owner_session_key="session-a"
    ) == "second\nthird"
    with pytest.raises(PermissionError, match="another session"):
        manager.logs(service["id"], owner_session_key="session-b")

async def test_start_tool_reuses_exec_workspace_guard(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    manager, popen = _manager(tmp_path / "runtime")
    tool = StartProcessTool(
        ExecTool(working_dir=str(workspace), restrict_to_workspace=True),
        manager,
    )

    result = await tool.execute(
        command="echo unsafe",
        working_dir=str(outside),
        lifecycle="task",
    )

    assert result.is_error is True
    assert "outside the configured workspace" in result
    assert not popen.processes

