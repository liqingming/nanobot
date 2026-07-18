"""File-logged background tasks and supervised local services."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlparse

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_session_key
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema
from nanobot.agent.tools.shell import ExecTool as _ExecTool
from nanobot.config.paths import get_runtime_subdir

_IS_WINDOWS = sys.platform == "win32"
_LIFECYCLES = {"task", "service"}
_RESTART_POLICIES = {"never", "on-failure", "always"}
_DEFAULT_LOG_LIMIT = 1024 * 1024 * 1024


class ManagedProcessManager:
    """Persist process metadata and supervise services without changing exec sessions."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        popen: Any = subprocess.Popen,
        run: Any = subprocess.run,
        platform_name: str | None = None,
        start_watcher: bool = True,
    ) -> None:
        self.root = root or get_runtime_subdir("processes")
        self.logs_dir = self.root / "logs"
        self.state_path = self.root / "state.json"
        self._popen = popen
        self._run = run
        self.platform_name = platform_name or ("Windows" if _IS_WINDOWS else "POSIX")
        self._lock = threading.RLock()
        self._children: dict[str, Any] = {}
        self._records = self._load()
        self._stop_event = threading.Event()
        self._closed = False
        self._watcher: threading.Thread | None = None
        if start_watcher:
            self._watcher = threading.Thread(
                target=self._watch_loop,
                name="nanobot-process-watchdog",
                daemon=True,
            )
            self._watcher.start()

    def start(
        self,
        *,
        command: str,
        cwd: str,
        env: dict[str, str],
        shell_program: str | None,
        login: bool,
        lifecycle: str,
        restart_policy: str,
        name: str | None,
        owner_session_key: str | None,
        health_url: str | None,
        health_grace_s: int,
        max_restarts: int,
        max_log_bytes: int,
    ) -> dict[str, Any]:
        self._validate(lifecycle, restart_policy, health_url)
        with self._lock:
            self._reconcile_locked()
            if name and any(
                rec.get("name") == name and rec.get("desired") == "running"
                for rec in self._records.values()
            ):
                raise ValueError(f"managed process name already exists: {name}")
            process_id = uuid.uuid4().hex[:12]
            record = {
                "id": process_id,
                "name": name or process_id,
                "command": command,
                "cwd": cwd,
                "env": env,
                "shell_program": shell_program,
                "login": bool(login),
                "lifecycle": lifecycle,
                "restart_policy": restart_policy,
                "owner_session_key": owner_session_key,
                "health_url": health_url,
                "health_grace_s": max(0, health_grace_s),
                "health_failures": 0,
                "max_restarts": max(0, max_restarts),
                "restart_count": 0,
                "max_log_bytes": max(1024 * 1024, max_log_bytes),
                "desired": "running",
                "status": "starting",
                "created_at": time.time(),
                "next_restart_at": 0.0,
                "log_path": str(self.logs_dir / f"{process_id}.log"),
            }
            self._records[process_id] = record
            try:
                self._spawn_locked(record)
                self._save_locked()
            except Exception:
                record["desired"] = "stopped"
                self._terminate_locked(record)
                self._records.pop(process_id, None)
                raise
            return self._public(record)

    def list(self, owner_session_key: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._reconcile_locked()
            self._save_locked()
            return [
                self._public(rec)
                for rec in self._records.values()
                if not owner_session_key
                or not rec.get("owner_session_key")
                or rec.get("owner_session_key") == owner_session_key
            ]

    def stop(
        self, process_id: str, owner_session_key: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            rec = self._require(process_id)
            self._check_owner(rec, owner_session_key)
            rec["desired"] = "stopped"
            self._terminate_locked(rec)
            rec["status"] = "stopped"
            self._save_locked()
            return self._public(rec)

    def restart(
        self, process_id: str, owner_session_key: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            rec = self._require(process_id)
            self._check_owner(rec, owner_session_key)
            self._terminate_locked(rec)
            rec["desired"] = "running"
            rec["restart_count"] = 0
            rec["next_restart_at"] = 0.0
            self._spawn_locked(rec)
            self._save_locked()
            return self._public(rec)

    def logs(
        self,
        process_id: str,
        tail: int = 100,
        owner_session_key: str | None = None,
    ) -> str:
        with self._lock:
            rec = self._require(process_id)
            self._check_owner(rec, owner_session_key)
            path = Path(str(rec["log_path"]))
        if not path.is_file():
            return ""
        return self._read_tail(path, tail)

    def shutdown_tasks(self) -> None:
        with self._lock:
            changed = False
            for rec in self._records.values():
                if rec.get("lifecycle") == "task" and rec.get("desired") == "running":
                    rec["desired"] = "stopped"
                    self._terminate_locked(rec)
                    rec["status"] = "stopped"
                    changed = True
            if changed:
                self._save_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        self.shutdown_tasks()

    def _validate(self, lifecycle: str, restart_policy: str, health_url: str | None) -> None:
        if lifecycle not in _LIFECYCLES:
            raise ValueError("lifecycle must be 'task' or 'service'")
        if restart_policy not in _RESTART_POLICIES:
            raise ValueError("restart_policy must be never, on-failure, or always")
        if lifecycle == "task" and restart_policy != "never":
            raise ValueError("task lifecycle only supports restart_policy='never'")
        if health_url:
            parsed = urlparse(health_url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("health_url must use http or https")
            if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("health_url must target localhost")

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(2.0):
            with suppress(Exception):
                with self._lock:
                    self._reconcile_locked()
                    self._save_locked()

    def _reconcile_locked(self) -> None:
        now = time.time()
        for rec in self._records.values():
            alive = self._alive(rec)
            log_path = Path(str(rec.get("log_path", "")))
            if (
                alive
                and log_path.is_file()
                and log_path.stat().st_size > int(rec["max_log_bytes"])
            ):
                rec["desired"] = "stopped"
                rec["status"] = "log_limit"
                self._terminate_locked(rec)
                alive = False
            if (
                alive
                and rec.get("health_url")
                and now >= float(rec.get("started_at", now)) + int(rec["health_grace_s"])
            ):
                if self._health_ok(str(rec["health_url"])):
                    rec["health_failures"] = 0
                else:
                    rec["health_failures"] = int(rec.get("health_failures", 0)) + 1
                    if rec["health_failures"] >= 3:
                        rec["status"] = "unhealthy"
                        self._terminate_locked(rec)
                        alive = False
            if alive:
                rec["status"] = "running"
                continue
            if rec.get("desired") != "running" or rec.get("lifecycle") != "service":
                if rec.get("status") not in {"completed", "log_limit", "stopped"}:
                    rec["status"] = "exited"
                continue
            if rec.get("restart_policy") == "never":
                rec["status"] = "exited"
                continue
            child = self._children.get(str(rec.get("id")))
            if (
                rec.get("restart_policy") == "on-failure"
                and child is not None
                and child.poll() == 0
            ):
                rec["status"] = "completed"
                rec["desired"] = "stopped"
                continue
            restarts = int(rec.get("restart_count", 0))
            if restarts >= int(rec.get("max_restarts", 0)):
                rec["status"] = "restart_exhausted"
                continue
            if now < float(rec.get("next_restart_at", 0)):
                rec["status"] = "restart_wait"
                continue
            rec["restart_count"] = restarts + 1
            rec["next_restart_at"] = now + min(60, 2 ** min(restarts, 6))
            self._spawn_locked(rec)

    def _spawn_locked(self, rec: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with open(rec["log_path"], "a", encoding="utf-8") as log:
            process = self._popen(
                self._argv(rec),
                cwd=rec["cwd"],
                env=rec["env"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **self._platform_kwargs(),
            )
        rec["pid"] = int(process.pid)
        rec["identity"] = self._identity(int(process.pid))
        rec["started_at"] = time.time()
        rec["health_failures"] = 0
        rec["status"] = "running"
        self._children[rec["id"]] = process

    def _argv(self, rec: dict[str, Any]) -> list[str]:
        command = str(rec["command"])
        program = rec.get("shell_program")
        if self.platform_name == "Windows":
            program = (
                program
                or shutil.which("pwsh")
                or shutil.which("powershell")
                or "powershell"
            )
            if PureWindowsPath(program).name.lower() in {"cmd", "cmd.exe"}:
                return [program, "/d", "/s", "/c", command]
            command = _ExecTool._normalize_powershell_command(command)
            return [program, "-NoProfile", "-NonInteractive", "-Command", command]
        program = program or shutil.which("bash") or "/bin/bash"
        args = [program]
        if rec.get("login") and Path(program).name in {"bash", "zsh"}:
            args.append("-l")
        return [*args, "-c", command]

    def _platform_kwargs(self) -> dict[str, Any]:
        if self.platform_name == "Windows":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            return {"creationflags": flags}
        return {"start_new_session": True}

    def _alive(self, rec: dict[str, Any]) -> bool:
        child = self._children.get(str(rec.get("id")))
        if child is not None:
            return child.poll() is None
        pid = int(rec.get("pid") or 0)
        if pid <= 0:
            return False
        identity = self._identity(pid)
        return identity is not None and identity == rec.get("identity")

    def _terminate_locked(self, rec: dict[str, Any]) -> None:
        pid = int(rec.get("pid") or 0)
        if pid <= 0 or not self._alive(rec):
            return
        if self.platform_name == "Windows":
            self._run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        child = self._children.pop(str(rec.get("id")), None)
        if child is not None:
            with suppress(Exception):
                child.wait(timeout=3)

    def _identity(self, pid: int) -> str | None:
        if self.platform_name == "Windows":
            from nanobot.gateway.runtime import _windows_process_identity

            return _windows_process_identity(pid)
        try:
            stat = Path(f"/proc/{pid}/stat")
            if stat.is_file():
                return stat.read_text(encoding="utf-8").split()[21]
            os.kill(pid, 0)
            return str(pid)
        except (OSError, IndexError):
            return None

    @staticmethod
    def _read_tail(path: Path, lines: int, max_bytes: int = 4 * 1024 * 1024) -> str:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            offset = max(0, size - max_bytes)
            handle.seek(offset)
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        if offset:
            text = text.partition("\n")[2]
        return "\n".join(text.splitlines()[-lines:])

    @staticmethod
    def _health_ok(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return 200 <= response.status < 500
        except Exception:
            return False

    def _require(self, process_id: str) -> dict[str, Any]:
        rec = self._records.get(process_id)
        if rec is None:
            raise KeyError(process_id)
        return rec

    @staticmethod
    def _check_owner(
        rec: dict[str, Any], owner_session_key: str | None
    ) -> None:
        if not owner_session_key:
            return
        owner = rec.get("owner_session_key")
        if owner and owner != owner_session_key:
            raise PermissionError("task belongs to another session")

    @staticmethod
    def _public(rec: dict[str, Any]) -> dict[str, Any]:
        hidden = {"env", "identity", "shell_program"}
        return {key: value for key, value in rec.items() if key not in hidden}

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            rows = data.get("processes", []) if isinstance(data, dict) else []
            return {
                row["id"]: row
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            }
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_locked(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f"state.{os.getpid()}.json.tmp"
        try:
            tmp.write_text(
                json.dumps(
                    {"version": 1, "processes": list(self._records.values())},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            with suppress(OSError):
                os.chmod(tmp, 0o600)
            tmp.replace(self.state_path)
        finally:
            with suppress(OSError):
                tmp.unlink()


_DEFAULT_MANAGED_PROCESS_MANAGER: ManagedProcessManager | None = None


def get_default_managed_process_manager() -> ManagedProcessManager:
    global _DEFAULT_MANAGED_PROCESS_MANAGER
    if _DEFAULT_MANAGED_PROCESS_MANAGER is None:
        _DEFAULT_MANAGED_PROCESS_MANAGER = ManagedProcessManager()
        atexit.register(_DEFAULT_MANAGED_PROCESS_MANAGER.close)
    return _DEFAULT_MANAGED_PROCESS_MANAGER


def shutdown_managed_tasks() -> None:
    if _DEFAULT_MANAGED_PROCESS_MANAGER is not None:
        _DEFAULT_MANAGED_PROCESS_MANAGER.shutdown_tasks()


@tool_parameters(
    tool_parameters_schema(
        command=StringSchema("Command to start"),
        name=StringSchema("Optional stable display name", nullable=True),
        working_dir=StringSchema("Optional working directory", nullable=True),
        lifecycle=StringSchema("task or service"),
        restart_policy=StringSchema("never, on-failure, or always", nullable=True),
        health_url=StringSchema("Optional localhost HTTP health URL", nullable=True),
        health_grace_s=IntegerSchema(
            10, description="Health check startup grace", minimum=0, maximum=600
        ),
        max_restarts=IntegerSchema(
            10, description="Maximum automatic restarts", minimum=0, maximum=1000
        ),
        max_log_bytes=IntegerSchema(
            _DEFAULT_LOG_LIMIT,
            description="Stop when log exceeds this size",
            minimum=1048576,
            maximum=5368709120,
        ),
        required=["command", "lifecycle"],
    )
)
class StartProcessTool(Tool):
    _scopes = {"core", "subagent"}
    config_key = "exec"

    def __init__(
        self,
        executor: _ExecTool,
        manager: ManagedProcessManager | None = None,
    ) -> None:
        self.executor = executor
        self.manager = manager or get_default_managed_process_manager()

    @classmethod
    def config_cls(cls):
        from nanobot.agent.tools.shell import ExecToolConfig

        return ExecToolConfig

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return ctx.config.exec.enable

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(_ExecTool.create(ctx))

    @property
    def name(self) -> str:
        return "start_process"

    @property
    def description(self) -> str:
        return (
            "Start a file-logged background task or supervised local service. "
            "Use service for HTTP/dev servers that should restart on failure."
        )

    async def execute(
        self,
        command: str,
        lifecycle: str,
        name: str | None = None,
        working_dir: str | None = None,
        restart_policy: str | None = None,
        health_url: str | None = None,
        health_grace_s: int = 10,
        max_restarts: int = 10,
        max_log_bytes: int = _DEFAULT_LOG_LIMIT,
        **kwargs: Any,
    ) -> str:
        prepared = self.executor._prepare_command(
            command, working_dir, 0, None, False
        )
        if isinstance(prepared, str):
            return prepared
        policy = restart_policy or (
            "on-failure" if lifecycle == "service" else "never"
        )
        try:
            rec = self.manager.start(
                command=prepared.command,
                cwd=prepared.cwd,
                env=prepared.env,
                shell_program=prepared.shell_program,
                login=prepared.login,
                lifecycle=lifecycle,
                restart_policy=policy,
                name=name,
                owner_session_key=current_request_session_key(),
                health_url=health_url,
                health_grace_s=health_grace_s,
                max_restarts=max_restarts,
                max_log_bytes=max_log_bytes,
            )
            return (
                f"Started {rec['lifecycle']} {rec['id']} ({rec['name']}); "
                f"pid={rec['pid']}; log={rec['log_path']}"
            )
        except Exception as exc:
            return ToolResult.error(f"Error starting managed process: {exc}")


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema("list, logs, stop, or restart"),
        process_id=StringSchema("Required except for list", nullable=True),
        tail=IntegerSchema(
            100, description="Log lines to return", minimum=1, maximum=5000
        ),
        required=["action"],
    )
)
class ProcessControlTool(Tool):
    _scopes = {"core", "subagent"}
    config_key = "exec"

    def __init__(self, manager: ManagedProcessManager | None = None) -> None:
        self.manager = manager or get_default_managed_process_manager()

    @classmethod
    def config_cls(cls):
        from nanobot.agent.tools.shell import ExecToolConfig

        return ExecToolConfig

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return ctx.config.exec.enable

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls()

    @property
    def name(self) -> str:
        return "process_control"

    @property
    def description(self) -> str:
        return (
            "List managed tasks/services, read file logs, stop a process tree, "
            "or restart a managed process."
        )

    @property
    def supports_read_only_calls(self) -> bool:
        return True

    def is_read_only_call(self, params: Any) -> bool:
        return isinstance(params, dict) and params.get("action") in {"list", "logs"}

    async def execute(
        self,
        action: str,
        process_id: str | None = None,
        tail: int = 100,
        **kwargs: Any,
    ) -> str:
        try:
            if action == "list":
                rows = self.manager.list(current_request_session_key())
                return (
                    json.dumps(rows, ensure_ascii=False, indent=2)
                    if rows
                    else "No managed processes."
                )
            if not process_id:
                return ToolResult.error("Error: process_id is required")
            owner = current_request_session_key()
            if action == "logs":
                return (
                    self.manager.logs(
                        process_id, tail=tail, owner_session_key=owner
                    )
                    or "(no output yet)"
                )
            if action == "stop":
                return json.dumps(
                    self.manager.stop(process_id, owner_session_key=owner),
                    ensure_ascii=False,
                )
            if action == "restart":
                return json.dumps(
                    self.manager.restart(process_id, owner_session_key=owner),
                    ensure_ascii=False,
                )
            return ToolResult.error(
                "Error: action must be list, logs, stop, or restart"
            )
        except KeyError:
            return ToolResult.error(
                f"Error: managed process not found: {process_id}"
            )
        except Exception as exc:
            return ToolResult.error(f"Error controlling managed process: {exc}")
