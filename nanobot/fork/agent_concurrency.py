"""Cross-process concurrency gate for fork-managed agent turns.

The gate is intentionally kept under ``nanobot.fork`` so upstream syncs only
need to preserve one small integration hook in the API server.  SQLite leases
make the limit shared by all local Nanobot serve processes and recoverable
when a process exits without cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_concurrency_runs (
    ticket_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('waiting', 'running')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    lease_until REAL,
    process_id INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    session_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_concurrency_waiting
    ON agent_concurrency_runs(status, created_at, ticket_id);
"""


@dataclass(frozen=True)
class AgentConcurrencyConfig:
    enabled: bool = False
    max_sessions: int = 1
    state_path: Path = Path.home() / ".nanobot" / "fork-concurrency.sqlite3"
    queue_timeout_seconds: float = 1800.0
    lease_timeout_seconds: float = 120.0
    heartbeat_interval_seconds: float = 15.0
    poll_interval_seconds: float = 0.25

    @classmethod
    def from_mapping(cls, value: object, *, base_dir: Path | None = None) -> "AgentConcurrencyConfig":
        data = value if isinstance(value, dict) else {}
        base = base_dir or Path.cwd()
        raw_path = data.get("statePath", data.get("state_path", str(cls.state_path)))
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_sessions=max(1, int(data.get("maxSessions", data.get("max_sessions", 1)))),
            state_path=path,
            queue_timeout_seconds=max(0.1, float(data.get("queueTimeoutSeconds", data.get("queue_timeout_seconds", 1800)))),
            lease_timeout_seconds=max(2.0, float(data.get("leaseTimeoutSeconds", data.get("lease_timeout_seconds", 120)))),
            heartbeat_interval_seconds=max(0.5, float(data.get("heartbeatIntervalSeconds", data.get("heartbeat_interval_seconds", 15)))),
            poll_interval_seconds=max(0.05, float(data.get("pollIntervalSeconds", data.get("poll_interval_seconds", 0.25)))),
        )


def load_agent_concurrency_config(config_path: Path | None = None) -> AgentConcurrencyConfig:
    """Load fork config without adding fields to upstream's Pydantic schema."""
    path_text = os.environ.get("NANOBOT_FORK_CONCURRENCY_CONFIG", "").strip()
    path = config_path or (Path(path_text).expanduser() if path_text else None)
    if path is None:
        return AgentConcurrencyConfig()
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        logger.warning("Fork concurrency config does not exist: {}", path)
        return AgentConcurrencyConfig(enabled=False)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Cannot load fork concurrency config {}: {}", path, exc)
        return AgentConcurrencyConfig(enabled=False)
    return AgentConcurrencyConfig.from_mapping(raw.get("forkConcurrency", raw), base_dir=path.parent)


class AgentConcurrencyLease:
    def __init__(self, gate: "AgentConcurrencyGate", ticket_id: str) -> None:
        self._gate = gate
        self.ticket_id = ticket_id
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._released = False

    async def __aenter__(self) -> "AgentConcurrencyLease":
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.release()

    async def _heartbeat(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._gate.config.heartbeat_interval_seconds)
                await asyncio.to_thread(self._gate._renew, self.ticket_id)
        except asyncio.CancelledError:
            return

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        await asyncio.to_thread(self._gate._release, self.ticket_id)


class AgentConcurrencyGate:
    def __init__(self, config: AgentConcurrencyConfig, *, instance_id: str = "") -> None:
        self.config = config
        self.instance_id = instance_id or socket.gethostname()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.config.state_path), timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(_SCHEMA)
        return connection

    def _initialize(self) -> None:
        if self._initialized:
            return
        with self._connect() as connection:
            self._cleanup_expired(connection, time.time())
        self._initialized = True

    def _cleanup_expired(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "DELETE FROM agent_concurrency_runs WHERE status = 'running' AND lease_until < ?",
            (now,),
        )
        connection.execute(
            "DELETE FROM agent_concurrency_runs WHERE status = 'waiting' AND created_at < ?",
            (now - self.config.queue_timeout_seconds,),
        )

    async def acquire(self, session_key: str) -> AgentConcurrencyLease | None:
        if not self.config.enabled:
            return None
        await asyncio.to_thread(self._initialize)
        ticket_id = uuid.uuid4().hex
        created_at = time.time()
        await asyncio.to_thread(self._enqueue, ticket_id, created_at, session_key)
        deadline = created_at + self.config.queue_timeout_seconds
        try:
            while True:
                if await asyncio.to_thread(self._try_start, ticket_id):
                    logger.info("Agent concurrency slot acquired ticket={} session={}", ticket_id, session_key)
                    return AgentConcurrencyLease(self, ticket_id)
                if time.time() >= deadline:
                    await asyncio.to_thread(self._delete_ticket, ticket_id)
                    logger.warning("Agent concurrency queue timeout session={}", session_key)
                    return None
                await asyncio.sleep(self.config.poll_interval_seconds)
        except BaseException:
            await asyncio.to_thread(self._delete_ticket, ticket_id)
            raise

    def _enqueue(self, ticket_id: str, created_at: float, session_key: str) -> None:
        with self._connect() as connection:
            self._cleanup_expired(connection, time.time())
            connection.execute(
                "INSERT INTO agent_concurrency_runs "
                "(ticket_id, status, created_at, updated_at, lease_until, process_id, instance_id, session_key) "
                "VALUES (?, 'waiting', ?, ?, NULL, ?, ?, ?)",
                (ticket_id, created_at, created_at, os.getpid(), self.instance_id, session_key),
            )

    def _try_start(self, ticket_id: str) -> bool:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._cleanup_expired(connection, now)
            running = connection.execute(
                "SELECT COUNT(*) FROM agent_concurrency_runs WHERE status = 'running'",
            ).fetchone()[0]
            first_waiting = connection.execute(
                "SELECT ticket_id FROM agent_concurrency_runs "
                "WHERE status = 'waiting' ORDER BY created_at, ticket_id LIMIT 1",
            ).fetchone()
            if running >= self.config.max_sessions or not first_waiting or first_waiting[0] != ticket_id:
                connection.commit()
                return False
            connection.execute(
                "UPDATE agent_concurrency_runs SET status = 'running', updated_at = ?, lease_until = ? "
                "WHERE ticket_id = ? AND status = 'waiting'",
                (now, now + self.config.lease_timeout_seconds, ticket_id),
            )
            connection.commit()
            return True

    def _renew(self, ticket_id: str) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "UPDATE agent_concurrency_runs SET updated_at = ?, lease_until = ? "
                "WHERE ticket_id = ? AND status = 'running'",
                (now, now + self.config.lease_timeout_seconds, ticket_id),
            )

    def _release(self, ticket_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM agent_concurrency_runs WHERE ticket_id = ?", (ticket_id,))
        logger.info("Agent concurrency slot released ticket={}", ticket_id)

    def _delete_ticket(self, ticket_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM agent_concurrency_runs WHERE ticket_id = ?", (ticket_id,))
