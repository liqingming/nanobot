"""OpenAI Codex provider backed by the official ``codex app-server``.

Codex owns OAuth, model transport, websocket continuation, and upstream retries.
Nanobot still owns its runner and tool execution: app-server dynamic-tool requests
are returned as ``ToolCallRequest`` values and resumed with nanobot's tool result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_workspace_cache_dir, is_default_workspace
from nanobot.fork.agent.execution_scope import current_execution_id
from nanobot.fork.providers.codex_context_checkpoint import (
    ContextCheckpoint,
    pending_checkpoint_messages,
)
from nanobot.fork.providers.codex_native_context import (
    NativeCompactionState,
    check_native_payload,
    native_input_budget,
)
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.providers.openai_codex_provider import (
    OpenAICodexProvider as LegacyOpenAICodexProvider,
)
from nanobot.security.workspace_access import current_workspace_scope
from nanobot.utils.atomic_write import replace_file_with_retry

CODEX_BIN_ENV = "NANOBOT_CODEX_BIN"
CODEX_RPC_TIMEOUT_ENV = "NANOBOT_CODEX_APP_SERVER_RPC_TIMEOUT_S"
CODEX_EVENT_TIMEOUT_ENV = "NANOBOT_CODEX_APP_SERVER_EVENT_TIMEOUT_S"
DEFAULT_CODEX_RPC_TIMEOUT_S = 45.0
DEFAULT_CODEX_EVENT_TIMEOUT_S = 240.0
DEFAULT_CODEX_RECOVERY_ATTEMPTS = 1
DEFAULT_NATIVE_TOOL_CORRECTION_ATTEMPTS = 1
DEFAULT_COMMAND_REFRESH_ATTEMPTS = 1
_APP_SERVER_STREAM_LIMIT = 4 * 1024 * 1024
_LEDGER_VERSION = 2
_LEDGER_MAX_BYTES = 4 * 1024 * 1024
_LEDGER_RETENTION_S = 7 * 24 * 60 * 60
_DISABLED_FEATURES = (
    "apps",
    "remote_plugin",
    "browser_use",
    "computer_use",
    "image_generation",
    "multi_agent",
    "skill_search",
    "tool_suggest",
    "goals",
    "hooks",
)
_THREAD_CONFIG_OVERRIDES: dict[str, Any] = {
    "web_search": "disabled",
    "agents": {"enabled": False},
    "mcp_servers": {},
    "plugins": {},
    "features": {feature: False for feature in _DISABLED_FEATURES},
}
_NATIVE_TOOL_ITEM_TYPES = frozenset(
    {
        "mcpToolCall",
        "collabAgentToolCall",
        "webSearch",
        "imageGeneration",
        "computerUse",
        "browserUse",
    }
)
_NATIVE_TOOL_GUARD = (
    "\n\nThe host application owns dynamic tool execution. Native Codex commands and file "
    "changes are supported inside the configured workspace sandbox. For every other "
    "capability, use only the dynamic tools provided by the client; do not use other "
    "native Codex tools. For subagents use the Nanobot spawn tool (fresh context), "
    "then subagent_control for status, wait, or cancel using the returned task id. "
    "Never use native collaboration tools. If the host tools are unavailable, report "
    "the blocker; do not substitute native agents or invent worker ids."
)
_NATIVE_TOOL_CORRECTION = (
    "The previous Codex turn attempted to use a native Codex tool, which the host "
    "blocked. Continue the same task using only the dynamic Nanobot tools provided "
    "by the client. Native commands and file changes are the only supported exceptions. "
    "This restriction applies to MCP, web access, images, browser/computer use, and "
    "subagents. To inspect a local image, call the Nanobot read_file tool with its path; "
    "the image will be returned in that tool result. Never call imageView. Do not repeat "
    "tool calls whose successful results already appear in the transcript."
)
_SKILL_DYNAMIC_TOOL_RE = re.compile(r"\bmcp__[A-Za-z0-9_]+")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)(\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


class CodexAppServerError(RuntimeError):
    """Official app-server process or protocol failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class CodexIdempotencyLedgerError(RuntimeError):
    """Durable tool-result checkpoint could not be trusted or written."""


class _ToolResultLedger:
    """Turn-scoped durable result index used while rebuilding a crashed bridge.

    Tool result bodies live in per-turn sidecar files.  The JSON ledger is only a
    small manifest, so a large image or command output cannot consume the 4 MiB
    idempotency budget by itself.
    """

    def __init__(
        self,
        path: Path,
        *,
        workspace: str,
        session_key: str,
        turn_id: str,
    ) -> None:
        self.path = path
        self.workspace = workspace
        self.session_key = session_key
        self.turn_id = turn_id
        self.result_root = path.with_name(f"{path.stem}.results")
        self.entries: list[dict[str, Any]] = []
        self._replay_index = 0
        self._native_image_replay_indexes: set[int] = set()
        self._load()

    @property
    def has_entries(self) -> bool:
        return bool(self.entries)

    def record_messages(self, messages: list[dict[str, Any]]) -> None:
        calls = _tool_calls_by_id(messages)
        results = _tool_results_by_id(messages)
        if not results:
            return
        existing = {
            str(entry.get("call_id")): entry
            for entry in self.entries
            if isinstance(entry.get("call_id"), str)
        }
        changed = False
        for call_id, content in results.items():
            call = calls.get(call_id)
            if call is None:
                continue
            name, arguments = call
            canonical_arguments = _canonical_tool_arguments(arguments)
            signature = _tool_signature(name, canonical_arguments)
            old = existing.get(call_id)
            if old is not None:
                if old.get("signature") != signature:
                    raise CodexIdempotencyLedgerError(
                        f"Conflicting durable result for tool call {call_id}."
                    )
                # The ledger is write-once. Context governance may replace an
                # older tool result with a digest before a later model request;
                # keep the original durable result for bridge recovery.
                continue
            serialized = _serialize_tool_result(content)
            entry = {
                "call_id": call_id,
                "name": name,
                "arguments": canonical_arguments,
                "signature": signature,
                "resultRef": self._store_result(call_id, serialized, content),
                "success": not _looks_like_tool_error(content),
            }
            self.entries.append(entry)
            existing[call_id] = entry
            changed = True
        if changed:
            self._write()

    def begin_recovery(self) -> None:
        self._replay_index = 0

    def begin_native_image_recovery(self) -> None:
        self._native_image_replay_indexes.clear()

    def cached_native_image_result(self, call: ToolCallRequest) -> tuple[Any, bool] | None:
        """Match a synthetic imageView result without imposing global tool order."""
        signature = _tool_signature(
            call.name,
            _canonical_tool_arguments(call.arguments),
        )
        for index, entry in enumerate(self.entries):
            if index in self._native_image_replay_indexes:
                continue
            call_id = entry.get("call_id")
            if not isinstance(call_id, str) or not call_id.startswith("native-image-view-"):
                continue
            if entry.get("signature") != signature:
                continue
            self._native_image_replay_indexes.add(index)
            return self._entry_result(entry), bool(entry.get("success"))
        return None

    def cached_result(
        self, call: ToolCallRequest, *, checkpoint_continuation: bool = False,
    ) -> tuple[Any, bool] | None:
        # 正常检查点续行允许重新观察已变化的宿主状态；不能返回旧 running 回执。
        # 只放行已知无副作用的操作和全新调用 ID，不猜测 exec/read_file 等工具的语义。
        # 故障恢复绝不传入此标记，取消/未知操作仍遵守原有顺序保护。
        if (
            checkpoint_continuation
            and isinstance(call.arguments, dict)
            and (
                (call.name == "subagent_control"
                 and call.arguments.get("action") in {"list", "status"})
                or (call.name == "write_stdin"
                    and call.arguments.get("chars") in (None, "")
                    and call.arguments.get("close_stdin", False) is False
                    and call.arguments.get("terminate", False) is False)
            )
            and call.id
            and not any(entry.get("call_id") == call.id for entry in self.entries)
        ):
            return None
        signature = _tool_signature(
            call.name,
            _canonical_tool_arguments(call.arguments),
        )
        if self._replay_index >= len(self.entries):
            if any(entry.get("signature") == signature for entry in self.entries):
                raise CodexIdempotencyLedgerError(
                    "Tool signature repeated again during the same recovery."
                )
            return None
        entry = self.entries[self._replay_index]
        if entry.get("signature") != signature:
            remaining_signatures = {
                item.get("signature") for item in self.entries[self._replay_index + 1 :]
            }
            if signature not in remaining_signatures:
                if any(item.get("signature") == signature
                       for item in self.entries[:self._replay_index]):
                    raise CodexIdempotencyLedgerError(
                        "Tool signature repeated again during the same recovery."
                    )
                return None
            raise CodexIdempotencyLedgerError(
                "Tool replay order diverged during Codex bridge recovery. "
                f"session={self.session_key!r}, turn={self.turn_id!r}, "
                f"index={self._replay_index}, "
                f"expected_tool={entry.get('name')!r}, "
                f"expected_call_id={entry.get('call_id')!r}, "
                f"actual_tool={call.name!r}, actual_call_id={call.id!r}. "
                "Tool arguments and results omitted."
            )
        self._replay_index += 1
        return self._entry_result(entry), bool(entry.get("success"))

    def clear(self) -> None:
        with suppress(OSError):
            self.path.unlink(missing_ok=True)
        with suppress(OSError):
            shutil.rmtree(self.result_root)
        self.entries.clear()
        self._replay_index = 0
        self._native_image_replay_indexes.clear()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size > _LEDGER_MAX_BYTES:
                raise CodexIdempotencyLedgerError("Idempotency ledger is too large.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") not in {1, _LEDGER_VERSION}:
                raise CodexIdempotencyLedgerError("Unsupported idempotency ledger format.")
            identity = (raw.get("workspace"), raw.get("session_key"), raw.get("turn_id"))
            if identity != (self.workspace, self.session_key, self.turn_id):
                raise CodexIdempotencyLedgerError("Idempotency ledger identity mismatch.")
            entries = raw.get("entries")
            if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
                raise CodexIdempotencyLedgerError("Invalid idempotency ledger entries.")
            for entry in entries:
                if not all(
                    isinstance(entry.get(field), str)
                    for field in ("call_id", "name", "arguments", "signature")
                ) or not isinstance(entry.get("success"), bool):
                    raise CodexIdempotencyLedgerError("Invalid idempotency ledger entry fields.")
                has_inline = isinstance(entry.get("content"), str)
                result_ref = entry.get("resultRef")
                has_external = isinstance(result_ref, dict) and all(
                    isinstance(result_ref.get(field), expected)
                    for field, expected in (
                        ("path", str),
                        ("size", int),
                        ("sha256", str),
                        ("contentType", str),
                        ("summary", str),
                    )
                )
                if not has_inline and not has_external:
                    raise CodexIdempotencyLedgerError(
                        "Ledger result reference is missing or invalid."
                    )
                if entry["signature"] != _tool_signature(entry["name"], entry["arguments"]):
                    raise CodexIdempotencyLedgerError("Idempotency ledger signature mismatch.")
            self.entries = entries
        except Exception as exc:
            logger.warning("Ignoring untrusted Codex idempotency ledger {}: {}", self.path, exc)
            self.entries = []

    def _write(self) -> None:
        payload = {
            "version": _LEDGER_VERSION,
            "workspace": self.workspace,
            "session_key": self.session_key,
            "turn_id": self.turn_id,
            "entries": self.entries,
        }
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > _LEDGER_MAX_BYTES:
            raise CodexIdempotencyLedgerError("Idempotency ledger exceeds 4 MiB.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_stale_ledgers(self.path.parent, keep=self.path)
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(tmp_path, flags, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            replace_file_with_retry(tmp_path, self.path)
            with suppress(OSError, NotImplementedError):
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except Exception as exc:
            raise CodexIdempotencyLedgerError(
                f"Failed to persist Codex tool-result ledger: {exc}"
            ) from exc
        finally:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    def _store_result(self, call_id: str, serialized: str, content: Any) -> dict[str, Any]:
        raw = serialized.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        call_digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:16]
        filename = f"{call_digest}-{digest[:16]}.result"
        target = self.result_root / filename
        self.result_root.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            _write_private_atomic(target, raw)
        relative = target.relative_to(self.path.parent).as_posix()
        return {
            "path": relative,
            "size": len(raw),
            "sha256": digest,
            "contentType": _tool_result_content_type(content),
            "summary": _tool_result_summary(content),
        }

    def _entry_result(self, entry: dict[str, Any]) -> Any:
        # Version 1 ledgers are still recoverable during a rolling upgrade.
        if isinstance(entry.get("content"), str):
            return _deserialize_tool_result(entry["content"])
        result_ref = entry.get("resultRef")
        if not isinstance(result_ref, dict):
            raise CodexIdempotencyLedgerError("Ledger result reference is missing.")
        candidate = (self.path.parent / str(result_ref.get("path", ""))).resolve()
        root = self.result_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CodexIdempotencyLedgerError(
                "Ledger result reference escapes its result directory."
            ) from exc
        try:
            raw = candidate.read_bytes()
        except OSError as exc:
            raise CodexIdempotencyLedgerError(
                f"Externalized tool result is unavailable: {candidate}"
            ) from exc
        if len(raw) != result_ref.get("size") or hashlib.sha256(raw).hexdigest() != result_ref.get(
            "sha256"
        ):
            raise CodexIdempotencyLedgerError(
                f"Externalized tool result failed integrity validation: {candidate}"
            )
        return _deserialize_tool_result(raw.decode("utf-8"))


def _cleanup_stale_ledgers(root: Path, *, keep: Path) -> None:
    cutoff = time.time() - _LEDGER_RETENTION_S
    for pattern in ("*.json", ".*.tmp"):
        for candidate in root.glob(pattern):
            if candidate == keep:
                continue
            with suppress(OSError):
                if candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
    for candidate in root.glob("*.results"):
        if candidate == keep.with_name(f"{keep.stem}.results"):
            continue
        with suppress(OSError):
            if candidate.stat().st_mtime < cutoff:
                shutil.rmtree(candidate)


def _write_private_atomic(path: Path, raw: bytes) -> None:
    """Atomically write a private sidecar before publishing its ledger reference."""
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(tmp_path, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file_with_retry(tmp_path, path)
    except Exception as exc:
        raise CodexIdempotencyLedgerError(
            f"Failed to persist externalized Codex tool result: {exc}"
        ) from exc
    finally:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)


class _CodexAppServerTurn:
    """One ephemeral app-server process for one nanobot outer turn."""

    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        rpc_timeout_s: float = DEFAULT_CODEX_RPC_TIMEOUT_S,
        event_timeout_s: float = DEFAULT_CODEX_EVENT_TIMEOUT_S,
    ):
        self.command = command
        self.env = env
        self.rpc_timeout_s = rpc_timeout_s
        self.event_timeout_s = event_timeout_s
        self.process: asyncio.subprocess.Process | None = None
        self._next_rpc_id = 1
        self._messages: deque[dict[str, Any]] = deque()
        self._pending_tools: dict[str, Any] = {}
        self._closed = False
        self._last_usage: dict[str, int] = {}
        self._reported_usage: dict[str, int] = {}
        self._context_input_tokens: int | None = None
        self.native_compaction = NativeCompactionState()
        self.context_checkpoint = ContextCheckpoint()
        self.recovery_replay_index: int | None = None
        self.checkpoint_continuation = False
        self._submitted_tool_results = False
        self._streamed_output = False
        self._native_file_changes: dict[str, str] = {}
        self._native_file_change_approvals_declined = 0
        self._native_command_executions: dict[str, str] = {}
        self._native_command_approvals_declined = 0
        self._dynamic_tool_names: set[str] = set()
        self._turn_id: str | None = None
        self._unsafe_protocol_event = False
        self._steered_input = False
        self._stderr_lines: deque[str] = deque(maxlen=24)
        self._stderr_task: asyncio.Task[None] | None = None

    @property
    def submitted_tool_results(self) -> bool:
        return self._submitted_tool_results

    @property
    def stderr_tail(self) -> str | None:
        if not self._stderr_lines:
            return None
        return "\n".join(self._stderr_lines)

    @property
    def streamed_output(self) -> bool:
        return self._streamed_output

    async def start(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        recovery: bool = False,
    ) -> None:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            limit=_APP_SERVER_STREAM_LIMIT,
        )
        assert self.process.stderr is not None
        self._stderr_task = asyncio.create_task(self._drain_stderr(self.process.stderr))
        await self._rpc(
            "initialize",
            {
                "clientInfo": {
                    "name": "nanobot",
                    "title": "Nanobot Codex Bridge",
                    "version": "1.0.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await self._notify("initialized", {})

        workspace = current_workspace_scope()
        cwd = str(workspace.project_path if workspace else Path.cwd().resolve())
        selected_tools, tool_choice_instruction = _apply_tool_choice(tools, tool_choice)
        self._dynamic_tool_names = {
            name for tool in selected_tools or [] if (name := _tool_schema_name(tool)) is not None
        }
        config = dict(_THREAD_CONFIG_OVERRIDES)
        config.update(self.native_compaction.config())
        disabled_skills = await self._unsupported_skill_overrides(cwd, selected_tools)
        if disabled_skills:
            config["skills"] = {"config": disabled_skills}
        instructions, turn_input = _messages_to_app_server_input(
            messages,
            include_current_turn_history=recovery,
            disabled_skill_paths={entry["path"] for entry in disabled_skills},
        )
        if tool_choice_instruction:
            instructions += "\n\n" + tool_choice_instruction
        thread_params: dict[str, Any] = {
            "model": _strip_codex_model_prefix(model),
            "cwd": cwd,
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
            "ephemeral": True,
            "baseInstructions": instructions + _NATIVE_TOOL_GUARD,
            "dynamicTools": _convert_dynamic_tools(selected_tools),
            "config": config,
        }
        result = await self._rpc("thread/start", thread_params)
        thread = result.get("thread") if isinstance(result, dict) else None
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexAppServerError("Codex app-server did not return a thread id.")
        self.native_compaction.thread_id = thread_id
        turn_params: dict[str, Any] = {"threadId": thread_id, "input": turn_input}
        if reasoning_effort:
            turn_params["effort"] = reasoning_effort
        result = await self._rpc("turn/start", turn_params)
        turn = result.get("turn") if isinstance(result, dict) else None
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexAppServerError("Codex app-server did not return a turn id.")
        self._turn_id = turn_id

    async def _unsupported_skill_overrides(
        self,
        cwd: str,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Hide Codex skills whose required native tools are unavailable to Nanobot."""
        try:
            result = await self._rpc("skills/list", {"cwds": [cwd], "forceReload": False})
        except Exception:
            # Older app-server versions may not expose skills/list. Skill filtering
            # is defensive isolation and must not prevent the model from starting.
            logger.debug("Codex skill dependency discovery unavailable", exc_info=True)
            return []
        tool_names = {name for tool in tools or [] if (name := _tool_schema_name(tool)) is not None}
        overrides: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for entry in result.get("data", []):
            if not isinstance(entry, dict) or entry.get("cwd") != cwd:
                continue
            for skill in entry.get("skills", []):
                if not isinstance(skill, dict):
                    continue
                path = skill.get("path")
                if not isinstance(path, str) or not path or path in seen_paths:
                    continue
                # skills.config is an array override. Preserve skills the user
                # already disabled instead of replacing their config with only
                # Nanobot's dependency-based entries.
                if skill.get("enabled") is False or _skill_requires_unavailable_tools(
                    skill, tool_names
                ):
                    overrides.append({"path": path, "enabled": False})
                    seen_paths.add(path)
        if overrides:
            logger.debug(
                "Disabled {} Codex skill(s) with unavailable tool dependencies",
                len(overrides),
            )
        return overrides

    async def steer(self, messages: list[dict[str, Any]]) -> None:
        """在待处理工具边界追加输入，保留原生执行历史，不重建或回放。"""
        inputs = [
            item for message in messages if message.get("role") == "user"
            for item in _content_to_user_inputs(message.get("content"))
        ]
        if not inputs:
            raise CodexIdempotencyLedgerError("Cannot steer Codex without new user input.")
        # 发出后即视为可能已接收；断线不能自动重建并重复注入。
        self._steered_input = True
        result = await self._rpc("turn/steer", {
            "threadId": self.native_compaction.thread_id,
            "expectedTurnId": self._turn_id,
            "input": inputs,
        })
        if not self._turn_id or result.get("turnId") != self._turn_id:
            self._unsafe_protocol_event = True
            raise CodexIdempotencyLedgerError("Codex steer acknowledgement turn mismatch.")

    async def submit_tool_results(self, messages: list[dict[str, Any]]) -> None:
        if not self._pending_tools:
            return
        results = _tool_results_by_id(messages)
        missing = [call_id for call_id in self._pending_tools if call_id not in results]
        if missing:
            raise CodexAppServerError(
                "Nanobot did not provide results for pending Codex tool calls: "
                + ", ".join(missing)
            )
        # Nanobot tools have already run by the time their result messages exist.
        # Mark the bridge before writing to stdin so a concurrently dead child
        # cannot turn a BrokenPipe into an unsafe replay of those side effects.
        self._submitted_tool_results = True
        for call_id, request_id in list(self._pending_tools.items()):
            text = results[call_id]
            await self._submit_tool_result(
                call_id,
                request_id,
                text,
                success=not _looks_like_tool_error(text),
            )

    async def submit_cached_tool_result(
        self,
        call_id: str,
        content: Any,
        *,
        success: bool,
    ) -> None:
        request_id = self._pending_tools.get(call_id)
        if request_id is None:
            raise CodexAppServerError(f"No pending Codex tool call named {call_id}.")
        self._submitted_tool_results = True
        await self._submit_tool_result(call_id, request_id, content, success=success)

    async def _submit_tool_result(
        self,
        call_id: str,
        request_id: Any,
        content: Any,
        *,
        success: bool,
    ) -> None:
        await self._send(
            {
                "id": request_id,
                "result": {
                    "contentItems": _tool_result_content_items(content),
                    "success": success,
                },
            }
        )
        self._pending_tools.pop(call_id, None)

    def _check_event_owner(self, method: Any, params: dict[str, Any]) -> None:
        """先核验归属，再处理工具、流、用量、压缩或完成；未知归属安全停止。"""
        if not isinstance(method, str) or not (
            method.startswith(("item/", "turn/")) or method == "thread/tokenUsage/updated"
        ):
            return
        expected_thread = self.native_compaction.thread_id
        expected_turn = self._turn_id
        actual_thread = params.get("threadId")
        turn = params.get("turn")
        actual_turn = params.get("turnId")
        if actual_turn is None and isinstance(turn, dict):
            actual_turn = turn.get("id")
        if (
            expected_thread is None or expected_turn is None
            or actual_thread != expected_thread or actual_turn != expected_turn
            or (isinstance(turn, dict) and turn.get("id") != expected_turn)
        ):
            self._unsafe_protocol_event = True
            raise CodexAppServerError(
                "Codex event ownership mismatch: "
                f"method={method} expected_thread={expected_thread} actual_thread={actual_thread} "
                f"expected_turn={expected_turn} actual_turn={actual_turn}.",
                code="event_owner_mismatch",
                retryable=False,
            )

    async def next_response(
        self,
        *,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_text: str | None = None
        while True:
            message = await self._next_message()
            method = message.get("method")
            raw_params = message.get("params")
            params = raw_params if isinstance(raw_params, dict) else {}
            self._check_event_owner(method, params)
            if method == "item/agentMessage/delta":
                delta = params.get("delta")
                if isinstance(delta, str) and delta:
                    content_parts.append(delta)
                    if on_content_delta:
                        self._streamed_output = True
                        await on_content_delta(delta)
                continue
            if method in {
                "item/reasoning/summaryTextDelta",
                "item/reasoning/textDelta",
            }:
                delta = params.get("delta")
                if isinstance(delta, str) and delta:
                    reasoning_parts.append(delta)
                    if on_thinking_delta:
                        self._streamed_output = True
                        await on_thinking_delta(delta)
                continue
            if self.native_compaction.observe(method, params):
                # 压缩前的 last 不得继续冒充压缩后的窗口；等待新的 usage 通知。
                self._context_input_tokens = None
                continue
            if method == "thread/tokenUsage/updated":
                if params.get("threadId") not in {None, self.native_compaction.thread_id}:
                    continue
                self._last_usage = _map_token_usage(params)
                token_usage = params.get("tokenUsage")
                last = token_usage.get("last") if isinstance(token_usage, dict) else None
                tokens = last.get("inputTokens") if isinstance(last, dict) else None
                self._context_input_tokens = tokens if isinstance(tokens, int) and tokens >= 0 else None
                continue
            if method == "item/commandExecution/requestApproval" and "id" in message:
                # Any approval request means the command needs permissions beyond the
                # non-interactive workspace sandbox (including network access). Fail closed.
                await self._send({"id": message["id"], "result": {"decision": "decline"}})
                self._native_command_approvals_declined += 1
                continue
            if method == "item/fileChange/requestApproval" and "id" in message:
                # The bridge is non-interactive. workspace-write may proceed without an
                # approval, but requests for broader roots must fail closed instead of
                # hanging the app-server waiting for input that can never arrive.
                await self._send({"id": message["id"], "result": {"decision": "decline"}})
                self._native_file_change_approvals_declined += 1
                continue
            if method in {"item/started", "item/completed"}:
                item = params.get("item")
                item_type = item.get("type") if isinstance(item, dict) else None
                if item_type == "commandExecution":
                    self._record_native_command(item, completed=method == "item/completed")
                    continue
                if item_type == "fileChange":
                    self._record_native_file_change(item, completed=method == "item/completed")
                    continue
                if item_type == "imageView":
                    if method == "item/completed":
                        continue
                    call = self._bridge_image_view(item)
                    if on_tool_call_delta:
                        await on_tool_call_delta(
                            {"id": call.id, "name": call.name, "arguments": call.arguments}
                        )
                    diagnostics = self._provider_diagnostics()
                    diagnostics["native_image_view_bridged"] = True
                    return LLMResponse(
                        content="".join(content_parts) or None,
                        tool_calls=[call],
                        finish_reason="tool_calls",
                        usage=self._take_usage_delta(),
                        reasoning_content="".join(reasoning_parts) or None,
                        provider_diagnostics=diagnostics,
                    )
                if item_type in _NATIVE_TOOL_ITEM_TYPES:
                    # 协作通知不是执行前审批：子线程可能已经启动，禁止纠偏重放。
                    if item_type == "collabAgentToolCall":
                        self._unsafe_protocol_event = True
                    raise CodexAppServerError(
                        f"Codex app-server attempted a native tool outside nanobot: {item_type}.",
                        code="native_tool_blocked",
                        retryable=False,
                    )
            if method == "item/completed":
                item = params.get("item")
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    text = item.get("text")
                    if isinstance(text, str):
                        final_text = text
                continue
            if method == "item/tool/call" and "id" in message:
                call = _tool_call_from_server_request(message)
                self._pending_tools[call.id] = message["id"]
                if on_tool_call_delta:
                    await on_tool_call_delta(
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                    )
                return LLMResponse(
                    content="".join(content_parts) or None,
                    tool_calls=[call],
                    finish_reason="tool_calls",
                    usage=self._take_usage_delta(),
                    reasoning_content="".join(reasoning_parts) or None,
                    provider_diagnostics=self._provider_diagnostics(),
                )

            if method == "turn/completed":
                raw_turn = params.get("turn")
                turn = raw_turn if isinstance(raw_turn, dict) else {}
                if turn.get("status") != "completed":
                    raise _turn_failure(turn)
                content = "".join(content_parts)
                if not content and final_text:
                    content = final_text
                    if on_content_delta:
                        self._streamed_output = True
                        await on_content_delta(content)
                return LLMResponse(
                    content=content or None,
                    finish_reason="stop",
                    usage=self._take_usage_delta(),
                    reasoning_content="".join(reasoning_parts) or None,
                    provider_diagnostics=self._provider_diagnostics(),
                )
            if "id" in message and isinstance(method, str):
                await self._send(
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": "Nanobot only handles dynamic tool calls.",
                        },
                    }
                )

    def _bridge_image_view(self, item: dict[str, Any]) -> ToolCallRequest:
        """Translate a native imageView event into Nanobot's guarded read_file tool."""
        if "read_file" not in self._dynamic_tool_names:
            raise CodexAppServerError(
                "Codex app-server requested imageView, but Nanobot read_file is unavailable.",
                code="native_tool_blocked",
                retryable=False,
            )
        path = next(
            (
                item.get(field)
                for field in ("path", "imagePath", "filePath", "image_path")
                if isinstance(item.get(field), str) and item.get(field).strip()
            ),
            None,
        )
        if path is None:
            raise CodexAppServerError(
                "Codex app-server imageView request is missing an image path.",
                code="native_tool_blocked",
                retryable=False,
            )
        item_id = item.get("id")
        suffix = item_id if isinstance(item_id, str) and item_id else uuid.uuid4().hex
        return ToolCallRequest(
            id=f"native-image-view-{suffix}",
            name="read_file",
            arguments={"path": path},
        )

    def _record_native_file_change(self, item: dict[str, Any], *, completed: bool) -> None:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            item_id = f"anonymous-{len(self._native_file_changes) + 1}"
        status = item.get("status")
        if not isinstance(status, str) or not status:
            status = "completed" if completed else "inProgress"
        self._native_file_changes[item_id] = status

    def _record_native_command(self, item: dict[str, Any], *, completed: bool) -> None:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            item_id = f"anonymous-{len(self._native_command_executions) + 1}"
        status = item.get("status")
        if not isinstance(status, str) or not status:
            status = "completed" if completed else "inProgress"
        self._native_command_executions[item_id] = status

    def _provider_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {"transport": "codex_app_server"}
        if self.native_compaction.budget > 0:
            diagnostics.update(self.native_compaction.diagnostics())
        if self._context_input_tokens is not None:
            diagnostics["context_input_tokens"] = self._context_input_tokens
        if self._native_file_changes:
            status_counts: dict[str, int] = {}
            for status in self._native_file_changes.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            diagnostics["native_file_changes"] = {
                "count": len(self._native_file_changes),
                "statuses": status_counts,
                "approval_requests_declined": self._native_file_change_approvals_declined,
            }
        if self._native_command_executions:
            status_counts = {}
            for status in self._native_command_executions.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            diagnostics["native_command_executions"] = {
                "count": len(self._native_command_executions),
                "statuses": status_counts,
                "approval_requests_declined": self._native_command_approvals_declined,
            }
        return diagnostics

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self.process
        self.process = None
        if process is None:
            return
        stdout_task: asyncio.Task[bytes] | None = None
        if process.stdin is not None:
            if not process.stdin.is_closing():
                process.stdin.close()
            with suppress(
                BrokenPipeError,
                ConnectionResetError,
                RuntimeError,
                ValueError,
            ):
                await process.stdin.wait_closed()
        if process.stdout is not None and not process.stdout.at_eof():
            stdout_task = asyncio.create_task(process.stdout.read())
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        reader_tasks = [task for task in (stdout_task, self._stderr_task) if task is not None]
        if reader_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*reader_tasks, return_exceptions=True),
                    timeout=1,
                )
            except asyncio.TimeoutError:
                for task in reader_tasks:
                    task.cancel()
                await asyncio.gather(*reader_tasks, return_exceptions=True)
        from nanobot.agent.tools.process_tree import close_subprocess_transport

        await close_subprocess_transport(process)

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_rpc_id
        self._next_rpc_id += 1
        await self._send({"method": method, "id": request_id, "params": params})
        while True:
            message = await self._read_message(timeout_s=self.rpc_timeout_s, phase=method)
            if message.get("id") == request_id and "method" not in message:
                error = message.get("error")
                if isinstance(error, dict):
                    raise CodexAppServerError(
                        str(error.get("message") or f"{method} failed"),
                        code=str(error.get("code")) if error.get("code") is not None else None,
                    )
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            self._messages.append(message)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"method": method, "params": params})

    async def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise CodexAppServerError("Codex app-server process is not running.")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        process.stdin.write(payload.encode("utf-8"))
        try:
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise CodexAppServerError("Codex app-server stdin closed unexpectedly.") from exc

    async def _next_message(self) -> dict[str, Any]:
        return (
            self._messages.popleft()
            if self._messages
            else await self._read_message(
                timeout_s=self.event_timeout_s,
                phase="turn event",
            )
        )

    async def _read_message(
        self,
        *,
        timeout_s: float,
        phase: str,
    ) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise CodexAppServerError("Codex app-server stdout is unavailable.")
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                raise CodexAppServerError(
                    f"Codex app-server timed out waiting for {phase} after {timeout_s:g}s.",
                    code="app_server_timeout",
                    retryable=True,
                ) from exc
            if not line:
                code = await process.wait()
                if self._stderr_task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
                stderr = self.stderr_tail
                detail = f" Stderr: {stderr}" if stderr else ""
                raise CodexAppServerError(
                    f"Codex app-server exited unexpectedly with code {code}.{detail}",
                    retryable=True,
                )
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON Codex app-server stdout line")
                continue
            if isinstance(message, dict):
                return message

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while True:
            line = await stderr.readline()
            if not line:
                return
            text = _redact_diagnostic(line.decode("utf-8", errors="replace").strip())
            if text:
                self._stderr_lines.append(text[:1000])

    def _take_usage_delta(self) -> dict[str, int]:
        current = dict(self._last_usage)
        delta = _usage_delta(current, self._reported_usage)
        if current:
            self._reported_usage = current
        return delta


class OpenAICodexProvider(LegacyOpenAICodexProvider):
    """OpenAI Codex provider using the official app-server transport stack."""

    supports_stream_recover_callback = True
    supports_native_context_compaction = True

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.1-codex",
        proxy: str | None = None,
        idempotency_dir: Path | None = None,
    ):
        super().__init__(default_model=default_model, proxy=proxy)
        self._app_server_command: list[str] | None = None
        self._app_server_env = _codex_subprocess_env(self.proxy)
        self._rpc_timeout_s = _positive_float_env(
            CODEX_RPC_TIMEOUT_ENV,
            DEFAULT_CODEX_RPC_TIMEOUT_S,
        )
        self._event_timeout_s = _positive_float_env(
            CODEX_EVENT_TIMEOUT_ENV,
            DEFAULT_CODEX_EVENT_TIMEOUT_S,
        )
        self._idempotency_dir = idempotency_dir
        self._turns: dict[tuple[str, str], _CodexAppServerTurn] = {}
        self._turn_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def aclose(self) -> None:
        """Close every app-server bridge still owned by this provider."""
        bridges = list(dict.fromkeys(self._turns.values()))
        self._turns.clear()
        self._turn_locks.clear()
        if bridges:
            await asyncio.gather(*(bridge.close() for bridge in bridges), return_exceptions=True)

    async def aclose_execution(self, execution_id: str) -> None:
        """子任务结束或取消时，只释放它的连接，不触碰主会话及其他执行者。"""
        key = _execution_turn_key(execution_id)
        bridge = self._turns.get(key)
        if bridge is not None:
            await self._finish_turn(key, bridge)
        else:
            self._turn_locks.pop(key, None)

    async def _call_codex(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        request_context: dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_stream_recover: Callable[[], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        key = _turn_key(request_context)
        lock = self._turn_locks.setdefault(key, asyncio.Lock())
        async with lock:
            turn_messages = _current_turn_messages(messages)
            try:
                native_budget = native_input_budget(self, request_context)
                ledger = await asyncio.to_thread(
                    _idempotency_ledger,
                    key,
                    root_override=self._idempotency_dir,
                )
                await asyncio.to_thread(ledger.record_messages, turn_messages)
                bridge = self._turns.get(key)
                if bridge is not None:
                    pending_ids = set(bridge._pending_tools) - _tool_results_by_id(turn_messages).keys()
                    if pending_ids:
                        await asyncio.to_thread(
                            ledger.record_messages,
                            pending_checkpoint_messages(messages, pending_ids),
                        )
            except Exception as exc:
                bridge = self._turns.get(key)
                if bridge is not None:
                    await self._finish_turn(key, bridge)
                else:
                    self._turn_locks.pop(key, None)
                return _app_server_error_response(exc, retry_allowed=False)
            recovering = (
                self._turns.get(key) is None
                and ledger.has_entries
                and bool(_tool_results_by_id(turn_messages))
            )
            native_image_recovery = recovering and _last_tool_result_is_native_image_view(
                turn_messages
            )
            strict_recovery = recovering and not native_image_recovery
            restored_from_ledger = recovering
            recovery_attempts = 0
            native_tool_corrections = 0
            command_refresh_attempts = 0
            replayed_results = 0
            active_messages = messages
            context_settings = (
                model or self.default_model, reasoning_effort, tools, tool_choice, native_budget,
            )
            context_rebased = False
            checkpoint_continuation = False
            context_sync = "append" if self._turns.get(key) is not None else "start"
            while True:
                bridge = self._turns.get(key)
                try:
                    if bridge is not None and bridge.context_checkpoint.needs_rebase(
                        active_messages, context_settings,
                    ):
                        # 只在工具已落盘、当前调用尚未续传的边界重建；不清账本、不重跑工具。
                        pending = set(bridge._pending_tools)
                        results = _tool_results_by_id(active_messages)
                        recorded = {entry["call_id"] for entry in ledger.entries}
                        if not pending or not pending <= results.keys() or not pending <= recorded:
                            raise CodexIdempotencyLedgerError(
                                "Cannot rebase Codex context without checkpointed pending results."
                            )
                        if bridge.context_checkpoint.can_append(active_messages, context_settings):
                            # 前缀与配置稳定时只注入新增消息，不重建并重新计量全量历史。
                            suffix = active_messages[len(bridge.context_checkpoint.messages):]
                            # 先核验全部新增输入预算，再发送 steer；结果尚未提交，不触发重复执行。
                            check_native_payload(
                                self, model or self.default_model, suffix, [],
                                max(1, native_budget // 5) if native_budget > 0 else 0,
                            )
                            await bridge.steer(suffix)
                            context_sync = "steered"
                        else:
                            if bridge._native_file_changes or bridge._native_command_executions:
                                raise CodexIdempotencyLedgerError(
                                    "Cannot safely rebase Codex context after native side effects; "
                                    "native operations are not covered by the dynamic-tool ledger."
                                )
                            await self._finish_turn(key, bridge, drop_lock=False)
                            bridge = None
                            recovering = strict_recovery = True
                            native_image_recovery = False
                            context_rebased = True
                            checkpoint_continuation = True
                    if bridge is None:
                        check_native_payload(
                            self, model or self.default_model, active_messages, tools, native_budget,
                        )
                        await self._close_stale_session_turns(key)
                        if self._app_server_command is None:
                            self._app_server_command = _resolve_codex_app_server_command()
                        bridge = _CodexAppServerTurn(
                            list(self._app_server_command),
                            env=self._app_server_env,
                            rpc_timeout_s=self._rpc_timeout_s,
                            event_timeout_s=self._event_timeout_s,
                        )
                        bridge.native_compaction.budget = native_budget
                        bridge.checkpoint_continuation = checkpoint_continuation
                        self._turns[key] = bridge
                        if strict_recovery:
                            ledger.begin_recovery()
                        elif native_image_recovery:
                            ledger.begin_native_image_recovery()
                        await bridge.start(
                            model=model or self.default_model,
                            reasoning_effort=reasoning_effort,
                            messages=active_messages,
                            tools=tools,
                            tool_choice=tool_choice,
                            recovery=recovering,
                        )
                    else:
                        if bridge.recovery_replay_index is not None:
                            strict_recovery = True
                            ledger._replay_index = bridge.recovery_replay_index
                        pending_messages = [
                            message for message in active_messages
                            if message.get("role") == "tool"
                            and message.get("tool_call_id") in bridge._pending_tools
                        ]
                        check_native_payload(
                            self, model or self.default_model, pending_messages, [],
                            max(1, native_budget // 5) if native_budget > 0 else 0,
                        )
                        await bridge.submit_tool_results(active_messages)
                    bridge.context_checkpoint.capture(active_messages, context_settings)
                    response, replay_count = await self._next_response_with_replay(
                        bridge,
                        ledger,
                        recovering=strict_recovery,
                        native_image_recovery=native_image_recovery,
                        on_content_delta=on_content_delta,
                        on_thinking_delta=on_thinking_delta,
                        on_tool_call_delta=on_tool_call_delta,
                    )
                    replayed_results += replay_count
                    if strict_recovery:
                        bridge.recovery_replay_index = ledger._replay_index
                    diagnostics = dict(response.provider_diagnostics or {})
                    diagnostics["context_sync"] = (
                        "rebased" if context_rebased else "recovered" if recovering else context_sync
                    )
                    if context_rebased:
                        diagnostics["context_rebased"] = True
                    if bridge.checkpoint_continuation:
                        diagnostics["checkpoint_continuation"] = True
                    if recovery_attempts:
                        diagnostics["bridge_recovery_attempts"] = recovery_attempts
                    if command_refresh_attempts:
                        diagnostics["app_server_command_refreshes"] = command_refresh_attempts
                    if restored_from_ledger:
                        diagnostics["restored_from_idempotency_ledger"] = True
                    if replayed_results:
                        diagnostics["idempotent_tool_replays"] = replayed_results
                    response.provider_diagnostics = diagnostics
                    if diagnostics.get("native_image_view_bridged"):
                        # imageView is a native Codex operation and has no dynamic-tool
                        # request id to answer. Close this bridge; the next provider call
                        # rebuilds the turn with the guarded read_file result in history.
                        await self._finish_turn(key, bridge, drop_lock=False)
                    if not response.has_tool_calls:
                        await self._finish_turn(key, bridge)
                        await asyncio.to_thread(ledger.clear)
                    return response
                except asyncio.CancelledError:
                    if bridge is not None:
                        await self._finish_turn(key, bridge)
                    raise
                except Exception as exc:
                    submitted = bridge is not None and bridge.submitted_tool_results
                    streamed = bridge is not None and bridge.streamed_output
                    stderr_tail = bridge.stderr_tail if bridge is not None else None
                    native_unsafe = bridge is not None and (
                        bridge._unsafe_protocol_event
                        or bridge._steered_input
                        or bridge.native_compaction.in_progress
                        or bool(bridge._native_file_changes or bridge._native_command_executions)
                    )
                    if bridge is not None:
                        await self._finish_turn(key, bridge, drop_lock=False)
                    can_refresh_command = (
                        isinstance(exc, FileNotFoundError)
                        and bridge is not None
                        and bridge.process is None
                        and command_refresh_attempts < DEFAULT_COMMAND_REFRESH_ATTEMPTS
                        and not submitted
                        and not streamed
                    )
                    if can_refresh_command:
                        command_refresh_attempts += 1
                        if self._app_server_command == bridge.command:
                            self._app_server_command = None
                        logger.warning(
                            "Re-resolving Codex app-server command after launch path vanished; "
                            "attempt={}",
                            command_refresh_attempts,
                        )
                        continue
                    can_correct_native_tool = (
                        isinstance(exc, CodexAppServerError)
                        and exc.code == "native_tool_blocked"
                        and not native_unsafe
                        and native_tool_corrections < DEFAULT_NATIVE_TOOL_CORRECTION_ATTEMPTS
                        and (not streamed or on_stream_recover is not None)
                    )
                    if can_correct_native_tool:
                        native_tool_corrections += 1
                        checkpoint_continuation = False
                        recovering = False
                        strict_recovery = False
                        native_image_recovery = False
                        active_messages = [
                            *messages,
                            {"role": "user", "content": _NATIVE_TOOL_CORRECTION},
                        ]
                        if streamed and on_stream_recover is not None:
                            try:
                                await on_stream_recover()
                            except Exception as callback_exc:
                                self._turn_locks.pop(key, None)
                                return _app_server_error_response(
                                    callback_exc,
                                    retry_allowed=False,
                                    stderr_tail=stderr_tail,
                                )
                        logger.warning(
                            "Restarting Codex turn after blocked native tool; attempt={}",
                            native_tool_corrections,
                        )
                        continue
                    can_recover = (
                        submitted
                        and not native_unsafe
                        and ledger.has_entries
                        and _is_recoverable_bridge_error(exc)
                        and recovery_attempts < DEFAULT_CODEX_RECOVERY_ATTEMPTS
                        and (not streamed or on_stream_recover is not None)
                    )
                    if can_recover:
                        recovery_attempts += 1
                        checkpoint_continuation = False
                        recovering = True
                        strict_recovery = True
                        native_image_recovery = False
                        if streamed and on_stream_recover is not None:
                            try:
                                await on_stream_recover()
                            except Exception as callback_exc:
                                self._turn_locks.pop(key, None)
                                return _app_server_error_response(
                                    callback_exc,
                                    retry_allowed=False,
                                    stderr_tail=stderr_tail,
                                )
                        logger.warning(
                            "Rebuilding Codex bridge after tool result; attempt={}",
                            recovery_attempts,
                        )
                        continue
                    self._turn_locks.pop(key, None)
                    response = _app_server_error_response(
                        exc,
                        retry_allowed=(
                            not submitted and not native_unsafe
                            and not isinstance(exc, CodexIdempotencyLedgerError)
                        ),
                        stderr_tail=stderr_tail,
                    )
                    if bridge is not None:
                        if native_budget > 0:
                            response.provider_diagnostics.update(bridge.native_compaction.diagnostics())
                        response.provider_diagnostics["native_recovery_suppressed"] = native_unsafe
                        response.provider_diagnostics["unsafe_protocol_event"] = bridge._unsafe_protocol_event
                    return response

    async def _next_response_with_replay(
        self,
        bridge: _CodexAppServerTurn,
        ledger: _ToolResultLedger,
        *,
        recovering: bool,
        native_image_recovery: bool,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> tuple[LLMResponse, int]:
        replayed = 0
        replay_usage: dict[str, int] = {}
        while True:
            response = await bridge.next_response(
                on_content_delta=on_content_delta,
                on_thinking_delta=on_thinking_delta,
                on_tool_call_delta=(
                    None if recovering or native_image_recovery else on_tool_call_delta
                ),
            )
            if not response.has_tool_calls:
                response.usage = _sum_usage(replay_usage, response.usage)
                return response, replayed
            if native_image_recovery and response.provider_diagnostics.get(
                "native_image_view_bridged"
            ):
                call = response.tool_calls[0]
                cached = ledger.cached_native_image_result(call)
                if cached is not None:
                    replay_usage = _sum_usage(replay_usage, response.usage)
                    replayed += 1
                    continue
                response.usage = _sum_usage(replay_usage, response.usage)
                if on_tool_call_delta is not None:
                    await on_tool_call_delta(
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                    )
                return response, replayed
            if not recovering:
                response.usage = _sum_usage(replay_usage, response.usage)
                return response, replayed
            if len(response.tool_calls) != 1:
                raise CodexIdempotencyLedgerError(
                    "Unexpected batched tool calls during Codex bridge recovery."
                )
            call = response.tool_calls[0]
            cached = ledger.cached_result(
                call, checkpoint_continuation=bridge.checkpoint_continuation,
            )
            if cached is None:
                response.usage = _sum_usage(replay_usage, response.usage)
                if on_tool_call_delta is not None:
                    await on_tool_call_delta(
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                    )
                return response, replayed
            content, success = cached
            replay_usage = _sum_usage(replay_usage, response.usage)
            if response.provider_diagnostics.get("native_image_view_bridged"):
                # imageView is a native event rather than a dynamic-tool RPC, so there is
                # no pending request id to answer. The matching ledger entry proves that
                # Nanobot already authorized and read this exact path. Ignore the repeated
                # native lifecycle event and continue consuming this recovery turn.
                replayed += 1
                continue
            await bridge.submit_cached_tool_result(
                call.id,
                content,
                success=success,
            )
            replayed += 1

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        request_context: dict[str, Any] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_stream_recover: Callable[[], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature
        return await self._call_codex(
            messages,
            tools,
            model,
            reasoning_effort,
            tool_choice,
            request_context=request_context,
            on_content_delta=on_content_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
            on_stream_recover=on_stream_recover,
        )

    async def _close_stale_session_turns(self, key: tuple[str, str]) -> None:
        session_key, _turn_id = key
        stale = [
            (other_key, bridge)
            for other_key, bridge in self._turns.items()
            if other_key[0] == session_key and other_key != key
        ]
        for other_key, bridge in stale:
            self._turns.pop(other_key, None)
            self._turn_locks.pop(other_key, None)
            await bridge.close()

    async def _finish_turn(
        self,
        key: tuple[str, str],
        bridge: _CodexAppServerTurn,
        *,
        drop_lock: bool = True,
    ) -> None:
        if self._turns.get(key) is bridge:
            self._turns.pop(key, None)
        if drop_lock:
            self._turn_locks.pop(key, None)
        await bridge.close()


CodexAppServerProvider = OpenAICodexProvider


def _resolve_codex_app_server_command() -> list[str]:
    configured = os.environ.get(CODEX_BIN_ENV, "").strip()
    candidate = Path(configured).expanduser() if configured else None
    if candidate is None:
        found = shutil.which("codex.exe" if os.name == "nt" else "codex")
        candidate = Path(found) if found else None
    if candidate is None and os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            root = Path(local_app_data) / "OpenAI" / "Codex" / "bin"
            matches = sorted(
                root.glob("*/codex.exe"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            candidate = matches[0] if matches else None
    if candidate is None:
        found = shutil.which("codex")
        candidate = Path(found) if found else None
    if candidate is None or not candidate.exists():
        raise RuntimeError(
            "Official Codex CLI was not found. Install Codex or set "
            f"{CODEX_BIN_ENV} to the codex executable."
        )
    if os.name == "nt" and candidate.suffix.lower() in {".cmd", ".bat"}:
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", str(candidate)]
    else:
        command = [str(candidate)]
    command.extend(["app-server", "--stdio"])
    command.extend(
        [
            "-c",
            'web_search="disabled"',
            "-c",
            "agents.enabled=false",
            "-c",
            "mcp_servers={}",
            "-c",
            "plugins={}",
        ]
    )
    for feature in _DISABLED_FEATURES:
        command.extend(["--disable", feature])
    return command


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid {}={}", name, raw)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive {}={}", name, raw)
        return default
    return value


def _codex_subprocess_env(proxy: str | None) -> dict[str, str] | None:
    if not proxy:
        return None
    env = dict(os.environ)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[name] = proxy
    return env


def _redact_diagnostic(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _strip_codex_model_prefix(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("openai-codex/") else model


def _execution_turn_key(execution_id: str) -> tuple[str, str]:
    identity = f"subagent:{execution_id}"
    return identity, identity


def _turn_key(request_context: dict[str, Any] | None) -> tuple[str, str]:
    # 话题键用于路由，不能作为并行执行者的连接所有权；主会话保持原有键格式。
    execution_id = current_execution_id()
    if execution_id is not None:
        return _execution_turn_key(execution_id)
    context = request_context or {}
    return (
        str(context.get("session_key") or "default"),
        str(context.get("turn_id") or "default"),
    )


def _convert_dynamic_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    namespace_tools: list[dict[str, Any]] = []
    for schema in tools or []:
        function = schema.get("function")
        source = function if isinstance(function, dict) else schema
        name = source.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = source.get("description")
        parameters = source.get("parameters")
        namespace_tools.append(
            {
                "type": "function",
                "name": name,
                "description": description if isinstance(description, str) else "",
                "inputSchema": parameters
                if isinstance(parameters, dict)
                else {
                    "type": "object",
                    "properties": {},
                },
            }
        )
    if not namespace_tools:
        return []
    return [
        {
            "type": "namespace",
            "name": "nanobot",
            "description": "Tools provided and executed by nanobot.",
            "tools": namespace_tools,
        }
    ]


def _apply_tool_choice(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if tool_choice is None or tool_choice == "auto":
        return tools, None
    if tool_choice == "none":
        return [], "Do not call any tool in this turn."
    if tool_choice == "required":
        return tools, "Call at least one provided dynamic tool before answering."
    if not isinstance(tool_choice, dict):
        return tools, None
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
    if not isinstance(name, str) or not name:
        return tools, None
    selected = [tool for tool in tools or [] if _tool_schema_name(tool) == name]
    return selected, f"Use only the provided dynamic tool named {name} when a tool is needed."


def _tool_schema_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    source = function if isinstance(function, dict) else tool
    name = source.get("name")
    return name if isinstance(name, str) and name else None


def _skill_requires_unavailable_tools(skill: dict[str, Any], tool_names: set[str]) -> bool:
    """Return whether a Codex skill depends on tools outside the dynamic namespace."""
    dependencies = skill.get("dependencies")
    dependency_tools = dependencies.get("tools", []) if isinstance(dependencies, dict) else []
    for dependency in dependency_tools:
        if not isinstance(dependency, dict):
            continue
        dependency_type = dependency.get("type")
        value = dependency.get("value")
        if not isinstance(value, str) or not value:
            continue
        if dependency_type == "mcp":
            prefix = f"mcp__{value}__".casefold()
            if not any(name.casefold().startswith(prefix) for name in tool_names):
                return True
        elif dependency_type == "tool" and value not in tool_names:
            return True

    path = skill.get("path")
    if not isinstance(path, str) or not path:
        return False
    try:
        markdown = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.debug("Could not inspect Codex skill dependencies: {}", path, exc_info=True)
        return False
    referenced_tools = set(_SKILL_DYNAMIC_TOOL_RE.findall(markdown))
    return bool(referenced_tools - tool_names)


def _messages_to_app_server_input(
    messages: list[dict[str, Any]],
    *,
    include_current_turn_history: bool = False,
    disabled_skill_paths: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    instruction_parts = [
        _message_content_text(message.get("content"))
        for message in messages
        if message.get("role") in {"system", "developer"}
    ]
    instructions = "\n\n".join(part for part in instruction_parts if part)
    if not instructions:
        instructions = "You are Nanobot, a helpful AI agent."

    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    last_user_index = user_indexes[-1] if user_indexes else None
    prior_messages = [
        message
        for index, message in enumerate(messages)
        if message.get("role") not in {"system", "developer"}
        and (last_user_index is None or index < last_user_index)
    ]
    omitted_tool_results = _disabled_skill_read_call_ids(
        messages,
        disabled_skill_paths or set(),
    )
    turn_input: list[dict[str, Any]] = []
    if prior_messages:
        turn_input.append(
            {
                "type": "text",
                "text": (
                    "Existing nanobot conversation transcript follows. Treat tool calls and "
                    "tool results as completed history; do not repeat them.\n\n"
                    + _render_transcript(
                        prior_messages,
                        omitted_tool_result_ids=omitted_tool_results,
                    )
                ),
            }
        )
    if last_user_index is not None:
        turn_input.extend(_content_to_user_inputs(messages[last_user_index].get("content")))
        current_turn_tail = messages[last_user_index + 1 :]
        if include_current_turn_history and current_turn_tail:
            turn_input.append(
                {
                    "type": "text",
                    "text": (
                        "Recovery checkpoint from this same nanobot turn follows. "
                        "Every listed tool call already ran and every tool result is "
                        "authoritative. Never execute those calls again; continue after "
                        "the final result.\n\n"
                        + _render_transcript(
                            current_turn_tail,
                            omitted_tool_result_ids=omitted_tool_results,
                        )
                    ),
                }
            )
    elif prior_messages:
        turn_input.append(
            {"type": "text", "text": "Continue from the final event in the transcript above."}
        )
    else:
        turn_input.append({"type": "text", "text": "(conversation continued)"})
    return instructions, turn_input


def _disabled_skill_read_call_ids(
    messages: list[dict[str, Any]],
    disabled_skill_paths: set[str],
) -> set[str]:
    """Find historical read_file calls that loaded a now-disabled Codex skill."""
    if not disabled_skill_paths:
        return set()
    disabled_path_keys = {_path_key(path) for path in disabled_skill_paths}
    call_ids: set[str] = set()
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or function.get("name") != "read_file":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            path = arguments.get("path") if isinstance(arguments, dict) else None
            call_id = tool_call.get("id")
            if (
                isinstance(path, str)
                and isinstance(call_id, str)
                and _path_key(path) in disabled_path_keys
            ):
                call_ids.add(call_id)
    return call_ids


def _path_key(path: str) -> str:
    portable_path = path.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normcase(os.path.abspath(os.path.normpath(portable_path)))


def _render_transcript(
    messages: list[dict[str, Any]],
    *,
    omitted_tool_result_ids: set[str] | None = None,
) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = _message_content_text(message.get("content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            rendered = json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
            content = f"{content}\nTool calls: {rendered}".strip()
        call_id = message.get("tool_call_id")
        if (
            role == "tool"
            and isinstance(call_id, str)
            and call_id in (omitted_tool_result_ids or set())
        ):
            content = (
                "[omitted: this result contained instructions from a Codex skill that is "
                "disabled because its required tools are unavailable]"
            )
        suffix = f" [{call_id}]" if isinstance(call_id, str) and call_id else ""
        lines.append(f"{role}{suffix}: {content}")
    return "\n".join(lines)


def _content_to_user_inputs(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": _message_content_text(content)}]
    inputs: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            inputs.append({"type": "text", "text": str(part)})
            continue
        if part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
            inputs.append({"type": "text", "text": part["text"]})
            continue
        if part.get("type") in {"image_url", "input_image"}:
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            url = url or part.get("url")
            if isinstance(url, str):
                inputs.append({"type": "image", "url": url})
                continue
        inputs.append(
            {"type": "text", "text": json.dumps(part, ensure_ascii=False, separators=(",", ":"))}
        )
    return inputs or [{"type": "text", "text": "(empty user message)"}]


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            else:
                parts.append(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(parts)
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return str(content)


def _tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            results[call_id] = message.get("content")
    return results


def _last_tool_result_is_native_image_view(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        return isinstance(call_id, str) and call_id.startswith("native-image-view-")
    return False


def _tool_result_content_items(content: Any) -> list[dict[str, str]]:
    """Map a Nanobot tool result to Codex dynamic-tool multimodal content items."""
    parts = content if isinstance(content, list) else [content]
    items: list[dict[str, str]] = []
    for part in parts:
        if isinstance(part, str):
            items.append({"type": "inputText", "text": part})
            continue
        if not isinstance(part, dict):
            items.append({"type": "inputText", "text": str(part)})
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text", "inputText"} and isinstance(part.get("text"), str):
            items.append({"type": "inputText", "text": part["text"]})
            continue
        if part_type in {"image_url", "input_image", "inputImage"}:
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            url = url or part.get("imageUrl") or part.get("url")
            if isinstance(url, str) and url:
                items.append({"type": "inputImage", "imageUrl": url})
                continue
        if part_type in {"audio_url", "input_audio", "inputAudio"}:
            audio = part.get("audio_url")
            url = audio.get("url") if isinstance(audio, dict) else audio
            url = url or part.get("audioUrl") or part.get("url")
            if isinstance(url, str) and url:
                items.append({"type": "inputAudio", "audioUrl": url})
                continue
        items.append(
            {
                "type": "inputText",
                "text": json.dumps(part, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return items or [{"type": "inputText", "text": ""}]


def _serialize_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(
        {"nanobot_multimodal_tool_result": content},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _tool_result_content_type(content: Any) -> str:
    if isinstance(content, str):
        return "text/plain"
    if isinstance(content, list):
        kinds = {str(item.get("type", "")) for item in content if isinstance(item, dict)}
        if any(kind in {"image_url", "input_image", "inputImage"} for kind in kinds):
            return "multipart/mixed; image"
        if any(kind in {"audio_url", "input_audio", "inputAudio"} for kind in kinds):
            return "multipart/mixed; audio"
        return "multipart/mixed"
    return "application/json"


def _tool_result_summary(content: Any) -> str:
    """Return a bounded, non-binary manifest summary for diagnostics and inspection."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        kinds: list[str] = []
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                kinds.append(type(item).__name__)
                continue
            kind = str(item.get("type", "unknown"))
            kinds.append(kind)
            if kind in {"text", "input_text", "inputText"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        prefix = f"{len(content)} multimodal items ({', '.join(kinds[:8])})"
        text = prefix + ((": " + " ".join(text_parts)) if text_parts else "")
    else:
        try:
            text = json.dumps(content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(content)
    text = re.sub(r"\s+", " ", text).strip()
    return _redact_diagnostic(text)[:240]


def _deserialize_tool_result(content: Any) -> Any:
    if not isinstance(content, str):
        return content or ""
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return content
    if isinstance(value, dict) and "nanobot_multimodal_tool_result" in value:
        return value["nanobot_multimodal_tool_result"]
    return content


def _tool_calls_by_id(
    messages: list[dict[str, Any]],
) -> dict[str, tuple[str, Any]]:
    calls: dict[str, tuple[str, Any]] = {}
    for message in messages:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            source = function if isinstance(function, dict) else raw_call
            name = source.get("name")
            if not isinstance(call_id, str) or not call_id:
                continue
            if not isinstance(name, str) or not name:
                continue
            calls[call_id] = (name, source.get("arguments"))
    return calls


def _current_turn_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the current user turn, excluding tool traffic from older turns."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return messages[index:]
    return messages


def _canonical_tool_arguments(arguments: Any) -> str:
    value = arguments
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError:
            value = arguments
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _tool_signature(name: str, canonical_arguments: str) -> str:
    payload = f"{name}\0{canonical_arguments}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _idempotency_ledger(
    key: tuple[str, str],
    *,
    root_override: Path | None,
) -> _ToolResultLedger:
    scope = current_workspace_scope()
    workspace_path = scope.project_path if scope is not None else Path.cwd().resolve(strict=False)
    workspace = str(workspace_path.resolve(strict=False))
    if root_override is not None:
        root = root_override
    else:
        data_dir = (
            workspace_path
            if is_default_workspace(workspace_path)
            else get_workspace_cache_dir(workspace_path)
        )
        root = data_dir / "codex-app-server" / "idempotency"
    session_key, turn_id = key
    digest = hashlib.sha256(f"{workspace}\0{session_key}\0{turn_id}".encode("utf-8")).hexdigest()
    return _ToolResultLedger(
        root / f"{digest}.json",
        workspace=workspace,
        session_key=session_key,
        turn_id=turn_id,
    )


def _is_recoverable_bridge_error(exc: Exception) -> bool:
    return isinstance(exc, CodexAppServerError) and exc.retryable is True


def _looks_like_tool_error(content: Any) -> bool:
    text = _message_content_text(content)
    return text.lstrip().lower().startswith(("error:", "tool error:", "failed:"))


def _tool_call_from_server_request(message: dict[str, Any]) -> ToolCallRequest:
    raw_params = message.get("params")
    params = raw_params if isinstance(raw_params, dict) else {}
    call_id = params.get("callId")
    name = params.get("tool")
    if not isinstance(call_id, str) or not call_id:
        raise CodexAppServerError("Codex dynamic tool request is missing callId.")
    if not isinstance(name, str) or not name:
        raise CodexAppServerError("Codex dynamic tool request is missing tool name.")
    arguments = params.get("arguments")
    return ToolCallRequest(
        id=call_id,
        name=name,
        arguments=arguments if arguments is not None else {},
    )


def _map_token_usage(params: dict[str, Any]) -> dict[str, int]:
    token_usage = params.get("tokenUsage")
    total = token_usage.get("total") if isinstance(token_usage, dict) else None
    breakdown = (
        total
        if isinstance(total, dict)
        else (token_usage.get("last") if isinstance(token_usage, dict) else None)
    )
    if not isinstance(breakdown, dict):
        return {}
    usage = {
        "prompt_tokens": int(breakdown.get("inputTokens") or 0),
        "completion_tokens": int(breakdown.get("outputTokens") or 0),
        "total_tokens": int(breakdown.get("totalTokens") or 0),
    }
    cached = int(breakdown.get("cachedInputTokens") or 0)
    reasoning = int(breakdown.get("reasoningOutputTokens") or 0)
    if cached:
        usage["cached_tokens"] = cached
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


def _usage_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    if not current:
        return {}
    if any(current.get(key, 0) < previous.get(key, 0) for key in current):
        return dict(current)
    return {
        key: value - previous.get(key, 0)
        for key, value in current.items()
        if value - previous.get(key, 0) > 0
    }


def _sum_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _turn_failure(turn: dict[str, Any]) -> CodexAppServerError:
    raw_error = turn.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    message = str(error.get("message") or f"Codex turn ended with status {turn.get('status')}.")
    info = error.get("codexErrorInfo")
    code = info if isinstance(info, str) else None
    retryable = code in {"serverOverloaded", "internalServerError"} or (
        isinstance(info, dict)
        and any(
            key in info
            for key in {
                "httpConnectionFailed",
                "responseStreamConnectionFailed",
                "responseStreamDisconnected",
                "responseTooManyFailedAttempts",
            }
        )
    )
    return CodexAppServerError(message, code=code, retryable=retryable)


def _app_server_error_response(
    exc: Exception,
    *,
    retry_allowed: bool = True,
    stderr_tail: str | None = None,
) -> LLMResponse:
    if isinstance(exc, CodexAppServerError):
        code = exc.code
        retryable = exc.retryable
    else:
        code = type(exc).__name__
        retryable = isinstance(exc, (BrokenPipeError, ConnectionError, TimeoutError))
    retryable = bool(retryable and retry_allowed)
    summary = str(exc)
    if stderr_tail and stderr_tail not in summary:
        summary = f"{summary} Stderr: {stderr_tail}"
    logger.warning(
        "Codex app-server bridge failed: type={} code={} retryable={} summary={}",
        type(exc).__name__,
        code,
        retryable,
        summary[:500],
    )
    return LLMResponse(
        content=f"Error calling Codex app-server: {summary}",
        finish_reason="error",
        error_kind="connection" if retryable else "provider",
        error_type="codex_app_server_error",
        error_code=code,
        error_should_retry=retryable,
        provider_diagnostics={
            "transport": "codex_app_server",
            "retry_suppressed_after_tool_result": not retry_allowed,
        },
    )
