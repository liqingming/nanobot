"""Session metadata helpers for explicit sustained goals.

Tools set ``metadata[GOAL_STATE_KEY]``. Reads accept the legacy session key ``thread_goal``
for older sessions. Callers use ``goal_state_runtime_lines`` and ``goal_state_ws_blob`` without
importing tool implementations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, MutableMapping, cast

from nanobot.session.manager import SessionManager

GOAL_STATE_KEY = "goal_state"
GOAL_COMMAND = "/goal"
MAX_GOAL_OBJECTIVE_CHARS = 4000
# Older builds stored the same JSON blob under this key.
_LEGACY_GOAL_STATE_SESSION_KEY = "thread_goal"
_MAX_OBJECTIVE_WS = 600
_DEFAULT_LONG_GOAL_LLM_TIMEOUT_S = 1200.0


def _session_goal_raw(metadata: Mapping[str, Any] | None) -> Any:
    if not metadata:
        return None
    if GOAL_STATE_KEY in metadata:
        return metadata.get(GOAL_STATE_KEY)
    return metadata.get(_LEGACY_GOAL_STATE_SESSION_KEY)


def discard_legacy_goal_state_key(metadata: MutableMapping[str, Any]) -> None:
    """Remove legacy metadata key after migrating writes to :data:`GOAL_STATE_KEY`."""
    metadata.pop(_LEGACY_GOAL_STATE_SESSION_KEY, None)


def goal_state_raw(metadata: Mapping[str, Any] | None) -> Any:
    """Return the session goal blob under :data:`GOAL_STATE_KEY` or the legacy key."""
    return _session_goal_raw(metadata)


def sustained_goal_active(metadata: Mapping[str, Any] | None) -> bool:
    """True when this session has an active sustained objective."""
    goal = parse_goal_state(goal_state_raw(metadata))
    return isinstance(goal, dict) and goal.get("status") == "active"


def sustained_goal_waiting_for_user(metadata: Mapping[str, Any] | None) -> bool:
    """True when an active goal deliberately yielded for a real user reply."""
    goal = parse_goal_state(goal_state_raw(metadata))
    return isinstance(goal, dict) and goal.get("status") == "active" and bool(
        goal.get("awaiting_user_input")
    )


def clear_goal_waiting_for_user(metadata: MutableMapping[str, Any]) -> bool:
    """Resume an active goal after its next real inbound user message."""
    goal = parse_goal_state(goal_state_raw(metadata))
    if not isinstance(goal, dict) or goal.get("status") != "active":
        return False
    if not goal.get("awaiting_user_input"):
        return False
    metadata[GOAL_STATE_KEY] = {
        **goal,
        "awaiting_user_input": False,
        "awaiting_user_input_reason": "",
    }
    discard_legacy_goal_state_key(metadata)
    return True


def explicit_goal_resume_requested(
    content: str | None,
    message_metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether an inbound message explicitly resumes a waiting goal."""
    if message_metadata and message_metadata.get("goal_reply") is True:
        return True
    original_command = str(
        (message_metadata or {}).get("original_command") or ""
    ).strip().casefold()
    if original_command.split("@", 1)[0] == "/continue":
        return True
    first_token = str(content or "").strip().split(maxsplit=1)[0:1]
    return bool(first_token and first_token[0].casefold().split("@", 1)[0] == "/continue")


def explicit_goal_requested(message_metadata: Mapping[str, Any] | None) -> bool:
    """True when this turn was explicitly started by the ``/goal`` command."""
    if not message_metadata:
        return False
    if message_metadata.get("goal_requested") is True:
        return True
    return str(message_metadata.get("original_command") or "").strip() == GOAL_COMMAND


def sustained_goal_turn(
    metadata: Mapping[str, Any] | None,
    *,
    message_metadata: Mapping[str, Any] | None = None,
) -> bool:
    """True when this turn should use sustained-goal runtime limits."""
    return sustained_goal_active(metadata) or explicit_goal_requested(message_metadata)


def parse_goal_state(blob: Any) -> dict[str, Any] | None:
    if blob is None:
        return None
    if isinstance(blob, dict):
        return cast(dict[str, Any], blob)
    if isinstance(blob, str):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
    return None


def goal_state_runtime_lines(metadata: Mapping[str, Any] | None) -> list[str]:
    """Lines appended inside the Runtime Context block when a goal is active."""
    if not metadata:
        return []
    goal = parse_goal_state(_session_goal_raw(metadata))
    if not isinstance(goal, dict) or goal.get("status") != "active":
        return []
    objective = str(goal.get("objective") or "").strip()
    if not objective:
        return ["Goal: active (no objective text stored)."]
    if len(objective) > MAX_GOAL_OBJECTIVE_CHARS:
        objective = objective[:MAX_GOAL_OBJECTIVE_CHARS].rstrip() + "\n… (truncated)"
    out = ["Goal (active):", objective]
    hint = str(goal.get("ui_summary") or "").strip()
    if hint:
        out.append(f"Summary: {hint}")
    if goal.get("awaiting_user_input"):
        reason = str(goal.get("awaiting_user_input_reason") or "").strip()
        out.append(f"Waiting for user input: {reason or 'yes'}")
    return out


def goal_state_ws_blob(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """JSON-safe snapshot for WebSocket ``goal_state`` events (one chat_id per frame)."""
    goal = parse_goal_state(_session_goal_raw(metadata)) if metadata else None
    if isinstance(goal, dict) and goal.get("status") in {"active", "blocked"}:
        status = str(goal.get("status"))
        objective = str(goal.get("objective") or "").strip()
        if len(objective) > _MAX_OBJECTIVE_WS:
            objective = objective[:_MAX_OBJECTIVE_WS].rstrip() + "…"
        summary = str(goal.get("ui_summary") or "").strip()[:120]
        blob: dict[str, Any] = {"active": status == "active", "status": status}
        if summary:
            blob["ui_summary"] = summary
        if objective:
            blob["objective"] = objective
        recap = str(goal.get("recap") or "").strip()[:240]
        if recap:
            blob["recap"] = recap
        return blob
    return {"active": False}


def runner_wall_llm_timeout_s(
    sessions: SessionManager,
    session_key: str | None,
    *,
    metadata: Mapping[str, Any] | None = None,
    message_metadata: Mapping[str, Any] | None = None,
) -> float | None:
    """Wall-clock cap for :class:`~nanobot.agent.runner.AgentRunner` when streaming an LLM.

    Sustained-goal turns use a longer but finite wall-clock cap so non-terminal
    provider events cannot hold the session lock forever. ``None`` means use the
    ordinary ``NANOBOT_LLM_TIMEOUT_S`` default. Override the long-goal cap with
    ``NANOBOT_LONG_GOAL_LLM_TIMEOUT_S``; invalid/non-positive values fall back to 1200s.
    """
    meta: Mapping[str, Any] | None = metadata
    if meta is None and session_key:
        meta = sessions.get_or_create(session_key).metadata
    if not sustained_goal_turn(meta, message_metadata=message_metadata):
        return None
    raw = os.environ.get(
        "NANOBOT_LONG_GOAL_LLM_TIMEOUT_S",
        str(_DEFAULT_LONG_GOAL_LLM_TIMEOUT_S),
    ).strip()
    try:
        timeout_s = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_LONG_GOAL_LLM_TIMEOUT_S
    return timeout_s if timeout_s > 0 else _DEFAULT_LONG_GOAL_LLM_TIMEOUT_S
