"""子执行的只读审计边界：复用 runtime JSONL writer，不写入父任务事件流。

parent_session_ref 是有效父 session key 的 SHA-256，可离线关联但不泄漏聊天标识。
每次执行独占日志文件；run_id/turn_id 是审计身份，provider_execution_id 仅观察现有
ContextVar，不设置或复用 provider 线程、路由、取消或停止凭据。
"""

from __future__ import annotations

import hashlib
import math
import uuid
from pathlib import Path
from typing import Any

from nanobot.fork.agent.execution_scope import current_execution_id
from nanobot.utils.session_runtime_log import append_session_runtime_log

_EVENTS = frozenset({
    "run.start", "run.end",
    "runner.iteration.start", "runner.context.governance",
    "runner.context.budget_exhausted", "runner.context.safety_stop",
    "runner.context.turn_summary", "runner.context_overflow.retry",
    "runner.model.request", "runner.model.request.done", "runner.model.response",
    "runner.model.retry", "runner.model.timeout", "runner.model.timeout_recovery",
    "runner.max_iterations.reached", "runner.max_iterations.finalization.skipped",
    "runner.tool_loop.stopped", "runner.tool_loop.warned",
})
_NUMBERS = frozenset({
    "iteration", "messages", "tools", "max_iterations", "model_requests",
    "local_projection_reduction_tokens", "local_projection_reduction_peak_tokens",
    "local_projection_reduction_token_observations", "context_input_peak_tokens",
    "context_input_estimated_peak_tokens", "context_input_measured_samples",
    "context_input_estimated_samples", "context_input_unknown_samples",
    "response_prompt_usage_peak_tokens", "projection_omitted_tool_results",
    "projection_changed_tool_results",
    "prompt_tokens", "completion_tokens", "prompt_peak_tokens",
    "context_input_tokens", "context_input_estimated", "governance_saved_total",
    "compacted_tool_results", "digested_tool_results", "saved_total", "context_version",
    "estimated_tokens", "target_tokens", "messages_before", "messages_after",
    "timeout_s", "duration_ms", "error_status_code", "error_retry_after_s",
    "repeat_count", "period", "attempt", "max_attempts", "delay_s", "retry_after_s",
})
_BOOLS = frozenset({
    "streaming", "progress_streaming", "should_execute_tools", "error_should_retry",
    "context_safety_failure", "finalize_enabled",
})
_ENUMS = {
    "context_scope": {"native_checkpoint_copy", "local_model_copy"},
    "projection_metrics_semantics": {"local_copy_not_remote_or_billing"},
    "governance_saved_total_semantics": {"repeated_local_token_observations"},
    "prompt_peak_tokens_semantics": {"response_usage_not_context_peak"},
    "compaction_owner": {"provider", "nanobot"},
    "strategy": {"native", "legacy", "transactional"},
    "metrics_source": {"local_estimate"},
    "estimate_source": {"provider", "tiktoken", "heuristic"},
    "finish_reason": {"stop", "end_turn", "length", "error", "tool_calls", "content_filter"},
    "error_kind": {"timeout", "connection", "rate_limit", "context_overflow", "auth", "quota", "transient"},
    "retry_mode": {"standard", "persistent"},
    "error_code": {"context_length_exceeded", "rate_limit_exceeded", "insufficient_quota",
                   "invalid_api_key", "server_error", "model_not_found"},
    "stop_reason": {"completed", "error", "tool_error", "cancelled", "max_iterations",
                    "tool_loop", "empty_final_response"},
    "outcome": {"completed", "error", "cancelled"},
    "reason": {"disabled"},
}
_USAGE = frozenset({
    "prompt_tokens", "completion_tokens", "total_tokens", "estimated_tokens", "provider_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens", "cached_tokens", "reasoning_tokens",
    "context_input_tokens", "context_input_estimated",
})
_GROUPS = frozenset({"system", "history", "tool_results", "tool_definitions", "total"})
_BUDGET = frozenset({
    "window_tokens", "output_reserve", "safety_reserve",
    "provider_budget", "block_limit", "input_tokens",
})
_PROVIDER_NUMBERS = frozenset({
    "context_input_tokens", "native_auto_compact_token_limit",
    "native_compactions_started", "native_compactions_completed",
    "bridge_recovery_attempts", "app_server_command_refreshes", "idempotent_tool_replays",
})
_PROVIDER_BOOLS = frozenset({
    "native_compaction_in_progress", "context_rebased", "checkpoint_continuation",
    "restored_from_idempotency_ledger", "native_recovery_suppressed",
    "retry_suppressed_after_tool_result", "native_image_view_bridged",
})
_PROVIDER_ENUMS = {
    "transport": {"codex_app_server"},
    "context_management": {"codex_native_auto"},
    "context_sync": {"fresh", "start", "steered", "append", "unchanged", "rebased", "recovered"},
}


def session_ref(key: str) -> str:
    """稳定关联值；不把 session key 当作路径片段或可见正文。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _safe_fields(
    fields: dict[str, Any], numbers: frozenset[str], booleans: frozenset[str],
    enums: dict[str, set[str]],
) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in numbers:
        if key in fields and (_number(fields[key]) or fields[key] is None):
            safe[key] = fields[key]
    for key in booleans:
        if type(fields.get(key)) is bool:
            safe[key] = fields[key]
    for key, allowed in enums.items():
        if key in fields:
            value = fields[key]
            safe[key] = value if type(value) is str and value in allowed else "other"
    return safe


class SubagentDiagnostics:
    """每个 Manager 执行创建一个实例；所有日志失败均不改变任务结果。"""

    def __init__(
        self, root: Path, *, task_id: str, parent_session_key: str, model: str,
        provider: Any,
    ) -> None:
        self.run_id = uuid.uuid4().hex
        self.turn_id = f"subagent:{self.run_id}"
        parent_ref = session_ref(parent_session_key)
        # spawn 产生的 task_id 保持可直接关联；非标准输入也不能穿越目录。
        safe_task_id = (
            task_id if len(task_id) == 32 and all(c in "0123456789abcdef" for c in task_id)
            else session_ref(task_id)
        )
        self.path = root / "subagent-runtime" / parent_ref / safe_task_id / self.run_id / "runtime.log"
        self._identity = {
            "scope": "subagent", "run_id": self.run_id, "turn_id": self.turn_id,
            "task_id": safe_task_id, "parent_session_ref": parent_ref,
            "model_ref": session_ref(model), "provider_type": type(provider).__name__,
        }
        self._provider_execution_id: str | None = None
        self._sequence = 0

    def __call__(self, event: str, fields: dict[str, Any]) -> None:
        try:
            if event not in _EVENTS:
                return
            execution_id = current_execution_id()
            if execution_id is not None:
                self._provider_execution_id = execution_id
            safe = _safe_fields(fields, _NUMBERS, _BOOLS, _ENUMS)
            for key, allowed in (
                ("usage", _USAGE), ("budget", _BUDGET),
                ("before", _GROUPS), ("after", _GROUPS), ("saved", _GROUPS),
            ):
                if type(fields.get(key)) is dict:
                    safe[key] = _safe_fields(fields[key], allowed, frozenset(), {})
            if type(fields.get("provider_diagnostics")) is dict:
                safe["provider_diagnostics"] = _safe_fields(
                    fields["provider_diagnostics"], _PROVIDER_NUMBERS,
                    _PROVIDER_BOOLS, _PROVIDER_ENUMS,
                )
            self._sequence += 1
            append_session_runtime_log(
                self.path, f"subagent.{event}", **safe, **self._identity,
                provider_execution_id=self._provider_execution_id, sequence=self._sequence,
            )
        except Exception:
            # 不输出原始异常或 traceback，其中可能含凭据和请求正文。
            return
