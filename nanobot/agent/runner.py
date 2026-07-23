"""Shared execution loop for tool-using agents.

Fork additions:
  * ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` in
    the ``usage`` dict — prompt-cache stats from Anthropic-style APIs.
  * ``await hook.after_execute_tools(context)`` — fork hook used by
    learning / TurnSummary capture to observe tool execution outcomes.
  * Rephrased ``_DEFAULT_MAX_ITERATIONS_MESSAGE`` to mention
    ``/continue`` (fork command).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from nanobot.agent.context_artifacts import ToolDigest, ToolDigestBuilder
from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
)
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.utils.helpers import (
    IncrementalThinkExtractor,
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    extract_reasoning,
    strip_reasoning_tags,
    strip_think,
)
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_budget_exhausted_finalization_message,
    build_finalization_retry_message,
    build_goal_continue_message,
    build_length_recovery_message,
    is_blank_text,
    repeated_external_lookup_error,
    repeated_workspace_violation_error,
)

GoalContinueMessage = str | Callable[[], str | None]

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_ARREARAGE_ERROR_MESSAGE = (
    "The AI provider rejected the request because the API key is out of quota or the "
    "account is in arrears. Please top up / check the billing status of your API key and try again."
)
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_MAX_INJECTIONS_PER_TURN = 3
_MAX_INJECTION_CYCLES = 5
# A hook must not indefinitely prevent a model-approved tool call from starting.
_PRE_TOOL_TRANSITION_WATCHDOG_S = 30.0

# Fork: stop model/tool feedback loops before the high per-turn iteration budget
# is exhausted. Three identical short cycles trigger a correction; five cycles
# terminate the turn. Polling tools are excluded because repetition is expected.
_TOOL_LOOP_WARN_REPEATS = 3
_TOOL_LOOP_STOP_REPEATS = 5
_TOOL_LOOP_MAX_PERIOD = 8
_TOOL_LOOP_EXEMPT_TOOLS = frozenset({"list_dir", "list_exec_sessions", "write_stdin"})
_TOOL_LOOP_STOP_MESSAGE = (
    "检测到工具调用陷入无进展循环，已安全停止本轮：相同的工具调用序列在纠偏后仍重复。"
    "请缩小任务范围、提供新的判定依据，或调整方案后再继续。"
)

@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    max_tool_result_chars: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    workspace: Path | None = None
    data_dir: Path | None = None
    session_key: str | None = None
    context_window_tokens: int | None = None
    context_block_limit: int | None = None
    provider_retry_mode: str = "standard"
    progress_callback: Any | None = None
    stream_progress_deltas: bool = True
    retry_wait_callback: Any | None = None
    checkpoint_callback: Any | None = None
    injection_callback: Any | None = None
    llm_timeout_s: float | None = None
    event_logger: Callable[[str, dict[str, Any]], None] | None = None
    goal_active_predicate: Callable[[], bool] | None = None
    goal_continue_message: GoalContinueMessage | None = None
    finalize_on_max_iterations: bool = True
    context_delta_callback: Callable[[dict[str, Any]], Any] | None = None

@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.context_governor = ContextGovernor()

    @staticmethod
    def _log_event(spec: AgentRunSpec, event: str, **fields: Any) -> None:
        if spec.event_logger is None:
            return
        try:
            spec.event_logger(event, fields)
        except Exception:
            return

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                    for item in value
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    @classmethod
    def _append_injected_messages(
        cls,
        messages: list[dict[str, Any]],
        injections: list[dict[str, Any]],
    ) -> None:
        """Append injected user messages while preserving role alternation."""
        for injection in injections:
            if (
                messages
                and injection.get("role") == "user"
                and messages[-1].get("role") == "user"
                and not is_hidden_history_message(injection)
                and not is_hidden_history_message(messages[-1])
            ):
                merged = dict(messages[-1])
                merged["content"] = cls._merge_message_content(
                    merged.get("content"),
                    injection.get("content"),
                )
                messages[-1] = merged
                continue
            messages.append(injection)

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        phase: str = "after error",
        iteration: int | None = None,
        allow_goal_continue: bool = False,
    ) -> tuple[bool, int]:
        """Drain pending injections. Returns (should_continue, updated_cycles).

        If injections are found and we haven't exceeded _MAX_INJECTION_CYCLES,
        append them to *messages* (and emit a checkpoint if *assistant_message*
        and *iteration* are both provided) and return (True, cycles+1) so the
        caller continues the iteration loop.  Otherwise return (False, cycles).
        """
        injections: list[dict[str, Any]] = []
        real_injection = False
        if injection_cycles < _MAX_INJECTION_CYCLES:
            injections = await self._drain_injections(spec)
            real_injection = bool(injections)
        if not injections and allow_goal_continue and assistant_message is not None:
            predicate = spec.goal_active_predicate
            if predicate is not None and predicate():
                injections = [self._build_goal_continue_message(spec)]
        if not injections:
            return False, injection_cycles
        if real_injection:
            injection_cycles += 1
        if assistant_message is not None:
            messages.append(assistant_message)
            if iteration is not None:
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "final_response",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [],
                    },
                )
        self._append_injected_messages(messages, injections)
        if real_injection:
            logger.info(
                "Injected {} follow-up message(s) {} ({}/{})",
                len(injections), phase, injection_cycles, _MAX_INJECTION_CYCLES,
            )
        else:
            logger.info("Injected sustained-goal continuation {}", phase)
        return True, injection_cycles

    def _build_goal_continue_message(self, spec: AgentRunSpec) -> dict[str, str]:
        custom = spec.goal_continue_message
        if callable(custom):
            try:
                custom = custom()
            except Exception:
                logger.exception("goal_continue_message callback failed")
                custom = None
        return build_goal_continue_message(custom)

    async def _drain_injections(self, spec: AgentRunSpec) -> list[dict[str, Any]]:
        """Drain pending user messages via the injection callback.

        Returns normalized user messages (capped by
        ``_MAX_INJECTIONS_PER_TURN``), or an empty list when there is
        nothing to inject. Messages beyond the cap are logged so they
        are not silently lost.
        """
        if spec.injection_callback is None:
            return []
        try:
            signature = inspect.signature(spec.injection_callback)
            accepts_limit = (
                "limit" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_limit:
                items = await spec.injection_callback(limit=_MAX_INJECTIONS_PER_TURN)
            else:
                items = await spec.injection_callback()
        except Exception:
            logger.exception("injection_callback failed")
            return []
        if not items:
            return []
        injected_messages: list[dict[str, Any]] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict) and item.get("role") == "user" and "content" in item:
                if self._has_injection_content(item.get("content")):
                    injected_messages.append(item)
                continue
            if isinstance(item, dict):
                continue
            content = getattr(item, "content") if hasattr(item, "content") else str(item)
            if self._has_injection_content(content):
                injected_messages.append({"role": "user", "content": content})
        if len(injected_messages) > _MAX_INJECTIONS_PER_TURN:
            dropped = len(injected_messages) - _MAX_INJECTIONS_PER_TURN
            logger.warning(
                "Injection callback returned {} messages, capping to {} ({} dropped)",
                len(injected_messages), _MAX_INJECTIONS_PER_TURN, dropped,
            )
            injected_messages = injected_messages[:_MAX_INJECTIONS_PER_TURN]
        return injected_messages

    @staticmethod
    def _has_injection_content(content: Any) -> bool:
        if content is None:
            return False
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(content)
        return True

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        context = AgentRunHookContext(messages=deepcopy(messages))

        try:
            await hook.before_run(context)
            result = await self._run_core(spec, hook, messages)
        except asyncio.CancelledError as exc:
            context.messages = deepcopy(messages)
            context.stop_reason = "cancelled"
            context.error = None
            context.exception = exc
            raise
        except Exception as exc:
            context.messages = deepcopy(messages)
            context.stop_reason = "error"
            context.error = f"Error: {type(exc).__name__}: {exc}"
            context.exception = exc
            await hook.on_error(context)
            raise
        else:
            context.messages = deepcopy(result.messages)
            context.final_content = result.final_content
            context.tools_used = list(result.tools_used)
            context.usage = dict(result.usage)
            context.stop_reason = result.stop_reason
            context.error = result.error
            context.tool_events = deepcopy(result.tool_events)
            context.had_injections = result.had_injections
            context.exception = None
            if context.error is not None:
                await hook.on_error(context)
            await hook.after_run(context)
            return result
        finally:
            context.messages = deepcopy(messages)
            if context.exception is None:
                await hook.on_finally(context)
            else:
                try:
                    await hook.on_finally(context)
                except Exception:
                    logger.exception(
                        "AgentHook.on_finally error after {}",
                        context.stop_reason or "run exception",
                    )

    async def _run_core(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        final_content: str | None = None
        tools_used: list[str] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        # Per-turn throttle for repeated attempts against the same outside target.
        workspace_violation_counts: dict[str, int] = {}
        empty_content_retries = 0
        length_recovery_count = 0
        had_injections = False
        injection_cycles = 0
        tool_batch_history: list[tuple[tuple[str, str], ...]] = []
        warned_tool_loops: set[tuple[tuple[tuple[str, str], ...], ...]] = set()
        compacted_tool_call_ids: set[str] = set()
        tool_digests: dict[str, ToolDigest] = {}
        adaptive_context_block_limit: int | None = None
        false_tool_budget_retries = 0
        model_request_count = 0
        prompt_peak_tokens = 0
        governance_saved_total = 0
        governance_config = ContextGovernanceConfig(
            provider=self.provider,
            model=spec.model,
            tools=spec.tools,
            workspace=spec.workspace,
            data_dir=spec.data_dir,
            session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            context_window_tokens=spec.context_window_tokens,
            context_block_limit=spec.context_block_limit,
            max_tokens=spec.max_tokens,
            inflight_start_index=len(spec.initial_messages),
        )

        for iteration in range(spec.max_iterations):
            self._log_event(
                spec,
                "runner.iteration.start",
                iteration=iteration,
                messages=len(messages),
            )
            try:
                # Keep the persisted conversation untouched. Context governance
                # may repair or compact historical messages for the model, but
                # those synthetic edits must not shift the append boundary used
                # later when the caller saves only the new turn.
                active_governance_config = (
                    replace(
                        governance_config,
                        context_block_limit=adaptive_context_block_limit,
                    )
                    if adaptive_context_block_limit is not None
                    else governance_config
                )
                tools_for_model = active_governance_config.tools.get_definitions()
                before_context = self.context_governor.context_metrics(messages, tools_for_model)
                messages_for_model = self.context_governor.prepare_for_model(
                    active_governance_config,
                    messages,
                    compacted_tool_call_ids,
                    tool_digests=tool_digests,
                )
                after_context = self.context_governor.context_metrics(
                    messages_for_model, tools_for_model
                )
                saved_by_group = {
                    key: max(0, before_context.get(key, 0) - after_context.get(key, 0))
                    for key in before_context
                }
                saved_total = max(0, before_context["total"] - after_context["total"])
                governance_saved_total += saved_total
                self._log_event(
                    spec,
                    "runner.context.governance",
                    iteration=iteration,
                    before=before_context,
                    after=after_context,
                    saved=saved_by_group,
                    saved_total=saved_total,
                    compacted_tool_results=len(compacted_tool_call_ids),
                    digested_tool_results=sum(
                        1 for call_id in compacted_tool_call_ids if call_id in tool_digests
                    ),
                )
            except Exception:
                logger.exception(
                    "Context governance failed on turn {} for {}; applying minimal repair",
                    iteration,
                    spec.session_key or "default",
                )
                try:
                    messages_for_model = ContextGovernor.strip_placeholder_assistant_messages(
                        messages
                    )
                    messages_for_model = ContextGovernor.strip_malformed_tool_calls(
                        messages_for_model
                    )
                    messages_for_model = ContextGovernor.drop_orphan_tool_results(
                        messages_for_model
                    )
                    messages_for_model = ContextGovernor.backfill_missing_tool_results(
                        messages_for_model
                    )
                except Exception:
                    messages_for_model = messages
            context = AgentHookContext(
                iteration=iteration,
                messages=messages,
                session_key=spec.session_key,
            )
            await hook.before_iteration(context)
            response = await self._request_model(spec, messages_for_model, hook, context)
            if (
                LLMProvider.is_context_length_response(response)
                and not context.streamed_content
            ):
                estimate, source = estimate_prompt_tokens_chain(
                    self.provider,
                    spec.model,
                    messages_for_model,
                    spec.tools.get_definitions(),
                )
                emergency_budget = max(1024, int(estimate * 0.75))
                if adaptive_context_block_limit is not None:
                    emergency_budget = min(
                        adaptive_context_block_limit, emergency_budget
                    )
                retry_config = replace(
                    governance_config,
                    context_block_limit=emergency_budget,
                )
                retry_messages = self.context_governor.snip_history(
                    retry_config, messages_for_model
                )
                retry_messages = ContextGovernor.drop_orphan_tool_results(
                    retry_messages
                )
                retry_messages = ContextGovernor.backfill_missing_tool_results(
                    retry_messages
                )
                if retry_messages != messages_for_model:
                    adaptive_context_block_limit = emergency_budget
                    self._log_event(
                        spec,
                        "runner.context_overflow.retry",
                        estimated_tokens=estimate,
                        target_tokens=emergency_budget,
                        estimate_source=source,
                        messages_before=len(messages_for_model),
                        messages_after=len(retry_messages),
                    )
                    logger.warning(
                        "Context overflow for {}; trimming {} -> {} messages "
                        "and retrying once",
                        spec.session_key or "default",
                        len(messages_for_model),
                        len(retry_messages),
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                    messages_for_model = retry_messages
                    response = await self._request_model(
                        spec, messages_for_model, hook, context
                    )
            if (
                false_tool_budget_retries < 1
                and self._claims_false_tool_budget(response)
                and bool(spec.tools.get_definitions())
            ):
                false_tool_budget_retries += 1
                rejected_usage = self._usage_or_estimate(
                    spec, messages_for_model, response
                )
                self._accumulate_usage(usage, rejected_usage)
                self._log_event(
                    spec,
                    "runner.false_tool_budget.retry",
                    iteration=iteration,
                    max_iterations=spec.max_iterations,
                )
                logger.warning(
                    "Model falsely reported exhausted tool budget for {}; retrying once",
                    spec.session_key or "default",
                )
                retry_messages = list(messages_for_model)
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[Runtime correction] Tool access is available for this new turn. "
                            f"This is iteration {iteration + 1} of {spec.max_iterations}; "
                            "tool calls in earlier user turns are historical and do not consume "
                            "the current turn's runtime budget. Use the available tools now if "
                            "the user's request requires action. Do not repeat the exhaustion claim."
                        ),
                    }
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)
                messages_for_model = retry_messages
                response = await self._request_model(
                    spec, messages_for_model, hook, context
                )
            context.response = response

            context.tool_calls = list(response.tool_calls)
            reasoning_text, cleaned_content = extract_reasoning(
                response.reasoning_content,
                response.thinking_blocks,
                response.content,
            )
            response.content = cleaned_content
            raw_usage = self._usage_or_estimate(spec, messages_for_model, response)
            context.usage = dict(raw_usage)
            self._accumulate_usage(usage, raw_usage)
            model_request_count += 1
            prompt_peak_tokens = max(prompt_peak_tokens, raw_usage.get("prompt_tokens", 0))
            response_log_fields: dict[str, Any] = {
                "iteration": iteration,
                "finish_reason": response.finish_reason,
                "should_execute_tools": response.should_execute_tools,
                "tool_calls": [tc.name for tc in response.tool_calls],
                "usage": raw_usage,
            }
            if response.finish_reason == "error":
                response_log_fields["error_content"] = response.content or ""
                response_log_fields["error_kind"] = response.error_kind
            if response.provider_diagnostics:
                response_log_fields["provider_diagnostics"] = response.provider_diagnostics
            self._log_event(spec, "runner.model.response", **response_log_fields)
            if reasoning_text and not context.streamed_reasoning:
                await hook.emit_reasoning(reasoning_text)
                await hook.emit_reasoning_end()
                context.streamed_reasoning = True

            if response.should_execute_tools:
                context.tool_calls = list(response.tool_calls)
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                loop_match = self._record_tool_batch(tool_batch_history, response.tool_calls)
                loop_correction: str | None = None
                if loop_match is not None:
                    loop_pattern, repeat_count = loop_match
                    pattern_tools = self._tool_loop_pattern_tools(loop_pattern)
                    if repeat_count >= _TOOL_LOOP_STOP_REPEATS:
                        final_content = (
                            f"{_TOOL_LOOP_STOP_MESSAGE}\n\n"
                            f"重复序列：{pattern_tools}；已连续出现 {repeat_count} 轮。"
                        )
                        stop_reason = "tool_loop"
                        self._append_final_message(messages, final_content)
                        context.final_content = final_content
                        context.stop_reason = stop_reason
                        self._log_event(
                            spec,
                            "runner.tool_loop.stopped",
                            iteration=iteration,
                            repeat_count=repeat_count,
                            period=len(loop_pattern),
                            tools=pattern_tools,
                        )
                        await hook.after_iteration(context)
                        break
                    if (
                        repeat_count >= _TOOL_LOOP_WARN_REPEATS
                        and loop_pattern not in warned_tool_loops
                    ):
                        warned_tool_loops.add(loop_pattern)
                        loop_correction = (
                            "[Runtime correction] The same tool-call sequence has repeated "
                            f"{repeat_count} times without a change in arguments: {pattern_tools}. "
                            "Do not repeat this sequence again. Use the results already returned, "
                            "change the investigation or implementation approach, or finish with "
                            "a clear explanation of what blocks progress."
                        )
                        self._log_event(
                            spec,
                            "runner.tool_loop.warned",
                            iteration=iteration,
                            repeat_count=repeat_count,
                            period=len(loop_pattern),
                            tools=pattern_tools,
                        )

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                messages.append(assistant_message)
                transition_started_at = time.monotonic()
                self._log_event(
                    spec,
                    "runner.pre_tools.start",
                    iteration=iteration,
                    tool_calls=[tc.name for tc in response.tool_calls],
                )
                self._log_event(spec, "runner.pre_tools.checkpoint.start", iteration=iteration)
                try:
                    await asyncio.wait_for(
                        self._emit_checkpoint(
                            spec,
                            {
                                "phase": "awaiting_tools",
                                "iteration": iteration,
                                "model": spec.model,
                                "assistant_message": assistant_message,
                                "completed_tool_results": [],
                                "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                            },
                        ),
                        timeout=_PRE_TOOL_TRANSITION_WATCHDOG_S,
                    )
                except asyncio.TimeoutError:
                    self._log_event(
                        spec,
                        "runner.pre_tools.watchdog_timeout",
                        iteration=iteration,
                        phase="checkpoint",
                        timeout_s=_PRE_TOOL_TRANSITION_WATCHDOG_S,
                        tool_calls=[tc.name for tc in response.tool_calls],
                    )
                    logger.warning(
                        "Pre-tool checkpoint timed out after {}s for {}; continuing to tools",
                        _PRE_TOOL_TRANSITION_WATCHDOG_S,
                        spec.session_key or "default",
                    )
                else:
                    self._log_event(
                        spec,
                        "runner.pre_tools.checkpoint.done",
                        iteration=iteration,
                        duration_ms=round((time.monotonic() - transition_started_at) * 1000, 1),
                    )

                hook_started_at = time.monotonic()
                self._log_event(spec, "runner.pre_tools.hook.start", iteration=iteration)
                try:
                    await asyncio.wait_for(
                        hook.before_execute_tools(context),
                        timeout=_PRE_TOOL_TRANSITION_WATCHDOG_S,
                    )
                except asyncio.TimeoutError:
                    self._log_event(
                        spec,
                        "runner.pre_tools.watchdog_timeout",
                        iteration=iteration,
                        phase="before_execute_tools",
                        timeout_s=_PRE_TOOL_TRANSITION_WATCHDOG_S,
                        tool_calls=[tc.name for tc in response.tool_calls],
                    )
                    logger.warning(
                        "Pre-tool hook timed out after {}s for {}; continuing to tools",
                        _PRE_TOOL_TRANSITION_WATCHDOG_S,
                        spec.session_key or "default",
                    )
                else:
                    self._log_event(
                        spec,
                        "runner.pre_tools.hook.done",
                        iteration=iteration,
                        duration_ms=round((time.monotonic() - hook_started_at) * 1000, 1),
                    )

                self._log_event(
                    spec,
                    "runner.pre_tools.dispatch_tools",
                    iteration=iteration,
                    transition_duration_ms=round((time.monotonic() - transition_started_at) * 1000, 1),
                )
                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    response.tool_calls,
                    external_lookup_counts,
                    workspace_violation_counts,
                    hook,
                    context,
                )
                tool_events.extend(new_events)
                self._log_event(
                    spec,
                    "runner.tools.completed",
                    iteration=iteration,
                    events=new_events,
                    fatal_error=(
                        f"{type(fatal_error).__name__}: {fatal_error}"
                        if fatal_error is not None else None
                    ),
                )
                tools_used.extend(
                    tool_call.name
                    for tool_call, event in zip(response.tool_calls, new_events)
                    if event.get("status") == "ok"
                )
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                await hook.after_execute_tools(context)
                completed_tool_results: list[dict[str, Any]] = []
                for tool_call, result, event in zip(response.tool_calls, results, new_events):
                    normalized = self.context_governor.normalize_tool_result(
                        governance_config,
                        tool_call.id,
                        tool_call.name,
                        result,
                    )
                    artifact_locator = self.context_governor.persisted_result_locator(normalized)
                    digest, evidence = ToolDigestBuilder.build(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        result=result,
                        status=str(event.get("status") or "ok"),
                        artifact_locator=artifact_locator,
                    )
                    tool_digests[tool_call.id] = digest
                    if spec.context_delta_callback is not None:
                        try:
                            spec.context_delta_callback({
                                "tool_digest": digest,
                                "evidence": evidence,
                            })
                        except Exception:
                            logger.exception("Context delta callback failed for {}", tool_call.id)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": normalized,
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                if loop_correction is not None:
                    messages.append({"role": "user", "content": loop_correction})
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    final_content = error
                    stop_reason = "tool_error"
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    should_continue, injection_cycles = await self._try_drain_injections(
                        spec, messages, None, injection_cycles,
                        phase="after tool error",
                    )
                    if should_continue:
                        had_injections = True
                        continue
                    break
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_content_retries = 0
                length_recovery_count = 0
                # Checkpoint 1: drain injections after tools, before next LLM call
                _drained, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after tool execution",
                )
                if _drained:
                    had_injections = True
                await hook.after_iteration(context)
                continue

            if response.has_tool_calls:
                logger.warning(
                    "Ignoring tool calls under finish_reason='{}' for {}",
                    response.finish_reason,
                    spec.session_key or "default",
                )

            clean = hook.finalize_content(context, response.content)
            if response.finish_reason != "error" and is_blank_text(clean):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_content_retries,
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)
                retry_messages = self._finalization_retry_messages(messages_for_model)
                response = await self._request_finalization_retry(spec, messages_for_model)
                retry_usage = self._usage_or_estimate(spec, retry_messages, response)
                self._accumulate_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = dict(raw_usage)
                context.tool_calls = list(response.tool_calls)
                clean = hook.finalize_content(context, response.content)

            if response.finish_reason == "length" and not is_blank_text(clean):
                length_recovery_count += 1
                if length_recovery_count <= _MAX_LENGTH_RECOVERIES:
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        length_recovery_count,
                        _MAX_LENGTH_RECOVERIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                    messages.append(build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))
                    messages.append(build_length_recovery_message())
                    await hook.after_iteration(context)
                    continue

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

            # Check real mid-turn injections before signaling stream end. A
            # sustained goal persists across user turns; it must not manufacture
            # another user message after a model has completed this turn.
            # If real injections are found we keep the stream alive (resuming=True)
            # so streaming channels don't prematurely finalize the card.
            should_continue, injection_cycles = await self._try_drain_injections(
                spec, messages, assistant_message, injection_cycles,
                phase="after final response",
                iteration=iteration,
            )
            if should_continue:
                had_injections = True

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=should_continue)

            if should_continue:
                await hook.after_iteration(context)
                continue

            if response.finish_reason == "error":
                if LLMProvider.is_arrearage_response(response):
                    final_content = _ARREARAGE_ERROR_MESSAGE
                else:
                    final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                self._append_model_error_placeholder(messages)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after LLM error",
                )
                if should_continue:
                    had_injections = True
                    continue
                break
            if is_blank_text(clean):
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after empty response",
                )
                if should_continue:
                    had_injections = True
                    continue
                break

            messages.append(assistant_message or build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            # Drain any remaining injections so they are appended to the
            # conversation history instead of being re-published as
            # independent inbound messages by _dispatch's finally block.
            # We include them before the no-tools finalization pass so the
            # final response can account for every known follow-up.
            drained_after_max_iterations, injection_cycles = await self._try_drain_injections(
                spec, messages, None, injection_cycles,
                phase="after max_iterations",
            )
            if drained_after_max_iterations:
                had_injections = True
            final_content = None
            if spec.finalize_on_max_iterations:
                final_content = await self._try_finalize_after_max_iterations(
                    spec,
                    hook,
                    messages,
                    usage,
                )
            if final_content is None:
                final_content = self._max_iterations_fallback(spec)
            self._append_final_message(messages, final_content)

        self._log_event(
            spec,
            "runner.context.turn_summary",
            model_requests=model_request_count,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_peak_tokens=prompt_peak_tokens,
            governance_saved_total=governance_saved_total,
            compacted_tool_results=len(compacted_tool_call_ids),
            digested_tool_results=sum(
                1 for call_id in compacted_tool_call_ids if call_id in tool_digests
            ),
        )
        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            had_injections=had_injections,
        )

    @staticmethod
    def _claims_false_tool_budget(response: LLMResponse) -> bool:
        if response.finish_reason == "error" or response.has_tool_calls:
            return False
        text = str(response.content or "").lower()
        english = (
            ("tool-call budget" in text or "tool call budget" in text)
            and any(token in text for token in ("this turn", "current turn", "this session"))
            and any(token in text for token in ("exhausted", "used up", "no remaining"))
        )
        chinese = (
            "工具调用" in text
            and any(token in text for token in ("当前会话", "本轮", "当前轮"))
            and any(token in text for token in ("额度", "预算", "次数"))
            and any(token in text for token in ("耗尽", "用完", "已满", "没有剩余"))
        )
        return english or chinese
    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": spec.model,
            "retry_mode": spec.provider_retry_mode,
            "on_retry_wait": spec.retry_wait_callback,
            "on_retry_event": lambda fields: self._log_event(
                spec, "runner.model.retry", **fields
            ),
        }
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            kwargs["max_tokens"] = spec.max_tokens
        if spec.reasoning_effort is not None:
            kwargs["reasoning_effort"] = spec.reasoning_effort
        return kwargs

    @staticmethod
    def _record_tool_batch(
        history: list[tuple[tuple[str, str], ...]],
        tool_calls: list[ToolCallRequest],
    ) -> tuple[tuple[tuple[tuple[str, str], ...], ...], int] | None:
        """Record a tool batch and detect a repeated suffix cycle."""
        batch = tuple(
            (
                call.name,
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
            for call in tool_calls
        )
        history.append(batch)
        if not batch or AgentRunner._tool_loop_batch_is_polling(batch):
            return None

        max_period = min(_TOOL_LOOP_MAX_PERIOD, len(history) // _TOOL_LOOP_WARN_REPEATS)
        for period in range(1, max_period + 1):
            pattern = tuple(history[-period:])
            repeat_count = 1
            cursor = len(history) - period
            while cursor >= period and tuple(history[cursor - period:cursor]) == pattern:
                repeat_count += 1
                cursor -= period
            if repeat_count >= _TOOL_LOOP_WARN_REPEATS:
                if not all(
                    AgentRunner._tool_loop_batch_is_polling(pattern_batch)
                    for pattern_batch in pattern
                ):
                    return pattern, repeat_count
        return None

    @staticmethod
    def _tool_loop_batch_is_polling(batch: tuple[tuple[str, str], ...]) -> bool:
        for name, arguments_json in batch:
            if name in _TOOL_LOOP_EXEMPT_TOOLS:
                continue
            if name == "process_control":
                try:
                    action = json.loads(arguments_json).get("action")
                except (AttributeError, json.JSONDecodeError):
                    return False
                if action in {"list", "logs"}:
                    continue
            return False
        return True

    @staticmethod
    def _tool_loop_pattern_tools(
        pattern: tuple[tuple[tuple[str, str], ...], ...],
    ) -> str:
        return " → ".join(
            "+".join(name for name, _arguments in batch) for batch in pattern
        )

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
        *,
        malformed_retry: bool = False,
    ):
        timeout_s: float | None = spec.llm_timeout_s
        if timeout_s is None:
            # Default to a finite timeout to avoid per-session lock starvation when an LLM
            # request hangs indefinitely (e.g. gateway/network stall).
            # Set NANOBOT_LLM_TIMEOUT_S=0 to disable.
            raw = os.environ.get("NANOBOT_LLM_TIMEOUT_S", "300").strip()
            try:
                timeout_s = float(raw)
            except (TypeError, ValueError):
                timeout_s = 300.0
        if timeout_s is not None and timeout_s <= 0:
            timeout_s = None

        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=spec.tools.get_definitions(),
        )
        wants_streaming = hook.wants_streaming()
        wants_progress_streaming = (
            not wants_streaming
            and spec.stream_progress_deltas
            and spec.progress_callback is not None
            and getattr(self.provider, "supports_progress_deltas", False) is True
        )

        progress_state: dict[str, bool] | None = None

        if wants_streaming:
            thinking_buf = ""

            async def _stream(delta: str) -> None:
                if delta:
                    context.streamed_content = True
                await hook.on_stream(context, delta)

            async def _thinking(delta: str) -> None:
                nonlocal thinking_buf
                if not delta:
                    return
                prev_clean = strip_reasoning_tags(thinking_buf)
                thinking_buf += delta
                new_clean = strip_reasoning_tags(thinking_buf)
                incremental = new_clean[len(prev_clean):]
                if incremental:
                    context.streamed_reasoning = True
                    await hook.emit_reasoning(incremental)

            async def _stream_recover() -> None:
                await hook.on_stream_end(context, resuming=True)

            coro = self.provider.chat_stream_with_retry(
                **kwargs,
                on_content_delta=_stream,
                on_thinking_delta=_thinking,
                on_stream_recover=_stream_recover,
            )
        elif wants_progress_streaming:
            stream_buf = ""
            think_extractor = IncrementalThinkExtractor()
            progress_state = {"reasoning_open": False}

            async def _stream_progress(delta: str) -> None:
                nonlocal stream_buf
                if not delta:
                    return
                prev_clean = strip_think(stream_buf)
                stream_buf += delta
                new_clean = strip_think(stream_buf)
                incremental = new_clean[len(prev_clean):]

                if await think_extractor.feed(stream_buf, hook.emit_reasoning):
                    context.streamed_reasoning = True
                    progress_state["reasoning_open"] = True

                if incremental:
                    if progress_state["reasoning_open"]:
                        await hook.emit_reasoning_end()
                        progress_state["reasoning_open"] = False
                    context.streamed_content = True
                    await spec.progress_callback(incremental)

            coro = self.provider.chat_stream_with_retry(
                **kwargs,
                on_content_delta=_stream_progress,
            )
        else:
            coro = self.provider.chat_with_retry(**kwargs)

        # Provider-level stream idle timeouts only detect a complete lack of events.
        # A broken SSE connection may keep emitting protocol events without ever completing,
        # so apply the finite wall-clock cap to streaming requests as well. Sustained goals
        # can still opt out explicitly by passing llm_timeout_s=0.
        outer_timeout_s = timeout_s
        self._log_event(
            spec,
            "runner.model.request",
            model=spec.model,
            messages=len(messages),
            tools=len(kwargs.get("tools") or []),
            streaming=wants_streaming,
            progress_streaming=wants_progress_streaming,
            timeout_s=outer_timeout_s,
        )
        try:
            response = (
                await coro if outer_timeout_s is None
                else await asyncio.wait_for(coro, timeout=outer_timeout_s)
            )
        except asyncio.TimeoutError:
            self._log_event(
                spec,
                "runner.model.timeout",
                timeout_s=outer_timeout_s,
                streaming=wants_streaming or wants_progress_streaming,
            )
            if outer_timeout_s is None:
                return LLMResponse(
                    content="Error calling LLM: stream stalled",
                    finish_reason="error",
                    error_kind="timeout",
                )
            return LLMResponse(
                content=f"Error calling LLM: timed out after {outer_timeout_s:g}s",
                finish_reason="error",
                error_kind="timeout",
            )
        model_done_fields: dict[str, Any] = {
            "finish_reason": response.finish_reason,
            "error_kind": response.error_kind,
        }
        if response.finish_reason == "error":
            model_done_fields.update({
                "error_content": response.content or "",
                "error_status_code": response.error_status_code,
                "error_type": response.error_type,
                "error_code": response.error_code,
                "error_retry_after_s": response.error_retry_after_s,
                "error_should_retry": response.error_should_retry,
            })
        self._log_event(
            spec,
            "runner.model.request.done",
            **model_done_fields,
        )
        if progress_state and progress_state.get("reasoning_open"):
            await hook.emit_reasoning_end()
        dropped, all_dropped, original_finish_reason = (
            self._drop_malformed_tool_calls(response)
        )
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and not malformed_retry
        ):
            logger.warning(
                "Retrying LLM request after all {} malformed tool call(s) were dropped",
                dropped,
            )
            retry_messages = self._malformed_tool_call_retry_messages(
                messages, response.content,
            )
            return await self._request_model(
                spec, retry_messages, hook, context,
                malformed_retry=True,
            )
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and malformed_retry
        ):
            logger.warning(
                "Malformed tool calls persisted after retry; falling back to no-tools request",
            )
            fallback_messages = self._malformed_tool_call_retry_messages(
                messages, response.content,
            )
            return await self._request_no_tools(spec, fallback_messages)
        return response

    @staticmethod
    def _drop_malformed_tool_calls(
        response: LLMResponse,
    ) -> tuple[int, bool, str | None]:
        """Strip tool calls whose name is missing/non-string from the response.

        Returns (dropped_count, all_dropped, original_finish_reason).

        A degenerate call (name=None or "") cannot be executed, and if it were
        persisted into the assistant message it would be replayed on every
        subsequent turn, causing upstream validation errors
        (``tool_use.name: Input should be a valid string``) that permanently
        wedge the session. Dropping it here keeps it out of execution, the
        assistant message, and the saved history in one place.
        """
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return (0, False, getattr(response, "finish_reason", None))
        valid = [tc for tc in calls if tc.has_valid_name()]
        if len(valid) == len(calls):
            return (0, False, getattr(response, "finish_reason", None))
        dropped = len(calls) - len(valid)
        original_finish_reason = getattr(response, "finish_reason", None)
        logger.warning(
            "Dropped {} malformed tool call(s) with missing/non-string name "
            "from LLM response (finish_reason={!r})",
            dropped,
            original_finish_reason,
        )
        response.tool_calls = valid
        if not valid:
            response.finish_reason = "stop"
        return (dropped, not valid, original_finish_reason)

    @staticmethod
    def _malformed_tool_call_retry_messages(
        messages: list[dict[str, Any]],
        assistant_text: str | None,
    ) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        note = (
            "The previous model response attempted to call tools, but every tool call "
            "was malformed: the tool_use blocks had missing or non-string tool names. "
            "Do not answer with a promise to use tools. Either call the required tools again "
            "using valid tool names from the provided tool list and JSON object inputs, or give "
            "a final answer only if no tool is required."
        )
        if assistant_text:
            note += (
                f"\n\nPrevious assistant text before the malformed calls:\n"
                f"{assistant_text}"
            )
        retry_messages.append({"role": "user", "content": note})
        return retry_messages

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ):
        retry_messages = self._finalization_retry_messages(messages)
        return await self._request_no_tools(spec, retry_messages)

    @staticmethod
    def _finalization_retry_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        return retry_messages

    async def _try_finalize_after_max_iterations(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> str | None:
        retry_messages = self._budget_exhausted_finalization_messages(messages)
        try:
            response = await self._request_no_tools(spec, retry_messages)
        except Exception:
            logger.exception(
                "Budget-exhausted finalization failed for {}; using fallback",
                spec.session_key or "default",
            )
            return None

        raw_usage = self._usage_or_estimate(spec, retry_messages, response)
        self._accumulate_usage(usage, raw_usage)
        if response.finish_reason == "error" or response.has_tool_calls:
            logger.warning(
                "Budget-exhausted finalization returned finish_reason='{}' "
                "with {} tool call(s) for {}; using fallback",
                response.finish_reason,
                len(response.tool_calls),
                spec.session_key or "default",
            )
            return None

        context = AgentHookContext(
            iteration=spec.max_iterations,
            messages=messages,
            response=response,
            usage=dict(raw_usage),
            session_key=spec.session_key,
        )
        clean = hook.finalize_content(context, response.content)
        if is_blank_text(clean):
            return None
        return clean

    async def _request_no_tools(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> LLMResponse:
        kwargs = self._build_request_kwargs(spec, messages, tools=None)
        return await self.provider.chat_with_retry(**kwargs)

    @staticmethod
    def _budget_exhausted_finalization_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        retry_messages.append(build_budget_exhausted_finalization_message())
        return retry_messages

    @staticmethod
    def _max_iterations_fallback(spec: AgentRunSpec) -> str:
        if spec.max_iterations_message:
            return spec.max_iterations_message.format(
                max_iterations=spec.max_iterations,
            )
        return render_template(
            "agent/max_iterations_message.md",
            strip=True,
            max_iterations=spec.max_iterations,
        )

    def _usage_or_estimate(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> dict[str, int]:
        usage = self._usage_dict(response.usage)
        total = self._usage_total(usage)
        if total > 0:
            usage["total_tokens"] = total
            usage.setdefault("provider_tokens", total)
            return usage
        if response.finish_reason == "error":
            return {}
        return self._estimate_response_usage(spec, messages, response)

    def _estimate_response_usage(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> dict[str, int]:
        try:
            tools = spec.tools.get_definitions()
        except Exception:
            tools = None
        prompt_tokens, _ = estimate_prompt_tokens_chain(self.provider, spec.model, messages, tools)
        assistant_message = build_assistant_message(
            response.content or "",
            tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        completion_tokens = estimate_message_tokens(assistant_message)
        total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)
        if total_tokens <= 0:
            return {}
        return {
            "prompt_tokens": max(0, prompt_tokens),
            "completion_tokens": max(0, completion_tokens),
            "total_tokens": total_tokens,
            "estimated_tokens": total_tokens,
        }

    @staticmethod
    def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _usage_total(usage: dict[str, int]) -> int:
        return max(0, usage.get("total_tokens", 0) or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        ))

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        merged = dict(left)
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        external_lookup_counts: dict[str, int],
        workspace_violation_counts: dict[str, int],
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        hook = hook or AgentHook()
        context = context or AgentHookContext(iteration=0, messages=[])
        deferred_searches = self._deferred_broad_searches(tool_calls)
        batches = self._partition_tool_batches(spec, tool_calls)
        self._log_event(
            spec,
            "runner.tools.start",
            count=len(tool_calls),
            batches=[len(batch) for batch in batches],
            tools=[tc.name for tc in tool_calls],
        )
        tool_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
        for batch in batches:
            if spec.concurrent_tools and len(batch) > 1:
                batch_results = await asyncio.gather(*(
                    self._deferred_search_result(spec, tool_call, deferred_searches[tool_call.id])
                    if tool_call.id in deferred_searches
                    else self._run_tool(
                        spec,
                        tool_call,
                        external_lookup_counts,
                        workspace_violation_counts,
                        hook,
                        context,
                    )
                    for tool_call in batch
                ))
                tool_results.extend(batch_results)
            else:
                batch_results = []
                for tool_call in batch:
                    if tool_call.id in deferred_searches:
                        result = await self._deferred_search_result(
                            spec, tool_call, deferred_searches[tool_call.id]
                        )
                    else:
                        result = await self._run_tool(
                            spec,
                            tool_call,
                            external_lookup_counts,
                            workspace_violation_counts,
                            hook,
                            context,
                        )
                    tool_results.append(result)
                    batch_results.append(result)

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        external_lookup_counts: dict[str, int],
        workspace_violation_counts: dict[str, int],
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        hook = hook or AgentHook()
        context = context or AgentHookContext(iteration=0, messages=[])
        hint = "\n\n[Analyze the error above and try a different approach.]"
        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        if lookup_error:
            self._log_event(
                spec,
                "runner.tool.blocked",
                tool=tool_call.name,
                call_id=tool_call.id,
                reason="repeated external lookup",
            )
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + hint, event, RuntimeError(lookup_error)
            return lookup_error + hint, event, None
        prepare_call = getattr(spec.tools, "prepare_call", None)
        tool, params, prep_error = None, tool_call.arguments, None
        if callable(prepare_call):
            with suppress(Exception):
                prepared = prepare_call(tool_call.name, tool_call.arguments)
                if isinstance(prepared, tuple) and len(prepared) == 3:
                    tool, params, prep_error = prepared
        if prep_error:
            self._log_event(
                spec,
                "runner.tool.prepare_error",
                tool=tool_call.name,
                call_id=tool_call.id,
                error=prep_error,
            )
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
            }
            handled = self._classify_violation(
                raw_text=prep_error,
                soft_payload=prep_error + hint,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            if handled is not None:
                return handled
            return prep_error + hint, event, (
                RuntimeError(prep_error) if spec.fail_on_tool_error else None
            )
        await hook.before_execute_tool(context, tool_call, tool, params)
        try:
            self._log_event(
                spec,
                "runner.tool.start",
                tool=tool_call.name,
                call_id=tool_call.id,
                arguments=params,
            )
            if tool is not None:
                execution = tool.execute(**params)
                timeout_s = tool.execution_timeout_s
                if timeout_s is not None:
                    result = await asyncio.wait_for(execution, timeout=timeout_s)
                else:
                    result = await execution
            else:
                result = await spec.tools.execute(tool_call.name, params)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            timeout_s = getattr(tool, "execution_timeout_s", None)
            payload = (
                "[Tool execution checkpoint — model decision]\n"
                f"The {tool_call.name} call exceeded its absolute {timeout_s:g}s guard. "
                "Decide whether the operation is still justified: stop, narrow/change approach, "
                "or retry explicitly. Search tools normally return a scan_cursor before this guard; "
                "if no cursor was returned, do not blindly repeat the identical broad call."
            )
            self._log_event(
                spec,
                "runner.tool.timeout_checkpoint",
                tool=tool_call.name,
                call_id=tool_call.id,
                timeout_s=timeout_s,
            )
            await hook.on_execute_tool_error(context, tool_call, tool, params, payload)
            return payload, {"name": tool_call.name, "status": "checkpoint", "detail": payload[:120]}, None
        except BaseException as exc:
            self._log_event(
                spec,
                "runner.tool.exception",
                tool=tool_call.name,
                call_id=tool_call.id,
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
            await hook.on_execute_tool_error(context, tool_call, tool, params, exc)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            payload = f"Error: {type(exc).__name__}: {exc}"
            handled = self._classify_violation(
                raw_text=str(exc),
                # Preserve legacy exception payloads without the retry hint.
                soft_payload=payload,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            if handled is not None:
                return handled
            if spec.fail_on_tool_error:
                return payload, event, exc
            return payload, event, None

        if is_tool_error_result(tool_call.name, result):
            self._log_event(
                spec,
                "runner.tool.error_result",
                tool=tool_call.name,
                call_id=tool_call.id,
                result=result,
            )
            await hook.on_execute_tool_error(context, tool_call, tool, params, result)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            handled = self._classify_violation(
                raw_text=result,
                soft_payload=result + hint,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            if handled is not None:
                return handled
            if spec.fail_on_tool_error:
                return result + hint, event, RuntimeError(result)
            return result + hint, event, None

        await hook.after_execute_tool(context, tool_call, tool, params, result)

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        self._log_event(
            spec,
            "runner.tool.done",
            tool=tool_call.name,
            call_id=tool_call.id,
            detail=detail,
        )
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}, None

    # SSRF is a hard security block at the tool boundary, but the agent turn
    # should recover conversationally instead of aborting the runtime.
    _SSRF_MARKERS: tuple[str, ...] = (
        "internal/private url detected",
        "private/internal address",
        "private address",
    )
    _SSRF_BOUNDARY_NOTE: str = (
        "This is a non-bypassable security boundary. Stop trying to access "
        "private/internal URLs. Do not retry with curl, wget, encoded IPs, "
        "alternate DNS, redirects, proxies, or another tool. Ask the user for "
        "local files, logs, screenshots, or an explicit safe public URL instead. "
        "If the user explicitly trusts this private URL, ask them to whitelist "
        "the exact IP/CIDR via tools.ssrfWhitelist."
    )

    # Non-SSRF boundary markers returned to the LLM as recoverable tool errors.
    _WORKSPACE_VIOLATION_MARKERS: tuple[str, ...] = (
        "outside the configured workspace",
        "outside allowed directory",
        "working_dir is outside",
        "working_dir could not be resolved",
        "path outside working dir",
        "path traversal detected",
    )

    @classmethod
    def _is_ssrf_violation(cls, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in cls._SSRF_MARKERS)

    @classmethod
    def _is_workspace_violation(cls, text: str) -> bool:
        """True when *text* looks like any policy boundary rejection."""
        if not text:
            return False
        lowered = text.lower()
        if cls._is_ssrf_violation(lowered):
            return True
        return any(marker in lowered for marker in cls._WORKSPACE_VIOLATION_MARKERS)

    def _classify_violation(
        self,
        *,
        raw_text: str,
        soft_payload: str,
        event: dict[str, str],
        tool_call: ToolCallRequest,
        workspace_violation_counts: dict[str, int],
    ) -> tuple[Any, dict[str, str], BaseException | None] | None:
        """Classify safety-boundary failures, or return ``None`` to pass through."""
        if self._is_ssrf_violation(raw_text):
            logger.warning(
                "Tool {} blocked by SSRF guard; returning non-retryable tool error: {}",
                tool_call.name,
                raw_text.replace("\n", " ").strip()[:200],
            )
            event["detail"] = self._event_detail("ssrf_violation: ", raw_text)
            return self._ssrf_soft_payload(raw_text), event, None

        if self._is_workspace_violation(raw_text):
            escalation = repeated_workspace_violation_error(
                tool_call.name,
                tool_call.arguments,
                workspace_violation_counts,
            )
            event["detail"] = self._event_detail("workspace_violation: ", raw_text)
            if escalation is not None:
                logger.warning(
                    "Tool {} hit workspace boundary repeatedly; escalating hint",
                    tool_call.name,
                )
                event["detail"] = self._event_detail(
                    "workspace_violation_escalated: ",
                    raw_text,
                )
                return escalation, event, None
            return soft_payload, event, None

        return None

    @classmethod
    def _ssrf_soft_payload(cls, raw_text: str) -> str:
        text = raw_text.strip() or "Error: request blocked by SSRF guard"
        return f"{text}\n\n{cls._SSRF_BOUNDARY_NOTE}"

    @staticmethod
    def _event_detail(prefix: str, text: str, limit: int = 160) -> str:
        return (prefix + text.replace("\n", " ").strip())[:limit]

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict[str, Any],
    ) -> None:
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = build_assistant_message(content)
            return
        messages.append(build_assistant_message(content))

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
        messages.append(build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))

    @staticmethod
    def _search_terms(tool_call: ToolCallRequest) -> set[str]:
        if not isinstance(tool_call.arguments, dict):
            return set()
        raw = tool_call.arguments.get("pattern") or tool_call.arguments.get("query") or ""
        return {term.lower() for term in re.findall(r"[\w.-]{4,}", str(raw))}

    @classmethod
    def _deferred_broad_searches(cls, tool_calls: list[ToolCallRequest]) -> dict[str, str]:
        """Defer same-turn fallback searches until the model sees the narrower result."""
        deferred: dict[str, str] = {}
        prior: list[ToolCallRequest] = []
        for call in tool_calls:
            if call.name not in {"grep", "find_files"} or not isinstance(call.arguments, dict):
                prior.append(call)
                continue
            path = Path(str(call.arguments.get("path", "."))).resolve(strict=False)
            terms = cls._search_terms(call)
            for narrow in prior:
                if narrow.name not in {"grep", "find_files"} or not isinstance(narrow.arguments, dict):
                    continue
                narrow_path = Path(str(narrow.arguments.get("path", "."))).resolve(strict=False)
                if path == narrow_path or path not in narrow_path.parents:
                    continue
                narrow_terms = cls._search_terms(narrow)
                if terms and narrow_terms and not terms.intersection(narrow_terms):
                    continue
                deferred[call.id] = (
                    "[Broader fallback search deferred — review narrow result first]\n"
                    f"A narrower overlapping search in '{narrow.arguments.get('path', '.')}' was requested "
                    f"in the same model response. The broader search in '{call.arguments.get('path', '.')}' "
                    "was not executed. Review the narrow result, then request the broader scope in a new "
                    "model step only if it remains necessary."
                )
                break
            prior.append(call)
        return deferred

    async def _deferred_search_result(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        payload: str,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        self._log_event(
            spec,
            "runner.tool.deferred",
            tool=tool_call.name,
            call_id=tool_call.id,
            reason="broader overlapping search",
        )
        return payload, {"name": tool_call.name, "status": "deferred", "detail": payload[:120]}, None

    def _partition_tool_batches(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        if not spec.concurrent_tools:
            return [[tool_call] for tool_call in tool_calls]

        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            get_tool = getattr(spec.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            can_batch = bool(
                tool
                and tool.concurrency_safe
                and tool.is_concurrency_safe_call(tool_call.arguments)
            )
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches
