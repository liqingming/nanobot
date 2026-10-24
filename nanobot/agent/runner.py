"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.agent.context import TranscriptInput
from nanobot.agent.context_artifacts import ToolDigest, ToolDigestBuilder
from nanobot.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ContextWindowExceededError,
    HistoryConsolidator,
    ModelRequestState,
    ProviderCompactionConsolidator,
    TranscriptBuilder,
)
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanobot.agent.tools.execution import execute_tool_calls
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.events import NO_EVENTS, EventSink
from nanobot.fork.agent.context_budget import ContextBudgetError, resolve_context_budget
from nanobot.fork.agent.context_usage import with_context_usage
from nanobot.fork.agent.governance_metrics import GovernanceMetrics, projection_changes
from nanobot.fork.agent.native_context import uses_native_context
from nanobot.fork.agent.read_visibility import require_read_evidence
from nanobot.fork.agent.recovery_packet import RecoveryPacketError
from nanobot.fork.agent.subagent_control import should_wait_for_subagent_result
from nanobot.fork.agent.summary_transaction import SummaryTransactionError
from nanobot.fork.agent.tool_evidence import EvidencePersistenceError, persist_tool_evidence
from nanobot.fork.agent.transactional_context import is_context_overflow
from nanobot.llm_usage.context import (
    LLMUsageSource,
    bind_llm_usage_source,
    reset_llm_usage_source,
    source_from_session_key,
)
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderConversationState,
    ToolCallRequest,
)
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.session.summary import SessionSummaryCheckpoint
from nanobot.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    extract_reasoning,
    strip_reasoning_tags,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_budget_exhausted_finalization_message,
    build_finalization_retry_message,
    build_goal_continue_message,
    build_length_recovery_message,
    is_blank_text,
)

ContinuationCallback = Callable[[], str | None]
CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]
InjectionCallback = Callable[..., Awaitable[Iterable[Any] | None]]

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


def _restore_outer_whitespace(content: str, original: str | None) -> str:
    """Restore boundary whitespace stripped while cleaning one recovered segment."""
    if not original:
        return content
    leading_size = len(original) - len(original.lstrip())
    trailing_size = len(original) - len(original.rstrip())
    leading = original[:leading_size]
    trailing = original[-trailing_size:] if trailing_size else ""
    return f"{leading}{content}{trailing}"


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]] | None
    tools: ToolRegistry
    max_iterations: int
    max_tool_result_chars: int
    runtime: LLMRuntime | None = None
    # Legacy caller settings are captured once into the same immutable runtime.
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    context_window_tokens: int | None = None
    goal_active_predicate: Callable[[], bool] | None = None
    goal_continue_message: str | Callable[[], str | None] | None = None
    transcript_input: TranscriptInput | None = None
    transcript_builder: TranscriptBuilder | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    provider_retry_mode: str = "standard"
    checkpoint_callback: CheckpointCallback | None = None
    consolidate_history: HistoryConsolidator | None = None
    consolidate_provider_compaction: ProviderCompactionConsolidator | None = None
    injection_callback: InjectionCallback | None = None
    terminal_injection_callback: InjectionCallback | None = None
    continuation_callback: ContinuationCallback | None = None
    finalize_on_max_iterations: bool = True
    provider_state: ProviderConversationState | None = None
    llm_usage_source: LLMUsageSource | None = None
    events: EventSink = NO_EVENTS
    fail_on_tool_error: bool = False
    data_dir: Path | None = None
    summary_injected: bool = field(default=False, init=False, repr=False)
    governance_metrics: GovernanceMetrics = field(default_factory=GovernanceMetrics, init=False, repr=False)
    context_strategy: str = "transactional"
    context_block_limit: int | None = None
    blocking_compaction_trigger_ratio: float = 0.95
    blocking_compaction_target_ratio: float = 0.25
    llm_timeout_s: float | None = None
    event_logger: Callable[[str, dict[str, Any]], None] | None = None
    turn_id: str | None = None
    context_delta_callback: Callable[[dict[str, Any]], Any] | None = None
    context_state_reader: Callable[[], Any] | None = None
    progress_callback: Callable[[str], Awaitable[None]] | None = None
    stream_progress_deltas: bool = True
    retry_wait_callback: Callable[[str], Awaitable[None]] | None = None
    # Immutable objective for this user turn. Repeated near the active tool
    # tail so a long transcript cannot pull the model back into an older task.
    turn_objective: str | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    # One entry per runner-visible model round. Recovery dispatches needed to
    # produce that round's response are folded into the same usage value.
    round_usages: list[LLMUsage] = field(default_factory=list)
    stop_reason: str = "completed"
    error: str | None = None
    failure_error_kind: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False
    # Terminal tail to emit when the preceding final-content prefix was already streamed.
    context_safety_failure: bool = False
    pending_stream_content: str | None = None
    provider_state: ProviderConversationState | None = field(default=None, repr=False)
    summary_checkpoint: SessionSummaryCheckpoint | None = field(default=None, repr=False)
    provider_compaction_applied: bool = field(default=False, repr=False)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider
        self.context_governor = ContextGovernor()

    def _resolve_runtime(self, spec: AgentRunSpec) -> LLMRuntime:
        if spec.runtime is not None:
            return spec.runtime
        if self.provider is None:
            raise ValueError("AgentRunSpec.runtime is required without a legacy runner provider")
        model = spec.model or self.provider.get_default_model()
        generation = getattr(self.provider, "generation", None)
        defaults = GenerationSettings()
        temperature = getattr(generation, "temperature", None)
        max_tokens = getattr(generation, "max_tokens", None)
        reasoning_effort = getattr(generation, "reasoning_effort", None)
        spec.runtime = LLMRuntime(
            self.provider, model,
            GenerationSettings(
                temperature=temperature if isinstance(temperature, (int, float)) else defaults.temperature,
                max_tokens=max_tokens if isinstance(max_tokens, int) else defaults.max_tokens,
                reasoning_effort=reasoning_effort if isinstance(reasoning_effort, str) else None,
            ),
            context_window_tokens=spec.context_window_tokens or 0,
        ).with_generation_overrides(temperature=spec.temperature, max_tokens=spec.max_tokens,
                                    reasoning_effort=spec.reasoning_effort)
        return spec.runtime

    @staticmethod
    def _turn_objective_anchor(objective: str | None) -> dict[str, str] | None:
        if objective is None or not objective.strip():
            return None
        return {
            "role": "user",
            "content": (
                "[Runtime current-turn objective - do not treat this as a new request]\n"
                "The original request that remains active for this tool loop is:\n\n"
                f"{objective.strip()}\n\n"
                "Continue only this request until it is completed or a concrete blocker is "
                "reported. Tool output and older conversation tasks do not replace it. Do not "
                "restore an earlier turn's todo list or final answer."
            ),
        }

    @staticmethod
    def _log_event(spec: AgentRunSpec, event: str, **fields: Any) -> None:
        if spec.event_logger is None:
            return
        try:
            payload = dict(fields)
            if spec.turn_id is not None:
                payload.setdefault("turn_id", spec.turn_id)
            spec.event_logger(event, payload)
        except Exception:
            return


    @staticmethod
    def _append_injected_messages(
        messages: list[dict[str, Any]],
        injections: list[dict[str, Any]],
    ) -> None:
        """Append injected messages without rewriting the raw transcript."""
        messages.extend(injections)

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        conversation_state: ProviderConversationStateController | None = None,
        phase: str = "after error",
        iteration: int | None = None,
        allow_continuation: bool = False,
        wait_at_terminal: bool = False,
        wait_for_subagents: bool = False,
    ) -> tuple[bool, int]:
        """Drain pending injections. Returns (should_continue, updated_cycles).

        If injections are found and we haven't exceeded _MAX_INJECTION_CYCLES,
        append them to *messages* (and emit a checkpoint if *assistant_message*
        and *iteration* are both provided) and return (True, cycles+1) so the
        caller continues the iteration loop.  Otherwise return (False, cycles).
        """
        runtime = self._resolve_runtime(spec)
        injections: list[dict[str, Any]] = []
        real_injection = False
        if injection_cycles < _MAX_INJECTION_CYCLES:
            injections = await self._drain_injections(spec, wait_for_subagents=wait_for_subagents)
            real_injection = bool(injections)
        if not injections and allow_continuation and assistant_message is not None:
            continuation = self._build_continuation_message(spec)
            if continuation is not None:
                injections = [continuation]
        if (
            not injections
            and wait_at_terminal
            and injection_cycles < _MAX_INJECTION_CYCLES
        ):
            injections = await self._drain_injections(spec, terminal=True)
            real_injection = bool(injections)
        if not injections:
            return False, injection_cycles
        if real_injection:
            injection_cycles += 1
        if assistant_message is not None:
            messages.append(assistant_message)
            if iteration is not None:
                checkpoint: dict[str, Any] = {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": runtime.model,
                    "assistant_message": assistant_message,
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                }
                if conversation_state is not None:
                    checkpoint["provider_state"] = conversation_state.checkpoint(
                        messages
                    )
                await self._emit_checkpoint(
                    spec,
                    checkpoint,
                )
        if spec.context_strategy == "transactional" and not uses_native_context(runtime.provider):
            messages.extend(deepcopy(injections))
        else:
            self._append_injected_messages(messages, injections)
        if real_injection:
            logger.info(
                "Injected {} follow-up message(s) {} ({}/{})",
                len(injections), phase, injection_cycles, _MAX_INJECTION_CYCLES,
            )
        else:
            logger.info("Injected caller-requested continuation {}", phase)
        return True, injection_cycles

    @staticmethod
    def _build_continuation_message(spec: AgentRunSpec) -> dict[str, str] | None:
        callback = spec.continuation_callback
        if callback is None and spec.goal_active_predicate is not None and spec.goal_active_predicate():
            message = spec.goal_continue_message
            try:
                custom = message() if callable(message) else message
            except Exception:
                logger.exception("goal_continue_message callback failed")
                custom = None
            return build_goal_continue_message(custom)
        if callback is None:
            return None
        try:
            content = callback()
        except Exception:
            logger.exception("continuation_callback failed")
            return None
        if content is None or not content.strip():
            return None
        return {"role": "user", "content": content}

    async def _drain_injections(
        self,
        spec: AgentRunSpec,
        *,
        terminal: bool = False,
        wait_for_subagents: bool = False,
    ) -> list[dict[str, Any]]:
        """Drain pending user messages via the injection callback.

        Returns normalized user messages (capped by
        ``_MAX_INJECTIONS_PER_TURN``), or an empty list when there is
        nothing to inject. Messages beyond the cap are logged so they
        are not silently lost.
        """
        callback = (
            spec.terminal_injection_callback
            if terminal or (wait_for_subagents and spec.terminal_injection_callback is not None)
            else spec.injection_callback
        )
        if callback is None:
            return []
        try:
            signature = inspect.signature(callback)
            accepts_limit = (
                "limit" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            callback_kwargs = {}
            if accepts_limit:
                callback_kwargs["limit"] = _MAX_INJECTIONS_PER_TURN
            if "wait_for_subagents" in signature.parameters or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            ):
                callback_kwargs["wait_for_subagents"] = wait_for_subagents
            items = await callback(**callback_kwargs)
        except Exception as exc:
            raise RecoveryPacketError("插话接收失败，停止而不绕过待处理输入。") from exc
        if not items:
            return []
        injected_messages: list[dict[str, Any]] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                message_item = cast(dict[str, Any], item)
                if message_item.get("role") == "user" and "content" in message_item:
                    if self._has_injection_content(message_item.get("content")):
                        injected_messages.append(message_item)
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
            return bool(cast(list[Any], content))
        return True

    @require_read_evidence
    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages, compaction = self._initial_transcript_and_compaction(spec)
        context = AgentRunHookContext(messages=deepcopy(messages))
        llm_usage_source_token = bind_llm_usage_source(
            spec.llm_usage_source or source_from_session_key(spec.session_key)
        )

        try:
            await hook.before_run(context)
            try:
                result = await self._run_core(spec, hook, messages, compaction)
            except (ContextBudgetError, RecoveryPacketError, EvidencePersistenceError,
                    SummaryTransactionError) as exc:
                self._log_event(spec, "runner.context.safety_failure", error=str(exc))
                self._log_event(spec, "runner.context.budget_exhausted" if isinstance(exc, ContextBudgetError) else "runner.context.safety_stop")
                result = AgentRunResult(
                    final_content=str(exc), messages=messages, stop_reason="error",
                    error=str(exc), context_safety_failure=True,
                )
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
            context.usage = result.usage
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
            try:
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
            finally:
                reset_llm_usage_source(llm_usage_source_token)

    @staticmethod
    def _initial_transcript_and_compaction(
        spec: AgentRunSpec,
    ) -> tuple[list[dict[str, Any]], ContextCompactionState | None]:
        """Build the initial transcript and its optional compaction state."""
        transcript_input = spec.transcript_input
        if transcript_input is not None:
            if spec.initial_messages is not None:
                raise ValueError("provide either transcript_input or initial_messages, not both")
            transcript_builder = spec.transcript_builder
            if transcript_builder is None:
                raise ValueError("transcript_builder is required with transcript_input")
            return ContextCompactionState.from_transcript(
                transcript_input,
                transcript_builder,
                spec.consolidate_history,
                spec.consolidate_provider_compaction,
            )
        if spec.initial_messages is None:
            raise ValueError("initial_messages is required without transcript_input")
        if spec.consolidate_history is not None:
            raise ValueError("consolidate_history requires transcript_input")
        return list(spec.initial_messages), None

    async def _run_core(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
        compaction: ContextCompactionState | None,
    ) -> AgentRunResult:
        runtime = self._resolve_runtime(spec)
        final_content: str | None = None
        tools_used: list[str] = []
        usage: LLMUsage | None = None
        round_usages: list[LLMUsage] = []
        error: str | None = None
        failure_error_kind: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        # Per-turn throttle for repeated attempts against the same outside target.
        workspace_violation_counts: dict[str, int] = {}
        empty_content_retries = 0
        # Segments from one uninterrupted length-recovery chain. Tool work or
        # injected user input starts a new logical answer and clears the chain.
        length_recovery_parts: list[str] = []
        had_injections = False
        injection_cycles = 0
        pending_stream_content: str | None = None
        tool_batch_history: list[tuple[tuple[str, str], ...]] = []
        warned_tool_loops: set[tuple[tuple[tuple[str, str], ...], ...]] = set()
        tool_digests: dict[str, ToolDigest] = {}
        context_safety_failure = False
        governance_metrics = spec.governance_metrics = GovernanceMetrics()
        spec.summary_injected = False
        latest_context_usage: dict[str, int] = {}
        native_context = uses_native_context(runtime.provider)
        conversation_state = ProviderConversationStateController(
            provider=runtime.provider,
            model=runtime.model,
            messages=messages,
            state=spec.provider_state,
            session_id=spec.session_key,
        )
        governance_config = ContextGovernanceConfig(
            provider=runtime.provider,
            model=runtime.model,
            tools=spec.tools,
            workspace=spec.workspace,
            session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            context_window_tokens=runtime.context_window_tokens,
            max_tokens=runtime.generation.max_tokens,
            context_strategy=spec.context_strategy,
            context_block_limit=spec.context_block_limit,
            blocking_compaction_trigger_ratio=spec.blocking_compaction_trigger_ratio,
            blocking_compaction_target_ratio=spec.blocking_compaction_target_ratio,
            data_dir=spec.data_dir,
        )
        request_state = ModelRequestState(
            config=governance_config,
            conversation=conversation_state,
            # Native preparation owns the stable projection of the raw transcript.
            compaction=None if native_context else compaction,
            events=spec.events,
            governance_state_reader=spec.context_state_reader,
            tool_digests=tool_digests,
        )
        turn_objective_anchor = self._turn_objective_anchor(spec.turn_objective)
        turn_objective_anchored = False

        for iteration in range(spec.max_iterations):
            self._log_event(spec, "runner.iteration.start", iteration=iteration, messages=len(messages))
            context = AgentHookContext(
                iteration=iteration,
                messages=messages,
                session_key=spec.session_key,
            )
            await hook.before_iteration(context)
            request_message_count = len(messages)
            request_messages = (
                request_state.compaction.request_messages(messages)
                if request_state.compaction is not None
                else messages
            )
            request_supplements = None
            if (
                iteration > 0
                and turn_objective_anchor is not None
                and (request_state.compaction is None or not turn_objective_anchored)
            ):
                request_supplements = [turn_objective_anchor]
            try:
                response, raw_usage = await self._request_model(
                    spec,
                    request_messages,
                    hook,
                    context,
                    request_state=request_state,
                    transcript=messages,
                    request_supplements=request_supplements,
                )
                if request_supplements is not None and request_state.compaction is not None:
                    turn_objective_anchored = True
            except (ContextBudgetError, RecoveryPacketError, EvidencePersistenceError, SummaryTransactionError) as exc:
                context_safety_failure = True
                final_content = error = str(exc)
                stop_reason = "error"
                self._log_event(spec, "runner.context.safety_failure", error=error)
                self._log_event(spec, "runner.context.budget_exhausted" if isinstance(exc, ContextBudgetError) else "runner.context.safety_stop")
                break
            assert request_state.messages is not None
            messages_for_model = request_state.messages
            conversation_state.observe_response(response, messages)
            request_state.accept_request(
                messages_for_model,
                raw_boundary=request_message_count,
            )
            context.response = response
            context.tool_calls = list(response.tool_calls)

            original_content = response.content
            reasoning_text, cleaned_content = extract_reasoning(
                response.reasoning_content,
                response.thinking_blocks,
                response.content,
            )
            response.content = cleaned_content
            round_usages.append(raw_usage)
            context.usage = raw_usage
            latest_context_usage = with_context_usage(
                raw_usage, response.provider_diagnostics or {},
                lambda: estimate_prompt_tokens_chain(runtime.provider, runtime.model,
                                                      messages_for_model, spec.tools.get_definitions())[0],
            )
            governance_metrics.observe_usage(latest_context_usage)
            self._log_event(spec, "runner.model.response", iteration=iteration,
                            finish_reason=response.finish_reason,
                            should_execute_tools=response.should_execute_tools,
                            tool_calls=[tc.name for tc in response.tool_calls],
                            usage=latest_context_usage, provider_diagnostics=response.provider_diagnostics,
                            error_content=response.content if response.finish_reason == "error" else None)
            usage = self._merge_usage(usage, raw_usage)
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
                        if any(name == "subagent_control" for batch in loop_pattern for name, _ in batch):
                            loop_correction += (
                                " If a child is still running and independent work is finished, "
                                "use subagent_control(action='wait', task_id='<actual-id>') once "
                                "to let the host wait for its receipt. Do not poll unchanged plans."
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
                    response_items=response.response_items,
                )
                assistant_message = conversation_state.project_response_message(
                    assistant_message,
                    response,
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
                                "model": runtime.model,
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
                results, new_events = await execute_tool_calls(
                    spec.tools,
                    response.tool_calls,
                    concurrent=spec.concurrent_tools,
                    external_lookup_counts=external_lookup_counts,
                    workspace_violation_counts=workspace_violation_counts,
                    hook=hook,
                    context=context,
                    event_logger=lambda event, fields: self._log_event(spec, event, **fields),
                )
                fatal_error = next((RuntimeError(event.get("detail", "tool failed"))
                                    for event in new_events if event.get("status") == "error"), None) if spec.fail_on_tool_error else None
                self._log_event(spec, "runner.tools.completed", iteration=iteration, events=new_events,
                                fatal_error=str(fatal_error) if fatal_error else None)
                tool_events.extend(new_events)
                tools_used.extend(
                    tool_call.name
                    for tool_call, event in zip(response.tool_calls, new_events)
                    if event.get("status") == "ok"
                )
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                await hook.after_execute_tools(context)
                completed_tool_results: list[dict[str, Any]] = []
                evidence_error: Exception | None = None
                for tool_call, result, event in zip(response.tool_calls, results, new_events):
                    normalized = result
                    try:
                        # 来源只取运行时实际保存结果，不解析不可信回执内的“已保存路径”。
                        artifact_locator = persist_tool_evidence(governance_config, result)
                        normalized = self.context_governor.normalize_tool_result(
                            governance_config, tool_call.id, tool_call.name, result,
                            artifact_locator=artifact_locator,
                        )
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
                            spec.context_delta_callback({
                                "tool_digest": digest,
                                "evidence": evidence,
                            })
                    except Exception as exc:
                        # 工具已经执行：保留整批回执，不重新执行、不走通用裁剪兜底。
                        normalized = result
                        evidence_error = EvidencePersistenceError(
                            "工具证据保存或登记失败，已保留原始回执；停止后续模型请求。"
                        )
                        logger.opt(exception=exc).error(
                            "Tool evidence failed for {}: {}", tool_call.id, type(exc).__name__
                        )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": normalized,
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                if evidence_error is not None:
                    context_safety_failure = True
                    await self._emit_checkpoint(spec, {
                        "phase": "tools_completed", "iteration": iteration, "model": runtime.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results, "pending_tool_calls": [],
                    })
                    error = final_content = str(evidence_error)
                    stop_reason = "error"
                    self._append_final_message(messages, final_content)
                    break
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
                checkpoint_model_messages = (
                    self.context_governor.prepare_messages_for_model(
                        governance_config,
                        messages,
                    )
                    if response.provider_state is not None
                    else None
                )
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": runtime.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                        "provider_state": conversation_state.checkpoint(
                            messages,
                            model_messages=checkpoint_model_messages,
                        ),
                    },
                )
                empty_content_retries = 0
                length_recovery_parts.clear()
                # Checkpoint 1: drain injections after tools, before next LLM call
                _drained, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after tool execution",
                    wait_for_subagents=should_wait_for_subagent_result(response.tool_calls, results, new_events),
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
            if (
                response.finish_reason
                not in {"error", "length", "refusal", "content_filter"}
                and is_blank_text(clean)
            ):
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
                response = await self._request_finalization_retry(
                    spec,
                    messages_for_model,
                    request_state=request_state,
                    transcript=messages,
                )
                retry_usage = self._record_request_usage(spec, request_state, response)
                round_usages.append(retry_usage)
                usage = self._merge_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = raw_usage
                context.tool_calls = list(response.tool_calls)
                original_content = response.content
                clean = hook.finalize_content(context, response.content)

            if response.finish_reason == "length":
                if len(length_recovery_parts) < _MAX_LENGTH_RECOVERIES:
                    length_recovery_parts.append(
                        _restore_outer_whitespace(clean or "", original_content)
                    )
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        len(length_recovery_parts),
                        _MAX_LENGTH_RECOVERIES,
                    )
                    if hook.wants_streaming():
                        context.stream_continues_current_message = True
                        await hook.on_stream_end(context, resuming=True)
                    messages.append(conversation_state.project_response_message(
                        build_assistant_message(
                            clean,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                    response_items=response.response_items,
                        ),
                        response,
                    ))
                    messages.append(build_length_recovery_message(clean or ""))
                    await hook.after_iteration(context)
                    continue

            # Some streaming providers recover with a complete response but no
            # content deltas. When an earlier length segment is already visible,
            # emit this terminal segment into the same stream; otherwise the
            # regular full response would duplicate the visible prefix.
            if (
                length_recovery_parts
                and hook.wants_streaming()
                and not context.streamed_content
                and response.finish_reason != "error"
                and not is_blank_text(clean)
            ):
                await hook.on_stream(
                    context,
                    _restore_outer_whitespace(clean or "", original_content),
                )
                context.streamed_content = True

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                    response_items=response.response_items,
                )
                assistant_message = conversation_state.project_response_message(
                    assistant_message,
                    response,
                )

            # Check for mid-turn injections BEFORE signaling stream end.
            # If injections are found we keep the stream alive (resuming=True)
            # so streaming channels don't prematurely finalize the card.
            should_continue, injection_cycles = await self._try_drain_injections(
                spec, messages, assistant_message, injection_cycles,
                conversation_state=conversation_state,
                phase="after final response",
                iteration=iteration,
                allow_continuation=(
                    response.finish_reason not in {"refusal", "content_filter"}
                ),
                wait_at_terminal=(
                    assistant_message is not None
                    and response.finish_reason
                    not in {"error", "length", "refusal", "content_filter"}
                ),
            )
            if should_continue:
                had_injections = True

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=should_continue)

            if should_continue:
                length_recovery_parts.clear()
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
                    length_recovery_parts.clear()
                    continue
                failure_error_kind = LLMProvider.public_error_kind(response)
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
                    length_recovery_parts.clear()
                    continue
                break

            messages.append(
                assistant_message
                or conversation_state.project_response_message(
                    build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    response_items=response.response_items,
                    ),
                    response,
                )
            )
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": runtime.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                    "provider_state": conversation_state.checkpoint(messages),
                },
            )
            if length_recovery_parts:
                final_content = (
                    "".join(length_recovery_parts)
                    + _restore_outer_whitespace(clean or "", original_content)
                ).strip()
            else:
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
            terminal_content = None
            if spec.finalize_on_max_iterations:
                terminal_content, usage = await self._try_finalize_after_max_iterations(
                    spec,
                    hook,
                    messages,
                    usage,
                    request_state=request_state,
                    round_usages=round_usages,
                )
            if terminal_content is None:
                terminal_content = self._max_iterations_fallback(spec)
            if length_recovery_parts:
                terminal_tail = f"\n\n{terminal_content.lstrip()}"
                final_content = (
                    "".join(length_recovery_parts).rstrip() + terminal_tail
                ).strip()
                pending_stream_content = terminal_tail
            else:
                final_content = terminal_content
            self._append_final_message(messages, terminal_content)

        if request_state.transaction is not None:
            # Sum summary consumption while keeping the final execution request's occupancy.
            usage = self._merge_usage(request_state.transaction.usage, usage)
        self._log_event(spec, "runner.context.turn_summary", model_requests=len(round_usages),
                        prompt_tokens=usage.input_tokens if usage else 0,
                        completion_tokens=usage.output_tokens if usage else 0,
                        **governance_metrics.summary(native=native_context),
                        context_input_tokens=latest_context_usage.get("context_input_tokens", 0),
                        context_input_estimated=latest_context_usage.get("context_input_estimated", 0))
        return AgentRunResult(
            context_safety_failure=context_safety_failure,
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            round_usages=round_usages,
            stop_reason=stop_reason,
            error=error,
            failure_error_kind=failure_error_kind,
            tool_events=tool_events,
            had_injections=had_injections or spec.summary_injected,
            pending_stream_content=pending_stream_content,
            provider_state=conversation_state.finish(messages),
            summary_checkpoint=(
                request_state.compaction.summary_checkpoint
                if request_state.compaction is not None
                else None
            ),
            provider_compaction_applied=request_state.provider_compaction_applied,
        )

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        runtime = self._resolve_runtime(spec)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": runtime.model,
            "retry_mode": spec.provider_retry_mode,
        }
        if spec.retry_wait_callback is not None:
            kwargs["on_retry_wait"] = spec.retry_wait_callback
        def log_retry(fields: dict[str, Any]) -> None:
            self._log_event(spec, "runner.model.retry", **fields)
        kwargs["on_retry_event"] = log_retry
        if getattr(runtime.provider, "supports_request_context", False) is True:
            request_context: dict[str, Any] = {"session_key": spec.session_key or "default", "turn_id": spec.turn_id}
            kwargs["request_context"] = request_context
            if uses_native_context(runtime.provider):
                request_context["native_context"] = {
                    "context_window_tokens": runtime.context_window_tokens,
                    "context_block_limit": spec.context_block_limit,
                    "max_tokens": runtime.generation.max_tokens,
                }
        generation = runtime.generation
        kwargs["temperature"] = generation.temperature
        kwargs["max_tokens"] = generation.max_tokens
        kwargs["reasoning_effort"] = generation.reasoning_effort
        return kwargs

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
        *,
        request_state: ModelRequestState,
        malformed_retry: bool = False,
        timeout_retry: bool = False,
        budget_retry: bool = False,
        overflow_retry: bool = False,
        request_supplements: list[dict[str, Any]] | None = None,
        transcript: list[dict[str, Any]] | None,
    ) -> tuple[LLMResponse, LLMUsage]:
        runtime = self._resolve_runtime(spec)
        tool_definitions = spec.tools.get_definitions()
        source_messages = messages
        previous_version = request_state.transaction.version if request_state.transaction else 0
        previous_checkpoint = (
            request_state.compaction.summary_checkpoint
            if request_state.compaction is not None
            else None
        )
        before_projection = list(messages)
        before_context = self.context_governor.context_metrics(messages, tool_definitions)
        try:
            messages, provider_context = await self.context_governor.prepare_request(
                request_state,
                messages,
                tool_definitions=tool_definitions,
                transcript=transcript,
            )
        except (ContextBudgetError, RecoveryPacketError, EvidencePersistenceError, SummaryTransactionError):
            raise
        except Exception as exc:
            if spec.context_strategy == "transactional" and not uses_native_context(runtime.provider):
                raise SummaryTransactionError("Context governance failed; the complete transcript is retained.") from exc
            raise

        checkpoint_advanced = (
            request_state.compaction is not None
            and request_state.compaction.summary_checkpoint is not previous_checkpoint
        )
        if (
            request_state.transaction is not None
            and request_state.transaction.version != previous_version
        ) or checkpoint_advanced:
            injections = await self._drain_injections(spec)
            if injections:
                if checkpoint_advanced:
                    source_messages = [*messages, *deepcopy(injections)]
                else:
                    source_messages.extend(deepcopy(injections))
                spec.summary_injected = True
                messages, provider_context = await self.context_governor.prepare_request(
                    request_state, source_messages, tool_definitions=tool_definitions,
                    transcript=transcript,
                )
        if request_supplements:
            fit_request = (self.context_governor.fit_to_budget
                           if spec.context_strategy == "legacy"
                           else self.context_governor.ensure_request_fits)
            try:
                messages = fit_request(request_state.config, [*messages, *request_supplements],
                                       tool_definitions=tool_definitions)
            except ContextWindowExceededError as exc:
                if spec.context_strategy == "transactional":
                    raise ContextBudgetError(str(exc)) from exc
                raise
            provider_context = request_state.conversation.independent_request_context(
                context_window_tokens=request_state.config.context_window_tokens,
            )
            request_state.messages = messages
        after_context = self.context_governor.context_metrics(messages, tool_definitions)
        saved = {key: max(0, value - after_context.get(key, 0)) for key, value in before_context.items()}
        saved_total = saved["total"]
        spec.governance_metrics.observe_projection(saved_total)
        native = uses_native_context(runtime.provider)
        budget = resolve_context_budget(runtime.provider, runtime.context_window_tokens,
                                        runtime.generation.max_tokens, spec.context_block_limit)
        self._log_event(
            spec, "runner.context.governance",
            context_scope="native_checkpoint_copy" if native else "local_model_copy",
            compaction_owner="provider" if native else "nanobot",
            strategy="native" if native else spec.context_strategy,
            before=before_context, after=after_context, saved=saved, saved_total=saved_total,
            budget=budget.diagnostics(), metrics_source="local_estimate",
            projection_metrics_semantics="local_copy_not_remote_or_billing",
            compacted_tool_results=projection_changes(before_projection, messages)["projection_changed_tool_results"],
            digested_tool_results=0,
            **projection_changes(before_projection, messages),
        )
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=tool_definitions,
        )
        wants_streaming = hook.wants_streaming()
        wants_progress_streaming = (not wants_streaming and spec.stream_progress_deltas
                                    and spec.progress_callback is not None
                                    and getattr(runtime.provider, "supports_progress_deltas", False) is True)

        active_hosted_tools: dict[str, dict[str, Any]] = {}
        native_reasoning_open = False
        native_reasoning_close_task: asyncio.Task[None] | None = None
        request_started_at = 0.0
        first_output_at: float | None = None
        generation_started_at: float | None = None
        generation_elapsed_s = 0.0

        def _generation_delta(delta: str) -> None:
            nonlocal first_output_at, generation_started_at
            if not delta:
                return
            now = time.perf_counter()
            if first_output_at is None:
                first_output_at = now
            if generation_started_at is None:
                generation_started_at = now

        def _pause_generation() -> None:
            nonlocal generation_elapsed_s, generation_started_at
            if generation_started_at is None:
                return
            generation_elapsed_s += max(0.0, time.perf_counter() - generation_started_at)
            generation_started_at = None

        async def _close_native_reasoning() -> None:
            nonlocal native_reasoning_open, native_reasoning_close_task
            if native_reasoning_close_task is None:
                if not native_reasoning_open:
                    return
                native_reasoning_open = False
                native_reasoning_close_task = asyncio.create_task(
                    hook.emit_reasoning_end()
                )

            close_task = native_reasoning_close_task
            cancellation: asyncio.CancelledError | None = None
            while not close_task.done():
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
            try:
                close_task.result()
            finally:
                if native_reasoning_close_task is close_task:
                    native_reasoning_close_task = None
            if cancellation is not None:
                raise cancellation

        async def _provider_tool_event(event: dict[str, Any]) -> None:
            if event.get("kind") != "hosted_tool":
                return
            await _close_native_reasoning()
            await hook.on_provider_tool_event(context, event)
            call_id = event.get("call_id")
            if not call_id:
                return
            call_id = str(call_id)
            if event.get("phase") == "start":
                active_hosted_tools[call_id] = dict(event)
            elif event.get("phase") in {"end", "error"}:
                active_hosted_tools.pop(call_id, None)

        if wants_streaming or wants_progress_streaming:
            thinking_buf = ""

            async def _stream(delta: str) -> None:
                _generation_delta(delta)
                if delta:
                    context.streamed_content = True
                    await _close_native_reasoning()
                if wants_streaming:
                    await hook.on_stream(context, delta)
                elif spec.progress_callback is not None:
                    await spec.progress_callback(delta)

            async def _thinking(delta: str) -> None:
                nonlocal native_reasoning_open, thinking_buf
                if not delta:
                    return
                _generation_delta(delta)
                prev_clean = strip_reasoning_tags(thinking_buf)
                thinking_buf += delta
                new_clean = strip_reasoning_tags(thinking_buf)
                incremental = new_clean[len(prev_clean):]
                if incremental:
                    context.streamed_reasoning = True
                    native_reasoning_open = True
                    await hook.emit_reasoning(incremental)

            async def _stream_recover() -> None:
                _pause_generation()
                await _close_native_reasoning()
                await hook.on_stream_end(context, resuming=True)

            coro = runtime.provider.chat_stream_with_retry(
                **kwargs,
                provider_context=provider_context,
                on_content_delta=_stream,
                on_thinking_delta=_thinking,
                on_tool_call_delta=_provider_tool_event,
                on_stream_recover=_stream_recover,
            )
        else:
            coro = runtime.provider.chat_stream_with_retry(
                **kwargs,
                provider_context=provider_context,
            )

        # Providers bound the wait for each stream event, including reasoning and tool deltas.
        request_started_at = time.perf_counter()
        timeout_s = spec.llm_timeout_s
        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get("NANOBOT_LLM_TIMEOUT_S", "300"))
            except (ValueError, TypeError):
                timeout_s = 300.0
        self._log_event(spec, "runner.model.request", model=runtime.model,
                        messages=len(messages), tools=len(tool_definitions),
                        streaming=wants_streaming, progress_streaming=wants_progress_streaming,
                        timeout_s=timeout_s)
        try:
            response = await asyncio.wait_for(coro, timeout_s) if timeout_s > 0 else await coro
        except asyncio.TimeoutError:
            _pause_generation()
            await _close_native_reasoning()
            self._log_event(spec, "runner.model.timeout", timeout_s=timeout_s)
            if not timeout_retry:
                self._log_event(spec, "runner.model.timeout_recovery", timeout_s=timeout_s)
                if wants_streaming:
                    await hook.on_stream_end(context, resuming=True)
                return await self._request_model(spec, source_messages, hook, context,
                                                 request_state=request_state, transcript=transcript,
                                                 malformed_retry=malformed_retry, timeout_retry=True,
                                                 request_supplements=request_supplements, budget_retry=budget_retry,
                                                 overflow_retry=overflow_retry)
            response = LLMResponse(content=f"Error calling LLM: timed out after {timeout_s:g}s",
                                   finish_reason="error", error_kind="timeout")
        except asyncio.CancelledError:
            _pause_generation()
            await _close_native_reasoning()
            raise
        _pause_generation()
        await _close_native_reasoning()
        self._log_event(spec, "runner.model.request.done", finish_reason=response.finish_reason,
                        error_kind=response.error_kind,
                        error_status_code=response.error_status_code,
                        error_code=response.error_code,
                        error_retry_after_s=response.error_retry_after_s,
                        error_should_retry=response.error_should_retry,
                        error_content=response.content if response.finish_reason == "error" else None)
        if first_output_at is not None:
            response.ttft_ms = max(0, round((first_output_at - request_started_at) * 1000))
        if generation_elapsed_s > 0:
            response.generation_ms = max(1, round(generation_elapsed_s * 1000))
        await self.context_governor.summarize_provider_compaction(
            request_state,
            response,
            current_request_boundary=(len(transcript) if transcript is not None else None),
        )
        request_state.provider_compaction_applied |= response.provider_compaction_applied
        round_usage = self._record_request_usage(spec, request_state, response)
        if spec.context_strategy == "transactional" and is_context_overflow(response):
            if overflow_retry or context.streamed_content:
                raise ContextBudgetError(
                    "Provider still rejected the safely reprojected context as too large."
                )
            estimated, _ = estimate_prompt_tokens_chain(
                runtime.provider, runtime.model, messages, tool_definitions,
            )
            emergency_budget = max(1024, int(estimated * 0.75))
            if request_state.config.context_block_limit is not None:
                emergency_budget = min(
                    emergency_budget, request_state.config.context_block_limit,
                )
            request_state.config = replace(
                request_state.config, context_block_limit=emergency_budget,
            )
            self._log_event(
                spec, "runner.context.overflow_reproject", budget=emergency_budget,
            )
            retry_response, retry_usage = await self._request_model(
                spec, source_messages, hook, context, request_state=request_state,
                transcript=transcript, request_supplements=request_supplements,
                overflow_retry=True,
            )
            return retry_response, round_usage + retry_usage
        if (spec.context_strategy == "legacy" and is_context_overflow(response)
                and not overflow_retry and not context.streamed_content):
            estimated, _ = estimate_prompt_tokens_chain(runtime.provider, runtime.model, messages, tool_definitions)
            emergency_budget = max(1024, int(estimated * 0.75))
            if request_state.config.context_block_limit is not None:
                emergency_budget = min(emergency_budget, request_state.config.context_block_limit)
            request_state.config = replace(request_state.config, context_block_limit=emergency_budget)
            reduced = self.context_governor.snip_history(request_state.config, messages)
            reduced = ContextGovernor.backfill_missing_tool_results(ContextGovernor.drop_orphan_tool_results(reduced))
            if reduced != messages:
                retry_response, retry_usage = await self._request_model(
                    spec, reduced, hook, context, request_state=request_state,
                    transcript=None, overflow_retry=True,
                )
                return retry_response, round_usage + retry_usage
        if not budget_retry and self._claims_false_tool_budget(response):
            correction = [{"role": "user", "content": (
                "[Runtime correction] Tool access is available for this new turn. "
                "Earlier user turns do not consume the current turn's tool budget. "
                "Use the available tools if needed; do not repeat the exhaustion claim."
            )}]
            self._log_event(spec, "runner.model.false_tool_budget_retry")
            if wants_streaming:
                await hook.on_stream_end(context, resuming=True)
            retry_response, retry_usage = await self._request_model(
                spec, source_messages, hook, context, request_state=request_state,
                transcript=None, budget_retry=True, request_supplements=correction,
            )
            return retry_response, round_usage + retry_usage
        # chat_stream_with_retry may recover internally, so only fail unfinished
        # hosted calls after the provider returns its final error response.
        if response.finish_reason == "error":
            for event in list(active_hosted_tools.values()):
                await _provider_tool_event({
                    **event,
                    "phase": "error",
                    "result": None,
                    "error": response.content
                    or "Model request failed before the provider-hosted tool completed.",
                })
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
                [], response.content,
            )
            retry_response, retry_usage = await self._request_model(
                spec, source_messages, hook, context,
                request_state=request_state,
                malformed_retry=True, request_supplements=retry_messages,
                transcript=None,
            )
            return retry_response, round_usage + retry_usage
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and malformed_retry
        ):
            logger.warning(
                "Malformed tool calls persisted after retry; falling back to no-tools request",
            )
            fallback_messages = self._malformed_tool_call_retry_messages(
                [], response.content,
            )
            fallback_response = await self._request_no_tools(
                spec,
                source_messages if spec.context_strategy == "transactional" else [*source_messages, *fallback_messages],
                request_state=request_state,
                request_supplements=fallback_messages if spec.context_strategy == "transactional" else None,
            )
            fallback_usage = self._record_request_usage(
                spec,
                request_state,
                fallback_response,
            )
            return fallback_response, round_usage + fallback_usage
        return response, round_usage

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
        # The opaque candidate still contains every raw function_call item.
        # Advancing it after dropping even one call would replay an unmatched
        # call without a corresponding tool output on the next request.
        response.provider_state = None
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
        *,
        request_state: ModelRequestState,
        transcript: list[dict[str, Any]],
    ) -> LLMResponse:
        retry_messages = self._finalization_retry_messages(messages)
        response = await self._request_no_tools(
            spec,
            transcript if spec.context_strategy == "transactional" else retry_messages,
            request_state=request_state,
            transcript=transcript,
            request_supplements=retry_messages[-1:] if spec.context_strategy == "transactional" else None,
        )
        request_state.conversation.observe_response(
            response,
            transcript,
            adopt_candidate_state=False,
        )
        return response

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
        usage: LLMUsage | None,
        *,
        request_state: ModelRequestState,
        round_usages: list[LLMUsage],
    ) -> tuple[str | None, LLMUsage | None]:
        compaction = request_state.compaction
        request_messages = (
            compaction.request_messages(messages)
            if compaction is not None
            else messages
        )
        retry_messages = self._budget_exhausted_finalization_messages(request_messages)
        self._log_event(spec, "runner.max_iterations.finalization.start")
        try:
            response = await self._request_no_tools(
                spec,
                messages if spec.context_strategy == "transactional" else retry_messages,
                request_state=request_state,
                transcript=messages if compaction is not None else None,
                request_supplements=retry_messages[-1:] if spec.context_strategy == "transactional" else None,
            )
        except (ContextBudgetError, RecoveryPacketError, EvidencePersistenceError, SummaryTransactionError):
            raise
        except Exception:
            self._log_event(spec, "runner.max_iterations.finalization.failed", reason="request_error")
            logger.exception(
                "Budget-exhausted finalization failed for {}; using fallback",
                spec.session_key or "default",
            )
            return None, usage

        self._log_event(spec, "runner.max_iterations.finalization.response",
                        finish_reason=response.finish_reason, has_tool_calls=response.has_tool_calls)
        raw_usage = self._record_request_usage(spec, request_state, response)
        round_usages.append(raw_usage)
        usage = self._merge_usage(usage, raw_usage)
        if response.finish_reason == "error" or response.has_tool_calls:
            logger.warning(
                "Budget-exhausted finalization returned finish_reason='{}' "
                "with {} tool call(s) for {}; using fallback",
                response.finish_reason,
                len(response.tool_calls),
                spec.session_key or "default",
            )
            return None, usage

        context = AgentHookContext(
            iteration=spec.max_iterations,
            messages=messages,
            response=response,
            usage=raw_usage,
            session_key=spec.session_key,
        )
        clean = hook.finalize_content(context, response.content)
        if is_blank_text(clean):
            return None, usage
        self._log_event(spec, "runner.max_iterations.finalization.success")
        return clean, usage

    async def _request_no_tools(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        request_state: ModelRequestState,
        transcript: list[dict[str, Any]] | None = None,
        request_supplements: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        runtime = self._resolve_runtime(spec)
        try:
            messages, provider_context = await self.context_governor.prepare_request(
                request_state,
                messages,
                tool_definitions=None,
                transcript=transcript,
            )
        except (ContextBudgetError, RecoveryPacketError, EvidencePersistenceError, SummaryTransactionError):
            raise
        except Exception as exc:
            if spec.context_strategy == "transactional" and not uses_native_context(runtime.provider):
                raise SummaryTransactionError("Finalization context preparation failed; the transcript is retained.") from exc
            raise
        if request_supplements:
            try:
                messages = self.context_governor.ensure_request_fits(
                    request_state.config, [*messages, *request_supplements], tool_definitions=None,
                )
            except ContextWindowExceededError as exc:
                raise ContextBudgetError(str(exc)) from exc
            provider_context = request_state.conversation.independent_request_context(
                context_window_tokens=request_state.config.context_window_tokens,
            )
            request_state.messages = messages
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=None,
        )
        request = runtime.provider.chat_stream_with_retry(**kwargs, provider_context=provider_context)
        timeout_s = spec.llm_timeout_s
        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get("NANOBOT_LLM_TIMEOUT_S", "300"))
            except (ValueError, TypeError):
                timeout_s = 300.0
        response = await asyncio.wait_for(request, timeout_s) if timeout_s > 0 else await request
        if spec.context_strategy == "transactional" and is_context_overflow(response):
            raise ContextBudgetError("Finalization exceeds the provider context budget; stopping safely.")
        await self.context_governor.summarize_provider_compaction(
            request_state,
            response,
            current_request_boundary=(len(transcript) if transcript is not None else None),
        )
        request_state.provider_compaction_applied |= response.provider_compaction_applied
        return response

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
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> LLMUsage:
        runtime = self._resolve_runtime(spec)
        usage = response.usage
        if isinstance(usage, dict):
            usage = cast(dict[str, Any], usage)
            # Older custom providers return Chat-style token dictionaries.
            usage = LLMUsage.reported(input_tokens=max(0, int(usage.get("prompt_tokens", 0))),
                                      output_tokens=max(0, int(usage.get("completion_tokens", 0))),
                                      total_tokens=max(0, int(usage.get("total_tokens", 0))),
                                      cache_read_tokens=usage.get("cached_tokens"),
                                      cache_write_tokens=usage.get("cache_creation_tokens"))
        if response.finish_reason == "error":
            if usage is None or usage.total_tokens == 0:
                return LLMUsage.empty_request().with_timing(generation_ms=response.generation_ms, ttft_ms=response.ttft_ms)
        elif usage is None or usage.total_tokens == 0:
            usage = self._estimate_response_usage(
                spec,
                messages,
                response,
                tool_definitions=tool_definitions,
            )
        context_usage = with_context_usage(
            usage, response.provider_diagnostics or {},
            lambda: estimate_prompt_tokens_chain(runtime.provider, runtime.model, messages, tool_definitions)[0],
        )
        response.provider_diagnostics = dict(response.provider_diagnostics or {})
        response.provider_diagnostics["context_input_tokens"] = context_usage["context_input_tokens"]
        response.provider_diagnostics["context_input_estimated"] = context_usage["context_input_estimated"]
        usage = replace(usage, context_tokens=context_usage["context_input_tokens"])
        return usage.with_timing(
            generation_ms=response.generation_ms,
            ttft_ms=response.ttft_ms,
        )

    def _record_request_usage(
        self,
        spec: AgentRunSpec,
        state: ModelRequestState,
        response: LLMResponse,
    ) -> LLMUsage:
        assert state.messages is not None
        state.usage = self._usage_or_estimate(
            spec,
            state.messages,
            response,
            tool_definitions=state.tool_definitions,
        )
        return state.usage

    def _estimate_response_usage(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> LLMUsage:
        runtime = self._resolve_runtime(spec)
        prompt_tokens, _ = estimate_prompt_tokens_chain(
            runtime.provider,
            runtime.model,
            messages,
            tool_definitions,
        )
        assistant_message = build_assistant_message(
            response.content or "",
            tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
                    response_items=response.response_items,
        )
        completion_tokens = estimate_message_tokens(assistant_message)
        return LLMUsage.estimated(
            input_tokens=max(0, prompt_tokens),
            output_tokens=max(0, completion_tokens),
        )

    @staticmethod
    def _merge_usage(
        left: LLMUsage | None,
        right: LLMUsage | None,
    ) -> LLMUsage | None:
        if left is None:
            return right
        if right is None:
            return left
        return left + right

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
            if name == "subagent_control":
                try:
                    if json.loads(arguments_json).get("action") == "wait":
                        continue
                except (AttributeError, json.JSONDecodeError):
                    return False
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
