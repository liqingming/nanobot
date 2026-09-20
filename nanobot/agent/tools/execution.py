"""Execute tool calls and turn their outcomes into model observations."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result
from nanobot.providers.base import ToolCallRequest
from nanobot.utils.runtime import (
    repeated_external_lookup_error,
    repeated_workspace_violation_error,
)

_RETRY_HINT = "\n\n[Analyze the error above and try a different approach.]"
# SSRF is a hard security block at the tool boundary, but the agent turn
# should recover conversationally instead of aborting the runtime.
_SSRF_MARKERS: tuple[str, ...] = (
    "internal/private url detected",
    "private/internal address",
    "private address",
)
_SSRF_BOUNDARY_NOTE = (
    "This is a non-bypassable security boundary. Stop trying to access "
    "private/internal URLs. Do not retry with curl, wget, encoded IPs, "
    "alternate DNS, redirects, proxies, or another tool. Ask the user for "
    "local files, logs, screenshots, or an explicit safe public URL instead. "
    "If the user explicitly trusts this private URL, ask them to whitelist "
    "the exact IP/CIDR via tools.ssrfWhitelist."
)
# Non-SSRF boundary markers returned to the model as recoverable tool errors.
_WORKSPACE_VIOLATION_MARKERS: tuple[str, ...] = (
    "outside the configured workspace",
    "outside allowed directory",
    "working_dir is outside",
    "working_dir could not be resolved",
    "path outside working dir",
    "path traversal detected",
)


def _with_retry_hint(payload: str) -> str:
    """Append the recovery hint exactly once."""
    if payload.endswith(_RETRY_HINT):
        return payload
    return payload + _RETRY_HINT


async def execute_tool_calls(
    tools: ToolRegistry,
    tool_calls: list[ToolCallRequest],
    *,
    concurrent: bool,
    external_lookup_counts: dict[str, int],
    workspace_violation_counts: dict[str, int],
    hook: AgentHook,
    context: AgentHookContext,
    event_logger: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Execute one model response's tool calls in stable result order."""
    deferred = _deferred_broad_searches(tool_calls)
    positions = {call.id: index for index, call in enumerate(tool_calls, 1)}

    async def execute_one(call: ToolCallRequest) -> tuple[Any, dict[str, str]]:
        started = time.monotonic()
        error_details: dict[str, str] = {}
        def log(event: str, **fields: Any) -> None:
            if event_logger is not None:
                event_logger(event, {"iteration": context.iteration, "position": positions[call.id],
                                     "tool": call.name, "call_id": call.id, **fields})
        log("runner.tool.audit.start", arguments=call.arguments)
        try:
            if call.id in deferred:
                result = deferred[call.id]
                event = {"name": call.name, "status": "deferred", "detail": result[:120]}
                log("runner.tool.deferred", reason="broader overlapping search")
            else:
                log("runner.tool.start", arguments=call.arguments)
                result, event = await _execute_tool_call(tools, call, external_lookup_counts,
                                                         workspace_violation_counts, hook, context, error_details)
        except asyncio.CancelledError:
            log("runner.tool.audit.end", status="cancelled",
                duration_ms=round((time.monotonic()-started)*1000, 1))
            raise
        except BaseException as exc:
            log("runner.tool.audit.end", status="exception", error_type=type(exc).__name__,
                error=str(exc), duration_ms=round((time.monotonic()-started)*1000, 1))
            raise
        log("runner.tool.audit.end", status=event.get("status"), result_preview=str(result),
            result_chars=len(str(result)), duration_ms=round((time.monotonic()-started)*1000, 1),
            **error_details)
        log("runner.tool.done", detail=event.get("detail"))
        return result, event

    tool_results: list[tuple[Any, dict[str, str]]] = []
    batches = _partition_tool_batches(tools, tool_calls, concurrent=concurrent)
    if event_logger is not None:
        event_logger("runner.tools.start", {
            "iteration": context.iteration, "concurrent_tools": concurrent,
            "batch_tools": [[call.name for call in batch] for batch in batches],
            "batch_call_ids": [[call.id for call in batch] for batch in batches],
        })
    for batch in batches:
        if concurrent and len(batch) > 1:
            tool_results.extend(await asyncio.gather(*(execute_one(call) for call in batch)))
        else:
            for call in batch:
                tool_results.append(await execute_one(call))

    results = [result for result, _event in tool_results]
    events = [event for _result, event in tool_results]
    return results, events


async def _execute_tool_call(
    tools: ToolRegistry,
    tool_call: ToolCallRequest,
    external_lookup_counts: dict[str, int],
    workspace_violation_counts: dict[str, int],
    hook: AgentHook,
    context: AgentHookContext,
    error_details: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    lookup_error = repeated_external_lookup_error(
        tool_call.name,
        tool_call.arguments,
        external_lookup_counts,
    )
    if lookup_error:
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": "repeated external lookup blocked",
        }
        return _with_retry_hint(lookup_error), event

    prepare_call = cast(
        Callable[[str, Any], object] | None,
        getattr(tools, "prepare_call", None),
    )
    tool, params, prep_error = None, tool_call.arguments, None
    if callable(prepare_call):
        prepared = prepare_call(tool_call.name, tool_call.arguments)
        if isinstance(prepared, tuple):
            prepared_tuple = cast(tuple[object, ...], prepared)
            if len(prepared_tuple) == 3:
                tool, params, prep_error = cast(tuple[Any, Any, str | None], prepared_tuple)
    if prep_error:
        payload = _with_retry_hint(prep_error)
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": prep_error.split(": ", 1)[-1][:120],
        }
        handled = _classify_violation(
            raw_text=prep_error,
            soft_payload=payload,
            event=event,
            tool_call=tool_call,
            workspace_violation_counts=workspace_violation_counts,
        )
        if handled is not None:
            return handled
        return payload, event

    await hook.before_execute_tool(context, tool_call, tool, params)
    try:
        if tool is not None:
            execution = tool.execute(**params)
            timeout_s = getattr(tool, "execution_timeout_s", None)
            result = await asyncio.wait_for(execution, timeout_s) if timeout_s is not None else await execution
        else:
            result = await tools.execute(tool_call.name, params)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        timeout_s = getattr(tool, "execution_timeout_s", None)
        payload = (
            "[Tool execution checkpoint ? model decision]\n"
            f"The {tool_call.name} call exceeded its absolute {timeout_s}s guard. "
            "Decide whether to stop, narrow/change approach, or retry explicitly. "
            "Do not blindly repeat the identical broad call without a scan cursor."
        )
        await hook.on_execute_tool_error(context, tool_call, tool, params, payload)
        return payload, {"name": tool_call.name, "status": "checkpoint", "detail": payload[:120]}
    except Exception as exc:
        if error_details is not None:
            error_details.update(error_type=type(exc).__name__, error=str(exc))
        await hook.on_execute_tool_error(context, tool_call, tool, params, exc)
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": str(exc),
        }
        payload = _with_retry_hint(f"Error: {type(exc).__name__}: {exc}")
        handled = _classify_violation(
            raw_text=str(exc),
            soft_payload=payload,
            event=event,
            tool_call=tool_call,
            workspace_violation_counts=workspace_violation_counts,
        )
        if handled is not None:
            return handled
        return payload, event

    if is_tool_error_result(result):
        await hook.on_execute_tool_error(context, tool_call, tool, params, result)
        payload = _with_retry_hint(result)
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": result.replace("\n", " ").strip()[:120],
        }
        handled = _classify_violation(
            raw_text=result,
            soft_payload=payload,
            event=event,
            tool_call=tool_call,
            workspace_violation_counts=workspace_violation_counts,
        )
        if handled is not None:
            return handled
        return payload, event

    await hook.after_execute_tool(context, tool_call, tool, params, result)

    detail = "" if result is None else str(result)
    detail = detail.replace("\n", " ").strip()
    if not detail:
        detail = "(empty)"
    elif len(detail) > 120:
        detail = detail[:120] + "..."
    return result, {"name": tool_call.name, "status": "ok", "detail": detail}


def is_ssrf_violation(text: str) -> bool:
    """Return whether a tool error describes a blocked private-network request."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _SSRF_MARKERS)


def _is_workspace_violation(text: str) -> bool:
    """Return whether text describes any workspace or network boundary rejection."""
    if not text:
        return False
    lowered = text.lower()
    if is_ssrf_violation(lowered):
        return True
    return any(marker in lowered for marker in _WORKSPACE_VIOLATION_MARKERS)


def _classify_violation(
    *,
    raw_text: str,
    soft_payload: str,
    event: dict[str, str],
    tool_call: ToolCallRequest,
    workspace_violation_counts: dict[str, int],
) -> tuple[Any, dict[str, str]] | None:
    if is_ssrf_violation(raw_text):
        logger.warning(
            "Tool {} blocked by SSRF guard; returning non-retryable tool error: {}",
            tool_call.name,
            raw_text.replace("\n", " ").strip()[:200],
        )
        event["detail"] = _event_detail("ssrf_violation: ", raw_text)
        return _ssrf_soft_payload(raw_text), event

    if _is_workspace_violation(raw_text):
        escalation = repeated_workspace_violation_error(
            tool_call.name,
            tool_call.arguments,
            workspace_violation_counts,
        )
        event["detail"] = _event_detail("workspace_violation: ", raw_text)
        if escalation is not None:
            logger.warning(
                "Tool {} hit workspace boundary repeatedly; escalating hint",
                tool_call.name,
            )
            event["detail"] = _event_detail(
                "workspace_violation_escalated: ",
                raw_text,
            )
            return escalation, event
        return soft_payload, event

    return None


def _ssrf_soft_payload(raw_text: str) -> str:
    text = raw_text.strip() or "Error: request blocked by SSRF guard"
    return f"{text}\n\n{_SSRF_BOUNDARY_NOTE}"


def _event_detail(prefix: str, text: str, limit: int = 160) -> str:
    return (prefix + text.replace("\n", " ").strip())[:limit]


def _partition_tool_batches(
    tools: ToolRegistry,
    tool_calls: list[ToolCallRequest],
    *,
    concurrent: bool,
) -> list[list[ToolCallRequest]]:
    if not concurrent:
        return [[tool_call] for tool_call in tool_calls]

    batches: list[list[ToolCallRequest]] = []
    current: list[ToolCallRequest] = []
    for tool_call in tool_calls:
        get_tool = cast(Callable[[str], Any] | None, getattr(tools, "get", None))
        tool = get_tool(tool_call.name) if callable(get_tool) else None
        can_batch = bool(tool and tool.concurrency_safe and tool.is_concurrency_safe_call(tool_call.arguments))
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


def _search_terms(tool_call: ToolCallRequest) -> set[str]:
    if not isinstance(tool_call.arguments, dict):
        return set()
    arguments = cast(dict[str, Any], tool_call.arguments)
    raw = arguments.get("pattern") or arguments.get("query") or ""
    return {term.lower() for term in re.findall(r"[\w.-]{4,}", str(raw))}


def _deferred_broad_searches(tool_calls: list[ToolCallRequest]) -> dict[str, str]:
    """Defer same-turn fallback searches until the model sees the narrower result."""
    deferred: dict[str, str] = {}
    prior: list[ToolCallRequest] = []
    for call in tool_calls:
        if call.name not in {"grep", "find_files"} or not isinstance(call.arguments, dict):
            prior.append(call)
            continue
        arguments = cast(dict[str, Any], call.arguments)
        path = Path(str(arguments.get("path", "."))).resolve(strict=False)
        terms = _search_terms(call)
        for narrow in prior:
            if narrow.name not in {"grep", "find_files"} or not isinstance(narrow.arguments, dict):
                continue
            narrow_arguments = cast(dict[str, Any], narrow.arguments)
            narrow_path = Path(str(narrow_arguments.get("path", "."))).resolve(strict=False)
            if path == narrow_path or path not in narrow_path.parents:
                continue
            narrow_terms = _search_terms(narrow)
            if terms and narrow_terms and not terms.intersection(narrow_terms):
                continue
            deferred[call.id] = (
                "[Broader fallback search deferred — review narrow result first]\n"
                f"A narrower overlapping search in '{narrow_arguments.get('path', '.')}' was requested "
                f"in the same model response. The broader search in '{arguments.get('path', '.')}' "
                "was not executed. Review the narrow result, then request the broader scope in a new "
                "model step only if it remains necessary."
            )
            break
        prior.append(call)
    return deferred
