"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from oauth_cli_kit import get_token as get_codex_token

from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    resolve_stream_idle_timeout_s,
)
from nanobot.providers.openai_responses import (
    consume_sse_with_reasoning,
    convert_messages,
    convert_tools,
)
from nanobot.utils.oauth_compat import call_with_optional_proxy

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "nanobot"
CODEX_FIRST_EVENT_TIMEOUT_ENV = "NANOBOT_CODEX_FIRST_EVENT_TIMEOUT_S"
DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S = 180.0
_DIAGNOSTIC_HMAC_KEY = secrets.token_bytes(32)


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    supports_progress_deltas = True

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.1-codex",
        proxy: str | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.proxy = proxy or None
        self._context_window, self._effective_input_window = (
            _load_codex_model_context(default_model)
        )

    def resolve_context_window_tokens(self, model: str, configured: int) -> int:
        hard_window, effective_window = _load_codex_model_context(model)
        self._context_window = hard_window
        self._effective_input_window = effective_window
        if hard_window:
            return min(configured, hard_window) if configured > 0 else hard_window
        return configured

    def input_token_budget(
        self,
        context_window_tokens: int,
        max_completion_tokens: int,
        safety_buffer: int = 1024,
    ) -> int:
        if self._context_window and context_window_tokens >= self._context_window:
            effective = self._effective_input_window or self._context_window
            return max(0, min(context_window_tokens, effective))
        return super().input_token_budget(
            context_window_tokens, max_completion_tokens, safety_buffer
        )

    async def _call_codex(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Shared request logic for both chat() and chat_stream()."""
        model = model or self.default_model
        system_prompt, input_items = convert_messages(messages)

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": _prompt_cache_key(messages[:2]),
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": True,
        }
        reasoning_options = _build_reasoning_options(reasoning_effort)
        if reasoning_options:
            body["reasoning"] = reasoning_options
        if tools:
            body["tools"] = convert_tools(tools)

        try:
            token = await asyncio.to_thread(call_with_optional_proxy, get_codex_token, proxy=self.proxy)
            headers = _build_headers(token.account_id, token.access)

            try:
                request_result = await _request_codex(
                    DEFAULT_CODEX_URL, headers, body, verify=True,
                    proxy=self.proxy,
                    on_content_delta=on_content_delta,
                    on_thinking_delta=on_thinking_delta,
                    on_tool_call_delta=on_tool_call_delta,
                )
                (
                    content,
                    tool_calls,
                    finish_reason,
                    usage,
                    reasoning_content,
                    diagnostics,
                ) = _unpack_codex_result(request_result)
            except Exception as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                    raise
                logger.warning("SSL verification failed for Codex API; retrying with verify=False")
                request_result = await _request_codex(
                    DEFAULT_CODEX_URL, headers, body, verify=False,
                    proxy=self.proxy,
                    on_content_delta=on_content_delta,
                    on_thinking_delta=on_thinking_delta,
                    on_tool_call_delta=on_tool_call_delta,
                )
                (
                    content,
                    tool_calls,
                    finish_reason,
                    usage,
                    reasoning_content,
                    diagnostics,
                ) = _unpack_codex_result(request_result)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                reasoning_content=reasoning_content,
                provider_diagnostics=diagnostics,
            )
        except Exception as e:
            response = _codex_error_response(e)
            exc_type = "CodexHTTPError" if isinstance(e, _CodexHTTPError) else type(e).__name__
            logger.warning(
                "Codex API request failed: type={} kind={} retryable={} status={} "
                "error_type={} error_code={} retry_after={} summary={}",
                exc_type,
                response.error_kind,
                response.error_should_retry,
                response.error_status_code,
                response.error_type,
                response.error_code,
                response.retry_after,
                _codex_log_summary(exc_type, response),
            )
            return response

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._call_codex(messages, tools, model, reasoning_effort, tool_choice)

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        return await self._call_codex(
            messages,
            tools,
            model,
            reasoning_effort,
            tool_choice,
            on_content_delta,
            on_thinking_delta,
            on_tool_call_delta,
        )

    def get_default_model(self) -> str:
        return self.default_model


def resolve_codex_first_event_timeout_s(
    *,
    env_value: str | None = None,
    default: float = DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S,
) -> float:
    """Return a bounded Codex wait for the first SSE line."""
    raw = os.environ.get(CODEX_FIRST_EVENT_TIMEOUT_ENV) if env_value is None else env_value
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid {}={!r}; using {}",
            CODEX_FIRST_EVENT_TIMEOUT_ENV,
            raw,
            default,
        )
        return default
    if value <= 0:
        logger.warning(
            "Ignoring non-positive {}={!r}; using {}",
            CODEX_FIRST_EVENT_TIMEOUT_ENV,
            raw,
            default,
        )
        return default
    return min(value, 3600.0)


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _load_codex_model_context(model: str) -> tuple[int | None, int | None]:
    """Read the current Codex route limits from the CLI model catalog."""
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    try:
        data = json.loads((root / "models_cache.json").read_text(encoding="utf-8"))
        slug = _strip_model_prefix(model)
        row = next(
            item
            for item in data.get("models", [])
            if isinstance(item, dict) and item.get("slug") == slug
        )
        hard = int(row.get("context_window") or row.get("max_context_window") or 0)
        percent = int(row.get("effective_context_window_percent") or 100)
        if hard <= 0:
            return None, None
        effective = max(1, hard * min(100, max(1, percent)) // 100)
        return hard, effective
    except (OSError, ValueError, TypeError, StopIteration, json.JSONDecodeError):
        return None, None


def _build_reasoning_options(reasoning_effort: str | None) -> dict[str, str] | None:
    """Opt in to visible summaries without changing provider-default effort."""
    if reasoning_effort and reasoning_effort.lower() == "none":
        return {"effort": "none"}
    options = {"summary": "auto"}
    if reasoning_effort:
        options["effort"] = reasoning_effort
    return options


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _format_codex_exception(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    return type(exc).__name__


class _CodexHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        should_retry: bool | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.error_type = error_type
        self.error_code = error_code
        self.should_retry = should_retry


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
    proxy: str | None = None,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
    on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[
    str,
    list[ToolCallRequest],
    str,
    dict[str, int],
    str | None,
    dict[str, Any],
]:
    idle_timeout_s = resolve_stream_idle_timeout_s()
    first_event_timeout_s = resolve_codex_first_event_timeout_s()
    client_kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(
            connect=idle_timeout_s,
            read=first_event_timeout_s,
            write=idle_timeout_s,
            pool=idle_timeout_s,
        ),
        "verify": verify,
    }
    if proxy:
        client_kwargs["proxy"] = proxy
        client_kwargs["trust_env"] = False
    async with httpx.AsyncClient(**client_kwargs) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raw = text.decode("utf-8", "ignore")
                retry_after = LLMProvider._extract_retry_after_from_headers(response.headers)
                error_type, error_code = LLMProvider._extract_error_type_code(raw)
                raise _CodexHTTPError(
                    _friendly_error(response.status_code, raw),
                    status_code=response.status_code,
                    retry_after=retry_after,
                    error_type=error_type,
                    error_code=error_code,
                    should_retry=_should_retry_status(response.status_code, error_type, error_code, raw),
                )
            diagnostics = _CodexSSEDiagnostics()
            result = await consume_sse_with_reasoning(
                response,
                on_content_delta=on_content_delta,
                on_tool_call_delta=on_tool_call_delta,
                on_reasoning_delta=on_thinking_delta,
                on_event=diagnostics.observe,
                first_line_timeout_s=first_event_timeout_s,
                idle_timeout_s=idle_timeout_s,
            )
            content, tool_calls, finish_reason, usage, reasoning_content = result
            return (
                content,
                tool_calls,
                finish_reason,
                usage,
                reasoning_content,
                diagnostics.finish(content),
            )


def _unpack_codex_result(
    result: tuple[Any, ...],
) -> tuple[str, list[ToolCallRequest], str, dict[str, int], str | None, dict[str, Any] | None]:
    """Accept legacy five-item test/provider adapters while adding diagnostics."""
    if len(result) == 5:
        content, tool_calls, finish_reason, usage, reasoning_content = result
        return content, tool_calls, finish_reason, usage, reasoning_content, None
    content, tool_calls, finish_reason, usage, reasoning_content, diagnostics = result
    return content, tool_calls, finish_reason, usage, reasoning_content, diagnostics


def _diagnostic_fingerprint(value: str) -> str:
    return hmac.new(
        _DIAGNOSTIC_HMAC_KEY,
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]


class _CodexSSEDiagnostics:
    """Collect metadata sufficient to diagnose replay without retaining response text."""

    def __init__(self) -> None:
        self.event_count = 0
        self.delta_count = 0
        self.delta_chars = 0
        self.repeated_event_count = 0
        self.repeated_delta_count = 0
        self.first_sequence_number: int | None = None
        self.last_sequence_number: int | None = None
        self.repeated_sequence_count = 0
        self.response_created_count = 0
        self.response_completed_count = 0
        self._seen_events: set[tuple[Any, ...]] = set()
        self._seen_deltas: set[tuple[Any, ...]] = set()
        self._seen_sequence_numbers: set[int] = set()
        self._response_fingerprints: set[str] = set()

    @staticmethod
    def _event_position(event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            event.get("sequence_number"),
            event.get("output_index"),
            event.get("content_index"),
            event.get("item_id"),
        )

    def observe(self, event: dict[str, Any]) -> None:
        self.event_count += 1
        event_type = str(event.get("type") or "")
        sequence_number = event.get("sequence_number")
        if isinstance(sequence_number, int):
            if self.first_sequence_number is None:
                self.first_sequence_number = sequence_number
            self.last_sequence_number = sequence_number
            if sequence_number in self._seen_sequence_numbers:
                self.repeated_sequence_count += 1
            else:
                self._seen_sequence_numbers.add(sequence_number)

        if event_type == "response.created":
            self.response_created_count += 1
        elif event_type == "response.completed":
            self.response_completed_count += 1
        response_obj = event.get("response")
        if isinstance(response_obj, dict) and response_obj.get("id"):
            self._response_fingerprints.add(
                _diagnostic_fingerprint(str(response_obj["id"]))
            )

        position = self._event_position(event)
        event_id = event.get("event_id") or event.get("id")
        event_key = (event_type, event_id, *position)
        if event_id is not None or any(value is not None for value in position):
            if event_key in self._seen_events:
                self.repeated_event_count += 1
            else:
                self._seen_events.add(event_key)

        if event_type != "response.output_text.delta":
            return
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        self.delta_count += 1
        self.delta_chars += len(delta)
        delta_key = (*position, len(delta), _diagnostic_fingerprint(delta))
        if delta_key in self._seen_deltas:
            self.repeated_delta_count += 1
        else:
            self._seen_deltas.add(delta_key)

    def finish(self, content: str) -> dict[str, Any]:
        exact_half_repeat = bool(content) and len(content) % 2 == 0
        if exact_half_repeat:
            midpoint = len(content) // 2
            exact_half_repeat = content[:midpoint] == content[midpoint:]
        return {
            "kind": "codex_sse",
            "event_count": self.event_count,
            "delta_count": self.delta_count,
            "delta_chars": self.delta_chars,
            "repeated_event_count": self.repeated_event_count,
            "repeated_delta_count": self.repeated_delta_count,
            "first_sequence_number": self.first_sequence_number,
            "last_sequence_number": self.last_sequence_number,
            "repeated_sequence_count": self.repeated_sequence_count,
            "response_created_count": self.response_created_count,
            "response_completed_count": self.response_completed_count,
            "unique_response_id_count": len(self._response_fingerprints),
            "content_chars": len(content),
            "content_fingerprint": _diagnostic_fingerprint(content),
            "exact_half_repeat": exact_half_repeat,
        }


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _friendly_error(status_code: int, raw: str) -> str:
    _ = raw
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: Codex API request failed"


def _codex_error_response(exc: Exception) -> LLMResponse:
    """Convert Codex transport/API failures into actionable, retryable metadata."""
    exc_type = "CodexHTTPError" if isinstance(exc, _CodexHTTPError) else type(exc).__name__
    detail = str(exc).strip()

    status_code = getattr(exc, "status_code", None)
    error_type = getattr(exc, "error_type", None)
    error_code = getattr(exc, "error_code", None)
    error_kind: str | None = None
    default_detail: str | None = None
    should_retry: bool | None = getattr(exc, "should_retry", None)

    if error_code == "context_length_exceeded":
        error_kind = "context_length"
        should_retry = False
    elif status_code is None and (
        error_type in {"server_error", "internal_server_error", "service_unavailable"}
        or error_code in {"server_error", "internal_server_error", "service_unavailable"}
    ):
        # Responses API may report transient server failures inside a completed
        # SSE stream without an HTTP status or explicit should_retry flag.
        error_kind = "server"
        should_retry = True if should_retry is None else should_retry
    elif isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
        error_kind = "timeout"
        default_detail = "timed out waiting for response"
        should_retry = True if should_retry is None else should_retry
    elif isinstance(exc, httpx.RemoteProtocolError):
        error_kind = "connection"
        default_detail = "network protocol error while reading response"
        should_retry = True if should_retry is None else should_retry
    elif isinstance(exc, (httpx.NetworkError, httpx.TransportError)):
        error_kind = "connection"
        default_detail = "network connection failed"
        should_retry = True if should_retry is None else should_retry
    elif isinstance(exc, _CodexHTTPError):
        error_kind = "http"
        default_detail = "HTTP request failed"

    if status_code is not None and should_retry is None:
        retry_content = None if int(status_code) == 429 and isinstance(exc, _CodexHTTPError) else detail
        should_retry = _should_retry_status(
            int(status_code),
            getattr(exc, "error_type", None),
            getattr(exc, "error_code", None),
            retry_content,
        )

    detail = detail or default_detail or "unexpected error"
    message = f"Error calling Codex ({exc_type}): {detail}"
    retry_after = getattr(exc, "retry_after", None) or LLMProvider._extract_retry_after(message)
    return LLMResponse(
        content=message,
        finish_reason="error",
        retry_after=retry_after,
        error_status_code=int(status_code) if status_code is not None else None,
        error_kind=error_kind,
        error_type=error_type,
        error_code=error_code,
        error_retry_after_s=retry_after,
        error_should_retry=should_retry,
    )


def _codex_log_summary(exc_type: str, response: LLMResponse) -> str:
    """Return a bounded diagnostic summary without request body or raw upstream payload."""
    if response.error_status_code is not None:
        parts = [f"HTTP {response.error_status_code}"]
        if response.error_type:
            parts.append(f"type={response.error_type}")
        if response.error_code:
            parts.append(f"code={response.error_code}")
        return " ".join(parts)

    kind = (response.error_kind or "").strip()
    if kind:
        return f"{exc_type} {kind}"

    return exc_type


def _should_retry_status(
    status_code: int,
    error_type: str | None,
    error_code: str | None,
    content: str | None,
) -> bool:
    if status_code == 429:
        return LLMProvider._is_retryable_429_response(
            LLMResponse(
                content=content or "",
                finish_reason="error",
                error_status_code=status_code,
                error_type=error_type,
                error_code=error_code,
            )
        )
    return status_code in LLMProvider._RETRYABLE_STATUS_CODES or status_code >= 500
