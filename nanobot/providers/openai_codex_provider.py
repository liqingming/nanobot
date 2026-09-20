"""OpenAI Codex Responses Provider."""

# pyright: reportMissingTypeStubs=false, reportPrivateUsage=false

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import ssl
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import httpx
from loguru import logger
from oauth_cli_kit import get_token as get_codex_token
from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER
from oauth_cli_kit.storage import FileTokenStorage

from nanobot import __version__
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
    resolve_stream_idle_timeout_s,
)
from nanobot.providers.oauth_model_catalog import (
    OAuthModelCatalog,
    OAuthModelCatalogSnapshot,
)
from nanobot.providers.openai_responses import (
    ResponsesStreamCapture,
    build_responses_compaction_state,
    build_responses_state,
    consume_sse_with_reasoning,
    convert_messages,
    convert_tools,
    is_compaction_compatibility_error,
    is_replayable_finish_reason,
    prepare_responses_input,
    resolve_compact_threshold,
    responses_state_context_tokens,
    responses_state_items,
    responses_state_matches,
)
from nanobot.providers.registry import ProviderModelSpec, find_by_name
from nanobot.utils.oauth_compat import call_with_optional_proxy

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_OPENAI_CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
# The server gates model visibility by client version; older catalogs omit Astra.
OPENAI_CODEX_CATALOG_CLIENT_VERSION = "0.153.4"
DEFAULT_ORIGINATOR = "nanobot"
CODEX_FIRST_EVENT_TIMEOUT_ENV = "NANOBOT_CODEX_FIRST_EVENT_TIMEOUT_S"
DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S = 180.0
CODEX_TURN_STATE_HEADER = "x-codex-turn-state"
_DIAGNOSTIC_HMAC_KEY = secrets.token_bytes(32)
_COMPACTION_RETAINED_CHAR_BUDGET = 256_000


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    supports_progress_deltas = True
    supports_request_context = True

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.6-sol",
        proxy: str | None = None,
        extra_body: dict[str, Any] | None = None,
        *,
        provider_name: str = "openai_codex",
    ):
        super().__init__(api_key=None, api_base=None, provider_name=provider_name)
        self.default_model = default_model
        self.proxy = proxy or None
        self._window_id = str(uuid.uuid4())
        self._turn_states: dict[tuple[str, str], str] = {}
        self._context_window, self._effective_input_window = (
            _load_codex_model_context(default_model)
        )

        self._extra_body = dict(extra_body or {})
        self._native_compaction_available = True
        self._ssl_contexts: dict[bool, ssl.SSLContext] = {}

    def _ssl_context(self, *, verify: bool) -> ssl.SSLContext:
        """Reuse synchronous TLS setup across requests on the shared event loop."""
        context = self._ssl_contexts.get(verify)
        if context is None:
            context = httpx.create_ssl_context(
                verify=verify,
                trust_env=self.proxy is None,
            )
            self._ssl_contexts[verify] = context
        return context
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
    def _remember_turn_state(self, key: tuple[str, str], value: str) -> None:
        """Keep recent Codex sticky-routing state bounded and isolated by turn."""
        self._turn_states.pop(key, None)
        self._turn_states[key] = value
        while len(self._turn_states) > 256:
            oldest = next(iter(self._turn_states))
            self._turn_states.pop(oldest, None)



    async def _call_codex(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        request_context: dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        """Shared request logic for both chat() and chat_stream()."""
        model = model or self.default_model
        model_info = _load_codex_model_info(model)
        use_responses_lite = model_info.get("use_responses_lite") is True
        sanitized_messages = self._sanitize_empty_content(messages)
        sanitized_state = (
            provider_context.conversation_state if provider_context is not None else None
        )
        if sanitized_state is not None:
            sanitized_state = sanitized_state.with_pending_messages(
                self._sanitize_empty_content(sanitized_state.pending_messages)
            )
        system_prompt, input_items, replayed = prepare_responses_input(
            sanitized_messages,
            state=sanitized_state,
            provider=self._responses_state_provider(),
            model=_strip_model_prefix(model),
        )
        session_id = provider_context.session_id if provider_context is not None else None
        session_routing_key = _prompt_cache_key(session_id) if session_id else None

        body = _build_request_body(
            sanitized_messages, tools, model, reasoning_effort, tool_choice,
            model_info=model_info, prepared_input=(system_prompt, input_items),
        )
        if session_routing_key:
            body["prompt_cache_key"] = session_routing_key
        body["include"] = ["reasoning.encrypted_content"]
        reasoning_options = _build_reasoning_options(reasoning_effort, use_responses_lite=use_responses_lite)
        if replayed and "gpt-5.6" in _strip_model_prefix(model).lower():
            reasoning_options = dict(reasoning_options or {})
            reasoning_options["context"] = "all_turns"
        if reasoning_options:
            body["reasoning"] = reasoning_options
        if tools and not use_responses_lite:
            body["tools"] = convert_tools(tools)
        if self._extra_body:
            # Apply explicit provider overrides last, matching other provider backends.
            body.update(self._extra_body)
        effective_cache_key = body.get("prompt_cache_key")

        stage = "oauth_token"
        native_compaction_applied = False
        native_compaction_state: ProviderConversationState | None = None
        try:
            token = await asyncio.to_thread(
                call_with_optional_proxy, get_codex_token, proxy=self.proxy
            )
            if not isinstance(token.account_id, str) or not token.account_id:
                raise RuntimeError("Codex OAuth token has no account identifier")
            request_metadata = _build_codex_request_metadata(
                token.account_id,
                request_context,
                window_id=self._window_id,
            )
            body["client_metadata"] = request_metadata["client_metadata"]
            if request_context is not None:
                body["prompt_cache_key"] = request_metadata["session_id"]

            turn_state_key = _codex_turn_state_key(request_metadata)
            turn_state = self._turn_states.get(turn_state_key) if turn_state_key else None
            raw_headers = _build_headers(
                token.account_id,
                token.access,
                use_responses_lite=use_responses_lite,
                request_metadata=request_metadata if request_context is not None else None,
                turn_state=turn_state,
                session_routing_key=(effective_cache_key if isinstance(effective_cache_key, str) else None),
            )
            headers = _CodexRequestHeaders(
                raw_headers,
                on_turn_state=(
                    (lambda value: self._remember_turn_state(turn_state_key, value))
                    if turn_state_key else None
                ),

            )

            async def _send(
                request_body: dict[str, Any],
                *,
                emit_deltas: bool,
            ) -> LLMResponse:
                wire_body = _without_response_item_ids(request_body)
                try:
                    return await _request_codex(
                        DEFAULT_CODEX_URL,
                        headers,
                        wire_body,
                        verify=self._ssl_context(verify=True),
                        proxy=self.proxy,
                        on_content_delta=on_content_delta if emit_deltas else None,
                        on_thinking_delta=on_thinking_delta if emit_deltas else None,
                        on_tool_call_delta=on_tool_call_delta if emit_deltas else None,
                    )
                except Exception as exc:
                    if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                        raise
                    logger.warning(
                        "SSL verification failed for Codex API; retrying with verify=False"
                    )
                    return await _request_codex(
                        DEFAULT_CODEX_URL,
                        headers,
                        wire_body,
                        verify=self._ssl_context(verify=False),
                        proxy=self.proxy,
                        on_content_delta=on_content_delta if emit_deltas else None,
                        on_thinking_delta=on_thinking_delta if emit_deltas else None,
                        on_tool_call_delta=on_tool_call_delta if emit_deltas else None,
                    )

            compact_threshold = resolve_compact_threshold(
                (provider_context.context_window_tokens if provider_context is not None else None),
                max_tokens,
            )
            if (
                self.supports_native_compaction(model)
                and replayed
                and sanitized_state is not None
                and compact_threshold is not None
                and responses_state_context_tokens(sanitized_state) >= compact_threshold
            ):
                stage = "codex_compaction"
                history_items = responses_state_items(sanitized_state) or []
                delta_items = input_items[len(history_items):]
                compact_body = {
                    **body,
                    "input": [*history_items, {"type": "compaction_trigger"}],
                }
                try:
                    compact_result = await _send(compact_body, emit_deltas=False)
                    compact_items = (
                        responses_state_items(compact_result.provider_state)
                        if compact_result.provider_state is not None
                        else None
                    )
                    if not compact_items or compact_items[-1].get("type") not in {
                        "compaction",
                        "compaction_summary",
                        "context_compaction",
                    }:
                        raise RuntimeError("Codex compaction returned no compaction item")
                    body["input"] = [
                        *_retained_compaction_messages(history_items),
                        *compact_items,
                        *delta_items,
                    ]
                    native_compaction_state = build_responses_compaction_state(
                        provider=self._responses_state_provider(),
                        model=_strip_model_prefix(model),
                        output_items=compact_items,
                    )
                    native_compaction_applied = True
                except Exception as compact_error:
                    if is_compaction_compatibility_error(compact_error):
                        self._native_compaction_available = False
                    logger.warning(
                        "Codex native compaction unavailable; continuing without it "
                        "(type={} status={} disabled={})",
                        type(compact_error).__name__,
                        getattr(compact_error, "status_code", None),
                        not self._native_compaction_available,
                    )

            stage = "codex_request"
            result = await _send(body, emit_deltas=True)
            result.provider_compaction_applied = (
                result.provider_compaction_applied or native_compaction_applied
            )
            if native_compaction_state is not None:
                result.provider_compaction_state = native_compaction_state
                result.provider_compaction_scope = "prior_context"
            return result
        except Exception as e:
            response = _codex_error_response(e)
            exc_type = "CodexHTTPError" if isinstance(e, _CodexHTTPError) else type(e).__name__
            logger.warning(
                "Codex API request failed: stage={} type={} kind={} retryable={} status={} "
                "error_type={} error_code={} retry_after={} summary={}",
                stage,
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
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_context: dict[str, Any] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        return await self._call_codex(
            messages,
            tools,
            model,
            max_tokens,
            reasoning_effort,
            tool_choice,
            request_context=request_context,
            provider_context=provider_context,
        )

    async def chat_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.chat(
            **kwargs,
            provider_context=provider_context,
        )

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
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        request_context: dict[str, Any] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        return await self._call_codex(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            request_context=request_context,
            on_content_delta=on_content_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
            provider_context=provider_context,
        )

    async def chat_stream_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.chat_stream(
            **kwargs,
            provider_context=provider_context,
        )

    def get_default_model(self) -> str:
        return self.default_model

    @staticmethod
    def _responses_state_provider() -> str:
        return f"openai_codex:{DEFAULT_CODEX_URL.rstrip('/')}"

    def can_resume_conversation_state(
        self,
        state: ProviderConversationState,
        model: str | None = None,
    ) -> bool:
        return responses_state_matches(
            state,
            provider=self._responses_state_provider(),
            model=_strip_model_prefix(model or self.default_model),
        )

    def supports_native_compaction(self, model: str | None = None) -> bool:
        """Use the Codex backend's inline compaction trigger when needed."""
        _ = model
        return self._native_compaction_available


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


def _without_response_item_ids(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Match Codex's default ``store=false`` request-item contract."""
    if request_body.get("store") is True:
        return request_body
    raw_input = request_body.get("input")
    if not isinstance(raw_input, list):
        return request_body

    input_items: list[object] = cast(list[object], raw_input)
    sanitized_input: list[object] = []
    for raw_item in input_items:
        if not isinstance(raw_item, dict):
            sanitized_input.append(raw_item)
            continue
        item = cast(dict[str, Any], raw_item)
        sanitized_input.append({key: value for key, value in item.items() if key != "id"})

    body = dict(request_body)
    body["input"] = sanitized_input
    return body


def _retained_compaction_messages(
    input_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mirror Codex's bounded retention of user/developer/system messages."""
    retained_reversed: list[dict[str, Any]] = []
    remaining = _COMPACTION_RETAINED_CHAR_BUDGET
    for item in reversed(input_items):
        if item.get("type") not in {None, "message"} or item.get("role") not in {
            "user",
            "developer",
            "system",
        }:
            continue
        size = len(json.dumps(item, ensure_ascii=False))
        if size > remaining and retained_reversed:
            continue
        retained_reversed.append(item)
        remaining = max(0, remaining - size)
        if remaining == 0:
            break
    retained_reversed.reverse()
    return retained_reversed


def _load_codex_model_context(model: str) -> tuple[int | None, int | None]:
    """Read the current Codex route limits from the CLI model catalog."""
    row = _load_codex_model_info(model)
    try:
        hard = int(row.get("context_window") or row.get("max_context_window") or 0)
        percent = int(row.get("effective_context_window_percent") or 100)
        if hard <= 0:
            return None, None
        effective = max(1, hard * min(100, max(1, percent)) // 100)
        return hard, effective
    except (ValueError, TypeError):
        return None, None


def _load_codex_model_info(model: str) -> dict[str, Any]:
    """Read one model's current capabilities from the Codex CLI catalog."""
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    try:
        data = json.loads((root / "models_cache.json").read_text(encoding="utf-8"))
        slug = _strip_model_prefix(model)
        return next(
            cast(dict[str, Any], item)
            for item in data.get("models", [])
            if isinstance(item, dict) and cast(dict[str, Any], item).get("slug") == slug
        )
    except (OSError, ValueError, TypeError, StopIteration, json.JSONDecodeError):
        return {}


def _build_request_body(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    reasoning_effort: str | None,
    tool_choice: str | dict[str, Any] | None,
    *,
    model_info: dict[str, Any] | None = None,
    prepared_input: tuple[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build the legacy or Responses Lite request selected by model capabilities."""
    model_info = model_info if model_info is not None else _load_codex_model_info(model)
    use_responses_lite = model_info.get("use_responses_lite") is True
    system_prompt, conversation_items = (
        prepared_input if prepared_input is not None else convert_messages(messages)
    )
    converted_tools = convert_tools(tools or [])

    if use_responses_lite:
        input_items: list[dict[str, Any]] = [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": converted_tools,
            }
        ]
        if system_prompt:
            input_items.append({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}],
            })
        input_items.extend(conversation_items)
    else:
        input_items = conversation_items

    verbosity = "medium"
    if use_responses_lite:
        default_verbosity = model_info.get("default_verbosity")
        verbosity = default_verbosity if isinstance(default_verbosity, str) else "low"

    body: dict[str, Any] = {
        "model": _strip_model_prefix(model),
        "store": False,
        "stream": True,
        "instructions": "" if use_responses_lite else system_prompt,
        "input": input_items,
        "text": {"verbosity": verbosity},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": tool_choice or "auto",
        "parallel_tool_calls": not use_responses_lite,
    }
    reasoning_options = _build_reasoning_options(
        reasoning_effort,
        use_responses_lite=use_responses_lite,
    )
    if reasoning_options:
        body["reasoning"] = reasoning_options
    if converted_tools and not use_responses_lite:
        body["tools"] = converted_tools
    return body


def _build_reasoning_options(
    reasoning_effort: str | None,
    *,
    use_responses_lite: bool = False,
) -> dict[str, str] | None:
    """Opt in to visible summaries without changing provider-default effort."""
    if use_responses_lite:
        options = {"context": "all_turns"}
        if reasoning_effort:
            options["effort"] = reasoning_effort
        return options
    if reasoning_effort and reasoning_effort.lower() == "none":
        return {"effort": "none"}
    options = {"summary": "auto"}
    if reasoning_effort:
        options["effort"] = reasoning_effort
    return options


def _stable_codex_id(kind: str, *parts: str) -> str:
    """Return a non-sensitive stable UUID for Codex routing metadata."""
    seed = "\x1f".join(("nanobot", "codex", kind, *parts))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _build_codex_request_metadata(
    account_id: str,
    request_context: dict[str, Any] | None,
    *,
    window_id: str,
) -> dict[str, Any]:
    """Build the canonical Codex session/thread/client metadata snapshot."""
    context = request_context or {}
    session_key = str(context.get("session_key") or "default")
    raw_turn_id = context.get("turn_id")
    raw_turn_id = str(raw_turn_id) if raw_turn_id else None

    installation_id = _stable_codex_id("installation", account_id)
    session_id = _stable_codex_id("session", account_id, session_key)
    thread_id = _stable_codex_id("thread", account_id, session_key)
    turn_id = (
        _stable_codex_id("turn", account_id, session_key, raw_turn_id)
        if raw_turn_id else None
    )

    client_metadata: dict[str, str] = {
        "x-codex-installation-id": installation_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "x-codex-window-id": window_id,
    }
    turn_metadata_json: str | None = None
    if turn_id:
        turn_metadata_json = json.dumps(
            {
                "installation_id": installation_id,
                "session_id": session_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "window_id": window_id,
                "request_kind": "turn",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        client_metadata["turn_id"] = turn_id
        client_metadata["x-codex-turn-metadata"] = turn_metadata_json

    return {
        "installation_id": installation_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "window_id": window_id,
        "turn_metadata_json": turn_metadata_json,
        "client_metadata": client_metadata,
    }


def _codex_turn_state_key(metadata: dict[str, Any]) -> tuple[str, str] | None:
    session_id = metadata.get("session_id")
    turn_id = metadata.get("turn_id")
    if isinstance(session_id, str) and isinstance(turn_id, str):
        return session_id, turn_id
    return None


class _CodexRequestHeaders(dict[str, str]):
    """Request headers that persist a response turn-state before SSE parsing."""

    def __init__(
        self,
        values: dict[str, str],
        *,
        on_turn_state: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(values)
        self._on_turn_state = on_turn_state

    def capture_turn_state(self, value: str | None) -> None:
        if value and self._on_turn_state:
            self._on_turn_state(value)


def _build_headers(
    account_id: str,
    token: str,
    *,
    use_responses_lite: bool = False,
    request_metadata: dict[str, Any] | None = None,
    turn_state: str | None = None,
    session_routing_key: str | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if session_routing_key:
        headers["session-id"] = session_routing_key
    if use_responses_lite:
        headers["x-openai-internal-codex-responses-lite"] = "true"
    if request_metadata:
        headers.update({
            "session-id": str(request_metadata["session_id"]),
            "thread-id": str(request_metadata["thread_id"]),
            "x-client-request-id": str(request_metadata["thread_id"]),
            "x-codex-installation-id": str(request_metadata["installation_id"]),
            "x-codex-window-id": str(request_metadata["window_id"]),
        })
        turn_metadata_json = request_metadata.get("turn_metadata_json")
        if isinstance(turn_metadata_json, str):
            headers["x-codex-turn-metadata"] = turn_metadata_json
    if turn_state:
        headers[CODEX_TURN_STATE_HEADER] = turn_state
    return headers


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
        compaction_unsupported: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.error_type = error_type
        self.error_code = error_code
        self.should_retry = should_retry
        self.compaction_unsupported = compaction_unsupported


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: ssl.SSLContext | bool,
    proxy: str | None = None,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
    on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> LLMResponse:
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
            if isinstance(headers, _CodexRequestHeaders):
                headers.capture_turn_state(response.headers.get(CODEX_TURN_STATE_HEADER))
            if response.status_code != 200:
                text = await response.aread()
                raw = text.decode("utf-8", "ignore")
                retry_after = LLMProvider._extract_retry_after_from_headers(response.headers)
                error_type, error_code = LLMProvider._extract_error_type_code(raw)
                compaction_unsupported = response.status_code in {400, 404, 422} and any(
                    marker in raw.lower()
                    for marker in (
                        "context_management",
                        "compact_threshold",
                        "compaction_trigger",
                    )
                )
                raise _CodexHTTPError(
                    _friendly_error(response.status_code, raw),
                    status_code=response.status_code,
                    retry_after=retry_after,
                    error_type=error_type,
                    error_code=error_code,
                    should_retry=_should_retry_status(
                        response.status_code, error_type, error_code, raw
                    ),
                    compaction_unsupported=compaction_unsupported,
                )
            diagnostics = _CodexSSEDiagnostics()
            capture = ResponsesStreamCapture()
            (
                content,
                tool_calls,
                finish_reason,
                usage,
                reasoning_content,
            ) = await consume_sse_with_reasoning(
                response,
                on_content_delta=on_content_delta,
                on_tool_call_delta=on_tool_call_delta,
                on_reasoning_delta=on_thinking_delta,
                capture=capture,
                on_event=diagnostics.observe,
                first_line_timeout_s=first_event_timeout_s,
                idle_timeout_s=idle_timeout_s,
            )
            result = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                reasoning_content=reasoning_content,
                response_items=capture.output_items,
                provider_diagnostics=diagnostics.finish(content),
            )
            if capture.completed and is_replayable_finish_reason(finish_reason):
                result.provider_state = build_responses_state(
                    provider=f"openai_codex:{url.rstrip('/')}",
                    model=str(body.get("model") or ""),
                    input_items=cast(list[dict[str, Any]], body.get("input") or []),
                    output_items=capture.output_items,
                    usage=usage,
                )
            return result
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
        raw_response = event.get("response")
        response_obj = cast(dict[str, Any], raw_response) if isinstance(raw_response, dict) else {}
        if response_obj.get("id"):
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


def _prompt_cache_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


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
    elif isinstance(exc, (ConnectionError, httpx.NetworkError, httpx.TransportError)):
        error_kind = "connection"
        default_detail = "network connection failed"
        should_retry = True if should_retry is None else should_retry
    elif isinstance(exc, _CodexHTTPError):
        error_kind = "http"
        default_detail = "HTTP request failed"

    if status_code is not None and should_retry is None:
        retry_content = (
            None if int(status_code) == 429 and isinstance(exc, _CodexHTTPError) else detail
        )
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


def get_openai_codex_model_catalog(
    proxy: str | None = None,
) -> OAuthModelCatalogSnapshot:
    storage = FileTokenStorage(token_filename=OPENAI_CODEX_PROVIDER.token_filename)
    token = storage.load()
    account_id = getattr(token, "account_id", None)
    account_key = _catalog_account_key(account_id)
    cache_key = f"{storage.get_token_path()}\0{account_key}\0{proxy or ''}"
    return _OPENAI_CODEX_MODEL_CATALOG.get(cache_key=cache_key, proxy=proxy)


def invalidate_openai_codex_model_catalog() -> None:
    _OPENAI_CODEX_MODEL_CATALOG.invalidate()


def _fetch_openai_codex_models(proxy: str | None) -> tuple[ProviderModelSpec, ...]:
    token = call_with_optional_proxy(get_codex_token, proxy=proxy)
    account_id = getattr(token, "account_id", None)
    if not isinstance(account_id, str) or not account_id:
        raise RuntimeError("OpenAI Codex OAuth token has no account ID")
    client_kwargs: dict[str, Any] = {"timeout": 10.0, "follow_redirects": False}
    if proxy:
        client_kwargs.update(proxy=proxy, trust_env=False)
    with httpx.Client(**client_kwargs) as client:
        response = client.get(
            DEFAULT_OPENAI_CODEX_MODELS_URL,
            params={"client_version": OPENAI_CODEX_CATALOG_CLIENT_VERSION},
            headers={
                "Authorization": f"Bearer {token.access}",
                "chatgpt-account-id": account_id,
                "originator": DEFAULT_ORIGINATOR,
                "User-Agent": f"nanobot/{__version__} (python)",
                "accept": "application/json",
            },
        )
    response.raise_for_status()
    return _parse_openai_codex_models(response.json())


def _parse_openai_codex_models(payload: Any) -> tuple[ProviderModelSpec, ...]:
    rows = cast(dict[str, Any], payload).get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()

    fallback_models = _oauth_fallback_models("openai_codex")
    fallback_by_id = {model.id.split("/", 1)[-1]: model for model in fallback_models}
    parsed: list[tuple[int, ProviderModelSpec]] = []
    seen: set[str] = set()
    for value in cast(list[object], rows):
        if not isinstance(value, dict):
            continue
        row = cast(dict[str, Any], value)
        wire_id = _catalog_first_text(row, "slug", "id")
        if not wire_id or wire_id in seen or row.get("visibility") in {"hide", "none"}:
            continue
        seen.add(wire_id)
        fallback = fallback_by_id.get(wire_id)
        priority = row.get("priority")
        parsed.append(
            (
                priority if isinstance(priority, int) and not isinstance(priority, bool) else 2**31,
                ProviderModelSpec(
                    id=f"openai-codex/{wire_id}",
                    label=(
                        _catalog_first_text(row, "display_name", "name")
                        or (fallback.label if fallback is not None else wire_id)
                    ),
                    description=(
                        _catalog_first_text(row, "description")
                        or (fallback.description if fallback is not None else "")
                    ),
                    owned_by="OpenAI Codex",
                    context_window=(
                        _catalog_positive_int(row, "context_window")
                        or (fallback.context_window if fallback is not None else None)
                    ),
                    reasoning_efforts=(
                        _catalog_reasoning_efforts(row.get("supported_reasoning_levels"))
                        or (fallback.reasoning_efforts if fallback is not None else ())
                    ),
                ),
            )
        )
    parsed.sort(key=lambda item: item[0])
    return tuple(model for _, model in parsed)


def _oauth_fallback_models(provider_name: str) -> tuple[ProviderModelSpec, ...]:
    spec = find_by_name(provider_name)
    assert spec is not None
    return spec.builtin_models


def _catalog_account_key(account_id: object) -> str:
    value = account_id if isinstance(account_id, str) else ""
    return hashlib.sha256(value.encode()).hexdigest()[:16] if value else "anonymous"


def _catalog_first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _catalog_positive_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


def _catalog_reasoning_efforts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    efforts: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, str):
            effort = item.strip()
        elif isinstance(item, dict):
            effort = _catalog_first_text(cast(dict[str, Any], item), "effort", "value", "id")
        else:
            effort = ""
        if effort and effort not in efforts:
            efforts.append(effort)
    return tuple(efforts)


_OPENAI_CODEX_MODEL_CATALOG = OAuthModelCatalog(
    fallback_models=_oauth_fallback_models("openai_codex"),
    fetch=_fetch_openai_codex_models,
)
