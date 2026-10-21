"""Provider-level OpenAI Codex relay transport.

The client keeps AgentLoop, sessions, workspaces, and tools local.  Only the
normalized provider request crosses the network; OAuth credentials remain on
the server running ``nanobot serve``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

import httpx

from nanobot.config.schema import CodexRelayConfig
from nanobot.providers.base import LLMResponse, LLMUsage, ToolCallRequest
from nanobot.providers.openai_codex_provider import OpenAICodexProvider

RELAY_PATH = "/v1/codex/relay"
RELAY_PROTOCOL_VERSION = 1


def response_to_payload(response: LLMResponse) -> dict[str, Any]:
    """Serialize the normalized provider response without local-only state."""
    return {
        "content": response.content,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "extra_content": call.extra_content,
                "provider_specific_fields": call.provider_specific_fields,
                "function_provider_specific_fields": call.function_provider_specific_fields,
            }
            for call in response.tool_calls
        ],
        "finish_reason": response.finish_reason,
        "usage": response.usage.to_dict() if response.usage else None,
        "generation_ms": response.generation_ms,
        "ttft_ms": response.ttft_ms,
        "retry_after": response.retry_after,
        "reasoning_content": response.reasoning_content,
        "thinking_blocks": response.thinking_blocks,
        "response_items": response.response_items,
        "error_status_code": response.error_status_code,
        "error_kind": response.error_kind,
        "error_type": response.error_type,
        "error_code": response.error_code,
        "error_retry_after_s": response.error_retry_after_s,
        "error_should_retry": response.error_should_retry,
        "provider_diagnostics": response.provider_diagnostics,
    }


def response_from_payload(value: object) -> LLMResponse:
    """Validate and deserialize a relay response."""
    if not isinstance(value, dict):
        raise ValueError("relay result must be an object")
    data = cast(dict[str, Any], value)
    raw_calls = data.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise ValueError("relay result tool_calls must be an array")
    calls: list[ToolCallRequest] = []
    for raw in cast(list[object], raw_calls):
        if not isinstance(raw, dict):
            raise ValueError("relay result contains an invalid tool call")
        item = cast(dict[str, Any], raw)
        call_id, name = item.get("id"), item.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ValueError("relay tool call id and name must be strings")
        calls.append(
            ToolCallRequest(
                id=call_id,
                name=name,
                arguments=item.get("arguments"),
                extra_content=_optional_dict(item.get("extra_content")),
                provider_specific_fields=_optional_dict(item.get("provider_specific_fields")),
                function_provider_specific_fields=_optional_dict(
                    item.get("function_provider_specific_fields")
                ),
            )
        )
    content = data.get("content")
    finish_reason = data.get("finish_reason", "stop")
    if content is not None and not isinstance(content, str):
        raise ValueError("relay result content must be text or null")
    if not isinstance(finish_reason, str):
        raise ValueError("relay result finish_reason must be text")
    return LLMResponse(
        content=content,
        tool_calls=calls,
        finish_reason=finish_reason,
        usage=LLMUsage.from_dict(data.get("usage")),
        generation_ms=_optional_int(data.get("generation_ms")),
        ttft_ms=_optional_int(data.get("ttft_ms")),
        retry_after=_optional_number(data.get("retry_after")),
        reasoning_content=_optional_str(data.get("reasoning_content")),
        thinking_blocks=_optional_dict_list(data.get("thinking_blocks")),
        response_items=_optional_dict_list(data.get("response_items")),
        error_status_code=_optional_int(data.get("error_status_code")),
        error_kind=_optional_str(data.get("error_kind")),
        error_type=_optional_str(data.get("error_type")),
        error_code=_optional_str(data.get("error_code")),
        error_retry_after_s=_optional_number(data.get("error_retry_after_s")),
        error_should_retry=_optional_bool(data.get("error_should_retry")),
        provider_diagnostics=_optional_dict(data.get("provider_diagnostics")),
    )


def _optional_dict(value: object) -> dict[str, Any] | None:
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _optional_dict_list(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None
    return cast(list[dict[str, Any]], value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


class CodexRelayProvider(OpenAICodexProvider):
    """OpenAI Codex provider whose transport is a remote nanobot relay."""

    def __init__(
        self,
        *,
        default_model: str,
        relay_config: CodexRelayConfig,
        provider_name: str = "openai_codex",
    ) -> None:
        super().__init__(
            default_model=default_model,
            enable_native_context=False,
            provider_name=provider_name,
        )
        self._relay_endpoint = relay_config.endpoint.rstrip("/") + RELAY_PATH
        self._relay_api_key = relay_config.api_key.strip()
        self._relay_timeout = relay_config.request_timeout

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
        provider_context: Any = None,
    ) -> LLMResponse:
        del request_context, provider_context
        payload = {
            "version": RELAY_PROTOCOL_VERSION,
            "model": model or self.default_model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "tool_choice": tool_choice,
        }
        timeout = httpx.Timeout(
            self._relay_timeout,
            connect=min(30.0, self._relay_timeout),
            read=self._relay_timeout,
            write=self._relay_timeout,
            pool=self._relay_timeout,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    self._relay_endpoint,
                    headers={
                        "Authorization": f"Bearer {self._relay_api_key}",
                        "Accept": "application/x-ndjson",
                    },
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        raw = (await response.aread()).decode("utf-8", "replace")
                        return _http_error_response(response.status_code, raw)
                    result: LLMResponse | None = None
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError("relay event must be an object")
                        event_type = event.get("type")
                        if event_type == "content_delta" and on_content_delta:
                            delta = event.get("delta")
                            if isinstance(delta, str):
                                await on_content_delta(delta)
                        elif event_type == "thinking_delta" and on_thinking_delta:
                            delta = event.get("delta")
                            if isinstance(delta, str):
                                await on_thinking_delta(delta)
                        elif event_type == "tool_call_delta" and on_tool_call_delta:
                            delta = event.get("delta")
                            if isinstance(delta, dict):
                                await on_tool_call_delta(cast(dict[str, Any], delta))
                        elif event_type == "result":
                            result = response_from_payload(event.get("response"))
                        elif event_type == "error":
                            message = event.get("message")
                            raise RuntimeError(
                                message if isinstance(message, str) else "Codex relay failed"
                            )
                    if result is None:
                        raise RuntimeError("Codex relay stream ended without a result")
                    return result
        except httpx.TimeoutException as exc:
            return _transport_error("timeout", f"Codex relay timed out: {exc}")
        except httpx.HTTPError as exc:
            return _transport_error("connection", f"Codex relay connection failed: {exc}")
        except Exception as exc:
            return _transport_error("protocol", f"Codex relay protocol failed: {exc}", retry=False)


def _http_error_response(status: int, raw: str) -> LLMResponse:
    message = raw[:1000]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
    except json.JSONDecodeError:
        pass
    retry = status in {408, 409, 425, 429} or status >= 500
    return LLMResponse(
        content=f"Codex relay HTTP {status}: {message}",
        finish_reason="error",
        error_status_code=status,
        error_kind="http",
        error_should_retry=retry,
    )


def _transport_error(kind: str, message: str, *, retry: bool = True) -> LLMResponse:
    return LLMResponse(
        content=message,
        finish_reason="error",
        error_kind=kind,
        error_should_retry=retry,
    )
