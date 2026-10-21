"""Authenticated provider-only relay for OpenAI Codex OAuth calls."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from aiohttp import web
from loguru import logger

from nanobot.config.schema import CodexRelayConfig
from nanobot.fork.providers.codex_relay import (
    RELAY_PROTOCOL_VERSION,
    response_to_payload,
)
from nanobot.providers.openai_codex_provider import OpenAICodexProvider


def _error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "codex_relay_error", "code": status}},
        status=status,
    )


def _required_string(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


class CodexRelayService:
    """Execute bounded, stateless Codex provider requests on the OAuth host."""

    def __init__(self, config: CodexRelayConfig) -> None:
        self._allowed_models = frozenset(config.allowed_models)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._provider = OpenAICodexProvider(enable_native_context=False)
        self.request_timeout = config.request_timeout

    def _parse_request(self, value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        data = cast(dict[str, Any], value)
        if data.get("version") != RELAY_PROTOCOL_VERSION:
            raise ValueError(f"unsupported relay protocol version: {data.get('version')!r}")
        model = _required_string(data, "model")
        if self._allowed_models and model not in self._allowed_models:
            raise PermissionError(f"model is not allowed by this relay: {model}")
        messages = data.get("messages")
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise ValueError("messages must be an array of objects")
        tools = data.get("tools")
        if tools is not None and (
            not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools)
        ):
            raise ValueError("tools must be an array of objects or null")
        max_tokens = data.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        reasoning_effort = _optional_string(data, "reasoning_effort")
        tool_choice = data.get("tool_choice")
        if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
            raise ValueError("tool_choice must be a string, object, or null")
        return {
            "messages": cast(list[dict[str, Any]], messages),
            "tools": cast(list[dict[str, Any]] | None, tools),
            "model": model,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "tool_choice": cast(str | dict[str, Any] | None, tool_choice),
        }

    async def handle(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            call = self._parse_request(body)
        except PermissionError as exc:
            return _error(403, str(exc))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return _error(400, str(exc))

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/x-ndjson; charset=utf-8",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
        await response.prepare(request)

        async def emit(event_type: str, field: str, value: object) -> None:
            event = {"type": event_type, field: value}
            await response.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))

        async def content_delta(delta: str) -> None:
            await emit("content_delta", "delta", delta)

        async def thinking_delta(delta: str) -> None:
            await emit("thinking_delta", "delta", delta)

        async def tool_call_delta(delta: dict[str, Any]) -> None:
            await emit("tool_call_delta", "delta", delta)

        try:
            async with self._semaphore:
                async with asyncio.timeout(self.request_timeout):
                    result = await self._provider._call_codex(
                        **call,
                        on_content_delta=content_delta,
                        on_thinking_delta=thinking_delta,
                        on_tool_call_delta=tool_call_delta,
                    )
            await emit("result", "response", response_to_payload(result))
        except asyncio.TimeoutError:
            await emit("error", "message", "Codex relay request timed out")
        except (ConnectionError, RuntimeError, ValueError) as exc:
            logger.warning("Codex relay request failed: {}", exc)
            await emit("error", "message", str(exc))
        except Exception:
            logger.exception("Unexpected Codex relay request failure")
            await emit("error", "message", "Internal relay error")
        return response


def create_codex_relay_handler(
    config: CodexRelayConfig,
) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """Create one stateful handler with a shared concurrency limit/provider."""
    return CodexRelayService(config).handle
