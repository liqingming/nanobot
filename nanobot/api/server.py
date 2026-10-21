"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import time
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from aiohttp import web
from loguru import logger

from nanobot.agent.hook import AgentHook, AgentRunHookContext
from nanobot.config.paths import get_media_dir
from nanobot.providers.base import LLMUsage
from nanobot.security.constant_time import constant_time_text_equal
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScopeError,
    validate_workspace_scope_payload,
)
from nanobot.utils.helpers import safe_filename
from nanobot.utils.media_decode import (
    MAX_FILE_SIZE,
)
from nanobot.utils.media_decode import (
    FileSizeExceeded as _FileSizeExceeded,
)
from nanobot.utils.media_decode import (
    save_base64_data_url as _save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import CodexRelayConfig

__all__ = (
    "MAX_FILE_SIZE",
    "_FileSizeExceeded",
    "_save_base64_data_url",
    "create_app",
    "handle_chat_completions",
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
_AGENT_LOOP_KEY = web.AppKey[Any]("agent_loop")
_MODEL_NAME_KEY = web.AppKey[str]("model_name")
_REQUEST_TIMEOUT_KEY = web.AppKey[float]("request_timeout")
_SESSION_LOCKS_KEY = web.AppKey[dict[str, asyncio.Lock]]("session_locks")
_PREPARE_AGENT_KEY = web.AppKey[Callable[[], Awaitable[None]] | None]("prepare_agent")
_MISSING = object()


class _UsageCaptureHook(AgentHook):
    """Capture the aggregate usage owned by one API run."""

    def __init__(self) -> None:
        super().__init__()
        self.usage: LLMUsage | None = None

    async def after_run(self, context: AgentRunHookContext) -> None:
        self.usage = context.usage


def _app_value(
    app: Any,
    key: web.AppKey[Any],
    legacy_key: str,
    default: Any = _MISSING,
) -> Any:
    """Read typed aiohttp state while accepting lightweight dict test doubles."""
    try:
        return app[key]
    except KeyError:
        if default is _MISSING:
            return app[legacy_key]
        return app.get(legacy_key, default)


async def _prepare_agent(app: Any) -> None:
    prepare: Callable[[], Awaitable[None]] | None = _app_value(
        app,
        _PREPARE_AGENT_KEY,
        "prepare_agent",
        None,
    )
    if prepare is not None:
        await prepare()


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(
    content: str,
    model: str,
    usage: LLMUsage | None = None,
) -> dict[str, Any]:
    prompt = usage.input_tokens if usage else 0
    completion = usage.output_tokens if usage else 0
    total = usage.total_tokens if usage else 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


def _as_str(value: object) -> str:
    """Return *value* when it is text, otherwise an empty string."""
    return value if isinstance(value, str) else ""


def _require_json_object(value: object, field: str) -> dict[str, Any]:
    """Validate an object-valued field from an untrusted JSON request."""
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _require_json_string(value: object, field: str) -> str:
    """Validate a string-valued field from an untrusted JSON request."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """Format a single OpenAI-compatible SSE chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(payload)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"

# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _parse_json_content(body: dict[str, Any]) -> tuple[str, list[str]]:
    """Parse JSON request body. Returns (text, media_paths)."""
    messages_value = cast(object, body.get("messages"))
    if not isinstance(messages_value, list):
        raise ValueError("Only a single user message is supported")
    messages = cast(list[object], messages_value)
    if len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message_value: object = messages[0]
    if not isinstance(message_value, dict):
        raise ValueError("Only a single user message is supported")
    message = cast(dict[str, Any], message_value)
    if message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part_value in cast(list[object], user_content):
            if not isinstance(part_value, dict):
                continue
            part = cast(dict[str, Any], part_value)
            if part.get("type") == "text":
                text_parts.append(
                    _require_json_string(
                        cast(object, part.get("text", "")),
                        "messages[0].content[].text",
                    )
                )
            elif part.get("type") == "image_url":
                image_url = _require_json_object(
                    cast(object, part.get("image_url", {})),
                    "messages[0].content[].image_url",
                )
                url = _require_json_string(
                    cast(object, image_url.get("url", "")),
                    "messages[0].content[].image_url.url",
                )
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(request: web.Request) -> tuple[str, list[str], str | None, str | None, str | None, float | None, Any]:
    """Parse multipart/form-data and return message, media, routing fields, and tool policy."""
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    workspace = None
    timeout = None
    tool_policy: Any = None
    media_paths: list[str] = []

    while True:
        part: Any = await reader.next()
        if part is None:
            break
        if part.name == "message":
            text = (await part.read()).decode("utf-8")
        elif part.name == "session_id":
            session_id = (await part.read()).decode("utf-8").strip()
        elif part.name == "model":
            model = (await part.read()).decode("utf-8").strip()
        elif part.name == "workspace":
            workspace = (await part.read()).decode("utf-8").strip()
        elif part.name == "timeout":
            raw_timeout = (await part.read()).decode("utf-8").strip()
            timeout = float(raw_timeout) if raw_timeout else None
        elif part.name == "tool_policy":
            raw_policy = (await part.read()).decode("utf-8").strip()
            tool_policy = _json.loads(raw_policy) if raw_policy else None
        elif part.name == "files":
            raw = await part.read()
            if len(raw) > MAX_FILE_SIZE:
                raise _FileSizeExceeded(
                    f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                )
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            dest.write_bytes(raw)
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return text, media_paths, session_id, model, workspace, timeout, tool_policy


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response | web.StreamResponse:
    """POST /v1/chat/completions — supports JSON and multipart/form-data."""
    content_type = _as_str(cast(object, request.content_type or ""))

    agent_loop = _app_value(request.app, _AGENT_LOOP_KEY, "agent_loop")
    timeout_s: float = _app_value(
        request.app,
        _REQUEST_TIMEOUT_KEY,
        "request_timeout",
        120.0,
    )
    model_name: str = _app_value(request.app, _MODEL_NAME_KEY, "model_name", "nanobot")

    stream = False
    try:
        requested_workspace = None
        requested_timeout = None
        requested_max_iterations = None
        requested_tool_policy: Any = None
        if content_type.startswith("multipart/"):
            text, media_paths, session_id, requested_model, requested_workspace, requested_timeout, requested_tool_policy = await _parse_multipart(request)
        else:
            try:
                body = await request.json()
            except Exception:
                return _error_json(400, "Invalid JSON body")
            if not isinstance(body, dict):
                return _error_json(400, "Invalid JSON body")
            body = cast(dict[str, Any], body)
            stream = body.get("stream", False)
            requested_model = body.get("model")
            requested_workspace = body.get("workspace")
            requested_timeout = body.get("timeout")
            requested_max_iterations = body.get("max_iterations")
            requested_tool_policy = body.get("tool_policy")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    if requested_model and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    request_timeout_s = timeout_s
    if requested_timeout is not None:
        try:
            request_timeout_s = float(requested_timeout)
        except (TypeError, ValueError):
            return _error_json(400, "timeout must be a number")
        if request_timeout_s <= 0:
            return _error_json(400, "timeout must be greater than 0")

    metadata: dict[str, Any] = {}
    if requested_max_iterations is not None:
        if not isinstance(requested_max_iterations, int) or isinstance(requested_max_iterations, bool) or requested_max_iterations <= 0:
            return _error_json(400, "max_iterations must be a positive integer")
        metadata["max_iterations"] = requested_max_iterations
    if requested_tool_policy is not None:
        if not isinstance(requested_tool_policy, dict):
            return _error_json(400, "tool_policy must be an object")
        normalized_policy: dict[str, list[str]] = {}
        for key in ("blocked_find_files_paths", "blocked_grep_paths", "blocked_read_file_paths", "allowed_grep_paths", "blocked_tool_names"):
            paths = requested_tool_policy.get(key)
            if paths is None:
                continue
            if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                return _error_json(400, f"tool_policy.{key} must be an array of strings")
            normalized_policy[key] = list(paths)
        for boolean_key in ("disable_all_tools", "disable_exec", "read_only_mode"):
            if boolean_key in requested_tool_policy:
                if not isinstance(requested_tool_policy[boolean_key], bool):
                    return _error_json(400, f"tool_policy.{boolean_key} must be a boolean")
                normalized_policy[boolean_key] = requested_tool_policy[boolean_key]
        blocked_exec_patterns = requested_tool_policy.get("blocked_exec_patterns")
        if blocked_exec_patterns is not None:
            if not isinstance(blocked_exec_patterns, list) or not all(isinstance(pattern, str) for pattern in blocked_exec_patterns):
                return _error_json(400, "tool_policy.blocked_exec_patterns must be an array of strings")
            normalized_policy["blocked_exec_patterns"] = list(blocked_exec_patterns)
        metadata["tool_policy"] = normalized_policy
    if session_id and str(session_id).startswith("review_"):
        metadata["review_session"] = True
    if requested_workspace:
        try:
            scope = validate_workspace_scope_payload(
                {"project_path": str(requested_workspace), "access_mode": "full"},
                default_workspace=agent_loop.workspace,
                default_restrict_to_workspace=agent_loop.restrict_to_workspace,
                source_channel="api",
            )
        except WorkspaceScopeError as exc:
            return _error_json(exc.status, exc.message)
        metadata[WORKSPACE_SCOPE_METADATA_KEY] = scope.metadata()

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = _app_value(
        request.app,
        _SESSION_LOCKS_KEY,
        "session_locks",
    )
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )
    async def _discard_api_progress(*_args: Any, **_kwargs: Any) -> None:
        # HTTP callers receive only the final response. Publishing intermediate
        # progress to the message bus can synchronously block the agent loop.
        return None

    # -- streaming path --
    if stream:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        await resp.prepare(request)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_failed = False
        emitted_content = False

        async def _on_stream(token: str) -> None:
            nonlocal emitted_content
            if token:
                emitted_content = True
            await queue.put(token)

        async def _on_stream_end(*_a: Any, **_kw: Any) -> None:
            # Agent stream-end callbacks mark generation segment boundaries.
            # Tool-backed requests may continue after a segment ends, so the
            # HTTP SSE stream is closed only when process_direct returns.
            return None

        async def _run() -> None:
            nonlocal stream_failed
            try:
                async with session_lock:
                    async with asyncio.timeout(timeout_s):
                        await _prepare_agent(request.app)
                        response = await agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_progress=_discard_api_progress,
                            on_stream=_on_stream,
                            on_stream_end=_on_stream_end,
                            metadata=metadata,
                        )
                    if not emitted_content:
                        response_text = _response_text(response)
                        if response_text.strip():
                            await queue.put(response_text)
            except Exception:
                stream_failed = True
                logger.exception("Streaming error for session {}", session_key)
            finally:
                await queue.put(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                await resp.write(_sse_chunk(token, model_name, chunk_id))
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if not stream_failed:
            await resp.write(_sse_chunk("", model_name, chunk_id, finish_reason="stop"))
            await resp.write(_SSE_DONE)
        return resp

    # -- non-streaming path (original logic) --
    usage_capture = _UsageCaptureHook()
    try:
        async with session_lock:
            try:
                async with asyncio.timeout(timeout_s):
                    await _prepare_agent(request.app)
                    response = await agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                        on_progress=_discard_api_progress,
                        metadata=metadata,
                        hooks=[usage_capture],
                    )
                response_text = _response_text(response)
                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, using fallback", session_key)
                    response_text = EMPTY_FINAL_RESPONSE_MESSAGE

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {request_timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(
        _chat_completion_response(response_text, model_name, usage_capture.usage)
    )


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models"""
    model_name = _app_value(request.app, _MODEL_NAME_KEY, "model_name", "nanobot")
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop: "AgentLoop | None",
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    api_key: str = "",
    prepare_agent: Callable[[], Awaitable[None]] | None = None,
    codex_relay: "CodexRelayConfig | None" = None,
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance, or None for provider-only relay mode.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
        api_key: Optional API key for Bearer-token authentication on API routes.
        prepare_agent: Optional application-owned readiness callback run before each turn.
        codex_relay: Optional server-mode Codex relay configuration.
    """
    relay_only = codex_relay is not None and codex_relay.mode == "server"
    if agent_loop is None and not relay_only:
        raise ValueError("agent_loop is required outside provider-only relay mode")

    app = web.Application(client_max_size=20 * 1024 * 1024)  # 20MB for base64 images
    if not relay_only:
        assert agent_loop is not None
        app[_AGENT_LOOP_KEY] = agent_loop
        app[_MODEL_NAME_KEY] = model_name
        app[_REQUEST_TIMEOUT_KEY] = request_timeout
        app[_SESSION_LOCKS_KEY] = {}  # per-user locks, keyed by session_key
        app[_PREPARE_AGENT_KEY] = prepare_agent

    @web.middleware
    async def auth_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        # Allow unauthenticated health checks.
        if request.path == "/health":
            return await handler(request)
        route_api_key = api_key
        if (
            request.path == "/v1/codex/relay"
            and codex_relay is not None
            and codex_relay.mode == "server"
        ):
            route_api_key = codex_relay.api_key.strip()
        if not route_api_key:
            return await handler(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _error_json(401, "Missing Authorization header. Use: Bearer <api_key>")
        if not constant_time_text_equal(auth[len("Bearer "):], route_api_key):
            return _error_json(401, "Invalid API key")
        return await handler(request)

    app.middlewares.append(auth_middleware)

    app.router.add_get("/health", handle_health)
    if relay_only:
        assert codex_relay is not None
        from nanobot.fork.api.codex_relay import create_codex_relay_handler

        # Relay mode is provider-only. Do not expose the normal AgentLoop API on
        # the OAuth host, otherwise relay clients could execute central tools.
        app.router.add_post("/v1/codex/relay", create_codex_relay_handler(codex_relay))
    else:
        app.router.add_post("/v1/chat/completions", handle_chat_completions)
        app.router.add_get("/v1/models", handle_models)
    return app
