from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from loguru import logger

import nanobot.providers.base as provider_base
from nanobot.providers.openai_codex_provider import (
    DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S,
    OpenAICodexProvider,
    _build_reasoning_options,
    _codex_error_response,
    _CodexHTTPError,
    _CodexSSEDiagnostics,
    _friendly_error,
    _request_codex,
    _should_retry_status,
    resolve_codex_first_event_timeout_s,
)
from nanobot.providers.openai_responses.parsing import ResponsesAPIError


def _mock_codex_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_token(**_kwargs):
        return SimpleNamespace(account_id="acct", access="token")

    monkeypatch.setattr(
        "nanobot.providers.openai_codex_provider.get_codex_token",
        fake_token,
    )


class _WarningCaptureLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args[0], args[1:]))

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("Codex diagnostics must not log exception tracebacks")


def _capture_codex_warnings(monkeypatch: pytest.MonkeyPatch) -> _WarningCaptureLogger:
    capture = _WarningCaptureLogger()
    monkeypatch.setattr("nanobot.providers.openai_codex_provider.logger", capture)
    return capture


def test_codex_first_event_timeout_parser_defaults_and_bounds() -> None:
    assert resolve_codex_first_event_timeout_s(env_value=None) == DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S
    assert resolve_codex_first_event_timeout_s(env_value="240") == 240
    assert resolve_codex_first_event_timeout_s(env_value="0") == DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S
    assert resolve_codex_first_event_timeout_s(env_value="bad") == DEFAULT_CODEX_FIRST_EVENT_TIMEOUT_S
    assert resolve_codex_first_event_timeout_s(env_value="7200") == 3600


def test_codex_blank_timeout_root_cause_reproduction() -> None:
    """Document why upstream produced a bare ``Error calling Codex:`` message."""
    exc = httpx.ReadTimeout("")
    legacy_content = f"Error calling Codex: {exc}"

    assert str(exc) == ""
    assert legacy_content == "Error calling Codex: "
    legacy_response = provider_base.LLMResponse(content=legacy_content, finish_reason="error")
    assert legacy_response.error_kind is None
    assert legacy_response.error_should_retry is None


def test_codex_http_friendly_error_omits_raw_body() -> None:
    raw = "raw upstream body with PRIVATE PROMPT MUST NOT APPEAR"

    message = _friendly_error(500, raw)

    assert message == "HTTP 500: Codex API request failed"
    assert "PRIVATE PROMPT MUST NOT APPEAR" not in message


@pytest.mark.asyncio
async def test_codex_request_non_200_populates_http_metadata(monkeypatch) -> None:
    original_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "2"},
            json={"error": {"type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}},
            request=request,
        )

    def fake_client(
        *,
        timeout: httpx.Timeout,
        verify: bool,
        **_kwargs: object,
    ) -> httpx.AsyncClient:
        assert timeout.connect == 90
        assert timeout.read == 180
        assert timeout.write == 90
        assert timeout.pool == 90
        assert verify is True
        return original_client(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr("nanobot.providers.openai_codex_provider.httpx.AsyncClient", fake_client)

    with pytest.raises(_CodexHTTPError) as caught:
        await _request_codex("https://codex.example/responses", {}, {"input": []}, verify=True)

    error = caught.value
    assert str(error) == "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    assert error.status_code == 429
    assert error.retry_after == 2.0
    assert error.error_type == "rate_limit_exceeded"
    assert error.error_code == "rate_limit_exceeded"
    assert error.should_retry is True


@pytest.mark.asyncio
async def test_codex_request_honors_stream_idle_timeout_env(monkeypatch) -> None:
    """NANOBOT_STREAM_IDLE_TIMEOUT_S overrides the default Codex stream timeout."""
    monkeypatch.setenv("NANOBOT_STREAM_IDLE_TIMEOUT_S", "5")
    original_client = httpx.AsyncClient
    seen: dict[str, httpx.Timeout] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    def fake_client(
        *,
        timeout: httpx.Timeout,
        verify: bool,
        **_kwargs: object,
    ) -> httpx.AsyncClient:
        seen["timeout"] = timeout
        return original_client(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr("nanobot.providers.openai_codex_provider.httpx.AsyncClient", fake_client)

    await _request_codex("https://codex.example/responses", {}, {"input": []}, verify=True)

    assert seen["timeout"].connect == 5
    assert seen["timeout"].read == 180
    assert seen["timeout"].write == 5
    assert seen["timeout"].pool == 5


@pytest.mark.asyncio
async def test_codex_request_uses_configured_proxy(monkeypatch) -> None:
    original_client = httpx.AsyncClient
    seen: dict[str, object] = {}
    proxy = "http://127.0.0.1:23458"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    def fake_client(
        *,
        timeout: int,
        verify: bool,
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> httpx.AsyncClient:
        seen["proxy"] = proxy
        seen["trust_env"] = trust_env
        return original_client(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr("nanobot.providers.openai_codex_provider.httpx.AsyncClient", fake_client)

    await _request_codex(
        "https://codex.example/responses",
        {},
        {"input": []},
        verify=True,
        proxy=proxy,
    )

    assert seen == {"proxy": proxy, "trust_env": False}


@pytest.mark.asyncio
async def test_codex_prompt_cache_key_uses_stable_conversation_prefix(monkeypatch) -> None:
    bodies: list[dict] = []

    _mock_codex_token(monkeypatch)

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = proxy, on_thinking_delta, on_tool_call_delta
        bodies.append(body)
        return "ok", [], "stop", {}, None

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    await provider.chat(
        [
            {"role": "system", "content": "You are nanobot."},
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first answer"},
        ],
    )
    await provider.chat(
        [
            {"role": "system", "content": "You are nanobot."},
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ],
    )
    await provider.chat(
        [
            {"role": "system", "content": "You are nanobot."},
            {"role": "user", "content": "different request"},
            {"role": "assistant", "content": "first answer"},
        ],
    )

    assert bodies[0]["prompt_cache_key"] == bodies[1]["prompt_cache_key"]
    assert bodies[0]["prompt_cache_key"] != bodies[2]["prompt_cache_key"]


@pytest.mark.asyncio
async def test_codex_provider_exposes_response_items_for_runner_persistence(monkeypatch) -> None:
    _mock_codex_token(monkeypatch)
    response_items = [{
        "type": "reasoning", "id": "rs_1",
        "encrypted_content": "opaque-token", "summary": [],
    }]

    async def fake_request(*args, **kwargs):
        return "", [], "stop", {}, None, response_items, {"kind": "codex_sse"}

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    response = await OpenAICodexProvider().chat([{"role": "user", "content": "hello"}])

    assert response.response_items == response_items
    assert response.provider_diagnostics == {"kind": "codex_sse"}


@pytest.mark.asyncio
async def test_codex_timeout_error_is_typed_and_retryable(monkeypatch) -> None:
    _mock_codex_token(monkeypatch)

    async def fake_request(*args, **kwargs):
        raise httpx.ReadTimeout("")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert response.content == (
        "Error calling Codex (ReadTimeout): timed out waiting for response"
    )
    assert response.error_kind == "timeout"
    assert response.error_should_retry is True


@pytest.mark.asyncio
async def test_codex_provider_passes_proxy_to_oauth_and_response_request(monkeypatch) -> None:
    proxy = "http://127.0.0.1:23458"
    seen: dict[str, object] = {}

    def fake_token(*, proxy=None):
        seen["token_proxy"] = proxy
        return SimpleNamespace(account_id="acct", access="token")

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = url, headers, body, verify, on_content_delta, on_thinking_delta, on_tool_call_delta
        seen["request_proxy"] = proxy
        return "ok", [], "stop", {}, None

    monkeypatch.setattr("nanobot.providers.openai_codex_provider.get_codex_token", fake_token)
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider(proxy=proxy)
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.content == "ok"
    assert seen["token_proxy"] == proxy
    assert seen["request_proxy"] == proxy


@pytest.mark.asyncio
async def test_codex_timeout_error_writes_diagnostic_log(monkeypatch) -> None:
    log_capture = _capture_codex_warnings(monkeypatch)
    _mock_codex_token(monkeypatch)

    async def fake_request(*args: Any, **kwargs: Any):
        raise httpx.ReadTimeout("")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.content == (
        "Error calling Codex (ReadTimeout): timed out waiting for response"
    )
    assert log_capture.calls == [
        (
            "Codex API request failed: type={} kind={} retryable={} status={} "
            "error_type={} error_code={} retry_after={} summary={}",
            (
                "ReadTimeout",
                "timeout",
                True,
                None,
                None,
                None,
                None,
                "ReadTimeout timeout",
            ),
        )
    ]


@pytest.mark.asyncio
async def test_codex_diagnostic_log_omits_prompt_content(monkeypatch) -> None:
    sink = io.StringIO()
    logger.enable("nanobot")
    handler_id = logger.add(sink, format="{message}", backtrace=True, diagnose=True)
    try:
        _mock_codex_token(monkeypatch)

        async def fake_request(*args: Any, **kwargs: Any):
            raise httpx.ReadTimeout("")

        monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

        provider = OpenAICodexProvider()
        response = await provider.chat(
            [{"role": "user", "content": "PRIVATE PROMPT MUST NOT APPEAR"}]
        )
    finally:
        logger.remove(handler_id)

    log_text = sink.getvalue()
    assert response.error_kind == "timeout"
    assert "Codex API request failed" in log_text
    assert "ReadTimeout" in log_text
    assert "PRIVATE PROMPT MUST NOT APPEAR" not in log_text


@pytest.mark.asyncio
async def test_codex_retry_uses_structured_timeout_metadata(monkeypatch) -> None:
    calls = 0
    delays: list[float] = []

    _mock_codex_token(monkeypatch)

    async def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("")
        return "ok", [], "stop", {}, None

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)
    monkeypatch.setattr(provider_base.asyncio, "sleep", fake_sleep)

    provider = OpenAICodexProvider()
    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert response.content == "ok"
    assert calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_codex_http_error_preserves_status_and_retry_after(monkeypatch) -> None:
    _mock_codex_token(monkeypatch)

    async def fake_request(*args, **kwargs):
        raise _CodexHTTPError(
            "HTTP 503: backend unavailable",
            status_code=503,
            retry_after=2.5,
            error_type="server_error",
            error_code="overloaded",
        )

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert response.content == "Error calling Codex (CodexHTTPError): HTTP 503: backend unavailable"
    assert response.error_status_code == 503
    assert response.error_kind == "http"
    assert response.error_type == "server_error"
    assert response.error_code == "overloaded"
    assert response.retry_after == 2.5
    assert response.error_should_retry is True


@pytest.mark.asyncio
async def test_codex_http_diagnostic_log_omits_raw_body(monkeypatch) -> None:
    log_capture = _capture_codex_warnings(monkeypatch)
    _mock_codex_token(monkeypatch)

    async def fake_request(*args: Any, **kwargs: Any):
        raise _CodexHTTPError(
            _friendly_error(500, "raw upstream body with PRIVATE PROMPT MUST NOT APPEAR"),
            status_code=500,
            error_type="server_error",
            error_code="overloaded",
        )

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.content == "Error calling Codex (CodexHTTPError): HTTP 500: Codex API request failed"
    assert log_capture.calls == [
        (
            "Codex API request failed: type={} kind={} retryable={} status={} "
            "error_type={} error_code={} retry_after={} summary={}",
            (
                "CodexHTTPError",
                "http",
                True,
                500,
                "server_error",
                "overloaded",
                None,
                "HTTP 500 type=server_error code=overloaded",
            ),
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "error_code", "expected_retry"),
    [
        ("rate_limit_exceeded", "rate_limit_exceeded", True),
        ("insufficient_quota", "insufficient_quota", False),
    ],
)
async def test_codex_429_preserves_retry_semantics(
    monkeypatch,
    error_type: str,
    error_code: str,
    expected_retry: bool,
) -> None:
    _mock_codex_token(monkeypatch)

    async def fake_request(*args: Any, **kwargs: Any):
        raise _CodexHTTPError(
            "ChatGPT usage quota exceeded or rate limit triggered. Please try again later.",
            status_code=429,
            error_type=error_type,
            error_code=error_code,
            should_retry=expected_retry,
        )

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.error_status_code == 429
    assert response.error_type == error_type
    assert response.error_code == error_code
    assert response.error_should_retry is expected_retry


def test_codex_429_friendly_message_fallback_does_not_override_unknown_retry() -> None:
    response = _codex_error_response(
        _CodexHTTPError(_friendly_error(429, ""), status_code=429)
    )

    assert response.error_status_code == 429
    assert response.error_should_retry is True


@pytest.mark.parametrize(
    ("raw", "expected_retry"),
    [
        ('{"error":{"type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}', True),
        ('{"error":{"type":"insufficient_quota","code":"insufficient_quota"}}', False),
    ],
)
def test_codex_429_classification_uses_raw_error_semantics(
    raw: str,
    expected_retry: bool,
) -> None:
    error_type, error_code = provider_base.LLMProvider._extract_error_type_code(raw)

    assert _should_retry_status(429, error_type, error_code, raw) is expected_retry


def test_codex_reasoning_options_request_summary_without_forcing_effort() -> None:
    assert _build_reasoning_options(None) == {"summary": "auto"}
    assert _build_reasoning_options("high") == {"summary": "auto", "effort": "high"}
    assert _build_reasoning_options("none") == {"effort": "none"}


@pytest.mark.asyncio
async def test_codex_stream_surfaces_reasoning_summary(monkeypatch) -> None:
    def fake_token(**_kwargs):
        return SimpleNamespace(account_id="acct", access="token")

    monkeypatch.setattr(
        "nanobot.providers.openai_codex_provider.get_codex_token",
        fake_token,
    )

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = url, headers, verify, proxy, on_tool_call_delta
        assert body["reasoning"] == {"summary": "auto", "effort": "medium"}
        if on_content_delta:
            await on_content_delta("answer")
        if on_thinking_delta:
            await on_thinking_delta("summary")
        return "answer", [], "stop", {"prompt_tokens": 10, "completion_tokens": 5}, "summary"

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    content_deltas: list[str] = []
    thinking_deltas: list[str] = []

    response = await provider.chat_stream(
        [{"role": "user", "content": "hi"}],
        reasoning_effort="medium",
        on_content_delta=lambda delta: _append(content_deltas, delta),
        on_thinking_delta=lambda delta: _append(thinking_deltas, delta),
    )

    assert content_deltas == ["answer"]
    assert thinking_deltas == ["summary"]
    assert response.content == "answer"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert response.reasoning_content == "summary"


async def _append(target: list[str], value: str) -> None:
    target.append(value)


def test_codex_sse_diagnostics_normal_stream_is_low_sensitivity() -> None:
    secret = "private answer that must never appear in runtime diagnostics"
    diagnostics = _CodexSSEDiagnostics()
    diagnostics.observe({
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "output_index": 0,
        "content_index": 0,
        "delta": secret,
    })
    diagnostics.observe({
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"status": "completed", "output": [{"text": secret}]},
    })

    result = diagnostics.finish(secret)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result == {
        "kind": "codex_sse",
        "event_count": 2,
        "delta_count": 1,
        "delta_chars": len(secret),
        "repeated_event_count": 0,
        "repeated_delta_count": 0,
        "first_sequence_number": 1,
        "last_sequence_number": 2,
        "repeated_sequence_count": 0,
        "response_created_count": 0,
        "response_completed_count": 1,
        "unique_response_id_count": 0,
        "content_chars": len(secret),
        "content_fingerprint": result["content_fingerprint"],
        "exact_half_repeat": False,
    }
    assert len(result["content_fingerprint"]) == 16
    assert secret not in serialized
    assert "private answer" not in serialized


def test_codex_sse_diagnostics_detects_event_and_exact_half_replay() -> None:
    diagnostics = _CodexSSEDiagnostics()
    event = {
        "type": "response.output_text.delta",
        "sequence_number": 7,
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "delta": "same answer",
    }

    diagnostics.observe(event)
    diagnostics.observe(dict(event))
    result = diagnostics.finish("same answersame answer")

    assert result["event_count"] == 2
    assert result["delta_count"] == 2
    assert result["repeated_event_count"] == 1
    assert result["repeated_delta_count"] == 1
    assert result["repeated_sequence_count"] == 1
    assert result["first_sequence_number"] == 7
    assert result["last_sequence_number"] == 7
    assert result["exact_half_repeat"] is True


def test_codex_sse_diagnostics_handles_non_ascii_content() -> None:
    diagnostics = _CodexSSEDiagnostics()

    assert diagnostics.finish("\u4e2d\u6587\u4e2d\u6587")["exact_half_repeat"] is True
    assert diagnostics.finish("\u4e2d\u6587\u56de\u7b54")["exact_half_repeat"] is False


@pytest.mark.asyncio
async def test_codex_provider_attaches_diagnostics_without_changing_content(monkeypatch) -> None:
    _mock_codex_token(monkeypatch)
    diagnostics = {
        "kind": "codex_sse",
        "event_count": 4,
        "exact_half_repeat": True,
    }

    async def fake_request(*args: Any, **kwargs: Any):
        return "answeranswer", [], "stop", {}, None, diagnostics

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    response = await OpenAICodexProvider().chat([{"role": "user", "content": "hello"}])

    assert response.content == "answeranswer"
    assert response.provider_diagnostics == diagnostics


@pytest.mark.asyncio
async def test_fork_codex_empty_generic_exception_message_includes_exception_type(monkeypatch) -> None:
    _mock_codex_token(monkeypatch)

    async def fake_request(*args: Any, **kwargs: Any):
        raise RuntimeError("")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider()
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert response.content == "Error calling Codex (RuntimeError): unexpected error"

@pytest.mark.asyncio
async def test_codex_provider_tolerates_oauth_get_token_without_proxy_parameter(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_token():
        seen["token_called"] = True
        return SimpleNamespace(account_id="acct", access="token")

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = url, headers, body, verify, on_content_delta, on_thinking_delta, on_tool_call_delta
        seen["request_proxy"] = proxy
        return "ok", [], "stop", {}, None

    monkeypatch.setattr("nanobot.providers.openai_codex_provider.get_codex_token", fake_token)
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)

    provider = OpenAICodexProvider(proxy="http://127.0.0.1:23458")
    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response.content == "ok"
    assert seen == {"token_called": True, "request_proxy": "http://127.0.0.1:23458"}

@pytest.mark.asyncio
async def test_codex_responses_lite_uses_catalog_protocol(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "use_responses_lite": True,
                        "default_verbosity": "low",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    _mock_codex_token(monkeypatch)
    seen: dict[str, Any] = {}

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = url, verify, proxy, on_content_delta, on_thinking_delta, on_tool_call_delta
        seen["headers"] = headers
        seen["body"] = body
        return "ok", [], "stop", {}, None

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)
    provider = OpenAICodexProvider(default_model="openai-codex/gpt-5.6-sol")

    response = await provider.chat(
        [
            {"role": "system", "content": "You are nanobot."},
            {"role": "user", "content": "hello"},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one file",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        reasoning_effort="high",
    )

    assert response.content == "ok"
    headers = seen["headers"]
    body = seen["body"]
    assert headers["x-openai-internal-codex-responses-lite"] == "true"
    assert body["instructions"] == ""
    assert "tools" not in body
    assert body["input"][0] == {
        "type": "additional_tools",
        "role": "developer",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read one file",
            "parameters": {"type": "object", "properties": {}},
        }],
    }
    assert body["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are nanobot."}],
    }
    assert body["input"][2]["role"] == "user"
    assert body["reasoning"] == {"context": "all_turns", "effort": "high"}
    assert body["parallel_tool_calls"] is False
    assert body["text"] == {"verbosity": "low"}


@pytest.mark.asyncio
async def test_codex_legacy_responses_preserves_existing_protocol(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models_cache.json").write_text(
        json.dumps({"models": [{"slug": "gpt-5.5", "use_responses_lite": False}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    _mock_codex_token(monkeypatch)
    seen: dict[str, Any] = {}

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = url, verify, proxy, on_content_delta, on_thinking_delta, on_tool_call_delta
        seen["headers"] = headers
        seen["body"] = body
        return "ok", [], "stop", {}, None

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)
    provider = OpenAICodexProvider(default_model="gpt-5.5")

    await provider.chat(
        [
            {"role": "system", "content": "You are nanobot."},
            {"role": "user", "content": "hello"},
        ],
        tools=[{"name": "read_file", "parameters": {"type": "object"}}],
    )

    headers = seen["headers"]
    body = seen["body"]
    assert "x-openai-internal-codex-responses-lite" not in headers
    assert body["instructions"] == "You are nanobot."
    assert body["input"][0]["role"] == "user"
    assert body["tools"][0]["name"] == "read_file"
    assert body["reasoning"] == {"summary": "auto"}
    assert body["parallel_tool_calls"] is True
    assert body["text"] == {"verbosity": "medium"}


def test_codex_responses_lite_reasoning_requires_all_turns_context() -> None:
    assert _build_reasoning_options(None, use_responses_lite=True) == {
        "context": "all_turns"
    }


def test_codex_model_catalog_clamps_configured_context_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "context_window": 372_000,
                        "max_context_window": 372_000,
                        "effective_context_window_percent": 95,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    provider = OpenAICodexProvider(default_model="gpt-5.6-sol")

    assert provider.resolve_context_window_tokens("gpt-5.6-sol", 1_000_000) == 372_000
    assert provider.input_token_budget(372_000, 65_536) == 353_400


def test_codex_model_catalog_falls_back_to_config_when_model_is_unknown(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "known-model",
                        "context_window": 372_000,
                        "effective_context_window_percent": 95,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    provider = OpenAICodexProvider(default_model="known-model")
    assert provider.input_token_budget(372_000, 65_536) == 353_400

    assert provider.resolve_context_window_tokens("unknown-model", 1_000_000) == 1_000_000
    assert provider.input_token_budget(1_000_000, 65_536) == 933_440


def test_codex_response_failed_preserves_context_error_metadata() -> None:
    error = ResponsesAPIError(
        {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "input too long",
            "param": "input",
        }
    )

    response = _codex_error_response(error)

    assert response.finish_reason == "error"
    assert response.error_type == "invalid_request_error"
    assert response.error_code == "context_length_exceeded"
    assert response.error_kind == "context_length"
    assert response.error_should_retry is False


def test_codex_response_server_error_without_status_is_retryable() -> None:
    error = ResponsesAPIError({
        "type": "server_error",
        "code": "server_error",
        "message": "You can retry your request.",
        "param": None,
    })

    response = _codex_error_response(error)

    assert response.finish_reason == "error"
    assert response.error_kind == "server"
    assert response.error_type == "server_error"
    assert response.error_code == "server_error"
    assert response.error_should_retry is True

@pytest.mark.asyncio
async def test_codex_reuses_turn_state_only_within_same_turn(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models_cache.json").write_text(
        json.dumps({"models": [{"slug": "gpt-5.6-sol", "use_responses_lite": True}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    _mock_codex_token(monkeypatch)
    requests: list[tuple[dict[str, str], dict[str, Any]]] = []

    async def fake_request(
        url,
        headers,
        body,
        verify,
        proxy=None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ):
        _ = url, verify, proxy, on_content_delta, on_thinking_delta, on_tool_call_delta
        requests.append((dict(headers), body))
        if len(requests) == 1:
            headers.capture_turn_state("route-state-1")
        return "ok", [], "stop", {}, None

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", fake_request)
    provider = OpenAICodexProvider(default_model="gpt-5.6-sol")
    messages = [{"role": "user", "content": "inspect workspace"}]

    await provider.chat(
        messages,
        request_context={"session_key": "cli:topic", "turn_id": "turn-1"},
    )
    await provider.chat(
        messages,
        request_context={"session_key": "cli:topic", "turn_id": "turn-1"},
    )
    await provider.chat(
        messages,
        request_context={"session_key": "cli:topic", "turn_id": "turn-2"},
    )

    first_headers, first_body = requests[0]
    second_headers, second_body = requests[1]
    third_headers, third_body = requests[2]
    assert "x-codex-turn-state" not in first_headers
    assert second_headers["x-codex-turn-state"] == "route-state-1"
    assert "x-codex-turn-state" not in third_headers
    assert first_headers["session-id"] == second_headers["session-id"]
    assert second_headers["session-id"] == third_headers["session-id"]
    assert first_headers["thread-id"] == second_headers["thread-id"]
    assert second_headers["thread-id"] == third_headers["thread-id"]
    assert first_headers["x-codex-turn-metadata"] == second_headers["x-codex-turn-metadata"]
    assert second_headers["x-codex-turn-metadata"] != third_headers["x-codex-turn-metadata"]
    assert first_body["prompt_cache_key"] == first_headers["session-id"]
    assert second_body["client_metadata"]["thread_id"] == second_headers["thread-id"]
    assert third_body["client_metadata"]["turn_id"] != second_body["client_metadata"]["turn_id"]
