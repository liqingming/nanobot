"""Tests for the config-driven provider-only Codex relay."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.api.server import create_app
from nanobot.config.schema import CodexRelayConfig, Config
from nanobot.fork.api.codex_relay import CodexRelayService
from nanobot.fork.providers.codex_relay import response_from_payload, response_to_payload
from nanobot.providers.base import LLMResponse, LLMUsage, ToolCallRequest
from nanobot.providers.factory import make_provider


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="unused")
    return agent


async def _client(app) -> TestClient:
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def _request(model: str = "openai-codex/gpt-5.6-sol") -> dict[str, object]:
    return {
        "version": 1,
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": None,
        "max_tokens": 100,
        "reasoning_effort": None,
        "tool_choice": None,
    }


def test_response_payload_round_trip_preserves_provider_result() -> None:
    original = LLMResponse(
        content="done",
        tool_calls=[ToolCallRequest(id="call-1", name="lookup", arguments={"q": "x"})],
        finish_reason="tool_calls",
        usage=LLMUsage.reported(input_tokens=10, output_tokens=4),
        reasoning_content="thinking",
        response_items=[{"type": "message"}],
        provider_diagnostics={"stage": "complete"},
    )

    restored = response_from_payload(response_to_payload(original))

    assert restored.content == original.content
    assert restored.finish_reason == "tool_calls"
    assert restored.tool_calls[0].arguments == {"q": "x"}
    assert restored.usage is not None
    assert restored.usage.total_tokens == 14
    assert restored.reasoning_content == "thinking"
    assert restored.response_items == [{"type": "message"}]


@pytest.mark.asyncio
async def test_relay_route_uses_dedicated_key_and_streams_deltas() -> None:
    relay = CodexRelayConfig(
        mode="server",
        api_key="relay-secret",
        allowed_models=["openai-codex/gpt-5.6-sol"],
    )

    async def fake_call(**kwargs):
        await kwargs["on_content_delta"]("hel")
        await kwargs["on_content_delta"]("lo")
        return LLMResponse(
            content="hello",
            usage=LLMUsage.reported(input_tokens=3, output_tokens=1),
        )

    with patch(
        "nanobot.providers.openai_codex_provider.OpenAICodexProvider._call_codex",
        new=AsyncMock(side_effect=fake_call),
    ):
        client = await _client(create_app(_agent(), api_key="normal-secret", codex_relay=relay))
        try:
            central_agent_api = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer normal-secret"},
                json={"messages": [{"role": "user", "content": "run a tool"}]},
            )
            assert central_agent_api.status == 404

            unauthorized = await client.post("/v1/codex/relay", json=_request())
            assert unauthorized.status == 401

            wrong_api_key = await client.post(
                "/v1/codex/relay",
                headers={"Authorization": "Bearer normal-secret"},
                json=_request(),
            )
            assert wrong_api_key.status == 401

            response = await client.post(
                "/v1/codex/relay",
                headers={"Authorization": "Bearer relay-secret"},
                json=_request(),
            )
            assert response.status == 200
            events = [json.loads(line) for line in (await response.text()).splitlines()]
            assert events[:2] == [
                {"type": "content_delta", "delta": "hel"},
                {"type": "content_delta", "delta": "lo"},
            ]
            assert events[-1]["type"] == "result"
            assert events[-1]["response"]["content"] == "hello"
        finally:
            await client.close()


def test_relay_rejects_disallowed_model_before_provider_call() -> None:
    relay = CodexRelayConfig(
        mode="server",
        api_key="relay-secret",
        allowed_models=["allowed-model"],
    )
    service = CodexRelayService(relay)

    with pytest.raises(PermissionError, match="not allowed"):
        service._parse_request(_request("other-model"))


def test_client_config_requires_secure_remote_endpoint() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        CodexRelayConfig(
            mode="client",
            endpoint="http://relay.example.test:8900",
            api_key="secret",
        )

    config = CodexRelayConfig(
        mode="client",
        endpoint="https://relay.example.test/",
        api_key="secret",
    )
    assert config.endpoint == "https://relay.example.test"


def test_root_config_round_trips_camel_case_relay_settings() -> None:
    config = Config.model_validate(
        {
            "codexRelay": {
                "mode": "client",
                "endpoint": "https://relay.example.test/",
                "apiKey": "secret",
                "requestTimeout": 321,
                "maxConcurrency": 7,
                "allowedModels": ["openai-codex/gpt-5.6-sol"],
            }
        }
    )

    assert config.codex_relay.request_timeout == 321
    dumped = config.model_dump(by_alias=True)
    assert dumped["codexRelay"]["endpoint"] == "https://relay.example.test"
    assert dumped["codexRelay"]["maxConcurrency"] == 7


def test_factory_routes_every_codex_provider_through_client_relay() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"model": "openai-codex/gpt-5.6-sol"}},
            "codexRelay": {
                "mode": "client",
                "endpoint": "https://relay.example.test",
                "apiKey": "secret",
            },
        }
    )

    provider = make_provider(config)

    assert provider.__class__.__name__ == "CodexRelayProvider"
    assert provider.default_model == "openai-codex/gpt-5.6-sol"
