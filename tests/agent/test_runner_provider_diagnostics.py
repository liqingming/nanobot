from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider, LLMResponse


@pytest.mark.asyncio
async def test_runner_writes_provider_diagnostics_to_model_response_event() -> None:
    diagnostics = {
        "kind": "codex_sse",
        "event_count": 6,
        "repeated_event_count": 2,
        "exact_half_repeat": True,
    }
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="answeranswer",
        provider_diagnostics=diagnostics,
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    events: list[tuple[str, dict]] = []

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
        event_logger=lambda event, fields: events.append((event, fields)),
    ))

    response_event = [fields for event, fields in events if event == "runner.model.response"][-1]
    assert result.final_content == "answeranswer"
    assert response_event["provider_diagnostics"] == diagnostics
