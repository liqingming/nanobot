"""预算在普通请求、会话归档和 Codex 路径中保持一致。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.context_governance import ContextGovernanceConfig, ContextGovernor
from nanobot.agent.memory import Consolidator
from nanobot.fork.agent.context_budget import resolve_context_budget
from nanobot.fork.providers.codex_native_context import native_input_budget
from nanobot.providers.base import GenerationSettings, LLMProvider


@pytest.mark.parametrize("limit, expected", [(None, 6976), (2000, 2000), (20000, 6976)])
def test_all_consumers_share_budget(limit, expected):
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings(max_tokens=2000)
    config = ContextGovernanceConfig(
        provider=provider, model="test", tools=None, workspace=None, data_dir=None,
        session_key=None, max_tool_result_chars=16000,
        context_window_tokens=10000, context_block_limit=limit,
    )
    consolidator = Consolidator(
        store=MagicMock(), provider=provider, model="test", sessions=MagicMock(),
        context_window_tokens=10000, build_messages=MagicMock(),
        get_tool_definitions=lambda: [], max_completion_tokens=2000,
        context_block_limit=limit,
    )
    assert ContextGovernor.input_budget(config) == expected
    assert consolidator._input_token_budget == expected
    assert native_input_budget(provider, {"native_context": {
        "context_window_tokens": 10000, "context_block_limit": limit,
    }}) == expected


@pytest.mark.parametrize("window", [None, 0, -1, True])
def test_unknown_window_is_not_exhaustion(window):
    budget = resolve_context_budget(SimpleNamespace(), window)
    assert budget.input_tokens is None
    assert budget.window_tokens is None
    assert budget.output_reserve == 4096


def test_exhaustion_is_known_zero():
    budget = resolve_context_budget(SimpleNamespace(), 1000, 2000, 9000)
    assert budget.input_tokens == 0
    assert budget.provider_budget == 0


def test_provider_override_is_authoritative():
    class Provider:
        def input_token_budget(self, window, output, safety):
            return 3000

    assert resolve_context_budget(Provider(), 10000, 1000).input_tokens == 3000
    assert resolve_context_budget(Provider(), 10000, 1000, 2500).input_tokens == 2500


def test_invalid_mock_output_uses_safe_default():
    budget = resolve_context_budget(MagicMock(), 10000, True)
    assert budget.output_reserve == 4096
    assert budget.input_tokens == 4880
    assert set(budget.diagnostics()) == {
        "window_tokens", "output_reserve", "safety_reserve",
        "provider_budget", "block_limit", "input_tokens",
    }


def test_provider_change_invalidates_consolidation_probe():
    provider = SimpleNamespace(generation=GenerationSettings(max_tokens=2000))
    consolidator = Consolidator(
        store=MagicMock(), provider=provider, model="test", sessions=MagicMock(),
        context_window_tokens=10000, build_messages=MagicMock(),
        get_tool_definitions=lambda: [],
    )
    consolidator._background_probe_cache["test"] = (3000, 4, 0)
    consolidator.set_provider(provider, "other", 8000)
    assert consolidator._background_probe_cache == {}


async def test_runner_stops_on_zero_budget_without_model_request():
    from unittest.mock import AsyncMock

    from nanobot.agent.runner import AgentRunner, AgentRunSpec

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []
    messages = [{"role": "user", "content": "不得修改数据"}]
    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=messages, tools=tools, model="test", max_iterations=1,
        max_tool_result_chars=16000, context_window_tokens=2000, max_tokens=4000,
        context_block_limit=100000,
    ))
    assert result.stop_reason == "error"
    assert "预算已耗尽" in result.error
    assert result.messages == messages
    provider.chat_with_retry.assert_not_awaited()


async def test_zero_budget_does_not_archive_history():
    from unittest.mock import AsyncMock

    from nanobot.session.manager import Session

    session = Session(key="test")
    session.add_message("user", "保留原始需求")
    sessions = MagicMock()
    sessions.get_or_create.return_value = session
    consolidator = Consolidator(
        store=MagicMock(), provider=MagicMock(), model="test", sessions=sessions,
        context_window_tokens=1000, build_messages=MagicMock(),
        get_tool_definitions=lambda: [],
    )
    consolidator.archive = AsyncMock()
    await consolidator.maybe_consolidate_by_tokens(session)
    consolidator.archive.assert_not_awaited()
    assert session.last_consolidated == 0


@pytest.mark.parametrize("limit, expected", [(None, 6976), (2000, 2000), (20000, 6976)])
def test_history_replay_uses_same_budget(limit, expected):
    from nanobot.agent.loop import AgentLoop

    loop = object.__new__(AgentLoop)
    loop.provider = SimpleNamespace(generation=GenerationSettings(max_tokens=2000))
    loop.context_window_tokens = 10000
    loop.context_block_limit = limit
    assert loop._replay_token_budget() == expected
