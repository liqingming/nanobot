"""上下文压力使用最近输入量，消耗统计保持累计。"""

from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.fork.agent.context_usage import (
    accumulate_usage,
    context_input_tokens,
    with_context_usage,
)
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def test_consumption_sums_but_context_replaces():
    usage = {}
    for tokens in (100, 200, 50):
        accumulate_usage(usage, with_context_usage(
            {"prompt_tokens": tokens, "cached_tokens": tokens // 2}, {}, lambda: 999,
        ))
    assert usage["prompt_tokens"] == 350
    assert usage["cached_tokens"] == 175
    assert context_input_tokens(usage) == 50
    assert context_input_tokens({"prompt_tokens": 8_290_000}) == 0


def test_codex_cumulative_delta_is_never_context_size():
    usage = {"prompt_tokens": 1_600_000}
    measured = with_context_usage(usage, {
        "transport": "codex_app_server", "context_input_tokens": 42_000,
    }, lambda: 55_000)
    assert measured["prompt_tokens"] == 1_600_000
    assert measured["context_input_tokens"] == 42_000
    assert measured["context_input_estimated"] == 0
    estimated = with_context_usage(usage, {"transport": "codex_app_server"}, lambda: 55_000)
    assert estimated["context_input_tokens"] == 55_000
    assert estimated["context_input_estimated"] == 1


async def test_runner_retains_total_and_latest_context_through_tools():
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="c", name="read_file", arguments={})],
                    usage={"prompt_tokens": 100, "completion_tokens": 10}),
        LLMResponse(content="done", usage={"prompt_tokens": 200, "completion_tokens": 20}),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")
    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "task"}],
        tools=tools, model="test", max_iterations=3, max_tool_result_chars=16000,
    ))
    assert result.usage["prompt_tokens"] == 300
    assert result.usage["completion_tokens"] == 30
    assert context_input_tokens(result.usage) == 200


def test_turn_summary_uses_latest_context_not_turn_consumption():
    from types import SimpleNamespace

    from nanobot.agent.loop import AgentLoop

    loop = SimpleNamespace(
        _last_usage={"prompt_tokens": 8_290_000, "context_input_tokens": 42_000},
        context_window_tokens=100_000, _last_tool_events=[], _pattern_store=None,
        _prev_consolidated={}, _last_user_input={}, _last_turn_summary={},
    )
    session = SimpleNamespace(last_consolidated=0, messages=[{"role": "user", "content": "task"}])
    AgentLoop._capture_turn_summary(loop, "topic", [], session, "task", "completed")
    assert loop._last_turn_summary["topic"].pressure_pct == 42


def test_finalization_usage_merge_preserves_latest_context():
    merged = AgentRunner._merge_usage(
        {"prompt_tokens": 1000, "context_input_tokens": 1000},
        {"prompt_tokens": 500, "context_input_tokens": 500},
    )
    assert merged["prompt_tokens"] == 1500
    assert merged["context_input_tokens"] == 500
