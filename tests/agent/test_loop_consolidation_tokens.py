from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    ProviderConversationState,
)
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT


def _make_loop(
    tmp_path,
    *,
    estimated_tokens: int,
    context_window_tokens: int,
    max_tokens: int = 0,
) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=max_tokens)
    provider.estimate_prompt_tokens.return_value = (estimated_tokens, "test-counter")
    _response = LLMResponse(content="ok", tool_calls=[])
    provider.chat_with_retry = AsyncMock(return_value=_response)
    provider.chat_stream_with_retry = AsyncMock(return_value=_response)

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=context_window_tokens,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator._SAFETY_BUFFER = 0
    return loop


@pytest.mark.asyncio
async def test_runner_pressure_commits_summary_and_current_delta(tmp_path) -> None:
    loop = _make_loop(
        tmp_path, estimated_tokens=100, context_window_tokens=1_624, max_tokens=100,
    )
    loop.context_strategy = "legacy"
    loop.provider.can_resume_conversation_state.return_value = False
    loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": role, "content": f"old-{role}-{turn}"}
        for turn in range(6)
        for role in ("user", "assistant")
    ]
    loop.sessions.save(session)

    def estimate(messages, _tools, _model):
        contents = [str(message.get("content")) for message in messages]
        if contents and "SNIP" in contents[-1]:
            return 300, "test-counter"
        if any(content.startswith("old-") for content in contents):
            return 600, "test-counter"
        return 100, "test-counter"

    loop.provider.estimate_prompt_tokens.side_effect = estimate
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="<continuation>Current checkpoint.</continuation>"
                "<memory-candidates>(nothing)</memory-candidates>",
    ))
    loop.provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="done"))

    result = await loop.process_direct("continue the task", session_key="cli:test")

    assert result.content == "done"
    assert loop.provider.chat_with_retry.await_count == 1
    assert loop.provider.chat_stream_with_retry.await_count == 1
    model_request = loop.provider.chat_stream_with_retry.await_args_list[0].kwargs["messages"]
    assert "Current checkpoint." in model_request[0]["content"]
    assert model_request[1]["content"] == SUMMARY_CONTINUATION_TEXT
    assert model_request[2]["content"] == "continue the task"

    reloaded = loop.sessions.get_or_create("cli:test")
    assert reloaded.messages[0]["content"] == "old-user-0"
    assert reloaded.metadata["_last_summary"]["text"] == "Current checkpoint."
    assert reloaded.messages[reloaded.last_archived]["content"] == (
        SUMMARY_CONTINUATION_TEXT
    )
    assert [message["content"] for message in reloaded.get_history()] == [
        SUMMARY_CONTINUATION_TEXT,
        "continue the task",
        "done",
    ]


@pytest.mark.asyncio
async def test_native_provider_compaction_commits_portable_terminal_checkpoint(
    tmp_path,
) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=2_000)
    session = loop.sessions.get_or_create("cli:native")
    session.messages = [
        {"role": "user", "content": "accepted history"},
        {"role": "assistant", "content": "accepted answer"},
    ]
    loop.sessions.save(session)
    compacted_state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="test-model",
        version=1,
        payload={"items": [{"type": "compaction", "encrypted_content": "opaque"}]},
    )
    loop.provider.can_resume_conversation_state.return_value = True
    loop.provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="done",
        provider_state=compacted_state,
        provider_compaction_applied=True,
        provider_compaction_state=compacted_state,
        provider_compaction_scope="current_request",
    ))
    loop.consolidator.summarize_provider_compaction = AsyncMock(
        return_value="portable terminal checkpoint",
    )

    result = await loop.process_direct("continue", session_key="cli:native")

    assert result.content == "done"
    summarize = loop.consolidator.summarize_provider_compaction
    summarize.assert_awaited_once()
    assert summarize.await_args.args[0] == compacted_state
    accepted = summarize.await_args.args[1]
    accepted_contents = [message.get("content") for message in accepted]
    assert "accepted history" in accepted_contents
    assert "accepted answer" in accepted_contents
    assert "continue" in accepted_contents
    assert "done" not in accepted_contents
    reloaded = loop.sessions.get_or_create("cli:native")
    assert reloaded.provider_state is None
    assert reloaded.metadata["_last_summary"]["text"] == (
        "portable terminal checkpoint"
    )
    assert reloaded.messages[reloaded.last_archived]["content"] == (
        SUMMARY_CONTINUATION_TEXT
    )
    assert [message["content"] for message in reloaded.get_history()] == [
        SUMMARY_CONTINUATION_TEXT,
        "done",
    ]

@pytest.mark.parametrize("strategy", ["transactional", "legacy"])
async def test_native_preflight_reaches_model_with_full_history(tmp_path, strategy):
    from copy import deepcopy

    loop = _make_loop(tmp_path, estimated_tokens=8000, context_window_tokens=10000)
    loop.provider.supports_native_context_compaction = True
    loop.context_strategy = strategy
    loop._max_messages = 2
    loop._schedule_background = lambda coro: coro.close()
    session = loop.sessions.get_or_create("cli:native-resume")
    for i in range(3):
        session.add_message("user", f"历史约束 {i}：禁止提交")
        session.add_message("assistant", f"未完成任务 {i}")
    original = deepcopy(session.messages)
    loop.sessions.save(session)

    response = await loop.process_direct("继续", session_key=session.key)

    assert response is not None and response.content == "ok"
    assert session.last_consolidated == 0
    assert session.messages[:len(original)] == original
    assert loop.context.memory.read_unprocessed_history(0) == []
    calls = (loop.provider.chat_with_retry.await_args_list
             + loop.provider.chat_stream_with_retry.await_args_list)
    assert len(calls) == 1
    sent = calls[0].kwargs["messages"]
    for i in range(3):
        assert any(f"历史约束 {i}：禁止提交" in str(m.get("content")) for m in sent)


async def test_dispatch_exposes_summary_failure_with_request_correlation(tmp_path):
    from nanobot.bus.events import InboundMessage
    from nanobot.fork.agent.summary_transaction import SummaryTransactionError

    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=10000)
    reason = "上下文摘要失败，已保留原文；本次请求停止，未裁剪重试。"
    loop._run_agent_loop = AsyncMock(
        side_effect=SummaryTransactionError(reason),
    )
    await loop._dispatch(InboundMessage(
        channel="cli", sender_id="user", chat_id="summary-failure", content="继续",
        metadata={"_turn_request_id": "summary-request"},
    ))
    outbound = []
    while loop.bus.outbound_size:
        outbound.append(await loop.bus.consume_outbound())
    failures = [m for m in outbound if m.metadata.get("_error")]
    assert len(failures) == 1
    assert failures[0].content == reason
    assert failures[0].metadata["_turn_request_id"] == "summary-request"
    loop.provider.chat_with_retry.assert_not_awaited()
    loop.provider.chat_stream_with_retry.assert_not_awaited()
