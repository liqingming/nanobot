from unittest.mock import AsyncMock, MagicMock

import pytest

import nanobot.agent.memory as memory_module
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path, *, estimated_tokens: int, context_window_tokens: int) -> AgentLoop:
    from nanobot.providers.base import GenerationSettings
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
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
async def test_prompt_below_threshold_does_not_consolidate(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="有效续接摘要")  # type: ignore[method-assign]

    await loop.process_direct("hello", session_key="cli:test")

    loop.consolidator.archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_above_threshold_triggers_consolidation(tmp_path, monkeypatch) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=1000, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="有效续接摘要")  # type: ignore[method-assign]
    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
    ]
    loop.sessions.save(session)
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _message: 500)

    await loop.process_direct("hello", session_key="cli:test")

    assert loop.consolidator.archive.await_count >= 1


@pytest.mark.asyncio
async def test_prompt_above_threshold_archives_until_next_user_boundary(tmp_path, monkeypatch) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=1000, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="有效续接摘要")  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
        {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
    ]
    loop.sessions.save(session)

    token_map = {"u1": 120, "a1": 120, "u2": 120, "a2": 120, "u3": 120}
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda message: token_map[message["content"]])

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    archived_chunk = loop.consolidator.archive.await_args.args[0]
    assert [message["content"] for message in archived_chunk] == ["u1", "a1", "u2", "a2"]
    assert session.last_consolidated == 4


@pytest.mark.asyncio
async def test_consolidation_loops_until_target_met(tmp_path, monkeypatch) -> None:
    """Verify maybe_consolidate_by_tokens keeps looping until under threshold."""
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="有效续接摘要")  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
        {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
        {"role": "assistant", "content": "a3", "timestamp": "2026-01-01T00:00:05"},
        {"role": "user", "content": "u4", "timestamp": "2026-01-01T00:00:06"},
    ]
    loop.sessions.save(session)

    call_count = [0]
    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500, "test")
        if call_count[0] == 2:
            return (300, "test")
        return (80, "test")

    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 100)

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    assert loop.consolidator.archive.await_count == 2
    assert session.last_consolidated == 6


@pytest.mark.asyncio
async def test_consolidation_continues_below_trigger_until_half_target(tmp_path, monkeypatch) -> None:
    """Once triggered, consolidation should continue until it drops below half threshold."""
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="有效续接摘要")  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
        {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
        {"role": "assistant", "content": "a3", "timestamp": "2026-01-01T00:00:05"},
        {"role": "user", "content": "u4", "timestamp": "2026-01-01T00:00:06"},
    ]
    loop.sessions.save(session)

    call_count = [0]

    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500, "test")
        if call_count[0] == 2:
            return (150, "test")
        return (80, "test")

    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 100)

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    assert loop.consolidator.archive.await_count == 2
    assert session.last_consolidated == 6


@pytest.mark.asyncio
async def test_consolidation_persists_summary_for_next_prepare_session(tmp_path, monkeypatch) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="User discussed project status.")  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
    ]
    loop.sessions.save(session)

    call_count = [0]

    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500, "test")
        return (80, "test")

    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 150)

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    reloaded = loop.sessions.get_or_create("cli:test")
    meta = reloaded.metadata.get("_last_summary")
    assert meta is not None
    assert meta["text"] == "User discussed project status."

    reloaded, pending = loop.auto_compact.prepare_session(reloaded, "cli:test")
    assert pending is not None
    assert "User discussed project status." in pending
    # _last_summary persists for restart survival.
    assert "_last_summary" in reloaded.metadata


@pytest.mark.asyncio
async def test_preflight_consolidation_receives_pending_summary(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=200)
    session = loop.sessions.get_or_create("cli:test")
    loop.auto_compact.prepare_session = MagicMock(
        return_value=(session, "Previous conversation summary: earlier context")
    )  # type: ignore[method-assign]
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)  # type: ignore[method-assign]
    loop._schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]

    await loop.process_direct("hello", session_key="cli:test")

    loop.consolidator.maybe_consolidate_by_tokens.assert_any_await(
        session,
        replay_max_messages=loop._max_messages,
        completed_goal=False,
    )


@pytest.mark.asyncio
async def test_preflight_consolidation_before_llm_call(tmp_path, monkeypatch) -> None:
    """Verify preflight consolidation runs before the LLM call in process_direct."""
    order: list[str] = []

    # 集成路径需要给 runner 的 1024 token 安全预留留出真实空间。
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=2000)

    archived_session_keys: list[str | None] = []

    async def track_consolidate(messages, *, session_key=None, prior_continuation=None):
        order.append("consolidate")
        archived_session_keys.append(session_key)
        return "有效续接摘要"
    loop.consolidator.archive = track_consolidate  # type: ignore[method-assign]

    async def track_llm(*args, **kwargs):
        order.append("llm")
        return LLMResponse(content="ok", tool_calls=[])
    loop.provider.chat_with_retry = track_llm
    loop.provider.chat_stream_with_retry = track_llm
    # 本测试验证调用顺序，提供与归档后估算一致的完整请求计数。
    loop.provider.estimate_prompt_tokens = lambda *_args, **_kwargs: (80, "test")
    loop._schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
    ]
    loop.sessions.save(session)
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 1500)

    call_count = [0]
    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        return (3000 if call_count[0] <= 1 else 80, "test")
    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]

    await loop.process_direct("hello", session_key="cli:test")

    assert "consolidate" in order
    assert "llm" in order
    assert order.index("consolidate") < order.index("llm")
    assert archived_session_keys == ["cli:test"]


@pytest.mark.asyncio
async def test_failed_preflight_stops_before_history_clipping_or_model(tmp_path) -> None:
    from nanobot.fork.agent.summary_transaction import SummaryTransactionError

    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=10000)
    loop._schedule_background = lambda coro: coro.close()
    session = loop.sessions.get_or_create("cli:test")
    for i in range(3):
        session.add_message("user", f"约束 {i}：禁止提交")
        session.add_message("assistant", "尚未验收")
    session.metadata["_completed_goal_needs_compaction"] = True
    loop.sessions.save(session)
    loop.consolidator.archive = AsyncMock(return_value=None)
    loop.consolidator.estimate_session_prompt_tokens = lambda _: (12000, "test")
    loop.consolidator.pick_consolidation_boundary = lambda *_: (2, 100)
    with pytest.raises(SummaryTransactionError, match="本次请求停止"):
        await loop.process_direct("继续", session_key=session.key)
    assert session.last_consolidated == 0
    assert session.metadata["_completed_goal_needs_compaction"] is True
    loop.provider.chat_with_retry.assert_not_awaited()
    loop.provider.chat_stream_with_retry.assert_not_awaited()


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
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(
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
