"""摘要覆盖必须经完整输入、有效输出及单次持久化提交。"""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore
from nanobot.fork.agent.summary_transaction import (
    COVERAGE_KEY,
    SummarySnapshot,
    SummaryTransactionError,
    commit_summary,
)
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.session.manager import SessionManager


@pytest.fixture
def setup(tmp_path):
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:summary-test")
    for i in range(3):
        session.add_message("user", f"需求 {i}：禁止修改数据")
        session.add_message("assistant", f"待验收 {i}")
    session.metadata["authorization"] = {"source": "user", "forbidden": "修改数据"}
    session.todos = [{"content": "验收", "status": "pending"}]
    sessions.save(session)
    provider = MagicMock()
    provider.generation = GenerationSettings(max_tokens=1000)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="<continuation>禁止修改数据；尚待验收。</continuation>"
                "<memory-candidates>(nothing)</memory-candidates>",
    ))
    consolidator = Consolidator(
        store=MemoryStore(tmp_path), provider=provider, model="test", sessions=sessions,
        context_window_tokens=10000, max_completion_tokens=1000,
        build_messages=lambda **_: [], get_tool_definitions=lambda: [],
    )
    consolidator.estimate_session_prompt_tokens = lambda current: (
        (7000 if current.last_consolidated == 0 else 100), "test"
    )
    consolidator.pick_consolidation_boundary = lambda *_: (2, 100)
    return consolidator, sessions, session, provider


async def run_path(consolidator, session, mode):
    if mode == "idle":
        return await consolidator.compact_idle_session(session.key, max_suffix=2)
    if mode == "replay":
        return await consolidator.maybe_consolidate_by_tokens(session, replay_max_messages=4)
    return await consolidator.maybe_consolidate_by_tokens(session)


@pytest.mark.parametrize("mode", ["tokens", "replay", "idle"])
async def test_failure_preserves_disk_and_live_history(setup, mode):
    consolidator, sessions, session, provider = setup
    before = deepcopy(session.messages)
    metadata = deepcopy(session.metadata)
    provider.chat_with_retry.side_effect = RuntimeError("offline")
    if mode == "idle":
        assert await run_path(consolidator, session, mode) is None
    else:
        with pytest.raises(SummaryTransactionError):
            await run_path(consolidator, session, mode)
    sessions.invalidate(session.key)
    reloaded = sessions.get_or_create(session.key)
    assert reloaded.messages == before
    assert reloaded.metadata == metadata
    assert reloaded.last_consolidated == 0
    assert provider.chat_with_retry.await_count == 1
    assert "[RAW]" in consolidator.store.read_unprocessed_history(0)[0]["content"]


@pytest.mark.parametrize("mode", ["tokens", "replay", "idle"])
async def test_success_commits_summary_cursor_and_version_together(setup, mode):
    consolidator, sessions, session, provider = setup
    await run_path(consolidator, session, mode)
    sessions.invalidate(session.key)
    reloaded = sessions.get_or_create(session.key)
    record = reloaded.metadata[COVERAGE_KEY]
    assert record["version"] == 1
    assert record["cursor"] == reloaded.last_consolidated
    assert record["covered_message_count"] == (4 if mode == "idle" else 2)
    assert reloaded.metadata["_continuation_summary"] == reloaded.metadata["_last_summary"]
    assert reloaded.metadata["authorization"] == session.metadata["authorization"]
    assert reloaded.todos == session.todos
    kwargs = provider.chat_with_retry.await_args.kwargs
    assert kwargs["tools"] is None
    assert kwargs["max_tokens"] == 1000


@pytest.mark.parametrize("mode", ["tokens", "replay", "idle"])
async def test_cancel_does_not_commit(setup, mode):
    consolidator, sessions, session, provider = setup
    before = deepcopy(session.messages)
    provider.chat_with_retry.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run_path(consolidator, session, mode)
    sessions.invalidate(session.key)
    reloaded = sessions.get_or_create(session.key)
    assert reloaded.messages == before
    assert reloaded.last_consolidated == 0
    assert COVERAGE_KEY not in reloaded.metadata


@pytest.mark.parametrize("change", ["messages", "authorization", "todos"])
async def test_inflight_change_rejects_stale_summary(setup, change):
    consolidator, sessions, session, provider = setup

    async def changed(**_):
        if change == "messages":
            session.add_message("user", "停止修改")
        elif change == "authorization":
            session.metadata["authorization"]["forbidden"] = "任何写入"
        else:
            session.todos.append({"content": "额外验证", "status": "pending"})
        return LLMResponse(content="旧摘要")

    provider.chat_with_retry.side_effect = changed
    with pytest.raises(SummaryTransactionError, match="会话发生变化"):
        await consolidator.maybe_consolidate_by_tokens(session)
    assert session.last_consolidated == 0
    assert COVERAGE_KEY not in session.metadata


def test_save_failure_rolls_back_memory_and_disk(setup, monkeypatch):
    _, sessions, session, _ = setup
    snapshot = SummarySnapshot.capture(session)
    before = deepcopy(session.metadata)

    def fail(*_, **__):
        raise OSError("replace denied")

    monkeypatch.setattr("nanobot.session.manager.replace_file_with_retry", fail)
    with pytest.raises(OSError, match="replace denied"):
        commit_summary(
            sessions, session, snapshot, "禁止修改数据，待验收",
            covered_messages=session.messages[:2], end_cursor=2, reason="test",
        )
    assert session.metadata == before
    assert session.last_consolidated == 0
    sessions.invalidate(session.key)
    assert sessions.get_or_create(session.key).metadata == before


@pytest.mark.parametrize("response", [
    LLMResponse(content=""),
    LLMResponse(content="(nothing)"),
    LLMResponse(content="<continuation>残缺"),
    LLMResponse(content="<memory-candidates>记忆</memory-candidates>"),
    LLMResponse(content="摘要", finish_reason="length"),
    LLMResponse(content="摘要", finish_reason="error"),
    LLMResponse(content="摘" * 8001),
])
async def test_invalid_output_does_not_cover_history(setup, response):
    consolidator, _, session, provider = setup
    provider.chat_with_retry.return_value = response
    with pytest.raises(SummaryTransactionError):
        await consolidator.maybe_consolidate_by_tokens(session)
    assert session.last_consolidated == 0
    assert COVERAGE_KEY not in session.metadata


async def test_full_request_budget_rejects_without_truncation(setup):
    consolidator, _, _, provider = setup
    provider.estimate_prompt_tokens.return_value = (9000, "test")
    assert await consolidator.archive(
        [{"role": "user", "content": "结尾约束不可丢"}], prior_continuation="旧约束" * 4000,
    ) is None
    provider.chat_with_retry.assert_not_awaited()


async def test_prior_summary_is_not_truncated(setup):
    consolidator, _, _, provider = setup
    provider.estimate_prompt_tokens.return_value = (4000, "test")
    prior = "旧约束" * 4000 + "结尾禁止推送"
    assert await consolidator.archive(
        [{"role": "user", "content": "继续"}], prior_continuation=prior,
    )
    assert prior in provider.chat_with_retry.await_args.kwargs["messages"][1]["content"]


async def test_each_round_sees_committed_prior_summary(setup):
    consolidator, _, session, provider = setup
    consolidator.estimate_session_prompt_tokens = lambda current: (
        (7000 if current.last_consolidated < 4 else 100), "test"
    )
    consolidator.pick_consolidation_boundary = lambda current, _: (
        current.last_consolidated + 2, 100,
    )
    provider.chat_with_retry.side_effect = [
        LLMResponse(content="第一轮禁止写入"),
        LLMResponse(content="第二轮禁止写入，尚未验收"),
    ]
    await consolidator.maybe_consolidate_by_tokens(session)
    assert session.last_consolidated == 4
    assert session.metadata[COVERAGE_KEY]["version"] == 2
    assert "第一轮禁止写入" in provider.chat_with_retry.await_args.kwargs["messages"][1]["content"]


async def test_later_failure_keeps_previous_committed_version(setup):
    consolidator, _, session, provider = setup
    consolidator.estimate_session_prompt_tokens = lambda _: (7000, "test")
    consolidator.pick_consolidation_boundary = lambda current, _: (
        current.last_consolidated + 2, 100,
    )
    provider.chat_with_retry.side_effect = [LLMResponse(content="首轮摘要"), RuntimeError("offline")]
    with pytest.raises(SummaryTransactionError):
        await consolidator.maybe_consolidate_by_tokens(session)
    assert session.last_consolidated == 2
    assert session.metadata[COVERAGE_KEY]["version"] == 1
    assert session.metadata["_continuation_summary"]["text"] == "首轮摘要"


def test_error_after_replace_does_not_restore_stale_memory(setup, monkeypatch):
    _, sessions, session, _ = setup
    snapshot = SummarySnapshot.capture(session)
    original_save = sessions.save

    def committed_then_failed(current, **kwargs):
        original_save(current, **kwargs)
        raise OSError("post-commit failure")

    monkeypatch.setattr(sessions, "save", committed_then_failed)
    with pytest.raises(OSError, match="post-commit"):
        commit_summary(
            sessions, session, snapshot, "已提交的摘要",
            covered_messages=session.messages[:2], end_cursor=2, reason="test",
        )
    assert session.last_consolidated == 2
    reloaded = sessions.get_or_create(session.key)
    assert reloaded.metadata == session.metadata
    assert reloaded.last_consolidated == 2


async def test_replaced_session_reference_rejects_stale_commit(setup):
    consolidator, sessions, session, provider = setup

    async def replaced(**_):
        sessions.invalidate(session.key)
        fresh = sessions.get_or_create(session.key)
        fresh.add_message("user", "新的授权边界：只读")
        sessions.save(fresh)
        return LLMResponse(content="过期摘要")

    provider.chat_with_retry.side_effect = replaced
    with pytest.raises(SummaryTransactionError, match="会话发生变化"):
        await consolidator.maybe_consolidate_by_tokens(session)
    fresh = sessions.get_or_create(session.key)
    assert fresh.messages[-1]["content"] == "新的授权边界：只读"
    assert fresh.last_consolidated == 0
    assert COVERAGE_KEY not in fresh.metadata


@pytest.mark.parametrize("retained", [None, []])
def test_mismatched_coverage_cannot_remove_unseen_messages(setup, retained):
    _, sessions, session, _ = setup
    with pytest.raises(SummaryTransactionError, match="覆盖与实际移除消息不一致"):
        commit_summary(
            sessions, session, SummarySnapshot.capture(session), "不完整摘要",
            covered_messages=session.messages[:1],
            end_cursor=2 if retained is None else len(session.messages),
            retained_messages=retained, reason="test",
        )
    assert session.last_consolidated == 0
    assert COVERAGE_KEY not in session.metadata


@pytest.mark.parametrize("mode", ["tokens", "replay", "idle", "background", "completed_goal"])
async def test_native_provider_keeps_history_without_ordinary_summary(setup, mode):
    consolidator, sessions, session, provider = setup
    provider.supports_native_context_compaction = True
    session.metadata["_completed_goal_needs_compaction"] = True
    sessions.save(session)
    before = deepcopy(session)
    history_path = sessions._get_session_path(session.key)
    persisted = history_path.read_bytes()
    if mode in {"background", "completed_goal"}:
        await consolidator.maybe_consolidate_by_tokens(session, **{mode: True})
    else:
        await run_path(consolidator, session, mode)
    assert session == before
    assert history_path.read_bytes() == persisted
    assert consolidator.store.read_unprocessed_history(0) == []
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.parametrize(("failure", "stage", "reason"), [
    ("provider", "request", "request_failed"),
    ("timeout", "request", "request_timeout"),
    ("budget", "input_budget", "input_budget_exceeded"),
    ("length", "response_validation", "response_incomplete"),
    ("provider_response", "response_validation", "response_incomplete"),
    ("missing_section", "summary_validation", "continuation_missing"),
    ("empty", "summary_validation", "summary_invalid"),
    ("oversized", "summary_validation", "summary_invalid"),
])
async def test_failure_diagnostic_locates_stage_without_payloads(setup, failure, stage, reason):
    consolidator, sessions, session, provider = setup
    before = SummarySnapshot.capture(session)
    persisted = sessions._get_session_path(session.key).read_bytes()
    secret = "SECRET-PROVIDER-PAYLOAD"
    if failure == "provider":
        provider.chat_with_retry.side_effect = RuntimeError(secret)
    elif failure == "timeout":
        provider.chat_with_retry.side_effect = TimeoutError(secret)
    elif failure == "budget":
        provider.estimate_prompt_tokens.return_value = (9000, "test")
    else:
        provider.chat_with_retry.return_value = {
            "length": LLMResponse(content=secret, finish_reason="length"),
            "provider_response": LLMResponse(
                content=secret, finish_reason="error", error_status_code=429,
                error_code="rate_limit_exceeded",
            ),
            "missing_section": LLMResponse(content="<continuation>" + secret),
            "empty": LLMResponse(content=""),
            "oversized": LLMResponse(content=secret * 1000),
        }[failure]
    with pytest.raises(SummaryTransactionError):
        await consolidator.maybe_consolidate_by_tokens(session)
    path = sessions.get_session_runtime_log_path(session.key)
    raw = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "context.summary.failed"
    assert row["stage"] == stage
    assert row["reason"] == reason
    assert row["session_key"] == session.key
    assert row["model"] == "test"
    assert row["estimated_tokens"] > 0
    assert row["input_budget"] == 7976
    assert secret not in raw
    assert session.messages[0]["content"] not in raw
    if failure == "provider_response":
        assert row["error_status_code"] == 429
        assert row["error_code"] == "rate_limit_exceeded"
    if failure == "timeout":
        assert row["exception_type"] == "TimeoutError"
    assert before.matches(session)
    assert sessions._get_session_path(session.key).read_bytes() == persisted
    assert provider.chat_with_retry.await_count == (0 if failure == "budget" else 1)


async def test_native_summary_refusal_is_diagnosed_without_executing_provider(setup):
    consolidator, sessions, session, provider = setup
    provider.supports_native_context_compaction = True
    # 直接验证摘要隔离门禁，不绕过普通入口已有的原生 Provider 分工。
    assert await consolidator.archive(session.messages, session_key=session.key) is None
    row = json.loads(sessions.get_session_runtime_log_path(session.key).read_text(encoding="utf-8"))
    assert row["reason"] == "native_summary_unsupported"
    provider.chat_with_retry.assert_not_awaited()
    assert session.last_consolidated == 0


async def test_candidate_and_raw_archive_failures_are_distinguishable(setup, monkeypatch):
    consolidator, sessions, session, _ = setup
    before = SummarySnapshot.capture(session)
    persisted = sessions._get_session_path(session.key).read_bytes()
    monkeypatch.setattr(consolidator.store, "append_history", MagicMock(side_effect=OSError("disk")))
    with pytest.raises(OSError):
        await consolidator.maybe_consolidate_by_tokens(session)
    rows = [json.loads(line) for line in sessions.get_session_runtime_log_path(session.key)
            .read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == ["candidate_archive", "raw_archive"]
    assert [row["exception_type"] for row in rows] == ["OSError", "OSError"]
    assert before.matches(session)
    assert sessions._get_session_path(session.key).read_bytes() == persisted


async def test_diagnostic_storage_failure_does_not_change_failure_semantics(setup, monkeypatch):
    consolidator, sessions, session, provider = setup
    before = SummarySnapshot.capture(session)
    provider.chat_with_retry.side_effect = RuntimeError("offline")
    monkeypatch.setattr(sessions, "get_session_runtime_log_path", MagicMock(side_effect=OSError))
    with pytest.raises(SummaryTransactionError):
        await consolidator.maybe_consolidate_by_tokens(session)
    assert before.matches(session)
    assert "[RAW]" in consolidator.store.read_unprocessed_history(0)[0]["content"]


async def test_cancel_does_not_log_as_failure_or_raw_archive(setup, monkeypatch):
    consolidator, sessions, session, provider = setup
    before = SummarySnapshot.capture(session)
    provider.chat_with_retry.side_effect = asyncio.CancelledError()
    raw_archive = MagicMock()
    monkeypatch.setattr(consolidator.store, "raw_archive", raw_archive)
    with pytest.raises(asyncio.CancelledError):
        await consolidator.maybe_consolidate_by_tokens(session)
    assert before.matches(session)
    assert not sessions.get_session_runtime_log_path(session.key).exists()
    raw_archive.assert_not_called()


async def test_real_summary_timeout_cancels_request_without_committing(setup, monkeypatch):
    from nanobot.fork.agent.transactional_context import request_summary

    consolidator, sessions, session, provider = setup
    before = SummarySnapshot.capture(session)
    cancelled = asyncio.Event()

    async def blocked(**_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def short_timeout(*args, **kwargs):
        return await request_summary(*args, **kwargs, timeout=0.01)

    provider.chat_with_retry.side_effect = blocked
    monkeypatch.setattr("nanobot.agent.memory.request_summary", short_timeout)
    with pytest.raises(SummaryTransactionError):
        await consolidator.maybe_consolidate_by_tokens(session)
    assert cancelled.is_set()
    assert before.matches(session)
    row = json.loads(sessions.get_session_runtime_log_path(session.key).read_text(encoding="utf-8"))
    assert row["reason"] == "request_timeout"
    assert provider.chat_with_retry.await_count == 1
