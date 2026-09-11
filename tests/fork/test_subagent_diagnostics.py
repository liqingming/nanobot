"""真实 Manager + runner，模型及工具只用内存替身，所有文件位于 tmp_path。"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.fork.agent import subagent_diagnostics
from nanobot.fork.agent.execution_scope import current_execution_id
from nanobot.fork.agent.subagent_diagnostics import SubagentDiagnostics, session_ref
from nanobot.providers.base import LLMProvider, LLMResponse

_SECRET = "SECRET-body-arguments-api-key"


def _manager(tmp_path, *, native=False, block_limit=3000):
    provider = MagicMock(spec=LLMProvider)
    provider.supports_native_context_compaction = native
    provider.supports_request_context = True
    provider.aclose_execution = AsyncMock()
    manager = SubagentManager(
        provider, tmp_path / "workspace", MessageBus(), 16000, model="test-model",
        data_dir=tmp_path / "data", context_window_tokens=8000,
        context_block_limit=block_limit,
    )
    manager.workspace.mkdir()
    manager._build_tools = lambda **kwargs: ToolRegistry()
    manager._build_subagent_prompt = lambda **kwargs: "test system"
    return manager, provider


def _logs(tmp_path):
    return [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in sorted((tmp_path / "data" / "subagent-runtime").rglob("runtime.log"))
    ]


async def _run(manager, *, session="cli:topic", task_id="a" * 32):
    status = SubagentStatus(task_id, "label", _SECRET, 0)
    await manager._run_subagent(
        task_id, _SECRET, "label",
        {"channel": "cli", "chat_id": "topic", "session_key": session}, status,
        origin_message_id="parent-message",
    )
    return status


@pytest.mark.parametrize("native,measured", [(False, True), (False, False), (True, True), (True, False)])
async def test_real_manager_audits_governance_usage_and_native_counts(tmp_path, native, measured):
    manager, provider = _manager(tmp_path, native=native)
    diagnostics = {"authorization": _SECRET, "raw_payload": {"text": _SECRET}}
    if native:
        diagnostics.update({
            "transport": "codex_app_server", "context_management": "codex_native_auto",
            "native_compactions_started": 3, "native_compactions_completed": 2,
            "native_compaction_in_progress": True,
        })
        if measured:
            diagnostics["context_input_tokens"] = 321
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=_SECRET,
        usage={"prompt_tokens": 900, "completion_tokens": 10} if measured or native else {},
        provider_diagnostics=diagnostics,
    ))
    status = await _run(manager)
    records, = _logs(tmp_path)
    events = {row["event"]: row for row in records}
    assert status.stop_reason == "completed"
    assert records[0]["event"] == "subagent.run.start"
    assert records[-1]["event"] == "subagent.run.end"
    assert records[-1]["outcome"] == "completed"
    assert [row["sequence"] for row in records] == list(range(1, len(records) + 1))
    governance = events["subagent.runner.context.governance"]
    assert governance["budget"]["block_limit"] == 3000
    assert governance["budget"]["input_tokens"] == 8000 - 4096 - 1024
    assert governance["before"]["total"] >= governance["after"]["total"]
    assert governance["strategy"] == ("native" if native else "transactional")
    response = events["subagent.runner.model.response"]
    usage = response["usage"]
    assert usage["context_input_estimated"] == int(not measured)
    assert usage["context_input_tokens"] > 0
    if measured:
        assert usage["context_input_tokens"] == (321 if native else 900)
    assert records[-1]["usage"] == usage
    summary = events["subagent.runner.context.turn_summary"]
    assert summary["context_input_tokens"] == usage["context_input_tokens"]
    assert summary["context_input_estimated"] == usage["context_input_estimated"]
    if native:
        assert response["provider_diagnostics"]["native_compactions_completed"] == 2
        assert response["provider_diagnostics"]["native_compactions_started"] == 3
    assert _SECRET not in json.dumps(records)
    assert records[-1]["parent_session_ref"] == session_ref("cli:topic")
    assert records[-1]["task_id"] == "a" * 32
    request = provider.chat_with_retry.await_args.kwargs["request_context"]
    assert request["session_key"] == "cli:topic"
    assert request["turn_id"] == records[-1]["turn_id"]
    assert request["turn_id"].startswith("subagent:")
    execution_id = records[-1]["provider_execution_id"]
    assert execution_id and execution_id != records[-1]["run_id"]
    provider.aclose_execution.assert_awaited_once_with(execution_id)
    assert current_execution_id() is None
    announcement = await manager.bus.consume_inbound()
    assert announcement.session_key_override == "cli:topic"
    assert announcement.metadata["subagent_task_id"] == "a" * 32
    assert announcement.metadata["origin_message_id"] == "parent-message"
    assert not (tmp_path / "data" / "sessions").exists()


async def test_spawn_concurrency_keeps_audit_and_provider_identity_isolated(tmp_path):
    manager, provider = _manager(tmp_path, native=True)
    entered = asyncio.Event()
    calls = []

    async def respond(**kwargs):
        calls.append((current_execution_id(), kwargs["request_context"]))
        if len(calls) == 3:
            entered.set()
        await asyncio.wait_for(entered.wait(), timeout=5)
        return LLMResponse(content="ok", usage={"prompt_tokens": 100, "completion_tokens": 5})

    provider.chat_with_retry = AsyncMock(side_effect=respond)
    tasks = []
    for session in ("cli:same", "cli:same", "cli:other"):
        await manager.spawn("task", session_key=session, origin_message_id="parent")
        tasks.append(list(manager._running_tasks.values())[-1])
    await asyncio.gather(*tasks)
    logs = _logs(tmp_path)
    assert len(logs) == 3
    assert len({rows[-1]["run_id"] for rows in logs}) == 3
    assert len({rows[-1]["task_id"] for rows in logs}) == 3
    assert len({rows[-1]["provider_execution_id"] for rows in logs}) == 3
    assert sorted(rows[-1]["parent_session_ref"] for rows in logs) == sorted(
        session_ref(key) for key in ("cli:same", "cli:same", "cli:other")
    )
    for rows in logs:
        assert len({row["turn_id"] for row in rows}) == 1
        assert len({row["task_id"] for row in rows}) == 1
        assert rows[-1]["outcome"] == "completed"
    assert {call.args[0] for call in provider.aclose_execution.await_args_list} == {
        execution_id for execution_id, _ in calls
    }
    assert current_execution_id() is None


@pytest.mark.parametrize("failure", ["provider_error", "exception", "cancelled", "budget"])
async def test_terminal_audit_survives_non_success_paths(tmp_path, failure):
    manager, provider = _manager(tmp_path, block_limit=1 if failure == "budget" else 3000)
    if failure == "provider_error":
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content=_SECRET, finish_reason="error", error_status_code=429,
            error_kind="rate_limit", error_code="rate_limit_exceeded",
            error_type=_SECRET,
        ))
    else:
        error = asyncio.CancelledError() if failure == "cancelled" else RuntimeError(_SECRET)
        provider.chat_with_retry = AsyncMock(side_effect=error)
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await _run(manager)
    else:
        await _run(manager)
    records, = _logs(tmp_path)
    assert records[-1]["outcome"] == ("cancelled" if failure == "cancelled" else "error")
    assert _SECRET not in json.dumps(records)
    if failure == "provider_error":
        done = next(row for row in records if row["event"] == "subagent.runner.model.request.done")
        assert done["error_status_code"] == 429
        assert done["error_code"] == "rate_limit_exceeded"
    if failure == "budget":
        provider.chat_with_retry.assert_not_awaited()
        assert records[-1]["context_safety_failure"] is True
        assert any(row["event"] == "subagent.runner.context.budget_exhausted" for row in records)
    assert current_execution_id() is None


@pytest.mark.parametrize("failure", ["completed", "error", "cancelled"])
async def test_log_writer_failure_does_not_change_result_or_cleanup(tmp_path, monkeypatch, failure):
    manager, provider = _manager(tmp_path)
    attempts = []

    def broken_writer(*args, **kwargs):
        attempts.append(args)
        raise OSError(_SECRET)

    monkeypatch.setattr(subagent_diagnostics, "append_session_runtime_log", broken_writer)
    if failure == "cancelled":
        provider.chat_with_retry = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await _run(manager)
    else:
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="result", finish_reason="error" if failure == "error" else "stop",
        ))
        status = await _run(manager)
        assert status.stop_reason == failure
        announcement = await manager.bus.consume_inbound()
        assert "result" in announcement.content
    assert attempts[0][1] == "subagent.run.start"
    assert attempts[-1][1] == "subagent.run.end"
    provider.aclose_execution.assert_awaited_once()
    assert current_execution_id() is None


def test_allowlist_drops_nested_payloads_and_identity_spoofing(tmp_path):
    audit = SubagentDiagnostics(
        tmp_path / "data", task_id="../" + _SECRET, parent_session_key="../" + _SECRET,
        model=_SECRET, provider=object(),
    )
    audit("runner.tool.audit.start", {"arguments": {"password": _SECRET}})
    audit("runner.model.response", {
        "turn_id": _SECRET, "task_id": _SECRET, "parent_session_ref": _SECRET,
        "usage": {"prompt_tokens": 12, "context_input_tokens": _SECRET, _SECRET: 123},
        "finish_reason": _SECRET, "error_content": _SECRET,
        "provider_diagnostics": {
            "native_compactions_completed": 2, "native_compactions_started": _SECRET,
            "transport": _SECRET, "credentials": {"api_key": _SECRET},
        },
        "timeout_s": float("inf"),
    })
    records, = _logs(tmp_path)
    assert len(records) == 1
    assert _SECRET not in json.dumps(records)
    assert records[0]["turn_id"] == audit.turn_id
    assert records[0]["finish_reason"] == "other"
    assert records[0]["usage"] == {"prompt_tokens": 12}
    assert records[0]["provider_diagnostics"] == {
        "native_compactions_completed": 2, "transport": "other",
    }
    assert audit.path.is_relative_to(tmp_path / "data")
    assert _SECRET not in str(audit.path)


async def test_initialization_failure_still_has_terminal_audit(tmp_path):
    manager, provider = _manager(tmp_path)

    def fail(**kwargs):
        raise RuntimeError(_SECRET)

    manager._build_tools = fail
    await _run(manager, session=None)
    rows, = _logs(tmp_path)
    assert [row["event"] for row in rows] == ["subagent.run.start", "subagent.run.end"]
    assert rows[-1]["outcome"] == "error"
    assert rows[-1]["parent_session_ref"] == session_ref("cli:topic")
    assert _SECRET not in json.dumps(rows)
    provider.chat_with_retry.assert_not_called()


async def test_same_task_reexecution_has_new_audit_identity_and_safe_retry(tmp_path):
    manager, provider = _manager(tmp_path)

    async def respond(**kwargs):
        kwargs["on_retry_event"]({
            "attempt": 1, "retry_mode": "standard", "retry_after_s": 2,
            "error_kind": "rate_limit", "error_summary": _SECRET,
        })
        return LLMResponse(content="ok")

    provider.chat_with_retry = AsyncMock(side_effect=respond)
    await _run(manager)
    await _run(manager)
    logs = _logs(tmp_path)
    assert len(logs) == 2
    assert len({rows[-1]["task_id"] for rows in logs}) == 1
    assert len({rows[-1]["run_id"] for rows in logs}) == 2
    for rows in logs:
        retry = next(row for row in rows if row["event"] == "subagent.runner.model.retry")
        assert retry["attempt"] == 1
        assert retry["retry_after_s"] == 2
        assert retry["retry_mode"] == "standard"
    assert _SECRET not in json.dumps(logs)
