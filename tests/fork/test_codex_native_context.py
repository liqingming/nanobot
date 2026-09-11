"""原生自动压缩：真实 stdio 模拟验证同线程续传、预算及失败安全。"""

import asyncio
import json
import sys
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_governance import ContextGovernanceConfig, ContextGovernor
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.fork.agent.native_context import NativeContextPreparation, uses_native_context
from nanobot.fork.providers.codex_app_server_provider import CodexAppServerProvider
from nanobot.fork.providers.codex_native_context import (
    NativeCompactionState,
    native_input_budget,
)
from nanobot.providers.base import LLMProvider, LLMResponse

_SERVER = r'''
import json, sys
from pathlib import Path
capture, mode = sys.argv[1:]
native_effects = not mode.startswith("plain:")
mode = mode.removeprefix("plain:")
def send(value):
    if value.get("method", "").startswith(("item/", "turn/")) or value.get("method") == "thread/tokenUsage/updated":
        value.setdefault("params", {}).setdefault("threadId", "thread")
        value["params"].setdefault("turnId", "turn")
    print(json.dumps(value), flush=True)
def event(method, **params):
    send({"method": method, "params": {"threadId": "thread", "turnId": "turn", **params}})
def call(n):
    send({"id": 100 + n, "method": "item/tool/call", "params": {
        "callId": "call-" + str(n), "tool": "read_file", "arguments": {"path": str(n)}
    }})
for line in sys.stdin:
    msg = json.loads(line)
    with Path(capture).open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")
    method = msg.get("method")
    if method == "initialize":
        send({"id": msg["id"], "result": {"userAgent": "codex-cli/0.153.4"}})
    elif method == "skills/list":
        send({"id": msg["id"], "result": {"data": []}})
    elif method == "thread/start":
        if mode == "unsupported":
            send({"id": msg["id"], "error": {"message": "unsupported compaction config"}})
        else:
            send({"id": msg["id"], "result": {"thread": {"id": "thread"}}})
    elif method == "turn/start":
        send({"id": msg["id"], "result": {"turn": {"id": "turn"}}})
        if native_effects and (mode in {"native_fail", "native_safe"} or mode.startswith("steer")):
            event("item/completed", item={"type": "commandExecution", "id": "cmd", "status": "completed"})
        if native_effects and mode in {"file_fail", "native_safe"}:
            event("item/completed", item={"type": "fileChange", "id": "file", "status": "completed"})
        call(1)
    elif method == "turn/steer":
        assert msg["params"]["threadId"] == "thread"
        assert msg["params"]["expectedTurnId"] == "turn"
        if mode == "steer_reject":
            send({"id": msg["id"], "error": {"message": "steer unavailable"}})
        elif mode == "steer_disconnect":
            sys.exit(2)
        else:
            send({"id": msg["id"], "result": {
                "turnId": "wrong" if mode == "steer_mismatch" else "turn",
            }})
    elif "result" in msg:
        n = msg["id"] - 100
        if mode in {"native_fail", "file_fail"} or (mode == "steer_later_disconnect" and n == 3):
            sys.exit(2)
        if n in (2, 5):
            event("thread/tokenUsage/updated", tokenUsage={
                "total": {"inputTokens": 90000, "outputTokens": 10},
                "last": {"inputTokens": 70000}})
            event("item/started", item={"type": "contextCompaction", "id": "compact-" + str(n)})
            if mode == "compact_fail":
                sys.exit(2)
            event("item/completed", item={"type": "contextCompaction", "id": "compact-" + str(n)})
            if mode != "no_usage":
                event("thread/tokenUsage/updated", tokenUsage={
                    "total": {"inputTokens": 100000 * n, "outputTokens": 20},
                    "last": {"inputTokens": 1234}})
        if n < 7:
            call(n + 1)
        else:
            event("item/completed", item={"type": "agentMessage", "id": "final", "text": "done"})
            event("turn/completed", turn={"id": "turn", "status": "completed"})
'''


def _provider(tmp_path, mode="normal"):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    capture = tmp_path / "rpc.jsonl"
    provider._app_server_command = [sys.executable, "-u", "-c", _SERVER, str(capture), mode]
    return provider, capture


def _config(provider, tmp_path, budget=8000):
    tools = MagicMock()
    tools.get_definitions.return_value = []
    return ContextGovernanceConfig(
        provider=provider, model="test", tools=tools, workspace=tmp_path, data_dir=tmp_path,
        session_key="test", max_tool_result_chars=16000, context_window_tokens=100000,
        context_block_limit=budget,
    )


def test_capability_only_applies_to_app_server(tmp_path):
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    assert not uses_native_context(MagicMock())
    assert not uses_native_context(MagicMock(spec=LLMProvider))
    assert not uses_native_context(OpenAICodexProvider())
    assert uses_native_context(_provider(tmp_path)[0])


def test_native_budget_cannot_enlarge_provider_budget():
    provider = MagicMock(spec=LLMProvider)
    assert native_input_budget(provider, None) == 0
    assert native_input_budget(provider, {"native_context": {"context_window_tokens": True}}) == 0
    budget = native_input_budget(provider, {"native_context": {
        "context_window_tokens": 8000, "context_block_limit": 100000, "max_tokens": 2000,
    }})
    assert budget == 4976
    assert NativeCompactionState(budget=budget).config()["model_auto_compact_token_limit"] == 3980


def test_native_prefix_is_stable_but_real_edits_still_visible(tmp_path):
    provider, _ = _provider(tmp_path)
    config = _config(provider, tmp_path)
    preparation = NativeContextPreparation(ContextGovernor())
    messages = [{"role": "user", "content": "task"}]
    first = preparation.prepare_for_model(config, messages, set())
    messages.extend([
        {"role": "assistant", "tool_calls": [
            {"id": "read", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "name": "read_file", "tool_call_id": "read", "content": "evidence"},
    ])
    second = preparation.prepare_for_model(config, messages, set())
    assert second[:len(first)] == first
    # 调用者修改输出副本，不能污染冻结检查点。
    second[0]["content"] = "accidental mutation"
    third = preparation.prepare_for_model(config, messages, set())
    assert third[0]["content"] == "task"
    messages[0]["content"] = "new requirement"
    fourth = preparation.prepare_for_model(config, messages, set())
    assert fourth[0]["content"] == "new requirement"


def test_compaction_events_are_correlated_and_deduplicated():
    state = NativeCompactionState(budget=8000, thread_id="mine")
    params = {"threadId": "other", "item": {"id": "c", "type": "contextCompaction"}}
    assert not state.observe("item/started", params)
    params["threadId"] = "mine"
    assert state.observe("item/started", params)
    assert state.in_progress
    for _ in range(2):
        assert state.observe("item/completed", params)
    assert not state.in_progress
    assert state.diagnostics()["native_compactions_completed"] == 1


@pytest.mark.parametrize("mode", ["normal", "no_usage", "native_safe"])
async def test_native_compaction_keeps_one_thread_and_all_receipts(tmp_path, mode):
    provider, capture = _provider(tmp_path, mode)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    receipt = "attemptId=47 sha256=123 receipt=result.json " + "evidence " * 1000
    tools.execute = AsyncMock(return_value=receipt)
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "task"}],
        tools=tools, model="test", max_iterations=10, max_tool_result_chars=16000,
        context_window_tokens=100000, context_block_limit=80000,
        session_key="topic", turn_id="turn", workspace=tmp_path, data_dir=tmp_path,
    )
    diagnostics = []
    original_chat = provider.chat_with_retry

    async def chat(**kwargs):
        response = await original_chat(**kwargs)
        diagnostics.append(deepcopy(response.provider_diagnostics))
        return response

    provider.chat_with_retry = chat
    try:
        result = await AgentRunner(provider).run(spec)
        assert result.final_content == "done"
        assert tools.execute.await_count == 7
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        starts = [m for m in rpc if m.get("method") == "thread/start"]
        assert len(starts) == 1
        assert starts[0]["params"]["config"]["model_auto_compact_token_limit"] == 64000
        assert starts[0]["params"]["config"]["model_auto_compact_token_limit_scope"] == "total"
        assert all(m.get("method") != "thread/compact/start" for m in rpc)
        receipts = [m for m in rpc if "result" in m and m["id"] >= 101]
        assert len(receipts) == 7
        assert all(receipt in json.dumps(m) for m in receipts)
        assert diagnostics[-1]["native_compactions_completed"] == 2
        assert not any(d.get("context_rebased") for d in diagnostics)
        assert result.usage["context_input_estimated"] == (mode == "no_usage")
        if mode != "no_usage":
            assert result.usage["context_input_tokens"] == 1234
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()


@pytest.mark.parametrize("mode", ["compact_fail", "native_fail", "file_fail", "unsupported"])
async def test_native_failure_never_replays_side_effects(tmp_path, mode):
    provider, capture = _provider(tmp_path, mode)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="receipt")
    try:
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "task"}],
            tools=tools, model="test", max_iterations=10, max_tool_result_chars=16000,
            context_window_tokens=100000, context_block_limit=80000,
            session_key="topic", turn_id="turn", workspace=tmp_path, data_dir=tmp_path,
        ))
        assert result.stop_reason == "error"
        assert tools.execute.await_count == {
            "compact_fail": 2, "native_fail": 1, "file_fail": 1, "unsupported": 0,
        }[mode]
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert sum(m.get("method") == "thread/start" for m in rpc) == 1
        assert provider._turns == provider._turn_locks == {}
        if mode != "unsupported":
            assert list((tmp_path / "ledger").glob("*.json"))
    finally:
        await provider.aclose()


async def test_initial_oversize_fails_before_launch(tmp_path):
    provider, capture = _provider(tmp_path)
    try:
        result = await provider.chat(
            messages=[{"role": "system", "content": "huge " * 10000}],
            model="test", request_context={"native_context": {
                "context_window_tokens": 8000, "context_block_limit": 1000,
            }},
        )
        assert result.finish_reason == "error"
        assert not result.error_should_retry
        assert not capture.exists()
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()


async def test_native_overflow_does_not_trigger_runner_local_rebuild():
    provider = MagicMock(spec=LLMProvider)
    provider.supports_native_context_compaction = True
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="context length exceeded", finish_reason="error",
        error_code="context_length_exceeded", error_should_retry=False,
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "task"}],
        tools=tools, model="test", max_iterations=2, max_tool_result_chars=16000,
        context_window_tokens=100000,
    ))
    assert provider.chat_with_retry.await_count == 1


def test_large_new_receipt_is_saved_before_compaction(tmp_path):
    from pathlib import Path

    from nanobot.agent.context_artifacts import ToolDigestBuilder

    provider, _ = _provider(tmp_path)
    config = _config(provider, tmp_path, budget=10000)
    # 工具定义本身再大，也不应作为续传结果被重新计费、迫使小回执压缩。
    preparation = NativeContextPreparation(ContextGovernor())
    messages = [{"role": "user", "content": "task"}]
    preparation.prepare_for_model(config, messages, set())
    raw = "attemptId=77 full snapshot " * 10000
    digest, _ = ToolDigestBuilder.build(
        tool_call_id="large", tool_name="read_file", arguments={"path": "source"},
        result=raw,
    )
    messages.extend([
        {"role": "assistant", "tool_calls": [{"id": "large", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "name": "read_file", "tool_call_id": "large", "content": raw},
    ])
    result = preparation.prepare_for_model(config, messages, set(), tool_digests={"large": digest})
    compacted = result[-1]["content"]
    locator = ContextGovernor.persisted_result_locator(compacted)
    assert locator
    assert Path(locator).read_text(encoding="utf-8") == raw
    assert "ToolDigest" in compacted
    assert len(compacted) < 2000
    assert messages[-1]["content"] == raw
    assert preparation.prepare_for_model(config, messages, set()) == result


def test_native_append_does_not_recount_known_tool_definitions(tmp_path):
    provider, _ = _provider(tmp_path)
    config = _config(provider, tmp_path, budget=10000)
    config.tools.get_definitions.return_value = [{
        "type": "function", "function": {"name": "read_file", "description": "known " * 3000},
    }]
    preparation = NativeContextPreparation(ContextGovernor())
    messages = [{"role": "user", "content": "task"}]
    preparation.prepare_for_model(config, messages, set())
    receipt = "needed evidence " * 100
    messages.extend([
        {"role": "assistant", "tool_calls": [{"id": "small", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "name": "read_file", "tool_call_id": "small", "content": receipt},
    ])
    result = preparation.prepare_for_model(config, messages, set())
    assert result[-1]["content"] == receipt


async def test_direct_oversized_pending_receipt_is_not_sent_or_replayed(tmp_path):
    provider, capture = _provider(tmp_path)
    context = {"session_key": "topic", "turn_id": "turn", "native_context": {
        "context_window_tokens": 100000, "context_block_limit": 10000,
    }}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, model="test", request_context=context)
        call = first.tool_calls[0]
        messages.extend([
            {"role": "assistant", "tool_calls": [call.to_openai_tool_call()]},
            {"role": "tool", "name": call.name, "tool_call_id": call.id, "content": "huge " * 10000},
        ])
        result = await provider.chat(messages=messages, model="test", request_context=context)
        assert result.finish_reason == "error"
        assert not result.error_should_retry
        assert "exceeds local budget" in result.content
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert sum(m.get("method") == "thread/start" for m in rpc) == 1
        assert not any("result" in m and m["id"] == 101 for m in rpc)
        assert list((tmp_path / "ledger").glob("*.json"))
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()


async def test_cancellation_closes_native_thread_without_replay(tmp_path):
    provider, capture = _provider(tmp_path)
    context = {"session_key": "topic", "turn_id": "turn", "native_context": {
        "context_window_tokens": 100000, "context_block_limit": 80000,
    }}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, model="test", request_context=context)
        call = first.tool_calls[0]
        messages.extend([
            {"role": "assistant", "tool_calls": [call.to_openai_tool_call()]},
            {"role": "tool", "name": call.name, "tool_call_id": call.id, "content": "receipt"},
        ])
        bridge = provider._turns[("topic", "turn")]
        entered = asyncio.Event()

        async def paused(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        bridge.next_response = paused
        task = asyncio.create_task(provider.chat(messages=messages, model="test", request_context=context))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider._turns == provider._turn_locks == {}
        assert bridge.process is None
        assert list((tmp_path / "ledger").glob("*.json"))
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert sum(m.get("method") == "thread/start" for m in rpc) == 1
    finally:
        await provider.aclose()


def test_initial_trimming_does_not_reappear_and_runs_are_isolated(tmp_path):
    provider, _ = _provider(tmp_path)
    config = _config(provider, tmp_path, budget=10000)
    config.inflight_start_index = 3
    messages = [
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "OLD_BULK " * 20000},
        {"role": "user", "content": "new task"},
    ]
    first_run = NativeContextPreparation(ContextGovernor())
    first = first_run.prepare_for_model(config, messages, set())
    assert "OLD_BULK" not in str(first)
    messages.append({"role": "assistant", "content": "thinking"})
    second = first_run.prepare_for_model(config, messages, set())
    assert second[:-1] == first
    second_run = NativeContextPreparation(ContextGovernor())
    assert second_run.prepare_for_model(
        config, [{"role": "user", "content": "other child"}], set(),
    ) == [{"role": "user", "content": "other child"}]
    assert first_run.prepare_for_model(config, messages, set()) == second


async def test_zero_input_budget_is_not_treated_as_unlimited(tmp_path):
    provider, capture = _provider(tmp_path)
    # 避免模型目录的容量分支，用远小于实际窗口的显式限制验证输出预留。
    try:
        result = await provider.chat(
            messages=[{"role": "user", "content": "task"}], model="test",
            request_context={"native_context": {
                "context_window_tokens": 1000, "context_block_limit": 500, "max_tokens": 2000,
            }},
        )
        assert result.finish_reason == "error"
        assert "no input budget" in result.content
        assert not result.error_should_retry
        assert not capture.exists()
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()


@pytest.mark.parametrize("mode", [
    "native_safe", "steer_reject", "steer_mismatch", "steer_disconnect", "steer_later_disconnect",
])
@pytest.mark.parametrize("native_effects", [True, False])
async def test_native_receipt_steers_existing_thread_without_replaying(tmp_path, mode, native_effects):
    provider, capture = _provider(tmp_path, mode if native_effects else "plain:" + mode)
    config = _config(provider, tmp_path, budget=80000)
    preparation = NativeContextPreparation(ContextGovernor())
    # 统计全量治理次数：回执不应导致旧前缀再次治理。
    original_prepare = preparation.governor.prepare_for_model
    full_prepares = []

    def prepare(*args, **kwargs):
        full_prepares.append(True)
        return original_prepare(*args, **kwargs)

    preparation.governor.prepare_for_model = prepare
    context = {"session_key": "topic", "turn_id": "receipt"}
    messages = [{"role": "user", "content": "task"}]
    try:
        for n in range(1, 4):
            projected = preparation.prepare_for_model(config, messages, set())
            response = await provider.chat(messages=projected, request_context=context)
            assert response.has_tool_calls, response.content
            call = response.tool_calls[0]
            messages.extend([
                {"role": "assistant", "tool_calls": [call.to_openai_tool_call()]},
                {"role": "tool", "name": call.name, "tool_call_id": call.id,
                 "content": "receipt-" + str(n)},
            ])
        bridge = provider._turns[("topic", "receipt")]
        messages.append({"role": "user", "content": "WORKER_COMPLETED",
                         "injected_event": "subagent_result"})
        projected = preparation.prepare_for_model(config, messages, set())
        response = await provider.chat(messages=projected, request_context=context)
        assert len(full_prepares) == 1
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert sum(m.get("method") == "thread/start" for m in rpc) == 1
        steers = [m for m in rpc if m.get("method") == "turn/steer"]
        assert len(steers) == 1
        assert steers[0]["params"]["input"] == [{"type": "text", "text": "WORKER_COMPLETED"}]
        if mode != "native_safe":
            assert response.finish_reason == "error"
            assert not response.error_should_retry
            assert not response.has_tool_calls
            assert provider._turns == provider._turn_locks == {}
            assert list((tmp_path / "ledger").glob("*.json"))
            # 不确定是否接收成功时，不能重建或重复提交待处理结果。
            assert sum(m.get("id") == 103 and "result" in m for m in rpc) == (
                1 if mode == "steer_later_disconnect" else 0
            )
            return
        assert provider._turns[("topic", "receipt")] is bridge
        assert response.provider_diagnostics["context_sync"] == "steered"
        assert not response.provider_diagnostics.get("context_rebased")
        assert response.tool_calls[0].id == "call-4"
        assert not response.provider_diagnostics.get("idempotent_tool_replays")
        # 原线程继续到结束；不同回执分别注入一次，原生事件状态保留。
        for n in range(4, 8):
            call = response.tool_calls[0]
            messages.extend([
                {"role": "assistant", "tool_calls": [call.to_openai_tool_call()]},
                {"role": "tool", "name": call.name, "tool_call_id": call.id,
                 "content": "receipt-" + str(n)},
            ])
            if n == 5:
                messages.append({"role": "user", "content": "SECOND_RECEIPT"})
            response = await provider.chat(
                messages=preparation.prepare_for_model(config, messages, set()),
                request_context=context,
            )
        assert response.content == "done"
        assert response.provider_diagnostics.get("native_command_executions", {}).get("count", 0) == int(native_effects)
        assert response.provider_diagnostics.get("native_file_changes", {}).get("count", 0) == int(native_effects)
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert [m["params"]["input"] for m in rpc if m.get("method") == "turn/steer"] == [
            [{"type": "text", "text": "WORKER_COMPLETED"}],
            [{"type": "text", "text": "SECOND_RECEIPT"}],
        ]
        assert len(full_prepares) == 1
        assert sum(m.get("method") == "turn/start" for m in rpc) == 1
        assert [m["id"] for m in rpc if "result" in m] == list(range(101, 108))
    finally:
        await provider.aclose()


@pytest.mark.parametrize("mode", ["native_safe", "plain:native_safe"])
async def test_native_steer_checks_new_input_budget_before_sending(tmp_path, mode):
    provider, capture = _provider(tmp_path, mode)
    context = {"session_key": "topic", "turn_id": "oversized-steer", "native_context": {
        "context_window_tokens": 8000, "context_block_limit": 1000,
    }}
    messages = [{"role": "user", "content": "task"}]
    try:
        response = await provider.chat(messages=messages, request_context=context)
        assert response.has_tool_calls
        call = response.tool_calls[0]
        messages.extend([
            {"role": "assistant", "tool_calls": [call.to_openai_tool_call()]},
            {"role": "tool", "tool_call_id": call.id, "content": "ok"},
            {"role": "user", "content": "large receipt " * 10000},
        ])
        response = await provider.chat(messages=messages, request_context=context)
        assert response.finish_reason == "error"
        assert not response.has_tool_calls
        assert not response.error_should_retry
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert not any(m.get("method") == "turn/steer" for m in rpc)
        assert not any("result" in m for m in rpc)
        assert sum(m.get("method") == "thread/start" for m in rpc) == 1
        assert list((tmp_path / "ledger").glob("*.json"))
    finally:
        await provider.aclose()


async def test_receipt_after_history_exceeds_start_budget_keeps_existing_thread(tmp_path, monkeypatch):
    from nanobot.fork.providers import codex_native_context

    # 用确定性估算复现：单次增量合法，累计历史超过新线程启动预算。
    monkeypatch.setattr(codex_native_context, "estimate_prompt_tokens_chain", lambda p, m, rows, t: (
        sum(len(row.get("content") or "") for row in rows), "test",
    ))
    provider, capture = _provider(tmp_path, "normal")
    context = {"session_key": "topic", "turn_id": "large-receipt", "native_context": {
        "context_window_tokens": 10000, "context_block_limit": 1000,
    }}
    messages = [{"role": "user", "content": "h" * 200}]
    try:
        response = await provider.chat(messages=messages, request_context=context)
        bridge = provider._turns[("topic", "large-receipt")]
        for n in range(1, 8):
            assert response.has_tool_calls, response.content
            call = response.tool_calls[0]
            assert call.id == f"call-{n}"
            messages.extend([
                {"role": "assistant", "tool_calls": [call.to_openai_tool_call()]},
                {"role": "tool", "name": call.name, "tool_call_id": call.id, "content": "r" * 150},
            ])
            if n == 6:
                messages.append({"role": "user", "content": "WORKER_COMPLETED",
                                 "injected_event": "subagent_result"})
                with pytest.raises(ValueError, match="exceeds local budget"):
                    codex_native_context.check_native_payload(provider, "test", messages, [], 1000)
            response = await provider.chat(messages=messages, request_context=context)
            if n == 6:
                assert response.finish_reason != "error", response.content
                assert provider._turns[("topic", "large-receipt")] is bridge
                assert response.provider_diagnostics["context_sync"] == "steered"
                assert not response.provider_diagnostics.get("idempotent_tool_replays")
        assert response.content == "done"
        rpc = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert sum(m.get("method") == "thread/start" for m in rpc) == 1
        assert sum(m.get("method") == "turn/start" for m in rpc) == 1
        assert [m["params"]["input"] for m in rpc if m.get("method") == "turn/steer"] == [
            [{"type": "text", "text": "WORKER_COMPLETED"}],
        ]
        assert [m["id"] for m in rpc if "result" in m] == list(range(101, 108))
    finally:
        await provider.aclose()
