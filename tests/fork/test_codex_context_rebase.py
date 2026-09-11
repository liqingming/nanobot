"""真实 stdio 模拟：治理重建、最新证据可见、跨多轮禁止重复副作用。"""

import json
import sys
from copy import deepcopy

import pytest

from nanobot.fork.providers.codex_app_server_provider import (
    CodexAppServerProvider,
    _idempotency_ledger,
)
from nanobot.fork.providers.codex_context_checkpoint import ContextCheckpoint

_SERVER = r'''
import json
import sys
from pathlib import Path

mode, capture = sys.argv[1:]
def send(value):
    if value.get("method", "").startswith(("item/", "turn/")) or value.get("method") == "thread/tokenUsage/updated":
        value.setdefault("params", {}).setdefault("threadId", "thread")
        value["params"].setdefault("turnId", "turn")
    print(json.dumps(value), flush=True)
def call(number):
    send({"id": number, "method": "item/tool/call", "params": {
        "callId": "call-" + str(number),
        "tool": "write" if number == 1 else "read_file",
        "arguments": {"value": number},
    }})
steered = False
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        send({"id": msg["id"], "result": {}})
    elif method == "skills/list":
        send({"id": msg["id"], "result": {"data": []}})
    elif method == "thread/start":
        send({"id": msg["id"], "result": {"thread": {"id": "thread"}}})
    elif method == "turn/start":
        with Path(capture).open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg["params"]) + "\n")
        send({"id": msg["id"], "result": {"turn": {"id": "turn"}}})
        call(3 if "READ_RECEIPT" in json.dumps(msg["params"]) else 1)
    elif method == "turn/steer":
        with Path(str(capture) + ".steer").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg["params"]) + "\n")
        send({"id": msg["id"], "result": {"turnId": "turn"}})
        steered = True
    elif "result" in msg:
        n = msg["id"]
        if n == 1:
            call(2)
        elif n == 3 and mode.startswith("repeat"):
            call(1 if mode == "repeat_ordered" else 2)
        else:
            send({"method": "item/completed", "params": {
                "item": {"type": "agentMessage", "text": "done"},
            }})
            send({"method": "turn/completed", "params": {
                "turn": {"id": "turn", "status": "completed"},
            }})
'''


def _finish(messages, response, result):
    call = response.tool_calls[0]
    messages.extend([
        {"role": "assistant", "content": "", "tool_calls": [call.to_openai_tool_call()]},
        {"role": "tool", "name": call.name, "tool_call_id": call.id, "content": result},
    ])


def test_checkpoint_only_rebases_changed_history_or_settings():
    messages = [{"role": "user", "content": "task"}]
    checkpoint = ContextCheckpoint()
    checkpoint.capture(messages, ["model", []])
    assert not checkpoint.needs_rebase(messages + [{"role": "tool", "content": "ok"}], ["model", []])
    assert checkpoint.needs_rebase([{"role": "user", "content": "new"}], ["model", []])
    assert checkpoint.needs_rebase(messages, ["other", []])
    assert checkpoint.needs_rebase([], ["model", []])
    assert checkpoint.needs_rebase(messages + [{"role": "user", "content": "steer"}], ["model", []])


@pytest.mark.parametrize("mode", ["complete", "repeat_ordered", "repeat_out_of_order"])
async def test_governed_checkpoint_reaches_stdio_and_replay_stays_safe(tmp_path, mode):
    capture = tmp_path / "starts.jsonl"
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    provider._app_server_command = [sys.executable, "-u", "-c", _SERVER, mode, str(capture)]
    context = {"session_key": "topic", "turn_id": "turn"}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "OLD_BULK " * 1000},
    ]
    exposed = []
    try:
        first = await provider.chat(messages=messages, request_context=context)
        exposed.extend(call.name for call in first.tool_calls)
        _finish(messages, first, "WRITE_RECEIPT")
        bridge = provider._turns[("topic", "turn")]
        second = await provider.chat(messages=messages, request_context=context)
        assert provider._turns[("topic", "turn")] is bridge
        exposed.extend(call.name for call in second.tool_calls)
        receipt = "READ_RECEIPT attemptId=9745 sha256=1cc7 receipt=attempt/result.json"
        _finish(messages, second, receipt)
        original = deepcopy(messages)
        governed = deepcopy(messages)
        governed[2]["content"] = "Earlier generated bulk was compacted."
        governed[4]["content"] = "Prior write completed."
        third = await provider.chat(messages=governed, request_context=context)
        assert third.provider_diagnostics["context_rebased"] is True
        assert bridge.process is None
        assert messages == original
        starts = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        assert len(starts) == 2
        rebuilt = json.dumps(starts[-1])
        assert "OLD_BULK" not in rebuilt
        assert receipt in rebuilt
        assert "Prior write completed." in rebuilt
        # 摘要同步线程，但原始幂等结果保持 write-once。
        ledger = _idempotency_ledger(("topic", "turn"), root_override=tmp_path / "ledger")
        assert ledger._entry_result(ledger.entries[0]) == "WRITE_RECEIPT"
        exposed.extend(call.name for call in third.tool_calls)
        _finish(governed, third, "fresh read")
        fourth = await provider.chat(messages=governed, request_context=context)
        if mode == "repeat_out_of_order":
            assert fourth.finish_reason == "error"
            assert "order diverged" in fourth.content
            assert ledger.path.exists()
        else:
            assert fourth.content == "done"
            if mode == "repeat_ordered":
                assert fourth.provider_diagnostics["idempotent_tool_replays"] == 2
        assert exposed == ["write", "read_file", "read_file"]
        assert not fourth.has_tool_calls
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()


@pytest.mark.parametrize("change", ["history", "model", "system", "missing_result"])
async def test_native_effects_block_automatic_governance_rebase(tmp_path, change):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    provider._app_server_command = [
        sys.executable, "-u", "-c", _SERVER, "complete", str(tmp_path / "starts"),
    ]
    context = {"session_key": "topic", "turn_id": "native"}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, request_context=context)
        bridge = provider._turns[("topic", "native")]
        bridge._native_command_executions["command"] = "completed"
        _finish(messages, first, "written")
        kwargs = {}
        if change == "history":
            messages[0] = {"role": "user", "content": "governed task"}
        elif change == "model":
            kwargs["model"] = "different-model"
        elif change == "system":
            messages.append({"role": "system", "content": "new system"})
        else:
            messages.pop()
            messages.append({"role": "user", "content": "receipt"})
        response = await provider.chat(messages=messages, request_context=context, **kwargs)
        assert response.finish_reason == "error"
        assert ("checkpointed pending results" if change == "missing_result"
                else "native side effects") in response.content
        assert not response.has_tool_calls
        assert bridge.process is None
        assert _idempotency_ledger(
            ("topic", "native"), root_override=tmp_path / "ledger",
        ).has_entries is (change != "missing_result")
        assert len((tmp_path / "starts").read_text(encoding="utf-8").splitlines()) == 1
    finally:
        await provider.aclose()


async def test_user_injection_keeps_pending_result_checkpoint(tmp_path):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    capture = tmp_path / "starts"
    provider._app_server_command = [
        sys.executable, "-u", "-c", _SERVER, "complete", str(capture),
    ]
    context = {"session_key": "topic", "turn_id": "injection"}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, request_context=context)
        _finish(messages, first, "READ_RECEIPT already written")
        messages.append({"role": "user", "content": "NEW_REQUIREMENT"})
        response = await provider.chat(messages=messages, request_context=context)
        assert response.has_tool_calls
        assert response.tool_calls[0].name == "read_file"
        assert response.provider_diagnostics["context_sync"] == "steered"
        assert len(capture.read_text(encoding="utf-8").splitlines()) == 1
        assert "NEW_REQUIREMENT" in capture.with_name(capture.name + ".steer").read_text(encoding="utf-8")
        ledger = _idempotency_ledger(("topic", "injection"), root_override=tmp_path / "ledger")
        assert ledger._entry_result(ledger.entries[0]) == "READ_RECEIPT already written"
    finally:
        await provider.aclose()


async def test_missing_pending_result_fails_before_rebase(tmp_path):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    capture = tmp_path / "starts"
    provider._app_server_command = [
        sys.executable, "-u", "-c", _SERVER, "complete", str(capture),
    ]
    context = {"session_key": "topic", "turn_id": "missing"}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, request_context=context)
        assert first.has_tool_calls
        # 不可把被裁掉的待处理工具结果当成成功检查点，亦不可新开线程重做。
        messages[0]["content"] = "changed"
        result = await provider.chat(messages=messages, request_context=context)
        assert result.finish_reason == "error"
        assert "checkpointed pending results" in result.content
        assert len(capture.read_text(encoding="utf-8").splitlines()) == 1
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()


# 模拟主线程：先产生副作用，再查 running；收到回执后重建并查询最新状态。
_OBSERVATION_SERVER = r'''
import json
import sys
from pathlib import Path

mode, action, capture = sys.argv[1:]
def send(value):
    if value.get("method", "").startswith(("item/", "turn/")):
        value.setdefault("params", {}).update(threadId="thread", turnId="turn")
    print(json.dumps(value), flush=True)
def call(number):
    send({"id": number, "method": "item/tool/call", "params": {
        "callId": "call-" + str(number),
        "tool": "exec" if number == 1 else (
            "write_stdin" if action.startswith("poll") else "subagent_control"
        ),
        "arguments": {"command": "write once"} if number == 1 else (
            {"session_id": "process", "yield_time_ms": 1000, "max_output_chars": 4000,
             **{"poll": {"chars": ""}, "poll_default": {}, "poll_null": {"chars": None},
                "poll_input": {"chars": "continue\n"}, "poll_close": {"close_stdin": True},
                "poll_terminate": {"terminate": True}}[action]}
            if action.startswith("poll") else (
                {"action": action} if action == "list" else {"action": action, "task_id": "worker"}
            )
        ),
    }})
steered = False
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        send({"id": msg["id"], "result": {}})
    elif method == "skills/list":
        send({"id": msg["id"], "result": {"data": []}})
    elif method == "thread/start":
        send({"id": msg["id"], "result": {"thread": {"id": "thread"}}})
    elif method == "turn/start":
        with Path(capture).open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg["params"]) + "\n")
        send({"id": msg["id"], "result": {"turn": {"id": "turn"}}})
        rebuilt = "WORKER_COMPLETED" in json.dumps(msg["params"])
        number = (2 if mode == "same_id" else 3) if rebuilt else 1
        if mode == "disconnect" and len(Path(capture).read_text().splitlines()) > 2:
            number = 5  # 真断线后用全新 ID，防止测试因旧 ID 碰巧阻断而假通过。
        call(number)
    elif method == "turn/steer":
        with Path(str(capture) + ".steer").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg["params"]) + "\n")
        send({"id": msg["id"], "result": {"turnId": "turn"}})
        steered = True
    elif "result" in msg:
        n = msg["id"]
        if n == 1:
            call(2)
        elif n == 2 and steered:
            call(3)
        elif n == 3:
            if mode == "disconnect":
                sys.exit(0)
            call(4)
        else:
            send({"method": "item/completed", "params": {
                "item": {"type": "agentMessage", "text": "done"},
            }})
            send({"method": "turn/completed", "params": {
                "turn": {"id": "turn", "status": "completed"},
            }})
'''


@pytest.mark.parametrize("action", ["status", "list", "poll", "poll_default", "poll_null"])
@pytest.mark.parametrize("trigger", ["receipt", "governance"])
async def test_rebase_refreshes_observation_instead_of_replaying_running(tmp_path, action, trigger):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    capture = tmp_path / "starts"
    provider._app_server_command = [
        sys.executable, "-u", "-c", _OBSERVATION_SERVER, "fresh", action, str(capture),
    ]
    context = {"session_key": "topic", "turn_id": "observation"}
    messages = [{"role": "user", "content": "task"}]
    emitted = []

    async def on_tool(delta):
        emitted.append(delta["id"])

    try:
        first = await provider.chat(messages=messages, request_context=context)
        _finish(messages, first, "WRITE_RECEIPT")
        second = await provider.chat(messages=messages, request_context=context)
        _finish(messages, second, "running")
        if trigger == "receipt":
            messages.append({
                "role": "user", "content": "WORKER_COMPLETED",
                "injected_event": "subagent_result", "subagent_task_id": "worker",
            })
        else:
            messages[0]["content"] = "task WORKER_COMPLETED"
        third = await provider.chat_stream(
            messages=messages, request_context=context, on_tool_call_delta=on_tool,
        )
        assert third.finish_reason != "error", third.content
        assert third.tool_calls[0].id == "call-3"
        assert third.provider_diagnostics["context_sync"] == ("steered" if trigger == "receipt" else "rebased")
        assert bool(third.provider_diagnostics.get("checkpoint_continuation")) is (trigger == "governance")
        assert not third.provider_diagnostics.get("idempotent_tool_replays")
        _finish(messages, third, "stopped")
        # 同一重建线程的后续轮次仍须读取新状态，不能退回旧账本游标。
        fourth = await provider.chat_stream(
            messages=messages, request_context=context, on_tool_call_delta=on_tool,
        )
        assert fourth.finish_reason != "error", fourth.content
        assert fourth.tool_calls[0].id == "call-4"
        assert emitted == ["call-3", "call-4"]
        ledger = _idempotency_ledger(("topic", "observation"), root_override=tmp_path / "ledger")
        assert [ledger._entry_result(entry) for entry in ledger.entries] == [
            "WRITE_RECEIPT", "running", "stopped",
        ]
        assert len(capture.read_text(encoding="utf-8").splitlines()) == (1 if trigger == "receipt" else 2)
        _finish(messages, fourth, "stopped")
        final = await provider.chat(messages=messages, request_context=context)
        assert final.content == "done"
    finally:
        await provider.aclose()


@pytest.mark.parametrize("mode,action", [
    ("same_id", "status"), ("fresh", "cancel"), ("same_id", "poll"),
    ("fresh", "poll_input"), ("fresh", "poll_close"), ("fresh", "poll_terminate"),
])
async def test_rebase_does_not_bypass_call_identity_or_cancel_replay(tmp_path, mode, action):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    provider._app_server_command = [
        sys.executable, "-u", "-c", _OBSERVATION_SERVER, mode, action, str(tmp_path / "starts"),
    ]
    context = {"session_key": "topic", "turn_id": "protected"}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, request_context=context)
        _finish(messages, first, "WRITE_RECEIPT")
        second = await provider.chat(messages=messages, request_context=context)
        _finish(messages, second, "old receipt")
        messages[0]["content"] = "task WORKER_COMPLETED"
        response = await provider.chat(messages=messages, request_context=context)
        assert response.finish_reason == "error"
        assert "order diverged" in response.content
        assert not response.has_tool_calls
        assert _idempotency_ledger(
            ("topic", "protected"), root_override=tmp_path / "ledger",
        ).has_entries
    finally:
        await provider.aclose()


@pytest.mark.parametrize("action", ["status", "poll"])
async def test_disconnect_after_checkpoint_continuation_restores_strict_replay(tmp_path, action):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    capture = tmp_path / "starts"
    provider._app_server_command = [
        sys.executable, "-u", "-c", _OBSERVATION_SERVER, "disconnect", action, str(capture),
    ]
    context = {"session_key": "topic", "turn_id": "disconnect"}
    messages = [{"role": "user", "content": "task"}]
    try:
        first = await provider.chat(messages=messages, request_context=context)
        _finish(messages, first, "WRITE_RECEIPT")
        second = await provider.chat(messages=messages, request_context=context)
        _finish(messages, second, "running")
        messages[0]["content"] = "task WORKER_COMPLETED"
        third = await provider.chat(messages=messages, request_context=context)
        assert third.has_tool_calls
        _finish(messages, third, "stopped")
        response = await provider.chat(messages=messages, request_context=context)
        assert response.finish_reason == "error"
        assert "order diverged" in response.content
        assert not response.has_tool_calls
        assert len(capture.read_text(encoding="utf-8").splitlines()) == 3
        ledger = _idempotency_ledger(("topic", "disconnect"), root_override=tmp_path / "ledger")
        assert [ledger._entry_result(entry) for entry in ledger.entries] == [
            "WRITE_RECEIPT", "running", "stopped",
        ]
        assert provider._turns == provider._turn_locks == {}
    finally:
        await provider.aclose()
