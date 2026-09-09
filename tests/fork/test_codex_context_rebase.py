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
    print(json.dumps(value), flush=True)
def call(number):
    send({"id": number, "method": "item/tool/call", "params": {
        "callId": "call-" + str(number),
        "tool": "write" if number == 1 else "read_file",
        "arguments": {"value": number},
    }})
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


async def test_native_effects_block_automatic_governance_rebase(tmp_path):
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
        messages[0] = {"role": "user", "content": "governed task"}
        response = await provider.chat(messages=messages, request_context=context)
        assert response.finish_reason == "error"
        assert "native side effects" in response.content
        assert not response.has_tool_calls
        assert bridge.process is None
        assert _idempotency_ledger(
            ("topic", "native"), root_override=tmp_path / "ledger",
        ).has_entries
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
        assert response.provider_diagnostics["context_rebased"]
        assert "NEW_REQUIREMENT" in capture.read_text(encoding="utf-8").splitlines()[-1]
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
