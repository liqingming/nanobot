"""原生协作禁用与父/子线程事件边界；全部使用离线假协议。"""

from unittest.mock import AsyncMock

import pytest

from nanobot.fork.providers.codex_app_server_provider import (
    CodexAppServerError,
    CodexAppServerProvider,
    _CodexAppServerTurn,
    _resolve_codex_app_server_command,
)

EVENTS = [
    ("item/agentMessage/delta", {"delta": "child text"}),
    ("item/reasoning/summaryTextDelta", {"delta": "child reasoning"}),
    ("item/tool/call", {"callId": "c", "tool": "write", "arguments": {}}),
    ("thread/tokenUsage/updated", {"tokenUsage": {"last": {"inputTokens": 999}}}),
    ("item/started", {"item": {"id": "compact", "type": "contextCompaction"}}),
    ("item/started", {"item": {"id": "cmd", "type": "commandExecution"}}),
    ("item/started", {"item": {"id": "patch", "type": "fileChange"}}),
    ("item/completed", {"item": {"id": "answer", "type": "agentMessage", "text": "child"}}),
    ("turn/completed", {"turn": {"id": "turn-1", "status": "completed"}}),
    ("item/commandExecution/requestApproval", {"itemId": "cmd"}),
]


@pytest.mark.parametrize("method,extra", EVENTS)
@pytest.mark.parametrize("identity", [
    {"threadId": "child", "turnId": "turn-1"},
    {"threadId": "parent", "turnId": "old-turn"},
    {},
])
async def test_foreign_or_missing_owner_stops_before_handling(method, extra, identity):
    bridge = _CodexAppServerTurn(["unused"])
    bridge.native_compaction.thread_id = "parent"
    bridge._turn_id = "turn-1"
    bridge._next_message = AsyncMock(return_value={
        "id": 80, "method": method, "params": {**extra, **identity},
    })
    bridge._send = AsyncMock()
    streamed = AsyncMock()
    with pytest.raises(CodexAppServerError, match="ownership mismatch") as error:
        await bridge.next_response(on_content_delta=streamed,
                                   on_thinking_delta=streamed, on_tool_call_delta=streamed)
    assert error.value.code == "event_owner_mismatch"
    streamed.assert_not_called()
    bridge._send.assert_not_called()
    assert bridge._pending_tools == {}
    assert bridge._context_input_tokens is None
    assert bridge.native_compaction.started == set()
    assert bridge._native_command_executions == {}
    assert bridge._native_file_changes == {}
    assert bridge._unsafe_protocol_event is True


async def test_matching_parent_events_still_work():
    bridge = _CodexAppServerTurn(["unused"])
    bridge.native_compaction.thread_id = "parent"
    bridge._turn_id = "turn-1"
    bridge._next_message = AsyncMock(side_effect=[
        {"method": "item/agentMessage/delta",
         "params": {"threadId": "parent", "turnId": "turn-1", "delta": "parent"}},
        {"method": "turn/completed",
         "params": {"threadId": "parent", "turn": {"id": "turn-1", "status": "completed"}}},
    ])
    stream = AsyncMock()
    response = await bridge.next_response(on_content_delta=stream,
                                         on_thinking_delta=None, on_tool_call_delta=None)
    assert response.content == "parent"
    stream.assert_awaited_once_with("parent")


def test_command_disables_new_and_legacy_agent_switches(monkeypatch, tmp_path):
    executable = tmp_path / "codex.exe"
    executable.touch()
    monkeypatch.setenv("NANOBOT_CODEX_BIN", str(executable))
    command = _resolve_codex_app_server_command()
    assert "agents.enabled=false" in command
    i = command.index("multi_agent")
    assert command[i - 1] == "--disable"


@pytest.mark.parametrize("collaboration", [True, False])
@pytest.mark.parametrize("budget", [0, 100000])
async def test_collaboration_or_foreign_event_never_restarts(
    monkeypatch, tmp_path, collaboration, budget,
):
    starts = []

    async def start(self, **kwargs):
        starts.append(self)
        self.native_compaction.thread_id = "parent"
        self._turn_id = "turn-1"

    async def next_message(self):
        if collaboration:
            return {"method": "item/started", "params": {
                "threadId": "parent", "turnId": "turn-1",
                "item": {"id": "spawn", "type": "collabAgentToolCall"},
            }}
        return {"method": "item/tool/call", "id": 5, "params": {
            "threadId": "child", "turnId": "child-turn",
            "callId": "side-effect", "tool": "write", "arguments": {},
        }}

    monkeypatch.setattr(_CodexAppServerTurn, "start", start)
    monkeypatch.setattr(_CodexAppServerTurn, "_next_message", next_message)
    monkeypatch.setattr(_CodexAppServerTurn, "close", AsyncMock())
    monkeypatch.setattr(
        "nanobot.fork.providers.codex_app_server_provider.native_input_budget",
        lambda *args: budget,
    )
    provider = CodexAppServerProvider(default_model="gpt-test", idempotency_dir=tmp_path)
    provider._app_server_command = ["unused"]
    response = await provider.chat(
        [{"role": "user", "content": "task"}],
        request_context={"session_key": "parent-session", "turn_id": "parent-turn"},
    )
    assert response.finish_reason == "error"
    assert response.error_code == (
        "native_tool_blocked" if collaboration else "event_owner_mismatch"
    )
    assert response.error_should_retry is False
    assert response.provider_diagnostics["native_recovery_suppressed"] is True
    assert len(starts) == 1
    assert provider._turns == {}
    assert provider._turn_locks == {}


async def test_conflicting_nested_turn_identity_is_rejected():
    bridge = _CodexAppServerTurn(["unused"])
    bridge.native_compaction.thread_id = "parent"
    bridge._turn_id = "turn-1"
    with pytest.raises(CodexAppServerError, match="ownership mismatch"):
        bridge._check_event_owner("turn/completed", {
            "threadId": "parent", "turnId": "turn-1",
            "turn": {"id": "foreign", "status": "completed"},
        })


async def test_foreign_event_after_tool_result_preserves_ledger_without_replay(monkeypatch, tmp_path):
    starts = []

    async def start(self, **kwargs):
        starts.append(self)
        self.native_compaction.thread_id = "parent"
        self._turn_id = "turn-1"
        self._send = AsyncMock()
        self._next_message = AsyncMock(side_effect=[
            {"id": 9, "method": "item/tool/call", "params": {
                "threadId": "parent", "turnId": "turn-1",
                "callId": "write-once", "tool": "write_file", "arguments": {"path": "a"},
            }},
            {"method": "turn/completed", "params": {
                "threadId": "child", "turn": {"id": "child-turn", "status": "completed"},
            }},
        ])

    monkeypatch.setattr(_CodexAppServerTurn, "start", start)
    monkeypatch.setattr(_CodexAppServerTurn, "close", AsyncMock())
    provider = CodexAppServerProvider(default_model="gpt-test", idempotency_dir=tmp_path)
    provider._app_server_command = ["unused"]
    ctx = {"session_key": "parent", "turn_id": "turn"}
    messages = [{"role": "user", "content": "task"}]
    first = await provider.chat(messages, request_context=ctx)
    assert first.has_tool_calls
    messages.extend([
        {"role": "assistant", "content": None,
         "tool_calls": [first.tool_calls[0].to_openai_tool_call()]},
        {"role": "tool", "tool_call_id": "write-once", "content": "written exactly once"},
    ])
    last = await provider.chat(messages, request_context=ctx)
    assert last.error_code == "event_owner_mismatch"
    assert not last.error_should_retry
    assert len(starts) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    starts[0]._send.assert_awaited_once()
