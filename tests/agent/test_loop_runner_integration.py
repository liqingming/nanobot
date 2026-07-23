"""Tests for AgentLoop integration with AgentRunner: streaming, think-filter, error handling, subagent."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.outbound_events import StreamedResponseEvent
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMResponse, ToolCallRequest

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)
    return loop

@pytest.mark.asyncio
async def test_loop_max_iterations_message_stays_stable(tmp_path):
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")
    loop.max_iterations = 2

    final_content, _, _, _, _ = await loop._run_agent_loop([])

    # Fork: bilingual max_iterations message injected by loop._run_agent_loop
    assert final_content == (
        "我已经用完了本轮的工具调用预算 (2 次)，但任务还没完成。"
        "目前进度已保存——输入 `/continue` 可以从断点继续，"
        "或者把任务拆成更小的步骤。"
    )


@pytest.mark.asyncio
async def test_loop_goal_turn_uses_standard_iteration_budget(tmp_path):
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")
    loop.max_iterations = 2

    final_content, _, _, stop_reason, _ = await loop._run_agent_loop(
        [],
        metadata={"original_command": "/goal"},
    )

    assert stop_reason == "max_iterations"
    assert loop.provider.chat_with_retry.await_count == 3
    assert loop.provider.chat_with_retry.await_args_list[-1].kwargs["tools"] is None
    assert "工具调用预算 (2 次)" in final_content
    assert "/continue" in final_content


@pytest.mark.asyncio
async def test_loop_stream_filter_handles_think_only_prefix_without_crashing(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("<think>hidden")
        await on_content_delta("</think>Hello")
        return LLMResponse(content="<think>hidden</think>Hello", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        endings.append(resuming)

    final_content, _, _, _, _ = await loop._run_agent_loop(
        [],
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert final_content == "Hello"
    assert deltas == ["Hello"]
    assert endings == [False]


@pytest.mark.asyncio
async def test_loop_stream_filter_hides_partial_trailing_think_prefix(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("Hello <thin")
        await on_content_delta("k>hidden</think>World")
        return LLMResponse(content="Hello <think>hidden</think>World", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    final_content, _, _, _, _ = await loop._run_agent_loop([], on_stream=on_stream)

    assert final_content == "Hello World"
    assert deltas == ["Hello", " World"]


@pytest.mark.asyncio
async def test_loop_stream_filter_hides_complete_trailing_think_tag(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("Hello <think>")
        await on_content_delta("hidden</think>World")
        return LLMResponse(content="Hello <think>hidden</think>World", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    final_content, _, _, _, _ = await loop._run_agent_loop([], on_stream=on_stream)

    assert final_content == "Hello World"
    assert deltas == ["Hello", " World"]


@pytest.mark.asyncio
async def test_loop_retries_think_only_final_response(tmp_path):
    loop = _make_loop(tmp_path)
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(content="<think>hidden</think>", tool_calls=[], usage={})
        return LLMResponse(content="Recovered answer", tool_calls=[], usage={})

    loop.provider.chat_with_retry = chat_with_retry

    final_content, _, _, _, _ = await loop._run_agent_loop([])

    assert final_content == "Recovered answer"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_streamed_flag_not_set_on_llm_error(tmp_path):
    """When LLM errors during a streaming-capable channel interaction,
    _streamed must NOT be set so ChannelManager delivers the error."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")
    error_resp = LLMResponse(
        content="503 service unavailable", finish_reason="error", tool_calls=[], usage={},
    )
    loop.provider.chat_with_retry = AsyncMock(return_value=error_resp)
    loop.provider.chat_stream_with_retry = AsyncMock(return_value=error_resp)
    loop.tools.get_definitions = MagicMock(return_value=[])

    msg = InboundMessage(
        channel="feishu", sender_id="u1", chat_id="c1", content="hi",
    )
    result = await loop._process_message(
        msg,
        on_stream=AsyncMock(),
        on_stream_end=AsyncMock(),
    )

    assert result is not None
    assert "503" in result.content
    assert not isinstance(result.event, StreamedResponseEvent), (
        "streamed response event must not be set when stop_reason is error"
    )
    assert not result.metadata.get("_streamed"), \
        "_streamed must not be set when stop_reason is error"
    assert result.metadata.get("_error") is True
    assert result.metadata.get("render_as") == "error"
    assert result.metadata.get("stop_reason") == "error"


@pytest.mark.asyncio
async def test_ssrf_soft_block_can_finalize_after_streamed_tool_call(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    tool_call_resp = LLMResponse(
        content="checking metadata",
        tool_calls=[ToolCallRequest(
            id="call_ssrf",
            name="exec",
            arguments={"command": "curl http://169.254.169.254/latest/meta-data/"},
        )],
        usage={},
    )
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        tool_call_resp,
        LLMResponse(
            content="I cannot access private URLs. Please share the local file.",
            tool_calls=[],
            usage={},
        ),
    ])

    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.prepare_call = MagicMock(return_value=(None, {}, None))
    loop.tools.execute = AsyncMock(return_value=(
        "Error: Command blocked by safety guard (internal/private URL detected)"
    ))

    result = await loop._process_message(
        InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="hi"),
        on_stream=AsyncMock(),
        on_stream_end=AsyncMock(),
    )

    assert result is not None
    assert result.content == "I cannot access private URLs. Please share the local file."
    assert isinstance(result.event, StreamedResponseEvent)


@pytest.mark.asyncio
async def test_next_turn_after_llm_error_keeps_turn_boundary(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.runner import _PERSISTED_MODEL_ERROR_PLACEHOLDER
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="429 rate limit exceeded", finish_reason="error", tool_calls=[], usage={}),
        LLMResponse(content="Recovered answer", tool_calls=[], usage={}),
    ])

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    first = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="first question")
    )
    assert first is not None
    assert first.content == "429 rate limit exceeded"

    session = loop.sessions.get_or_create("cli:test")
    assert [
        {key: value for key, value in message.items() if key in {"role", "content"}}
        for message in session.messages
    ] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": _PERSISTED_MODEL_ERROR_PLACEHOLDER},
    ]

    second = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="second question")
    )
    assert second is not None
    assert second.content == "Recovered answer"

    request_messages = provider.chat_with_retry.await_args_list[1].kwargs["messages"]
    non_system = [message for message in request_messages if message.get("role") != "system"]
    assert non_system[0]["role"] == "user"
    assert "first question" in non_system[0]["content"]
    assert non_system[1]["role"] == "assistant"
    assert _PERSISTED_MODEL_ERROR_PLACEHOLDER in non_system[1]["content"]
    assert non_system[2]["role"] == "user"
    assert "second question" in non_system[2]["content"]


@pytest.mark.asyncio
async def test_subagent_max_iterations_announces_existing_fallback(tmp_path, monkeypatch):
    from nanobot.agent.subagent import SubagentManager, SubagentStatus
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    mgr = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    )
    mgr._announce_result = AsyncMock()

    async def fake_execute(self, **kwargs):
        return "tool result"

    monkeypatch.setattr("nanobot.agent.tools.filesystem.ListDirTool.execute", fake_execute)

    status = SubagentStatus(task_id="sub-1", label="label", task_description="do task", started_at=time.monotonic())
    await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"}, status)

    mgr._announce_result.assert_awaited_once()
    args = mgr._announce_result.await_args.args
    assert args[3] == "Task completed but no final response was generated."
    assert args[5] == "ok"


@pytest.mark.asyncio
async def test_dispatch_auto_recovers_transient_model_error(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    error_response = LLMResponse(
        content="503 server error from model",
        finish_reason="error",
        tool_calls=[],
        usage={},
    )
    provider.chat_with_retry = AsyncMock(return_value=error_response)
    provider.chat_stream_with_retry = AsyncMock(return_value=error_response)
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    await loop._dispatch(InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="test",
        content="do the task",
        metadata={"_wants_stream": True},
    ))
    if loop._background_tasks:
        await asyncio.gather(*list(loop._background_tasks), return_exceptions=True)

    outbound = [
        await asyncio.wait_for(bus.consume_outbound(), timeout=1)
        for _ in range(bus.outbound_size)
    ]
    error_out = next(msg for msg in outbound if msg.metadata.get("_error"))
    notice_out = next(msg for msg in outbound if msg.metadata.get("_system_message"))
    recovery_in = await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    assert error_out.metadata.get("_error") is True
    assert error_out.metadata.get("stop_reason") == "error"
    assert notice_out.metadata.get("_system_message") is True
    assert "自动恢复任务" in notice_out.content
    assert recovery_in.content == "请继续上次因模型服务错误中断的任务。"
    assert recovery_in.metadata.get("_auto_recovery") is True
    assert recovery_in.metadata.get("_auto_recover_attempt") == 1
    assert recovery_in.session_key == "cli:test"


def test_newly_completed_goal_fallback_uses_current_turn_recap() -> None:
    from nanobot.agent.loop import _newly_completed_goal_fallback

    metadata = {
        "goal_state": {
            "status": "completed",
            "completed_at": "2026-07-20T11:40:06Z",
            "recap": "取证完成，未修改文件。",
        }
    }

    fallback = _newly_completed_goal_fallback(
        metadata,
        completed_at_before=None,
    )

    assert fallback == (
        "最终回复生成失败，已恢复任务完成时保存的摘要：\n\n"
        "取证完成，未修改文件。"
    )


def test_newly_completed_goal_fallback_does_not_reuse_stale_recap() -> None:
    from nanobot.agent.loop import _newly_completed_goal_fallback

    metadata = {
        "goal_state": {
            "status": "completed",
            "completed_at": "2026-07-20T11:40:06Z",
            "recap": "旧任务摘要",
        }
    }

    assert _newly_completed_goal_fallback(
        metadata,
        completed_at_before="2026-07-20T11:40:06Z",
    ) is None


@pytest.mark.asyncio
async def test_loop_runtime_log_correlates_complete_tool_trace_by_turn_id(tmp_path):
    loop = _make_loop(tmp_path)
    runtime_log = tmp_path / "runtime.log"
    loop.sessions.get_session_runtime_log_path.return_value = runtime_log
    loop.provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="working",
            tool_calls=[ToolCallRequest(
                id="call_read",
                name="read_file",
                arguments={
                    "path": "a.py",
                    "offset": 10,
                    "limit": 20,
                    "pages": "",
                    "force": False,
                },
            )],
        ),
        LLMResponse(content="done", tool_calls=[]),
    ])
    audit_tools = MagicMock()
    audit_tools.get_definitions.return_value = []
    audit_tools.prepare_call.return_value = (None, {}, None)
    audit_tools.execute = AsyncMock(return_value="file body")

    final_content, _, _, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "inspect"}],
        session_key="cli:topic",
        turn_id="cli:topic:turn-42",
        tools=audit_tools,
    )

    records = [
        json.loads(line)
        for line in runtime_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runner_records = [row for row in records if row["event"].startswith("runner.")]
    audit_start = next(row for row in records if row["event"] == "runner.tool.audit.start")
    audit_end = next(row for row in records if row["event"] == "runner.tool.audit.end")
    assert final_content == "done"
    assert runner_records
    assert {row["turn_id"] for row in runner_records} == {"cli:topic:turn-42"}
    assert audit_start["iteration"] == 0
    assert audit_start["call_id"] == "call_read"
    assert audit_start["arguments"] == {
        "path": "a.py",
        "offset": 10,
        "limit": 20,
        "pages": "",
        "force": False,
    }
    assert audit_end["iteration"] == 0
    assert audit_end["status"] == "ok"
    assert audit_end["result_preview"] == "file body"
