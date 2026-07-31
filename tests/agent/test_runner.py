"""Tests for the shared agent runner and its integration contracts."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)
    return loop


@pytest.mark.asyncio
async def test_active_goal_does_not_continue_after_normal_final_response():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "finish this"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=16000,
        goal_active_predicate=lambda: True,
        goal_continue_message="Continue the active goal.",
    ))

    assert result.final_content == "done"
    assert provider.chat_with_retry.await_count == 1
    assert all(message.get("content") != "Continue the active goal." for message in result.messages)


@pytest.mark.asyncio
async def test_real_injection_still_continues_after_normal_final_response():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="first", tool_calls=[]),
        LLMResponse(content="second", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    injections = [[{"role": "user", "content": "real follow-up"}], []]

    async def drain_injections(**_kwargs):
        return injections.pop(0)

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "start"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=16000,
        injection_callback=drain_injections,
        goal_active_predicate=lambda: True,
    ))

    assert result.final_content == "second"
    assert provider.chat_with_retry.await_count == 2
    assert any(message.get("content") == "real follow-up" for message in result.messages)


@pytest.mark.asyncio
async def test_runner_preserves_reasoning_fields_and_tool_results():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                reasoning_content="hidden reasoning",
                thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                usage={"prompt_tokens": 5, "completion_tokens": 3},
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=3, max_tool_result_chars=16000,
    ))

    assert result.final_content == "done"
    assert result.tools_used == ["list_dir"]
    assert result.tool_events == [
        {"name": "list_dir", "status": "ok", "detail": "tool result"}
    ]

    assistant_messages = [
        msg for msg in captured_second_call
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
    assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]
    assert any(
        msg.get("role") == "tool" and msg.get("content") == "tool result"
        for msg in captured_second_call
    )


@pytest.mark.asyncio
async def test_runner_calls_hooks_in_order():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}
    events: list[tuple] = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    class RecordingHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            events.append(("before_iteration", context.iteration))

        async def before_execute_tools(self, context: AgentHookContext) -> None:
            events.append((
                "before_execute_tools",
                context.iteration,
                [tc.name for tc in context.tool_calls],
            ))

        async def after_iteration(self, context: AgentHookContext) -> None:
            events.append((
                "after_iteration",
                context.iteration,
                context.final_content,
                list(context.tool_results),
                list(context.tool_events),
                context.stop_reason,
            ))

        def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
            events.append(("finalize_content", context.iteration, content))
            return content.upper() if content else content

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=16000,
        hook=RecordingHook(),
    ))

    assert result.final_content == "DONE"
    assert events == [
        ("before_iteration", 0),
        ("before_execute_tools", 0, ["list_dir"]),
        (
            "after_iteration",
            0,
            None,
            ["tool result"],
            [{"name": "list_dir", "status": "ok", "detail": "tool result"}],
            None,
        ),
        ("before_iteration", 1),
        ("finalize_content", 1, "done"),
        ("after_iteration", 1, "DONE", [], [], "completed"),
    ]


@pytest.mark.asyncio
async def test_runner_streaming_hook_receives_deltas_and_end_signal():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    streamed: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("he")
        await on_content_delta("llo")
        return LLMResponse(content="hello", tool_calls=[], usage={})

    provider.chat_stream_with_retry = chat_stream_with_retry
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            streamed.append(delta)

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=16000,
        hook=StreamingHook(),
    ))

    assert result.final_content == "hello"
    assert streamed == ["he", "llo"]
    assert endings == [False]
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_returns_max_iterations_fallback():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="still working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2, max_tool_result_chars=16000,
    ))

    assert result.stop_reason == "max_iterations"
    assert result.final_content == (
        "I reached the maximum number of tool call iterations (2) without "
        "completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_runner_warns_then_stops_identical_tool_loop():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    requests: list[list[dict]] = []

    async def chat_with_retry(*, messages, **_kwargs):
        requests.append(messages)
        call_number = len(requests)
        return LLMResponse(
            content="still checking",
            tool_calls=[ToolCallRequest(
                id=f"call_{call_number}",
                name="grep",
                arguments={"path": "a.py", "pattern": "target"},
            )],
        )

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="same result")

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=20,
        max_tool_result_chars=16000,
    ))

    assert result.stop_reason == "tool_loop"
    assert "grep" in result.final_content
    assert len(requests) == 5
    assert tools.execute.await_count == 4
    assert any(
        message.get("role") == "user"
        and "[Runtime correction]" in message.get("content", "")
        for message in requests[3]
    )


@pytest.mark.asyncio
async def test_runner_detects_repeated_multi_call_cycle():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    sequence = ["first", "second", "third", "fourth", "fifth"]
    call_count = 0

    async def chat_with_retry(**_kwargs):
        nonlocal call_count
        name = sequence[call_count % len(sequence)]
        call_count += 1
        return LLMResponse(
            content="checking",
            tool_calls=[ToolCallRequest(
                id=f"call_{call_count}",
                name="grep",
                arguments={"path": "a.py", "pattern": name},
            )],
        )

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="result")

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=40,
        max_tool_result_chars=16000,
    ))

    assert result.stop_reason == "tool_loop"
    assert "grep → grep → grep → grep → grep" in result.final_content
    assert call_count == 25
    assert tools.execute.await_count == 24


@pytest.mark.asyncio
async def test_runner_tool_loop_correction_can_recover():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    call_count = 0

    async def chat_with_retry(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return LLMResponse(
                content="checking",
                tool_calls=[ToolCallRequest(
                    id=f"call_{call_count}",
                    name="grep",
                    arguments={"path": "a.py", "pattern": "target"},
                )],
            )
        return LLMResponse(content="changed approach and finished", tool_calls=[])

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="same result")

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=16000,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "changed approach and finished"
    assert tools.execute.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("write_stdin", {"session_id": "s1", "chars": ""}),
        ("list_dir", {"path": "."}),
        ("process_control", {"action": "logs", "process_id": "p1", "tail": 20}),
    ],
)
async def test_runner_exempts_polling_tool_repetition(tool_name, arguments):
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    call_count = 0

    async def chat_with_retry(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 6:
            return LLMResponse(
                content="waiting",
                tool_calls=[ToolCallRequest(
                    id=f"call_{call_count}",
                    name=tool_name,
                    arguments=arguments,
                )],
            )
        return LLMResponse(content="process completed", tool_calls=[])

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="still running")

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=16000,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "process completed"
    assert tools.execute.await_count == 6


@pytest.mark.asyncio
async def test_runner_returns_structured_tool_error():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=RuntimeError("boom"))

    runner = AgentRunner(provider)

    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=16000,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "tool_error"
    assert result.error == "Error: RuntimeError: boom"
    assert result.tool_events == [
        {"name": "list_dir", "status": "error", "detail": "boom"}
    ]


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
async def test_subagent_max_iterations_announces_existing_fallback(tmp_path, monkeypatch):
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus, max_tool_result_chars=16000)
    mgr.max_iterations = 2
    mgr._announce_result = AsyncMock()

    async def fake_execute(self, name, arguments):
        return "tool result"

    monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", fake_execute)

    import time
    from nanobot.agent.subagent import SubagentStatus
    status = SubagentStatus(task_id="sub-1", label="label", task_description="do task", started_at=time.monotonic())
    await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"}, status)

    mgr._announce_result.assert_awaited_once()
    args = mgr._announce_result.await_args.args
    assert args[3] == "Task completed but no final response was generated."
    assert args[5] == "ok"


@pytest.mark.asyncio
async def test_runner_runtime_audit_reconstructs_each_tool_iteration():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    responses = [
        LLMResponse(
            content="step one",
            tool_calls=[ToolCallRequest(
                id="call_read",
                name="read_file",
                arguments={"path": "a.py", "offset": 1},
            )],
        ),
        LLMResponse(
            content="step two",
            tool_calls=[ToolCallRequest(
                id="call_grep",
                name="grep",
                arguments={"path": ".", "pattern": "target"},
            )],
        ),
        LLMResponse(content="done", tool_calls=[]),
    ]
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=responses)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=["file body", "matching line"])
    events: list[tuple[str, dict]] = []

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "inspect"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=16000,
        event_logger=lambda event, fields: events.append((event, fields)),
        turn_id="cli:topic:turn-1",
    ))

    starts = [fields for event, fields in events if event == "runner.tool.audit.start"]
    ends = [fields for event, fields in events if event == "runner.tool.audit.end"]
    assert result.final_content == "done"
    assert [(row["iteration"], row["call_id"], row["tool"]) for row in starts] == [
        (0, "call_read", "read_file"),
        (1, "call_grep", "grep"),
    ]
    assert [row["arguments"] for row in starts] == [
        {"path": "a.py", "offset": 1},
        {"path": ".", "pattern": "target"},
    ]
    assert [(row["iteration"], row["call_id"], row["status"]) for row in ends] == [
        (0, "call_read", "ok"),
        (1, "call_grep", "ok"),
    ]
    assert [row["result_chars"] for row in ends] == [9, 13]
    assert [row["result_preview"] for row in ends] == ["file body", "matching line"]
    assert all(row["duration_ms"] >= 0 for row in ends)
    plans = [fields for event, fields in events if event == "runner.tools.start"]
    assert [row["concurrent_tools"] for row in plans] == [False, False]
    assert [row["batch_tools"] for row in plans] == [[['read_file']], [['grep']]]
    assert [row["batch_call_ids"] for row in plans] == [[['call_read']], [['call_grep']]]
    assert all(fields["turn_id"] == "cli:topic:turn-1" for _event, fields in events)


@pytest.mark.asyncio
async def test_runner_runtime_audit_records_tool_error_outcome():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_bad", name="read_file", arguments={"path": "x"})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=RuntimeError("boom"))
    events: list[tuple[str, dict]] = []

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "inspect"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=16000,
        event_logger=lambda event, fields: events.append((event, fields)),
        fail_on_tool_error=True,
    ))

    end = [fields for event, fields in events if event == "runner.tool.audit.end"][-1]
    assert result.stop_reason == "tool_error"
    assert end["iteration"] == 0
    assert end["call_id"] == "call_bad"
    assert end["status"] == "error"
    assert end["error_type"] == "RuntimeError"
    assert end["error"] == "boom"
    assert "boom" in end["result_preview"]


@pytest.mark.asyncio
async def test_runner_runtime_audit_keeps_full_varying_long_loop_trace():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    call_count = 0

    async def chat_with_retry(**_kwargs):
        nonlocal call_count
        if call_count == 12:
            return LLMResponse(content="done", tool_calls=[])
        call_count += 1
        return LLMResponse(
            content="working",
            tool_calls=[ToolCallRequest(
                id=f"call_{call_count}",
                name="read_file",
                arguments={"path": "large.py", "offset": call_count * 100},
            )],
        )

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=[f"chunk {index}" for index in range(1, 13)])
    events: list[tuple[str, dict]] = []

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "inspect all chunks"}],
        tools=tools,
        model="test-model",
        max_iterations=20,
        max_tool_result_chars=16000,
        event_logger=lambda event, fields: events.append((event, fields)),
        turn_id="cli:topic:long-turn",
    ))

    starts = [fields for event, fields in events if event == "runner.tool.audit.start"]
    ends = [fields for event, fields in events if event == "runner.tool.audit.end"]
    assert result.final_content == "done"
    assert len(starts) == len(ends) == 12
    assert [row["iteration"] for row in starts] == list(range(12))
    assert [row["call_id"] for row in starts] == [f"call_{index}" for index in range(1, 13)]
    assert [row["arguments"]["offset"] for row in starts] == [
        index * 100 for index in range(1, 13)
    ]
    assert [row["result_preview"] for row in ends] == [
        f"chunk {index}" for index in range(1, 13)
    ]


@pytest.mark.asyncio
async def test_runner_runtime_audit_closes_cancelled_tool_span():
    import asyncio

    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute_tool(_name, _arguments):
        tool_started.set()
        await release_tool.wait()
        return "unreachable"

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_wait", name="exec", arguments={"command": "wait"})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = execute_tool
    events: list[tuple[str, dict]] = []
    task = asyncio.create_task(AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "wait"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=16000,
        event_logger=lambda event, fields: events.append((event, fields)),
        turn_id="cli:topic:cancelled-turn",
    )))

    await tool_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    end = [fields for event, fields in events if event == "runner.tool.audit.end"][-1]
    assert end["turn_id"] == "cli:topic:cancelled-turn"
    assert end["iteration"] == 0
    assert end["call_id"] == "call_wait"
    assert end["status"] == "cancelled"
    assert end["duration_ms"] >= 0

@pytest.mark.asyncio
async def test_runner_passes_request_context_to_opted_in_provider():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.supports_request_context = True
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "inspect"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=16000,
        session_key="cli:topic",
        turn_id="turn-123",
    ))

    request_kwargs = provider.chat_with_retry.await_args.kwargs
    assert request_kwargs["request_context"] == {
        "session_key": "cli:topic",
        "turn_id": "turn-123",
    }
