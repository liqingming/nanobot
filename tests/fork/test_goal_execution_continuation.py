"""持续目标接续：阶段回执后收口，真实等待/完成/错误仍停止。"""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.long_task import AwaitUserInputTool, CompleteGoalTool, LongTaskTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.goal_state import sustained_goal_active, sustained_goal_waiting_for_user
from nanobot.session.manager import SessionManager


@pytest.fixture
def goal_runtime(tmp_path):
    sessions = SessionManager(tmp_path / "data")
    session = sessions.get_or_create("cli:goal-test")
    registry = ToolRegistry()
    for cls in (LongTaskTool, CompleteGoalTool, AwaitUserInputTool):
        tool = cls(sessions)
        tool.set_context(RequestContext(channel="cli", chat_id="goal-test", session_key=session.key))
        registry.register(tool)
    provider = MagicMock(spec=LLMProvider)
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "完成已授权任务，不修改范围外文件"}],
        tools=registry, model="test", max_iterations=8, max_tool_result_chars=16000,
        workspace=tmp_path, data_dir=tmp_path / "data", session_key=session.key,
        goal_active_predicate=lambda: (
            sustained_goal_active(session.metadata)
            and not sustained_goal_waiting_for_user(session.metadata)
        ),
        finalize_on_max_iterations=False,
    )
    return session, registry, provider, spec


@pytest.mark.parametrize("ending", ["complete_goal", "await_user_input"])
async def test_receipt_progress_continues_until_explicit_terminal_tool(goal_runtime, ending):
    session, registry, provider, spec = goal_runtime
    await registry.execute("long_task", {"goal": "完成授权任务及最终核验"})
    receipt = {"role": "user", "content": "WORKER_RECEIPT：章节已完成，主任务仍需核验",
               "injected_event": "subagent_result"}
    spec.injection_callback = AsyncMock(side_effect=[[receipt], [], [], []])
    checkpoints = []
    spec.checkpoint_callback = AsyncMock(side_effect=lambda row: checkpoints.append(deepcopy(row)))
    requests = []
    responses = iter([
        LLMResponse(content="等待回执"),
        LLMResponse(content="章节已完成，接下来可继续"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="terminal", name=ending,
            arguments={"recap": "已核验完成"} if ending == "complete_goal"
            else {"reason": "范围外修改需要授权"},
        )]),
        LLMResponse(content="已完成" if ending == "complete_goal" else "请确认范围外修改。"),
    ])

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        return next(responses)

    provider.chat_with_retry = chat
    result = await AgentRunner(provider).run(spec)
    assert result.stop_reason == "completed"
    assert len(requests) == 4
    assert sum(m.get("content") == receipt["content"] for m in result.messages) == 1
    assert any("active sustained goal" in str(m.get("content")) for m in requests[2])
    assert sum(m.get("tool_call_id") == "terminal" for m in result.messages) == 1
    assert sum(c.get("phase") == "final_response" for c in checkpoints) == 3
    assert sustained_goal_active(session.metadata) is (ending == "await_user_input")
    assert sustained_goal_waiting_for_user(session.metadata) is (ending == "await_user_input")


async def test_goal_created_during_run_is_observed_lazily(goal_runtime):
    session, _, provider, spec = goal_runtime
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="start", name="long_task", arguments={"goal": "完成明确授权的任务"},
        )]),
        LLMResponse(content="阶段完成，尚待验收"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="finish", name="complete_goal", arguments={"recap": "验收完成"},
        )]),
        LLMResponse(content="整体完成"),
    ])
    result = await AgentRunner(provider).run(spec)
    assert provider.chat_with_retry.await_count == 4
    assert result.final_content == "整体完成"
    assert not sustained_goal_active(session.metadata)


@pytest.mark.parametrize("response, expected, calls", [
    (LLMResponse(content="模型失败", finish_reason="error"), "error", 1),
    (LLMResponse(content=""), "empty_final_response", 3),
])
async def test_active_goal_does_not_repeat_invalid_response(goal_runtime, response, expected, calls):
    _, registry, provider, spec = goal_runtime
    await registry.execute("long_task", {"goal": "完成授权任务"})
    provider.chat_with_retry = AsyncMock(return_value=response)
    result = await AgentRunner(provider).run(spec)
    assert result.stop_reason == expected
    # 保留原有两次空答与一次最终纠偏，不因持续目标无限重试。
    assert provider.chat_with_retry.await_count == calls
    assert not any("active sustained goal" in str(m.get("content")) for m in result.messages)


async def test_waiting_goal_does_not_resume_on_worker_receipt(goal_runtime):
    session, registry, provider, spec = goal_runtime
    await registry.execute("long_task", {"goal": "完成授权任务"})
    await registry.execute("await_user_input", {"reason": "需要用户明确范围"})
    spec.injection_callback = AsyncMock(side_effect=[
        [{"role": "user", "content": "WORKER_RECEIPT", "injected_event": "subagent_result"}], [],
    ])
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="仍需用户确认范围。"))
    result = await AgentRunner(provider).run(spec)
    assert provider.chat_with_retry.await_count == 2
    assert result.stop_reason == "completed"
    assert sustained_goal_waiting_for_user(session.metadata)
    assert not any("active sustained goal" in str(m.get("content")) for m in result.messages)


async def test_unresolved_todo_cannot_be_completed_by_stage_summary(goal_runtime):
    session, registry, provider, spec = goal_runtime
    await registry.execute("long_task", {"goal": "完成授权任务及最终门禁"})
    session.todos = [{"content": "最终门禁", "status": "pending"}]
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="premature", name="complete_goal", arguments={"recap": "章节完成"},
        )]),
        LLMResponse(content="章节完成"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="wait", name="await_user_input", arguments={"reason": "门禁需范围外授权"},
        )]),
        LLMResponse(content="需要授权后执行最终门禁。"),
    ])
    result = await AgentRunner(provider).run(spec)
    assert result.stop_reason == "completed"
    assert provider.chat_with_retry.await_count == 4
    assert sustained_goal_active(session.metadata)
    assert sustained_goal_waiting_for_user(session.metadata)
    assert session.todos[0]["status"] == "pending"
    assert any("unresolved" in str(m.get("content")) for m in result.messages)


async def test_stream_remains_open_for_goal_continuation(goal_runtime):
    from nanobot.agent.hook import AgentHook

    _, registry, provider, spec = goal_runtime
    await registry.execute("long_task", {"goal": "完成授权任务"})
    endings = []

    class TrackingHook(AgentHook):
        def wants_streaming(self):
            return True

        async def on_stream_end(self, context, *, resuming):
            endings.append(resuming)

    spec.hook = TrackingHook()
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="章节完成，尚待最终核验"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="done", name="complete_goal", arguments={"recap": "最终核验完成"},
        )]),
        LLMResponse(content="全部完成"),
    ])
    result = await AgentRunner(provider).run(spec)
    assert provider.chat_stream_with_retry.await_count == 3
    assert endings[0] is True
    assert endings[-1] is False
    assert result.final_content == "全部完成"


async def test_cancel_during_continuation_is_not_swallowed(goal_runtime):
    import asyncio

    session, registry, provider, spec = goal_runtime
    await registry.execute("long_task", {"goal": "完成授权任务"})
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="阶段完成"), asyncio.CancelledError(),
    ])
    with pytest.raises(asyncio.CancelledError):
        await AgentRunner(provider).run(spec)
    assert provider.chat_with_retry.await_count == 2
    assert sustained_goal_active(session.metadata)
