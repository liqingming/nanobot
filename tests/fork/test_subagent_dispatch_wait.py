"""主调度先绑定再等待；真实队列回执、用户插话与取消不丢失。"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import Session


class TrackingQueue(asyncio.Queue):
    def __init__(self):
        super().__init__()
        self.waits = asyncio.Queue()

    async def get(self):
        self.waits.put_nowait(None)
        return await super().get()


class StepTool(Tool):
    def __init__(self, name, action):
        self._name = name
        self.action = action

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Test dispatch step"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return await self.action()


async def runtime(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    session = Session(key="cli:dispatch")
    queue = TrackingQueue()
    loop.runner.run = AsyncMock(return_value=AgentRunResult(final_content="", messages=[]))
    await loop._run_agent_loop(
        [{"role": "user", "content": "task"}], session=session,
        pending_queue=queue, channel="cli", chat_id="dispatch",
    )
    callback = loop.runner.run.await_args.args[0].injection_callback
    return loop, session, queue, provider, callback


@pytest.mark.parametrize("ending", ["receipt", "user_then_receipt", "cancel"])
@pytest.mark.parametrize("wait_method", ["final", "tool"])
async def test_spawn_binds_before_waiting_and_final_wait_is_interruptible(tmp_path, ending, wait_method):
    loop, session, queue, provider, callback = await runtime(tmp_path)
    release = asyncio.Event()
    bound = asyncio.Event()
    workers = []

    async def child():
        await release.wait()
        await queue.put(InboundMessage(
            channel="cli", sender_id="subagent", chat_id="dispatch", content="WORKER_RECEIPT",
            metadata={"injected_event": "subagent_result", "subagent_task_id": "worker"},
        ))

    async def spawn():
        worker = asyncio.create_task(child())
        workers.append(worker)
        loop.subagents._running_tasks["worker"] = worker
        loop.subagents._session_tasks[session.key] = {"worker"}
        return "worker"

    async def bind():
        assert not release.is_set()
        bound.set()
        return "bound"

    tools = ToolRegistry()
    tools.register(StepTool("spawn", spawn))
    tools.register(StepTool("bind", bind))

    async def wait_result():
        return json.dumps({"task_id": "worker", "state": "running", "worker_stopped": False})

    tools.register(StepTool("subagent_control", wait_result))
    responses = [
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="s", name="spawn", arguments={})]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="b", name="bind", arguments={})]),
        (LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="w", name="subagent_control", arguments={"action": "wait", "task_id": "worker"},
        )]) if wait_method == "tool" else LLMResponse(content="阶段完成，等待执行者")),
    ]
    if ending == "user_then_receipt":
        responses.append(LLMResponse(content="已收到用户约束，继续等待"))
    responses.append(LLMResponse(content="全部完成"))
    provider.chat_with_retry = AsyncMock(side_effect=responses)
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "task"}], tools=tools, model="test-model",
        max_iterations=6, max_tool_result_chars=16000, context_strategy="legacy",
        workspace=tmp_path, data_dir=tmp_path / "data", session_key=session.key,
        injection_callback=callback,
    )
    task = asyncio.create_task(AgentRunner(provider).run(spec))
    try:
        # 旧实现会在 spawn 后等待 300 秒，bind 根本不会执行。
        await asyncio.wait_for(bound.wait(), timeout=2)
        await asyncio.wait_for(queue.waits.get(), timeout=2)
        assert provider.chat_with_retry.await_count == 3
        assert not task.done()
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return
        if ending == "user_then_receipt":
            await queue.put(InboundMessage(
                channel="cli", sender_id="user", chat_id="dispatch", content="USER_CONSTRAINT",
            ))
            await asyncio.wait_for(queue.waits.get(), timeout=2)
            assert provider.chat_with_retry.await_count == 4
        release.set()
        result = await asyncio.wait_for(task, timeout=3)
        assert result.stop_reason == "completed"
        assert result.final_content == "全部完成"
        contents = [m.get("content") for m in result.messages]
        assert contents.count("WORKER_RECEIPT") == 1
        if ending == "user_then_receipt":
            assert contents.count("USER_CONSTRAINT") == 1
            assert contents.index("USER_CONSTRAINT") < contents.index("WORKER_RECEIPT")
        assert queue.empty()
        assert [m.get("tool_call_id") for m in result.messages if m.get("role") == "tool"] == (
            ["s", "b", "w"] if wait_method == "tool" else ["s", "b"]
        )
    finally:
        task.cancel()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(task, *workers, return_exceptions=True)


async def test_error_does_not_wait_for_running_worker(tmp_path):
    loop, session, queue, provider, callback = await runtime(tmp_path)
    worker = asyncio.create_task(asyncio.Event().wait())
    loop.subagents._running_tasks["worker"] = worker
    loop.subagents._session_tasks[session.key] = {"worker"}
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="failure", finish_reason="error"))
    try:
        result = await asyncio.wait_for(AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "task"}], tools=ToolRegistry(),
            model="test-model", max_iterations=2, max_tool_result_chars=16000,
            context_strategy="legacy", injection_callback=callback,
        )), timeout=2)
        assert result.stop_reason == "error"
        assert queue.waits.empty()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_explicit_user_wait_does_not_block_on_worker(tmp_path):
    loop, session, queue, _, callback = await runtime(tmp_path)
    session.metadata["goal_state"] = {"status": "active", "awaiting_user_input": True}
    worker = asyncio.create_task(asyncio.Event().wait())
    loop.subagents._running_tasks["worker"] = worker
    loop.subagents._session_tasks[session.key] = {"worker"}
    try:
        assert await asyncio.wait_for(callback(wait_for_subagents=True), timeout=2) == []
        assert queue.waits.empty()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("same_session", [True, False])
async def test_model_error_recovery_reuses_only_current_session_workers(tmp_path, same_session):
    from nanobot.bus.events import OutboundMessage

    loop, session, _, _, _ = await runtime(tmp_path)
    worker = asyncio.create_task(asyncio.Event().wait())
    loop.subagents._running_tasks["worker"] = worker
    loop.subagents._session_tasks[session.key if same_session else "cli:other"] = {"worker"}
    loop.subagents.spawn = AsyncMock()
    try:
        await loop._publish_auto_recovery_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="dispatch", content="task"),
            OutboundMessage(channel="cli", chat_id="dispatch", content="model timed out"),
            session.key,
        )
        recovered = await asyncio.wait_for(loop.bus.consume_inbound(), timeout=2)
        assert ("不要重复派发" in recovered.content) is same_session
        assert ("subagent_control(action='wait'" in recovered.content) is same_session
        assert recovered.metadata["_auto_recover_attempt"] == 1
        assert recovered.session_key == session.key
        assert not worker.done()
        loop.subagents.spawn.assert_not_awaited()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("action,should_warn", [("wait", False), ("status", True), ("cancel", True)])
def test_only_explicit_wait_is_exempt_from_repeated_tool_loop(action, should_warn):
    history = []
    warnings = []
    for n in range(6):
        warnings.append(AgentRunner._record_tool_batch(history, [ToolCallRequest(
            id=str(n), name="subagent_control", arguments={"action": action, "task_id": "worker"},
        )]))
    assert any(warnings) is should_warn
