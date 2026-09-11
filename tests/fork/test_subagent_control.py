"""宿主子任务的创建、终止确认、跨话题拒绝与有界回执。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.context import RequestContext, ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.fork.agent.execution_scope import current_execution_id
from nanobot.fork.agent.subagent_control import SubagentControlState, SubagentControlTool
from nanobot.providers.base import LLMProvider


def manager(tmp_path, **kwargs):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test-model"
    provider.aclose_execution = AsyncMock()
    sm = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus(),
                         max_tool_result_chars=16000, max_concurrent_subagents=3, **kwargs)
    sm._build_tools = MagicMock(return_value=ToolRegistry())
    sm._build_subagent_prompt = MagicMock(return_value="fresh system")
    sm._announce_result = AsyncMock()
    return sm


def control(sm, session="cli:one"):
    tool = SubagentControlTool(sm)
    tool.set_context(RequestContext(channel="cli", chat_id="one", session_key=session))
    return tool


async def test_spawn_is_fresh_and_completed_receipt_is_not_business_success(tmp_path):
    sm = manager(tmp_path)
    seen = []

    async def run(spec):
        seen.append((spec.initial_messages, current_execution_id()))
        return AgentRunResult(final_content="blocked by business gate", messages=[],
                              stop_reason="completed")

    sm.runner.run = run
    spawner = SpawnTool(sm)
    spawner.set_context(RequestContext(channel="cli", chat_id="one", session_key="cli:one"))
    task_text = "仅处理指定目录下的任务，输出报告作为交付，验收要求核对结果并确认通过，不修改其他文件。"
    result = await spawner.execute(task_text)
    tid = result.split("(id: ")[1].split(")")[0]
    ctl = control(sm)
    result = json.loads(await ctl.execute("wait", tid))
    assert result["worker_stopped"] is True
    assert result["business_success"] == "unverified"
    await asyncio.sleep(0)
    assert json.loads(await ctl.execute("status", tid)) == result
    assert tid in {r["task_id"] for r in json.loads(await ctl.execute("list"))["tasks"]}
    assert seen[0][0] == [
        {"role": "system", "content": "fresh system"},
        {"role": "user", "content": task_text},
    ]
    sm.provider.aclose_execution.assert_awaited_once_with(seen[0][1])
    assert sm._task_statuses == {}


async def test_cancel_one_waits_for_cleanup_without_touching_other_session(tmp_path):
    sm = manager(tmp_path)
    began = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    alive = asyncio.Event()

    async def run(spec):
        began.set()
        await alive.wait()
        return AgentRunResult(final_content="done", messages=[], stop_reason="completed")

    async def close(execution_id):
        cleanup_started.set()
        await release_cleanup.wait()

    sm.runner.run = run
    sm.provider.aclose_execution.side_effect = close
    await sm.spawn("first", session_key="cli:one")
    first = next(iter(sm._running_tasks))
    await began.wait()
    await sm.spawn("second", session_key="cli:two")
    second = next(tid for tid in sm._running_tasks if tid != first)
    await asyncio.sleep(0)
    ctl = control(sm)
    foreign = json.loads(await ctl.execute("cancel", second, timeout_s=0))
    assert foreign["state"] == "unknown"
    assert sm._running_tasks[second].cancelling() == 0
    pending = json.loads(await ctl.execute("cancel", first, timeout_s=0))
    assert pending["worker_stopped"] is False
    await cleanup_started.wait()
    await ctl.execute("cancel", first, timeout_s=0)
    assert sm._running_tasks[first].cancelling() == 1
    assert json.loads(await ctl.execute("status", first))["worker_stopped"] is False
    release_cleanup.set()
    stopped = json.loads(await ctl.execute("wait", first))
    assert stopped["worker_stopped"] is True
    assert stopped["stop_reason"] == "cancelled"
    assert not sm._running_tasks[second].done()
    alive.set()
    await asyncio.gather(*sm._running_tasks.values(), return_exceptions=True)


async def test_wait_timeout_and_cancelled_wait_do_not_cancel_worker(tmp_path):
    sm = manager(tmp_path)
    release = asyncio.Event()

    async def run(spec):
        await release.wait()
        return AgentRunResult(final_content="ok", messages=[], stop_reason="completed")

    sm.runner.run = run
    await sm.spawn("task", session_key="cli:one")
    tid, worker = next(iter(sm._running_tasks.items()))
    ctl = control(sm)
    assert json.loads(await ctl.execute("wait", tid, timeout_s=0))["state"] == "running"
    waiter = asyncio.create_task(ctl.execute("wait", tid, timeout_s=30))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    assert not worker.done()
    assert worker.cancelling() == 0
    release.set()
    await worker


async def test_unknown_restart_and_foreign_workspace_never_prove_stopped(tmp_path):
    sm = manager(tmp_path)
    assert json.loads(await control(sm).execute("status", "old-agent"))["worker_stopped"] is False
    state = SubagentControlState()
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    status = SimpleNamespace(label="test", phase="done", iteration=0, stop_reason="completed")
    state.register("id", "cli:one", tmp_path, done, status)
    state.finish("id")
    for session, root in [("cli:two", tmp_path), ("cli:one", tmp_path / "other")]:
        receipt = await state.execute("cancel", session, root, "id", 0)
        assert receipt["state"] == "unknown"
        assert receipt["worker_stopped"] is False
    for i in range(257):
        state.register(str(i), "cli:one", tmp_path, done, status)
        state.finish(str(i))
    assert len(state.finished) == 256
    assert (await state.execute("status", "cli:one", tmp_path, "id", 0))["state"] == "unknown"


def test_control_tool_is_discovered_in_parent_but_not_child(tmp_path):
    sm = manager(tmp_path)
    ctx = ToolContext(config=ToolsConfig(), workspace=str(tmp_path), subagent_manager=sm)
    loader = ToolLoader(test_classes=[SpawnTool, SubagentControlTool])
    parent, child = ToolRegistry(), ToolRegistry()
    loader.load(ctx, parent)
    loader.load(ctx, child, scope="subagent")
    assert parent.has("spawn")
    assert parent.has("subagent_control")
    assert not child.has("subagent_control")
    assert not child.has("spawn")
    assert "without parent conversation history" in SpawnTool(sm).description


async def test_cleanup_failure_does_not_confirm_worker_stopped(tmp_path):
    sm = manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=AgentRunResult(
        final_content="done", messages=[], stop_reason="completed",
    ))
    sm.provider.aclose_execution.side_effect = RuntimeError("cleanup failed")
    await sm.spawn("task", session_key="cli:one")
    tid = next(iter(sm._running_tasks))
    receipt = json.loads(await control(sm).execute("wait", tid))
    assert receipt["state"] == "unknown"
    assert receipt["worker_stopped"] is False


def test_control_tool_is_auto_discovered():
    assert SubagentControlTool in ToolLoader().discover()


async def test_concurrent_host_agents_have_distinct_execution_ids_and_reply_origins(tmp_path):
    sm = manager(tmp_path)
    seen = []
    ready = asyncio.Event()

    async def run(spec):
        seen.append((spec.session_key, current_execution_id()))
        if len(seen) == 2:
            ready.set()
        await ready.wait()
        return AgentRunResult(final_content="done", messages=[], stop_reason="completed")

    sm.runner.run = run
    await sm.spawn("one", session_key="cli:one")
    await sm.spawn("two", session_key="cli:one")
    tasks = list(sm._running_tasks.values())
    await asyncio.gather(*tasks)
    assert len({execution for _, execution in seen}) == 2
    assert {session for session, _ in seen} == {"cli:one"}
    assert {call.args[0] for call in sm.provider.aclose_execution.await_args_list} == {
        execution for _, execution in seen
    }
    assert len(sm._announce_result.await_args_list) == 2
    assert all(call.args[4]["session_key"] == "cli:one"
               for call in sm._announce_result.await_args_list)


async def test_invalid_wait_timeout_is_rejected_before_dispatch_and_can_be_corrected(tmp_path):
    sm = manager(tmp_path)
    sm._task_control.execute = AsyncMock(return_value={"state": "running"})
    ctl = control(sm)
    registry = ToolRegistry()
    registry.register(ctl)
    result = await registry.execute("subagent_control", {
        "action": "wait", "task_id": "worker", "timeout_s": 120,
    })
    assert "Error:" in result
    assert "30" in result
    sm._task_control.execute.assert_not_awaited()
    assert "Error:" in await ctl.execute("wait", "worker", timeout_s=120)
    sm._task_control.execute.assert_not_awaited()
    result = await registry.execute("subagent_control", {
        "action": "wait", "task_id": "worker", "timeout_s": 30,
    })
    assert json.loads(result)["state"] == "running"
    sm._task_control.execute.assert_awaited_once_with("wait", "cli:one", tmp_path, "worker", 30)


@pytest.mark.parametrize("action,state,stopped,returned_id,status,mixed,expected", [
    ("wait", "running", False, "worker", "ok", False, True),
    ("status", "running", False, "worker", "ok", False, False),
    ("list", "running", False, "worker", "ok", False, False),
    ("cancel", "running", False, "worker", "ok", False, False),
    ("wait", "unknown", False, "worker", "ok", False, False),
    ("wait", "stopped", True, "worker", "ok", False, False),
    ("wait", "running", False, "other", "ok", False, False),
    ("wait", "running", False, "worker", "error", False, False),
    ("wait", "running", False, "worker", "ok", True, False),
])
def test_host_wait_requires_successful_explicit_wait_for_matching_worker(
    action, state, stopped, returned_id, status, mixed, expected,
):
    from nanobot.fork.agent.subagent_control import should_wait_for_subagent_result
    from nanobot.providers.base import ToolCallRequest

    calls = [ToolCallRequest(id="w", name="subagent_control", arguments={
        "action": action, "task_id": "worker",
    })]
    results = [json.dumps({"state": state, "worker_stopped": stopped, "task_id": returned_id})]
    events = [{"status": status}]
    if mixed:
        calls.append(ToolCallRequest(id="b", name="exec", arguments={"command": "bind"}))
        results.append("bound")
        events.append({"status": "ok"})
    assert should_wait_for_subagent_result(calls, results, events) is expected
