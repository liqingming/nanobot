"""Tests for sustained goal tools (`long_task`, `complete_goal`)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_artifacts import CONTEXT_STATE_KEY, ContextState
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.long_task import (
    AwaitUserInputTool,
    CompleteGoalTool,
    LongTaskTool,
)
from nanobot.bus.outbound_events import GoalStateSyncEvent
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import RuntimeEventBus
from nanobot.session.goal_state import GOAL_STATE_KEY
from nanobot.session.manager import SessionManager
from nanobot.session.resume_state import AMBIGUOUS_RESUME_META_KEY
from nanobot.session.webui_turns import WebuiTurnCoordinator


def _tools(sm: SessionManager) -> tuple[LongTaskTool, CompleteGoalTool]:
    lt = LongTaskTool(sessions=sm)
    cg = CompleteGoalTool(sessions=sm)
    rc = RequestContext(
        channel="websocket",
        chat_id="c1",
        session_key="websocket:c1",
        metadata={},
    )
    lt.set_context(rc)
    cg.set_context(rc)
    return lt, cg


@pytest.mark.asyncio
async def test_long_task_records_goal_metadata(tmp_path):
    sm = SessionManager(tmp_path)
    lt, _cg = _tools(sm)

    out = await lt.execute(goal="Do the thing", ui_summary="thing")
    assert "Goal recorded" in out

    sess = sm.get_or_create("websocket:c1")
    blob = sess.metadata.get(GOAL_STATE_KEY)
    assert isinstance(blob, dict)
    assert blob["status"] == "active"
    assert blob["objective"] == "Do the thing"
    assert blob["ui_summary"] == "thing"


@pytest.mark.asyncio
async def test_long_task_rejects_second_active_goal(tmp_path):
    sm = SessionManager(tmp_path)
    lt, _cg = _tools(sm)

    await lt.execute(goal="First")
    out = await lt.execute(goal="Second")
    assert "already active" in out


@pytest.mark.asyncio
async def test_complete_goal_closes_active_goal(tmp_path):
    sm = SessionManager(tmp_path)
    lt, cg = _tools(sm)

    await lt.execute(goal="X")
    out = await cg.execute(recap="Done.")
    assert "marked complete" in out

    sess = sm.get_or_create("websocket:c1")
    blob = sess.metadata.get(GOAL_STATE_KEY)
    assert blob["status"] == "completed"
    assert blob["recap"] == "Done."
    assert sess.metadata["_completed_goal_needs_compaction"] is True
    context_state = ContextState.from_metadata(sess.metadata)
    assert len(context_state.completion_stubs) == 1
    assert context_state.completion_stubs[0].result == "Done."
    assert CONTEXT_STATE_KEY in sess.metadata


@pytest.mark.asyncio
async def test_goal_tools_keep_request_context_per_task(tmp_path):
    sm = SessionManager(tmp_path)
    lt = LongTaskTool(sessions=sm)
    cg = CompleteGoalTool(sessions=sm)
    ctx_a = RequestContext(channel="websocket", chat_id="a", session_key="websocket:a")
    ctx_b = RequestContext(channel="websocket", chat_id="b", session_key="websocket:b")

    lt.set_context(ctx_a)
    task_a = asyncio.create_task(lt.execute(goal="Goal A"))
    lt.set_context(ctx_b)
    task_b = asyncio.create_task(lt.execute(goal="Goal B"))
    await asyncio.gather(task_a, task_b)

    assert sm.get_or_create("websocket:a").metadata[GOAL_STATE_KEY]["objective"] == "Goal A"
    assert sm.get_or_create("websocket:b").metadata[GOAL_STATE_KEY]["objective"] == "Goal B"

    cg.set_context(ctx_a)
    done_a = asyncio.create_task(cg.execute(recap="Done A"))
    cg.set_context(ctx_b)
    done_b = asyncio.create_task(cg.execute(recap="Done B"))
    await asyncio.gather(done_a, done_b)

    assert sm.get_or_create("websocket:a").metadata[GOAL_STATE_KEY]["recap"] == "Done A"
    assert sm.get_or_create("websocket:b").metadata[GOAL_STATE_KEY]["recap"] == "Done B"


@pytest.mark.asyncio
async def test_goal_tools_context_isolated_across_tool_types(tmp_path):
    """LongTaskTool and CompleteGoalTool must not share routing context."""
    sm = SessionManager(tmp_path)
    lt = LongTaskTool(sessions=sm)
    cg = CompleteGoalTool(sessions=sm)
    ctx = RequestContext(channel="websocket", chat_id="a", session_key="websocket:a")

    lt.set_context(ctx)
    assert cg._request_ctx.get() is None

    cg.set_context(ctx)
    assert lt._request_ctx.get() is ctx
    assert cg._request_ctx.get() is ctx


@pytest.mark.asyncio
async def test_long_task_publishes_goal_state_ws_after_save(tmp_path):
    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    runtime_events = RuntimeEventBus()
    sm = SessionManager(tmp_path)
    WebuiTurnCoordinator(
        bus=bus,
        sessions=sm,
        schedule_background=lambda _coro: None,
    ).subscribe(runtime_events)
    lt = LongTaskTool(sessions=sm, runtime_events=runtime_events)
    rc = RequestContext(
        channel="websocket",
        chat_id="chat-99",
        session_key="websocket:chat-99",
        metadata={},
    )
    lt.set_context(rc)

    await lt.execute(goal="Objective alpha", ui_summary="alpha")

    bus.publish_outbound.assert_awaited_once()
    call = bus.publish_outbound.await_args.args[0]
    assert call.channel == "websocket"
    assert call.chat_id == "chat-99"
    assert isinstance(call.event, GoalStateSyncEvent)
    assert call.event.goal_state == {
        "active": True,
        "ui_summary": "alpha",
        "objective": "Objective alpha",
    }


@pytest.mark.asyncio
async def test_complete_goal_publishes_inactive_goal_state_ws(tmp_path):
    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    runtime_events = RuntimeEventBus()
    sm = SessionManager(tmp_path)
    WebuiTurnCoordinator(
        bus=bus,
        sessions=sm,
        schedule_background=lambda _coro: None,
    ).subscribe(runtime_events)
    lt = LongTaskTool(sessions=sm, runtime_events=runtime_events)
    cg = CompleteGoalTool(sessions=sm, runtime_events=runtime_events)
    rc = RequestContext(
        channel="websocket",
        chat_id="chat-z",
        session_key="websocket:chat-z",
        metadata={},
    )
    lt.set_context(rc)
    await lt.execute(goal="X")

    bus.publish_outbound.reset_mock()
    cg.set_context(rc)
    await cg.execute(recap="Done.")

    bus.publish_outbound.assert_awaited_once()
    call = bus.publish_outbound.await_args.args[0]
    assert isinstance(call.event, GoalStateSyncEvent)
    assert call.event.goal_state == {"active": False}


@pytest.mark.asyncio
async def test_complete_goal_without_active_is_noop_message(tmp_path):
    sm = SessionManager(tmp_path)
    _lt, cg = _tools(sm)

    out = await cg.execute(recap="n/a")
    assert "No active" in out


@pytest.mark.asyncio
async def test_long_task_skips_ws_publish_without_bus(tmp_path):
    sm = SessionManager(tmp_path)
    lt, _cg = _tools(sm)
    out = await lt.execute(goal="Solo", ui_summary="s")
    assert "Goal recorded" in out


@pytest.mark.asyncio
async def test_long_task_and_complete_goal_registered(tmp_path):
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    lt = loop.tools.get("long_task")
    cg = loop.tools.get("complete_goal")
    wait = loop.tools.get("await_user_input")
    assert lt is not None and lt.name == "long_task"
    assert cg is not None and cg.name == "complete_goal"
    assert wait is not None and wait.name == "await_user_input"
    assert isinstance(wait, AwaitUserInputTool)
    assert wait._sessions is loop.sessions


@pytest.mark.asyncio
async def test_await_user_input_marks_active_goal_without_completing_it(tmp_path):
    sm = SessionManager(tmp_path)
    lt = LongTaskTool(sessions=sm)
    wait = AwaitUserInputTool(sessions=sm)
    rc = RequestContext(channel="websocket", chat_id="c1", session_key="websocket:c1")
    lt.set_context(rc)
    wait.set_context(rc)
    await lt.execute(goal="Ship phase one")
    out = await wait.execute(reason="Reply start phase one")
    assert "paused" in out
    blob = sm.get_or_create("websocket:c1").metadata[GOAL_STATE_KEY]
    assert blob["status"] == "active"
    assert blob["awaiting_user_input"] is True
    assert blob["awaiting_user_input_reason"] == "Reply start phase one"


@pytest.mark.asyncio
async def test_ambiguous_resume_cannot_start_guessed_goal_after_completed_goal(tmp_path):
    sm = SessionManager(tmp_path)
    lt, cg = _tools(sm)
    await lt.execute(goal="Build the overview")
    await cg.execute(recap="Overview completed")

    lt.set_context(RequestContext(
        channel="websocket",
        chat_id="c1",
        session_key="websocket:c1",
        metadata={AMBIGUOUS_RESUME_META_KEY: True},
    ))
    out = await lt.execute(goal="Guess that BinaryAddressableProvider is next")

    assert "ambiguous resume request" in out
    assert sm.get_or_create("websocket:c1").metadata[GOAL_STATE_KEY]["status"] == "completed"
    assert "BinaryAddressableProvider" not in str(
        sm.get_or_create("websocket:c1").metadata[GOAL_STATE_KEY]
    )


@pytest.mark.asyncio
async def test_named_new_goal_is_allowed_after_completed_goal(tmp_path):
    sm = SessionManager(tmp_path)
    lt, cg = _tools(sm)
    await lt.execute(goal="Old goal")
    await cg.execute(recap="Done")

    out = await lt.execute(goal="Explicitly requested new goal")

    assert "Goal recorded" in out
    assert sm.get_or_create("websocket:c1").metadata[GOAL_STATE_KEY]["objective"] == (
        "Explicitly requested new goal"
    )


@pytest.mark.asyncio
async def test_complete_goal_rejects_unresolved_todos(tmp_path):
    sm = SessionManager(tmp_path)
    lt, cg = _tools(sm)
    await lt.execute(goal="Finish all tracked work")
    session = sm.get_or_create("websocket:c1")
    session.todos = [{"content": "Write final diagram", "status": "in_progress"}]
    sm.save(session)

    out = await cg.execute(recap="Done")

    assert "todos remain unresolved" in out
    assert sm.get_or_create("websocket:c1").metadata[GOAL_STATE_KEY]["status"] == "active"


@pytest.mark.asyncio
async def test_ambiguous_resume_guard_survives_search_failure(tmp_path):
    sm = SessionManager(tmp_path)
    lt, cg = _tools(sm)
    await lt.execute(goal="Completed source study")
    await cg.execute(recap="Study complete")

    # A failed history lookup must not authorize a guessed replacement goal;
    # the guard is attached to the inbound request context, not search output.
    lt.set_context(RequestContext(
        channel="websocket",
        chat_id="c1",
        session_key="websocket:c1",
        metadata={AMBIGUOUS_RESUME_META_KEY: True, "history_search_result": "not found"},
    ))
    out = await lt.execute(goal="Guessed next study topic")

    assert "Do not call long_task" in out
    assert sm.get_or_create("websocket:c1").metadata[GOAL_STATE_KEY]["objective"] == (
        "Completed source study"
    )
