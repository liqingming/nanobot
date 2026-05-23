"""Tests for AskUserTool — bus-based interactive question dispatcher."""
from __future__ import annotations

import asyncio
import json

import pytest

from nanobot.fork.agent.tools.ask_user import AskUserTool, deliver_reply
from nanobot.bus.queue import MessageBus


@pytest.mark.asyncio
async def test_non_cli_channel_returns_unsupported_error() -> None:
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("telegram", "12345")

    out = await tool.execute(questions=[
        {"question": "pick?", "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]},
    ])
    data = json.loads(out)
    assert "error" in data
    assert "not supported" in data["error"].lower()
    assert data["channel"] == "telegram"


@pytest.mark.asyncio
async def test_missing_bus_returns_error() -> None:
    tool = AskUserTool(bus=None)
    tool.set_context("cli", "test")
    out = await tool.execute(questions=[
        {"question": "x?", "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]},
    ])
    data = json.loads(out)
    assert "error" in data


@pytest.mark.asyncio
async def test_empty_questions_returns_error() -> None:
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("cli", "test")
    out = await tool.execute(questions=[])
    data = json.loads(out)
    assert "error" in data


@pytest.mark.asyncio
async def test_publishes_ask_user_message_to_bus() -> None:
    """Tool should immediately push a question to the bus with a correlation id."""
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("cli", "test")

    questions = [
        {"question": "yes?", "options": [{"label": "y", "description": ""}, {"label": "n", "description": ""}]},
    ]
    # Run execute in background and inspect the bus
    task = asyncio.create_task(tool.execute(questions=questions))
    await asyncio.sleep(0.05)
    try:
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Tool never published the _ask_user message")

    assert msg.metadata.get("_ask_user") is True
    cid = msg.metadata.get("_ask_user_id")
    assert isinstance(cid, str) and len(cid) > 0
    assert msg.metadata.get("_ask_user_questions") == questions
    assert msg.channel == "cli"
    assert msg.chat_id == "test"

    # Resolve the awaiting future so the task finishes cleanly
    deliver_reply(cid, {"yes?": "y"})
    out = await asyncio.wait_for(task, timeout=1.0)
    data = json.loads(out)
    assert data == {"answers": {"yes?": "y"}}


@pytest.mark.asyncio
async def test_deliver_reply_resolves_pending_future() -> None:
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("cli", "test")
    questions = [
        {"question": "color?", "options": [
            {"label": "red", "description": ""},
            {"label": "blue", "description": ""},
        ]},
    ]
    task = asyncio.create_task(tool.execute(questions=questions))
    await asyncio.sleep(0.05)
    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    cid = msg.metadata["_ask_user_id"]

    deliver_reply(cid, {"color?": "red"})
    out = await asyncio.wait_for(task, timeout=1.0)
    assert json.loads(out)["answers"]["color?"] == "red"


@pytest.mark.asyncio
async def test_cancellation_via_deliver_reply() -> None:
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("cli", "test")
    task = asyncio.create_task(tool.execute(questions=[
        {"question": "?", "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]},
    ]))
    await asyncio.sleep(0.05)
    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    cid = msg.metadata["_ask_user_id"]
    deliver_reply(cid, None, cancelled=True)
    out = await asyncio.wait_for(task, timeout=1.0)
    assert json.loads(out) == {"cancelled": True}


@pytest.mark.asyncio
async def test_stale_reply_after_resolution_is_ignored() -> None:
    """Calling deliver_reply twice (e.g. late TUI reply after timeout) is safe."""
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("cli", "test")
    task = asyncio.create_task(tool.execute(questions=[
        {"question": "?", "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]},
    ]))
    await asyncio.sleep(0.05)
    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    cid = msg.metadata["_ask_user_id"]
    deliver_reply(cid, {"?": "a"})
    out = await asyncio.wait_for(task, timeout=1.0)
    assert json.loads(out)["answers"]["?"] == "a"
    # Second delivery should be a no-op (no exception)
    deliver_reply(cid, {"?": "b"})
    deliver_reply("nonexistent-id", {"x": "y"})


@pytest.mark.asyncio
async def test_timeout_returns_error(monkeypatch) -> None:
    bus = MessageBus()
    tool = AskUserTool(bus=bus)
    tool.set_context("cli", "test")
    monkeypatch.setattr(AskUserTool, "DEFAULT_TIMEOUT_SEC", 0.1)
    out = await tool.execute(questions=[
        {"question": "?", "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]},
    ])
    data = json.loads(out)
    assert "error" in data
    assert "timeout" in data["error"].lower()


def test_summarize_result_answers() -> None:
    tool = AskUserTool()
    args = {"questions": [{"question": "a?"}, {"question": "b?"}]}
    out = tool.summarize_result(args, json.dumps({"answers": {"a?": "yes", "b?": "no"}}))
    assert out == "answered 2/2"


def test_summarize_result_cancelled() -> None:
    tool = AskUserTool()
    out = tool.summarize_result({}, json.dumps({"cancelled": True}))
    assert out == "cancelled"


def test_summarize_result_error() -> None:
    tool = AskUserTool()
    out = tool.summarize_result({}, json.dumps({"error": "channel not supported"}))
    assert out.startswith("error:")
