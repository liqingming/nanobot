import pytest

from nanobot.fork.agent.tools.todo import (
    MAX_TODO_ITEMS,
    TodoWriteTool,
    format_todo_diff,
    format_todos,
)
from nanobot.session.manager import SessionManager
from nanobot.agent.tools.context import RequestContext


@pytest.mark.asyncio
async def test_todo_write_persists_to_session(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    items = [
        {"content": "Plan refactor", "status": "completed"},
        {"content": "Implement TodoTool", "status": "in_progress"},
        {"content": "Write tests", "status": "pending"},
    ]
    result = await tool.execute(items=items)

    assert "Updated todo list" in result
    assert "[x] Plan refactor" in result
    assert "[~] Implement TodoTool" in result
    assert "[ ] Write tests" in result

    session = sessions.get_or_create("cli:test")
    assert [(t["content"], t["status"]) for t in session.todos] == [
        (it["content"], it["status"]) for it in items
    ]


@pytest.mark.asyncio
async def test_todo_write_rejects_multiple_in_progress(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    items = [
        {"content": "A", "status": "in_progress"},
        {"content": "B", "status": "in_progress"},
    ]
    result = await tool.execute(items=items)

    assert result.startswith("Error:")
    assert "at most one" in result
    session = sessions.get_or_create("cli:test")
    assert session.todos == []


@pytest.mark.asyncio
async def test_todo_write_empty_clears_list(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "X", "status": "pending"}])
    result = await tool.execute(items=[])

    assert result == "Todo list cleared."
    session = sessions.get_or_create("cli:test")
    assert session.todos == []


@pytest.mark.asyncio
async def test_todo_overwrite_replaces_full_list(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[
        {"content": "Old A", "status": "pending"},
        {"content": "Old B", "status": "pending"},
    ])
    await tool.execute(items=[
        {"content": "New only", "status": "in_progress"},
    ])

    session = sessions.get_or_create("cli:test")
    assert len(session.todos) == 1
    assert session.todos[0]["content"] == "New only"


def test_todos_persist_across_session_reload(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:test")
    todo = {
        "content": "Persist me",
        "status": "in_progress",
        "created_at": "2026-05-20T10:00:00",
        "started_at": "2026-05-20T10:05:00",
    }
    session.todos = [todo]
    sessions.save(session)

    # Drop cache to force reload from disk
    sessions.invalidate("cli:test")
    reloaded = sessions.get_or_create("cli:test")
    assert reloaded.todos == [todo]


def test_session_clear_resets_todos(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:test")
    session.todos = [{"content": "X", "status": "pending"}]
    session.clear()
    assert session.todos == []


def test_format_todos_handles_unknown_status() -> None:
    rendered = format_todos([{"content": "weird", "status": "bogus"}])
    assert "[?] weird" in rendered


def test_format_todos_empty() -> None:
    assert format_todos([]) == "(no active todos)"


@pytest.mark.asyncio
async def test_todo_rejects_completed_to_pending(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "task", "status": "completed"}])
    result = await tool.execute(items=[{"content": "task", "status": "pending"}])

    assert result.startswith("Error:")
    assert "completed" in result and "not allowed" in result
    session = sessions.get_or_create("cli:test")
    assert session.todos[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_todo_rejects_completed_to_in_progress(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "task", "status": "completed"}])
    result = await tool.execute(items=[{"content": "task", "status": "in_progress"}])

    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_todo_rejects_empty_content(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    result = await tool.execute(items=[{"content": "   ", "status": "pending"}])

    assert result.startswith("Error:")
    assert "empty content" in result


@pytest.mark.asyncio
async def test_todo_rejects_duplicate_content(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    result = await tool.execute(items=[
        {"content": "dup", "status": "pending"},
        {"content": "dup", "status": "in_progress"},
    ])

    assert result.startswith("Error:")
    assert "duplicate" in result


@pytest.mark.asyncio
async def test_todo_rejects_oversized_list(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    items = [{"content": f"task {i}", "status": "pending"} for i in range(MAX_TODO_ITEMS + 1)]
    result = await tool.execute(items=items)

    assert result.startswith("Error:")
    assert "exceeds limit" in result


@pytest.mark.asyncio
async def test_todo_timestamps_initialized_on_create(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "a", "status": "pending"}])
    session = sessions.get_or_create("cli:test")
    assert "created_at" in session.todos[0]
    assert "started_at" not in session.todos[0]
    assert "completed_at" not in session.todos[0]


@pytest.mark.asyncio
async def test_todo_started_at_set_on_first_in_progress(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "a", "status": "pending"}])
    await tool.execute(items=[{"content": "a", "status": "in_progress"}])
    session = sessions.get_or_create("cli:test")
    assert "started_at" in session.todos[0]
    started = session.todos[0]["started_at"]

    # Subsequent updates preserve the original started_at
    await tool.execute(items=[{"content": "a", "status": "in_progress"}])
    session = sessions.get_or_create("cli:test")
    assert session.todos[0]["started_at"] == started


@pytest.mark.asyncio
async def test_todo_completed_at_set_and_preserved(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "a", "status": "in_progress"}])
    await tool.execute(items=[{"content": "a", "status": "completed"}])
    session = sessions.get_or_create("cli:test")
    assert "completed_at" in session.todos[0]
    completed = session.todos[0]["completed_at"]

    # Re-asserting completed status keeps original timestamp
    await tool.execute(items=[{"content": "a", "status": "completed"}])
    session = sessions.get_or_create("cli:test")
    assert session.todos[0]["completed_at"] == completed


@pytest.mark.asyncio
async def test_todo_created_at_preserved_across_updates(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "a", "status": "pending"}])
    session = sessions.get_or_create("cli:test")
    created = session.todos[0]["created_at"]

    await tool.execute(items=[{"content": "a", "status": "in_progress"}])
    session = sessions.get_or_create("cli:test")
    assert session.todos[0]["created_at"] == created


@pytest.mark.asyncio
async def test_todo_allows_pending_to_completed_directly(tmp_path) -> None:
    """Skipping in_progress is allowed — useful for trivially-fast steps."""
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "a", "status": "pending"}])
    result = await tool.execute(items=[{"content": "a", "status": "completed"}])
    assert not result.startswith("Error:")
    session = sessions.get_or_create("cli:test")
    assert session.todos[0]["status"] == "completed"
    assert "completed_at" in session.todos[0]


@pytest.mark.asyncio
async def test_todo_in_progress_to_pending_revert_allowed(tmp_path) -> None:
    """Reverting an active task back to pending is allowed (e.g. blocked work)."""
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    await tool.execute(items=[{"content": "a", "status": "in_progress"}])
    result = await tool.execute(items=[{"content": "a", "status": "pending"}])
    assert not result.startswith("Error:")
    session = sessions.get_or_create("cli:test")
    assert session.todos[0]["status"] == "pending"
    # started_at should still be present from the original transition
    assert "started_at" in session.todos[0]


# ── format_todo_diff ──────────────────────────────────────────────────────


def test_diff_first_creation_shows_full_plan() -> None:
    curr = [
        {"content": "step 1", "status": "in_progress"},
        {"content": "step 2", "status": "pending"},
        {"content": "step 3", "status": "pending"},
    ]
    result = format_todo_diff([], curr)
    assert result is not None
    assert "📋 计划已制定 (3 步)" in result
    assert "[~] step 1" in result
    assert "[ ] step 2" in result
    assert "[ ] step 3" in result


def test_diff_clear_returns_cleared_message() -> None:
    prev = [{"content": "x", "status": "completed"}]
    assert format_todo_diff(prev, []) == "📋 计划已清空"


def test_diff_empty_to_empty_returns_none() -> None:
    assert format_todo_diff([], []) is None


def test_diff_no_change_returns_none() -> None:
    items = [{"content": "x", "status": "in_progress"}]
    assert format_todo_diff(items, items) is None


def test_diff_completion_shows_done_marker() -> None:
    prev = [{"content": "step 1", "status": "in_progress"}, {"content": "step 2", "status": "pending"}]
    curr = [{"content": "step 1", "status": "completed"}, {"content": "step 2", "status": "in_progress"}]
    result = format_todo_diff(prev, curr)
    assert result is not None
    assert "✅ step 1" in result
    assert "⚡ step 2" in result
    assert "📊 进度: 1/2" in result


def test_diff_all_completed_shows_summary_not_per_item() -> None:
    prev = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
    ]
    curr = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
    ]
    result = format_todo_diff(prev, curr)
    assert result == "✅ 全部 2 步完成"


def test_diff_all_completed_already_returns_none() -> None:
    """If everything was already completed and still is, no message."""
    items = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
    ]
    assert format_todo_diff(items, items) is None


def test_diff_added_item() -> None:
    prev = [{"content": "a", "status": "completed"}]
    curr = [{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]
    result = format_todo_diff(prev, curr)
    assert result is not None
    assert "➕ 新增: b" in result
    assert "📊 进度: 1/2" in result


def test_diff_removed_item() -> None:
    prev = [{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]
    curr = [{"content": "a", "status": "pending"}]
    result = format_todo_diff(prev, curr)
    assert result is not None
    assert "➖ 删除: b" in result


def test_diff_revert_to_pending() -> None:
    prev = [{"content": "a", "status": "in_progress"}]
    curr = [{"content": "a", "status": "pending"}]
    result = format_todo_diff(prev, curr)
    assert result is not None
    assert "↩ 重置: a" in result


def test_diff_multiple_changes_in_one_call() -> None:
    """LLM batched two transitions in one todo_write call."""
    prev = [
        {"content": "a", "status": "in_progress"},
        {"content": "b", "status": "pending"},
        {"content": "c", "status": "pending"},
    ]
    curr = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
        {"content": "c", "status": "in_progress"},
    ]
    result = format_todo_diff(prev, curr)
    assert result is not None
    assert "✅ a" in result
    assert "✅ b" in result
    assert "⚡ c" in result
    assert "📊 进度: 2/3" in result


# ── Live diff broadcasting via MessageBus ─────────────────────────────────


@pytest.mark.asyncio
async def test_tool_pushes_diff_to_bus_on_each_call(tmp_path) -> None:
    """Each todo_write call must publish a system message with the diff so the
    TUI shows progress immediately (not only at turn end)."""
    import asyncio
    from nanobot.bus.queue import MessageBus

    sessions = SessionManager(tmp_path)
    bus = MessageBus()
    tool = TodoWriteTool(sessions=sessions, bus=bus)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    # First call: empty → 2 items → expect a "first creation" diff
    await tool.execute(items=[
        {"content": "step 1", "status": "in_progress"},
        {"content": "step 2", "status": "pending"},
    ])
    msg1 = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    assert msg1.channel == "cli"
    assert msg1.chat_id == "test"
    assert msg1.metadata.get("_system_message") is True
    assert "📋 计划已制定 (2 步)" in msg1.content

    # Second call: complete step 1, start step 2 → expect per-item diff
    await tool.execute(items=[
        {"content": "step 1", "status": "completed"},
        {"content": "step 2", "status": "in_progress"},
    ])
    msg2 = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    assert "✅ step 1" in msg2.content
    assert "⚡ step 2" in msg2.content
    assert "📊 进度: 1/2" in msg2.content


@pytest.mark.asyncio
async def test_tool_works_without_bus(tmp_path) -> None:
    """Bus is optional — passing None must not break execution."""
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions, bus=None)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))
    result = await tool.execute(items=[{"content": "x", "status": "pending"}])
    assert result.startswith("Updated todo list")


# ── summarize_result for tool-trace UI ────────────────────────────────────


def test_summary_in_progress_list(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    args = {"items": [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ]}
    out = tool.summarize_result(args, "Updated todo list:\n[x] a\n[~] b\n[ ] c")
    assert out == "3 todos · 1/3 done"


def test_summary_all_done(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    args = {"items": [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
    ]}
    out = tool.summarize_result(args, "Updated todo list:\n[x] a\n[x] b")
    # all-done state never carries the +N badge (would be redundant)
    assert out == "all 2 done"


def test_summary_cleared(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    out = tool.summarize_result({"items": []}, "Todo list cleared.")
    assert out == "cleared"


def test_summary_error_returns_error(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    out = tool.summarize_result({}, "Error: at most one todo may be in_progress")
    assert out.startswith("Error")


@pytest.mark.asyncio
async def test_summary_delta_appears_after_completion(tmp_path) -> None:
    """+N suffix should appear when a new execute() completes more todos."""
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    items_1 = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ]
    result_1 = await tool.execute(items=items_1)
    s1 = tool.summarize_result({"items": items_1}, result_1)
    # First call: prev_done=0, new_done=1, delta=+1
    assert s1 == "3 todos · 1/3 done · +1"

    items_2 = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
        {"content": "c", "status": "in_progress"},
    ]
    result_2 = await tool.execute(items=items_2)
    s2 = tool.summarize_result({"items": items_2}, result_2)
    # Second call: prev_done=1, new_done=2, delta=+1
    assert s2 == "3 todos · 2/3 done · +1"

    items_3 = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
        {"content": "c", "status": "completed"},
    ]
    result_3 = await tool.execute(items=items_3)
    s3 = tool.summarize_result({"items": items_3}, result_3)
    # All-done state suppresses the +N suffix (it'd be redundant — "all N done"
    # already conveys completion). Mid-progress states still show delta.
    assert s3 == "all 3 done"


@pytest.mark.asyncio
async def test_summary_no_delta_when_unchanged(tmp_path) -> None:
    """If no new todos completed this call, no +N suffix shown."""
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    items = [
        {"content": "a", "status": "in_progress"},
        {"content": "b", "status": "pending"},
    ]
    await tool.execute(items=items)
    s = tool.summarize_result({"items": items}, "Updated...")
    # 0 newly completed → no delta suffix
    assert "+" not in s
    assert s == "2 todos · 0/2 done"


@pytest.mark.asyncio
async def test_summary_delta_isolated_per_session(tmp_path) -> None:
    """Two sessions' delta counters must not interfere."""
    sessions = SessionManager(tmp_path)
    tool = TodoWriteTool(sessions=sessions)

    tool.set_context(RequestContext(channel="cli", chat_id="session_a"))
    await tool.execute(items=[{"content": "x", "status": "completed"}])

    tool.set_context(RequestContext(channel="cli", chat_id="session_b"))
    items_b = [{"content": "y", "status": "completed"}]
    result_b = await tool.execute(items=items_b)
    s = tool.summarize_result({"items": items_b}, result_b)
    # session_b's first all-done state — delta would be +1 but all-done
    # suppresses the +N suffix (verifies session_b's counter is independent
    # of session_a — if it weren't, this would crash or show wrong total).
    assert s == "all 1 done"


@pytest.mark.asyncio
async def test_tool_does_not_push_when_no_change(tmp_path) -> None:
    """If two identical todo_write calls happen, the second emits nothing."""
    import asyncio
    from nanobot.bus.queue import MessageBus

    sessions = SessionManager(tmp_path)
    bus = MessageBus()
    tool = TodoWriteTool(sessions=sessions, bus=bus)
    tool.set_context(RequestContext(channel="cli", chat_id="test"))

    items = [{"content": "x", "status": "in_progress"}]
    await tool.execute(items=items)
    await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

    await tool.execute(items=items)
    # Second call: identical items → format_todo_diff returns None → no message
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(bus.consume_outbound(), timeout=0.2)
