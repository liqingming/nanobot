"""Regression: TextualTUI.load_session_history must replay tool traces.

This broke when _replay_tool_trace imported ``format_tool_hint`` from
``nanobot.agent.loop`` (which only imports it function-locally as ``_fork_fmt``),
raising ImportError that the surrounding ``except`` silently swallowed — so
every restored tool trace vanished on restart while live runs looked fine.
"""
from __future__ import annotations

import pytest

from nanobot.fork.cli.tui_textual import _TEXTUAL_AVAILABLE, TextualTUI

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE, reason="textual library is not installed"
)


@pytest.mark.asyncio
async def test_load_session_history_replays_tool_trace():
    tui = TextualTUI()
    messages = [
        {"role": "user", "content": "问我几个问题"},
        {
            "role": "assistant",
            "content": "好的，我先确认几个选择：",
            "tool_calls": [
                {"id": "c1", "function": {"name": "ask_user", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "用户已回复"},
    ]

    app = tui._app
    async with app.run_test() as pilot:
        await pilot.pause()
        tui.load_session_history(messages)
        await pilot.pause()
        out = app.query_one("#output")
        rendered = "\n".join(strip.text for strip in out.lines)

    # The assistant text is restored under one turn header …
    assert "确认几个选择" in rendered
    assistant_headers = [
        line for line in rendered.splitlines() if line.startswith("🐈 nanobot 20")
    ]
    assert len(assistant_headers) == 1
    assert rendered.index(assistant_headers[0]) < rendered.index("确认几个选择")
    assert rendered.index("确认几个选择") < rendered.index("ask_user")
    # … and crucially the tool trace is replayed (before the fix the swallowed
    # ImportError left no trace at all).
    assert "ask_user" in rendered


@pytest.mark.asyncio
async def test_load_session_history_replays_readable_multitool_batch() -> None:
    tui = TextualTUI(workspace=".")
    here_python = "@'\nimport inspect\nprint(inspect.signature(len))\n'@ | python -X utf8 -"
    messages = [
        {"role": "user", "content": "检查实现"},
        {
            "role": "assistant",
            "content": "我先检查。",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "exec",
                        "arguments": __import__("json").dumps({"command": here_python}),
                    },
                },
                {
                    "id": "c2",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"nanobot/fork/cli/tui_textual.py"}',
                    },
                },
                {
                    "id": "c3",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"nanobot/fork/cli/tui_textual.py"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "tool", "tool_call_id": "c3", "content": "ok"},
    ]

    async with tui._app.run_test(size=(48, 20)) as pilot:
        tui.load_session_history(messages, workspace=".")
        await pilot.pause()
        out = tui._app.query_one("#output")
        rendered = "\n".join(strip.text for strip in out.lines)

    assert "工具 · 3 项" in rendered
    assert "运行 Python 检查脚本" in rendered
    assert 'read_file("nanobot/fork/cli/tui_textual.py") × 2' in rendered
    assert "@'" not in rendered
    assert "├─" in rendered
    assert "└─" in rendered


@pytest.mark.asyncio
async def test_history_tool_only_assistant_writes_header_before_tools_and_final_reply() -> None:
    tui = TextualTUI(render_markdown=False)
    messages = [
        {
            "role": "user",
            "content": "不需要了，提交",
            "timestamp": "2026-07-21T14:08:42",
        },
        {
            "role": "assistant",
            "content": "",
            "timestamp": "2026-07-21T14:08:43",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"git status --short"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "clean"},
        {
            "role": "assistant",
            "content": "已提交。",
            "timestamp": "2026-07-21T14:09:37",
        },
    ]

    async with tui._app.run_test() as pilot:
        tui.load_session_history(messages)
        await pilot.pause()
        rendered = "\n".join(strip.text for strip in tui._app.query_one("#output").lines)

    assert rendered.count("nanobot 2026-") == 1
    header_index = rendered.index("nanobot 2026-")
    assert header_index < rendered.index("exec") < rendered.index("已提交。")
    assert "2026-07-21 14:08:43" in rendered


@pytest.mark.asyncio
async def test_history_resets_assistant_header_once_per_user_turn() -> None:
    tui = TextualTUI(render_markdown=False)
    messages = [
        {"role": "user", "content": "第一问", "timestamp": "2026-07-21T10:00:00"},
        {"role": "assistant", "content": "第一答", "timestamp": "2026-07-21T10:00:01"},
        {"role": "user", "content": "第二问", "timestamp": "2026-07-21T10:01:00"},
        {
            "role": "assistant",
            "content": "",
            "timestamp": "2026-07-21T10:01:01",
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "第二答", "timestamp": "2026-07-21T10:01:02"},
    ]

    async with tui._app.run_test() as pilot:
        tui.load_session_history(messages)
        await pilot.pause()
        rendered = "\n".join(strip.text for strip in tui._app.query_one("#output").lines)

    assert rendered.count("nanobot 2026-") == 2
    assert rendered.index("第二问") < rendered.index("read_file") < rendered.index("第二答")


@pytest.mark.asyncio
async def test_history_initial_limit_counts_complete_user_turns() -> None:
    tui = TextualTUI(render_markdown=False)
    messages = []
    for turn in range(3):
        messages.extend([
            {"role": "user", "content": f"question-{turn}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"c{turn}",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": f"c{turn}", "content": "ok"},
            {"role": "assistant", "content": f"answer-{turn}"},
        ])

    async with tui._app.run_test() as pilot:
        tui.load_session_history(messages, max_messages=2)
        await pilot.pause()
        rendered = "\n".join(strip.text for strip in tui._app.query_one("#output").lines)

    assert "question-0" not in rendered
    assert "question-1" in rendered and "answer-1" in rendered
    assert "question-2" in rendered and "answer-2" in rendered


@pytest.mark.asyncio
async def test_older_history_page_prepends_and_preserves_view_anchor() -> None:
    tui = TextualTUI(render_markdown=False)
    initial = [
        {"role": "user", "content": "new-question", "_transcript_id": "new-u"},
        {"role": "assistant", "content": "new-answer", "_transcript_id": "new-a"},
    ]
    older = [
        {"role": "user", "content": "old-question", "_transcript_id": "old-u"},
        {"role": "assistant", "content": "old-answer", "_transcript_id": "old-a"},
    ]
    calls = []

    async def loader(cursor):
        calls.append(cursor)
        return older, None, False

    async with tui._app.run_test(size=(60, 12)) as pilot:
        tui.load_session_history(initial)
        out = tui._app.query_one("#output")
        tui.set_history_page_loader(loader, before_offset=123, has_older=True)
        out.scroll_to(y=0, animate=False, immediate=True, force=True)
        old_new_line = out.message_start_line("new-u")
        assert old_new_line is not None

        out._notify_top_if_needed()
        await pilot.pause()
        await pilot.pause()

        new_new_line = out.message_start_line("new-u")
        old_line = out.message_start_line("old-u")
        assert calls == [123]
        assert old_line is not None and new_new_line is not None
        assert old_line < new_new_line
        assert int(out.scroll_offset.y) == new_new_line - old_new_line
        rendered = "\n".join(strip.text for strip in out.lines)
        assert rendered.index("old-question") < rendered.index("new-question")
        assert not tui._history_has_older


@pytest.mark.asyncio
async def test_stale_older_page_is_ignored_after_topic_reset() -> None:
    import asyncio

    tui = TextualTUI(render_markdown=False)
    release = asyncio.Event()

    async def loader(cursor):
        await release.wait()
        return [{"role": "user", "content": "stale-topic"}], None, False

    async with tui._app.run_test() as pilot:
        tui.load_session_history([
            {"role": "user", "content": "current-topic", "_transcript_id": "current"}
        ])
        out = tui._app.query_one("#output")
        tui.set_history_page_loader(loader, before_offset=99, has_older=True)
        out.scroll_to(y=0, animate=False, immediate=True, force=True)
        out._notify_top_if_needed()
        await pilot.pause()

        tui.reset_history()
        tui.load_session_history([
            {"role": "user", "content": "next-topic", "_transcript_id": "next"}
        ])
        release.set()
        await pilot.pause()
        await pilot.pause()

        rendered = "\n".join(strip.text for strip in out.lines)
        assert "next-topic" in rendered
        assert "stale-topic" not in rendered


@pytest.mark.asyncio
async def test_programmatic_scroll_to_top_requests_older_history() -> None:
    tui = TextualTUI(render_markdown=False)
    called = []

    async def loader(cursor):
        called.append(cursor)
        return [], None, False

    async with tui._app.run_test(size=(50, 10)) as pilot:
        tui.load_session_history([
            {"role": "user", "content": "new\n" * 80, "_transcript_id": "new"}
        ])
        out = tui._app.query_one("#output")
        tui.set_history_page_loader(loader, before_offset=7, has_older=True)
        out.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()
        assert out.scroll_offset.y > 0

        out.scroll_to(y=0, animate=False, immediate=True, force=True)
        await pilot.pause()
        await pilot.pause()

        assert called == [7]
