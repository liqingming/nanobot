"""Topic-local persistent reading-position bookmarks for the Textual TUI."""
from __future__ import annotations

import json

import pytest
from rich.cells import cell_len
from rich.segment import Segment
from textual.strip import Strip

from nanobot.fork.cli.tui_textual import (
    _TEXTUAL_AVAILABLE,
    TextualTUI,
    _OutputLog,
)

pytestmark = pytest.mark.skipif(not _TEXTUAL_AVAILABLE, reason="textual not installed")


MESSAGES = [
    {"role": "user", "content": "第一条用户消息", "timestamp": "2026-07-21T09:00:00"},
    {"role": "assistant", "content": "第一条助手回复", "timestamp": "2026-07-21T09:00:01"},
    {"role": "user", "content": "第二条用户消息", "timestamp": "2026-07-21T09:00:02"},
    {"role": "assistant", "content": "第二条助手回复", "timestamp": "2026-07-21T09:00:03"},
]
LONG_MESSAGES = [
    {
        "role": "assistant",
        "_transcript_id": "assistant-long-stable-id",
        "content": "\n\n".join(
            f"第{index}阶段：执行系统流程步骤 {index}，校验输入并保存该阶段结果。"
            for index in range(1, 41)
        ),
        "timestamp": "2026-07-21T09:00:00",
    }
]


def _make_tui(tmp_path, topic: str = "cli:topic-a") -> TextualTUI:
    tui = TextualTUI(render_markdown=False, history_file=str(tmp_path / "history"))
    tui.set_topic("topic-a")
    tui.set_input_history_topic(topic)
    return tui


def _bookmark_current_line(tui: TextualTUI, output: _OutputLog, line: int) -> dict:
    output.scroll_to(y=line, animate=False, immediate=True, force=True)
    tui.toggle_bookmark_at_view()
    assert len(tui._bookmarks) >= 1
    return list(tui._bookmarks.values())[-1]


@pytest.mark.asyncio
async def test_bookmark_persists_and_restores_same_reading_position(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        target = output.message_blocks()[0]
        start, end = output.message_line_range(target["id"])
        line = min(start + 4, end)
        bookmark = _bookmark_current_line(tui, output, line)

        assert bookmark["message_id"] == target["id"]
        assert bookmark["char_offset"] > 0
        payload = json.loads(tui._bookmark_file.read_text(encoding="utf-8"))
        assert payload["schema"] == 2
        assert payload["bookmarks"][0]["bookmark_id"] == bookmark["bookmark_id"]

    restored = _make_tui(tmp_path)
    async with restored._app.run_test():
        restored.load_session_history(LONG_MESSAGES)
        output = restored._app.query_one("#output", _OutputLog)
        restored_bookmark = restored._bookmarks[bookmark["bookmark_id"]]
        restored_line = output.bookmark_line(restored_bookmark)
        assert restored_line is not None
        assert restored.jump_to_bookmark(bookmark["bookmark_id"])
        assert int(output.scroll_offset.y) <= restored_line
        assert output._bookmark_highlight == (restored_line, restored_line)


@pytest.mark.asyncio
async def test_bookmark_prefers_mouse_selection_endpoint_over_viewport_top(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        target_line = min(start + 6, end)
        output.scroll_to(y=start, animate=False, immediate=True, force=True)
        output._sel_start = (min(start + 2, end), 0)
        output._sel_end = (target_line, 5)
        output._sel_moved = True

        tui.toggle_bookmark_at_view()

        bookmark = next(iter(tui._bookmarks.values()))
        assert output.bookmark_line(bookmark) == target_line
        assert output._selection_points() is None


@pytest.mark.asyncio
async def test_same_long_message_supports_multiple_position_bookmarks(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])

        first = _bookmark_current_line(tui, output, min(start + 2, end))
        second = _bookmark_current_line(tui, output, min(start + 6, end))

        assert first["message_id"] == second["message_id"]
        assert first["bookmark_id"] != second["bookmark_id"]
        assert output.bookmark_line(first) != output.bookmark_line(second)
        assert len(tui._bookmarks) == 2


@pytest.mark.asyncio
async def test_position_bookmark_survives_output_reflow(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        bookmark = _bookmark_current_line(tui, output, min(start + 4, end))
        before = output.bookmark_line(bookmark)

        output._reflow(24)
        after = output.bookmark_line(bookmark)

        assert before is not None and after is not None
        assert bookmark["char_offset"] > 0
        assert tui.jump_to_bookmark(bookmark["bookmark_id"])
        assert output._bookmark_highlight == (after, after)


@pytest.mark.asyncio
async def test_previous_bookmark_uses_position_and_wraps(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        first = _bookmark_current_line(tui, output, min(start + 2, end))
        last = _bookmark_current_line(tui, output, min(start + 6, end))
        first_line = output.bookmark_line(first)
        last_line = output.bookmark_line(last)
        assert first_line is not None and last_line is not None and first_line < last_line

        output.scroll_to(y=last_line + 1, animate=False, immediate=True, force=True)
        assert tui.jump_to_previous_bookmark()
        assert output._bookmark_highlight == (last_line, last_line)

        output.scroll_to(y=first_line, animate=False, immediate=True, force=True)
        assert tui.jump_to_previous_bookmark()
        assert output._bookmark_highlight == (last_line, last_line)


@pytest.mark.asyncio
async def test_delete_clear_and_invalid_bookmark_are_safe(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        line = output.message_start_line(output.message_blocks()[0]["id"])
        assert line is not None
        bookmark = _bookmark_current_line(tui, output, line)
        invalid = {
            "bookmark_id": "missing",
            "message_id": "missing",
            "record_index": 0,
            "char_offset": 0,
            "role": "assistant",
            "summary": "已被压缩的消息",
            "created_at": "2026-07-21T09:00:00",
        }
        tui._bookmarks["missing"] = invalid

        assert not tui.jump_to_bookmark("missing")
        assert tui.delete_bookmark(bookmark["bookmark_id"])
        assert not tui.delete_bookmark(bookmark["bookmark_id"])
        assert tui.clear_topic_bookmarks() == 1
        assert tui._bookmarks == {}


@pytest.mark.asyncio
async def test_compacted_duplicate_content_does_not_retarget_bookmark(tmp_path) -> None:
    duplicate_messages = [
        {"role": "user", "content": "相同内容", "timestamp": "2026-07-21T09:00:00"},
        {"role": "assistant", "content": "中间回复", "timestamp": "2026-07-21T09:00:01"},
        {"role": "user", "content": "相同内容", "timestamp": "2026-07-21T10:00:00"},
    ]
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(duplicate_messages)
        output = tui._app.query_one("#output", _OutputLog)
        first_id = output.message_blocks()[0]["id"]
        second_id = output.message_blocks()[2]["id"]
        line = output.message_start_line(first_id)
        assert line is not None
        anchor = output.bookmark_anchor_at_line(line)
        assert anchor is not None
        bookmark_id = tui._bookmark_id(
            anchor["message_id"], anchor["record_index"], anchor["char_offset"]
        )
        bookmark = {
            "bookmark_id": bookmark_id,
            **anchor,
            "created_at": "2026-07-21T09:30:00",
        }
        tui._bookmarks[bookmark_id] = bookmark
        tui._save_bookmarks()

    restored = _make_tui(tmp_path)
    async with restored._app.run_test():
        restored.load_session_history(duplicate_messages[1:])
        output = restored._app.query_one("#output", _OutputLog)
        assert output.message_start_line(first_id) is None
        assert output.message_start_line(second_id) is not None
        assert not restored.jump_to_bookmark(bookmark["bookmark_id"])


@pytest.mark.asyncio
async def test_topics_use_separate_bookmark_files(tmp_path) -> None:
    tui = _make_tui(tmp_path, "cli:topic-a")
    first_file = tui._bookmark_file
    tui._bookmarks["a"] = {
        "bookmark_id": "a",
        "message_id": "message-a",
        "record_index": 0,
        "char_offset": 0,
        "role": "user",
        "summary": "A",
        "created_at": "2026-07-21T09:00:00",
    }
    tui._save_bookmarks()

    tui.set_input_history_topic("cli:topic-b")
    assert tui._bookmark_file != first_file
    assert tui._bookmarks == {}


@pytest.mark.asyncio
async def test_bookmark_popup_selects_position_and_marks_invalid_entry(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        bookmark = _bookmark_current_line(tui, output, min(start + 4, end))
        tui._bookmarks["missing"] = {
            "bookmark_id": "missing",
            "message_id": "missing",
            "record_index": 0,
            "char_offset": 0,
            "role": "assistant",
            "summary": "旧消息",
            "created_at": "2026-07-21T09:00:00",
        }

        tui.show_bookmark_popup()

        assert tui._popup_mode == "topic"
        assert tui._popup_items[0][0] == bookmark["bookmark_id"]
        assert "【第2阶段" in tui._popup_items[0][1]
        assert "前：第1阶段" in tui._popup_items[0][1]
        assert "后：第3阶段" in tui._popup_items[0][1]
        assert "[失效]" in tui._popup_items[1][1]
        await tui._popup_on_select(bookmark["bookmark_id"])
        line = output.bookmark_line(bookmark)
        assert output._bookmark_highlight == (line, line)


@pytest.mark.asyncio
async def test_bookmark_popup_rebuilds_unhelpful_legacy_summary(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        bookmark = _bookmark_current_line(tui, output, min(start + 4, end))
        bookmark["summary"] = "▼"

        items = dict(tui._bookmark_popup_items())

        label = items[bookmark["bookmark_id"]]
        assert "▼" not in label
        assert "【第2阶段" in label
        assert "前：第1阶段" in label
        assert "后：第3阶段" in label


@pytest.mark.asyncio
async def test_bookmark_keyboard_bindings_work_with_focused_input(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test() as pilot:
        tui.load_session_history(MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)

        await pilot.press("ctrl+b")
        await pilot.pause()
        assert len(tui._bookmarks) == 1
        bookmark = next(iter(tui._bookmarks.values()))

        tui.show_bookmark_popup()
        assert tui._popup_mode == "topic"
        assert tui._popup_items

        await pilot.press("escape")
        await pilot.press("f6")
        await pilot.pause()
        line = output.bookmark_line(bookmark)
        assert output._bookmark_highlight == (line, line)

        assert tui._bookmarks


@pytest.mark.asyncio
async def test_ctrl_c_copies_retained_selection_without_exiting(tmp_path, monkeypatch) -> None:
    tui = _make_tui(tmp_path)
    copied: list[str] = []
    async with tui._app.run_test() as pilot:
        tui.load_session_history(MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        output._sel_start = (0, 0)
        output._sel_end = (0, 2)
        output._sel_moved = True
        monkeypatch.setattr(output, "_copy_to_clipboard", copied.append)

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert copied
        assert output._selection_points() is None
        assert tui._app.is_running


@pytest.mark.asyncio
async def test_bookmark_delete_popup_removes_selected_bookmark(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        line = output.message_start_line(output.message_blocks()[0]["id"])
        assert line is not None
        bookmark = _bookmark_current_line(tui, output, line)

        tui.show_bookmark_delete_popup()
        assert tui._popup_mode == "topic"
        await tui._popup_on_select(bookmark["bookmark_id"])

        assert bookmark["bookmark_id"] not in tui._bookmarks


def test_unmarked_line_preserves_cjk_character_crossing_gutter_boundary() -> None:
    source = " 平时在 Editor 中"
    strip = Strip([Segment(source)], cell_len(source))

    rendered = _OutputLog._add_bookmark_gutter(
        strip,
        strip.cell_length,
        marked=False,
        overlay=False,
    )

    assert rendered is strip
    assert rendered.text == source


@pytest.mark.asyncio
async def test_bookmark_icon_is_drawn_on_exact_reading_line(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.load_session_history(LONG_MESSAGES)
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        target_line = min(start + 4, end)
        bookmark = _bookmark_current_line(tui, output, target_line)
        marker_line = output.bookmark_line(bookmark)
        assert marker_line == target_line
        screen_line = marker_line - int(output.scroll_offset.y)
        rendered = output.render_line(screen_line).text
        assert rendered.startswith("🔖")

        tui.toggle_bookmark_at_view()
        assert bookmark["bookmark_id"] not in tui._bookmarks
        rendered_without_bookmark = output.render_line(screen_line).text
        assert "🔖" not in rendered_without_bookmark


@pytest.mark.asyncio
async def test_live_bookmark_restores_from_persisted_transcript_id(tmp_path) -> None:
    """实时渲染和最终持久化时间/内容路径不同，书签仍绑定同一条消息。"""
    transcript_id = "assistant-live-stable-id"
    content = LONG_MESSAGES[0]["content"]
    tui = _make_tui(tmp_path)
    async with tui._app.run_test():
        tui.add_response(
            content,
            ts="2026-07-21 10:38:44",
            message_id=transcript_id,
        )
        output = tui._app.query_one("#output", _OutputLog)
        block = output.message_blocks()[0]
        start, end = output.message_line_range(block["id"])
        bookmark = _bookmark_current_line(tui, output, min(start + 4, end))
        assert bookmark["message_id"] == transcript_id

    persisted = [
        {
            "role": "assistant",
            "content": content,
            "timestamp": "2026-07-21T10:39:12.123456",
            "_transcript_id": transcript_id,
        }
    ]
    restored = _make_tui(tmp_path)
    async with restored._app.run_test():
        restored.load_session_history(persisted)
        output = restored._app.query_one("#output", _OutputLog)
        restored_bookmark = restored._bookmarks[bookmark["bookmark_id"]]
        restored_line = output.bookmark_line(restored_bookmark)
        assert restored_line is not None
        assert restored.jump_to_previous_bookmark()
        assert output._bookmark_highlight == (restored_line, restored_line)
        screen_line = restored_line - int(output.scroll_offset.y)
        assert "🔖" in output.render_line(screen_line).text
