"""Regression coverage for Textual streaming segments around tool calls."""
from __future__ import annotations

import asyncio

import pytest

from nanobot.fork.cli.tui_textual import (
    _TEXTUAL_AVAILABLE,
    TextualTUI,
    _OutputLog,
)

pytestmark = pytest.mark.skipif(not _TEXTUAL_AVAILABLE, reason="textual not installed")


def _output_text(output: _OutputLog) -> str:
    return "\n".join("".join(segment.text for segment in line) for line in output.lines)


async def _settle_stream_render() -> None:
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_tool_preface_is_replaced_by_final_response_but_trace_survives() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test():
        output = tui._app.query_one("#output", _OutputLog)
        tui.stream_start()
        tui.stream_delta("我先检查运行日志。")
        await _settle_stream_render()
        tui.flush_stream()
        tui.tool_phase_start()
        tui.add_progress("exec")
        tui.add_tool_result("exit 0")

        during_tool = _output_text(output)
        assert "我先检查运行日志。" in during_tool
        assert "exec" in during_tool

        tui.stream_delta("最终结论：没有重复事件。")
        await _settle_stream_render()
        final = tui.pop_stream()
        tui.flush_accumulator()
        tui.add_response(final)

        completed = _output_text(output)
        assert "我先检查运行日志。" not in completed
        assert "exec" in completed
        assert completed.count("最终结论：没有重复事件。") == 1


@pytest.mark.asyncio
async def test_multiple_tool_prefaces_are_removed_without_removing_tool_traces() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test():
        output = tui._app.query_one("#output", _OutputLog)
        tui.stream_start()

        for preface, tool in (("先查日志。", "grep"), ("再核对源码。", "read_file")):
            tui.stream_delta(preface)
            await _settle_stream_render()
            tui.flush_stream()
            tui.tool_phase_start()
            tui.add_progress(tool)
            tui.add_tool_result("ok")

        tui.stream_delta("最终答复。")
        await _settle_stream_render()
        final = tui.pop_stream()
        tui.flush_accumulator()
        tui.add_response(final)

        completed = _output_text(output)
        assert "先查日志。" not in completed
        assert "再核对源码。" not in completed
        assert "grep" in completed
        assert "read_file" in completed
        assert completed.count("最终答复。") == 1


@pytest.mark.asyncio
async def test_resuming_stream_without_tool_progress_keeps_prior_segment() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test():
        output = tui._app.query_one("#output", _OutputLog)
        tui.stream_start()
        tui.stream_delta("长度续写第一段。")
        await _settle_stream_render()
        tui.flush_stream()

        # A new stream segment without add_progress is a continuation/retry,
        # not a tool-preface segment.
        tui.stream_delta("长度续写第二段。")
        await _settle_stream_render()
        final = tui.pop_stream()
        tui.flush_accumulator()
        tui.add_response(final)

        completed = _output_text(output)
        assert completed.count("长度续写第一段。") == 1
        assert completed.count("长度续写第二段。") == 1


@pytest.mark.asyncio
async def test_plain_stream_is_rendered_once() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test():
        output = tui._app.query_one("#output", _OutputLog)
        tui.stream_start()
        tui.stream_delta("普通最终答复。")
        await _settle_stream_render()
        final = tui.pop_stream()
        tui.flush_accumulator()
        tui.add_response(final)

        assert _output_text(output).count("普通最终答复。") == 1


@pytest.mark.asyncio
async def test_error_response_replaces_tool_preface_and_keeps_trace() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test():
        output = tui._app.query_one("#output", _OutputLog)
        live = tui._app.query_one("#live")
        tui.stream_start()
        assert live.has_class("visible")
        assert "思考中" not in _output_text(output)
        tui.stream_delta("先尝试读取。")
        await _settle_stream_render()
        tui.flush_stream()
        tui.tool_phase_start()
        tui.add_progress("read_file")
        tui.add_tool_result("failed")

        # The outbound error branch also finalizes through pop_stream before
        # writing the error response.
        tui.pop_stream()
        tui.flush_accumulator()
        tui.add_response("读取失败。", {"render_as": "error"})

        completed = _output_text(output)
        assert "先尝试读取。" not in completed
        assert "read_file" in completed
        assert completed.count("读取失败。") == 1
        assert not live.has_class("visible")
        assert "思考中" not in completed


@pytest.mark.asyncio
async def test_cancel_flush_keeps_visible_text_and_next_turn_resets_state() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test():
        output = tui._app.query_one("#output", _OutputLog)
        tui.stream_start()
        tui.stream_delta("取消前已经输出。")
        await _settle_stream_render()
        tui.flush_stream()
        tui.add_system("已取消当前请求。")

        cancelled = _output_text(output)
        assert not tui._app.query_one("#live").has_class("visible")
        assert "思考中" not in cancelled
        assert cancelled.count("取消前已经输出。") == 1
        assert "已取消当前请求。" in cancelled

        tui.stream_start()
        assert tui._pending_tool_text_records == []
        assert tui._temporary_tool_text_records == []


@pytest.mark.asyncio
async def test_transient_activity_stays_out_of_output_for_wrapped_tool_traces() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test() as pilot:
        output = tui._app.query_one("#output", _OutputLog)
        live = tui._app.query_one("#live")
        tui.stream_start()
        await pilot.pause()

        assert live.has_class("visible")
        assert "思考中" in str(live.render())
        assert "思考中" not in _output_text(output)

        long_hint = "read " + "very-long-path/" * 20
        tui.flush_stream()
        tui.tool_phase_start()
        tui.add_progress(long_hint)
        await pilot.pause(0.1)

        assert live.has_class("visible")
        assert "very-long-path" in str(live.render())
        assert "执行中" not in _output_text(output)
        assert "very-long-path" not in _output_text(output)

        tui.add_tool_result("ok")
        await asyncio.sleep(0.6)
        await pilot.pause()

        completed_tool = _output_text(output)
        assert completed_tool.count("very-long-path") >= 1
        assert "思考中" not in completed_tool
        assert live.has_class("visible")
        assert "思考中" in str(live.render())

        tui.pop_stream()
        await pilot.pause()

        assert not live.has_class("visible")
        assert "思考中" not in _output_text(output)


@pytest.mark.asyncio
async def test_consecutive_wrapped_tool_traces_all_survive_live_status_cycles() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test() as pilot:
        output = tui._app.query_one("#output", _OutputLog)
        live = tui._app.query_one("#live")
        tui.stream_start()
        await pilot.pause()

        for index in range(4):
            tui.flush_stream()
            tui.tool_phase_start()
            tui.add_progress(f"read-{index} " + "very-long-path/" * 20)
            await pilot.pause(0.1)
            tui.add_tool_result(f"summary-{index}")
            await asyncio.sleep(0.6)
            await pilot.pause()

            history = _output_text(output)
            assert all(f"summary-{seen}" in history for seen in range(index + 1))
            assert "思考中" not in history
            assert "执行中" not in history
            assert live.has_class("visible")

        tui.pop_stream()
        await pilot.pause()

        history = _output_text(output)
        assert all(f"summary-{index}" in history for index in range(4))
        assert not live.has_class("visible")


@pytest.mark.asyncio
async def test_idle_live_status_preserves_bottom_follow_for_next_stream_delta() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test(size=(80, 24)) as pilot:
        output = tui._app.query_one("#output", _OutputLog)
        for index in range(80):
            output.write(f"history-{index}")
        output.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()

        tui.stream_start()
        tui.stream_delta("first " * 500)
        await _settle_stream_render()
        assert output.is_at_bottom()

        # Reproduce the quiet-period transition that expands #live by one row.
        tui._schedule_idle_thinking(delay=0.01)
        await asyncio.sleep(0.05)
        await pilot.pause()

        assert tui._app.query_one("#live").has_class("visible")
        assert output.is_at_bottom()

        tui.stream_delta("LATEST-STREAM-OUTPUT " * 100)
        await _settle_stream_render()
        # The debounce task posts rendering back to Textual's message loop.
        await pilot.pause()

        assert "LATEST-STREAM-OUTPUT" in _output_text(output)
        assert output.is_at_bottom()


@pytest.mark.asyncio
async def test_idle_live_status_does_not_pull_history_reader_to_bottom() -> None:
    tui = TextualTUI(render_markdown=False)

    async with tui._app.run_test(size=(80, 24)) as pilot:
        output = tui._app.query_one("#output", _OutputLog)
        for index in range(80):
            output.write(f"history-{index}")
        output.scroll_end(animate=False, immediate=True, force=True)
        await pilot.pause()

        tui.stream_start()
        tui.stream_delta("first " * 500)
        await _settle_stream_render()
        # Finish the preceding #live collapse refresh before simulating a user scroll.
        await pilot.pause()
        output.scroll_to(
            y=max(0, int(output.max_scroll_y) - 5),
            animate=False,
            immediate=True,
            force=True,
        )
        await pilot.pause()
        reading_position = int(output.scroll_offset.y)
        assert not output.is_at_bottom()

        tui._schedule_idle_thinking(delay=0.01)
        await asyncio.sleep(0.05)
        await pilot.pause()

        assert tui._app.query_one("#live").has_class("visible")
        assert int(output.scroll_offset.y) == reading_position
        assert not output.is_at_bottom()
