"""待发送附件栏不改变输入、焦点或发起模型请求。"""

from unittest.mock import AsyncMock

import pytest

from nanobot.fork.cli.tui_textual import TextualTUI


@pytest.mark.asyncio
async def test_context_bar_is_literal_and_does_not_submit():
    tui = TextualTUI()
    submit = AsyncMock()
    tui.set_on_submit(submit)
    async with tui._app.run_test() as pilot:
        await pilot.pause()
        inp = tui._app.query_one("#input")
        inp.value = "待提交的问题"
        inp.cursor_position = 2
        selection = inp.selection
        tui.set_ide_context(["[red]玩家.cs:10–12"])
        await pilot.pause()
        bar = tui._app.query_one("#ide-context")
        assert bar.has_class("visible")
        assert not bar._render_markup
        assert "[red]玩家.cs:10–12" in bar.content
        assert inp.value == "待提交的问题"
        assert inp.selection == selection
        assert tui._app.focused is inp
        submit.assert_not_called()
        tui.set_ide_context([])
        await pilot.pause()
        assert not bar.has_class("visible")



@pytest.mark.asyncio
async def test_live_queued_and_replayed_context_copy_match(tmp_path):
    import copy

    from nanobot.fork.cli.ide_context import PendingIDEContext

    pending = PendingIDEContext(tmp_path, lambda labels: None)
    pending.switch("cli:test")
    for index in range(2):
        pending.add({
            "request_id": str(index), "path": str(tmp_path / f"玩家{index}.cs"),
            "start_line": 10, "end_line": 12, "text": "未保存代码\n下一行",
            "session_key": pending.session_key, "generation": pending.generation,
        })
    preview = pending.preview("分析")
    consumed = pending.consume("分析")
    messages = [{"role": "user", "content": consumed, "_transcript_id": "test"}]
    original = copy.deepcopy(messages)
    tui = TextualTUI()
    async with tui._app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        out = tui._app.query_one("#output")

        async def copied_body(render):
            tui.reset_history()
            render()
            await pilot.pause()
            start, end = out._user_ranges[-1]
            out._sel_start = (start, 0)
            out._sel_end = (end, 10000)
            # 使用真实鼠标复制的文本提取路径，忽略每次生成的时间头。
            copied = out._extract_selected_text()
            return copied[copied.index("分析"):].strip()

        live = await copied_body(lambda: tui.add_user_echo(preview))
        queued = await copied_body(lambda: tui.add_user_echo(consumed))
        replayed = await copied_body(lambda: tui.load_session_history(messages))
        paged = await copied_body(lambda: tui._render_session_messages(messages))
        assert live == queued == replayed == paged == preview
        assert "request_id" not in replayed
        assert messages == original
        assert "未保存代码" in messages[0]["content"]


@pytest.mark.parametrize("raw", [
    "不是 JSON", "[]", "{}", "[null]", '[{"path": "玩家.cs"}]',
    '[{"request_id": "1", "path": "玩家.cs", "start_line": true,'
    ' "end_line": 2, "text": "代码"}]',
    '[{"request_id": "1", "path": "玩家.cs", "start_line": 3,'
    ' "end_line": 2, "text": "代码"}]',
])
def test_invalid_snapshot_display_preserves_original(raw):
    from nanobot.fork.cli.ide_context import IDE_SNAPSHOT_MARKER, format_ide_context_display

    text = "问题" + IDE_SNAPSHOT_MARKER + raw
    assert format_ide_context_display(text) == text


def test_plain_message_and_snapshot_with_trailing_prose_are_preserved():
    from nanobot.fork.cli.ide_context import IDE_SNAPSHOT_MARKER, format_ide_context_display

    for text in ("普通问题", "问题\n[IDE 附件] 玩家.cs:1–2",
                 "提及" + IDE_SNAPSHOT_MARKER + "[]\n还有问题"):
        assert format_ide_context_display(text) == text
