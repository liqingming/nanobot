"""运行内读取证据不能用文件去重占位替代，且不放宽既有安全边界。"""

import asyncio
import os

import pytest

from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.file_state import FileStates
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.fork.agent.read_visibility import read_evidence_required, require_read_evidence


async def test_repeated_read_returns_body_only_inside_run(tmp_path):
    path = tmp_path / "evidence.txt"
    path.write_text("first\nsecond\nthird", encoding="utf-8")
    tool = ReadFileTool(workspace=tmp_path, file_states=FileStates())
    assert "2| second" in await tool.execute(path.name, offset=2, limit=1)
    assert "File unchanged" in await tool.execute(path.name, offset=2, limit=1)

    @require_read_evidence
    async def run(*, count):
        for _ in range(count):
            # 不假设历史中的读取回执仍在当前模型窗口。
            result = await tool.execute(path.name, offset=2, limit=1)
            assert "2| second" in result
            assert "1| first" not in result
            assert "3| third" not in result
        return "done"

    assert await run(count=2) == "done"
    assert run.__name__ == "run"
    assert not read_evidence_required()
    assert "File unchanged" in await tool.execute(path.name, offset=2, limit=1)


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("same_mtime", [False, True])
async def test_force_and_external_change_preserve_read_hash_guard(tmp_path, scoped, same_mtime):
    path = tmp_path / "evidence.txt"
    path.write_text("before", encoding="utf-8")
    states = FileStates()
    tool = ReadFileTool(workspace=tmp_path, file_states=states)

    async def run():
        assert states.check_read(path) is not None
        assert "1| before" in await tool.execute(path.name)
        assert "1| before" in await tool.execute(path.name, force=True)
        assert states.check_read(path) is None
        prior = path.stat()
        path.write_text("after!", encoding="utf-8")
        os.utime(path, ns=(prior.st_atime_ns, prior.st_mtime_ns if same_mtime
                          else prior.st_mtime_ns + 2_000_000_000))
        assert "modified since last read" in states.check_read(path)
        assert "1| after!" in await tool.execute(path.name)
        assert states.check_read(path) is None
        path.write_text("third!", encoding="utf-8")
        os.utime(path, ns=(prior.st_atime_ns, prior.st_mtime_ns))
        assert "modified since last read" in states.check_read(path)

    await (require_read_evidence(run)() if scoped else run())
    assert not read_evidence_required()


@pytest.mark.parametrize("force", [False, True])
async def test_scoped_read_keeps_workspace_and_request_policy(tmp_path, force):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private outside", encoding="utf-8")
    inside = root / "blocked.txt"
    inside.write_text("private inside", encoding="utf-8")
    tool = ReadFileTool(workspace=root, allowed_dir=root)
    await tool.execute(inside.name)
    token = bind_request_context(RequestContext(
        channel="test", chat_id="test",
        metadata={"tool_policy": {"blocked_read_file_paths": ["blocked.txt"]}},
    ))

    @require_read_evidence
    async def run():
        result = await tool.execute(str(outside), force=force)
        assert result.is_error
        assert "private outside" not in result
        result = await tool.execute(inside.name, force=force)
        assert result.is_error
        assert "blocked by the request tool_policy" in result
        assert "private inside" not in result

    try:
        await run()
    finally:
        reset_request_context(token)


async def test_nested_scope_and_exception_restore_previous_value():
    @require_read_evidence
    async def inner():
        assert read_evidence_required()
        raise ValueError("test failure")

    @require_read_evidence
    async def outer():
        with pytest.raises(ValueError, match="test failure"):
            await inner()
        assert read_evidence_required()

    await outer()
    assert not read_evidence_required()
    with pytest.raises(ValueError, match="test failure"):
        await inner()
    assert not read_evidence_required()


async def test_cancelled_run_resets_in_same_task():
    started = asyncio.Event()

    @require_read_evidence
    async def run():
        assert read_evidence_required()
        started.set()
        await asyncio.Event().wait()

    async def worker():
        try:
            await run()
        except asyncio.CancelledError:
            # 必须检查被取消任务自身，而不只是父任务的独立上下文。
            assert not read_evidence_required()
            raise

    task = asyncio.create_task(worker())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not read_evidence_required()


async def test_concurrent_runs_do_not_change_independent_tool_mode(tmp_path):
    path = tmp_path / "evidence.txt"
    path.write_text("body", encoding="utf-8")
    tool = ReadFileTool(workspace=tmp_path, file_states=FileStates())
    await tool.execute(path.name)
    started = asyncio.Event()
    release = asyncio.Event()

    @require_read_evidence
    async def run():
        started.set()
        await release.wait()
        # 工具执行子任务继承本 run 的模式。
        result = await asyncio.create_task(tool.execute(path.name))
        assert "1| body" in result

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not read_evidence_required()
        assert "File unchanged" in await tool.execute(path.name)
        await require_read_evidence(tool.execute)(path.name)
        assert not read_evidence_required()
        assert "File unchanged" in await tool.execute(path.name)
    finally:
        release.set()
        await task
    assert not read_evidence_required()
