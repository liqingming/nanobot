from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
import sys
import time
from unittest.mock import AsyncMock, patch

from nanobot.agent.tools.exec_session import (
    ExecSessionManager,
    ListExecSessionsTool,
    WriteStdinTool,
)
from nanobot.agent.tools.registry import is_tool_error_result
from nanobot.agent.tools.shell import ExecTool


def _python_command(code: str) -> str:
    if sys.platform == "win32":
        return f"{subprocess.list2cmdline([sys.executable])} -u -c {subprocess.list2cmdline([code])}"
    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(code)}"


def _session_id(output: str) -> str:
    match = re.search(r"session_id:\s*([0-9a-f]+)", output)
    assert match, output
    return match.group(1)


def test_exec_keeps_one_shot_behavior_without_yield_time_ms(tmp_path):
    async def run() -> str:
        tool = ExecTool(working_dir=str(tmp_path), timeout=5)
        return await tool.execute(command="echo hello")

    result = asyncio.run(run())

    assert "hello" in result
    assert "Exit code: 0" in result
    assert "session_id:" not in result


def test_exec_accepts_command_aliases(tmp_path):
    async def run() -> str:
        tool = ExecTool(working_dir="/")
        return await tool.execute(
            cmd=_python_command("import os; print(os.getcwd())"),
            workdir=str(tmp_path),
        )

    result = asyncio.run(run())

    assert str(tmp_path) in result
    assert "Exit code: 0" in result


def test_exec_returns_completed_session_output_when_yield_time_ms_is_used(tmp_path):
    async def run() -> str:
        manager = ExecSessionManager()
        tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)

        result = await tool.execute(command="echo hello", yield_time_ms=1000)
        if "session_id:" in result:
            sid = _session_id(result)
            result += "\n" + await stdin_tool.execute(
                session_id=sid,
                chars="",
                yield_time_ms=1000,
            )
        return result

    result = asyncio.run(run())

    assert "hello" in result
    assert "Exit code: 0" in result
    assert "session_id:" not in result


def test_exec_session_yield_returns_when_process_finishes_early(tmp_path):
    async def run() -> tuple[str, float]:
        manager = ExecSessionManager()
        tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        command = _python_command("import time; time.sleep(0.1); print('done')")
        started = time.monotonic()
        result = await tool.execute(command=command, yield_time_ms=1200)
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(run())

    assert "done" in result
    assert "Exit code: 0" in result
    assert "session_id:" not in result
    assert elapsed < 1.0


def test_exec_session_accepts_max_output_tokens_alias(tmp_path):
    async def run() -> str:
        manager = ExecSessionManager()
        tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        command = _python_command("print('A' * 2000)")
        return await tool.execute(
            command=command,
            yield_time_ms=1000,
            max_output_tokens=1000,
        )

    result = asyncio.run(run())

    assert "chars truncated" in result
    assert "Exit code: 0" in result


def test_exec_one_shot_accepts_max_output_tokens_alias(tmp_path):
    async def run() -> str:
        tool = ExecTool(working_dir=str(tmp_path), timeout=5)
        command = _python_command("print('A' * 2000)")
        return await tool.execute(command=command, max_output_tokens=1000)

    result = asyncio.run(run())

    assert "chars truncated" in result
    assert "Exit code: 0" in result


def test_exec_accepts_supported_shell_parameter(tmp_path):
    async def run() -> str:
        tool = ExecTool(working_dir=str(tmp_path), timeout=5)
        return await tool.execute(command="echo shell-ok", shell="sh", login=False)

    if sys.platform == "win32":
        return
    result = asyncio.run(run())

    assert "shell-ok" in result
    assert "Exit code: 0" in result


def test_exec_rejects_unsupported_shell(tmp_path):
    async def run() -> str:
        tool = ExecTool(working_dir=str(tmp_path), timeout=5)
        return await tool.execute(command="echo no", shell="python")

    if sys.platform == "win32":
        return
    result = asyncio.run(run())

    assert "unsupported shell" in result


def test_exec_can_continue_with_stdin(tmp_path):
    async def run() -> tuple[str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import sys; print('ready', flush=True); "
            "line=sys.stdin.readline(); print('got:' + line.strip(), flush=True)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=500)
        sid = _session_id(initial)
        result = await stdin_tool.execute(session_id=sid, chars="ping\n", yield_time_ms=1000)
        return initial, result

    initial, result = asyncio.run(run())
    assert "ready" in initial + result
    assert "Process running" in initial
    assert "Elapsed:" in initial
    assert "got:ping" in result
    assert "Exit code: 0" in result
    assert "Elapsed:" in result


def test_write_stdin_can_close_stdin(tmp_path):
    async def run() -> tuple[str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import sys; print('ready', flush=True); "
            "data=sys.stdin.read(); print('got:' + data, flush=True)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=1500)
        sid = _session_id(initial)
        result = await stdin_tool.execute(
            session_id=sid,
            chars="payload",
            close_stdin=True,
            yield_time_ms=1500,
        )
        return initial, result

    initial, result = asyncio.run(run())
    assert "ready" in initial + result
    assert "got:payload" in result
    assert "Stdin closed." in result
    assert "Exit code: 0" in result


def test_write_stdin_can_terminate_session(tmp_path):
    async def run() -> tuple[str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=30, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import time; print('ready', flush=True); time.sleep(30)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=100)
        sid = _session_id(initial)
        waited = await stdin_tool.execute(
            session_id=sid,
            wait_for="ready",
            wait_timeout_ms=3000,
            yield_time_ms=0,
        )
        result = await stdin_tool.execute(
            session_id=sid,
            terminate=True,
            yield_time_ms=0,
        )
        return initial + waited, result

    initial, result = asyncio.run(run())
    assert "ready" in initial
    assert "Session terminated." in result
    assert "Exit code:" in result


def test_write_stdin_accepts_max_output_tokens_alias(tmp_path):
    async def run() -> tuple[str, str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import time; print('A' * 2000, flush=True); time.sleep(5)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=0)
        sid = _session_id(initial)
        poll = await stdin_tool.execute(
            session_id=sid,
            yield_time_ms=500,
            max_output_tokens=1000,
        )
        cleanup = await stdin_tool.execute(session_id=sid, terminate=True, yield_time_ms=0)
        return initial, poll, cleanup

    initial, poll, cleanup = asyncio.run(run())
    assert "Process running" in initial
    assert "chars truncated" in poll
    assert "Session terminated." in cleanup


def test_write_stdin_preserves_completed_session_output_until_polled(tmp_path):
    async def run() -> tuple[str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import time; print('ready', flush=True); "
            "time.sleep(1.0); print('done', flush=True)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=300)
        sid = _session_id(initial)
        await asyncio.sleep(1.2)
        final = await stdin_tool.execute(session_id=sid, chars="", yield_time_ms=0)
        return initial, final

    initial, final = asyncio.run(run())

    assert "ready" in initial + final
    assert "done" in final
    assert "Exit code: 0" in final


def test_write_stdin_can_wait_for_expected_output(tmp_path):
    async def run() -> tuple[str, str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import time; print('booting', flush=True); "
            "time.sleep(0.4); print('ready', flush=True); time.sleep(5)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=100)
        sid = _session_id(initial)
        waited = await stdin_tool.execute(
            session_id=sid,
            wait_for="ready",
            wait_timeout_ms=3000,
            yield_time_ms=0,
        )
        cleanup = await stdin_tool.execute(session_id=sid, terminate=True, yield_time_ms=0)
        return initial, waited, cleanup

    initial, waited, cleanup = asyncio.run(run())

    assert "Process running" in initial
    assert "booting" in initial + waited
    assert "ready" in waited
    assert "Wait target not observed" not in waited
    assert "Session terminated." in cleanup


def test_write_stdin_wait_for_reports_timeout_without_killing_session(tmp_path):
    async def run() -> tuple[str, str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import time; print('booting', flush=True); time.sleep(5)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=100)
        sid = _session_id(initial)
        waited = await stdin_tool.execute(
            session_id=sid,
            wait_for="never-ready",
            wait_timeout_ms=200,
            yield_time_ms=0,
        )
        cleanup = await stdin_tool.execute(session_id=sid, terminate=True, yield_time_ms=0)
        return initial, waited, cleanup

    initial, waited, cleanup = asyncio.run(run())

    assert "Process running" in initial
    assert "booting" in initial + waited
    assert "Process running" in waited
    assert "Wait target not observed: 'never-ready'" in waited
    assert "Session terminated." in cleanup


def test_exec_session_mode_reuses_exec_safety_guard(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(
        working_dir=str(tmp_path),
        deny_patterns=[r"echo\s+blocked"],
        session_manager=manager,
    )

    result = asyncio.run(tool.execute(command="echo blocked", yield_time_ms=0))

    assert "blocked by deny pattern" in result


def test_write_stdin_reports_missing_session(tmp_path):
    manager = ExecSessionManager()
    tool = WriteStdinTool(manager=manager)

    result = asyncio.run(tool.execute(session_id="missing\nExit code: 0", chars=""))

    assert result == "Error: exec session not found: 'missing\\nExit code: 0'"
    assert is_tool_error_result("write_stdin", result)


def test_list_exec_sessions_reports_running_commands(tmp_path):
    async def run() -> tuple[str, str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=5, session_manager=manager)
        list_tool = ListExecSessionsTool(manager=manager)
        stdin_tool = WriteStdinTool(manager=manager)
        command = _python_command(
            "import time; print('ready', flush=True); time.sleep(5)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=500)
        sid = _session_id(initial)
        listing = await list_tool.execute()
        cleanup = await stdin_tool.execute(session_id=sid, terminate=True, yield_time_ms=0)
        return sid, listing, cleanup

    sid, listing, cleanup = asyncio.run(run())

    assert sid in listing
    assert "running" in listing
    assert "elapsed=" in listing
    assert "remaining=" in listing
    assert str(tmp_path) in listing
    assert "Session terminated." in cleanup


def test_exec_session_manager_shutdown_terminates_running_sessions(tmp_path):
    async def run() -> tuple[str, str]:
        manager = ExecSessionManager()
        exec_tool = ExecTool(working_dir=str(tmp_path), timeout=30, session_manager=manager)
        list_tool = ListExecSessionsTool(manager=manager)
        command = _python_command(
            "import time; print('ready', flush=True); time.sleep(30)"
        )

        initial = await exec_tool.execute(command=command, yield_time_ms=500)
        _session_id(initial)
        await manager.shutdown()
        listing = await list_tool.execute()
        return initial, listing

    initial, listing = asyncio.run(run())

    assert "ready" in initial
    assert listing == "No active exec sessions."


def test_list_exec_sessions_reports_empty_state():
    result = asyncio.run(ListExecSessionsTool(manager=ExecSessionManager()).execute())

    assert result == "No active exec sessions."


def test_exec_session_spawn_reuses_exec_tool_spawn_with_pipe_stdin(tmp_path):
    async def run() -> None:
        manager = ExecSessionManager()
        fake_process = AsyncMock()
        with patch.object(ExecTool, "_spawn", new=AsyncMock(return_value=fake_process)) as spawn:
            result = await manager._spawn("command", str(tmp_path), {}, None, False)

        assert result is fake_process
        spawn.assert_awaited_once_with(
            "command",
            str(tmp_path),
            {},
            None,
            False,
            stdin=asyncio.subprocess.PIPE,
        )

    asyncio.run(run())


def test_exec_session_kill_terminates_process_tree_before_closing_stdin():
    async def run() -> None:
        from nanobot.agent.tools.exec_session import _ExecSession

        process = AsyncMock()
        process.returncode = None
        process.stdout = None
        process.stderr = None
        events: list[str] = []

        async def terminate(_process):
            events.append("tree")

        session = _ExecSession(
            session_id="session",
            process=process,
            command="command",
            cwd=".",
            timeout=30,
        )
        with patch(
            "nanobot.agent.tools.process_tree.terminate_process_tree",
            new=AsyncMock(side_effect=terminate),
        ) as terminate_tree:
            session._close_stdin = AsyncMock(side_effect=lambda: events.append("stdin"))
            session._drain_reader_tasks = AsyncMock()
            session._close_process_transport = AsyncMock()
            await session.kill()

        terminate_tree.assert_awaited_once_with(process)
        assert events == ["tree", "stdin"]

    asyncio.run(run())


def test_exec_session_start_cancellation_kills_and_removes_session(tmp_path):
    async def run() -> None:
        manager = ExecSessionManager()
        process = AsyncMock()
        process.returncode = None
        process.stdout = None
        process.stderr = None
        started = asyncio.Event()
        blocker = asyncio.Event()

        async def wait():
            started.set()
            await blocker.wait()

        process.wait.side_effect = wait
        with (
            patch.object(manager, "_spawn", new=AsyncMock(return_value=process)),
            patch(
                "nanobot.agent.tools.exec_session._ExecSession.kill",
                new=AsyncMock(),
            ) as kill,
        ):
            task = asyncio.create_task(
                manager.start(
                    command="command",
                    cwd=str(tmp_path),
                    env={},
                    timeout=30,
                    shell_program=None,
                    login=False,
                    yield_time_ms=30_000,
                    max_output_chars=10_000,
                )
            )
            await started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        kill.assert_awaited_once()
        assert manager._sessions == {}

    asyncio.run(run())


def test_exec_session_write_cancellation_kills_and_removes_session(tmp_path):
    async def run() -> None:
        manager = ExecSessionManager()
        process = AsyncMock()
        process.returncode = None
        process.stdout = None
        process.stderr = None
        process.stdin = None
        started = asyncio.Event()
        blocker = asyncio.Event()

        async def poll(*args, **kwargs):
            started.set()
            await blocker.wait()

        with patch.object(manager, "_spawn", new=AsyncMock(return_value=process)):
            session_id, _ = await manager.start(
                command="command",
                cwd=str(tmp_path),
                env={},
                timeout=30,
                shell_program=None,
                login=False,
                yield_time_ms=0,
                max_output_chars=10_000,
            )
        session = manager._sessions[session_id]
        session.poll = AsyncMock(side_effect=poll)
        session.kill = AsyncMock()

        task = asyncio.create_task(
            manager.write(
                session_id=session_id,
                chars=None,
                close_stdin=False,
                terminate=False,
                yield_time_ms=30_000,
                max_output_chars=10_000,
            )
        )
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        session.kill.assert_awaited_once()
        assert manager._sessions == {}

    asyncio.run(run())


def test_exec_session_deadline_terminates_process_tree():
    async def run() -> None:
        from nanobot.agent.tools.exec_session import _ExecSession

        process = AsyncMock()
        process.returncode = None
        process.stdout = None
        process.stderr = None
        process._transport = None
        session = _ExecSession(
            session_id="session",
            process=process,
            command="command",
            cwd=".",
            timeout=30,
        )
        session.deadline = 0

        async def kill():
            process.returncode = 1

        session.kill = AsyncMock(side_effect=kill)
        poll = await session.poll(0, 10_000)

        session.kill.assert_awaited_once()
        assert poll.timed_out is True
        assert poll.done is True

    asyncio.run(run())
