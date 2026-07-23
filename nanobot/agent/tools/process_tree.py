"""Cross-platform subprocess-tree termination helpers."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from contextlib import suppress

from loguru import logger

_IS_WINDOWS = sys.platform == "win32"
_PROCESS_WAIT_TIMEOUT_S = 5.0
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


async def terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    """Force-stop *process* and every descendant, then reap the direct child.

    Tool commands run through a shell, so killing only that direct shell can
    orphan native grandchildren (for example a long-running ``git fsck``).
    Windows has no asyncio process-group API, therefore ``taskkill /T`` is used
    while the shell is still alive. POSIX subprocesses are started in their own
    session and can be terminated through their process group.
    """
    if process.returncode is not None:
        return

    if _IS_WINDOWS:
        await _terminate_windows_tree(process.pid)
    else:
        _terminate_posix_tree(process.pid)

    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    with suppress(asyncio.TimeoutError, ProcessLookupError):
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_WAIT_TIMEOUT_S)


async def _terminate_windows_tree(pid: int) -> None:
    try:
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(pid),
            "/T",
            "/F",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        await asyncio.wait_for(killer.wait(), timeout=_PROCESS_WAIT_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("Failed to terminate Windows process tree {}: {}", pid, exc)


def _terminate_posix_tree(pid: int) -> None:
    try:
        os.killpg(pid, _KILL_SIGNAL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        logger.warning("Failed to terminate POSIX process tree {}: {}", pid, exc)
