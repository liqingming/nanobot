"""宿主回执的持久化、跨进程重启、失联恢复和占用交还。"""

import asyncio
import json
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from nanobot.fork.agent.subagent_control import SubagentControlState
from nanobot.fork.agent.subagent_receipts import SubagentReceiptStore
from nanobot.fork.cli.subagent_recovery import main


def status():
    return SimpleNamespace(label="chapter", phase="done", iteration=1, stop_reason="completed")


async def test_terminal_receipt_survives_new_manager_and_cache_eviction(tmp_path):
    state = SubagentControlState(tmp_path)
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    for i in range(257):
        state.register(str(i), "cli:one", tmp_path, task, status())
        state.finish(str(i))
    assert len(state.finished) == 256
    other = SubagentControlState(tmp_path)
    receipt = await other.execute("status", "cli:one", tmp_path, "0", 0)
    assert receipt["worker_stopped"] is True
    assert receipt["confirmation_source"] == "host_cleanup"
    assert receipt["business_success"] == "unverified"
    for owner, workspace in [("cli:other", tmp_path), ("cli:one", tmp_path / "elsewhere")]:
        hidden = await other.execute("status", owner, workspace, "0", 0)
        assert hidden["worker_stopped"] is False
        assert hidden["state"] == "unknown"


async def test_another_manager_cannot_cancel_or_reconcile_live_worker(tmp_path):
    state = SubagentControlState(tmp_path)
    task = asyncio.create_task(asyncio.sleep(60))
    state.register("id", "cli:one", tmp_path, task, status())
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("operator note", encoding="utf-8")
    other = SubagentControlState(tmp_path)
    try:
        result = await other.execute("cancel", "cli:one", tmp_path, "id", 0)
        assert result["owner_available"] is True
        assert result["worker_stopped"] is False
        assert not task.cancelling()
        with pytest.raises(ValueError, match="宿主锁"):
            other.store.reconcile("id", "cli:one", tmp_path, confirmed_stopped=True,
                                  reason="verified", evidence_file=evidence)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        state.finish("id")


async def test_receipt_write_failure_keeps_lease_and_retries(tmp_path, monkeypatch):
    state = SubagentControlState(tmp_path)
    task = asyncio.create_task(asyncio.sleep(0))
    state.register("id", "cli:one", tmp_path, task, status())
    await task
    original = state.store._write
    monkeypatch.setattr(state.store, "_write", lambda _: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        await state.execute("status", "cli:one", tmp_path, "id", 0)
    with pytest.raises(OSError):
        await state.execute("list", "cli:one", tmp_path, None, 0)
    other = SubagentControlState(tmp_path)
    assert not (await other.execute("status", "cli:one", tmp_path, "id", 0))["worker_stopped"]
    monkeypatch.setattr(state.store, "_write", original)
    assert (await state.execute("wait", "cli:one", tmp_path, "id", 0))["worker_stopped"]
    assert (await other.execute("status", "cli:one", tmp_path, "id", 0))["worker_stopped"]


@pytest.mark.parametrize("confirm,reason", [(False, "checked"), (True, " ")])
def test_operator_recovery_requires_confirmation_and_evidence(tmp_path, confirm, reason):
    store = SubagentReceiptStore(tmp_path)
    evidence = tmp_path / "note.txt"
    evidence.write_text("inspection", encoding="utf-8")
    with pytest.raises(ValueError):
        store.reconcile("old", "cli:one", tmp_path, confirmed_stopped=confirm,
                        reason=reason, evidence_file=evidence)
    assert store.read("old") is None


def test_recovery_is_scoped_idempotent_and_keeps_audit(tmp_path):
    store = SubagentReceiptStore(tmp_path)
    evidence = tmp_path / "note.txt"
    evidence.write_text("old host and provider exited; artifacts preserved", encoding="utf-8")
    receipt = store.reconcile("old", "cli:one", tmp_path, confirmed_stopped=True,
                              reason="operator checked", evidence_file=evidence)
    record = deepcopy(store.read("old"))
    assert receipt["confirmation_source"] == "operator_verified"
    assert record["recovery"]["evidence_sha256"]
    assert record["previous_record"] is None
    again = store.reconcile("old", "cli:one", tmp_path, confirmed_stopped=True,
                            reason="do not overwrite", evidence_file=evidence)
    assert again == receipt and store.read("old") == record
    with pytest.raises(ValueError, match="其他话题"):
        store.reconcile("old", "cli:other", tmp_path, confirmed_stopped=True,
                        reason="wrong scope", evidence_file=evidence)


def test_corrupt_receipt_cannot_be_overwritten_as_stopped(tmp_path):
    store = SubagentReceiptStore(tmp_path)
    store.root.mkdir()
    store._path("id").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        store.inspect("id", "cli:one", tmp_path)
    with pytest.raises(ValueError):
        store.reconcile("id", "cli:one", tmp_path, confirmed_stopped=True,
                        reason="note", evidence_file=store._path("id"))


async def test_real_process_crash_recovery_then_release_claim_preserves_artifacts(tmp_path):
    # 真实进程持锁登记后崩溃，没有运行模型或外部工具。
    source = """
import os, sys
from pathlib import Path
from nanobot.fork.agent.subagent_receipts import SubagentReceiptStore
root=Path(sys.argv[1])
s=SubagentReceiptStore(root)
s.register('crashed', 'cli:one', root, {'task_id':'crashed', 'state':'running', 'worker_stopped':False})
os._exit(9)
"""
    child = subprocess.run([sys.executable, "-c", source, str(tmp_path)], timeout=20,
                           capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert child.returncode == 9, child.stderr
    artifact = tmp_path / "finished.xml"
    artifact.write_text("completed-resource", encoding="utf-8")
    claim = tmp_path / "claim.json"
    claim.write_text(json.dumps({"active": "crashed"}), encoding="utf-8")
    restarted = SubagentControlState(tmp_path)
    unknown = await restarted.execute("status", "cli:one", tmp_path, "crashed", 0)
    assert unknown["recovery_required"] and not unknown["worker_stopped"]
    assert unknown["owner_available"] is False
    assert json.loads(claim.read_text())["active"] == "crashed"
    evidence = tmp_path / "inspection.json"
    evidence.write_text(json.dumps({"process_exit_code": child.returncode,
                                    "external_workers": []}), encoding="utf-8")
    restarted.store.reconcile("crashed", "cli:one", tmp_path, confirmed_stopped=True,
                               reason="test process exited; no descendants launched",
                               evidence_file=evidence)
    after_second_restart = SubagentControlState(tmp_path)
    receipt = await after_second_restart.execute("status", "cli:one", tmp_path, "crashed", 0)
    assert receipt["worker_stopped"] is True
    row = after_second_restart.store.read("crashed")
    assert row["previous_record"]["receipt"]["state"] == "running"
    # 消费者只在取得真实持久回执后交还旧 claim，不重做已完成资源。
    claim.write_text(json.dumps({"active": None, "last_closed": receipt}), encoding="utf-8")
    assert artifact.read_text(encoding="utf-8") == "completed-resource"
    assert json.loads(claim.read_text())["active"] is None


def test_recovery_cli_requires_flag(tmp_path, monkeypatch):
    evidence = tmp_path / "note.txt"
    evidence.write_text("operator inspection", encoding="utf-8")
    argv = ["recover", "--data-dir", str(tmp_path), "--workspace", str(tmp_path),
            "--session", "cli:one", "--task-id", "old", "--reason", "checked",
            "--evidence-file", str(evidence)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 2
    monkeypatch.setattr(sys, "argv", argv + ["--confirm-stopped"])
    assert main() == 0
    assert SubagentReceiptStore(tmp_path).inspect("old", "cli:one", tmp_path)["worker_stopped"]


async def test_recovery_refreshes_existing_manager_failed_cleanup_cache(tmp_path):
    state = SubagentControlState(tmp_path)
    async def failure():
        raise RuntimeError("cleanup failed")
    task = asyncio.create_task(failure())
    state.register("id", "cli:one", tmp_path, task, status())
    await asyncio.gather(task, return_exceptions=True)
    state.finish("id")
    assert not state.finished["id"][2]["worker_stopped"]
    evidence = tmp_path / "verified.txt"
    evidence.write_text("external worker stopped by operator", encoding="utf-8")
    SubagentReceiptStore(tmp_path).reconcile(
        "id", "cli:one", tmp_path, confirmed_stopped=True, reason="verified",
        evidence_file=evidence,
    )
    assert (await state.execute("status", "cli:one", tmp_path, "id", 0))["worker_stopped"]
    listing = await state.execute("list", "cli:one", tmp_path, None, 0)
    assert listing["tasks"][0]["worker_stopped"]


async def test_spawn_registration_failure_never_runs_unregistered_worker(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMProvider

    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    manager = SubagentManager(provider, tmp_path, MessageBus(), max_tool_result_chars=1000)
    manager._run_subagent = AsyncMock()
    monkeypatch.setattr(manager._task_control.store, "_write",
                        lambda _: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        await manager.spawn("task", session_key="cli:one")
    await asyncio.sleep(0)
    manager._run_subagent.assert_not_awaited()
    assert not manager._running_tasks and not manager._task_statuses and not manager._session_tasks
