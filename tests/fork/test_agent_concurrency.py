"""Tests for the fork-local cross-process AgentLoop concurrency gate."""

import asyncio

import pytest

from nanobot.fork.agent_concurrency import AgentConcurrencyConfig, AgentConcurrencyGate


def make_gate(tmp_path, *, queue_timeout=1.0):
    return AgentConcurrencyGate(
        AgentConcurrencyConfig(
            enabled=True,
            max_sessions=1,
            state_path=tmp_path / "concurrency.sqlite3",
            queue_timeout_seconds=queue_timeout,
            lease_timeout_seconds=5,
            heartbeat_interval_seconds=0.5,
            poll_interval_seconds=0.01,
        )
    )


@pytest.mark.asyncio
async def test_gate_limits_independent_process_instances(tmp_path):
    first = make_gate(tmp_path)
    second = make_gate(tmp_path, queue_timeout=0.05)

    first_lease = await first.acquire("api:first")
    assert first_lease is not None
    try:
        assert await second.acquire("api:second") is None
    finally:
        await first_lease.release()

    second_lease = await second.acquire("api:second")
    assert second_lease is not None
    await second_lease.release()


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed(tmp_path):
    first = make_gate(tmp_path)
    second = make_gate(tmp_path, queue_timeout=5)
    first_lease = await first.acquire("api:first")
    assert first_lease is not None
    try:
        waiter = asyncio.create_task(second.acquire("api:second"))
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        await first_lease.release()

    lease = await second.acquire("api:third")
    assert lease is not None
    await lease.release()
