"""验证 fork 执行作用域不泄漏，也不要求其他 provider 实现新接口。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.fork.agent.execution_scope import (
    current_execution_id,
    run_isolated_subagent,
)


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
async def test_execution_scope_restores_context_and_closes_owner(failure):
    close = AsyncMock()
    provider = SimpleNamespace(aclose_execution=close)
    seen = []

    async def run(spec):
        seen.append(current_execution_id())
        if failure is not None:
            raise failure()
        return spec

    runner = SimpleNamespace(provider=provider, run=run)
    for _ in range(2):
        if failure is None:
            assert await run_isolated_subagent(runner, "result") == "result"
        else:
            with pytest.raises(failure):
                await run_isolated_subagent(runner, "result")
        assert current_execution_id() is None
    assert all(seen)
    assert len(set(seen)) == 2
    assert [call.args[0] for call in close.await_args_list] == seen


async def test_nested_scope_restores_parent_without_provider_cleanup_capability():
    async def child_run(spec):
        assert current_execution_id() != spec
        return current_execution_id()

    child = SimpleNamespace(provider=object(), run=child_run)

    async def parent_run(spec):
        parent_id = current_execution_id()
        await run_isolated_subagent(child, parent_id)
        assert current_execution_id() == parent_id
        return spec

    parent = SimpleNamespace(provider=object(), run=parent_run)
    assert await run_isolated_subagent(parent, "ok") == "ok"
    assert current_execution_id() is None


async def test_execution_scope_restores_context_even_if_cleanup_fails():
    provider = SimpleNamespace(aclose_execution=AsyncMock(side_effect=RuntimeError("close")))
    runner = SimpleNamespace(provider=provider, run=AsyncMock(return_value="ok"))
    with pytest.raises(RuntimeError, match="close"):
        await run_isolated_subagent(runner, None)
    assert current_execution_id() is None
