"""子任务的模型传输身份；不改变话题归属或消息路由。"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec

_execution_id: ContextVar[str | None] = ContextVar("subagent_execution_id", default=None)


def current_execution_id() -> str | None:
    return _execution_id.get()


async def run_isolated_subagent(runner: AgentRunner, spec: AgentRunSpec) -> AgentRunResult:
    """每次运行独立命名空间；退出时仅清理支持此能力的 provider 的自有连接。"""
    execution_id = uuid.uuid4().hex
    token = _execution_id.set(execution_id)
    provider = runner.provider
    try:
        return await runner.run(spec)
    finally:
        try:
            close = getattr(provider, "aclose_execution", None)
            if callable(close):
                await close(execution_id)
        finally:
            _execution_id.reset(token)
