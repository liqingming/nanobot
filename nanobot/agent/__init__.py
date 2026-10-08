"""Agent core exports, loaded lazily to avoid fork-registration import cycles."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentHook",
    "AgentHookContext",
    "AgentRunHookContext",
    "AgentTurnHookContext",
    "AgentTurnHookFactory",
    "AgentLoop",
    "CompositeHook",
    "ContextBuilder",
    "MemoryStore",
    "SkillsLoader",
    "SubagentManager",
]

_MODULES = {
    "ContextBuilder": ".context",
    "AgentLoop": ".loop",
    "MemoryStore": ".memory",
    "SkillsLoader": ".skills",
    "SubagentManager": ".subagent",
}
if TYPE_CHECKING:
    from nanobot.agent.context import ContextBuilder
    from nanobot.agent.hook import (
        AgentHook,
        AgentHookContext,
        AgentRunHookContext,
        AgentTurnHookContext,
        AgentTurnHookFactory,
        CompositeHook,
    )
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.memory import MemoryStore
    from nanobot.agent.skills import SkillsLoader
    from nanobot.agent.subagent import SubagentManager


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULES.get(name, ".hook"), __name__), name)
