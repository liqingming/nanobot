"""Tests for AgentLoop.warmup_caches (问题1: 进入会话/切换话题缓存预热).

warmup_caches must (a) touch every lazy cache warm point and (b) be strictly
best-effort — any failure is swallowed so it can never break entering a
session or switching topics. We drive it with a lightweight stand-in as
``self`` to avoid building a full AgentLoop.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from nanobot.agent.loop import AgentLoop


async def test_warmup_caches_triggers_all_warm_points():
    skills = MagicMock()
    memory = MagicMock()
    topic_store = MagicMock()
    context = SimpleNamespace(
        skills=skills,
        memory=memory,
        _get_topic_store=lambda _sk: topic_store,
    )
    search = MagicMock()
    tools = MagicMock()
    tools.tool_names = ["search_history"]
    tools.get.return_value = search

    loop_obj = SimpleNamespace(context=context, tools=tools)

    # Call the coroutine with our stand-in as `self`.
    await AgentLoop.warmup_caches(loop_obj, "cli:topic1")

    skills.build_skills_summary.assert_called_once()
    memory.read_memory.assert_called_once()
    topic_store.read_memory.assert_called_once()
    search._load_index.assert_called_once()


async def test_warmup_caches_skips_search_when_not_registered():
    context = SimpleNamespace(
        skills=MagicMock(),
        memory=MagicMock(),
        _get_topic_store=lambda _sk: None,  # topic memory disabled
    )
    tools = MagicMock()
    tools.tool_names = []  # search_history not registered (enable_learning off)

    loop_obj = SimpleNamespace(context=context, tools=tools)

    await AgentLoop.warmup_caches(loop_obj, "cli:topic1")

    tools.get.assert_not_called()  # guarded by `in tool_names`


async def test_warmup_caches_swallows_all_errors():
    # Every warm point blows up — warmup must still complete without raising.
    skills = MagicMock()
    skills.build_skills_summary.side_effect = RuntimeError("boom")
    memory = MagicMock()
    memory.read_memory.side_effect = RuntimeError("boom")
    context = SimpleNamespace(
        skills=skills,
        memory=memory,
        _get_topic_store=MagicMock(side_effect=RuntimeError("boom")),
    )
    tools = MagicMock()
    tools.tool_names = ["search_history"]
    tools.get.return_value = MagicMock(_load_index=MagicMock(side_effect=RuntimeError("boom")))

    loop_obj = SimpleNamespace(context=context, tools=tools)

    # Must not raise.
    await AgentLoop.warmup_caches(loop_obj, "cli:topic1")
