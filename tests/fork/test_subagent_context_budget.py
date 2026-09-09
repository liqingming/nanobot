"""子任务继承主模型预算，并向治理器传入硬限制。"""

from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider


async def test_child_spec_has_window_and_block_limit(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    manager = SubagentManager(
        provider, tmp_path, MessageBus(), 16000, model="test",
        context_window_tokens=8000, context_block_limit=3000,
    )
    manager._build_tools = lambda **kwargs: MagicMock()
    manager._build_subagent_prompt = lambda **kwargs: "test"
    manager.runner.run = AsyncMock(return_value=AgentRunResult(final_content="ok", messages=[]))
    await manager._run_subagent(
        "id", "task", "label",
        {"channel": "cli", "chat_id": "topic", "session_key": "cli:topic"},
        SubagentStatus(task_id="id", label="label", task_description="task", started_at=0),
    )
    spec = manager.runner.run.await_args.args[0]
    assert spec.context_window_tokens == 8000
    assert spec.context_block_limit == 3000


def test_default_and_model_switch_keep_a_budget(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    manager = SubagentManager(provider, tmp_path, MessageBus(), 16000, model="test")
    assert manager.context_window_tokens == AgentDefaults().context_window_tokens
    manager.set_provider(provider, "small", context_window_tokens=6000)
    assert manager.context_window_tokens == 6000
    manager.set_provider(provider, "next")
    assert manager.context_window_tokens == 6000


async def test_child_hard_budget_actually_shapes_large_tool_result(tmp_path):
    from nanobot.providers.base import LLMResponse, ToolCallRequest
    from nanobot.utils.helpers import estimate_prompt_tokens_chain

    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="read", name="read_file", arguments={"path": "receipt"}),
        ]),
        LLMResponse(content="done"),
    ])
    manager = SubagentManager(
        provider, tmp_path, MessageBus(), 16000, model="test",
        context_window_tokens=8000, context_block_limit=800,
    )
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="large proof " * 1000)
    manager._build_tools = lambda **kwargs: tools
    manager._build_subagent_prompt = lambda **kwargs: "test"
    await manager._run_subagent(
        "id", "task", "label",
        {"channel": "cli", "chat_id": "topic", "session_key": "cli:topic"},
        SubagentStatus(task_id="id", label="label", task_description="task", started_at=0),
    )
    assert provider.chat_with_retry.await_count == 2
    second = provider.chat_with_retry.await_args.kwargs["messages"]
    assert estimate_prompt_tokens_chain(provider, "test", second, [])[0] <= 800
    assert all("large proof " * 1000 != msg.get("content") for msg in second)
