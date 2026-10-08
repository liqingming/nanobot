"""子任务继承主模型预算，并向治理器传入硬限制。"""

from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.utils.llm_runtime import LLMRuntime


async def test_child_spec_has_window_and_block_limit(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings(max_tokens=100)
    manager = SubagentManager(
        workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=16000,
        context_block_limit=3000,
    )
    manager._build_tools = lambda **kwargs: MagicMock()
    manager._build_subagent_prompt = lambda **kwargs: "test"
    manager.runner.run = AsyncMock(return_value=AgentRunResult(final_content="ok", messages=[]))
    await manager._run_subagent(
        "id", "task", "label",
        {"channel": "cli", "chat_id": "topic", "session_key": "cli:topic"},
        SubagentStatus(task_id="id", label="label", task_description="task", started_at=0),
        LLMRuntime.capture(provider, "test", context_window_tokens=8000),
    )
    spec = manager.runner.run.await_args.args[0]
    assert spec.context_window_tokens == 8000
    assert spec.context_block_limit == 3000


async def test_child_budget_is_bound_to_each_runtime(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings()
    manager = SubagentManager(workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=16000)
    manager._build_tools = lambda **kwargs: MagicMock()
    manager._build_subagent_prompt = lambda **kwargs: "test"
    manager.runner.run = AsyncMock(return_value=AgentRunResult(final_content="ok", messages=[]))
    specs = []
    for model, window in [("default", AgentDefaults().context_window_tokens), ("small", 6000)]:
        runtime = LLMRuntime.capture(provider, model, context_window_tokens=window)
        await manager._run_subagent(
            model, "task", "label", {"channel": "cli", "chat_id": "topic"},
            SubagentStatus(task_id=model, label="label", task_description="task", started_at=0),
            runtime,
        )
        spec = manager.runner.run.await_args.args[0]
        assert spec.runtime is runtime
        assert spec.context_window_tokens == window
        specs.append(spec)
    assert specs[0].context_window_tokens == AgentDefaults().context_window_tokens


async def test_child_hard_budget_actually_shapes_large_tool_result(tmp_path):
    from nanobot.providers.base import LLMResponse, ToolCallRequest
    from nanobot.utils.helpers import estimate_prompt_tokens_chain

    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings(max_tokens=100)
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="read", name="read_file", arguments={"path": "receipt"}),
        ]),
        LLMResponse(content="done"),
    ])
    manager = SubagentManager(
        workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=16000,
        context_block_limit=800,
    )
    manager.context_strategy = "legacy"  # 旧策略回退仍可整形最新大回执。
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="large proof " * 1000)
    manager._build_tools = lambda **kwargs: tools
    manager._build_subagent_prompt = lambda **kwargs: "test"
    await manager._run_subagent(
        "id", "task", "label",
        {"channel": "cli", "chat_id": "topic", "session_key": "cli:topic"},
        SubagentStatus(task_id="id", label="label", task_description="task", started_at=0),
        LLMRuntime.capture(provider, "test", context_window_tokens=8000),
    )
    assert provider.chat_stream_with_retry.await_count == 2
    second = provider.chat_stream_with_retry.await_args.kwargs["messages"]
    assert estimate_prompt_tokens_chain(provider, "test", second, [])[0] <= 800
    assert all("large proof " * 1000 != msg.get("content") for msg in second)
