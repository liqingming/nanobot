"""原生首次整形保留超过 64 条的旧证据，但仍执行软预算及硬门禁。"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context_governance import ContextGovernanceConfig, ContextGovernor
from nanobot.fork.agent.native_context import NativeContextPreparation
from nanobot.fork.providers.codex_app_server_provider import CodexAppServerProvider
from nanobot.fork.providers.codex_native_context import check_native_payload
from nanobot.utils.helpers import estimate_prompt_tokens_chain


def _config(tmp_path, boundary, *, budget=100000):
    # 离线计数器：测试预算路径而非特定 tokenizer 的字节压缩比。
    provider = SimpleNamespace(
        estimate_prompt_tokens=lambda messages, tools, model: (
            sum(len(str(m.get("content", ""))) // 4 + 8 for m in messages), "test",
        ),
    )
    return ContextGovernanceConfig(
        provider=provider, model="test",
        tools=SimpleNamespace(get_definitions=lambda: []),
        workspace=tmp_path, data_dir=tmp_path, session_key="visibility",
        max_tool_result_chars=16000, context_window_tokens=200000,
        context_block_limit=budget, max_tokens=1000, inflight_start_index=boundary,
    )


def _history(count, size):
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "old task"}]
    for index in range(count):
        call_id = f"read-{index}"
        messages.extend([
            {"role": "assistant", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "name": "read_file", "tool_call_id": call_id,
             "content": f"evidence-{index}: " + "body " * (size // 5)},
        ])
    messages.append({"role": "user", "content": "continue"})
    return messages


def test_first_projection_keeps_more_than_64_old_bodies_and_stable_prefix(tmp_path):
    messages = _history(70, 100)
    original = deepcopy(messages)
    config = _config(tmp_path, len(messages))
    governor = ContextGovernor()
    # 证明本例真正覆盖原 64 条裁剪路径，而不是边界配置恰巧绕开它。
    assert len(governor.trim_historical_tool_exchanges(config, messages)) < len(messages)
    preparation = NativeContextPreparation(governor)
    compacted = set()
    projected = preparation.prepare_for_model(config, messages, compacted)
    assert projected == original
    assert not compacted
    assert config.inflight_start_index == len(messages)
    assert messages == original
    checkpoint_prefix = list(preparation.source_checkpoint.messages)
    messages.append({"role": "user", "content": "new appended instruction"})
    again = preparation.prepare_for_model(config, messages, compacted)
    assert again[:-1] == projected
    assert preparation.source_checkpoint.messages[:-1] == checkpoint_prefix
    assert len(preparation.source_checkpoint.messages) == len(messages)
    assert messages[:-1] == original


def test_first_projection_still_applies_soft_budget_to_old_evidence(tmp_path):
    messages = _history(70, 600)
    original = deepcopy(messages)
    config = _config(tmp_path, len(messages))
    compacted = set()
    projected = NativeContextPreparation(ContextGovernor()).prepare_for_model(
        config, messages, compacted,
    )
    assert compacted
    assert len(projected) == len(messages)  # 软压缩不是删除工具交换。
    assert sum(len(m["content"]) for m in projected if m["role"] == "tool") < 24000
    assert messages == original


def test_first_projection_still_compacts_to_hard_budget(tmp_path):
    messages = _history(10, 1000)  # 总正文低于软阈值，只可能由硬预算触发。
    original = deepcopy(messages)
    config = _config(tmp_path, len(messages), budget=1200)
    governor = ContextGovernor()
    compacted = set()
    assert estimate_prompt_tokens_chain(config.provider, config.model, messages, [])[0] > 1200
    projected = NativeContextPreparation(governor).prepare_for_model(config, messages, compacted)
    assert compacted
    assert governor.input_budget(config) == governor.input_budget(
        replace(config, inflight_start_index=0),
    ) == 1200
    check_native_payload(config.provider, config.model, projected, [], 1200)
    assert messages == original


async def test_unshrinkable_first_projection_is_rejected_before_provider_launch(tmp_path, monkeypatch):
    provider = CodexAppServerProvider(idempotency_dir=tmp_path / "ledger")
    messages = [{"role": "system", "content": "huge evidence " * 10000}]
    config = replace(_config(tmp_path, len(messages), budget=1000), provider=provider)
    preparation = NativeContextPreparation(ContextGovernor())
    projected = preparation.prepare_for_model(config, messages, set())
    assert projected == messages
    with pytest.raises(ValueError, match="exceeds local budget"):
        check_native_payload(provider, config.model, projected, [], 1000)
    # 即便整形无法减到预算，实际 Provider 入口仍拒绝，不启动进程或模型。
    launch = AsyncMock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(
        "nanobot.fork.providers.codex_app_server_provider.asyncio.create_subprocess_exec", launch,
    )
    provider._app_server_command = ["offline-must-not-launch"]
    try:
        result = await provider.chat(
            messages=projected, model="test",
            request_context={"native_context": {
                "context_window_tokens": 200000, "context_block_limit": 1000, "max_tokens": 1000,
            }},
        )
        assert result.finish_reason == "error"
        assert not result.error_should_retry
        launch.assert_not_called()
        assert not (tmp_path / "ledger").exists()
    finally:
        await provider.aclose()
