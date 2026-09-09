"""Fresh, bounded recovery reads must reach the next model decision intact."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from nanobot.agent.context_artifacts import ToolDigestBuilder
from nanobot.agent.context_governance import ContextGovernanceConfig, ContextGovernor


def _exchange(call_id, content, tool_name="read_file"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": '{"path":"review-contract.json","limit":40,"force":true}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": content},
    ]


def _config(*, budget=12_000):
    tools = MagicMock()
    tools.get_definitions.return_value = []
    return ContextGovernanceConfig(
        provider=MagicMock(),
        model="test-model",
        tools=tools,
        workspace=None,
        data_dir=None,
        session_key=None,
        max_tool_result_chars=16_000,
        context_window_tokens=100_000,
        context_block_limit=budget,
    )


@pytest.fixture
def char_estimates(monkeypatch):
    # Simulate history/tool-call arguments dominating even after all old results
    # are compacted; a fresh small exchange still fits on its own.
    def tokens(message):
        return len(str(message.get("content", ""))) + sum(
            len(str(call.get("function", {}).get("arguments", "")))
            for call in message.get("tool_calls", [])
        )

    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        lambda _provider, _model, messages, _tools: (
            sum(tokens(message) for message in messages), "test"
        ),
    )
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens", tokens,
    )


@pytest.mark.parametrize("tool_name", ["read_file", "grep"])
def test_small_recovery_read_survives_history_overflow_and_reread(char_estimates, tool_name):
    contract = 'reviewAttemptId=9745; sha256=1cc7; receipt=attempt/review-receipt.json\n' * 30
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "核对当前合同"},
        {"role": "assistant", "content": "historical generated payload " * 1000},
        *_exchange("old", "old evidence " * 500),
        *_exchange("fresh-1", contract, tool_name),
    ]
    governor = ContextGovernor()
    compacted_ids = set()
    digests = {}
    for call_id in ("fresh-1", "fresh-2"):
        if call_id == "fresh-2":
            messages.extend(_exchange(call_id, contract, tool_name))
        digest, _ = ToolDigestBuilder.build(
            tool_call_id=call_id,
            tool_name=tool_name,
            arguments={"path": "review-contract.json"},
            result=contract,
        )
        digests[call_id] = digest
        original = deepcopy(messages)

        result = governor.prepare_for_model(
            _config(), messages, compacted_ids, tool_digests=digests,
        )

        assert result[-1]["content"] == contract
        assert sum(len(str(msg.get("content", ""))) for msg in result) <= 12_000
        assert result[0] == original[0]
        assert result[1] == original[1]
        assert result[-2]["tool_calls"][0]["id"] == call_id
        assert call_id not in compacted_ids
        assert messages == original
        assert "old" in compacted_ids
    # Protection is for the newest exchange, not a permanent exemption for reads.
    assert "fresh-1" in compacted_ids


@pytest.mark.parametrize("hard_overflow", [False, True])
def test_entire_small_parallel_batch_is_protected(char_estimates, hard_overflow):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "read the six fields"},
    ]
    if hard_overflow:
        messages.append({"role": "assistant", "content": "old payload " * 2000})
    else:
        # Non-compactable content keeps soft pressure high even after all old
        # read results are reclaimed. More than KEEP_RECENT fresh calls remain.
        messages.extend(_exchange("skill", "instructions " * 2500, "load_skill"))
    for index in range(8):
        messages.extend(_exchange(f"old-{index}", "x" * 5000))
    batch = [_exchange(f"fresh-{index}", str(index) * 600) for index in range(6)]
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [pair[0]["tool_calls"][0] for pair in batch],
    })
    messages.extend(pair[1] for pair in batch)
    original = deepcopy(messages)
    compacted_ids = set()
    governor = ContextGovernor()

    if hard_overflow:
        result = governor.prepare_for_model(_config(), messages, compacted_ids)
    else:
        result = governor.compact_inflight_soft_budget(_config(), messages, compacted_ids)

    assert result[-6:] == original[-6:]
    if hard_overflow:
        assert sum(len(str(msg.get("content", ""))) for msg in result) <= 12_000
        assert result[-7] == original[-7]
    assert compacted_ids
    assert all(call_id.startswith("old-") for call_id in compacted_ids)
    assert messages == original


def test_oversized_fresh_result_still_compacts(char_estimates):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "read"},
        {"role": "assistant", "content": "old payload " * 3000},
        *_exchange("large", "x" * 20_000),
    ]
    compacted_ids = set()

    result = ContextGovernor().compact_inflight_overflow(
        _config(budget=30_000), messages, compacted_ids,
    )

    assert "compacted" in result[-1]["content"]
    assert compacted_ids == {"large"}


@pytest.mark.parametrize("large_part", ["system", "user", "tool_arguments"])
def test_small_batch_does_not_bypass_minimum_request_budget(char_estimates, large_part):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "read"},
        *_exchange("fresh", "x" * 600),
    ]
    if large_part == "tool_arguments":
        messages[-2]["tool_calls"][0]["function"]["arguments"] = "x" * 20_000
    else:
        index = 0 if large_part == "system" else 1
        messages[index]["content"] = "mandatory instructions " * 1000
    compacted_ids = set()

    result = ContextGovernor().compact_inflight_overflow(_config(), messages, compacted_ids)

    assert "compacted" in result[-1]["content"]
    assert compacted_ids == {"fresh"}


def test_fresh_batch_protection_has_aggregate_size_limit(char_estimates):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "read"},
        {"role": "assistant", "content": "old payload " * 3000},
    ]
    batch = [_exchange(f"large-{index}", "x" * 8000) for index in range(2)]
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [pair[0]["tool_calls"][0] for pair in batch],
    })
    messages.extend(pair[1] for pair in batch)
    compacted_ids = set()

    result = ContextGovernor().compact_inflight_overflow(
        _config(budget=30_000), messages, compacted_ids,
    )

    assert all("compacted" in msg["content"] for msg in result[-2:])
    assert compacted_ids == {"large-0", "large-1"}
