"""治理统计口径及 runner 读取模式的真实接入回归。"""

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.file_state import FileStates
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.fork.agent.governance_metrics import GovernanceMetrics, projection_changes
from nanobot.fork.agent.read_visibility import read_evidence_required
from nanobot.fork.agent.subagent_diagnostics import SubagentDiagnostics
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def test_repeated_projection_is_not_unique_savings_or_context_peak():
    metrics = GovernanceMetrics()
    for _ in range(68):
        metrics.observe_projection(42459)
    metrics.observe_usage({"prompt_tokens": 914662, "context_input_tokens": 130932,
                           "context_input_estimated": 0})
    metrics.observe_usage({"context_input_tokens": 250000, "context_input_estimated": 1})
    metrics.observe_usage({"context_input_tokens": 195900, "context_input_estimated": 0})
    metrics.observe_usage({"prompt_tokens": 1000000})
    summary = metrics.summary(native=True)
    assert summary["local_projection_reduction_tokens"] == 42459
    assert summary["local_projection_reduction_peak_tokens"] == 42459
    assert summary["local_projection_reduction_token_observations"] == 2887212
    assert summary["context_input_peak_tokens"] == 195900
    assert summary["context_input_estimated_peak_tokens"] == 250000
    assert summary["context_input_measured_samples"] == 2
    assert summary["context_input_estimated_samples"] == 1
    assert summary["context_input_unknown_samples"] == 1
    assert summary["prompt_peak_tokens"] == 1000000
    assert summary["prompt_peak_tokens_semantics"] == "response_usage_not_context_peak"
    assert summary["projection_metrics_semantics"] == "local_copy_not_remote_or_billing"


def test_unknown_peak_and_projection_counts_do_not_modify_source():
    metrics = GovernanceMetrics()
    assert metrics.summary(native=False)["context_input_peak_tokens"] is None
    metrics.observe_projection(10)
    metrics.observe_projection(0)
    assert metrics.local_projection_reduction_tokens == 0
    assert metrics.local_projection_reduction_peak_tokens == 10
    before = [{"role": "tool", "tool_call_id": str(i), "content": "body"} for i in range(3)]
    original = deepcopy(before)
    after = [dict(before[0]), {**before[1], "content": "bounded preview"}]
    assert projection_changes(before, after) == {
        "projection_omitted_tool_results": 1, "projection_changed_tool_results": 1,
    }
    assert before == original


@pytest.mark.parametrize("strategy,native", [("legacy", False), ("transactional", False),
                                             ("transactional", True)])
async def test_real_runner_reloads_body_and_reports_separate_peaks(tmp_path, strategy, native):
    path = tmp_path / "evidence.txt"
    path.write_text("evidence body", encoding="utf-8")
    tool = ReadFileTool(workspace=tmp_path, file_states=FileStates())
    await tool.execute(path.name)
    assert "File unchanged" in await tool.execute(path.name)
    registry = ToolRegistry()
    registry.register(tool)
    provider = MagicMock(spec=LLMProvider)
    provider.supports_native_context_compaction = native
    provider.supports_request_context = True
    responses = []
    for i in range(3):
        responses.append(LLMResponse(
            content="done" if i == 2 else None,
            tool_calls=[] if i == 2 else [ToolCallRequest(
                id=f"read-{i}", name="read_file", arguments={"path": path.name},
            )],
            usage={"prompt_tokens": 900000, "completion_tokens": 10},
            provider_diagnostics={"context_input_tokens": 100 + i},
        ))
    provider.chat_with_retry = AsyncMock(side_effect=responses)
    events = []
    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "system", "content": "system"},
                          {"role": "user", "content": "read twice"}],
        tools=registry, model="test", max_iterations=3, max_tool_result_chars=16000,
        context_strategy=strategy, workspace=tmp_path, data_dir=tmp_path / "data",
        session_key="test", event_logger=lambda event, fields: events.append((event, fields)),
    ))
    assert result.stop_reason == "completed"
    reads = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert len(reads) == 2
    assert all("1| evidence body" in value and "File unchanged" not in value for value in reads)
    assert not read_evidence_required()
    summary = next(fields for event, fields in events if event == "runner.context.turn_summary")
    assert summary["context_input_peak_tokens"] == 102
    assert summary["response_prompt_usage_peak_tokens"] == 900000
    assert summary["context_input_measured_samples"] == 3
    assert summary["context_input_estimated_peak_tokens"] is None
    assert summary["context_scope"] == ("native_checkpoint_copy" if native else "local_model_copy")
    assert result.usage["context_input_tokens"] == 102
    # 原始累计消耗口径不被观察统计改变。
    assert result.usage["prompt_tokens"] == 2700000


@pytest.mark.parametrize("sync", ["start", "steered"])
def test_child_audit_preserves_new_metrics_and_sync_without_raw_text(tmp_path, sync):
    audit = SubagentDiagnostics(tmp_path, task_id="a" * 32, parent_session_key="parent",
                                model="test", provider=object())
    metrics = GovernanceMetrics()
    metrics.observe_projection(42)
    audit("runner.context.turn_summary", {**metrics.summary(native=True), "content": "SECRET"})
    audit("runner.model.response", {
        "provider_diagnostics": {"context_sync": sync, "raw": "SECRET"},
        "usage": {"cached_tokens": 12, "reasoning_tokens": 3},
    })
    text = audit.path.read_text(encoding="utf-8")
    assert "SECRET" not in text
    summary, response = map(json.loads, text.splitlines())
    assert summary["context_input_peak_tokens"] is None
    assert summary["local_projection_reduction_peak_tokens"] == 42
    assert summary["projection_metrics_semantics"] == "local_copy_not_remote_or_billing"
    assert response["provider_diagnostics"]["context_sync"] == sync
    assert response["usage"] == {"cached_tokens": 12, "reasoning_tokens": 3}
