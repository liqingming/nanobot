"""只读治理统计：本地投影差值不是远端压缩量，更不是费用节省。"""

from dataclasses import dataclass
from typing import Any


def projection_changes(
    before: list[dict[str, Any]], after: list[dict[str, Any]],
) -> dict[str, int]:
    """统计当前投影相对源记录的结果变化；非本轮新增压缩次数。"""
    source = {m["tool_call_id"]: m.get("content") for m in before
              if m.get("role") == "tool" and m.get("tool_call_id")}
    projected = {m["tool_call_id"]: m.get("content") for m in after
                 if m.get("role") == "tool" and m.get("tool_call_id")}
    return {
        "projection_omitted_tool_results": len(source.keys() - projected.keys()),
        "projection_changed_tool_results": sum(
            source[key] != projected[key] for key in source.keys() & projected.keys()
        ),
    }


@dataclass
class GovernanceMetrics:
    """每次 runner.run 独立；实际与估算峰值分开，未知不当作实际零占用。"""

    local_projection_reduction_tokens: int = 0
    local_projection_reduction_peak_tokens: int = 0
    local_projection_reduction_token_observations: int = 0
    context_input_peak_tokens: int | None = None
    context_input_estimated_peak_tokens: int | None = None
    context_input_measured_samples: int = 0
    context_input_estimated_samples: int = 0
    context_input_unknown_samples: int = 0
    response_prompt_usage_peak_tokens: int = 0

    def observe_projection(self, reduction: int) -> None:
        self.local_projection_reduction_tokens = max(0, reduction)
        self.local_projection_reduction_peak_tokens = max(
            self.local_projection_reduction_peak_tokens, max(0, reduction),
        )
        # 重复观察同一差值也累计，只表示 token×观察次数，不声称唯一节省量。
        self.local_projection_reduction_token_observations += max(0, reduction)

    def observe_usage(self, usage: dict[str, int]) -> None:
        prompt = usage.get("prompt_tokens")
        if type(prompt) is int and prompt >= 0:
            self.response_prompt_usage_peak_tokens = max(
                self.response_prompt_usage_peak_tokens, prompt,
            )
        tokens = usage.get("context_input_tokens")
        estimated = usage.get("context_input_estimated")
        if type(tokens) is not int or tokens < 0 or type(estimated) is not int or estimated not in (0, 1):
            self.context_input_unknown_samples += 1
        elif estimated:
            self.context_input_estimated_samples += 1
            self.context_input_estimated_peak_tokens = max(
                self.context_input_estimated_peak_tokens or 0, tokens,
            )
        else:
            self.context_input_measured_samples += 1
            self.context_input_peak_tokens = max(self.context_input_peak_tokens or 0, tokens)

    def summary(self, *, native: bool) -> dict[str, Any]:
        return {
            **vars(self),
            "context_scope": "native_checkpoint_copy" if native else "local_model_copy",
            "projection_metrics_semantics": "local_copy_not_remote_or_billing",
            # 兼容已有消费者；旧字段保留但明确其单位和含义。
            "governance_saved_total": self.local_projection_reduction_token_observations,
            "governance_saved_total_semantics": "repeated_local_token_observations",
            "prompt_peak_tokens": self.response_prompt_usage_peak_tokens,
            "prompt_peak_tokens_semantics": "response_usage_not_context_peak",
        }
