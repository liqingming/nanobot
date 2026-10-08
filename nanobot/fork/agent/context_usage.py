"""区分整轮消耗与最近一次上下文占用，禁止用累计输入量计算压力。"""

from collections.abc import Callable
from typing import Any

from nanobot.providers.base import LLMUsage

_CONTEXT_FIELDS = frozenset({"context_input_tokens", "context_input_estimated"})


def with_context_usage(
    usage: LLMUsage | dict[str, int] | None,
    diagnostics: dict[str, Any],
    estimate: Callable[[], int],
) -> dict[str, int]:
    if isinstance(usage, LLMUsage):
        result = {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "reported_tokens": usage.reported_tokens,
            "estimated_tokens": usage.estimated_tokens,
        }
        if usage.cache_read_tokens is not None:
            result["cached_tokens"] = usage.cache_read_tokens
        if usage.cache_write_tokens is not None:
            result["cache_creation_input_tokens"] = usage.cache_write_tokens
    else:
        result = dict(usage or {})
    measured = diagnostics.get("context_input_tokens")
    if isinstance(measured, int) and measured >= 0:
        tokens, estimated = measured, int(bool(diagnostics.get("context_input_estimated")))
    elif diagnostics.get("transport") == "codex_app_server":
        # Codex 的累计量差分可能跨越多次内部请求；缺少 last 时只能标记本地估算。
        tokens, estimated = estimate(), 1
    else:
        tokens = (
            usage.context_tokens
            if isinstance(usage, LLMUsage) and usage.context_tokens is not None
            else result.get("prompt_tokens", 0)
        )
        estimated = int(bool(result.get("estimated_tokens")))
    result["context_input_tokens"] = max(0, tokens)
    result["context_input_estimated"] = estimated
    return result


def accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        target[key] = value if key in _CONTEXT_FIELDS else target.get(key, 0) + value


def context_input_tokens(usage: LLMUsage | dict[str, int] | None) -> int:
    # 旧累计 usage 没有独立上下文字段时视为未知，不以累计消耗冒充占用。
    if isinstance(usage, LLMUsage):
        return max(0, usage.context_tokens or 0)
    return max(0, (usage or {}).get("context_input_tokens", 0))
