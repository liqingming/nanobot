"""区分整轮消耗与最近一次上下文占用，禁止用累计输入量计算压力。"""

from collections.abc import Callable
from typing import Any

_CONTEXT_FIELDS = frozenset({"context_input_tokens", "context_input_estimated"})


def with_context_usage(
    usage: dict[str, int],
    diagnostics: dict[str, Any],
    estimate: Callable[[], int],
) -> dict[str, int]:
    result = dict(usage)
    measured = diagnostics.get("context_input_tokens")
    if isinstance(measured, int) and measured >= 0:
        tokens, estimated = measured, 0
    elif diagnostics.get("transport") == "codex_app_server":
        # Codex 的累计量差分可能跨越多次内部请求；缺少 last 时只能标记本地估算。
        tokens, estimated = estimate(), 1
    else:
        tokens = usage.get("prompt_tokens", 0)
        estimated = int(bool(usage.get("estimated_tokens")))
    result["context_input_tokens"] = max(0, tokens)
    result["context_input_estimated"] = estimated
    return result


def accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        target[key] = value if key in _CONTEXT_FIELDS else target.get(key, 0) + value


def context_input_tokens(usage: dict[str, int] | None) -> int:
    # 旧累计 usage 没有独立上下文字段时视为未知，不以累计消耗冒充占用。
    return max(0, (usage or {}).get("context_input_tokens", 0))
