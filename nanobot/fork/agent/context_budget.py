"""统一预算口径；未知窗口与已耗尽预算必须可以区分。"""

from dataclasses import asdict, dataclass
from typing import Any

from nanobot.providers.base import provider_input_token_budget


class ContextBudgetError(ValueError):
    """已知预算耗尽时停止请求，不能降级成无预算限制。"""


@dataclass(frozen=True, slots=True)
class ContextBudget:
    window_tokens: int | None
    output_reserve: int
    safety_reserve: int
    provider_budget: int | None
    block_limit: int | None
    input_tokens: int | None

    def diagnostics(self) -> dict[str, Any]:
        # 仅记录数值，不携带提示词、工具回执或证据路径。
        return asdict(self)


def resolve_context_budget(
    provider: Any,
    window_tokens: int | None,
    max_tokens: int | None = None,
    block_limit: int | None = None,
    safety_buffer: int = 1024,
) -> ContextBudget:
    output = max_tokens
    if type(output) is not int:
        output = getattr(getattr(provider, "generation", None), "max_tokens", 4096)
    output = max(0, output) if type(output) is int else 4096
    safety = max(0, safety_buffer)
    limit = block_limit if type(block_limit) is int and block_limit > 0 else None
    if type(window_tokens) is not int or window_tokens <= 0:
        return ContextBudget(None, output, safety, None, limit, None)
    available = provider_input_token_budget(provider, window_tokens, output, safety)
    effective = min(available, limit) if limit is not None else available
    return ContextBudget(window_tokens, output, safety, available, limit, effective)
