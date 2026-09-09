"""Codex 原生自动压缩的预算和事件状态，不在等待工具回执时强插手动压缩。

协议依据：Codex 0.153.4 的本机 JSON Schema，以及官方 config-reference / app-server。
原生压缩失败时不能通过重建线程重放副作用；配置仅作用于临时线程，不写用户配置。
"""

from dataclasses import dataclass, field
from typing import Any

from nanobot.providers.base import provider_input_token_budget
from nanobot.utils.helpers import estimate_prompt_tokens_chain


def native_input_budget(provider: Any, context: dict[str, Any] | None) -> int:
    settings = (context or {}).get("native_context")
    if not isinstance(settings, dict):
        return 0
    window = settings.get("context_window_tokens")
    if type(window) is not int or window <= 0:
        return 0
    output = settings.get("max_tokens")
    if type(output) is not int:
        output = getattr(getattr(provider, "generation", None), "max_tokens", 4096)
    output = output if type(output) is int else 4096
    budget = provider_input_token_budget(provider, window, output)
    if budget <= 0:
        raise ValueError("Codex native context has no input budget after output reservation.")
    limit = settings.get("context_block_limit")
    if type(limit) is int and limit > 0:
        budget = min(budget, limit)
    return budget


def check_native_payload(
    provider: Any, model: str, messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None, budget: int,
) -> None:
    if budget <= 0:
        return
    estimate, _ = estimate_prompt_tokens_chain(provider, model, messages, tools or [])
    if estimate > budget:
        raise ValueError(
            f"Codex native context input exceeds local budget ({estimate} > {budget}); "
            "reduce the initial context or read a smaller evidence range. "
            "The active thread will not be rebuilt automatically."
        )


@dataclass
class NativeCompactionState:
    budget: int = 0
    thread_id: str | None = None
    started: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)

    @property
    def in_progress(self) -> bool:
        return bool(self.started - self.completed)

    def config(self) -> dict[str, Any]:
        if self.budget <= 0:
            return {}
        # 不覆盖模型窗口、压缩提示词或用户的工具输出配置，更不抬高模型容量。
        return {
            "model_auto_compact_token_limit": max(1, self.budget * 4 // 5),
            "model_auto_compact_token_limit_scope": "total",
        }

    def observe(self, method: str, params: dict[str, Any]) -> bool:
        if params.get("threadId") != self.thread_id or self.thread_id is None:
            return False
        item = params.get("item")
        if not isinstance(item, dict) or item.get("type") != "contextCompaction":
            return False
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return False
        if method == "item/started":
            self.started.add(item_id)
        elif method == "item/completed":
            self.completed.add(item_id)
        else:
            return False
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "context_management": "codex_native_auto",
            "native_auto_compact_token_limit": self.config().get("model_auto_compact_token_limit"),
            "native_compactions_started": len(self.started),
            "native_compactions_completed": len(self.completed),
            "native_compaction_in_progress": self.in_progress,
        }
