"""有状态 Provider 的上下文分工：首次整形，随后仅约束新增证据。

已发送的前缀保持稳定；线程内历史由 Provider 原生压缩。此副本用于续传身份与
故障检查点，不代表远端压缩后的窗口。状态仅属于一次 runner.run，主子任务不共享。
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from nanobot.agent.context_artifacts import ToolDigest
from nanobot.agent.context_governance import ContextGovernanceConfig, ContextGovernor
from nanobot.fork.providers.codex_context_checkpoint import ContextCheckpoint
from nanobot.fork.providers.codex_native_context import native_input_budget
from nanobot.utils.helpers import maybe_persist_tool_result


def uses_native_context(provider: Any) -> bool:
    # 必须显式声明，避免 MagicMock 或普通/旧版 Provider 意外启用。
    return getattr(provider, "supports_native_context_compaction", False) is True


class NativeContextPreparation:
    def __init__(self, governor: ContextGovernor):
        self.governor = governor
        self.source_checkpoint = ContextCheckpoint()
        self.projected: list[dict[str, Any]] = []

    def prepare_for_model(
        self, config: ContextGovernanceConfig, messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str], *, tool_digests: dict[str, ToolDigest] | None = None,
    ) -> list[dict[str, Any]]:
        settings = (
            config.model, config.context_window_tokens, config.context_block_limit,
            config.max_tokens, config.max_tool_result_chars, config.tools.get_definitions(),
        )
        if self.source_checkpoint.settings is None or self.source_checkpoint.needs_rebase(
            messages, settings,
        ):
            # 真正的用户插话、设置变化或历史编辑不能被稳定前缀掩盖。
            # Provider 的检查点/幂等规则仍决定能否重建。
            projected = self.governor.prepare_for_model(
                config, messages, compacted_tool_call_ids, tool_digests=tool_digests,
            )
        else:
            suffix = messages[len(self.source_checkpoint.messages):]
            suffix = self.governor.apply_tool_result_budget(config, suffix)
            budget = native_input_budget(config.provider, {"native_context": {
                "context_window_tokens": config.context_window_tokens,
                "context_block_limit": config.context_block_limit,
                "max_tokens": config.max_tokens,
            }})
            if budget > 0:
                # 续传只发送结果，不把已知的调用参数和工具定义重复计入新增预算。
                batch = [message for message in suffix if message.get("role") == "tool"]
                batch_config = replace(
                    config, context_block_limit=max(1, budget // 5), inflight_start_index=0,
                    tools=SimpleNamespace(get_definitions=lambda: []),
                )
                bounded, _ = self.governor._compact_inflight_overflow_with_estimate(
                    batch_config, batch, set(), tool_digests=tool_digests,
                )
                replacements = {}
                for original, compacted in zip(batch, bounded):
                    if original == compacted:
                        continue
                    call_id = str(original.get("tool_call_id") or "")
                    locator = self.governor.persisted_result_locator(original.get("content"))
                    if not locator:
                        saved = maybe_persist_tool_result(
                            config.workspace, config.session_key, call_id, original.get("content"),
                            max_chars=1, data_dir=config.data_dir,
                        )
                        locator = self.governor.persisted_result_locator(saved)
                    if locator:
                        replacements[call_id] = {
                            **compacted,
                            "content": compacted["content"] + "\nFull output saved to: " + locator,
                        }
                    # 无法保留可读证据则不压缩，让 Provider 的预算门禁明确停止。
                suffix = [replacements.get(str(message.get("tool_call_id")), message)
                          if message.get("role") == "tool" else message for message in suffix]
            projected = self.projected + suffix
        self.source_checkpoint.capture(messages, settings)
        self.projected = deepcopy(projected)
        return deepcopy(projected)
