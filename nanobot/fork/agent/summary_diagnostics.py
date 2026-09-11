"""摘要失败的有界诊断；不把请求、响应正文或异常中的凭据写入日志。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.fork.agent.summary_transaction import SummaryTransactionError
from nanobot.utils.session_runtime_log import append_session_runtime_log


class SummaryValidationError(ValueError):
    """由本地摘要校验产生的稳定原因码，不接受模型生成的错误描述。"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def record_summary_failure(
    sessions: Any, *, session_key: str | None, provider: Any, model: str,
    stage: str, error: Exception, estimated_tokens: int | None,
    input_budget: int | None, response: Any = None, summary_chars: int | None = None,
) -> None:
    """仅追加诊断，不更改摘要状态；诊断存储失败不能影响原文保护路径。"""
    if isinstance(error, SummaryValidationError):
        reason = error.reason
    elif stage == "request" and isinstance(error, TimeoutError):
        reason = "request_timeout"
    elif stage == "request" and isinstance(error, SummaryTransactionError) and (
        getattr(provider, "supports_native_context_compaction", False) is True
    ):
        reason = "native_summary_unsupported"
    else:
        reason = f"{stage}_failed"
    finish_reason = getattr(response, "finish_reason", None)
    content = getattr(response, "content", None)
    tool_calls = getattr(response, "tool_calls", None)
    status = getattr(response, "error_status_code", None)
    if type(status) is not int:
        status = getattr(error, "status_code", None)
    error_code = getattr(response, "error_code", None)
    fields = {
        "session_key": session_key,
        "provider": type(provider).__name__,
        "model": model,
        "stage": stage,
        "reason": reason,
        "exception_type": type(error).__name__,
        "error_status_code": status if type(status) is int else None,
        "error_code": error_code if error_code in (
            None, "context_length_exceeded", "rate_limit_exceeded", "insufficient_quota",
            "invalid_api_key", "server_error", "model_not_found",
        ) else "other",
        "estimated_tokens": estimated_tokens,
        "input_budget": input_budget,
        "finish_reason": finish_reason if finish_reason in (
            None, "stop", "end_turn", "length", "error", "tool_calls", "content_filter",
        ) else "other",
        "response_chars": len(content) if isinstance(content, str) else None,
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else None,
        "summary_chars": summary_chars,
    }
    # 不输出 str(error) 或 traceback：SDK 异常可能嵌有请求正文、URL 和密钥。
    logger.warning("Context summary failed: {}", fields)
    if session_key:
        try:
            path = sessions.get_session_runtime_log_path(session_key)
            append_session_runtime_log(path, "context.summary.failed", **fields)
        except Exception:
            # 仍有普通日志；不能因观测失败而跳过归档或改变事务结果。
            logger.warning("Context summary diagnostic could not be persisted")
