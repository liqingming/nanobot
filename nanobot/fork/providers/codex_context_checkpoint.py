"""检测本地治理副本与有状态 Codex 线程是否仍可仅追加工具结果。

前缀变化必须在已落盘的工具边界重建，不能将本地 saved_total 当成服务端节省量。
具体重建与幂等回放仍由 provider 管理；原生命令、文件副作用不在动态工具账本内。
"""

import hashlib
import json
from typing import Any


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class ContextCheckpoint:
    """只保留哈希，不复制原始历史或在日志中泄漏内容。"""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.settings: str | None = None

    def capture(self, messages: list[dict[str, Any]], settings: Any) -> None:
        self.messages = [_fingerprint(message) for message in messages]
        self.settings = _fingerprint(settings)

    def needs_rebase(self, messages: list[dict[str, Any]], settings: Any) -> bool:
        if self.settings is None:
            return False
        if self.settings != _fingerprint(settings) or len(messages) < len(self.messages):
            return True
        if any(old != _fingerprint(new) for old, new in zip(self.messages, messages)):
            return True
        # 续传只能提交工具结果，新增用户指令也必须进入新线程。
        return any(message.get("role") in {"user", "system", "developer"}
                   for message in messages[len(self.messages):])


def pending_checkpoint_messages(
    messages: list[dict[str, Any]], pending_ids: set[str],
) -> list[dict[str, Any]]:
    """用户插话可能改变当前 user-tail，但不能丢掉已经执行的待提交结果。"""
    return [
        message for message in messages
        if (message.get("role") == "tool" and message.get("tool_call_id") in pending_ids)
        or (message.get("role") == "assistant" and any(
            call.get("id") in pending_ids for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        ))
    ]
