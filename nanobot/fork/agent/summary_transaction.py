"""会话摘要的覆盖事务；原文归档不等于可安全移除上下文。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from nanobot.fork.agent.recovery_packet import recovery_sha256, render_recovery_context

SUMMARY_MAX_CHARS = 8000
COVERAGE_KEY = "_context_compaction"


class SummaryTransactionError(RuntimeError):
    """摘要未能安全替代原文；调用方必须停止而非静默裁剪。"""


def valid_summary(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip() != "(nothing)"
        and len(value) <= SUMMARY_MAX_CHARS
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class SummarySnapshot:
    messages: list[dict]
    metadata: dict
    todos: list[dict]
    cursor: int
    updated_at: Any

    @classmethod
    def capture(cls, session: Any) -> SummarySnapshot:
        return cls(
            deepcopy(session.messages), deepcopy(session.metadata),
            deepcopy(session.todos), session.last_consolidated, session.updated_at,
        )

    def matches(self, session: Any) -> bool:
        return (
            self.messages == session.messages and self.metadata == session.metadata
            and self.todos == session.todos and self.cursor == session.last_consolidated
            and self.updated_at == session.updated_at
        )


def commit_summary(
    sessions: Any,
    session: Any,
    snapshot: SummarySnapshot,
    summary: str,
    *,
    covered_messages: list[dict],
    end_cursor: int,
    retained_messages: list[dict] | None = None,
    reason: str,
) -> None:
    """在调用者持有会话锁时提交；没有 await，保存失败回滚内存视图。"""
    if not valid_summary(summary):
        raise SummaryTransactionError("摘要无效或超长，原始上下文保持不变。")
    if sessions.get_or_create(session.key) is not session or not snapshot.matches(session):
        raise SummaryTransactionError("摘要期间会话发生变化，拒绝提交过期覆盖。")
    if not snapshot.cursor < end_cursor <= len(snapshot.messages) or not covered_messages:
        raise SummaryTransactionError("摘要覆盖边界无效，原始上下文保持不变。")
    source = snapshot.messages[snapshot.cursor:end_cursor]
    if retained_messages is None:
        covered = covered_messages == source
    else:
        # 合法尾部修复可能非连续；用多重集验证每条原文恰好被保留或覆盖。
        covered = end_cursor == len(snapshot.messages) and (
            Counter(map(_hash, covered_messages + retained_messages))
            == Counter(map(_hash, source))
        )
    if not covered:
        raise SummaryTransactionError("摘要覆盖与实际移除消息不一致，拒绝提交。")
    previous = snapshot.metadata.get(COVERAGE_KEY, {})
    previous = previous if isinstance(previous, dict) else {}
    version = previous.get("version", 0)
    if type(version) is not int or version < 0:
        raise SummaryTransactionError("摘要覆盖版本无效，拒绝覆盖。")
    entry = {"text": summary, "last_active": session.updated_at.isoformat()}
    metadata = deepcopy(session.metadata)
    metadata["_continuation_summary"] = entry
    metadata["_last_summary"] = entry
    recovery = render_recovery_context(metadata, todos=session.todos, legacy_summary=summary)
    metadata[COVERAGE_KEY] = {
        "version": version + 1,
        "previous_version": version,
        "reason": reason,
        "source_cursor": snapshot.cursor,
        "source_message_count": len(snapshot.messages),
        "covered_message_count": len(covered_messages),
        "covered_sha256": _hash(covered_messages),
        "summary_sha256": _hash(summary),
        "recovery_sha256": recovery_sha256(recovery),
        "prior_summary_sha256": _hash(snapshot.metadata.get("_continuation_summary")),
        "cursor": end_cursor if retained_messages is None else 0,
    }
    old_messages, old_metadata, old_cursor = (
        session.messages, session.metadata, session.last_consolidated,
    )
    session.metadata = metadata
    session.messages = (
        deepcopy(retained_messages) if retained_messages is not None else session.messages
    )
    session.last_consolidated = end_cursor if retained_messages is None else 0
    try:
        sessions.save(session, fsync=True)
    except BaseException as exc:
        # replace 已完成后的 fsync/日志也可能失败，不能盲目回滚成旧内存版本。
        try:
            path = sessions._get_session_path(session.key)
            try:
                with path.open(encoding="utf-8") as stream:
                    persisted = json.loads(stream.readline())
                committed = persisted.get("metadata", {}).get(COVERAGE_KEY) == metadata[COVERAGE_KEY]
            except FileNotFoundError:
                committed = False
        except Exception:
            sessions.invalidate(session.key)
            raise SummaryTransactionError(
                "摘要保存结果无法确认，已停止；重新读取持久化状态后才能恢复。"
            ) from exc
        if not committed:
            session.messages, session.metadata, session.last_consolidated = (
                old_messages, old_metadata, old_cursor,
            )
        sessions.invalidate(session.key)
        raise
