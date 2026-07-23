"""Helpers for deterministic recovery of interrupted user work."""

from __future__ import annotations

import re

AMBIGUOUS_RESUME_META_KEY = "_ambiguous_resume_request"

# Deliberately narrow: named requests such as ``继续重构登录模块`` carry their
# own objective and must not be blocked. These phrases ask the agent to recover
# an unspecified prior objective from session state.
_AMBIGUOUS_RESUME_RE = re.compile(
    r"^(?:请)?(?:继续|接着|恢复)(?:一下)?(?:上次|之前|刚才)?(?:的)?"
    r"(?:中断(?:的)?|未完成(?:的)?|暂停(?:的)?)?(?:任务|工作|计划)?[。.!！?？]*$"
)


def is_ambiguous_resume_request(text: str | None) -> bool:
    """Return whether *text* asks to resume without naming an objective."""
    normalized = "".join(str(text or "").strip().split())
    return bool(normalized and _AMBIGUOUS_RESUME_RE.fullmatch(normalized))
