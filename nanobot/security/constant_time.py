"""Constant-time comparison helpers for authentication secrets."""

from __future__ import annotations

import hmac


def constant_time_text_equal(left: str, right: str) -> bool:
    """Compare Unicode text as UTF-8 bytes without content-based short circuiting."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
