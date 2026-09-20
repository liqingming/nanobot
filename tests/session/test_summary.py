from datetime import datetime

import pytest

from nanobot.session.summary import session_summary_from_metadata


@pytest.mark.parametrize("continuation", [None, [], {}, {"text": ""}, {"text": 3}])
def test_invalid_continuation_falls_back_to_last_summary(continuation):
    fallback = datetime(2026, 9, 11)
    result = session_summary_from_metadata(
        {"_continuation_summary": continuation, "_last_summary": {"text": "saved"}},
        fallback_last_active=fallback,
    )
    assert result == {"text": "saved", "last_active": fallback.isoformat()}


def test_valid_continuation_takes_priority_and_validates_timestamp():
    fallback = datetime(2026, 9, 11)
    result = session_summary_from_metadata(
        {
            "_continuation_summary": {"text": "resume", "last_active": "invalid"},
            "_last_summary": {"text": "saved"},
        },
        fallback_last_active=fallback,
    )
    assert result == {"text": "resume", "last_active": fallback.isoformat()}
