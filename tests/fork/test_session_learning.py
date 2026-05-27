"""Tests for AgentLoop.clear_session_learning (问题2: 切话题清理 learning 字典).

On CLI topic switch the loop drops the previous topic's per-session learning
state so the dicts don't grow unbounded. It must remove only the target key
(never another topic's) and be safe for keys that were never recorded.
"""

from types import SimpleNamespace

from nanobot.agent.loop import AgentLoop


def test_clear_session_learning_pops_only_target_key():
    loop_obj = SimpleNamespace(
        _last_turn_summary={"cli:a": "sumA", "cli:b": "sumB"},
        _prev_consolidated={"cli:a": 1, "cli:b": 2},
        _last_user_input={"cli:a": "x", "cli:b": "y"},
    )

    AgentLoop.clear_session_learning(loop_obj, "cli:a")

    # Target key dropped from all three dicts.
    assert "cli:a" not in loop_obj._last_turn_summary
    assert "cli:a" not in loop_obj._prev_consolidated
    assert "cli:a" not in loop_obj._last_user_input
    # Other topics untouched.
    assert loop_obj._last_turn_summary == {"cli:b": "sumB"}
    assert loop_obj._prev_consolidated == {"cli:b": 2}
    assert loop_obj._last_user_input == {"cli:b": "y"}


def test_clear_session_learning_safe_for_unknown_key():
    loop_obj = SimpleNamespace(
        _last_turn_summary={},
        _prev_consolidated={},
        _last_user_input={},
    )

    # Must not raise for a key that was never recorded.
    AgentLoop.clear_session_learning(loop_obj, "cli:never-seen")
