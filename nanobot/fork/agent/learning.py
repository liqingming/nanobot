"""Turn-level learning context for LLM-driven self-improvement.

Injects structured metadata about the previous turn before the next user message,
so the LLM can make learning decisions without scanning raw message history.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PATTERN_THRESHOLD = 3  # times a tool sequence must repeat before flagging

# Fork(perf): PatternStore throttles disk writes to at most one per this many
# seconds. increment() runs on every tool-using turn; in-memory counts stay
# exact (get() is always current), only persistence is deferred. A crash loses
# at most this window of increments — acceptable for non-critical pattern stats.
_PATTERN_SAVE_MIN_INTERVAL_SEC = 30.0


def _compress_tool_sequence(tools: list[str]) -> tuple[str, ...]:
    """Preserve order but collapse consecutive duplicate tool calls.

    ["read_file", "exec", "exec", "write_file"] → ("read_file", "exec", "write_file")
    This distinguishes A→B→C from A→C→B while being robust to retry loops.
    """
    result: list[str] = []
    for t in tools:
        if not result or result[-1] != t:
            result.append(t)
    return tuple(result)


class PatternStore:
    """Persistent cross-session tool-sequence pattern counter.

    Stores counts in ``<data_dir>/memory/patterns.json`` so patterns
    accumulate across restarts and sessions.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "memory" / "patterns.json"
        self._counts: dict[str, int] = self._load()
        self._dirty = False
        self._last_save = 0.0

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> dict[str, int]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._counts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False
        self._last_save = time.monotonic()

    def _save_throttled(self) -> None:
        """Persist at most once per _PATTERN_SAVE_MIN_INTERVAL_SEC; else mark dirty."""
        self._dirty = True
        if time.monotonic() - self._last_save >= _PATTERN_SAVE_MIN_INTERVAL_SEC:
            self._save()

    def flush(self) -> None:
        """Force-persist any pending counts. Call on shutdown / session end."""
        if self._dirty:
            self._save()

    # ── public API ───────────────────────────────────────────────────

    def increment(self, pattern: tuple[str, ...]) -> int:
        """Increment count for *pattern* and return the new count."""
        key = ",".join(pattern)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._save_throttled()
        return self._counts[key]

    def get(self, pattern: tuple[str, ...]) -> int:
        return self._counts.get(",".join(pattern), 0)

    def reset(self, pattern: tuple[str, ...]) -> None:
        """Reset count for *pattern* (call after a skill is created for it)."""
        key = ",".join(pattern)
        if key in self._counts:
            del self._counts[key]
            self._save()


# ── TurnSummary ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class TurnSummary:
    """Metadata about a completed agent turn, injected before the next user message.

    Follows the same ``[Tag — metadata only]`` convention as
    ``ContextBuilder._RUNTIME_CONTEXT_TAG``.
    """

    turn_index: int
    pressure_pct: int = 0
    pressure_detail: str = ""
    tools_count: int = 0
    tools_list: list[str] = field(default_factory=list)
    error_count: int = 0
    stop_reason: str = "end_turn"
    consolidation_note: str | None = None
    user_delta: str = "continuation"
    # Repeated pattern: set when the same tool combination has been used >= threshold times.
    repeated_pattern: tuple[str, ...] | None = None
    repeated_count: int = 0

    # ── format ───────────────────────────────────────────────────────

    def format_for_injection(self) -> str:
        """Format as a nanobot-style metadata block."""
        lines = ["[Turn Summary — metadata only, not instructions]"]
        lines.append(f"turn_index: {self.turn_index}")
        if self.pressure_detail:
            lines.append(f"pressure: {self.pressure_pct}% ({self.pressure_detail})")
        if self.tools_count > 0:
            tools_str = ", ".join(self.tools_list)
            lines.append(f"tools: {tools_str} ({self.tools_count} calls)")
        if self.error_count > 0:
            lines.append(f"errors: {self.error_count}")
        lines.append(f"stop: {self.stop_reason}")
        if self.consolidation_note:
            lines.append(f"consolidation: {self.consolidation_note}")
        lines.append(f"user_delta: {self.user_delta}")
        if self.repeated_pattern:
            pattern_str = ", ".join(self.repeated_pattern)
            lines.append(f"repeated_pattern: {pattern_str} ({self.repeated_count} times)")
        return "\n".join(lines)

    # ── significance ──────────────────────────────────────────────────

    @property
    def is_significant(self) -> bool:
        """Only inject if something meaningful happened in the previous turn."""
        return (
            self.tools_count > 0
            or self.error_count > 0
            or self.consolidation_note is not None
            or self.stop_reason not in ("end_turn", "completed")
            or self.repeated_pattern is not None
        )


# ── user_delta heuristic ─────────────────────────────────────────────────

# Specific correction patterns — kept narrow to avoid false positives.
# Removed standalone "别"/"不要"/"不应该" (too common; match "别忘了", "不要担心").
_CORRECTION_PATTERNS = (
    "不对", "错了", "不是", "纠正", "改一下", "重新来", "更正",
    "你理解错了", "理解错了", "说错了", "搞错了",
    "不要这样", "不应该这样", "不是这样",
    "wrong", "incorrect", "correction", "that's not right", "you're wrong",
)

# Patterns that suggest the user is switching to a new subject.
_NEW_TOPIC_PATTERNS = (
    "换个话题", "换一个问题", "顺便问", "另外问", "说点别的",
    "by the way", "btw", "new question", "different topic", "changing subject",
)


def detect_user_delta(prev_msg: str | None, current_msg: str) -> str:
    """Heuristic to classify the user's intent relative to the previous message.

    Returns one of ``"correction"``, ``"new_topic"``, or ``"continuation"``.
    """
    if not prev_msg:
        return "new_topic"
    lower = current_msg.strip().lower()
    for pattern in _CORRECTION_PATTERNS:
        if pattern in lower:
            return "correction"
    for pattern in _NEW_TOPIC_PATTERNS:
        if pattern in lower:
            return "new_topic"
    return "continuation"
