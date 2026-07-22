"""Shared result models for the optional code-intelligence tool."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class CodeLocation:
    path: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    kind: str = "reference"
    container: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(slots=True)
class RiskAssessment:
    score: int = 0
    reasons: list[str] = field(default_factory=list)
    suggested_scope: str = "semantic"
    exhaustive_intent: bool = False

    @property
    def macro_sensitive(self) -> bool:
        return self.score >= 2

    def add(self, points: int, reason: str) -> None:
        if reason not in self.reasons:
            self.score += points
            self.reasons.append(reason)

    def finish(self) -> "RiskAssessment":
        if self.score >= 6:
            self.suggested_scope = "workspace"
        elif self.score >= 2:
            self.suggested_scope = "module"
        return self
