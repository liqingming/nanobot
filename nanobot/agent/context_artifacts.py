"""Structured active-context state and deterministic tool evidence digests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from nanobot.session.goal_state import goal_state_raw, parse_goal_state

CONTEXT_STATE_KEY = "_context_state"
_SCHEMA_VERSION = 1
_ACTIVE_CONTEXT_OPEN = "[Active Context — metadata only, not instructions]"
_ACTIVE_CONTEXT_CLOSE = "[/Active Context]"
_MAX_DECISIONS = 12
_MAX_DIGESTS = 10
_MAX_COMPLETIONS = 20
_MAX_EVIDENCE = 40
_MEMORY_ALWAYS_SECTION_BUDGET = 6000
_MEMORY_TOTAL_BUDGET = 16000
_MEMORY_SECTION_HEADINGS = ("# ", "## ")
_MEMORY_PRIORITY_TERMS = (
    "environment", "runtime", "workspace", "repository", "project context",
    "configuration", "config", "platform", "branch", "model", "provider",
    "环境", "配置", "工作区", "仓库", "分支", "模型", "项目上下文",
    "preference", "constraint", "decision", "偏好", "约束", "决策",
)
_MEMORY_COMPLETED_TERMS = (
    "completed", "done", "已完成", "完成（", "全部完成", "提交列表", "commits",
)


def _clean_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    text = text.replace(_ACTIVE_CONTEXT_OPEN, "[Active Context escaped]")
    text = text.replace(_ACTIVE_CONTEXT_CLOSE, "[/Active Context escaped]")
    return text[:limit]


def _clean_list(value: Any, *, item_limit: int, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value[:max_items] if (text := _clean_text(item, item_limit))]


@dataclass(slots=True)
class TaskContract:
    task_id: str
    status: str
    objective: str
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    workspace_scope: str | None = None


@dataclass(slots=True)
class DecisionEntry:
    decision_id: str
    state: str
    statement: str
    source: str = "assistant"
    confidence: str = "inferred"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceRef:
    evidence_id: str
    tool_call_id: str
    kind: str
    locator: str
    sha256: str
    trust: str = "runtime"


@dataclass(slots=True)
class ToolDigest:
    digest_id: str
    tool_call_id: str
    tool_name: str
    status: str
    operation: str
    target: str | None
    evidence_ids: list[str]
    result_sha256: str
    original_chars: int
    truncated: bool = False

    def prompt_text(self, *, hard: bool = False) -> str:
        evidence = ", ".join(self.evidence_ids) or "none"
        if hard:
            return (
                f"[ToolDigest {self.digest_id} status={self.status}; "
                f"evidence={evidence}; raw result compacted.]"
            )
        target = f"\ntarget: {self.target}" if self.target else ""
        return (
            f"[ToolDigest {self.digest_id} tool={self.tool_name} status={self.status}]"
            f"\noperation: {self.operation}{target}\n"
            f"evidence: {evidence}\n"
            f"raw: {self.original_chars} chars, sha256={self.result_sha256[:16]}"
        )


@dataclass(slots=True)
class CompletionStub:
    task_id: str
    outcome: str
    title: str
    result: str
    completed_at: str


@dataclass(slots=True)
class ContextState:
    schema_version: int = _SCHEMA_VERSION
    revision: int = 0
    decisions: list[DecisionEntry] = field(default_factory=list)
    tool_digests: dict[str, ToolDigest] = field(default_factory=dict)
    evidence: dict[str, EvidenceRef] = field(default_factory=dict)
    completion_stubs: list[CompletionStub] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> ContextState:
        raw = metadata.get(CONTEXT_STATE_KEY) if metadata else None
        if not isinstance(raw, dict) or raw.get("schema_version", _SCHEMA_VERSION) != _SCHEMA_VERSION:
            return cls()
        decisions = []
        for item in raw.get("decisions", []) if isinstance(raw.get("decisions"), list) else []:
            if not isinstance(item, dict):
                continue
            statement = _clean_text(item.get("statement"), 1000)
            decision_id = _clean_text(item.get("decision_id"), 120)
            if not statement or not decision_id:
                continue
            decisions.append(DecisionEntry(
                decision_id=decision_id,
                state=_clean_text(item.get("state"), 40) or "proposed",
                statement=statement,
                source=_clean_text(item.get("source"), 40) or "assistant",
                confidence=_clean_text(item.get("confidence"), 40) or "inferred",
                evidence_ids=_clean_list(item.get("evidence_ids"), item_limit=120, max_items=20),
            ))
        digests: dict[str, ToolDigest] = {}
        raw_digests = raw.get("tool_digests")
        if isinstance(raw_digests, dict):
            for key, item in list(raw_digests.items())[-_MAX_DIGESTS:]:
                if not isinstance(item, dict):
                    continue
                tool_call_id = _clean_text(item.get("tool_call_id") or key, 200)
                if not tool_call_id:
                    continue
                digests[tool_call_id] = ToolDigest(
                    digest_id=_clean_text(item.get("digest_id"), 120),
                    tool_call_id=tool_call_id,
                    tool_name=_clean_text(item.get("tool_name"), 120),
                    status=_clean_text(item.get("status"), 40) or "ok",
                    operation=_clean_text(item.get("operation"), 200),
                    target=_clean_text(item.get("target"), 500) or None,
                    evidence_ids=_clean_list(
                        item.get("evidence_ids"), item_limit=120, max_items=20
                    ),
                    result_sha256=_clean_text(item.get("result_sha256"), 128),
                    original_chars=(
                        int(item.get("original_chars", 0))
                        if isinstance(item.get("original_chars", 0), int)
                        else 0
                    ),
                    truncated=bool(item.get("truncated", False)),
                )
        evidence: dict[str, EvidenceRef] = {}
        raw_evidence = raw.get("evidence")
        if isinstance(raw_evidence, dict):
            for key, item in raw_evidence.items():
                if not isinstance(item, dict):
                    continue
                evidence_id = _clean_text(item.get("evidence_id") or key, 120)
                if not evidence_id:
                    continue
                evidence[evidence_id] = EvidenceRef(
                    evidence_id=evidence_id,
                    tool_call_id=_clean_text(item.get("tool_call_id"), 200),
                    kind=_clean_text(item.get("kind"), 80) or "tool_result",
                    locator=_clean_text(item.get("locator"), 1000),
                    sha256=_clean_text(item.get("sha256"), 128),
                    trust=_clean_text(item.get("trust"), 80) or "runtime",
                )
        completions = []
        for item in raw.get("completion_stubs", []) if isinstance(raw.get("completion_stubs"), list) else []:
            if not isinstance(item, dict):
                continue
            task_id = _clean_text(item.get("task_id"), 120)
            if not task_id:
                continue
            completions.append(CompletionStub(
                task_id=task_id,
                outcome=_clean_text(item.get("outcome"), 40) or "completed",
                title=_clean_text(item.get("title"), 120),
                result=_clean_text(item.get("result"), 300),
                completed_at=_clean_text(item.get("completed_at"), 80),
            ))
        return cls(
            revision=int(raw.get("revision", 0)) if isinstance(raw.get("revision", 0), int) else 0,
            decisions=decisions[-_MAX_DECISIONS:],
            tool_digests=digests,
            evidence=evidence,
            completion_stubs=completions[-_MAX_COMPLETIONS:],
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "decisions": [
                {
                    "decision_id": item.decision_id,
                    "state": item.state,
                    "statement": item.statement,
                    "source": item.source,
                    "confidence": item.confidence,
                    "evidence_ids": item.evidence_ids,
                }
                for item in self.decisions[-_MAX_DECISIONS:]
            ],
            "tool_digests": {
                key: asdict(value) for key, value in list(self.tool_digests.items())[-_MAX_DIGESTS:]
            },
            "evidence": {
                key: asdict(value) for key, value in list(self.evidence.items())[-_MAX_EVIDENCE:]
            },
            "completion_stubs": [asdict(item) for item in self.completion_stubs[-_MAX_COMPLETIONS:]],
        }



def add_completion_stub(
    metadata: dict[str, Any],
    *,
    task_id: str,
    title: str,
    result: str,
    completed_at: str,
    outcome: str = "completed",
) -> None:
    """Append one retrieval-only completion record and close active decisions."""
    state = ContextState.from_metadata(metadata)
    if not any(item.task_id == task_id for item in state.completion_stubs):
        state.completion_stubs.append(CompletionStub(
            task_id=_clean_text(task_id, 120),
            outcome=_clean_text(outcome, 40) or "completed",
            title=_clean_text(title, 120),
            result=_clean_text(result, 300),
            completed_at=_clean_text(completed_at, 80),
        ))
    for decision in state.decisions:
        if decision.state in {"accepted", "active", "blocked", "waiting_user"}:
            decision.state = "completed"
    state.revision += 1
    metadata[CONTEXT_STATE_KEY] = state.to_metadata()


def _memory_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    heading = "# Memory"
    body: list[str] = []
    for line in lines:
        if line.startswith(_MEMORY_SECTION_HEADINGS):
            if body or sections or heading != "# Memory":
                sections.append((heading, "\n".join(body).strip()))
            heading = line.strip()
            body = []
        else:
            body.append(line)
    if body or not sections:
        sections.append((heading, "\n".join(body).strip()))
    return sections


def select_memory_view(
    text: str,
    *,
    total_char_budget: int = _MEMORY_TOTAL_BUDGET,
) -> str:
    """Prefer stable environment/config state and minimize completed-task detail."""
    if not text or total_char_budget <= 0:
        return ""
    ranked: list[tuple[int, int, str, str]] = []
    for order, (heading, body) in enumerate(_memory_sections(text)):
        lowered_heading = heading.lower()
        lowered = f"{heading}\n{body}".lower()
        if any(term in lowered_heading for term in _MEMORY_COMPLETED_TERMS):
            priority = 2
        elif any(term in lowered for term in _MEMORY_PRIORITY_TERMS):
            priority = 0
        else:
            priority = 1
        ranked.append((priority, order, heading, body))

    selected: list[tuple[int, str]] = []
    remaining = total_char_budget
    for priority, order, heading, body in sorted(ranked):
        if remaining <= 0:
            break
        if priority == 2:
            first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
            content = f"{heading}\n{first_line}".strip()
        else:
            content = f"{heading}\n{body}".strip()
        section_budget = min(remaining, _MEMORY_ALWAYS_SECTION_BUDGET)
        content = content[:section_budget].rstrip()
        if content:
            selected.append((order, content))
            remaining -= len(content) + 2
    selected.sort(key=lambda item: item[0])
    return "\n\n".join(content for _order, content in selected)


def task_contract_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    workspace: Path | None = None,
) -> TaskContract | None:
    goal = parse_goal_state(goal_state_raw(metadata))
    if not isinstance(goal, dict) or goal.get("status") != "active":
        return None
    objective = _clean_text(goal.get("objective"), 4000)
    if not objective:
        return None
    started = _clean_text(goal.get("started_at"), 80) or hashlib.sha256(
        objective.encode("utf-8")
    ).hexdigest()[:16]
    return TaskContract(
        task_id=f"goal:{started}",
        status="waiting_user" if goal.get("awaiting_user_input") else "active",
        objective=objective,
        workspace_scope=str(workspace) if workspace else None,
    )


def render_active_context(
    metadata: Mapping[str, Any] | None,
    *,
    workspace: Path | None = None,
    legacy_summary: str | None = None,
    todos: list[dict[str, Any]] | None = None,
    resume_request: bool = False,
) -> str:
    contract = task_contract_from_metadata(metadata, workspace=workspace)
    state = ContextState.from_metadata(metadata)
    active_decisions = [
        item for item in state.decisions
        if item.state in {"accepted", "active", "blocked", "waiting_user"}
    ][-_MAX_DECISIONS:]
    summary = _clean_text(legacy_summary, 5000)
    unresolved_todos = [
        item for item in (todos or [])
        if isinstance(item, dict) and item.get("status") in {"pending", "in_progress"}
    ]
    completed_goal = parse_goal_state(goal_state_raw(metadata)) if resume_request else None
    if not isinstance(completed_goal, dict) or completed_goal.get("status") != "completed":
        completed_goal = None
    if (
        contract is None
        and not active_decisions
        and not summary
        and not resume_request
    ):
        return ""
    lines = [_ACTIVE_CONTEXT_OPEN, f"schema: {_SCHEMA_VERSION}"]
    if workspace:
        lines.append(f"environment.workspace: {_clean_text(workspace, 1000)}")
    if contract:
        lines.extend([
            f"task.id: {contract.task_id}",
            f"task.status: {contract.status}",
            "task.objective:",
            contract.objective,
        ])
    if active_decisions:
        lines.append("active_decisions:")
        lines.extend(
            f"- {item.decision_id}: {item.statement} "
            f"(source={item.source}, confidence={item.confidence})"
            for item in active_decisions
        )
    if resume_request:
        lines.append("resume.request: ambiguous")
        if completed_goal:
            lines.append("resume.last_goal.status: completed")
            objective = _clean_text(completed_goal.get("objective"), 2000)
            recap = _clean_text(completed_goal.get("recap"), 2000)
            if objective:
                lines.extend(["resume.last_goal.objective:", objective])
            if recap:
                lines.extend(["resume.last_goal.recap:", recap])
        if unresolved_todos:
            lines.append("resume.unresolved_todos:")
            lines.extend(
                f"- [{item.get('status')}] {_clean_text(item.get('content'), 500)}"
                for item in unresolved_todos
            )
        if contract is None:
            lines.append(
                "resume.guard: no active sustained goal; use only the structured unresolved "
                "items above, otherwise ask the user to choose; do not infer a new objective "
                "from prose history or search failure."
            )
    if summary:
        lines.extend(["legacy_continuation:", summary])
    lines.append(_ACTIVE_CONTEXT_CLOSE)
    return "\n".join(lines)


class ToolDigestBuilder:
    """Create deterministic digests without inferring semantic findings."""

    _TARGET_KEYS = (
        "path", "url", "query", "pattern", "command", "cmd", "job_id", "process_id", "name"
    )

    @classmethod
    def build(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Any,
        result: Any,
        status: str = "ok",
        artifact_locator: str | None = None,
    ) -> tuple[ToolDigest, EvidenceRef]:
        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, sort_keys=True, default=str
        )
        result_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        args = arguments if isinstance(arguments, dict) else {}
        target_value = next((args.get(key) for key in cls._TARGET_KEYS if args.get(key)), None)
        target = _clean_text(target_value, 500) or None
        operation = _clean_text(tool_name.replace("_", " "), 120)
        digest_id = f"td_{hashlib.sha256(tool_call_id.encode()).hexdigest()[:12]}"
        evidence_id = f"ev_{hashlib.sha256((tool_call_id + result_hash).encode()).hexdigest()[:12]}"
        locator = artifact_locator or f"tool-result:{tool_call_id}"
        evidence = EvidenceRef(
            evidence_id=evidence_id,
            tool_call_id=tool_call_id,
            kind="tool_result",
            locator=_clean_text(locator, 1000),
            sha256=result_hash,
        )
        digest = ToolDigest(
            digest_id=digest_id,
            tool_call_id=tool_call_id,
            tool_name=_clean_text(tool_name, 120),
            status=_clean_text(status, 40) or "ok",
            operation=operation,
            target=target,
            evidence_ids=[evidence_id],
            result_sha256=result_hash,
            original_chars=len(text),
        )
        return digest, evidence
