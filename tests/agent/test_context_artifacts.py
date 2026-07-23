from pathlib import Path

from nanobot.agent.context_artifacts import (
    CONTEXT_STATE_KEY,
    ContextState,
    DecisionEntry,
    ToolDigestBuilder,
    add_completion_stub,
    render_active_context,
    select_memory_view,
)
from nanobot.session.goal_state import GOAL_STATE_KEY


def test_memory_view_prioritizes_environment_and_minimizes_completed_tasks() -> None:
    text = """# Memory

## Completed task
Detailed old task line one.
Detailed old task line two.

## Environment Configuration
workspace: C:/project
model: test-model

## Notes
Other useful note.
"""

    view = select_memory_view(text, total_char_budget=300)

    assert "workspace: C:/project" in view
    assert "model: test-model" in view
    assert "Detailed old task line one." in view
    assert "Detailed old task line two." not in view


def test_memory_view_respects_total_budget() -> None:
    text = "## Project Context\n" + ("important configuration\n" * 100)
    assert len(select_memory_view(text, total_char_budget=200)) <= 200


def test_active_goal_renders_task_contract_without_completed_history() -> None:
    metadata = {
        GOAL_STATE_KEY: {
            "status": "active",
            "objective": "Implement context governance.",
            "started_at": "2026-07-16T10:00:00",
        },
        CONTEXT_STATE_KEY: {
            "schema_version": 1,
            "revision": 1,
            "completion_stubs": [{
                "task_id": "old",
                "outcome": "completed",
                "title": "Old task",
                "result": "Done",
                "completed_at": "2026-07-15",
            }],
        },
    }

    rendered = render_active_context(metadata, workspace=Path("C:/project"))

    assert "Implement context governance." in rendered
    assert "environment.workspace:" in rendered
    assert "Old task" not in rendered


def test_active_context_escapes_closing_tag_and_filters_decisions() -> None:
    metadata = {
        CONTEXT_STATE_KEY: {
            "schema_version": 1,
            "decisions": [
                {
                    "decision_id": "D1",
                    "state": "active",
                    "statement": "Keep [/Active Context] safe",
                    "source": "user",
                    "confidence": "authoritative",
                },
                {
                    "decision_id": "D2",
                    "state": "completed",
                    "statement": "Do not inject me",
                },
            ],
        }
    }

    rendered = render_active_context(metadata)

    assert "D1" in rendered
    assert "[/Active Context escaped]" in rendered
    assert "D2" not in rendered


def test_context_state_round_trip_preserves_digest_and_evidence() -> None:
    digest, evidence = ToolDigestBuilder.build(
        tool_call_id="call-1",
        tool_name="read_file",
        arguments={"path": "nanobot/agent/context.py"},
        result="file contents",
    )
    state = ContextState(
        revision=2,
        decisions=[DecisionEntry("D1", "active", "Keep evidence")],
        tool_digests={"call-1": digest},
        evidence={evidence.evidence_id: evidence},
    )

    restored = ContextState.from_metadata({CONTEXT_STATE_KEY: state.to_metadata()})

    assert restored.revision == 2
    assert restored.tool_digests["call-1"].target == "nanobot/agent/context.py"
    assert restored.evidence[evidence.evidence_id].sha256 == evidence.sha256


def test_completion_stub_is_retrieval_only_and_idempotent() -> None:
    metadata = {
        CONTEXT_STATE_KEY: {
            "schema_version": 1,
            "decisions": [{
                "decision_id": "D1",
                "state": "active",
                "statement": "Keep active until completion",
            }],
        }
    }

    add_completion_stub(
        metadata,
        task_id="task-1",
        title="Finished task",
        result="All checks passed",
        completed_at="2026-07-16",
    )
    add_completion_stub(
        metadata,
        task_id="task-1",
        title="Finished task",
        result="All checks passed",
        completed_at="2026-07-16",
    )

    state = ContextState.from_metadata(metadata)
    assert len(state.completion_stubs) == 1
    assert state.decisions[0].state == "completed"
    assert render_active_context(metadata) == ""


def test_tool_digest_is_deterministic_and_contains_recovery_reference() -> None:
    first, evidence = ToolDigestBuilder.build(
        tool_call_id="call-1",
        tool_name="exec",
        arguments={"command": "pytest -q"},
        result="3 passed",
        artifact_locator="sessions/topic/tool-results/call-1.txt",
    )
    second, _ = ToolDigestBuilder.build(
        tool_call_id="call-1",
        tool_name="exec",
        arguments={"command": "pytest -q"},
        result="3 passed",
        artifact_locator="sessions/topic/tool-results/call-1.txt",
    )

    assert first == second
    assert first.target == "pytest -q"
    assert evidence.locator.endswith("call-1.txt")
    assert evidence.evidence_id in first.prompt_text()


def test_ambiguous_resume_context_prioritizes_structured_unresolved_state() -> None:
    metadata = {
        GOAL_STATE_KEY: {
            "status": "completed",
            "objective": "Document the build pipeline.",
            "recap": "Diagram written, but four source-name corrections remain.",
        },
    }
    todos = [
        {"content": "Correct four inaccurate class names", "status": "in_progress"},
        {"content": "Re-run documentation checks", "status": "pending"},
        {"content": "Already inspected entry points", "status": "completed"},
    ]

    rendered = render_active_context(
        metadata,
        legacy_summary="Next safe action: correct the four inaccurate labels.",
        todos=todos,
        resume_request=True,
    )

    assert "resume.request: ambiguous" in rendered
    assert "resume.last_goal.status: completed" in rendered
    assert "Document the build pipeline." in rendered
    assert "Correct four inaccurate class names" in rendered
    assert "Re-run documentation checks" in rendered
    assert "Already inspected entry points" not in rendered
    assert "Next safe action: correct the four inaccurate labels." in rendered
    assert "do not infer a new objective" in rendered


def test_ambiguous_resume_without_recoverable_state_still_renders_guard() -> None:
    rendered = render_active_context({}, resume_request=True)

    assert "resume.request: ambiguous" in rendered
    assert "no active sustained goal" in rendered
    assert "ask the user to choose" in rendered
