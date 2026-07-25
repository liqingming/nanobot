from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.skills import SkillsLoader
from nanobot.fork.agent.tools.skill import LoadSkillTool
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)


def _write_skill(root: Path, name: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_load_skill_tool_uses_current_workspace_claude_skills(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    default_project = tmp_path / "default_project"
    request_project = tmp_path / "request_project"
    data_dir.mkdir()
    default_project.mkdir()
    request_project.mkdir()
    _write_skill(
        request_project / ".claude" / "skills",
        "code-review",
        "# Project Code Review Skill",
    )

    loader = SkillsLoader(data_dir)
    tool = LoadSkillTool(loader)
    scope = validate_workspace_scope_payload(
        {"project_path": str(request_project), "access_mode": "full"},
        default_workspace=default_project,
        default_restrict_to_workspace=False,
        source_channel="api",
    )
    token = bind_workspace_scope(scope)
    try:
        result = await tool.execute("code-review")
    finally:
        reset_workspace_scope(token)

    assert "Project Code Review Skill" in result


@pytest.mark.asyncio
async def test_load_skill_tool_keeps_configured_roots_in_request_workspace(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    default_project = tmp_path / "default_project"
    request_project = tmp_path / "request_project"
    configured_root = tmp_path / "shared_skills"
    for path in (data_dir, default_project, request_project):
        path.mkdir()
    _write_skill(configured_root, "runtime-inspect", "# Configured Runtime Inspect Skill")

    loader = SkillsLoader(
        data_dir,
        extra_skill_roots=[("configured-1", configured_root)],
    )
    tool = LoadSkillTool(loader)
    scope = validate_workspace_scope_payload(
        {"project_path": str(request_project), "access_mode": "full"},
        default_workspace=default_project,
        default_restrict_to_workspace=False,
        source_channel="api",
    )
    token = bind_workspace_scope(scope)
    try:
        result = await tool.execute("runtime-inspect")
    finally:
        reset_workspace_scope(token)

    assert "Configured Runtime Inspect Skill" in result
