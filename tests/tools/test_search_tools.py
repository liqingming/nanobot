"""Tests for grep search tools."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.search import FindFilesTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import WebSearchConfig


@pytest.mark.asyncio
async def test_web_search_tool_refreshes_dynamic_config_loader(monkeypatch) -> None:
    tool = WebSearchTool(
        config=WebSearchConfig(provider="brave"),
        config_loader=lambda: WebSearchConfig(provider="duckduckgo", max_results=3),
    )

    async def fake_duckduckgo(self, query: str, n: int) -> str:
        return f"{self.config.provider}:{query}:{n}"

    monkeypatch.setattr(WebSearchTool, "_search_duckduckgo", fake_duckduckgo)

    assert await tool.execute("nanobot") == "duckduckgo:nanobot:3"


@pytest.mark.asyncio
async def test_find_files_filters_by_query_glob_and_type(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "settings_view.tsx").write_text("export {}\n", encoding="utf-8")
    (tmp_path / "src" / "settings_api.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("settings\n", encoding="utf-8")

    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        path=".",
        query="settings",
        glob="src/**",
        type="ts",
    )

    assert result.splitlines() == ["src/settings_view.tsx"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["", ".", "./"])
async def test_find_files_rejects_paths_from_request_tool_policy(tmp_path: Path, path: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api",
        chat_id="default",
        session_key="api:review_123_code_review",
        metadata={"tool_policy": {"blocked_find_files_paths": ["", ".", "./"]}},
    ))
    try:
        result = await tool.execute(path=path, query="main")
    finally:
        reset_request_context(token)

    assert "blocked by the request tool_policy for find_files" in str(result)


@pytest.mark.asyncio
async def test_find_files_without_policy_is_not_restricted_for_review_session(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api",
        chat_id="default",
        session_key="api:review_123_code_review",
    ))
    try:
        result = await tool.execute(path="src", query="main")
    finally:
        reset_request_context(token)

    assert result.splitlines() == ["src/main.py"]


@pytest.mark.asyncio
async def test_find_files_can_include_directories(tmp_path: Path) -> None:
    (tmp_path / "src" / "settings").mkdir(parents=True)
    (tmp_path / "src" / "settings" / "index.ts").write_text("export {}\n", encoding="utf-8")

    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(path="src", query="settings", include_dirs=True)

    assert "src/settings/" in result.splitlines()
    assert "src/settings/index.ts" in result.splitlines()


@pytest.mark.asyncio
async def test_find_files_supports_modified_sort_and_pagination(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    for idx, name in enumerate(("a.py", "b.py", "c.py"), start=1):
        file_path = tmp_path / "src" / name
        file_path.write_text("pass\n", encoding="utf-8")
        os.utime(file_path, (idx, idx))

    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        path="src",
        type="py",
        sort="modified",
        head_limit=1,
        offset=1,
    )

    assert result.splitlines()[0] == "src/b.py"
    assert "pagination: limit=1, offset=1" in result


@pytest.mark.asyncio
async def test_find_files_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-find-files.txt"
    outside.write_text("secret\n", encoding="utf-8")

    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(path=str(outside))

    assert result.startswith("Error:")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["", ".", "./", "Assets", "Assets/Script", "Assets/Script/Game", "Assets/Script/Game/moduls", "Assets/ScriptGenerated", "Assets/ResourcesAssets"])
async def test_grep_rejects_paths_from_request_tool_policy(tmp_path: Path, path: str) -> None:
    (tmp_path / "Assets" / "Script" / "Game" / "moduls" / "Dragon").mkdir(parents=True)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api",
        chat_id="default",
        session_key="api:review_123_code_review",
        metadata={"tool_policy": {"blocked_grep_paths": ["", ".", "./", "Assets", "Assets/Script", "Assets/Script/Game", "Assets/Script/Game/moduls", "Assets/ScriptGenerated", "Assets/ResourcesAssets"]}},
    ))
    try:
        result = await tool.execute(pattern="needle", path=path)
    finally:
        reset_request_context(token)

    assert "blocked by the request tool_policy for grep" in str(result)
    assert "specific module/subdirectory" in str(result)


@pytest.mark.asyncio
async def test_grep_rejects_subpaths_of_request_tool_policy(tmp_path: Path) -> None:
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="default", session_key="api:review_123_code_review",
        metadata={"tool_policy": {"blocked_grep_paths": ["Assets/ResourcesAssets"]}},
    ))
    try:
        result = await tool.execute(pattern="needle", path="Assets/ResourcesAssets/Prefabs/Test")
    finally:
        reset_request_context(token)
    assert "blocked by the request tool_policy for grep" in str(result)


@pytest.mark.asyncio
async def test_exec_rejects_request_with_disable_exec_policy(tmp_path: Path) -> None:
    tool = ExecTool(working_dir=str(tmp_path))
    token = bind_request_context(RequestContext(
        channel="api", chat_id="default", session_key="api:review_123_code_review",
        metadata={"tool_policy": {"disable_exec": True}},
    ))
    try:
        result = await tool.execute(command="git diff --check", shell="cmd")
    finally:
        reset_request_context(token)
    assert "exec is blocked by the request tool_policy" in str(result)


@pytest.mark.asyncio
async def test_exec_rejects_request_blocked_command_pattern(tmp_path: Path) -> None:
    tool = ExecTool(working_dir=str(tmp_path))
    token = bind_request_context(RequestContext(
        channel="api", chat_id="default", session_key="api:repair",
        metadata={"tool_policy": {"blocked_exec_patterns": [r"\bgit\s+commit\b"]}},
    ))
    try:
        result = await tool.execute(command="git commit -m test", shell="cmd")
    finally:
        reset_request_context(token)
    assert "command is blocked by the request tool_policy" in str(result)


async def test_grep_allows_explicit_leaf_module_despite_blocked_parent(tmp_path: Path) -> None:
    module = tmp_path / "Assets" / "Script" / "Game" / "moduls" / "DragonInvadeActivity"
    module.mkdir(parents=True)
    (module / "UI_item_resource.cs").write_text("void UpdateView() {}\n", encoding="utf-8")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="default", session_key="api:review_123_code_review",
        metadata={"tool_policy": {
            "blocked_grep_paths": ["Assets/Script/Game/moduls"],
            "allowed_grep_paths": ["Assets/Script/Game/moduls/DragonInvadeActivity"],
        }},
    ))
    try:
        result = await tool.execute(pattern="UpdateView", path="Assets/Script/Game/moduls/DragonInvadeActivity", fixed_strings=True)
    finally:
        reset_request_context(token)
    assert "UI_item_resource.cs" in str(result)


async def test_grep_allows_specific_module_for_review_session(tmp_path: Path) -> None:
    module = tmp_path / "Assets" / "Script" / "Game" / "moduls" / "Dragon"
    module.mkdir(parents=True)
    (module / "DragonData.cs").write_text("needle\n", encoding="utf-8")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api",
        chat_id="default",
        session_key="api:review_123_code_review",
    ))
    try:
        result = await tool.execute(
            pattern="needle",
            path="Assets/Script/Game/moduls/Dragon",
            fixed_strings=True,
        )
    finally:
        reset_request_context(token)

    assert result.splitlines() == ["Assets/Script/Game/moduls/Dragon/DragonData.cs"]


@pytest.mark.asyncio
async def test_grep_respects_glob_filter_and_context(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "alpha\nbeta\nmatch_here\ngamma\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("match_here\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="match_here",
        path=".",
        glob="*.py",
        output_mode="content",
        context_before=1,
        context_after=1,
    )

    assert "src/main.py:3" in result
    assert "  2| beta" in result
    assert "> 3| match_here" in result
    assert "  4| gamma" in result
    assert "README.md" not in result


@pytest.mark.asyncio
async def test_grep_defaults_to_files_with_matches(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("match_here\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="match_here",
        path="src",
    )

    assert result.splitlines() == ["src/main.py"]
    assert "1|" not in result


@pytest.mark.asyncio
async def test_grep_supports_case_insensitive_search(tmp_path: Path) -> None:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "HISTORY.md").write_text(
        "[2026-04-02 10:00] OAuth token rotated\n",
        encoding="utf-8",
    )

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="oauth",
        path="memory/HISTORY.md",
        case_insensitive=True,
        output_mode="content",
    )

    assert "memory/HISTORY.md:1" in result
    assert "OAuth token rotated" in result


@pytest.mark.asyncio
async def test_grep_type_filter_limits_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "src" / "b.md").write_text("needle\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="needle",
        path="src",
        type="py",
    )

    assert result.splitlines() == ["src/a.py"]


@pytest.mark.asyncio
async def test_grep_fixed_strings_treats_regex_chars_literally(tmp_path: Path) -> None:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "HISTORY.md").write_text(
        "[2026-04-02 10:00] OAuth token rotated\n",
        encoding="utf-8",
    )

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="[2026-04-02 10:00]",
        path="memory/HISTORY.md",
        fixed_strings=True,
        output_mode="content",
    )

    assert "memory/HISTORY.md:1" in result
    assert "[2026-04-02 10:00] OAuth token rotated" in result


@pytest.mark.asyncio
async def test_grep_files_with_matches_mode_returns_unique_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    a = tmp_path / "src" / "a.py"
    b = tmp_path / "src" / "b.py"
    a.write_text("needle\nneedle\n", encoding="utf-8")
    b.write_text("needle\n", encoding="utf-8")
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="needle",
        path="src",
        output_mode="files_with_matches",
    )

    assert result.splitlines() == ["src/b.py", "src/a.py"]


@pytest.mark.asyncio
async def test_grep_files_with_matches_supports_head_limit_and_offset(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / "src" / name).write_text("needle\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="needle",
        path="src",
        head_limit=1,
        offset=1,
    )

    # Filesystem order is not deterministic across platforms, so just verify:
    # 1. Only one file path is returned (head_limit=1 after offset=1)
    # 2. The pagination info is correct
    assert "pagination: limit=1, offset=1" in result
    # Count non-empty lines that start with src/ (file paths)
    file_lines = [line for line in result.splitlines() if line.startswith("src/")]
    assert len(file_lines) == 1


@pytest.mark.asyncio
async def test_grep_count_mode_reports_counts_per_file(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "one.log").write_text("warn\nok\nwarn\n", encoding="utf-8")
    (tmp_path / "logs" / "two.log").write_text("warn\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="warn",
        path="logs",
        output_mode="count",
    )

    assert "logs/one.log: 2" in result
    assert "logs/two.log: 1" in result
    assert "total matches: 3 in 2 files" in result


@pytest.mark.asyncio
async def test_grep_files_with_matches_mode_respects_max_results(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    files = []
    for idx, name in enumerate(("a.py", "b.py", "c.py"), start=1):
        file_path = tmp_path / "src" / name
        file_path.write_text("needle\n", encoding="utf-8")
        os.utime(file_path, (idx, idx))
        files.append(file_path)

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        pattern="needle",
        path="src",
        output_mode="files_with_matches",
        max_results=2,
    )

    assert result.splitlines()[:2] == ["src/c.py", "src/b.py"]
    assert "pagination: limit=2, offset=0" in result


@pytest.mark.asyncio
async def test_grep_reports_skipped_binary_and_large_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "large.txt").write_text("x" * 20, encoding="utf-8")

    monkeypatch.setattr(GrepTool, "_MAX_FILE_BYTES", 10)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(pattern="needle", path=".")

    assert "No matches found" in result
    assert "skipped 1 binary/unreadable files" in result
    assert "skipped 1 large files" in result


@pytest.mark.asyncio
async def test_search_tools_reject_paths_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-search.txt"
    outside.write_text("secret\n", encoding="utf-8")

    grep_tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    grep_result = await grep_tool.execute(pattern="secret", path=str(outside))

    assert grep_result.startswith("Error:")


def test_agent_loop_registers_grep(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    assert "find_files" in loop.tools.tool_names
    assert "grep" in loop.tools.tool_names


@pytest.mark.asyncio
async def test_subagent_registers_grep(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=4096,
    )
    captured: dict[str, list[str]] = {}

    async def fake_run(spec):
        captured["tool_names"] = spec.tools.tool_names
        return SimpleNamespace(
            stop_reason="ok",
            final_content="done",
            tool_events=[],
            error=None,
        )

    mgr.runner.run = fake_run
    mgr._announce_result = AsyncMock()

    status = SubagentStatus(task_id="sub-1", label="label", task_description="search task", started_at=time.monotonic())
    await mgr._run_subagent("sub-1", "search task", "label", {"channel": "cli", "chat_id": "direct"}, status)

    assert "find_files" in captured["tool_names"]
    assert "grep" in captured["tool_names"]


def test_subagent_prompt_respects_disabled_skills(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    skills_dir = tmp_path / "skills"
    (skills_dir / "alpha").mkdir(parents=True)
    (skills_dir / "alpha" / "SKILL.md").write_text("# Alpha\n\nhidden\n", encoding="utf-8")
    (skills_dir / "beta").mkdir(parents=True)
    (skills_dir / "beta" / "SKILL.md").write_text("# Beta\n\nshown\n", encoding="utf-8")

    mgr = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=4096,
        disabled_skills=["alpha"],
    )

    prompt = mgr._build_subagent_prompt()

    assert "alpha" not in prompt
    assert "beta" in prompt


@pytest.mark.parametrize(
    ("command", "allowed"),
    [
        ("git status --short", True),
        ("git diff --check", True),
        ("git log -5 --oneline", True),
        ("git branch --show-current", True),
        ("git remote -v", True),
        ("where python", True),
        ("where powershell", True),
        ("tasklist", True),
        ("dotnet --info", True),
        ("git checkout main", False),
        ("git branch new-name", False),
        ("git tag v1", False),
        ("git remote add origin x", False),
        ("git diff --output=patch.txt", False),
        ("git diff --ext-diff", False),
        ("git show --textconv HEAD:file", False),
        ("git grep --open-files-in-pager=writer needle", False),
        ("git --paginate status", False),
        ("python -c \"open('x','w').write('y')\"", False),
        ("type source.txt > copy.txt", False),
        ("git status & echo x", False),
        ("git status | findstr M", False),
        ("powershell Get-Content x", False),
        ("dotnet build", False),
    ],
)
def test_exec_read_only_command_classifier(command: str, allowed: bool) -> None:
    assert ExecTool._is_read_only_command(command) is allowed


@pytest.mark.asyncio
async def test_exec_read_only_mode_allows_diagnostics_and_blocks_mutation(tmp_path: Path) -> None:
    tool = ExecTool(working_dir=str(tmp_path))
    token = bind_request_context(RequestContext(
        channel="api", chat_id="default", session_key="api:repair",
        metadata={"tool_policy": {"read_only_mode": True}},
    ))
    try:
        allowed = await tool.execute(command="git status --short", shell="cmd")
        blocked = await tool.execute(command="git checkout main", shell="cmd")
    finally:
        reset_request_context(token)
    assert "not allowed" not in str(allowed)
    assert "not allowed in request read-only mode" in str(blocked)


@pytest.mark.asyncio
async def test_search_tools_use_bounded_default_result_limit(tmp_path: Path) -> None:
    for idx in range(130):
        (tmp_path / f"match_{idx:03d}.txt").write_text("needle\n", encoding="utf-8")

    find_result = await FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path).execute(
        path=".", query="match_"
    )
    grep_result = await GrepTool(workspace=tmp_path, allowed_dir=tmp_path).execute(
        pattern="needle", path="."
    )

    assert len([line for line in find_result.splitlines() if line.endswith(".txt")]) == 100
    assert len([line for line in grep_result.splitlines() if line.endswith(".txt")]) == 100
    assert "pagination: limit=100" in find_result
    assert "pagination: limit=100" in grep_result
