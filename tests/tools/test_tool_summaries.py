"""Tests for per-tool summarize_result implementations + the dispatcher.

Each Tool subclass owns its own summarize_result; the dispatcher just routes
to it. Tests construct the tool directly (with dummy workspace) and call
summarize_result with mocked args/result.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.tools.summaries import (
    extract_error_summary,
    line_count,
    summarize_tool_result,
    truncate,
)


# ── shared helpers ────────────────────────────────────────────────────────


def test_line_count_basic() -> None:
    assert line_count("") == 0
    assert line_count("hello") == 1
    assert line_count("a\nb") == 2
    assert line_count("a\nb\n") == 2


def test_truncate_keeps_short() -> None:
    assert truncate("short", 60) == "short"


def test_truncate_collapses_newlines_and_cuts() -> None:
    text = "line1\nline2\n" + "x" * 100
    out = truncate(text, 20)
    assert len(out) == 20
    assert out.endswith("…")
    assert "\n" not in out


def test_extract_error_summary_strips_prefix() -> None:
    assert extract_error_summary("Error: permission denied").startswith("Error: permission denied")
    assert "Error:" in extract_error_summary("Error: permission denied")
    # Does NOT result in "Error: Error: ..."
    assert extract_error_summary("Error: permission denied").count("Error:") == 1


def test_extract_error_summary_uses_first_line() -> None:
    msg = "Error: command failed\nFull traceback below:\n  File X, line Y\n  Bad code"
    out = extract_error_summary(msg)
    assert "command failed" in out
    assert "traceback" not in out


# ── dispatcher behavior ──────────────────────────────────────────────────


def test_dispatcher_none_tool_returns_empty() -> None:
    assert summarize_tool_result(None, {}, "anything") == ""


def test_dispatcher_swallows_exceptions() -> None:
    class Broken:
        def summarize_result(self, args, result):
            raise RuntimeError("boom")
    assert summarize_tool_result(Broken(), {}, "x") == ""


def test_dispatcher_coerces_non_dict_args() -> None:
    class Echo:
        def summarize_result(self, args, result):
            assert isinstance(args, dict)
            return "ok"
    assert summarize_tool_result(Echo(), "not a dict", "x") == "ok"


# ── ExecTool ─────────────────────────────────────────────────────────────


def test_exec_summary_with_exit_marker(tmp_path: Path) -> None:
    from nanobot.agent.tools.shell import ExecTool
    tool = ExecTool(working_dir=str(tmp_path))
    out = tool.summarize_result({}, "line1\nline2\n[exit code: 0]")
    assert "exit code" in out
    assert "2 lines" in out


def test_exec_summary_error(tmp_path: Path) -> None:
    from nanobot.agent.tools.shell import ExecTool
    tool = ExecTool(working_dir=str(tmp_path))
    out = tool.summarize_result({}, "Error: command not found")
    assert "Error" in out


# ── ReadFileTool / WriteFileTool / EditFileTool / ListDirTool ────────────


def test_read_file_summary(tmp_path: Path) -> None:
    from nanobot.agent.tools.filesystem import ReadFileTool
    tool = ReadFileTool(workspace=tmp_path)
    out = tool.summarize_result({"path": "x"}, "alpha\nbeta\ngamma")
    assert "3 lines" in out
    assert "chars" in out


def test_read_file_summary_error(tmp_path: Path) -> None:
    from nanobot.agent.tools.filesystem import ReadFileTool
    tool = ReadFileTool(workspace=tmp_path)
    out = tool.summarize_result({"path": "x"}, "Error: file not found")
    assert "Error" in out


def test_write_file_summary_uses_input(tmp_path: Path) -> None:
    from nanobot.agent.tools.filesystem import WriteFileTool
    tool = WriteFileTool(workspace=tmp_path)
    out = tool.summarize_result(
        {"path": "x.py", "content": "import os\nimport sys\n"},
        "Successfully wrote",
    )
    assert "wrote" in out
    assert "line" in out


def test_edit_file_summary_shows_delta(tmp_path: Path) -> None:
    from nanobot.agent.tools.filesystem import EditFileTool
    tool = EditFileTool(workspace=tmp_path)
    out = tool.summarize_result(
        {"old_string": "a", "new_string": "a\nb\nc"},
        "Successfully edited",
    )
    assert "+2" in out


def test_edit_file_summary_shrink(tmp_path: Path) -> None:
    from nanobot.agent.tools.filesystem import EditFileTool
    tool = EditFileTool(workspace=tmp_path)
    out = tool.summarize_result(
        {"old_string": "a\nb\nc", "new_string": "a"},
        "Successfully edited",
    )
    assert "-2" in out


def test_list_dir_summary(tmp_path: Path) -> None:
    from nanobot.agent.tools.filesystem import ListDirTool
    tool = ListDirTool(workspace=tmp_path)
    result = "📄 file1.py\n📄 file2.py\n📁 subdir\n"
    out = tool.summarize_result({"path": "."}, result)
    assert "3 entries" in out


# ── WebSearchTool / WebFetchTool ─────────────────────────────────────────


def test_web_search_summary_counts_numbered() -> None:
    from nanobot.agent.tools.web import WebSearchTool
    tool = WebSearchTool()
    result = "Results for: foo\n\n1. Title A\n   url1\n2. Title B\n   url2\n3. Title C\n   url3"
    out = tool.summarize_result({"query": "foo"}, result)
    assert "3 results" in out


def test_web_search_summary_no_results() -> None:
    from nanobot.agent.tools.web import WebSearchTool
    tool = WebSearchTool()
    out = tool.summarize_result({"query": "x"}, "No results for: x")
    assert "no results" in out


def test_web_fetch_summary_parses_json() -> None:
    from nanobot.agent.tools.web import WebFetchTool
    import json
    tool = WebFetchTool()
    payload = json.dumps({
        "url": "https://example.com", "status": 200, "length": 1234,
        "truncated": False, "text": "hello",
    })
    out = tool.summarize_result({"url": "x"}, payload)
    assert "HTTP 200" in out
    assert "1234 chars" in out


def test_web_fetch_summary_truncated() -> None:
    from nanobot.agent.tools.web import WebFetchTool
    import json
    tool = WebFetchTool()
    payload = json.dumps({"status": 200, "length": 5000, "truncated": True})
    out = tool.summarize_result({"url": "x"}, payload)
    assert "truncated" in out


def test_web_fetch_summary_error() -> None:
    from nanobot.agent.tools.web import WebFetchTool
    import json
    tool = WebFetchTool()
    payload = json.dumps({"error": "timeout", "url": "x"})
    out = tool.summarize_result({"url": "x"}, payload)
    assert "Error" in out


# ── Tool base default ────────────────────────────────────────────────────


def test_tool_base_summarize_result_default() -> None:
    """Tools that don't override return empty string."""
    from nanobot.fork.agent.tools.todo import TodoWriteTool
    from nanobot.session.manager import SessionManager
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sessions = SessionManager(Path(tmp))
        tool = TodoWriteTool(sessions=sessions)
        # TodoWriteTool doesn't define summarize_result → uses base default
        assert tool.summarize_result({}, "anything") == ""


# ── LoadSkillTool: dedicated skill loader with section-count summary ───


def _make_loader_with_skill(tmp_path: Path, name: str, body: str):
    from nanobot.agent.skills import SkillsLoader
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return SkillsLoader(tmp_path)


def test_load_skill_returns_wrapped_skill_content(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    loader = _make_loader_with_skill(tmp_path, "foo", "# foo\n## A\nbody")
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="foo"))
    assert out.startswith('<skill name="foo">')
    assert out.endswith("</skill>")
    assert "# foo" in out and "## A" in out


def test_load_skill_summary_counts_sections(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    body = "# foo\n## Overview\n...\n## Usage\n...\n## Examples\n...\n"
    loader = _make_loader_with_skill(tmp_path, "foo", body)
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="foo"))
    assert tool.summarize_result({"name": "foo"}, out) == "loaded: 3 sections"


def test_load_skill_unknown_lists_available(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    loader = _make_loader_with_skill(tmp_path, "alpha", "# alpha\n")
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="nope"))
    assert "Error" in out and "alpha" in out


def test_load_skill_empty_name_returns_error(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    loader = _make_loader_with_skill(tmp_path, "x", "# x\n")
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name=""))
    assert "Error" in out


def test_load_skill_no_sections_falls_back_to_line_count(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    body = "Just plain text\nwith no headers\nat all\n"
    loader = _make_loader_with_skill(tmp_path, "plain", body)
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="plain"))
    summary = tool.summarize_result({"name": "plain"}, out)
    assert summary.startswith("loaded:") and "line" in summary


def test_read_file_summary_no_longer_specializes_skill_md(tmp_path: Path) -> None:
    """ReadFileTool reverts to plain N lines / M chars — LoadSkillTool
    owns the skill-flavoured summary now."""
    from nanobot.agent.tools.filesystem import ReadFileTool
    tool = ReadFileTool(workspace=tmp_path)
    out = tool.summarize_result(
        {"path": "skills/foo/SKILL.md"},
        "## A\n## B\nbody\n",
    )
    assert "loaded:" not in out
    assert "line" in out and "chars" in out


# ── frontmatter version extraction in LoadSkillTool summary ─────────────


def test_load_skill_summary_includes_version_from_frontmatter(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    body = "---\nversion: 1.2\nauthor: x\n---\n# foo\n## A\n## B\n"
    loader = _make_loader_with_skill(tmp_path, "foo", body)
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="foo"))
    summary = tool.summarize_result({"name": "foo"}, out)
    assert summary == "loaded: v1.2, 2 sections"


def test_load_skill_summary_no_version_when_no_frontmatter(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    body = "# foo\n## A\n## B\n"
    loader = _make_loader_with_skill(tmp_path, "foo", body)
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="foo"))
    summary = tool.summarize_result({"name": "foo"}, out)
    assert summary == "loaded: 2 sections"
    assert "v" not in summary or "version" not in summary


def test_load_skill_summary_handles_quoted_version(tmp_path: Path) -> None:
    import asyncio
    from nanobot.fork.agent.tools.skill import LoadSkillTool
    body = '---\nversion: "2.0.1"\n---\n# x\n## A\n'
    loader = _make_loader_with_skill(tmp_path, "x", body)
    tool = LoadSkillTool(loader=loader)
    out = asyncio.run(tool.execute(name="x"))
    summary = tool.summarize_result({"name": "x"}, out)
    assert "v2.0.1" in summary
