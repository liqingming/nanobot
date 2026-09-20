"""Tests for grep search tools."""

from __future__ import annotations

import asyncio
import os
import re
import threading
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
from nanobot.providers.base import GenerationSettings
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    default_workspace_scope,
    reset_workspace_scope,
)
from nanobot.utils.llm_runtime import LLMRuntime


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


@pytest.mark.parametrize("tool_name", ["find_files", "grep"])
@pytest.mark.parametrize(
    ("glob", "expected"),
    [
        ("**/*.py", ["main.py", "src/api.py", "src/nested/deep/worker.py"]),
        ("src/**/*.py", ["src/api.py", "src/nested/deep/worker.py"]),
        ("src/**", ["src/api.py", "src/nested/deep/worker.py"]),
        ("src/*.py", ["src/api.py"]),
        ("src/**/deep/*.py", ["src/nested/deep/worker.py"]),
        (r"src\**\*.py", ["src/api.py", "src/nested/deep/worker.py"]),
    ],
)
async def test_search_recursive_glob_matches_zero_or_more_directories(
    tmp_path: Path, tool_name: str, glob: str, expected: list[str]
) -> None:
    for name in ["main.py", "src/api.py", "src/nested/deep/worker.py", "notes.md"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("needle\n", encoding="utf-8")

    if tool_name == "find_files":
        result = await FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path).execute(
            path=".", glob=glob,
        )
    else:
        result = await GrepTool(workspace=tmp_path, allowed_dir=tmp_path).execute(
            pattern="needle", path=".", glob=glob, output_mode="files_with_matches",
        )

    assert sorted(result.splitlines()) == expected


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
async def test_find_files_worker_preserves_current_workspace_scope(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "inside.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("nope\n", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, restrict_to_workspace=False)
    token = bind_workspace_scope(default_workspace_scope(project, True))
    try:
        result = await tool.execute(path=".")
    finally:
        reset_workspace_scope(token)

    assert result == "inside.txt"


@pytest.mark.asyncio
async def test_find_files_scan_keeps_event_loop_responsive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "match.txt"
    target.write_text("ok\n", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    original_iter_paths = tool._iter_paths
    started = threading.Event()
    release = threading.Event()

    def blocking_iter_paths(root: Path, *, include_dirs: bool, budget):
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release find_files traversal")
        yield from original_iter_paths(root, include_dirs=include_dirs, budget=budget)

    monkeypatch.setattr(tool, "_iter_paths", blocking_iter_paths)
    task = asyncio.create_task(tool.execute(path="."))
    try:
        assert await asyncio.to_thread(started.wait, 0.5)
        for _ in range(3):
            await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()

    assert await asyncio.wait_for(task, timeout=0.5) == "match.txt"


@pytest.mark.asyncio
async def test_find_files_cancellation_stops_worker_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    started = threading.Event()
    stopped = threading.Event()

    def cancellable_iter_paths(root: Path, *, include_dirs: bool, budget):
        del root, include_dirs
        started.set()
        if not budget.cancelled.wait(timeout=1):
            raise TimeoutError("find_files worker did not receive cancellation")
        stopped.set()
        if False:
            yield

    monkeypatch.setattr(tool, "_iter_paths", cancellable_iter_paths)
    task = asyncio.create_task(tool.execute(path="."))
    assert await asyncio.to_thread(started.wait, 0.5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.2)

    assert await asyncio.to_thread(stopped.wait, 0.5)


@pytest.mark.asyncio
async def test_find_files_path_limit_stops_after_pagination_lookahead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / name for name in ("a.txt", "b.txt", "c.txt")]
    for path in paths:
        path.write_text("ok\n", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)
    visited: list[str] = []

    def ordered_iter_paths(root: Path, *, include_dirs: bool, budget):
        del include_dirs
        for path in paths:
            budget.checkpoint()
            visited.append(path.name)
            yield tool._entry(path, root, is_dir=False)

    monkeypatch.setattr(tool, "_iter_paths", ordered_iter_paths)
    result = await tool.execute(path=".", head_limit=1)

    assert result.splitlines()[0] == "a.txt"
    assert "pagination: limit=1, offset=0" in result
    assert visited == ["a.txt", "b.txt"]


@pytest.mark.asyncio
async def test_find_files_path_budget_counts_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    monkeypatch.setattr(FindFilesTool, "_MAX_SCAN_PATHS", 1)
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)

    result = await tool.execute(path=".")

    assert result.startswith("Error: find_files scan exceeded 1 paths")


@pytest.mark.asyncio
async def test_find_files_enforces_time_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "match.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(FindFilesTool, "_MAX_SCAN_SECONDS", 0.0)
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)

    result = await tool.execute(path=".")

    assert result.startswith("Error: find_files scan exceeded 0 seconds")


@pytest.mark.asyncio
async def test_find_files_path_sort_matches_existing_lexicographic_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "inside.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "a+").write_text("ok\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("ok\n", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)

    result = await tool.execute(path=".", head_limit=0)

    assert result.splitlines() == ["a+", "a.py", "a/inside.txt"]


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
async def test_grep_explicit_content_includes_requested_context(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "\n".join(f"line {line}" for line in range(1, 6))
        + "\nmatch_here\n"
        + "\n".join(f"line {line}" for line in range(7, 13)),
        encoding="utf-8",
    )

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        output_mode="content", context_before=5, context_after=5, pattern="match_here",
        path="src",
    )

    assert "src/main.py:6" in result
    assert "  1| line 1" in result
    assert "> 6| match_here" in result
    assert "  11| line 11" in result
    assert "line 12" not in result


@pytest.mark.asyncio
async def test_grep_searches_xlsx_with_sheet_cell_locator(tmp_path: Path) -> None:
    from openpyxl import Workbook

    workbook_path = tmp_path / "people.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "People"
    sheet.append(["Name", "Role"])
    sheet.append(["Ada", "Engineer"])
    workbook.save(workbook_path)
    workbook.close()

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        output_mode="content", pattern="Engineer",
        path="people.xlsx",
        fixed_strings=True,
    )

    assert "people.xlsx:3" in result
    assert "sheet='People',row=2,cell=B2" in result
    assert "Ada\tEngineer" in result


@pytest.mark.asyncio
async def test_grep_searches_docx_with_paragraph_locator(tmp_path: Path) -> None:
    from docx import Document

    document_path = tmp_path / "notes.docx"
    document = Document()
    document.add_paragraph("Introduction")
    document.add_paragraph("late needle")
    document.save(document_path)

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        output_mode="content", pattern="late needle",
        path="notes.docx",
        fixed_strings=True,
    )

    assert "notes.docx:3 [paragraph=2]" in result
    assert "late needle" in result


@pytest.mark.asyncio
async def test_grep_searches_pptx_with_slide_locator(tmp_path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    presentation_path = tmp_path / "deck.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    textbox = slide.shapes.add_textbox(
        Inches(1), Inches(1), Inches(4), Inches(1)
    )
    textbox.text_frame.text = "slide needle"
    presentation.save(presentation_path)

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        output_mode="content", pattern="slide needle",
        path="deck.pptx",
        fixed_strings=True,
    )

    assert "deck.pptx:2 [slide=1,line=1]" in result
    assert "slide needle" in result


@pytest.mark.asyncio
async def test_grep_searches_pdf_with_page_locator(tmp_path: Path) -> None:
    import fitz

    pdf_path = tmp_path / "notes.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "pdf needle")
    document.save(pdf_path)
    document.close()

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        output_mode="content", pattern="pdf needle",
        path="notes.pdf",
        fixed_strings=True,
    )

    assert "notes.pdf:2 [page=1,line=1]" in result
    assert "pdf needle" in result


@pytest.mark.asyncio
async def test_grep_searches_xlsx_beyond_attachment_preview_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openpyxl import Workbook

    from nanobot.utils import document as document_utils

    workbook_path = tmp_path / "long.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    for row in range(1, 20):
        sheet.append([f"ordinary-row-{row}"])
    sheet.append(["late-needle"])
    workbook.save(workbook_path)
    workbook.close()
    monkeypatch.setattr(document_utils, "_MAX_TEXT_LENGTH", 50)

    preview = document_utils.extract_text(workbook_path)
    assert preview is not None
    assert "late-needle" not in preview

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        output_mode="content", pattern="late-needle",
        path="long.xlsx",
        fixed_strings=True,
    )

    assert "late-needle" in result
    assert "sheet='Data',row=20,cell=A20" in result
    assert "No matches found" not in result


@pytest.mark.asyncio
async def test_grep_keeps_an_oversized_matching_line_visible(tmp_path: Path) -> None:
    long_line = "x" * 130_000 + "needle" + "y" * 10_000
    (tmp_path / "huge-line.txt").write_text(long_line, encoding="utf-8")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    result = await tool.execute(
        output_mode="content", pattern="needle",
        path="huge-line.txt",
        fixed_strings=True,
        context_before=0,
        context_after=0,
    )

    assert "huge-line.txt:1" in result
    assert "needle" in result
    assert "No matches found" not in result
    assert len(result) < GrepTool._MAX_RESULT_CHARS


@pytest.mark.asyncio
async def test_grep_size_limit_returns_a_resumable_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "\n".join(f"needle-{line}-" + "x" * 80 for line in range(1, 11))
    (tmp_path / "many.txt").write_text(content, encoding="utf-8")
    monkeypatch.setattr(GrepTool, "_MAX_RESULT_CHARS", 350)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    first = await tool.execute(
        output_mode="content", pattern="needle",
        path="many.txt",
        fixed_strings=True,
        context_before=0,
        context_after=0,
        head_limit=10,
    )
    continuation = re.search(r"use offset=(\d+) to continue", first)

    assert continuation is not None
    next_offset = int(continuation.group(1))
    assert next_offset > 0

    second = await tool.execute(
        output_mode="content", pattern="needle",
        path="many.txt",
        fixed_strings=True,
        context_before=0,
        context_after=0,
        head_limit=10,
        offset=next_offset,
    )
    first_headers = {
        line for line in first.splitlines() if line.startswith("many.txt:")
    }
    second_headers = {
        line for line in second.splitlines() if line.startswith("many.txt:")
    }
    assert second_headers
    assert first_headers.isdisjoint(second_headers)


@pytest.mark.asyncio
async def test_grep_reports_an_invalid_pdf_page_range(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    pdf_path = tmp_path / "one-page.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with pdf_path.open("wb") as output:
        writer.write(output)

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(pattern="needle", path="one-page.pdf", pages="bad")

    assert result.startswith("Error: Invalid PDF page range 'bad'")
    assert "binary/unreadable" not in result

    out_of_bounds = await tool.execute(
        pattern="needle",
        path="one-page.pdf",
        pages="99",
    )
    assert out_of_bounds == (
        "Error: Invalid PDF page range '99': document has 1 page; "
        "use a page number or range within 1-1."
    )


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
        output_mode="files_with_matches",
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
        output_mode="files_with_matches",
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
async def test_grep_uses_a_larger_bounded_limit_for_an_explicit_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large_file = tmp_path / "history.jsonl"
    large_file.write_text("needle\n" + "x" * 20, encoding="utf-8")
    monkeypatch.setattr(GrepTool, "_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(GrepTool, "_MAX_EXPLICIT_FILE_BYTES", 100)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    explicit_result = await tool.execute(
        pattern="needle",
        path=str(large_file),
        output_mode="content",
    )
    directory_result = await tool.execute(pattern="needle", path=".")
    monkeypatch.setattr(GrepTool, "_MAX_EXPLICIT_FILE_BYTES", 10)
    capped_result = await tool.execute(pattern="needle", path=str(large_file))

    assert "needle" in explicit_result
    assert "skipped 1 large files" in directory_result
    assert "skipped 1 large files" in capped_result


def test_grep_schema_is_concise_and_hides_internal_limits(tmp_path: Path) -> None:
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    properties = tool.parameters["properties"]

    assert len(tool.description) < 150
    assert "head_limit" in properties
    assert "max_matches" not in properties
    assert "max_results" not in properties


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
    provider.generation = GenerationSettings()
    mgr = SubagentManager(
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=4096,
    )
    captured: dict[str, list[str]] = {}

    async def fake_run(spec):
        captured["tool_names"] = spec.tools.tool_names
        return SimpleNamespace(
            usage=None,
            stop_reason="ok",
            final_content="done",
            tool_events=[],
            error=None,
        )

    mgr.runner.run = fake_run
    mgr._announce_result = AsyncMock()

    status = SubagentStatus(task_id="sub-1", label="label", task_description="search task", started_at=time.monotonic())
    await mgr._run_subagent(
        "sub-1",
        "search task",
        "label",
        {"channel": "cli", "chat_id": "direct"},
        status,
        LLMRuntime.capture(provider, "test-model", context_window_tokens=128_000),
    )

    assert "find_files" in captured["tool_names"]
    assert "grep" in captured["tool_names"]


def test_subagent_prompt_respects_disabled_skills(tmp_path: Path) -> None:
    bus = MessageBus()
    skills_dir = tmp_path / "skills"
    (skills_dir / "alpha").mkdir(parents=True)
    (skills_dir / "alpha" / "SKILL.md").write_text("# Alpha\n\nhidden\n", encoding="utf-8")
    (skills_dir / "beta").mkdir(parents=True)
    (skills_dir / "beta" / "SKILL.md").write_text("# Beta\n\nshown\n", encoding="utf-8")

    mgr = SubagentManager(
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


@pytest.mark.asyncio
async def test_find_files_query_modes_and_all_mode_hint(tmp_path: Path) -> None:
    (tmp_path / "thzxiu8m.xml").write_text("x", encoding="utf-8")
    (tmp_path / "au3n6j.xml").write_text("x", encoding="utf-8")
    tool = FindFilesTool(workspace=tmp_path, allowed_dir=tmp_path)

    all_result = await tool.execute(path=".", query="thzxiu8m.xml au3n6j.xml")
    any_result = await tool.execute(
        path=".", query="thzxiu8m.xml au3n6j.xml", query_mode="any"
    )
    phrase_result = await tool.execute(path=".", query="thzxiu8m", query_mode="phrase")

    assert "No files found" in all_result
    assert "query_mode='all'" in all_result
    assert "query_mode='any'" in all_result
    assert any_result.splitlines() == ["au3n6j.xml", "thzxiu8m.xml"]
    assert phrase_result == "thzxiu8m.xml"


@pytest.mark.asyncio
async def test_grep_broad_parent_scope_requires_model_confirmation(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "a.txt").write_text("needle", encoding="utf-8")
    tool = GrepTool(workspace=workspace, allowed_dir=tmp_path)

    checkpoint = await tool.execute(pattern="needle", path=str(tmp_path), glob="**/*.txt")
    confirmed = await tool.execute(
        pattern="needle",
        path=str(tmp_path),
        glob="**/*.txt",
        confirm_broad_search=True,
    )

    assert "Broad search confirmation required" in checkpoint
    assert "confirm_broad_search=true" in checkpoint
    assert confirmed == "a.txt"


@pytest.mark.asyncio
async def test_grep_budget_checkpoint_can_continue_with_scan_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for idx in range(5):
        (tmp_path / f"f{idx}.txt").write_text(
            "needle" if idx == 4 else "nope", encoding="utf-8"
        )
    monkeypatch.setattr(GrepTool, "_SCAN_BUDGET_FILES", 2)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    first = await tool.execute(pattern="needle", path=".")
    second = await tool.execute(
        pattern="needle", path=".", scan_cursor=2, confirm_broad_search=True
    )
    third = await tool.execute(
        pattern="needle", path=".", scan_cursor=4, confirm_broad_search=True
    )

    assert "Search budget checkpoint" in first
    assert "scan_cursor=2" in first
    assert "Search budget checkpoint" in second
    assert "scan_cursor=4" in second
    assert "f4.txt" in third


@pytest.mark.asyncio
async def test_directory_search_calls_are_not_concurrency_safe(tmp_path: Path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("needle", encoding="utf-8")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    assert tool.is_concurrency_safe_call({"path": str(file_path)}) is True
    assert tool.is_concurrency_safe_call({"path": "."}) is False


@pytest.mark.asyncio
async def test_grep_path_sort_stops_after_requested_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for idx in range(10):
        (tmp_path / f"f{idx}.txt").write_text("needle", encoding="utf-8")
    reads: list[str] = []
    original = Path.open

    def tracked_read(path: Path, *args, **kwargs):
        reads.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_read)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    result = await tool.execute(
        pattern="needle",
        path=".",
        output_mode="files_with_matches",
        sort="path",
        head_limit=2,
    )

    assert result.splitlines()[:2] == ["f0.txt", "f1.txt"]
    assert "pagination" in result
    assert reads == ["f0.txt", "f1.txt", "f2.txt"]
