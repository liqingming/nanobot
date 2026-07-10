"""Tests for ContextBuilder — system prompt and message assembly."""

from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.session.goal_state import GOAL_STATE_KEY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builder(tmp_path: Path, **kw) -> ContextBuilder:
    return ContextBuilder(workspace=tmp_path, **kw)


def _write_skill(root: Path, name: str, description: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "\n".join([
            "---",
            f"description: {description}",
            "---",
            "",
            f"# {name}",
        ]),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Skills roots
# ---------------------------------------------------------------------------


class TestSkillsRoots:
    def test_workspace_claude_skills_loaded_when_data_dir_is_separate(self, tmp_path):
        workspace = tmp_path / "project"
        data_dir = tmp_path / "data"
        claude_skill = _write_skill(
            workspace / ".claude" / "skills",
            "project_skill",
            "Project-local Claude skill",
        )
        data_dir.mkdir(parents=True)

        builder = ContextBuilder(workspace=workspace, data_dir=data_dir)
        entries = builder.skills.list_skills(filter_unavailable=False)
        project_entries = [entry for entry in entries if entry["name"] == "project_skill"]

        assert project_entries == [{
            "name": "project_skill",
            "path": str(claude_skill),
            "source": "claude",
        }]
        assert "Project-local Claude skill" in builder.skills.build_skills_summary()

    def test_workspace_without_claude_dir_uses_data_dir_skills_only(self, tmp_path):
        workspace = tmp_path / "project"
        data_dir = tmp_path / "data"
        data_skill = _write_skill(data_dir / "skills", "data_skill", "Nanobot data skill")
        workspace.mkdir(parents=True)

        builder = ContextBuilder(workspace=workspace, data_dir=data_dir)
        entries = builder.skills.list_skills(filter_unavailable=False)
        data_entries = [entry for entry in entries if entry["name"] == "data_skill"]

        assert data_entries == [{
            "name": "data_skill",
            "path": str(data_skill),
            "source": "workspace",
        }]
        assert all(entry["source"] != "claude" for entry in entries)


# ---------------------------------------------------------------------------
# _build_runtime_context (static)
# ---------------------------------------------------------------------------


class TestBuildRuntimeContext:
    def test_time_only(self):
        ctx = ContextBuilder._build_runtime_context(None, None)
        assert "[Runtime Context" in ctx
        assert "[/Runtime Context]" in ctx
        assert "Current Time:" in ctx
        assert "Channel:" not in ctx

    def test_with_channel_and_chat_id(self):
        ctx = ContextBuilder._build_runtime_context("telegram", "chat123")
        assert "Channel: telegram" in ctx
        assert "Chat ID: chat123" in ctx

    def test_with_sender_id(self):
        ctx = ContextBuilder._build_runtime_context("cli", "direct", sender_id="user1")
        assert "Sender ID: user1" in ctx

    def test_with_timezone(self):
        ctx = ContextBuilder._build_runtime_context(None, None, timezone="Asia/Shanghai")
        assert "Current Time:" in ctx

    def test_no_channel_no_chat_id_omits_both(self):
        ctx = ContextBuilder._build_runtime_context(None, None)
        assert "Channel:" not in ctx
        assert "Chat ID:" not in ctx

    def test_no_sender_id_omits(self):
        ctx = ContextBuilder._build_runtime_context("cli", "direct")
        assert "Sender ID:" not in ctx


# ---------------------------------------------------------------------------
# _merge_message_content (static)
# ---------------------------------------------------------------------------


class TestMergeMessageContent:
    def test_str_plus_str(self):
        result = ContextBuilder._merge_message_content("hello", "world")
        assert result == "hello\n\nworld"

    def test_empty_left_plus_str(self):
        result = ContextBuilder._merge_message_content("", "world")
        assert result == "world"

    def test_list_plus_list(self):
        left = [{"type": "text", "text": "a"}]
        right = [{"type": "text", "text": "b"}]
        result = ContextBuilder._merge_message_content(left, right)
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "b"

    def test_str_plus_list(self):
        right = [{"type": "text", "text": "b"}]
        result = ContextBuilder._merge_message_content("hello", right)
        assert len(result) == 2
        assert result[0]["text"] == "hello"
        assert result[1]["text"] == "b"

    def test_list_plus_str(self):
        left = [{"type": "text", "text": "a"}]
        result = ContextBuilder._merge_message_content(left, "world")
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "world"

    def test_none_plus_str(self):
        result = ContextBuilder._merge_message_content(None, "hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_str_plus_none(self):
        result = ContextBuilder._merge_message_content("hello", None)
        assert result == [{"type": "text", "text": "hello"}]

    def test_none_plus_none(self):
        result = ContextBuilder._merge_message_content(None, None)
        assert result == []

    def test_list_items_not_dicts_wrapped(self):
        result = ContextBuilder._merge_message_content(["raw_item"], None)
        assert result == [{"type": "text", "text": "raw_item"}]


# ---------------------------------------------------------------------------
# _load_bootstrap_files
# ---------------------------------------------------------------------------


class TestLoadBootstrapFiles:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    def test_no_bootstrap_files(self, tmp_path):
        builder = _builder(tmp_path)
        assert builder._load_bootstrap_files() == ""

    def test_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be helpful.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "## AGENTS.md" in result
        assert "Be helpful." in result

    def test_multiple_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Rules.", encoding="utf-8")
        (tmp_path / "SOUL.md").write_text("Soul.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "## AGENTS.md" in result
        assert "## SOUL.md" in result
        assert "Rules." in result
        assert "Soul." in result

    def test_all_bootstrap_files(self, tmp_path):
        for name in ContextBuilder.BOOTSTRAP_FILES:
            (tmp_path / name).write_text(f"Content of {name}", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        for name in ContextBuilder.BOOTSTRAP_FILES:
            assert f"## {name}" in result

    def test_loads_global_and_workspace_claude_md_before_agents(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        workspace = tmp_path / "project"
        global_claude = home / ".claude" / "CLAUDE.md"
        workspace_claude = workspace / "CLAUDE.md"
        global_claude.parent.mkdir(parents=True)
        workspace.mkdir()
        global_claude.write_text("Global Claude rules.", encoding="utf-8")
        workspace_claude.write_text("Workspace Claude rules.", encoding="utf-8")
        (workspace / "AGENTS.md").write_text("Nanobot rules.", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: home)

        builder = _builder(workspace)
        result = builder._load_bootstrap_files()

        assert "## ~/.claude/CLAUDE.md" in result
        assert "Global Claude rules." in result
        assert "## CLAUDE.md" in result
        assert "Workspace Claude rules." in result
        assert result.index("## ~/.claude/CLAUDE.md") < result.index("## CLAUDE.md")
        assert result.index("## CLAUDE.md") < result.index("## AGENTS.md")

    def test_workspace_claude_md_does_not_fallback_to_data_dir(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        workspace = tmp_path / "project"
        data_dir = tmp_path / "data"
        workspace.mkdir()
        data_dir.mkdir()
        (data_dir / "CLAUDE.md").write_text("Data dir Claude rules.", encoding="utf-8")
        (workspace / "CLAUDE.md").write_text("Workspace Claude rules.", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: home)

        builder = ContextBuilder(workspace=workspace, data_dir=data_dir)
        result = builder._load_bootstrap_files()

        assert "Workspace Claude rules." in result
        assert "Data dir Claude rules." not in result

    def test_legacy_tools_md_is_not_bootstrapped(self, tmp_path):
        (tmp_path / "TOOLS.md").write_text("workspace tool notes", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "TOOLS.md" not in result
        assert "workspace tool notes" not in result

    def test_utf8_content(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("用中文回复", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "用中文回复" in result


# ---------------------------------------------------------------------------
# _is_template_content (static)
# ---------------------------------------------------------------------------


class TestIsTemplateContent:
    def test_nonexistent_template_returns_false(self):
        assert ContextBuilder._is_template_content("anything", "nonexistent/path.md") is False

    def test_content_matching_template(self):
        from importlib.resources import files as pkg_files
        tpl = pkg_files("nanobot") / "templates" / "memory" / "MEMORY.md"
        if not tpl.is_file():
            pytest.skip("MEMORY.md template not bundled")
        original = tpl.read_text(encoding="utf-8")
        assert ContextBuilder._is_template_content(original, "memory/MEMORY.md") is True

    def test_modified_content_returns_false(self):
        from importlib.resources import files as pkg_files
        tpl = pkg_files("nanobot") / "templates" / "memory" / "MEMORY.md"
        if not tpl.is_file():
            pytest.skip("MEMORY.md template not bundled")
        assert ContextBuilder._is_template_content("totally different", "memory/MEMORY.md") is False


# ---------------------------------------------------------------------------
# Bundled bootstrap templates
# ---------------------------------------------------------------------------


class TestBundledToolContract:
    def test_tool_contract_balances_general_and_coding_workflows(self):
        from importlib.resources import files as pkg_files

        tpl = pkg_files("nanobot") / "templates" / "agent" / "tool_contract.md"
        content = tpl.read_text(encoding="utf-8")

        assert "## General Tool Contract" in content
        assert "Use the narrowest structured tool" in content
        assert "Do not use `exec` as a universal workaround" in content
        assert "## File and Coding Workflows" in content
        assert "apply_patch" in content
        assert "## Web and External Information" in content
        assert "## Messaging and Media" in content
        assert "## Scheduling and Background Work" in content
        assert "pure coding" not in content.lower()

    def test_tool_contract_is_injected_without_workspace_file(self, tmp_path):
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt()

        assert "# Tool Usage Notes" in prompt
        assert "## General Tool Contract" in prompt
        assert "Do not use `exec` as a universal workaround" in prompt


# ---------------------------------------------------------------------------
# _build_user_content
# ---------------------------------------------------------------------------


class TestBuildUserContent:
    def test_no_media_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", None)
        assert result == "hello"

    def test_empty_media_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [])
        assert result == "hello"

    def test_nonexistent_media_file_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", ["/nonexistent/image.png"])
        assert result == "hello"

    def test_non_image_file_returns_string(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("not an image", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [str(txt)])
        assert result == "hello"

    def test_valid_image_returns_list(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [str(png)])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "hello"

    def test_image_meta_includes_path(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [str(png)])
        assert "_meta" in result[0]
        assert "path" in result[0]["_meta"]


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_returns_nonempty_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_identity_section(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "workspace" in result.lower() or "python" in result.lower()

    def test_includes_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be helpful and concise.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "Be helpful and concise." in result

    def test_includes_session_summary(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt(session_summary="Previous chat about Python.")
        assert "Previous chat about Python." in result
        assert "[Archived Context Summary]" in result

    def test_sections_separated_by_separator(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Rules.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder.build_system_prompt(session_summary="Summary.")
        assert "\n\n---\n\n" in result

    def test_no_bootstrap_no_summary(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "## AGENTS.md" not in result
        assert "[Archived Context Summary]" not in result


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_basic_empty_history(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "hello" in str(messages[1]["content"])

    def test_runtime_context_injected(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello", channel="cli", chat_id="direct")
        user_msg = str(messages[-1]["content"])
        assert "[Runtime Context" in user_msg
        assert "hello" in user_msg

    def test_session_metadata_injects_active_goal_state(self, tmp_path):
        builder = _builder(tmp_path)
        meta = {
            GOAL_STATE_KEY: {"status": "active", "objective": "Finish docs migration."},
        }
        messages = builder.build_messages(
            [],
            "hi",
            channel="cli",
            chat_id="x",
            session_metadata=meta,
        )
        user_msg = str(messages[-1]["content"])
        assert "Goal (active):" in user_msg
        assert "Finish docs migration." in user_msg

    def test_goal_state_does_not_leak_without_session_metadata(self, tmp_path):
        builder = _builder(tmp_path)
        other_session_meta = {
            GOAL_STATE_KEY: {"status": "active", "objective": "Other chat goal."},
        }

        with_goal = builder.build_messages(
            [],
            "hi",
            channel="websocket",
            chat_id="chat-a",
            session_metadata=other_session_meta,
        )
        without_goal = builder.build_messages(
            [],
            "hi",
            channel="websocket",
            chat_id="chat-b",
            session_metadata={},
        )

        assert "Other chat goal." in str(with_goal[-1]["content"])
        assert "Other chat goal." not in str(without_goal[-1]["content"])
        assert "Goal (active):" not in str(without_goal[-1]["content"])

    def test_current_runtime_lines_are_injected(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages(
            [],
            "please use @zoom tonight",
            current_runtime_lines=[
                "CLI App Attachment: @zoom (installed; tool=run_cli_app; entry_point=cli-anything-zoom).",
            ],
        )
        user_msg = str(messages[-1]["content"])

        assert "CLI App Attachment: @zoom" in user_msg
        assert "tool=run_cli_app" in user_msg
        assert "entry_point=cli-anything-zoom" in user_msg

    def test_consecutive_same_role_merged(self, tmp_path):
        builder = _builder(tmp_path)
        history = [{"role": "user", "content": "previous user message"}]
        messages = builder.build_messages(history, "new message")
        assert len(messages) == 2  # system + merged user
        assert "previous user message" in str(messages[1]["content"])
        assert "new message" in str(messages[1]["content"])

    def test_different_role_appended(self, tmp_path):
        builder = _builder(tmp_path)
        history = [{"role": "assistant", "content": "previous response"}]
        messages = builder.build_messages(history, "new message")
        assert len(messages) == 3  # system + assistant + user

    def test_media_with_history(self, tmp_path):
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        history = [{"role": "assistant", "content": "see this"}]
        messages = builder.build_messages(history, "check image", media=[str(png)])
        user_msg = messages[-1]["content"]
        assert isinstance(user_msg, list)
        assert any(b.get("type") == "image_url" for b in user_msg)


def test_build_system_prompt_uses_request_workspace_claude_skills(tmp_path):
    default_workspace = tmp_path / "default"
    data_dir = tmp_path / "data"
    request_workspace = tmp_path / "request_project"
    default_workspace.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    request_workspace.mkdir(parents=True)
    _write_skill(
        request_workspace / ".claude" / "skills",
        "request_skill",
        "Request workspace skill",
    )

    builder = ContextBuilder(workspace=default_workspace, data_dir=data_dir)
    prompt = builder.build_system_prompt(workspace=request_workspace)

    assert "request_skill" in prompt
    assert "Request workspace skill" in prompt
