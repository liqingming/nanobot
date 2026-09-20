"""Tests for ContextBuilder — system prompt and message assembly."""

import hashlib
from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder, SystemPromptBuild, TranscriptInput
from nanobot.runtime_context import RuntimeContextBlock
from nanobot.session.goal_state import GOAL_STATE_KEY
from nanobot.utils.helpers import load_bundled_template

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

    def test_configured_skill_roots_persist_for_request_workspace(self, tmp_path):
        default_workspace = tmp_path / "default-project"
        request_workspace = tmp_path / "request-project"
        data_dir = tmp_path / "data"
        shared_root = tmp_path / "shared-skills"
        shared_skill = _write_skill(shared_root, "shared_skill", "Configured shared skill")
        request_skill = _write_skill(
            request_workspace / ".claude" / "skills",
            "request_skill",
            "Request-local skill",
        )
        default_workspace.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        builder = ContextBuilder(
            workspace=default_workspace,
            data_dir=data_dir,
            skill_roots=[str(shared_root)],
        )
        loader = builder._skills_for_workspace(request_workspace)
        entries = loader.list_skills(filter_unavailable=False)

        assert {entry["name"] for entry in entries} >= {"shared_skill", "request_skill"}
        assert next(entry for entry in entries if entry["name"] == "shared_skill") == {
            "name": "shared_skill",
            "path": str(shared_skill),
            "source": "configured-1",
        }
        assert next(entry for entry in entries if entry["name"] == "request_skill")["path"] == str(request_skill)


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

    def test_empty_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("\n", encoding="utf-8")
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
        local_claude = workspace / ".claude" / "CLAUDE.md"
        global_claude.parent.mkdir(parents=True)
        local_claude.parent.mkdir(parents=True)
        workspace.mkdir(exist_ok=True)
        global_claude.write_text("Global Claude rules.", encoding="utf-8")
        workspace_claude.write_text("Workspace Claude rules.", encoding="utf-8")
        local_claude.write_text("Local Claude rules.", encoding="utf-8")
        (workspace / "AGENTS.md").write_text("Nanobot rules.", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: home)

        builder = _builder(workspace)
        result = builder._load_bootstrap_files()

        assert "## ~/.claude/CLAUDE.md" in result
        assert "Global Claude rules." in result
        assert "## CLAUDE.md" in result
        assert "Workspace Claude rules." in result
        assert "## .claude/CLAUDE.md" in result
        assert "Local Claude rules." in result
        assert result.index("## ~/.claude/CLAUDE.md") < result.index("## CLAUDE.md")
        assert result.index("## CLAUDE.md") < result.index("## .claude/CLAUDE.md")
        assert result.index("## .claude/CLAUDE.md") < result.index("## AGENTS.md")

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

    def test_selected_project_supplies_only_agents_file(self, tmp_path):
        agent_home = tmp_path / "agent-home"
        project = tmp_path / "project"
        agent_home.mkdir()
        project.mkdir()
        (agent_home / "AGENTS.md").write_text("global project rules", encoding="utf-8")
        (agent_home / "SOUL.md").write_text("global soul", encoding="utf-8")
        (agent_home / "USER.md").write_text("global user", encoding="utf-8")
        (project / "AGENTS.md").write_text("selected project rules", encoding="utf-8")
        (project / "SOUL.md").write_text("project soul collision", encoding="utf-8")
        (project / "USER.md").write_text("project user collision", encoding="utf-8")

        result = ContextBuilder(agent_home).build_system_prompt(workspace=project)

        assert "selected project rules" in result
        assert "global project rules" not in result
        assert "global soul" in result
        assert "global user" in result
        assert "project soul collision" not in result
        assert "project user collision" not in result

    def test_selected_project_without_agents_does_not_fall_back(self, tmp_path):
        agent_home = tmp_path / "agent-home"
        project = tmp_path / "project"
        agent_home.mkdir()
        project.mkdir()
        (agent_home / "AGENTS.md").write_text("default workspace rules", encoding="utf-8")

        result = ContextBuilder(agent_home).build_system_prompt(workspace=project)

        assert "default workspace rules" not in result

    def test_unmodified_agents_and_user_templates_are_skipped(self, tmp_path):
        from nanobot.utils.helpers import sync_workspace_templates

        sync_workspace_templates(tmp_path, silent=True)

        result = ContextBuilder(tmp_path)._load_bootstrap_files()

        assert "## AGENTS.md" not in result
        assert "## USER.md" not in result
        assert "## SOUL.md" not in result

    def test_customized_user_template_is_loaded(self, tmp_path):
        from nanobot.utils.helpers import sync_workspace_templates

        sync_workspace_templates(tmp_path, silent=True)
        (tmp_path / "USER.md").write_text("User prefers Chinese.", encoding="utf-8")

        result = ContextBuilder(tmp_path)._load_bootstrap_files()

        assert "## USER.md" in result
        assert "User prefers Chinese." in result


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

        assert "Use the narrowest structured tool" in content
        assert "Do not use `exec` as a universal workaround" in content
        assert "## File and Coding Workflows" in content
        assert "`grep` defaults to file paths" in content
        assert "apply_patch" in content
        assert "acceptance criteria into concrete checks" in content
        assert "visual evidence reaches the model" in content
        assert "clear user request as authorization" in content
        assert "Never invent missing records or measurements" in content
        assert "## Web and External Information" in content
        assert "## Messaging and Media" in content
        assert "## Scheduling and Background Work" in content
        assert "independent lookup" in content

    def test_tool_contract_is_injected_without_workspace_file(self, tmp_path):
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt()

        assert "# Tool Usage Notes" in prompt
        assert "Use the narrowest structured tool" in prompt
        assert "Do not use `exec` as a universal workaround" in prompt


# ---------------------------------------------------------------------------
# build_user_content
# ---------------------------------------------------------------------------


class TestBuildUserContent:
    def test_no_media_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_user_content("hello", None)
        assert result == "hello"

    def test_empty_media_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_user_content("hello", [])
        assert result == "hello"

    def test_nonexistent_media_file_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_user_content("hello", ["/nonexistent/image.png"])
        assert result == "hello"

    def test_non_image_file_returns_string(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("not an image", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder.build_user_content("hello", [str(txt)])
        assert result == "hello"

    def test_valid_image_returns_list(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        result = builder.build_user_content("hello", [str(png)])
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
        result = builder.build_user_content("hello", [str(png)])
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

    def test_build_with_metrics_reports_named_sections(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Custom rules.", encoding="utf-8")
        builder = _builder(tmp_path)

        result = builder.build_system_prompt_with_metrics(session_summary="Archived state")

        assert isinstance(result, SystemPromptBuild)
        assert result.prompt == builder.build_system_prompt(session_summary="Archived state")
        assert result.section_tokens["identity"] > 0
        assert result.section_tokens["bootstrap"] > 0
        assert result.section_tokens["tool_contract"] > 0
        assert "session_summary" not in result.section_tokens
        assert result.section_tokens["total"] > 0

    def test_workspace_section_does_not_guess_memory_or_skill_paths(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt(session_key="cli:topic")

        assert "## Memory Paths" in result
        assert "Global memory:" in result
        assert "Topic memory:" in result
        assert "memory/history.jsonl" in result
        assert "memory/topics/cli_topic/history.jsonl" in result
        assert "HISTORY.md" not in result

    def test_default_identity_uses_relative_agent_paths(self, tmp_path):
        result = ContextBuilder(tmp_path)._get_identity()

        assert str(tmp_path.resolve()) not in result.split("## Memory Paths", 1)[0]
        assert "Agent profile: SOUL.md and USER.md" in result
        assert "History log: memory/history.jsonl" in result
        assert "Custom skills: skills/{skill-name}/SKILL.md" in result

    def test_selected_project_identity_keeps_agent_data_in_agent_workspace(self, tmp_path):
        agent_home = tmp_path / "agent-home"
        project = tmp_path / "project"
        agent_home.mkdir()
        project.mkdir()

        result = ContextBuilder(agent_home)._get_identity(workspace=project)

        assert str(project.resolve()) not in result
        assert f"agent workspace is at: {agent_home.resolve()}" in result
        assert f"{agent_home.resolve()}/SOUL.md" in result
        assert f"{project.resolve()}/SOUL.md" not in result

    def test_includes_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be helpful and concise.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "Be helpful and concise." in result

    @pytest.mark.parametrize("filename", ["AGENTS.md", "SOUL.md", "USER.md"])
    def test_skips_default_bootstrap_files(self, tmp_path, filename):
        template = load_bundled_template(filename)
        assert template is not None
        (tmp_path / filename).write_text(template, encoding="utf-8")

        result = _builder(tmp_path).build_system_prompt()

        assert f"## {filename}" not in result

    @pytest.mark.parametrize("filename", ["AGENTS.md", "SOUL.md"])
    def test_skips_locally_observed_legacy_default_bootstrap_files(self, tmp_path, filename, monkeypatch):
        content = f"legacy default {filename}\nsecond line"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        monkeypatch.setitem(ContextBuilder._LEGACY_DEFAULT_BOOTSTRAP_HASHES, filename, {digest})
        (tmp_path / filename).write_text(content.replace("\n", "\r\n"), encoding="utf-8", newline="")

        result = _builder(tmp_path).build_system_prompt()

        assert f"## {filename}" not in result

    @pytest.mark.parametrize("filename", ["AGENTS.md", "SOUL.md", "USER.md"])
    def test_keeps_custom_bootstrap_files(self, tmp_path, filename):
        marker = f"custom {filename} content"
        (tmp_path / filename).write_text(marker, encoding="utf-8")

        result = _builder(tmp_path).build_system_prompt()

        assert f"## {filename}" in result
        assert marker in result

    def test_session_summary_stays_out_of_stable_system_prompt(self, tmp_path):
        builder = _builder(tmp_path)
        summary = {
            "text": "Previous chat about Python.",
            "last_active": "2026-08-19T10:00:00",
        }
        result = builder.build_system_prompt(session_summary=summary)
        assert "Previous chat about Python." in result
        assert "[Archived Context Summary]" in result

    def test_nothing_summary_builds_the_prompt_without_archived_context(self, tmp_path):
        builder = _builder(tmp_path)
        summary = {"text": "(nothing)", "last_active": "2026-08-19T10:00:00"}
        assert builder.build_system_prompt(session_summary=summary) == builder.build_system_prompt()

    def test_sections_separated_by_separator(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Rules.", encoding="utf-8")
        builder = _builder(tmp_path)
        summary = {"text": "Summary.", "last_active": "2026-08-19T10:00:00"}
        result = builder.build_system_prompt(session_summary=summary)
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
    def test_session_summary_is_not_repeated_as_user_reminder(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello", session_summary="Archived summary")

        assert "Archived summary" not in messages[0]["content"]
        assert messages[-1]["content"].count("Archived summary") == 1
        assert "legacy_continuation:" in messages[-1]["content"]

    def test_pending_summary_is_injected_as_user_reminder(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello", pending_summary="Fresh summary")

        assert "Fresh summary" in messages[-1]["content"]
        assert "Fresh summary" not in messages[0]["content"]

    def test_optional_arguments_are_keyword_only(self, tmp_path):
        builder = _builder(tmp_path)

        with pytest.raises(TypeError):
            builder.build_system_prompt(["legacy-skill"])
        with pytest.raises(TypeError):
            builder.build_messages([], "hello", ["legacy-skill"])

    def test_basic_empty_history(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "hello" in str(messages[1]["content"])

    def test_public_builder_preserves_assistant_role_compatibility(self, tmp_path):
        from nanobot.agent import ContextBuilder as PublicContextBuilder

        builder = PublicContextBuilder(tmp_path)
        messages = builder.build_messages(
            history=[{"role": "assistant", "content": "previous result"}],
            current_message="subagent result",
            current_role="assistant",
            runtime_context_blocks=[
                RuntimeContextBlock(source="test", content="user-only runtime context"),
            ],
        )


        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "previous result\n\nsubagent result"
        assert "user-only runtime context" not in messages[-1]["content"]
        assert "_meta" not in messages[-1]

    def test_compatibility_builder_merges_system_role_without_history(self, tmp_path):
        builder = _builder(tmp_path)

        messages = builder.build_messages([], "system event", current_role="system")

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert str(messages[0]["content"]).endswith("system event")

    def test_explicit_skill_reference_loads_full_instructions_for_this_turn(self, tmp_path):
        skill_dir = tmp_path / "skills" / "review"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: review\n"
            "description: Review changes.\n"
            "---\n\n"
            "# Review workflow\n\nFollow the unique review checklist.",
            encoding="utf-8",
        )
        builder = _builder(tmp_path)

        messages = builder.build_messages([], "Please $review this patch and use $review carefully.")
        plain_messages = builder.build_messages([], "Please review this patch carefully.")

        system_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]
        assert system_prompt == plain_messages[0]["content"]
        assert "Follow the unique review checklist." not in system_prompt
        assert "Please $review this patch" in user_prompt
        assert "[Active Skills — instructions for this user turn]" in user_prompt
        assert "### Skill: review" in user_prompt
        assert "Follow the unique review checklist." in user_prompt
        assert user_prompt.count("### Skill: review") == 1
        assert messages[-1]["_meta"]["runtime_context"]["sources"] == [
            "explicit_skills"
        ]

    def test_unknown_skill_reference_does_not_change_active_skills(self, tmp_path):
        messages = _builder(tmp_path).build_messages([], "Keep the shell literal $HOME.")

        assert "# Active Skills" not in messages[0]["content"]

    def test_runtime_context_is_not_injected_by_default(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello", channel="cli")
        user_msg = str(messages[-1]["content"])
        assert user_msg == "hello"

    def test_explicit_runtime_context_blocks_are_appended(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages(
            [],
            "please use @zoom tonight",
            runtime_context_blocks=[
                RuntimeContextBlock(
                    source="cli_apps",
                    content="CLI App Attachment: @zoom (installed; tool=run_cli_app).",
                ),
            ],
        )
        user_msg = str(messages[-1]["content"])

        assert "CLI App Attachment: @zoom" in user_msg
        assert "tool=run_cli_app" in user_msg
        assert user_msg.index("please use @zoom tonight") < user_msg.index(
            "CLI App Attachment: @zoom"
        )
        assert messages[-1]["_meta"]["runtime_context"]["sources"] == ["cli_apps"]

    def test_consecutive_same_role_merged(self, tmp_path):
        builder = _builder(tmp_path)
        history = [{"role": "user", "content": "previous user message"}]
        messages = builder.build_messages(history, "new message")
        assert len(messages) == 2  # system + merged user
        assert "previous user message" in str(messages[1]["content"])
        assert "new message" in str(messages[1]["content"])

    def test_structured_transcript_preserves_fresh_turn_boundary(self, tmp_path):
        builder = _builder(tmp_path)
        transcript = TranscriptInput(
            history=[{"role": "user", "content": "previous user message"}],
            current_message="new message",
        )

        messages = builder.build_transcript(transcript)

        assert [message["role"] for message in messages] == ["system", "user", "user"]
        assert messages[-2]["content"] == "previous user message"
        assert messages[-1]["content"] == "new message"
        assert transcript.message_count == 3

    def test_current_message_can_be_built_without_history_merge(self, tmp_path):
        builder = _builder(tmp_path)
        current = builder.build_current_message(
            "new message",
            runtime_context_blocks=[
                RuntimeContextBlock(source="test", content="fresh context"),
            ],
        )

        assert current["role"] == "user"
        assert "new message" in current["content"]
        assert "fresh context" in current["content"]
        assert current["_meta"]["runtime_context"]["sources"] == ["test"]

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


def test_skill_match_reminder_uses_request_workspace_claude_skills(tmp_path):
    default_workspace = tmp_path / "default"
    data_dir = tmp_path / "data"
    request_workspace = tmp_path / "request_project"
    default_workspace.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    request_workspace.mkdir(parents=True)
    _write_skill(
        request_workspace / ".claude" / "skills",
        "request_skill",
        "Inspect request project widgets",
    )

    builder = ContextBuilder(workspace=default_workspace, data_dir=data_dir)
    messages = builder.build_messages(
        history=[],
        current_message="inspect request project widgets",
        workspace=request_workspace,
    )

    assert "request_skill" in str(messages[-1]["content"])


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


def test_build_messages_injects_resume_guard_and_unresolved_todos(tmp_path) -> None:
    builder = _builder(tmp_path)
    messages = builder.build_messages(
        [],
        "继续中断任务",
        channel="cli",
        chat_id="resume",
        session_metadata={
            GOAL_STATE_KEY: {
                "status": "completed",
                "objective": "Previous completed objective",
                "recap": "Completed with one documented follow-up.",
            }
        },
        todos=[{"content": "Finish documented follow-up", "status": "pending"}],
        resume_request=True,
    )

    user_msg = str(messages[-1]["content"])
    system_msg = str(messages[0]["content"])
    assert "resume.last_goal.status: completed" in user_msg
    assert "Finish documented follow-up" in user_msg
    assert "# Active Todos" in system_msg


def test_active_goal_is_rendered_in_current_user_context(tmp_path):
    builder = _builder(tmp_path)
    messages = builder.build_messages(
        [], "continue", session_metadata={GOAL_STATE_KEY: {
            "status": "active", "objective": "Finish docs migration.",
            "started_at": "2026-09-11T00:00:00",
        }},
    )
    assert "Finish docs migration." in str(messages[-1]["content"])
    assert "Goal (active):" not in str(messages[-1]["content"])
