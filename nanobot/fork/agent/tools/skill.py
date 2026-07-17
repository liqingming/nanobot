"""LoadSkillTool — load a nanobot skill by name.

Skills are markdown documents (``<skills-root>/<name>/SKILL.md``) that teach
the agent how to use specific tools or perform certain tasks. Historically
they were loaded by the LLM calling ``read_file(<skill path>)``. That works
but the trace was indistinguishable from a normal file read, and the loaded
content sat in conversation history without any "this came from a skill"
marker — making it easier for the LLM to forget which steps were skill
instructions.

LoadSkillTool fixes both problems:

  * Trace shows ``load-skill("name")`` and a skill-flavoured summary
    (``loaded: 7 sections``).
  * Tool output wraps the SKILL.md body in ``<skill name="...">…</skill>``
    so the LLM can see clearly that what follows is skill guidance.

The system prompt advertises only the skill names (see
``ContextBuilder.build_system_prompt``); the LLM doesn't need to know
filesystem paths.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nanobot.agent.skills import SkillsLoader, project_skill_roots
from nanobot.agent.tools.base import Tool
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.skills import SkillsLoader


class LoadSkillTool(Tool):
    """Load a nanobot skill by name and return its instructions."""

    def __init__(self, loader: "SkillsLoader") -> None:
        self._loader = loader

    def _effective_loader(self) -> "SkillsLoader":
        scope = current_workspace_scope()
        if scope is None or scope.project_path == self._loader.workspace:
            return self._loader
        return SkillsLoader(
            self._loader.workspace,
            builtin_skills_dir=self._loader.builtin_skills,
            disabled_skills=self._loader.disabled_skills,
            extra_skill_roots=project_skill_roots(scope.project_path),
        )

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        return (
            "Load a nanobot skill by name. Skills are pre-written task "
            "playbooks (e.g. dataset_explore, web_research) listed in the "
            "system prompt's Skills section. Call this BEFORE starting a "
            "task that matches a skill's description — the returned "
            "<skill> block contains step-by-step instructions to follow. "
            "Argument is the skill's <name> (NOT a file path)."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Skill name as listed in the <skills> section of "
                        "the system prompt (e.g. 'dataset_explore')."
                    ),
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, **_kwargs: Any) -> str:
        if not isinstance(name, str) or not name.strip():
            return "Error: 'name' must be a non-empty string"
        name = name.strip()
        loader = self._effective_loader()
        content = await asyncio.to_thread(loader.load_skill, name)
        if content is None:
            # Help the LLM recover quickly when it picked a wrong name —
            # list what IS available so the next attempt can self-correct.
            available = await asyncio.to_thread(
                lambda: [
                    s["name"]
                    for s in loader.list_skills(filter_unavailable=False)
                ]
            )
            return (
                f"Error: skill '{name}' not found. "
                f"Available skills: {', '.join(available) if available else '(none)'}"
            )
        return f'<skill name="{name}">\n{content}\n</skill>'

    def summarize_result(self, args: dict[str, Any], result: Any) -> str:
        if not isinstance(result, str):
            return ""
        if result.startswith("Error"):
            from nanobot.agent.tools.summaries import extract_error_summary
            return extract_error_summary(result)
        # Strip the <skill> wrapper for accurate counting; cheap string scan
        # rather than full parse since the wrapper is fixed.
        body = result
        if body.startswith("<skill ") and body.endswith("</skill>"):
            first_nl = body.find("\n")
            if first_nl > 0:
                body = body[first_nl + 1 : -len("</skill>")].rstrip()
        if not body:
            return "loaded: empty"
        version = _extract_frontmatter_version(body)
        body_after_fm = _strip_frontmatter(body)
        lines = body_after_fm.splitlines()
        section_count = sum(1 for line in lines if line.startswith("## "))
        if section_count == 0:
            base = f"{len(lines)} line{'s' if len(lines) != 1 else ''}"
        else:
            base = f"{section_count} section{'s' if section_count != 1 else ''}"
        if version:
            return f"loaded: v{version}, {base}"
        return f"loaded: {base}"


def _extract_frontmatter_version(text: str) -> str | None:
    """Pull ``version: X`` from a leading ``---\\n...\\n---`` YAML block.

    Returns None if there's no frontmatter, no version field, or any
    extraction error. Doesn't import yaml — a regex is enough for the
    one field we need and keeps the dependency surface small.
    """
    if not text.startswith("---"):
        return None
    # Find the closing --- on its own line
    end = text.find("\n---", 3)
    if end < 0:
        return None
    fm = text[3:end]
    import re
    m = re.search(r"^\s*version\s*:\s*(.+?)\s*$", fm, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---\\n...\\n---`` block so downstream line counts
    don't double-count the metadata header."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    after = end + len("\n---")
    if after < len(text) and text[after] == "\n":
        after += 1
    return text[after:]


# ── self-registration ────────────────────────────────────────────────────

from nanobot.agent.tools.registry import register_fork_tool  # noqa: E402

register_fork_tool(lambda loop: LoadSkillTool(loader=loop.context.skills))
