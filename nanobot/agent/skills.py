"""Skills loader for agent capabilities.

Fork additions:
  * ``build_skills_summary`` omits ``<location>`` entries — fork's
    ``load_skill(name=...)`` tool loads by name, exposing filesystem
    paths just wastes prompt tokens and tempts read_file misuse.
  * ``get_skill_metadata`` caches parsed frontmatter by mtime (perf) — it is
    called per-skill on every turn via the skill-match hint and
    build_skills_summary, and was previously a fresh read + YAML parse each time.
"""

import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from nanobot.runtime_context import RuntimeContextBlock

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)
_SKILL_NAME = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SKILL_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]+)")


def parse_skill_metadata(content: str) -> dict[str, object] | None:
    """Parse a skill document's YAML frontmatter."""
    if not (match := _STRIP_SKILL_FRONTMATTER.match(content)):
        return None
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in cast(dict[object, object], parsed).items()}


def valid_skill_metadata(metadata: dict[str, object], name: str) -> bool:
    """Return whether metadata satisfies the Agent Skills identity contract."""
    description = metadata.get("description")
    return (
        metadata.get("name") == name
        and len(name) <= 64
        and _SKILL_NAME.fullmatch(name) is not None
        and isinstance(description, str)
        and 1 <= len(description.strip()) <= 1024
    )


def configured_skill_roots(paths: Sequence[str | Path]) -> list[tuple[str, Path]]:
    """Convert configured Skill root paths into stable, ordered loader entries."""
    return [(f"configured-{index}", Path(path)) for index, path in enumerate(paths, start=1)]


def project_skill_roots(workspace: Path) -> list[tuple[str, Path]]:
    claude_dir = workspace / ".claude"
    if claude_dir.is_dir():
        return [("claude", claude_dir / "skills")]
    return []


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        disabled_skills: set[str] | None = None,
        extra_skill_roots: Sequence[tuple[str, Path]] | None = None,
    ):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()
        self.extra_skill_roots = list(extra_skill_roots or [])
        # Fork(perf): cache parsed frontmatter metadata keyed by skill name,
        # invalidated by SKILL.md mtime. Without this, get_skill_metadata is a
        # fresh read + yaml.safe_load on every call, and build_skills_summary /
        # list_skills / the per-turn skill-match hint invoke it per skill.
        self._meta_cache: dict[str, tuple[float, dict | None]] = {}

    def _skill_aliases(self) -> dict[str, str]:
        """Return compatibility aliases owned by installed CLI Apps."""
        from nanobot.apps.cli import CliAppManager

        try:
            return CliAppManager(workspace=self.workspace).installed_skill_aliases()
        except OSError:
            return {}

    def _skill_entries_from_dir(self, base: Path, source: str, *, skip_names: set[str] | None = None) -> list[dict[str, str]]:
        if not base.is_dir():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in sorted(base.iterdir(), key=lambda path: path.name):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        from nanobot.agent.plugins import enabled_agent_plugin_skills

        skills: list[dict[str, str]] = []
        seen_names: set[str] = set()
        for source, root in self.extra_skill_roots:
            entries = self._skill_entries_from_dir(root, source, skip_names=seen_names)
            skills.extend(entries)
            seen_names.update(entry["name"] for entry in entries)

        entries = self._skill_entries_from_dir(self.workspace_skills, "workspace", skip_names=seen_names)
        skills.extend(entries)
        seen_names.update(entry["name"] for entry in entries)
        plugin_skills = enabled_agent_plugin_skills(self.workspace)
        for name, path in plugin_skills:
            if name in seen_names:
                continue
            skills.append(
                {
                    "name": name,
                    "path": str(path),
                    "source": "plugin",
                }
            )
            seen_names.add(name)
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=seen_names)
            )

        if self.disabled_skills:
            disabled = set(self.disabled_skills)
            for legacy, canonical in self._skill_aliases().items():
                if legacy in disabled or canonical in disabled:
                    disabled.update((legacy, canonical))
            skills = [s for s in skills if s["name"] not in disabled]

        if filter_unavailable:
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
        return skills

    def format_listing(self) -> str:
        entries = [entry for entry in self.list_skills(filter_unavailable=False)
                   if entry["name"] != "long-goal"]
        if not entries:
            return "当前没有可用技能。"

        available_names = {
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
        }
        source_labels = {
            "claude": ".claude",
            "workspace": "workspace",
            "builtin": "builtin",
        }
        lines = [f"Skills ({len(entries)}):"]
        for entry in sorted(entries, key=lambda item: (item.get("source", ""), item.get("name", ""))):
            name = entry.get("name", "")
            source = source_labels.get(entry.get("source", ""), entry.get("source", "unknown"))
            description = self.get_skill_description(name) or "(no description)"
            status = "" if name in available_names else " [unavailable]"
            lines.append(f"- {name} ({source}){status}: {description}")
        return "\n".join(lines)

    def _resolve_skill_path(self, name: str) -> Path | None:
        skills = self.list_skills(filter_unavailable=False)
        available = {entry["name"] for entry in skills}
        resolved = name if name in available else self._skill_aliases().get(name, name)
        entry = next((entry for entry in skills if entry["name"] == resolved), None)
        return Path(entry["path"]) if entry else None

    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        skills = self.list_skills(filter_unavailable=False)
        available = {skill["name"] for skill in skills}
        resolved = name if name in available else self._skill_aliases().get(name, name)
        entry = next((skill for skill in skills if skill["name"] == resolved), None)
        return Path(entry["path"]).read_text(encoding="utf-8") if entry else None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def get_explicitly_invoked_skills(self, text: str) -> list[str]:
        """Resolve ``$skill-name`` references to enabled, available skills."""
        if not text:
            return []
        available = {
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
        }
        aliases = self._skill_aliases()
        invoked: list[str] = []
        for match in _SKILL_REFERENCE.finditer(text):
            requested = match.group(1)
            name = requested if requested in available else aliases.get(requested, requested)
            if name in available and name not in invoked:
                invoked.append(name)
        return invoked

    def build_explicit_skill_runtime_context(
        self,
        text: str,
    ) -> RuntimeContextBlock | None:
        """Load non-always skills explicitly invoked by the current message."""
        skill_names = self.get_explicitly_invoked_skills(text)
        if not skill_names:
            return None
        always_active = set(self.get_always_skills())
        skill_names = [name for name in skill_names if name not in always_active]
        content = self.load_skills_for_context(skill_names)
        if not content:
            return None
        return RuntimeContextBlock(
            source="explicit_skills",
            content=(
                "[Active Skills — instructions for this user turn]\n"
                f"{content}\n"
                "[/Active Skills]"
            ),
        )

    def build_skills_summary(
        self,
        exclude: set[str] | None = None,
        *,
        workspace: Path | None = None,
    ) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can load the full
        skill content by name when needed.

        Args:
            exclude: Set of skill names to omit from the summary.
            workspace: Effective project workspace used to choose safe display paths.

        Returns:
            Markdown-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
        exclude = {"long-goal", *(exclude or set())}
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        emitted = [skill for skill in all_skills if not exclude or skill["name"] not in exclude]
        if not emitted:
            return ""

        lines = ["<skills>"]
        for s in emitted:
            skill_name = s["name"]
            name = escape_xml(skill_name)
            desc = escape_xml(self.get_skill_description(skill_name))
            skill_meta = self._get_skill_meta(skill_name)
            available = self._check_requirements(skill_meta)

            # <location> is intentionally omitted — LLM loads by name via
            # load_skill(name=...), so exposing filesystem paths just
            # wastes prompt tokens (and tempts read_file misuse).
            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")

            # Show missing requirements for unavailable skills
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    @staticmethod
    def _requirement_lists(skill_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (bins, env) lists from skill metadata, tolerating null/wrong shapes."""
        requires = cast(dict[str, Any], skill_meta.get("requires") or {})
        if not isinstance(skill_meta.get("requires") or {}, dict):
            return [], []
        bins_raw: object = requires.get("bins") or []
        env_raw: object = requires.get("env") or []
        bins = [value for value in cast(list[object], bins_raw) if isinstance(value, str) and value.strip()] if isinstance(bins_raw, list) else []
        env = [value for value in cast(list[object], env_raw) if isinstance(value, str) and value.strip()] if isinstance(env_raw, list) else []
        return bins, env

    def _get_missing_requirements(self, skill_meta: dict[str, Any]) -> str:
        """Get a description of missing requirements."""
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        return ", ".join(
            [f"CLI: {command_name}" for command_name in required_bins if not shutil.which(command_name)]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """Return whether a skill can run and why not when it cannot."""
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        """Return explicit command/env requirements and currently missing entries."""
        bins, env = self._requirement_lists(self._get_skill_meta(name))
        return {
            "bins": bins,
            "env": env,
            "missing_bins": [value for value in bins if not shutil.which(value)],
            "missing_env": [value for value in env if not os.environ.get(value)],
        }

    def get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        description = meta.get("description") if meta else None
        if isinstance(description, str) and description:
            return description
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: object) -> dict[str, Any]:
        """Extract nanobot/openclaw metadata from a frontmatter field.

        ``raw`` may be a dict (already parsed by yaml.safe_load) or a JSON str.
        """
        if isinstance(raw, dict):
            data = cast(dict[str, Any], raw)
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        data_object = cast(dict[str, Any], data)
        payload = data_object.get("nanobot", data_object.get("openclaw", {}))
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _get_skill_meta(self, name: str) -> dict[str, Any]:
        """Get nanobot metadata for a skill (cached in frontmatter)."""
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")
            )
        ]

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        """
        Get metadata from a skill's frontmatter.

        Fork(perf): result is cached per skill name and invalidated when the
        SKILL.md mtime changes — callers (build_skills_summary, list_skills,
        skill-match hint) hit this per skill on every turn. The cached dict is
        treated as read-only by all callers; do not mutate the return value.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        path = self._resolve_skill_path(name)
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._meta_cache.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        meta = self._parse_skill_metadata(path)
        self._meta_cache[name] = (mtime, meta)
        return meta

    def _parse_skill_metadata(self, path: Path) -> dict | None:
        """Parse YAML frontmatter from a SKILL.md path (no caching)."""
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if not match:
            return None
        try:
            parsed = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(parsed, dict):
            return None
        # yaml.safe_load returns native types (int, bool, list, etc.);
        # keep values as-is so downstream consumers get correct types.
        metadata: dict[str, object] = {}
        for key, value in parsed.items():
            metadata[str(key)] = value
        return metadata
