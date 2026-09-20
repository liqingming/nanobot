"""Context builder for assembling agent prompts."""

import base64
import hashlib
import mimetypes
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from nanobot.agent.context_artifacts import render_active_context, select_memory_view
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import (
    SkillsLoader,
    configured_skill_roots,
    project_skill_roots,
)
from nanobot.agent.tools import image_generation as image_generation_tools
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools import sessions as session_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_SESSION_DISCARD,
    InboundMessage,
)
from nanobot.fork.agent.input_evidence import make_input_verifier
from nanobot.fork.agent.source_evidence import make_source_verifier
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    append_runtime_context,
)
from nanobot.security.workspace_access import WorkspaceScopeResolver
from nanobot.session.keys import last_channel_from_metadata
from nanobot.session.manager import Session
from nanobot.session.summary import SessionSummary
from nanobot.utils.helpers import (
    detect_image_mime,
    estimate_message_tokens,
    load_bundled_template,
    safe_filename,
    truncate_text_to_tokens,
)
from nanobot.utils.prompt_templates import render_template


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted kwargs for turn-attached capabilities."""
    return (
        cli_app_utils.session_extra(metadata)
        | mcp_tools.session_extra(metadata)
        | session_tools.session_extra(metadata)
    )


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    if msg.metadata.get(INBOUND_META_RUNTIME_CONTROL) == RUNTIME_CONTROL_SESSION_DISCARD:
        await state.discard_session(msg.session_key)
        return True
    return await image_generation_tools.handle_runtime_control(state, msg, tools)


@dataclass(frozen=True, slots=True)
class PersistedPromptContextResolver:
    """Restore prompt routing context when no inbound message is available."""

    workspace_scopes: WorkspaceScopeResolver
    unified_session: bool = False

    def __call__(self, session: Session) -> tuple[str | None, Path]:
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        if self.unified_session:
            route = last_channel_from_metadata(session.metadata)
            if route is not None:
                channel = route[0]
        scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=None,
            session_metadata=session.metadata,
        )
        return channel, scope.project_path


@dataclass(frozen=True, slots=True)
class TranscriptInput:
    """Raw turn inputs from which ``ContextBuilder`` assembles a transcript."""

    history: list[dict[str, Any]]
    current_message: str | None
    media: Sequence[str] | None = None
    current_role: str = "user"
    session_summary: SessionSummary | str | None = None
    runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None
    session_key: str | None = None
    session_metadata: Mapping[str, Any] | None = None
    todos: list[dict[str, Any]] | None = None
    learning_ctx: str | None = None
    resume_request: bool = False
    runtime_state: Any = None
    inbound_message: Any = None
    pending_summary: str | None = None
    include_memory_recent_history: bool = True
    unified_session: bool = False
    system_prompt_metrics: dict[str, int] | None = None

    @property
    def message_count(self) -> int:
        """Number of boundary-preserving messages in the assembled transcript."""
        return 1 + len(self.history) + (self.current_message is not None)


@dataclass(frozen=True, slots=True)
class SystemPromptBuild:
    prompt: str
    section_tokens: dict[str, int]


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    _SKIPPABLE_DEFAULTS = {"AGENTS.md", "USER.md"}
    _SKILL_MATCH_MIN_TOKEN_LEN = 4
    _SKILL_MATCH_MAX_SUGGESTIONS = 3
    _MAX_RECENT_HISTORY = 12
    _MAX_HISTORY_TOKENS = 1_500
    # Default bootstrap contents observed in this user's existing local data
    # directories before the bundled templates changed.  Keep these fingerprints
    # narrow: any other content remains user-authored and is injected normally.
    _LEGACY_DEFAULT_BOOTSTRAP_HASHES = {
        "AGENTS.md": {"15d53dd70ccb682673b87ef7d7c526d54db61e7a6b877253088370a7a0058cd4"},
        "SOUL.md": {"4c94411ddfd965484aa8ac9981561324b8035c026bb903d5d379adea41ec53e8"},
    }

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        skill_roots: list[str] | None = None,
        *,
        data_dir: Path | None = None,
        topic_memory_factory: Any = None,
    ):
        self.workspace = workspace
        self.data_dir = data_dir if data_dir is not None else workspace
        self.timezone = timezone
        self.memory = MemoryStore(self.data_dir)
        self.configured_skill_roots = configured_skill_roots(skill_roots or [])
        self.skills = SkillsLoader(
            self.data_dir,
            disabled_skills=set(disabled_skills) if disabled_skills else None,
            extra_skill_roots=[*self.configured_skill_roots, *project_skill_roots(self.workspace),
                               *([("workspace", self.workspace / "skills")] if self.workspace != self.data_dir else [])],
        )
        self._topic_memory_factory = topic_memory_factory

    def build_system_prompt(
        self,
        *,
        skill_names: list[str] | None = None,
        include_learning_rules: bool = False,
        session_key: str | None = None,
        todos: list[dict[str, Any]] | None = None,
        channel: str | None = None,
        session_summary: SessionSummary | str | None = None,
        workspace: Path | None = None,
        include_memory_recent_history: bool = True,
        unified_session: bool = False,
        include_memory: bool = True,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        return self.build_system_prompt_with_metrics(
            skill_names=skill_names,
            include_learning_rules=include_learning_rules,
            session_key=session_key,
            todos=todos,
            channel=channel,
            session_summary=session_summary,
            workspace=workspace,
            include_memory_recent_history=include_memory_recent_history,
            unified_session=unified_session,
            include_memory=include_memory,
        ).prompt

    def _get_identity(
        self,
        session_key: str | None = None,
        channel: str | None = None,
        workspace: Path | None = None,
    ) -> str:
        """Get the core identity section."""
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve())
        data_path = str(self.data_dir.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        identity = render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            agent_workspace_path=str(self.workspace.expanduser().resolve()),
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )
        if isinstance(session_key, str) and session_key:
            safe_key = safe_filename(session_key.replace(":", "_"))
            memory_lines = (
                f"- Global memory: {data_path}/memory/MEMORY.md (cross-topic facts, auto-injected)\n"
                f"- Global history: {data_path}/memory/history.jsonl (event log, search on demand)\n"
                f"- Topic memory: {data_path}/memory/topics/{safe_key}/MEMORY.md (current topic facts, auto-injected)\n"
                f"- Topic history: {data_path}/memory/topics/{safe_key}/history.jsonl (event log, search on demand)"
            )
        else:
            memory_lines = (
                f"- Long-term memory: {data_path}/memory/MEMORY.md (auto-injected facts)\n"
                f"- History log: {data_path}/memory/history.jsonl (search on demand)"
            )
        return f"{identity}\n\n## Memory Paths\n{memory_lines}\n- Custom skills: {data_path}/skills/{{skill-name}}/SKILL.md"

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            if not left:
                return right
            if not right:
                return left
            return f"{left}\n\n{right}"

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    cast(dict[str, Any], item)
                    if isinstance(item, dict)
                    else {"type": "text", "text": str(item)}
                    for item in cast(list[Any], value)
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self, workspace: Path | None = None) -> str:
        """Load bootstrap files from Claude globals, workspace, and nanobot data."""
        parts: list[str] = []
        seen_paths: set[Path] = set()
        root = workspace or self.workspace

        candidates: list[tuple[str, Path]] = [
            ("~/.claude/CLAUDE.md", Path.home() / ".claude" / "CLAUDE.md"),
            ("CLAUDE.md", root / "CLAUDE.md"),
            (".claude/CLAUDE.md", root / ".claude" / "CLAUDE.md"),
        ]
        for filename in self.BOOTSTRAP_FILES:
            source = root if filename == "AGENTS.md" else self.data_dir
            candidates.append((filename, source / filename))

        for label, file_path in candidates:
            if not file_path.exists():
                continue
            try:
                resolved = file_path.resolve()
            except OSError:
                resolved = file_path
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            content = file_path.read_text(encoding="utf-8")
            if label == "SOUL.md" and self._is_template_content(content, "legacy/SOUL.md"):
                content = load_bundled_template("SOUL.md") or content
            if not content.strip():
                continue
            if label in self.BOOTSTRAP_FILES and self._is_template_content(content, label):
                continue
            parts.append(f"## {label}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @classmethod
    def _is_template_content(cls, content: str, template_path: str) -> bool:
        """Check whether content is a current or locally observed default template."""
        normalized = content.strip()
        tpl = load_bundled_template(template_path)
        if tpl is not None and normalized == tpl.strip():
            return True
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return digest in cls._LEGACY_DEFAULT_BOOTSTRAP_HASHES.get(template_path, set())

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str | None,
        *,
        media: list[str] | None = None,
        channel: str | None = None,
        current_role: str = "user",
        session_summary: SessionSummary | str | None = None,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
        skill_names: list[str] | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        learning_ctx: str | None = None,
        session_key: str | None = None,
        todos: list[dict[str, Any]] | None = None,
        pending_summary: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        resume_request: bool = False,
        current_runtime_lines: Sequence[str] | None = None,
        runtime_state: Any = None,
        inbound_message: Any = None,
        skip_runtime_lines: bool = False,
        include_memory_recent_history: bool = True,
        unified_session: bool = False,
        system_prompt_metrics: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Adapt legacy turn inputs to the boundary-preserving transcript builder."""
        del skill_names
        blocks = list(runtime_context_blocks or ())
        if not skip_runtime_lines:
            lines = list(current_runtime_lines or ())
            if chat_id:
                lines.append(f"Chat ID: {chat_id}")
            if sender_id:
                lines.append(f"Sender ID: {sender_id}")
            if lines:
                blocks.append(RuntimeContextBlock(source="fork_request", content="\n".join(lines)))
        messages = self.build_transcript(
            TranscriptInput(
                history=history,
                current_message=current_message,
                media=media,
                current_role=current_role,
                session_summary=session_summary,
                runtime_context_blocks=blocks,
                session_key=session_key,
                session_metadata=session_metadata,
                todos=todos,
                learning_ctx=learning_ctx,
                pending_summary=pending_summary,
                resume_request=resume_request,
                runtime_state=runtime_state,
                inbound_message=inbound_message,
                include_memory_recent_history=include_memory_recent_history,
                unified_session=unified_session,
                system_prompt_metrics=system_prompt_metrics,
            ),
            channel=channel,
            workspace=workspace,
            include_memory=include_memory,
        )
        if current_message is None:
            return messages
        current = messages[-1]
        if len(messages) < 2 or messages[-2].get("role") != current.get("role"):
            return messages

        merged = dict(messages[-2])
        merged["content"] = self._merge_message_content(
            merged.get("content"),
            current.get("content"),
        )
        current_meta = current.get("_meta")
        if current.get("role") == "user" and isinstance(current_meta, dict):
            internal_meta = dict(merged.get("_meta") or {})
            internal_meta.update(cast(dict[str, Any], current_meta))
            merged["_meta"] = internal_meta
        return [*messages[:-2], merged]

    def build_transcript(
        self,
        transcript: TranscriptInput,
        *,
        channel: str | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
    ) -> list[dict[str, Any]]:
        """Build a model transcript while preserving the fresh-turn boundary."""
        root = workspace or self.workspace
        system_build = self.build_system_prompt_with_metrics(
            channel=channel,
            session_key=transcript.session_key,
            todos=transcript.todos,
            include_learning_rules=transcript.learning_ctx is not None,
            session_summary=transcript.session_summary,
            workspace=root,
            include_memory=include_memory,
            include_memory_recent_history=transcript.include_memory_recent_history,
            unified_session=transcript.unified_session,
        )
        if transcript.system_prompt_metrics is not None:
            transcript.system_prompt_metrics.clear()
            transcript.system_prompt_metrics.update(system_build.section_tokens)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_build.prompt},
            *transcript.history,
        ]
        if transcript.current_message is None:
            return messages

        registry = getattr(transcript.runtime_state, "tools", None)
        active = render_active_context(
            transcript.session_metadata, workspace=root,
            legacy_summary=transcript.session_summary if isinstance(transcript.session_summary, str) else None,
            todos=transcript.todos, resume_request=transcript.resume_request,
            source_verifier=make_source_verifier(
                root, registry.get("read_file") if isinstance(registry, ToolRegistry) else None,
                getattr(transcript.inbound_message, "metadata", None),
            ),
            input_verifier=make_input_verifier(self.data_dir, transcript.session_key),
        )
        reminder = ""
        if transcript.pending_summary and transcript.pending_summary.strip():
            reminder = (
                "<system-reminder>\nUpdated long-term memory (consolidated from earlier "
                "in this conversation; not yet promoted to MEMORY.md):\n\n"
                f"{transcript.pending_summary.strip()}\n</system-reminder>"
            )
        loader = self._skills_for_workspace(root)
        skill_hint = self._build_skill_match_reminder(
            transcript.current_message, loader,
        ) if (
            transcript.current_role == "user"
            and getattr(transcript.inbound_message, "sender_id", None) != "subagent"
            and not loader.get_explicitly_invoked_skills(transcript.current_message)
        ) else ""
        prefix = "\n\n".join(
            part for part in (active, transcript.learning_ctx, reminder, skill_hint) if part
        ) if transcript.current_role == "user" else ""
        blocks = list(transcript.runtime_context_blocks or ())
        if prefix:
            blocks.append(RuntimeContextBlock(source="fork_context", content=prefix))
        current = self.build_current_message(
            transcript.current_message,
            media=list(transcript.media) if transcript.media else None,
            current_role=transcript.current_role,
            runtime_context_blocks=blocks,
            skills=self._skills_for_workspace(root),
        )
        messages.append(current)
        return messages

    def build_current_message(
        self,
        current_message: str,
        *,
        media: list[str] | None = None,
        current_role: str = "user",
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        skills: SkillsLoader | None = None,
    ) -> dict[str, Any]:
        """Build only the fresh turn message without merging it into history."""
        content = self.build_user_content(current_message, image_paths=media)
        blocks: list[RuntimeContextBlock] = []
        if current_role == "user":
            blocks.extend(runtime_context_blocks or ())
            skill_context = (skills or self.skills).build_explicit_skill_runtime_context(current_message)
            if skill_context is not None and skill_context not in blocks:
                blocks.append(skill_context)
        merged, runtime_context_meta = append_runtime_context(content, blocks)
        current: dict[str, Any] = {"role": current_role, "content": merged}
        if current_role == "user" and runtime_context_meta is not None:
            current["_meta"] = {
                RUNTIME_CONTEXT_MESSAGE_META: runtime_context_meta,
            }
        return current

    def build_user_content(
        self,
        text: str,
        image_paths: list[str] | None,
    ) -> str | list[dict[str, Any]]:
        """Build user message content from prefiltered image paths."""
        if not image_paths:
            return text

        image_blocks: list[dict[str, Any]] = []
        for path in image_paths:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            # Re-detect from the bytes used for the request: the file may have
            # changed since attachment routing, and the data URL needs its MIME.
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not image_blocks:
            return text
        return image_blocks + [{"type": "text", "text": text}]

    def _get_topic_store(self, session_key: str | None) -> MemoryStore | None:
        if self._topic_memory_factory is None or not isinstance(session_key, str) or not session_key:
            return None
        return self._topic_memory_factory.get(session_key)


    def _skills_for_workspace(self, workspace: Path) -> SkillsLoader:
        if workspace == self.workspace:
            return self.skills
        return SkillsLoader(
            self.data_dir,
            disabled_skills=self.skills.disabled_skills,
            extra_skill_roots=[*self.configured_skill_roots, *project_skill_roots(workspace),
                               *([("workspace", workspace / "skills")] if workspace != self.data_dir else [])],
        )


    def build_system_prompt_with_metrics(
        self,
        skill_names: list[str] | None = None,
        include_learning_rules: bool = False,
        session_key: str | None = None,
        todos: list[dict[str, Any]] | None = None,
        channel: str | None = None,
        session_summary: SessionSummary | str | None = None,
        workspace: Path | None = None,
        include_memory_recent_history: bool = True,
        unified_session: bool = False,
        include_memory: bool = True,
    ) -> SystemPromptBuild:
        """Build the system prompt and expose comparable per-section estimates."""
        del skill_names  # Reserved for API compatibility; skills are loaded progressively.
        root = workspace or self.workspace
        sections: list[tuple[str, str]] = []

        def add(name: str, content: str | None) -> None:
            if content and content.strip():
                sections.append((name, content))

        add("identity", self._get_identity(session_key=session_key, channel=channel, workspace=root))
        add("bootstrap", self._load_bootstrap_files(root))
        add("tool_contract", render_template("agent/tool_contract.md"))
        if root.resolve() != self.workspace.resolve():
            add("project", f"# Current Project\n\nWorking directory: {root.resolve()}\nUse it as the default root for project files and relative tool paths.")

        mem_parts: list[str] = []
        raw_global_memory = self.memory.read_memory() if include_memory else ""
        if raw_global_memory and not self._is_template_content(
            raw_global_memory, "memory/MEMORY.md"
        ):
            global_memory = self.memory.format_memory_context(raw_global_memory)
            global_memory = select_memory_view(global_memory)
            if global_memory:
                mem_parts.append(global_memory)
        topic_store = self._get_topic_store(session_key) if include_memory else None
        topic_mem = topic_store.read_memory() if topic_store is not None else ""
        if topic_mem:
            topic_view = select_memory_view(topic_mem)
            if topic_view:
                mem_parts.append(f"### Topic Memory\n{topic_view}")
        add("memory", "\n\n".join(mem_parts))

        if todos:
            from nanobot.fork.agent.tools.todo import format_todos
            add("todos", "# Active Todos\n\n" + format_todos(todos))

        skills = self._skills_for_workspace(root)
        always_skills = skills.get_always_skills()
        always_content = skills.load_skills_for_context(always_skills) if always_skills else ""
        add("always_skills", f"# Active Skills\n\n{always_content}" if always_content else "")

        skills_summary = skills.build_skills_summary(exclude=set(always_skills), workspace=root)
        if skills_summary:
            add("skills_index", render_template("agent/skills_section.md", skills_summary=skills_summary))

        if include_memory and include_memory_recent_history:
            entries = self.memory.read_recent_history_for_prompt(
                since_cursor=self.memory.get_last_dream_cursor(),
                session_key=session_key,
                unified_session=unified_session,
            )
            if entries:
                capped = entries[-self._MAX_RECENT_HISTORY:]
                history_text = "\n".join(
                    f"- [{entry['timestamp']}] {entry['content']}" for entry in capped
                )
                history_text = truncate_text_to_tokens(history_text, self._MAX_HISTORY_TOKENS)
                add("recent_history", "# Recent History\n\n" + history_text)

        if isinstance(session_summary, dict) and session_summary.get("text") != "(nothing)":
            add("summary", f"[Archived Context Summary]\n\nPrevious conversation summary (last active {session_summary['last_active']}):\n{session_summary['text']}")

        # Active/legacy continuation state is injected with the current user
        # metadata so the stable system prefix remains cache-friendly.

        if include_learning_rules:
            lr_path = self.data_dir / "LEARNING_RULES.md"
            if lr_path.exists():
                add("learning_rules", lr_path.read_text(encoding="utf-8"))

        prompt = "\n\n---\n\n".join(content for _name, content in sections)
        section_tokens = {
            name: estimate_message_tokens({"role": "system", "content": content})
            for name, content in sections
        }
        section_tokens["total"] = estimate_message_tokens(
            {"role": "system", "content": prompt}
        )
        return SystemPromptBuild(prompt=prompt, section_tokens=section_tokens)


    def _build_skill_match_reminder(
        self,
        user_message: str,
        skills: SkillsLoader | None = None,
    ) -> str:
        if not user_message or not user_message.strip():
            return ""
        loader = skills or self.skills
        try:
            entries = loader.list_skills(filter_unavailable=True)
        except Exception:
            return ""
        if not entries:
            return ""
        msg_lower = user_message.lower()
        suggestions: list[tuple[str, str, int]] = []
        for skill in entries:
            name = skill.get("name", "")
            desc = loader.get_skill_description(name) or ""
            if not name or not desc:
                continue
            tokens = {
                token.strip(".,;:()[]{}\"'<>/").lower()
                for token in desc.split()
                if len(token) >= self._SKILL_MATCH_MIN_TOKEN_LEN
            }
            hits = sum(1 for token in tokens if token and token in msg_lower)
            if hits > 0:
                suggestions.append((name, desc.strip(), hits))
        if not suggestions:
            return ""
        suggestions.sort(key=lambda item: item[2], reverse=True)
        bullets = "\n".join(
            f"  - {name}: {desc} (call `load_skill(name=\"{name}\")`)"
            for name, desc, _ in suggestions[: self._SKILL_MATCH_MAX_SUGGESTIONS]
        )
        return (
            "<system-reminder>\n"
            "Your latest message may match these installed skills. "
            "Consider loading one BEFORE proceeding so you don't miss "
            "steps the skill author already worked out:\n\n"
            f"{bullets}\n"
            "</system-reminder>"
        )
