"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader, project_skill_roots
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import InboundMessage
from nanobot.session.goal_state import goal_state_runtime_lines
from nanobot.utils.helpers import (
    build_assistant_message,
    current_time_str,
    detect_image_mime,
    load_bundled_template,
    safe_filename,
    truncate_text_to_tokens,
)
from nanobot.utils.prompt_templates import render_template


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted kwargs for turn-attached capabilities."""
    return cli_app_utils.session_extra(metadata) | mcp_tools.session_extra(metadata)


def runtime_lines(state: Any, msg: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """Return model-visible runtime annotations for turn-attached capabilities."""
    return [
        *cli_app_utils.runtime_lines(msg, workspace, skip=skip),
        *mcp_tools.runtime_lines(
            msg,
            configured_server_names=set(state._mcp_servers),
            connected_server_names=set(state._mcp_stacks),
            skip=skip,
        ),
    ]


async def connect_mcp(state: Any, tools: ToolRegistry) -> None:
    await mcp_tools.connect_missing_servers(state, tools)


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    return await mcp_tools.handle_runtime_control(state, msg, tools)


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"
    _MAX_RECENT_HISTORY = 50
    _MAX_HISTORY_TOKENS = 8_000

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        *,
        data_dir: Path | None = None,
        topic_memory_factory: Any = None,
    ):
        self.workspace = workspace
        self.data_dir = data_dir if data_dir is not None else workspace
        self.timezone = timezone
        self.memory = MemoryStore(self.data_dir)
        self.skills = SkillsLoader(
            self.data_dir,
            disabled_skills=set(disabled_skills) if disabled_skills else None,
            extra_skill_roots=project_skill_roots(self.workspace),
        )
        self._topic_memory_factory = topic_memory_factory

    def _get_topic_store(self, session_key: str | None) -> MemoryStore | None:
        if self._topic_memory_factory is None or not session_key:
            return None
        return self._topic_memory_factory.get(session_key)

    def _skills_for_workspace(self, workspace: Path) -> SkillsLoader:
        if workspace == self.workspace:
            return self.skills
        return SkillsLoader(
            self.data_dir,
            disabled_skills=self.skills.disabled_skills,
            extra_skill_roots=project_skill_roots(workspace),
        )

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        include_learning_rules: bool = False,
        session_key: str | None = None,
        todos: list[dict[str, Any]] | None = None,
        channel: str | None = None,
        session_summary: str | None = None,
        workspace: Path | None = None,
        include_memory_recent_history: bool = True,
        unified_session: bool = False,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        root = workspace or self.workspace
        parts = [self._get_identity(session_key=session_key, channel=channel, workspace=root)]

        bootstrap = self._load_bootstrap_files(root)
        if bootstrap:
            parts.append(bootstrap)

        parts.append(render_template("agent/tool_contract.md"))

        mem_parts: list[str] = []
        global_memory = self.memory.get_memory_context()
        if global_memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            mem_parts.append(global_memory)
        topic_store = self._get_topic_store(session_key)
        topic_mem = topic_store.read_memory() if topic_store is not None else ""
        if topic_mem:
            mem_parts.append(f"### Topic Memory\n{topic_mem}")
        if mem_parts:
            parts.append("# Memory\n\n" + "\n\n".join(mem_parts))

        if todos:
            from nanobot.fork.agent.tools.todo import format_todos
            parts.append("# Active Todos\n\n" + format_todos(todos))

        skills = self._skills_for_workspace(root)
        always_skills = skills.get_always_skills()
        if always_skills:
            always_content = skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        if include_memory_recent_history:
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
                parts.append("# Recent History\n\n" + history_text)

        if session_summary:
            parts.append(f"[Archived Context Summary]\n\n{session_summary}")

        if include_learning_rules:
            lr_path = self.data_dir / "LEARNING_RULES.md"
            if lr_path.exists():
                parts.append(lr_path.read_text(encoding="utf-8"))

        return "\n\n---\n\n".join(parts)

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
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )
        if session_key is not None:
            safe_key = safe_filename(session_key.replace(":", "_"))
            memory_lines = (
                f"- Global memory: {data_path}/memory/MEMORY.md (cross-topic facts, auto-injected)\n"
                f"- Topic memory: {data_path}/memory/topics/{safe_key}/MEMORY.md (current topic facts, auto-injected; prefer writing here)\n"
                f"- Topic history: {data_path}/memory/topics/{safe_key}/HISTORY.md (event log, handled by consolidation)"
            )
        else:
            memory_lines = (
                f"- Long-term memory: {data_path}/memory/MEMORY.md (write important facts here)\n"
                f"- History log: {data_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM]."
            )
        return f"{identity}\n\n## Memory Paths\n{memory_lines}\n- Custom skills: {data_path}/skills/{{skill-name}}/SKILL.md"

    @staticmethod
    def _build_runtime_context(
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        sender_id: str | None = None,
        supplemental_lines: Sequence[str] | None = None,
    ) -> str:
        """Build untrusted runtime metadata block appended after user content."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if sender_id:
            lines += [f"Sender ID: {sender_id}"]
        if supplemental_lines:
            lines.extend(supplemental_lines)
        return (
            ContextBuilder._RUNTIME_CONTEXT_TAG
            + "\n"
            + "\n".join(lines)
            + "\n"
            + ContextBuilder._RUNTIME_CONTEXT_END
        )

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
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
        ]
        for filename in self.BOOTSTRAP_FILES:
            data_path = self.data_dir / filename
            candidates.append((filename, data_path))
            if self.data_dir != root:
                candidates.append((filename, root / filename))

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
            parts.append(f"## {label}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template."""
        tpl = load_bundled_template(template_path)
        if tpl is not None:
            return content.strip() == tpl.strip()
        return False

    _SKILL_MATCH_MIN_TOKEN_LEN = 4
    _SKILL_MATCH_MAX_SUGGESTIONS = 3

    def _build_skill_match_reminder(self, user_message: str) -> str:
        if not user_message or not user_message.strip():
            return ""
        try:
            skills = self.skills.list_skills(filter_unavailable=True)
        except Exception:
            return ""
        if not skills:
            return ""
        msg_lower = user_message.lower()
        suggestions: list[tuple[str, str, int]] = []
        for skill in skills:
            name = skill.get("name", "")
            desc = self.skills._get_skill_description(name) or ""
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

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        learning_ctx: str | None = None,
        session_key: str | None = None,
        todos: list[dict[str, Any]] | None = None,
        pending_summary: str | None = None,
        sender_id: str | None = None,
        session_summary: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        current_runtime_lines: Sequence[str] | None = None,
        workspace: Path | None = None,
        runtime_state: Any | None = None,
        inbound_message: Any | None = None,
        skip_runtime_lines: bool = False,
        include_memory_recent_history: bool = True,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        root = workspace or self.workspace
        extra = [*goal_state_runtime_lines(session_metadata)]
        if runtime_state is not None and inbound_message is not None:
            extra.extend(runtime_lines(runtime_state, inbound_message, root, skip=skip_runtime_lines))
        if current_runtime_lines:
            extra.extend(line for line in current_runtime_lines if line)
        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            sender_id=sender_id,
            supplemental_lines=extra or None,
        )
        user_content = self._build_user_content(current_message, media)

        reminder = ""
        effective_pending = pending_summary or session_summary
        if effective_pending and effective_pending.strip():
            reminder = (
                "<system-reminder>\n"
                "Updated long-term memory (consolidated from earlier in this "
                "conversation; not yet promoted to MEMORY.md):\n\n"
                f"{effective_pending.strip()}\n"
                "</system-reminder>"
            )
        skill_hint = self._build_skill_match_reminder(current_message)
        prefix_parts = [part for part in (learning_ctx, reminder, skill_hint) if part]
        prefix = "\n\n".join(prefix_parts) if prefix_parts else ""

        if isinstance(user_content, str):
            user_with_prefix = f"{prefix}\n\n{user_content}" if prefix else user_content
            merged = f"{user_with_prefix}\n\n{runtime_ctx}"
        else:
            merged = ([{"type": "text", "text": prefix}] if prefix else []) + user_content
            merged.append({"type": "text", "text": runtime_ctx})

        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    include_learning_rules=(learning_ctx is not None),
                    session_key=session_key,
                    todos=todos,
                    channel=channel,
                    session_summary=session_summary,
                    workspace=root,
                    include_memory_recent_history=include_memory_recent_history,
                    unified_session=unified_session,
                ),
            },
            *history,
        ]
        if messages[-1].get("role") == current_role and len(messages) > 1:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
