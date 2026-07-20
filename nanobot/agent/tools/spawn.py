"""Spawn tool for creating background subagents."""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.schema import NumberSchema, StringSchema, tool_parameters_schema
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


_SCOPE_RE = re.compile(
    r"(?:scope|范围|only|仅限|仅处理|under|目录|路径|files?|文件)",
    re.IGNORECASE,
)
_DELIVERABLE_RE = re.compile(
    r"(?:deliverable|交付|输出|产出|report|报告|summary|摘要|清单|实现|修复|文档|代码)",
    re.IGNORECASE,
)
_ACCEPTANCE_RE = re.compile(
    r"(?:acceptance|验收|完成条件|done when|verify|验证|测试|通过|必须|确保|确认)",
    re.IGNORECASE,
)


def _missing_task_boundaries(task: str) -> list[str]:
    """Return missing delegation boundaries for non-trivial subagent tasks."""
    text = task.strip()
    missing: list[str] = []
    if len(text) < 40:
        missing.append("objective/details")
    if not _SCOPE_RE.search(text):
        missing.append("scope")
    if not _DELIVERABLE_RE.search(text):
        missing.append("expected deliverable")
    if not _ACCEPTANCE_RE.search(text):
        missing.append("acceptance criteria")
    return missing


@tool_parameters(
    tool_parameters_schema(
        task=StringSchema("The task for the subagent to complete"),
        label=StringSchema("Optional short label for the task (for display)"),
        temperature=NumberSchema(
            description=(
                "Optional sampling temperature for the subagent "
                "(0.0 = deterministic, higher = more creative). "
                "Defaults to the provider's configured temperature."
            ),
            minimum=0.0,
            maximum=2.0,
        ),
        required=["task"],
    )
)
class SpawnTool(Tool, ContextAware):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel: ContextVar[str] = ContextVar("spawn_origin_channel", default="cli")
        self._origin_chat_id: ContextVar[str] = ContextVar("spawn_origin_chat_id", default="direct")
        self._session_key: ContextVar[str] = ContextVar("spawn_session_key", default="cli:direct")
        self._origin_message_id: ContextVar[str | None] = ContextVar(
            "spawn_origin_message_id",
            default=None,
        )

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(manager=ctx.subagent_manager)

    def set_context(self, ctx: RequestContext) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel.set(ctx.channel)
        self._origin_chat_id.set(ctx.chat_id)
        self._session_key.set(ctx.session_key or f"{ctx.channel}:{ctx.chat_id}")
        self._origin_message_id.set(ctx.message_id)

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use only when the task is independently executable and its objective, scope, "
            "expected deliverable, and acceptance criteria are explicit in the task text. "
            "Do not spawn for ambiguous work or when the next step depends on evidence not "
            "gathered yet; keep that work in the main agent. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    async def execute(
        self,
        task: str,
        label: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        missing = _missing_task_boundaries(task)
        if missing:
            return (
                "Cannot spawn subagent: task is not independently executable; missing "
                + ", ".join(missing)
                + ". Clarify these boundaries or keep the investigation in the main agent."
            )
        running = self._manager.get_running_count()
        limit = self._manager.max_concurrent_subagents
        if running >= limit:
            return (
                f"Cannot spawn subagent: concurrency limit reached "
                f"({running}/{limit} running). Wait for a running subagent "
                f"to complete before spawning a new one."
            )
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel.get(),
            origin_chat_id=self._origin_chat_id.get(),
            session_key=self._session_key.get(),
            origin_message_id=self._origin_message_id.get(),
            temperature=temperature,
            workspace_scope=current_workspace_scope(),
        )
