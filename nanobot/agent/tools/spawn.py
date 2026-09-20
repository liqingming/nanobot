"""Spawn tool for creating background subagents."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.schema import (
    BooleanSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.fork.agent.subagent_control import SubagentControlTool  # noqa: F401
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.context import ToolContext


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
        wait=BooleanSchema(
            description=(
                "Wait for the subagent and return its result directly. Use this for a "
                "blocking consultation that must inform the current turn. Defaults to "
                "false for background execution."
            ),
            default=False,
        ),
        required=["task"],
    )
)
class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        manager = ctx.subagent_manager
        if manager is None:
            raise RuntimeError("SpawnTool requires an initialized subagent manager")
        return cls(manager=manager)

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
            "Each task starts fresh with its own system prompt and task, without parent "
            "conversation history, and reports back when done. Bind only the returned "
            "task id after creation; use subagent_control to list/status/wait/cancel and "
            "confirm termination. After binding and independent work, use subagent_control wait "
            "to let the host await the receipt; do not poll unchanged plans or wait in model thinking. "
            "Never substitute native Codex collaboration tools. "
            "Set wait=true for a consultation whose result must inform the current turn. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    @property
    def concurrency_safe(self) -> bool:
        """Each call owns its task state; the manager serializes capacity admission."""
        return True

    async def execute(
        self,
        task: str,
        label: str | None = None,
        temperature: float | None = None,
        wait: bool = False,
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
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            return ToolResult.error("Error: spawn requires an active model runtime")
        origin_channel = request_ctx.channel
        origin_chat_id = request_ctx.chat_id
        session_key = request_ctx.session_key or f"{origin_channel}:{origin_chat_id}"
        method = self._manager.run_inline if wait else self._manager.spawn
        return await method(
            task=task,
            runtime=request_ctx.runtime,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            origin_message_id=request_ctx.message_id,
            temperature=temperature,
            workspace_scope=current_workspace_scope(),
        )
