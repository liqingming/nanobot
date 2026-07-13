"""Agent hook that observes file-editing tools and emits file-edit activity."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
import re
from typing import Any

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentRunHookContext,
    AgentTurnHookContext,
)
from nanobot.providers.base import ToolCallRequest
from nanobot.utils.file_edit_events import (
    FileEditTracker,
    build_file_edit_end_event,
    build_file_edit_error_event,
    build_file_edit_start_event,
    prepare_file_edit_trackers,
)
from nanobot.utils.progress_events import (
    invoke_file_edit_progress,
    on_progress_accepts_file_edit_events,
)


class FileEditActivityHook(AgentHook):
    """Translate file-editing tool lifecycle events into WebUI progress events."""

    def __init__(
        self,
        *,
        on_progress: Callable[..., Awaitable[None]] | None,
        workspace: Path | None,
    ) -> None:
        super().__init__()
        self._on_progress = (
            on_progress
            if on_progress is not None and on_progress_accepts_file_edit_events(on_progress)
            else None
        )
        self._workspace = workspace
        self._trackers_by_call: dict[str, list[FileEditTracker]] = {}

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._trackers_by_call.clear()

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> None:
        if self._on_progress is None or not isinstance(params, dict):
            return
        trackers = prepare_file_edit_trackers(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            tool=tool,
            workspace=self._workspace,
            params=params,
        )
        if not trackers:
            return
        self._trackers_by_call[self._tool_call_key(tool_call)] = trackers
        await self._emit([build_file_edit_start_event(tracker, params) for tracker in trackers])

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        key = self._tool_call_key(tool_call)
        trackers = self._trackers_by_call.get(key, [])
        if trackers:
            await self._emit([build_file_edit_end_event(tracker) for tracker in trackers])
            self._trackers_by_call.pop(key, None)

    async def on_execute_tool_error(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        error: Any,
    ) -> None:
        key = self._tool_call_key(tool_call)
        trackers = self._trackers_by_call.get(key, [])
        if trackers:
            # apply_patch validates a batch atomically. Its error names the
            # specific edit that failed, so do not falsely mark every other
            # tracked file as failed when none of them was applied.
            failed_trackers = self._trackers_matching_error_path(trackers, str(error))
            if failed_trackers:
                await self._emit([
                    build_file_edit_error_event(tracker, str(error)) for tracker in failed_trackers
                ])
            self._trackers_by_call.pop(key, None)

    async def on_finally(self, context: AgentRunHookContext) -> None:
        if context.stop_reason != "cancelled" or not self._trackers_by_call:
            return
        trackers = [
            tracker
            for trackers in self._trackers_by_call.values()
            for tracker in trackers
        ]
        self._trackers_by_call.clear()
        await self._emit([
            build_file_edit_error_event(
                tracker,
                "Task interrupted before this tool finished.",
            )
            for tracker in trackers
        ])

    async def _emit(self, events: list[dict[str, Any]]) -> None:
        if self._on_progress is not None:
            await invoke_file_edit_progress(self._on_progress, events)

    @staticmethod
    def _trackers_matching_error_path(
        trackers: list[FileEditTracker], error: str
    ) -> list[FileEditTracker]:
        """Return only trackers whose path is explicitly named by a tool error."""
        match = re.search(r"(?:not found|multiple times) in (?P<path>.+?)(?:\r?\n|$)", error)
        if match is None:
            return []
        raw_path = match.group("path").strip().replace("\\", "/")
        return [
            tracker
            for tracker in trackers
            if tracker.display_path == raw_path or tracker.path.as_posix() == raw_path
        ]

    @staticmethod
    def _tool_call_key(tool_call: ToolCallRequest) -> str:
        call_id = getattr(tool_call, "id", "") or ""
        return f"{call_id}|{tool_call.name}" if call_id else f"{id(tool_call)}|{tool_call.name}"


def create_file_edit_activity_hook(context: AgentTurnHookContext) -> AgentHook | None:
    """Create the default file-edit observer for one agent turn."""
    if context.on_progress is None:
        return None
    return FileEditActivityHook(
        on_progress=context.on_progress,
        workspace=context.workspace,
    )
