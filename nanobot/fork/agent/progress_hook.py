"""Fork-specific progress hook behavior."""

from __future__ import annotations

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.progress_hook import AgentProgressHook


class ForkAgentProgressHook(AgentProgressHook):
    """Restore CLI-visible pre-tool narration while keeping upstream hook intact."""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress and self._on_stream and not context.streamed_content:
            thought = self._strip_think(context.response.content if context.response else None)
            if thought:
                await self._on_progress(thought)
        await super().before_execute_tools(context)