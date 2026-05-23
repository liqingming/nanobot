"""Tool registry for dynamic tool management."""

from typing import Any, Callable

from nanobot.agent.tools.base import Tool


# Fork extension point — list of factory callables that produce
# fork-only Tool instances. Each callable receives the AgentLoop
# instance (so it can wire bus / sessions / context dependencies) and
# returns either a Tool to register or None to skip (e.g. a tool that
# depends on optional features the loop didn't enable).
#
# Fork modules append here at import time via ``register_fork_tool``.
# ``AgentLoop._register_default_tools`` iterates the list near the
# end and registers each non-None return value, so adding a new fork
# tool only requires:
#   1. Drop the file in ``nanobot/fork/agent/tools/``.
#   2. Import it from ``nanobot/fork/agent/tools/__init__.py`` so
#      bootstrap triggers the registration.
# Core never imports fork tool files directly.
_FORK_TOOL_FACTORIES: list[Callable[[Any], Tool | None]] = []


def register_fork_tool(factory: Callable[[Any], Tool | None]) -> None:
    """Register a fork tool factory. See ``_FORK_TOOL_FACTORIES`` doc."""
    _FORK_TOOL_FACTORIES.append(factory)


def iter_fork_tool_factories() -> list[Callable[[Any], Tool | None]]:
    """Return a snapshot of registered factories."""
    return list(_FORK_TOOL_FACTORIES)


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)
            
            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
