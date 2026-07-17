"""Tool registry for dynamic tool management."""

import json
from typing import Any, Callable

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import current_request_context


def is_tool_error_result(name: str, result: Any) -> bool:
    return isinstance(result, ToolResult) and result.is_error


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
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    @staticmethod
    def _lookup_key(name: str) -> str:
        """Normalize names for suggestions only; never for execution."""
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        key = self._lookup_key(str(name or ""))
        if not key:
            return None
        matches = [
            registered
            for registered in self._tools
            if self._lookup_key(registered) == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """Extract a normalized tool name from either OpenAI or flat schemas."""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions with stable ordering for cache-friendly prompts.

        Built-in tools are sorted first as a stable prefix, then MCP tools are
        sorted and appended.  The result is cached until the next
        register/unregister call.
        """
        if self._cached_definitions is None:
            definitions = [tool.to_schema() for tool in self._tools.values()]
            builtins: list[dict[str, Any]] = []
            mcp_tools: list[dict[str, Any]] = []
            for schema in definitions:
                name = self._schema_name(schema)
                if name.startswith("mcp_"):
                    mcp_tools.append(schema)
                else:
                    builtins.append(schema)

            builtins.sort(key=self._schema_name)
            mcp_tools.sort(key=self._schema_name)
            self._cached_definitions = builtins + mcp_tools

        request_ctx = current_request_context()
        policy = request_ctx.metadata.get("tool_policy") if request_ctx else None
        if isinstance(policy, dict) and policy.get("disable_all_tools") is True:
            return []
        definitions = self._cached_definitions
        if isinstance(policy, dict) and policy.get("read_only_mode") is True:
            definitions = [
                schema for schema in definitions
                if (tool := self._tools.get(self._schema_name(schema))) is not None and tool.supports_read_only_calls
            ]
        blocked_tool_names = policy.get("blocked_tool_names") if isinstance(policy, dict) else None
        if isinstance(blocked_tool_names, list):
            blocked = set(blocked_tool_names)
            return [schema for schema in definitions if self._schema_name(schema) not in blocked]
        return definitions

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, Any, str | None]:
        """Resolve, cast, and validate one tool call."""
        tool = self._tools.get(name)
        if not tool:
            suggestion = self._suggest_name(str(name))
            hint = f" Did you mean '{suggestion}'? Tool names must match exactly." if suggestion else ""
            return None, params, (
                ToolResult.error(
                    f"Error: Tool '{name}' not found.{hint} Available: {', '.join(self.tool_names)}"
                )
            )

        params = self._coerce_params(tool, params)
        if not isinstance(params, dict):
            return tool, params, (
                ToolResult.error(
                    f"Error: Tool '{name}' parameters must be a JSON object, got "
                    f"{type(params).__name__}. Use named parameters like "
                    'tool_name(param1="value1", param2="value2") matching the tool schema.'
                )
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                ToolResult.error(f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors))
            )
        return tool, cast_params, None

    @classmethod
    def _coerce_argument_value(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return {}

        if not stripped.startswith(("{", "[")):
            return value

        try:
            parsed = json.loads(stripped)
        except Exception:
            return value

        return parsed

    @classmethod
    def _coerce_params(cls, tool: Tool, params: Any) -> Any:
        params = cls._coerce_argument_value(params)
        return cls._unwrap_arguments_payload(tool, params)

    @classmethod
    def _unwrap_arguments_payload(cls, tool: Tool, params: Any) -> Any:
        if not isinstance(params, dict) or set(params) != {"arguments"}:
            return params
        properties = (tool.parameters or {}).get("properties", {})
        if isinstance(properties, dict) and "arguments" in properties:
            return params
        return cls._coerce_argument_value(params.get("arguments"))

    async def execute(self, name: str, params: Any) -> Any:
        """Execute a tool by name with given parameters."""
        request_ctx = current_request_context()
        policy = request_ctx.metadata.get("tool_policy") if request_ctx else None
        if isinstance(policy, dict) and policy.get("disable_all_tools") is True:
            return ToolResult.error("Error: all tools are blocked by the request tool_policy.")
        blocked_tool_names = policy.get("blocked_tool_names") if isinstance(policy, dict) else None
        if isinstance(blocked_tool_names, list) and name in blocked_tool_names:
            return ToolResult.error(f"Error: tool '{name}' is blocked by the request tool_policy.")
        hint = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if tool is not None and isinstance(policy, dict) and policy.get("read_only_mode") is True:
            if not tool.is_read_only_call(params):
                return ToolResult.error(f"Error: tool '{name}' call is not allowed in request read-only mode.")
        if error:
            return ToolResult.error(str(error) + hint)

        try:
            assert tool is not None  # guarded by prepare_call()
            result = await tool.execute(**params)
            if is_tool_error_result(name, result):
                return ToolResult.error(str(result) + hint)
            return result
        except Exception as e:
            return ToolResult.error(f"Error executing {name}: {str(e)}" + hint)

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
