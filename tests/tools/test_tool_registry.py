from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry


class _FakeTool(Tool):
    def __init__(self, name: str, schema: dict[str, Any] | None = None, *, read_only: bool = False):
        self._name = name
        self._schema = schema
        self._read_only = read_only

    @property
    def name(self) -> str:
        return self._name

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema or {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs


def _tool_names(definitions: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for definition in definitions:
        fn = definition.get("function", {})
        names.append(fn.get("name", ""))
    return names


def _registry_with_names(names: list[str]) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(_FakeTool(name))
    return registry


def test_get_definitions_orders_builtins_then_mcp_tools() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("mcp_git_status"))
    registry.register(_FakeTool("write_file"))
    registry.register(_FakeTool("mcp_fs_list"))
    registry.register(_FakeTool("read_file"))

    assert _tool_names(registry.get_definitions()) == [
        "read_file",
        "write_file",
        "mcp_fs_list",
        "mcp_git_status",
    ]


def test_prepare_call_rejects_near_miss_tool_name_with_suggestion() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("readFile", {"path": "foo.txt"})

    assert tool is None
    assert params == {"path": "foo.txt"}
    assert error is not None
    assert "Tool 'readFile' not found" in error
    assert "Did you mean 'read_file'?" in error
    assert "must match exactly" in error


def test_suggest_name_handles_canonical_tool_name_variants() -> None:
    registry = _registry_with_names(["read_file"])
    expected = {
        "readFile": "read_file",
        "read-file": "read_file",
        "READ_FILE": "read_file",
        "read file": "read_file",
        "readfile": "read_file",
    }

    assert {name: registry._suggest_name(name) for name in expected} == expected


def test_suggest_name_suppresses_low_confidence_and_non_unique_matches() -> None:
    registry = _registry_with_names(["read_file", "write_file"])

    for name in ["", "foo", "read", "file", "readfil", "read_file_tool"]:
        assert registry._suggest_name(name) is None

    ambiguous = _registry_with_names(["read_file", "readFile"])
    assert ambiguous._suggest_name("readfile") is None


def test_suggest_name_updates_after_register_and_unregister() -> None:
    registry = _registry_with_names(["read_file"])

    assert registry._suggest_name("readFile") == "read_file"

    registry.register(_FakeTool("readFile"))
    assert registry._suggest_name("read-file") is None

    registry.unregister("read_file")
    assert registry._suggest_name("read-file") == "readFile"


def test_prepare_call_read_file_rejects_non_object_params_with_actionable_hint() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("read_file", ["foo.txt"])

    assert tool is not None
    assert params == ["foo.txt"]
    assert error is not None
    assert "must be a JSON object" in error
    assert 'tool_name(param1="value1", param2="value2")' in error
    assert "matching the tool schema" in error


def test_prepare_call_parses_json_string_arguments() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("read_file", '{"path":"foo.txt"}')

    assert tool is not None
    assert params == {"path": "foo.txt"}
    assert error is None


def test_prepare_call_rejects_malformed_json_string_arguments() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("read_file", '{path:"foo.txt"}')

    assert tool is not None
    assert params == '{path:"foo.txt"}'
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_rejects_scalar_for_single_required_parameter() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool(
        "web_fetch",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    ))

    tool, params, error = registry.prepare_call("web_fetch", "https://example.com")

    assert tool is not None
    assert params == "https://example.com"
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_rejects_unquoted_scalar_strings_before_schema_cast() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool(
        "message",
        {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    ))

    tool, params, error = registry.prepare_call("message", "true")

    assert tool is not None
    assert params == "true"
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_unwraps_arguments_payload() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool(
        "read_file",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ))

    tool, params, error = registry.prepare_call(
        "read_file",
        {"arguments": '{"path":"foo.txt"}'},
    )

    assert tool is not None
    assert params == {"path": "foo.txt"}
    assert error is None


def test_prepare_call_treats_none_arguments_as_empty_object() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("list_exec_sessions"))

    tool, params, error = registry.prepare_call("list_exec_sessions", None)

    assert tool is not None
    assert params == {}
    assert error is None

    tool, params, error = registry.prepare_call("list_exec_sessions", "null")

    assert tool is not None
    assert params == "null"
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_other_tools_keep_generic_object_validation() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("grep"))

    tool, params, error = registry.prepare_call("grep", ["TODO"])

    assert tool is not None
    assert params == ["TODO"]
    assert error == (
        "Error: Tool 'grep' parameters must be a JSON object, got list. "
        'Use named parameters like tool_name(param1="value1", param2="value2") '
        "matching the tool schema."
    )


async def test_registry_rejects_unknown_builtin_tool_parameters(tmp_path) -> None:
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(
        ReadFileTool(
            workspace=tmp_path,
            allowed_dir=tmp_path,
            restrict_to_workspace=True,
        )
    )

    result = await registry.execute(
        "read_file",
        {"path": "sample.txt", "line_limit": 1},
    )

    assert "Invalid parameters" in result
    assert "unexpected parameter line_limit" in result
    assert "one" not in result


async def test_registry_preserves_successful_exec_output_that_starts_with_error() -> None:
    registry = ToolRegistry()
    output = "Error: generated report successfully\n\nExit code: 0"
    tool = _FakeTool("exec")
    tool.execute = AsyncMock(return_value=output)
    registry.register(tool)

    result = await registry.execute("exec", {})

    assert result == output


async def test_registry_uses_structured_tool_result_for_errors() -> None:
    registry = ToolRegistry()
    output = "Error: plain tool output, not a structured failure"
    raw_tool = _FakeTool("raw_output")
    raw_tool.execute = AsyncMock(return_value=output)
    registry.register(raw_tool)

    raw_result = await registry.execute("raw_output", {})

    assert raw_result == output

    failing_tool = _FakeTool("failing_tool")
    failing_tool.execute = AsyncMock(return_value=ToolResult.error("Error: real failure"))
    registry.register(failing_tool)

    error_result = await registry.execute("failing_tool", {})

    assert isinstance(error_result, ToolResult)
    assert error_result.is_error
    assert error_result.startswith("Error: real failure")
    assert "[Analyze the error above" in error_result


def test_get_definitions_returns_cached_result() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    first = registry.get_definitions()
    assert registry._cached_definitions is not None
    second = registry.get_definitions()
    assert first == second


def test_register_invalidates_cache() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    first = registry.get_definitions()
    registry.register(_FakeTool("write_file"))
    second = registry.get_definitions()
    assert first is not second
    assert len(second) == 2


def test_unregister_invalidates_cache() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    registry.register(_FakeTool("write_file"))
    first = registry.get_definitions()
    registry.unregister("write_file")
    second = registry.get_definitions()
    assert first is not second
    assert len(second) == 1


async def test_registry_blocks_every_tool_for_disable_all_tools_policy() -> None:
    registry = ToolRegistry()
    tool = _FakeTool("write_file")
    tool.execute = AsyncMock(return_value="written")
    registry.register(tool)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="repair", metadata={"tool_policy": {"disable_all_tools": True}},
    ))
    try:
        result = await registry.execute("write_file", {"path": "x", "content": "y"})
    finally:
        reset_request_context(token)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "all tools are blocked" in str(result)
    tool.execute.assert_not_awaited()


async def test_registry_blocks_named_tool_but_keeps_other_tools_available() -> None:
    registry = ToolRegistry()
    blocked = _FakeTool("long_task")
    allowed = _FakeTool("read_file")
    blocked.execute = AsyncMock(return_value="started")
    allowed.execute = AsyncMock(return_value="read")
    registry.register(blocked)
    registry.register(allowed)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="repair", metadata={"tool_policy": {"blocked_tool_names": ["long_task"]}},
    ))
    try:
        blocked_result = await registry.execute("long_task", {"goal": "x"})
        allowed_result = await registry.execute("read_file", {"path": "x"})
    finally:
        reset_request_context(token)

    assert isinstance(blocked_result, ToolResult)
    assert blocked_result.is_error
    assert "tool 'long_task' is blocked" in str(blocked_result)
    blocked.execute.assert_not_awaited()
    allowed.execute.assert_awaited_once_with(path="x")
    assert allowed_result == "read"


async def test_get_definitions_hides_tools_blocked_for_current_request() -> None:
    registry = _registry_with_names(["read_file", "long_task", "todo_write"])
    unfiltered = registry.get_definitions()
    token = bind_request_context(RequestContext(
        channel="api", chat_id="repair", metadata={"tool_policy": {"blocked_tool_names": ["long_task", "todo_write"]}},
    ))
    try:
        filtered = registry.get_definitions()
    finally:
        reset_request_context(token)

    assert _tool_names(unfiltered) == ["long_task", "read_file", "todo_write"]
    assert _tool_names(filtered) == ["read_file"]
    assert _tool_names(registry.get_definitions()) == ["long_task", "read_file", "todo_write"]


async def test_get_definitions_is_empty_when_all_tools_are_disabled() -> None:
    registry = _registry_with_names(["read_file", "write_file"])
    token = bind_request_context(RequestContext(
        channel="api", chat_id="repair", metadata={"tool_policy": {"disable_all_tools": True}},
    ))
    try:
        assert registry.get_definitions() == []
    finally:
        reset_request_context(token)


async def test_read_only_mode_hides_and_blocks_mutating_tools_but_allows_read_tools() -> None:
    registry = ToolRegistry()
    read_tool = _FakeTool("read_file", read_only=True)
    write_tool = _FakeTool("write_file")
    read_tool.execute = AsyncMock(return_value="read")
    write_tool.execute = AsyncMock(return_value="written")
    registry.register(read_tool)
    registry.register(write_tool)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="repair", metadata={"tool_policy": {"read_only_mode": True}},
    ))
    try:
        assert _tool_names(registry.get_definitions()) == ["read_file"]
        read_result = await registry.execute("read_file", {"path": "x"})
        write_result = await registry.execute("write_file", {"path": "x", "content": "y"})
    finally:
        reset_request_context(token)

    assert read_result == "read"
    read_tool.execute.assert_awaited_once_with(path="x")
    assert isinstance(write_result, ToolResult) and write_result.is_error
    assert "not allowed in request read-only mode" in str(write_result)
    write_tool.execute.assert_not_awaited()


async def test_read_only_mode_allows_only_safe_shape_of_mixed_tool() -> None:
    class _MixedTool(_FakeTool):
        @property
        def supports_read_only_calls(self) -> bool:
            return True

        def is_read_only_call(self, params: Any) -> bool:
            return isinstance(params, dict) and params.get("action") == "check"

    registry = ToolRegistry()
    tool = _MixedTool("my")
    tool.execute = AsyncMock(return_value="ok")
    registry.register(tool)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="repair", metadata={"tool_policy": {"read_only_mode": True}},
    ))
    try:
        assert _tool_names(registry.get_definitions()) == ["my"]
        check_result = await registry.execute("my", {"action": "check"})
        set_result = await registry.execute("my", {"action": "set", "key": "model", "value": "x"})
    finally:
        reset_request_context(token)

    assert check_result == "ok"
    assert isinstance(set_result, ToolResult) and set_result.is_error
    tool.execute.assert_awaited_once_with(action="check")
