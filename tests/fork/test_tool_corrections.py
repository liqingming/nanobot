from __future__ import annotations

import json

from nanobot.agent.tools.filesystem import ListDirTool
from nanobot.agent.tools.search import FindFilesTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.fork.agent.tools.corrections import CorrectedTool, ToolCorrectionStore


def test_grep_context_is_clamped_before_validation(tmp_path) -> None:
    store = ToolCorrectionStore(tmp_path)
    tool = CorrectedTool(GrepTool(), store)

    params = tool.cast_params({"pattern": "x", "context_after": 30})

    assert params["context_after"] == 20
    assert tool.validate_params(params) == []
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    entry = payload["corrections"]["grep:clamp:context_after"]
    assert entry["corrected_to"] == 20
    assert entry["hits"] == 1
    assert entry["observed_count"] == 2


def test_exec_allows_powershell_format_cmdlets_but_blocks_disk_format(tmp_path) -> None:
    store = ToolCorrectionStore(tmp_path)
    wrapped = ExecTool()
    tool = CorrectedTool(wrapped, store)

    assert wrapped._guard_command("Get-Item x | Format-List", ".") is None
    assert "deny pattern" in str(wrapped._guard_command("format C:", "."))

    assert tool.name == "exec"


def test_catalog_preserves_error_order_and_adds_preventive_hints(tmp_path) -> None:
    store = ToolCorrectionStore(tmp_path)
    exec_tool = CorrectedTool(ExecTool(), store)
    filesystem_tools = (
        CorrectedTool(ListDirTool(), store),
        CorrectedTool(FindFilesTool(), store),
        CorrectedTool(GrepTool(), store),
    )

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    entries = sorted(
        payload["corrections"].values(),
        key=lambda entry: entry["sequence"],
    )

    assert [entry["sequence"] for entry in entries] == [1, 2, 3, 4, 5]
    assert [entry["observed_count"] for entry in entries] == [2, 2, 1, 1, 3]
    assert "stdout.readline()" in exec_tool.description
    assert "here-string" in exec_tool.description
    assert "python -c" in exec_tool.description
    for tool in filesystem_tools:
        assert "authoritative workspace path" in tool.description
        assert "same missing path" in tool.description


def test_fork_factories_replace_registered_corrected_tools(tmp_path) -> None:
    from types import SimpleNamespace

    from nanobot.agent.tools.registry import ToolRegistry, iter_fork_tool_factories

    registry = ToolRegistry()
    registry.register(ExecTool())
    registry.register(GrepTool())
    registry.register(ListDirTool())
    registry.register(FindFilesTool())
    loop = SimpleNamespace(tools=registry, context=SimpleNamespace(data_dir=tmp_path))
    corrected_names = ("exec", "grep", "list_dir", "find_files")
    factories = {
        factory.__name__: factory
        for factory in iter_fork_tool_factories()
        if factory.__name__ in {f"correct_{name}_tool" for name in corrected_names}
    }

    for name in corrected_names:
        wrapped = factories[f"correct_{name}_tool"](loop)
        assert isinstance(wrapped, CorrectedTool)
        registry.register(wrapped)

    for name in corrected_names:
        assert isinstance(registry.get(name), CorrectedTool)
