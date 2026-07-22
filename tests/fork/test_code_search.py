from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent.tools.registry import iter_fork_tool_factories
from nanobot.fork.agent.tools.code_intelligence_manager import CodeIntelligenceManager
from nanobot.fork.agent.tools.code_intelligence_models import CodeLocation
from nanobot.fork.agent.tools.code_search import CodeSearchTool
from nanobot.fork.agent.tools.lsp_client import LspError
from nanobot.fork.agent.tools.macro_risk import assess_macro_risk
from nanobot.fork.agent.tools.text_candidates import search_text_candidates


class FakeManager:
    def __init__(self, rows: list[CodeLocation] | None = None, *, error: str = "") -> None:
        self.rows = rows or []
        self.error = error
        self.calls: list[dict] = []

    async def query(self, workspace: Path, **kwargs):
        self.calls.append({"workspace": workspace, **kwargs})
        if self.error:
            raise LspError(self.error)
        return list(self.rows), {"semantic_available": True, "degraded": False}


@pytest.mark.asyncio
async def test_stage1_semantic_reference_is_structured_and_limited(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Gameplay" / "Foo.cs"
    source.parent.mkdir(parents=True)
    source.write_text("class Foo { void Target() {} }", encoding="utf-8")
    rows = [
        CodeLocation(f"Assets/Gameplay/Call{i}.cs", i, 3)
        for i in range(1, 4)
    ]
    manager = FakeManager(rows)
    tool = CodeSearchTool(tmp_path, manager=manager)

    data = json.loads(await tool.execute(
        action="references", file="Assets/Gameplay/Foo.cs", line=1, column=18,
        mode="semantic", max_results=2,
    ))

    assert data["mode_used"] == "semantic"
    assert data["semantic_total"] == 3
    assert len(data["semantic_results"]) == 2
    assert data["truncated"] is True
    assert manager.calls[0]["action"] == "references"


@pytest.mark.asyncio
async def test_stage1_semantic_unavailable_does_not_claim_text_results(tmp_path: Path) -> None:
    source = tmp_path / "Foo.cs"
    source.write_text("class Foo {}", encoding="utf-8")
    tool = CodeSearchTool(tmp_path, manager=FakeManager(error="not installed"))

    data = json.loads(await tool.execute(action="definition", file="Foo.cs", mode="semantic"))

    assert data == {
        "status": "unavailable",
        "action": "definition",
        "semantic_available": False,
        "degraded": True,
        "warning": "not installed",
        "results": [],
    }


def test_stage1_manager_reuses_workspace_session(monkeypatch, tmp_path: Path) -> None:
    manager = CodeIntelligenceManager()
    fake = SimpleNamespace(alive=True)
    manager._sessions[str(tmp_path.resolve()).casefold()] = fake

    async def run():
        assert await manager.session(tmp_path) is fake
        assert await manager.session(tmp_path) is fake

    import asyncio
    asyncio.run(run())


def test_stage1_fork_factory_registers_code_search(tmp_path: Path) -> None:
    loop = SimpleNamespace(
        workspace=tmp_path,
        tools_config=SimpleNamespace(
            code_intelligence=SimpleNamespace(
                model_dump=lambda: {"enabled": True, "command": []}
            )
        ),
    )
    factories = [
        factory for factory in iter_fork_tool_factories()
        if factory.__module__.endswith("code_search")
    ]
    assert len(factories) == 1
    assert isinstance(factories[0](loop), CodeSearchTool)


def test_stage2_macro_risk_uses_path_source_and_asmdef(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Editor" / "Channel" / "Foo.cs"
    source.parent.mkdir(parents=True)
    source.write_text("#if OVERSEA\nclass Foo {}\n#endif\n", encoding="utf-8")
    (source.parent / "Game.asmdef").write_text(
        json.dumps({"defineConstraints": ["OVERSEA"]}), encoding="utf-8"
    )

    risk = assess_macro_risk(source, tmp_path, intent="检查全部引用和影响范围")

    assert risk.macro_sensitive
    assert risk.score >= 9
    assert risk.suggested_scope == "workspace"
    assert any("conditional compilation" in reason for reason in risk.reasons)
    assert any("defineConstraints" in reason for reason in risk.reasons)


@pytest.mark.asyncio
async def test_stage2_auto_only_returns_text_difference(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Editor" / "Foo.cs"
    active = tmp_path / "Assets" / "Editor" / "Active.cs"
    inactive = tmp_path / "Assets" / "Editor" / "Inactive.cs"
    source.parent.mkdir(parents=True)
    source.write_text("#if UNITY_EDITOR\nvoid Target() {}\n#endif", encoding="utf-8")
    active.write_text("Target();", encoding="utf-8")
    inactive.write_text("#if OVERSEA\nTarget();\n#endif", encoding="utf-8")
    semantic = [CodeLocation("Assets/Editor/Active.cs", 1, 1)]
    tool = CodeSearchTool(tmp_path, manager=FakeManager(semantic))

    data = json.loads(await tool.execute(
        action="references", file="Assets/Editor/Foo.cs", symbol="Target",
        scope="Assets/Editor", mode="auto",
    ))

    assert data["mode_used"] == "semantic+text-diff"
    assert data["macro_sensitive"] is True
    assert [row["path"] for row in data["text_candidates"]] == [
        "Assets/Editor/Foo.cs", "Assets/Editor/Inactive.cs"
    ]
    assert data["text_candidates"][1]["condition_stack"] == ["OVERSEA"]


class FakeLspClient:
    def __init__(self, workspace: Path, *, call_hierarchy: bool, references: bool = True) -> None:
        self.workspace = workspace
        self.capabilities = {
            "callHierarchyProvider": call_hierarchy,
            "referencesProvider": references,
        }
        self.requests: list[str] = []

    async def request(self, method: str, params):
        self.requests.append(method)
        if method == "textDocument/prepareCallHierarchy":
            return [{"name": "Target", "uri": (self.workspace / "Target.cs").as_uri(),
                     "selectionRange": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 6}}}]
        if method == "callHierarchy/incomingCalls":
            return [{"from": {"name": "Caller", "uri": (self.workspace / "Caller.cs").as_uri(),
                                "selectionRange": {"start": {"line": 4, "character": 2}, "end": {"line": 4, "character": 8}}}}]
        if method == "callHierarchy/outgoingCalls":
            return [{"to": {"name": "Callee", "uri": (self.workspace / "Callee.cs").as_uri(),
                              "selectionRange": {"start": {"line": 8, "character": 1}, "end": {"line": 8, "character": 7}}}}]
        if method == "textDocument/references":
            return [{"uri": (self.workspace / "MaybeCaller.cs").as_uri(),
                     "range": {"start": {"line": 2, "character": 1}, "end": {"line": 2, "character": 7}}}]
        return []


@pytest.mark.asyncio
async def test_stage3_callers_uses_call_hierarchy(tmp_path: Path) -> None:
    manager = CodeIntelligenceManager()
    client = FakeLspClient(tmp_path, call_hierarchy=True)
    rows, metadata = await manager._call_hierarchy(client, {}, "callers")
    assert [(row.path, row.line, row.container) for row in rows] == [("Caller.cs", 5, "Caller")]
    assert metadata["degraded"] is False
    assert metadata["relationship_source"] == "lsp_call_hierarchy"


@pytest.mark.asyncio
async def test_stage3_callers_explicitly_degrades_to_references(tmp_path: Path) -> None:
    manager = CodeIntelligenceManager()
    client = FakeLspClient(tmp_path, call_hierarchy=False)
    rows, metadata = await manager._call_hierarchy(client, {}, "callers")
    assert rows[0].path == "MaybeCaller.cs"
    assert metadata["degraded"] is True
    assert metadata["relationship_source"] == "references_fallback"
    assert "not proven call sites" in metadata["warning"]


@pytest.mark.asyncio
async def test_stage3_callees_without_hierarchy_fails_clearly(tmp_path: Path) -> None:
    manager = CodeIntelligenceManager()
    client = FakeLspClient(tmp_path, call_hierarchy=False)
    with pytest.raises(LspError, match="does not support callees"):
        await manager._call_hierarchy(client, {}, "callees")


def test_stage4_candidates_include_cross_language_and_unity_assets(tmp_path: Path) -> None:
    files = {
        "Assets/A.cs": 'GetMethod("Target")',
        "Lua/A.lua": 'CS.Game.Target()',
        "Assets/A.prefab": 'm_MethodName: Target',
        "Assets/A.mat": 'stringTagMap: Target',
        "Assets/No.txt": 'Target',
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    rows, total = search_text_candidates(
        tmp_path, tmp_path, "Target", include_resources=True, max_results=20
    )

    assert total == 4
    assert {row["classification"] for row in rows} == {
        "csharp_text_candidate", "cross_language_candidate", "unity_resource_candidate"
    }
    assert {row["path"] for row in rows} == {
        "Assets/A.cs", "Lua/A.lua", "Assets/A.prefab", "Assets/A.mat"
    }


@pytest.mark.asyncio
async def test_stage4_candidates_action_never_claims_semantic_binding(tmp_path: Path) -> None:
    path = tmp_path / "Assets" / "Config.xml"
    path.parent.mkdir(parents=True)
    path.write_text('<handler name="Target" />', encoding="utf-8")
    tool = CodeSearchTool(tmp_path, manager=FakeManager())

    data = json.loads(await tool.execute(
        action="candidates", symbol="Target", include_resources=True
    ))

    assert data["semantic_available"] is False
    assert data["semantic_results"] == []
    assert data["text_candidates"][0]["classification"] == "configuration_candidate"


def test_stage1_config_accepts_camel_case_and_validates_timeouts() -> None:
    from pydantic import ValidationError

    from nanobot.config.schema import Config

    config = Config.model_validate({
        "tools": {
            "codeIntelligence": {
                "command": ["server", "--stdio"],
                "startupTimeoutSeconds": 12,
                "requestTimeoutSeconds": 7,
            }
        }
    })
    assert config.tools.code_intelligence.enabled is False
    assert config.tools.code_intelligence.command == ["server", "--stdio"]
    assert config.tools.code_intelligence.startup_timeout_seconds == 12
    assert config.tools.code_intelligence.request_timeout_seconds == 7
    with pytest.raises(ValidationError):
        Config.model_validate({"tools": {"codeIntelligence": {"requestTimeoutSeconds": -1}}})


def test_stage2_only_if_directive_is_macro_sensitive(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Gameplay" / "Foo.cs"
    source.parent.mkdir(parents=True)
    source.write_text("#if FEATURE\nclass Foo {}\n#endif", encoding="utf-8")
    risk = assess_macro_risk(source, tmp_path)
    assert risk.macro_sensitive is True
    assert risk.suggested_scope == "module"


def test_stage2_module_scope_uses_asmdef_root(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Game" / "Sub" / "Foo.cs"
    sibling = tmp_path / "Assets" / "Game" / "Other" / "Call.cs"
    source.parent.mkdir(parents=True)
    sibling.parent.mkdir(parents=True)
    source.write_text("#if FEATURE\nTarget();\n#endif", encoding="utf-8")
    sibling.write_text("Target();", encoding="utf-8")
    (tmp_path / "Assets" / "Game" / "Game.asmdef").write_text("{}", encoding="utf-8")
    rows, total = search_text_candidates(
        tmp_path, (tmp_path / "Assets" / "Game"), "Target", max_results=10
    )
    assert total == 2
    assert {row["path"] for row in rows} == {
        "Assets/Game/Sub/Foo.cs", "Assets/Game/Other/Call.cs"
    }


def test_stage1_utf16_position_encoding(tmp_path: Path) -> None:
    path = tmp_path / "Foo.cs"
    text = "😀Target();"
    position = CodeIntelligenceManager.text_document_position(path, 1, 8, text)
    assert position["position"]["character"] == 8


def test_stage1_normalizer_filters_non_file_and_outside_workspace(tmp_path: Path) -> None:
    region = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}
    outside = (tmp_path.parent / "secret.cs").as_uri()
    rows = CodeIntelligenceManager._normalize_locations([
        {"uri": "https://example.com/Foo.cs", "range": region},
        {"uri": outside, "range": region},
    ], tmp_path, kind="reference")
    assert rows == []


class DocumentClient:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, dict]] = []

    async def notify(self, method: str, params: dict) -> None:
        self.notifications.append((method, params))


@pytest.mark.asyncio
async def test_stage1_document_sync_opens_once_then_changes(tmp_path: Path) -> None:
    manager = CodeIntelligenceManager()
    client = DocumentClient()
    source = tmp_path / "Foo.cs"
    await manager._sync_document(client, tmp_path, source, "first")
    await manager._sync_document(client, tmp_path, source, "first")
    await manager._sync_document(client, tmp_path, source, "second")
    assert [method for method, _params in client.notifications] == [
        "textDocument/didOpen", "textDocument/didChange"
    ]
    assert client.notifications[-1][1]["textDocument"]["version"] == 2


@pytest.mark.asyncio
async def test_stage4_candidates_respects_resource_flag(tmp_path: Path) -> None:
    resource = tmp_path / "Assets" / "A.prefab"
    resource.parent.mkdir(parents=True)
    resource.write_text("Target", encoding="utf-8")
    tool = CodeSearchTool(tmp_path, manager=FakeManager())
    without = json.loads(await tool.execute(action="candidates", symbol="Target"))
    with_resources = json.loads(await tool.execute(
        action="candidates", symbol="Target", include_resources=True
    ))
    assert without["text_candidates"] == []
    assert with_resources["text_candidates"][0]["classification"] == "unity_resource_candidate"


@pytest.mark.asyncio
async def test_stage1_tool_close_drains_manager(tmp_path: Path) -> None:
    class ClosingManager(FakeManager):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def close_all(self) -> None:
            self.closed = True

    manager = ClosingManager()
    tool = CodeSearchTool(tmp_path, manager=manager)
    await tool.aclose()
    assert manager.closed is True


@pytest.mark.asyncio
async def test_stage2_auto_derives_symbol_at_position(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Gameplay" / "Foo.cs"
    caller = tmp_path / "Assets" / "Gameplay" / "Caller.cs"
    source.parent.mkdir(parents=True)
    source.write_text("#if FEATURE\nvoid Target() {}\n#endif", encoding="utf-8")
    caller.write_text("Target();", encoding="utf-8")
    tool = CodeSearchTool(tmp_path, manager=FakeManager())
    data = json.loads(await tool.execute(
        action="references", file="Assets/Gameplay/Foo.cs", line=2, column=8, mode="auto"
    ))
    assert data["mode_used"] == "semantic+text-diff"
    assert {row["path"] for row in data["text_candidates"]} == {
        "Assets/Gameplay/Foo.cs", "Assets/Gameplay/Caller.cs"
    }


@pytest.mark.asyncio
async def test_stage3_call_hierarchy_deduplicates_locations(tmp_path: Path) -> None:
    class DuplicateClient(FakeLspClient):
        async def request(self, method: str, params):
            rows = await super().request(method, params)
            if method == "callHierarchy/incomingCalls":
                return rows + rows
            return rows

    manager = CodeIntelligenceManager()
    rows, _metadata = await manager._call_hierarchy(
        DuplicateClient(tmp_path, call_hierarchy=True), {}, "callers"
    )
    assert len(rows) == 1


def test_stage4_configuration_and_unity_ui_candidates_are_classified(tmp_path: Path) -> None:
    files = {
        "Assets/View.uxml": '<Button name="Target" />',
        "Assets/View.uss": '#Target {}',
        "Assets/Graph.shadergraph": '"m_Name": "Target"',
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    rows, total = search_text_candidates(
        tmp_path, tmp_path, "Target", include_resources=True, max_results=10
    )
    assert total == 3
    assert {row["classification"] for row in rows} == {
        "configuration_candidate", "unity_resource_candidate"
    }


def test_stage2_exhaustive_intent_requests_workspace_scope(tmp_path: Path) -> None:
    source = tmp_path / "Assets" / "Gameplay" / "Foo.cs"
    source.parent.mkdir(parents=True)
    source.write_text("class Foo {}", encoding="utf-8")
    risk = assess_macro_risk(source, tmp_path, intent="查找全部引用和影响范围")
    assert risk.exhaustive_intent is True
    assert risk.suggested_scope == "workspace"


def test_stage1_normalizer_converts_utf16_output_columns(tmp_path: Path) -> None:
    source = tmp_path / "Foo.cs"
    source.write_text("😀Target", encoding="utf-8")
    rows = CodeIntelligenceManager._normalize_locations([{
        "uri": source.as_uri(),
        "range": {
            "start": {"line": 0, "character": 2},
            "end": {"line": 0, "character": 8},
        },
    }], tmp_path, kind="reference")
    assert rows[0].column == 2
    assert rows[0].end_column == 8


@pytest.mark.asyncio
async def test_stage1_document_sync_is_serialized(tmp_path: Path) -> None:
    class SlowDocumentClient(DocumentClient):
        async def notify(self, method: str, params: dict) -> None:
            await asyncio.sleep(0.01)
            await super().notify(method, params)

    manager = CodeIntelligenceManager()
    client = SlowDocumentClient()
    source = tmp_path / "Foo.cs"
    await asyncio.gather(
        manager._sync_document(client, tmp_path, source, "same"),
        manager._sync_document(client, tmp_path, source, "same"),
    )
    assert [method for method, _params in client.notifications] == ["textDocument/didOpen"]


def test_stage2_text_search_does_not_read_symlink_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-code-search.cs"
    outside.write_text("Target", encoding="utf-8")
    link = tmp_path / "linked.cs"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    rows, total = search_text_candidates(tmp_path, tmp_path, "Target", max_results=10)
    assert rows == []
    assert total == 0


def test_stage2_text_diff_keeps_multiple_matches_and_only_exact_semantic_hit(tmp_path: Path) -> None:
    source = tmp_path / "A.cs"
    source.write_text('Target(); GetMethod("Target"); Target();', encoding="utf-8")
    semantic = [CodeLocation("A.cs", 1, 1)]
    rows, total = search_text_candidates(
        tmp_path, tmp_path, "Target", semantic=semantic, max_results=10
    )
    assert total == 2
    assert [row["column"] for row in rows] == [22, 32]


def test_stage3_capability_options_objects_count_as_supported(tmp_path: Path) -> None:
    client = SimpleNamespace(capabilities={
        "callHierarchyProvider": {},
        "referencesProvider": {},
    })
    assert CodeIntelligenceManager.supports(client, "callers") is True
    assert CodeIntelligenceManager.supports(client, "references") is True


def test_stage1_factory_can_disable_code_search(tmp_path: Path) -> None:
    loop = SimpleNamespace(
        workspace=tmp_path,
        tools_config=SimpleNamespace(
            code_intelligence=SimpleNamespace(
                model_dump=lambda: {"enabled": False}
            )
        ),
    )
    factory = next(
        factory for factory in iter_fork_tool_factories()
        if factory.__module__.endswith("code_search")
    )
    assert factory(loop) is None


def test_stage1_factory_defaults_to_disabled_when_flag_is_missing(tmp_path: Path) -> None:
    loop = SimpleNamespace(
        workspace=tmp_path,
        tools_config=SimpleNamespace(
            code_intelligence=SimpleNamespace(model_dump=lambda: {"command": []})
        ),
    )
    factory = next(
        factory for factory in iter_fork_tool_factories()
        if factory.__module__.endswith("code_search")
    )
    assert factory(loop) is None
