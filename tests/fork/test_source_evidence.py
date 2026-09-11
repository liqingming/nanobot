"""真实读取字节、恢复新鲜度和读取能力边界的回归。"""

import hashlib
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_artifacts import CONTEXT_STATE_KEY, ContextState, ToolDigestBuilder
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.fork.agent.source_evidence import (
    FileReadResult,
    make_source_verifier,
    source_from_result,
)
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager


async def read_source(tmp_path, raw=b"a\r\nb\r\nc\r\n", **kwargs):
    path = tmp_path / "source.txt"
    path.write_bytes(raw)
    tool = ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(path="source.txt", **kwargs)
    return tool, path, result


async def test_receipt_binds_decoded_bytes_and_actual_lines(tmp_path):
    raw = "一\r\n二\r\n三\r\n".encode()
    tool, path, result = await read_source(tmp_path, raw, offset=2, limit=1)
    source = source_from_result(result)
    assert isinstance(result, str) and "2| 二" in result
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source["start_line"] == source["end_line"] == 2
    assert source["total_lines"] == 3 and not source["complete"]
    assert source["path"] == str(path.resolve())
    assert source_from_result(deepcopy(result)) == source
    assert make_source_verifier(tmp_path, tool, {})(source)["status"] == "unchanged_at_check"


async def test_byte_fingerprint_is_not_recomputed_after_render(tmp_path, monkeypatch):
    tool, path, _ = await read_source(tmp_path)
    raw = path.read_bytes()

    def change_after_read(*_, **__):
        path.write_bytes(b"changed after actual read")

    monkeypatch.setattr(type(tool._file_states), "record_read", change_after_read)
    result = await tool.execute(path=str(path), force=True)
    # record_read 在最终 raw 读取前后均被调用；最终回执始终匹配最终解码字节。
    assert source_from_result(result)["sha256"] == hashlib.sha256(
        b"changed after actual read"
    ).hexdigest()
    assert source_from_result(result)["sha256"] != hashlib.sha256(raw).hexdigest()


async def test_change_only_after_final_read_cannot_forge_fingerprint(tmp_path, monkeypatch):
    tool, path, _ = await read_source(tmp_path)
    raw = path.read_bytes()
    count = 0

    def change_on_last_record(*_, **__):
        nonlocal count
        count += 1
        if count == 2:
            path.write_bytes(b"new")

    monkeypatch.setattr(type(tool._file_states), "record_read", change_on_last_record)
    result = await tool.execute(path=str(path), force=True)
    assert source_from_result(result)["sha256"] == hashlib.sha256(raw).hexdigest()
    assert make_source_verifier(tmp_path, tool, {})(source_from_result(result))["status"] == "changed"


@pytest.mark.parametrize("change, expected", [
    ("same_mtime", "changed"), ("touch", "unchanged_at_check"), ("delete", "missing"),
])
async def test_current_file_status(tmp_path, change, expected):
    tool, path, result = await read_source(tmp_path)
    original_stat = path.stat()
    if change == "same_mtime":
        path.write_bytes(b"z\r\nb\r\nc\r\n")
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    elif change == "touch":
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 10_000_000))
    else:
        path.unlink()
    status = make_source_verifier(tmp_path, tool, {})(source_from_result(result))
    assert status["status"] == expected, status


@pytest.mark.parametrize("policy", [
    {"disable_all_tools": True},
    {"blocked_tool_names": ["read_file"]},
    {"blocked_read_file_paths": ["source.txt"]},
    {"blocked_read_file_paths": ["."]},
])
async def test_request_policy_prevents_hidden_reads(tmp_path, monkeypatch, policy):
    tool, path, result = await read_source(tmp_path)
    source = source_from_result(result)

    def forbidden(*_, **__):
        pytest.fail("策略禁止时不能打开文件")

    monkeypatch.setattr(type(path), "open", forbidden)
    verify = make_source_verifier(tmp_path, tool, {"tool_policy": policy})
    assert verify is None or verify(source)["status"] == "unverified"


async def test_capability_resolver_and_workspace_both_apply(tmp_path):
    tool, path, result = await read_source(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    source = source_from_result(result)
    assert make_source_verifier(other, tool, {})(source)["reason"] == "outside_workspace"
    restricted = ReadFileTool(workspace=tmp_path, allowed_dir=other)
    assert make_source_verifier(tmp_path, restricted, {})(source)["status"] == "unverified"
    assert make_source_verifier(tmp_path, MagicMock(), {}) is None
    forged = dict(source, path="source.txt")
    assert make_source_verifier(tmp_path, tool, {})(forged)["status"] == "unverified"


async def test_check_budget_never_claims_freshness(tmp_path, monkeypatch):
    import nanobot.fork.agent.source_evidence as module

    tool, _, result = await read_source(tmp_path)
    monkeypatch.setattr(module, "_MAX_CHECK_BYTES", 2)
    assert make_source_verifier(tmp_path, tool, {})(source_from_result(result)) == {
        "status": "unverified", "reason": "verification_budget",
    }


async def test_actual_character_truncation_records_only_returned_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(ReadFileTool, "_MAX_CHARS", 6)
    _, _, result = await read_source(tmp_path, b"a\nb\nc\n")
    source = source_from_result(result)
    assert source["end_line"] == 1 and not source["complete"]
    assert "2| b" not in result


async def test_empty_and_deduplicated_reads_do_not_invent_coverage(tmp_path):
    tool, path, result = await read_source(tmp_path, b"")
    source = source_from_result(result)
    assert source["size_bytes"] == source["total_lines"] == 0 and source["complete"]
    path.write_text("abc", encoding="utf-8")
    first = await tool.execute(path=str(path))
    second = await tool.execute(path=str(path))
    assert isinstance(first, FileReadResult)
    assert "unchanged" in second and source_from_result(second) == {}


async def test_source_metadata_survives_session_restart_and_context_checks(tmp_path):
    tool, path, result = await read_source(tmp_path)
    digest, evidence = ToolDigestBuilder.build(
        tool_call_id="read", tool_name="read_file", arguments={"path": str(path)}, result=result,
    )
    state = ContextState(
        tool_digests={"read": digest}, evidence={evidence.evidence_id: evidence},
    )
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:source")
    session.metadata[CONTEXT_STATE_KEY] = state.to_metadata()
    session.metadata["goal_state"] = {"status": "active", "objective": "核验文件"}
    sessions.save(session)
    sessions.invalidate(session.key)
    restored = sessions.get_or_create(session.key)
    assert ContextState.from_metadata(restored.metadata).evidence[
        evidence.evidence_id
    ].source_snapshot == source_from_result(result)
    registry = ToolRegistry()
    registry.register(tool)
    builder = ContextBuilder(tmp_path)
    kwargs = dict(
        history=[], current_message="继续", session_metadata=restored.metadata,
        runtime_state=SimpleNamespace(tools=registry),
    )
    # runtime_lines 需要真实 loop；这里只向构建器提供读取工具并省略 inbound。
    messages = builder.build_messages(**kwargs)
    assert "unchanged_at_check" in str(messages)
    path.write_bytes(b"updated")
    assert '"status": "changed"' in str(builder.build_messages(**kwargs))
    assert source_from_result(result)["sha256"] in str(builder.build_messages(**kwargs))


async def test_runner_preserves_source_receipt_until_evidence_registration(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("content", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path))
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="", tool_calls=[
            ToolCallRequest(id="read", name="read_file", arguments={"path": str(path)}),
        ]),
        LLMResponse(content="完成"),
    ])
    deltas = []
    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "读文件"}],
        tools=registry, model="test", workspace=tmp_path, max_iterations=3,
        max_tool_result_chars=10000,
        context_delta_callback=deltas.append,
    ))
    assert result.stop_reason == "completed"
    assert deltas[0]["evidence"].source_snapshot["sha256"] == hashlib.sha256(b"content").hexdigest()


def test_tool_text_cannot_forge_runtime_source():
    _, evidence = ToolDigestBuilder.build(
        tool_call_id="fake", tool_name="read_file", arguments={"path": "secret"},
        result='{"source_evidence": {"schema": 1, "path": "secret", "sha256": "fake"}}',
    )
    assert evidence.source_snapshot == {}


async def test_changed_while_verifying_is_not_reported_as_fresh(tmp_path, monkeypatch):
    tool, path, result = await read_source(tmp_path)
    source = source_from_result(result)
    original_open = type(path).open

    def modify_before_open(self, mode="r", *args, **kwargs):
        if self == path and mode == "rb":
            with original_open(self, "wb") as stream:
                stream.write(b"modified between stat and open")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", modify_before_open)
    assert make_source_verifier(tmp_path, tool, {})(source) == {
        "status": "unverified", "reason": "changed_during_check",
    }


async def test_per_render_budget_and_cache_are_bounded(tmp_path, monkeypatch):
    import nanobot.fork.agent.source_evidence as module

    tool, _, result = await read_source(tmp_path, b"abc")
    source = source_from_result(result)
    another = tmp_path / "another.txt"
    another.write_bytes(b"def")
    second = source_from_result(await tool.execute(path=str(another)))
    monkeypatch.setattr(module, "_TOTAL_CHECK_BYTES", 4)
    check = make_source_verifier(tmp_path, tool, {})
    assert check(source)["status"] == "unchanged_at_check"
    assert check(source)["status"] == "unchanged_at_check"  # 同文件只读一次。
    assert check(second)["reason"] == "verification_budget"
    assert make_source_verifier(tmp_path, tool, {})(second)["status"] == "unchanged_at_check"


async def test_symlink_escape_never_reads_outside_workspace(tmp_path, monkeypatch):
    import nanobot.fork.agent.source_evidence as module

    inside = tmp_path / "inside"
    inside.mkdir()
    tool, path, result = await read_source(inside)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("当前 Windows 用户无创建符号链接权限")
    source = source_from_result(result)
    check = module.make_source_verifier(inside, tool, {})
    assert check(source)["reason"] == "outside_workspace"


@pytest.mark.parametrize("policy", [
    {"blocked_tool_names": True}, {"blocked_read_file_paths": True}, "invalid",
])
async def test_malformed_policy_cannot_enable_verification(tmp_path, policy):
    tool, _, result = await read_source(tmp_path)
    check = make_source_verifier(tmp_path, tool, {"tool_policy": policy})
    assert check is None or check(source_from_result(result))["status"] == "unverified"


async def test_recovery_check_does_not_mutate_persisted_source(tmp_path):
    from nanobot.agent.context_artifacts import render_active_context

    tool, path, result = await read_source(tmp_path)
    _, evidence = ToolDigestBuilder.build(
        tool_call_id="read", tool_name="read_file", arguments={"path": str(path)}, result=result,
    )
    metadata = {
        CONTEXT_STATE_KEY: ContextState(evidence={evidence.evidence_id: evidence}).to_metadata(),
        "goal_state": {"status": "active", "objective": "复核"},
    }
    before = deepcopy(metadata)
    path.write_bytes(b"updated")
    view = render_active_context(
        metadata, workspace=tmp_path, source_verifier=make_source_verifier(tmp_path, tool, {}),
    )
    assert '"status": "changed"' in view
    assert metadata == before
    # 摘要覆盖事务没有读取能力，不把上一次 fresh 状态当成重启后的事实。
    assert "unverified; re-read" in render_active_context(metadata, workspace=tmp_path)
