"""普通上下文策略的事务、原文保护、失败出口和显式回退。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.context_governance import ContextGovernanceConfig
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.config.schema import AgentDefaults
from nanobot.fork.agent.context_budget import ContextBudgetError
from nanobot.fork.agent.recovery_packet import RecoveryPacketError
from nanobot.fork.agent.summary_transaction import SummaryTransactionError
from nanobot.fork.agent.tool_evidence import EvidencePersistenceError
from nanobot.fork.agent.transactional_context import (
    TransactionalContextPreparation,
    assert_request_fits,
    request_summary,
    validate_strategy,
)
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


def exchange(i, text="结果"):
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": str(i), "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": str(i), "content": text},
    ]


def count(messages, tools=None, model=None):
    return len(json.dumps(messages, ensure_ascii=False)) // 4 + len(str(tools)), "test"


async def summarize(**kwargs):
    payload = json.loads(kwargs["messages"][1]["content"])
    return LLMResponse(content=json.dumps({
        "summary": "已检查旧证据；尚未实施。禁止提交，文件需复核。",
        "source_sha256": payload["source_sha256"],
    }, ensure_ascii=False), usage={"prompt_tokens": 12, "completion_tokens": 4})


def config(tmp_path):
    provider = Mock()
    provider.supports_native_context_compaction = False
    provider.estimate_prompt_tokens = count
    provider.chat_with_retry = AsyncMock(side_effect=summarize)
    return ContextGovernanceConfig(
        provider=provider, model="test", tools=SimpleNamespace(get_definitions=lambda: []),
        workspace=tmp_path, data_dir=tmp_path, session_key="cli:topic",
        max_tool_result_chars=16000, context_window_tokens=20000,
        context_block_limit=4000, max_tokens=1000,
    )


def source():
    return [
        {"role": "system", "content": "系统约束不变"},
        {"role": "user", "content": "只调查，不允许改代码，更不允许提交"},
        *exchange(1, "旧资料" * 4000),
        {"role": "user", "content": "继续检查，不扩展授权"},
        *exchange(2),
    ]


async def test_commit_is_durable_keeps_protected_rows_and_raw_transcript(tmp_path):
    cfg, messages = config(tmp_path), source()
    before = deepcopy(messages)
    tx = TransactionalContextPreparation()
    result = await tx.prepare(cfg, messages)
    assert messages == before and tx.version == 1
    assert [m for m in result if m["role"] in {"system", "user"}] == [
        m for m in before if m["role"] in {"system", "user"}
    ]
    assert result[-2:] == messages[-2:]
    assert tx.version_locator and Path(tx.version_locator).is_file()
    version = json.loads(Path(tx.version_locator).read_text(encoding="utf-8"))
    assert version["version"] == 1 and version["projected"] == result
    original = json.loads(Path(version["source_locator"]).read_text(encoding="utf-8"))
    assert original["messages"] == messages[2:4]
    assert cfg.provider.chat_with_retry.call_args.kwargs["tools"] is None
    assert tx.usage == {"prompt_tokens": 12, "completion_tokens": 4}
    again = await tx.prepare(cfg, messages + [{"role": "user", "content": "停止"}])
    assert tx.version == 1 and cfg.provider.chat_with_retry.await_count == 1
    assert again[-1]["content"] == "停止"


@pytest.mark.parametrize("failure", ["error", "length", "empty", "invalid", "hash", "tool", "exception"])
async def test_summary_failure_never_switches_context(tmp_path, failure):
    cfg, messages = config(tmp_path), source()
    tx = TransactionalContextPreparation()

    async def bad(**kwargs):
        if failure == "exception":
            raise OSError("网络失败")
        response = await summarize(**kwargs)
        if failure in {"error", "length"}:
            response.finish_reason = failure
        elif failure == "empty":
            response.content = ""
        elif failure == "invalid":
            response.content = '{"summary": ""}'
        elif failure == "hash":
            response.content = '{"summary":"声称允许提交","source_sha256":"forged"}'
        else:
            response.tool_calls = [Mock()]
        return response

    cfg.provider.chat_with_retry.side_effect = bad
    with pytest.raises(SummaryTransactionError):
        await tx.prepare(cfg, messages)
    assert tx.version == 0 and not tx.source and not tx.projected
    assert messages == source()


async def test_cancel_and_concurrent_edit_do_not_commit(tmp_path):
    cfg, messages = config(tmp_path), source()
    tx = TransactionalContextPreparation()
    cfg.provider.chat_with_retry.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await tx.prepare(cfg, messages)
    assert tx.version == 0

    async def change(**kwargs):
        messages.append({"role": "user", "content": "撤销授权"})
        return await summarize(**kwargs)

    cfg.provider.chat_with_retry.side_effect = change
    with pytest.raises(SummaryTransactionError, match="期间"):
        await tx.prepare(cfg, messages)
    assert tx.version == 0 and messages[-1]["content"] == "撤销授权"


async def test_version_write_failure_keeps_old_projection(tmp_path, monkeypatch):
    import nanobot.fork.agent.transactional_context as module

    cfg, messages = config(tmp_path), source()
    tx = TransactionalContextPreparation()
    original = module.persist_tool_evidence

    def fail_version(config, payload):
        if payload.get("kind") == "context_summary_version":
            raise EvidencePersistenceError("版本写入失败")
        return original(config, payload)

    monkeypatch.setattr(module, "persist_tool_evidence", fail_version)
    with pytest.raises(EvidencePersistenceError):
        await tx.prepare(cfg, messages)
    assert tx.version == 0 and not tx.source


async def test_no_storage_or_oversized_summary_input_does_not_call_model(tmp_path):
    cfg, tx = config(tmp_path), TransactionalContextPreparation()
    with pytest.raises(EvidencePersistenceError):
        await tx.prepare(replace(cfg, workspace=None, data_dir=None), source())
    cfg.provider.chat_with_retry.assert_not_called()
    huge = source()
    huge[2:4] = exchange(1, "大原文" * 30000)
    with pytest.raises(SummaryTransactionError, match="完整摘要输入"):
        await tx.prepare(cfg, huge)
    cfg.provider.chat_with_retry.assert_not_called()


@pytest.mark.parametrize("rows", [
    exchange(1)[:1], exchange(1)[1:],
    exchange(1)[:1] + [{"role": "user", "content": "新消息"}] + exchange(1)[1:],
])
async def test_incomplete_or_orphan_batches_stop_without_repair(tmp_path, rows):
    cfg = config(tmp_path)
    with pytest.raises(SummaryTransactionError):
        await TransactionalContextPreparation().prepare(cfg, rows)
    cfg.provider.chat_with_retry.assert_not_called()


async def test_no_fixed_soft_budget_or_64_exchange_eviction(tmp_path):
    cfg = replace(config(tmp_path), context_block_limit=100000, context_window_tokens=200000)
    messages = [{"role": "user", "content": "不得遗漏"}]
    for i in range(80):
        messages.extend(exchange(i, "证据" * 500))
    actual = await TransactionalContextPreparation().prepare(cfg, messages)
    assert actual == messages
    cfg.provider.chat_with_retry.assert_not_called()


async def test_native_summary_is_rejected_before_provider_call(tmp_path):
    cfg = config(tmp_path)
    cfg.provider.supports_native_context_compaction = True
    with pytest.raises(SummaryTransactionError, match="隔离摘要"):
        await request_summary(cfg.provider, model="test", messages=source(), max_tokens=1000)
    cfg.provider.chat_with_retry.assert_not_called()


def spec(cfg, messages, **kwargs):
    return AgentRunSpec(
        initial_messages=messages, tools=cfg.tools, model=cfg.model, max_iterations=2,
        max_tool_result_chars=cfg.max_tool_result_chars, workspace=cfg.workspace,
        data_dir=cfg.data_dir, session_key=cfg.session_key,
        context_window_tokens=cfg.context_window_tokens,
        context_block_limit=cfg.context_block_limit, max_tokens=cfg.max_tokens, **kwargs,
    )


async def test_runner_summary_failure_is_safety_stop_not_injection_retry(tmp_path):
    cfg = config(tmp_path)
    cfg.provider.chat_with_retry.side_effect = RuntimeError("摘要失败")
    injected = AsyncMock(return_value=[{"role": "user", "content": "继续"}])
    result = await AgentRunner(cfg.provider).run(spec(cfg, source(), injection_callback=injected))
    assert result.context_safety_failure and result.stop_reason == "error"
    assert cfg.provider.chat_with_retry.await_count == 1
    injected.assert_not_called()
    assert result.messages == source()


async def test_provider_overflow_never_snips_or_drains_injections(tmp_path):
    cfg = config(tmp_path)
    cfg.provider.chat_with_retry.side_effect = None
    cfg.provider.chat_with_retry.return_value = LLMResponse(
        content="maximum context length exceeded", finish_reason="error",
    )
    injected = AsyncMock()
    result = await AgentRunner(cfg.provider).run(spec(
        cfg, [{"role": "user", "content": "检查"}], injection_callback=injected,
    ))
    assert result.context_safety_failure
    assert cfg.provider.chat_with_retry.await_count == 1
    injected.assert_not_called()


async def test_injection_failure_is_not_swallowed(tmp_path):
    cfg = config(tmp_path)
    request = spec(cfg, [], injection_callback=AsyncMock(side_effect=OSError("失效")))
    with pytest.raises(RecoveryPacketError):
        await AgentRunner(cfg.provider)._drain_injections(request)


def test_full_replay_keeps_all_uncovered_rows_and_assistant_prefix():
    session = Session(key="cli:topic")
    session.messages = [{"role": "assistant", "content": "主动交付结果"}]
    for i in range(75):
        session.messages.extend([{"role": "user", "content": str(i)}, *exchange(i)])
    preserved = session.get_history(max_messages=4, max_tokens=1, preserve_unconsolidated=True)
    assert len(preserved) == len(session.messages)
    assert preserved[0]["content"] == "主动交付结果"
    assert len(session.get_history(max_messages=4, max_tokens=1)) < len(preserved)


def test_configuration_and_emergency_request_gate(tmp_path):
    assert AgentDefaults().context_strategy == "transactional"
    assert AgentDefaults(contextStrategy="legacy").context_strategy == "legacy"
    with pytest.raises(ValueError):
        AgentDefaults(contextStrategy="auto")
    with pytest.raises(ContextBudgetError):
        validate_strategy("auto")
    cfg = config(tmp_path)
    request = spec(cfg, [])
    huge = [{"role": "user", "content": "新约束" * 10000}]
    with pytest.raises(ContextBudgetError):
        assert_request_fits(cfg.provider, request, huge, [])
    assert_request_fits(cfg.provider, replace(request, context_strategy="legacy"), huge, [])


async def test_runner_complete_batches_then_summary_then_new_user_input(tmp_path):
    from nanobot.providers.base import ToolCallRequest

    cfg = config(tmp_path)
    cfg.tools = Mock()
    cfg.tools.get_definitions.return_value = []
    cfg.tools.execute = AsyncMock(side_effect=["旧资料" * 4000, "新证据"])
    main_calls = []
    summary_done = False
    injected = False

    async def chat(**kwargs):
        nonlocal summary_done
        if kwargs["messages"][0].get("content", "").startswith("你是无工具"):
            summary_done = True
            return await summarize(**kwargs)
        main_calls.append(deepcopy(kwargs["messages"]))
        if len(main_calls) <= 2:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id=str(len(main_calls)), name="read_file", arguments={"path": "fixture"},
            )])
        return LLMResponse(content="已停止，不提交", usage={"prompt_tokens": 10, "completion_tokens": 2})

    async def inject():
        nonlocal injected
        if summary_done and not injected:
            injected = True
            return [{"role": "user", "content": "撤销继续执行；立即停止", "_input_evidence_id": "I"}]
        return []

    cfg.provider.chat_with_retry.side_effect = chat
    result = await AgentRunner(cfg.provider).run(replace(
        spec(cfg, [{"role": "user", "content": "只检查，不提交"}], injection_callback=inject),
        max_iterations=4,
    ))
    assert result.stop_reason == "completed" and result.final_content == "已停止，不提交"
    assert len(main_calls) == 3 and cfg.tools.execute.await_count == 2
    assert main_calls[-1][-1]["content"] == "撤销继续执行；立即停止"
    assert any("历史摘要" in str(row.get("content")) for row in main_calls[-1])
    assert next(row["content"] for row in result.messages if row.get("tool_call_id") == "1") == "旧资料" * 4000
    assert result.usage["completion_tokens"] >= 6


async def test_summary_timeout_and_later_failure_keep_committed_version(tmp_path):
    cfg = config(tmp_path)

    async def slow(**kwargs):
        await asyncio.sleep(10)

    cfg.provider.chat_with_retry.side_effect = slow
    with pytest.raises(asyncio.TimeoutError):
        await request_summary(cfg.provider, model="test", messages=[], max_tokens=10, timeout=0.01)
    cfg.provider.chat_with_retry.side_effect = summarize
    tx = TransactionalContextPreparation()
    await tx.prepare(cfg, source())
    old = (tx.version, deepcopy(tx.projected), tx.version_locator)
    changed = source() + exchange(3, "又一份资料" * 4000) + exchange(4)
    cfg.provider.chat_with_retry.side_effect = OSError("第二版失败")
    with pytest.raises(SummaryTransactionError):
        await tx.prepare(cfg, changed)
    assert (tx.version, tx.projected, tx.version_locator) == old


async def test_finalization_budget_guard_does_not_call_provider(tmp_path):
    cfg = config(tmp_path)
    runner = AgentRunner(cfg.provider)
    with pytest.raises(ContextBudgetError):
        await runner._request_no_tools(spec(cfg, []), [{"role": "user", "content": "约束" * 20000}])
    cfg.provider.chat_with_retry.assert_not_called()


def test_new_persistence_keeps_long_receipt_and_user_marker_text(tmp_path):
    from nanobot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    loop.max_tool_result_chars = 64
    loop.context_strategy = "transactional"
    session = Session(key="cli:topic")
    raw = "全部证据" * 500
    rows = exchange(1, raw) + [{
        "role": "user", "content": "[Runtime Context]\n\n不得提交", "_input_evidence_id": "input-real",
    }]
    loop._save_turn(session, rows, 0)
    assert session.messages[1]["content"] == raw
    assert session.messages[2]["content"] == rows[-1]["content"]
    assert session.messages[2]["_input_evidence_id"] == "input-real"


async def test_pending_input_evidence_survives_restart_and_failure_requeues_in_order(tmp_path, monkeypatch):
    from nanobot.agent.context_artifacts import ContextState
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.fork.agent.input_evidence import capture_input

    cfg = config(tmp_path)
    cfg.provider.get_default_model.return_value = "test"
    loop = AgentLoop(bus=MessageBus(), provider=cfg.provider, workspace=tmp_path, enable_learning=False)
    session = loop.sessions.get_or_create("cli:topic")
    pending = asyncio.Queue()
    for text in ["只检查", "撤销"]:
        pending.put_nowait(capture_input(InboundMessage(
            channel="cli", sender_id="user", chat_id="topic", content=text,
        ), "cli_interactive"))
    seen = []

    async def run(request):
        rows = await request.injection_callback(limit=1)
        seen.extend(rows)
        from nanobot.agent.runner import AgentRunResult
        return AgentRunResult(final_content="ok", messages=[*request.initial_messages, *rows])

    loop.runner.run = run
    await loop._run_agent_loop(
        [{"role": "user", "content": "初始"}], session=session, pending_queue=pending,
    )
    loop.sessions.invalidate(session.key)
    loaded = loop.sessions.get_or_create(session.key)
    evidence_id = seen[0]["_input_evidence_id"]
    evidence = ContextState.from_metadata(loaded.metadata).evidence[evidence_id]
    assert json.loads(Path(evidence.locator).read_text(encoding="utf-8"))["text"] == "只检查"

    # 保存失败后整批重新入队，原有队尾不能跑到撤销消息之前。
    pending.put_nowait(InboundMessage(channel="cli", sender_id="u", chat_id="topic", content="后续"))
    monkeypatch.setattr(loop.sessions, "save", Mock(side_effect=OSError("存储故障")))
    with pytest.raises(OSError):
        await loop._run_agent_loop(
            [{"role": "user", "content": "初始"}], session=loaded, pending_queue=pending,
        )
    assert [pending.get_nowait().content, pending.get_nowait().content] == ["撤销", "后续"]


@pytest.mark.parametrize("change", ["state", "tools"])
async def test_summary_rejects_stale_task_state_or_tool_capabilities(tmp_path, change):
    cfg = config(tmp_path)
    state = {"constraint": "只读", "todos": ["检查"]}
    tx = TransactionalContextPreparation(lambda: state)

    async def changed(**kwargs):
        if change == "state":
            state["todos"].append("停止")
        else:
            cfg.tools.get_definitions = lambda: [{"name": "new_tool"}]
        return await summarize(**kwargs)

    cfg.provider.chat_with_retry.side_effect = changed
    with pytest.raises(SummaryTransactionError, match="任务状态或工具能力"):
        await tx.prepare(cfg, source())
    assert tx.version == 0


async def test_latest_oversized_batch_stops_without_discarding_raw_result(tmp_path):
    cfg = config(tmp_path)
    rows = [{"role": "user", "content": "检查"}] + exchange(1, "完整结果" * 10000)
    before = deepcopy(rows)
    tx = TransactionalContextPreparation()
    with pytest.raises(ContextBudgetError, match="最新完整工具批次"):
        await tx.prepare(cfg, rows)
    assert rows == before and tx.version == 0
    cfg.provider.chat_with_retry.assert_not_called()


async def test_unexpected_governance_error_cannot_fall_back_to_raw_request(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(TransactionalContextPreparation, "prepare", AsyncMock(side_effect=TypeError("故障")))
    result = await AgentRunner(cfg.provider).run(spec(cfg, [{"role": "user", "content": "检查"}]))
    assert result.context_safety_failure and result.stop_reason == "error"
    cfg.provider.chat_with_retry.assert_not_called()
