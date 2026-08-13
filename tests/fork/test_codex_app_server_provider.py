from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from nanobot.config.schema import Config
from nanobot.fork.providers.codex_app_server_provider import (
    CodexAppServerError,
    CodexAppServerProvider,
    CodexIdempotencyLedgerError,
    _apply_tool_choice,
    _canonical_tool_arguments,
    _codex_subprocess_env,
    _CodexAppServerTurn,
    _convert_dynamic_tools,
    _current_turn_messages,
    _idempotency_ledger,
    _map_token_usage,
    _messages_to_app_server_input,
    _skill_requires_unavailable_tools,
    _strip_codex_model_prefix,
    _tool_call_from_server_request,
    _tool_result_content_items,
    _tool_results_by_id,
    _usage_delta,
)
from nanobot.providers.factory import make_provider
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)

_FAKE_APP_SERVER = r"""
import json
import sys
from pathlib import Path

mode = sys.argv[1]
state_path = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None

def send(message):
    print(json.dumps(message, separators=(",", ":")), flush=True)

for raw_line in sys.stdin:
    message = json.loads(raw_line)
    method = message.get("method")
    if method == "initialize":
        if mode == "hang":
            continue
        send({"id": message["id"], "result": {}})
    elif method == "skills/list":
        skills = []
        if mode == "capture_skills" and state_path is not None:
            skills.append({
                "name": "browser:control-in-app-browser",
                "description": "browser",
                "path": str(state_path / "SKILL.md"),
                "scope": "user",
                "enabled": True,
            })
        send({
            "id": message["id"],
            "result": {"data": [{"cwd": message["params"]["cwds"][0], "skills": skills, "errors": []}]},
        })
    elif method == "thread/start":
        params = message["params"]
        config = params.get("config", {})
        if mode == "capture_sandbox" and state_path is not None:
            state_path.write_text(json.dumps(params), encoding="utf-8")
        if mode == "capture_skills" and state_path is not None:
            (state_path / "params.json").write_text(json.dumps(params), encoding="utf-8")
        full = (
            params.get("sandbox") == "danger-full-access"
            and "runtimeWorkspaceRoots" not in params
            and "sandbox_workspace_write" not in config
        )
        isolated = (
            config.get("web_search") == "disabled"
            and config.get("mcp_servers") == {}
            and config.get("plugins") == {}
            and full
            and params.get("approvalPolicy") == "never"
        )
        if not isolated:
            send({"id": message["id"], "error": {"code": -1, "message": "not isolated"}})
        else:
            send({"id": message["id"], "result": {"thread": {"id": "thread-1"}}})
    elif method == "turn/start":
        if mode == "capture_skills" and state_path is not None:
            (state_path / "turn.json").write_text(
                json.dumps(message["params"]), encoding="utf-8"
            )
        send({"id": message["id"], "result": {"turn": {"id": "turn-1"}}})
        if mode == "hang_turn":
            continue
        if mode in {"answer", "large_event", "capture_sandbox", "capture_skills"}:
            text = "x" * (128 * 1024) if mode == "large_event" else "done"
            send({
                "method": "item/completed",
                "params": {"item": {"id": "answer-1", "type": "agentMessage", "text": text}},
            })
            send({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn-1", "status": "completed"}},
            })
        elif mode == "image_view" or (
            mode == "image_after_tool" and state_path is not None and state_path.exists()
        ):
            send({
                "method": "item/started",
                "params": {
                    "item": {
                        "id": "image-1",
                        "type": "imageView",
                        "path": "reference.png",
                    }
                },
            })
            send({
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": "image-1",
                        "type": "imageView",
                        "path": "reference.png",
                    }
                },
            })
            send({
                "method": "item/completed",
                "params": {
                    "item": {"id": "answer-1", "type": "agentMessage", "text": "done"}
                },
            })
            send({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn-1", "status": "completed"}},
            })
        elif mode in {"unsupported_native", "unsupported_native_once"}:
            already_blocked = state_path is not None and state_path.exists()
            if mode == "unsupported_native_once" and already_blocked:
                send({
                    "method": "item/completed",
                    "params": {
                        "item": {"id": "answer-1", "type": "agentMessage", "text": "done"}
                    },
                })
                send({
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-1", "status": "completed"}},
                })
                continue
            if mode == "unsupported_native_once" and state_path is not None:
                state_path.write_text("blocked", encoding="utf-8")
            send({
                "method": "item/started",
                "params": {"item": {"id": "native-1", "type": "mcpToolCall"}},
            })
        elif mode in {"native", "native_failed", "native_approval"}:
            send({
                "method": "item/started",
                "params": {"item": {
                    "id": "patch-1",
                    "type": "fileChange",
                    "status": "inProgress",
                    "changes": [{"path": "example.txt", "kind": "update", "diff": "@@"}],
                }},
            })
            if mode == "native_approval":
                send({
                    "id": 700,
                    "method": "item/fileChange/requestApproval",
                    "params": {
                        "itemId": "patch-1",
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "startedAtMs": 1,
                        "grantRoot": "outside-workspace",
                    },
                })
                continue
            send({
                "method": "item/completed",
                "params": {"item": {
                    "id": "patch-1",
                    "type": "fileChange",
                    "status": "failed" if mode == "native_failed" else "completed",
                    "changes": [{"path": "example.txt", "kind": "update", "diff": "@@"}],
                }},
            })
            send({
                "method": "item/completed",
                "params": {"item": {"id": "answer-1", "type": "agentMessage", "text": "done"}},
            })
            send({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn-1", "status": "completed"}},
            })
        elif mode in {"native_command", "native_command_failed", "native_command_approval"}:
            send({
                "method": "item/started",
                "params": {"item": {
                    "id": "command-1",
                    "type": "commandExecution",
                    "status": "inProgress",
                    "command": "git status --short",
                    "cwd": ".",
                    "commandActions": [],
                }},
            })
            if mode == "native_command_approval":
                send({
                    "id": 701,
                    "method": "item/commandExecution/requestApproval",
                    "params": {
                        "itemId": "command-1",
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "startedAtMs": 1,
                        "reason": "network access",
                    },
                })
                continue
            send({
                "method": "item/commandExecution/outputDelta",
                "params": {"itemId": "command-1", "delta": "output"},
            })
            send({
                "method": "item/completed",
                "params": {"item": {
                    "id": "command-1",
                    "type": "commandExecution",
                    "status": "failed" if mode == "native_command_failed" else "completed",
                    "command": "git status --short",
                    "cwd": ".",
                    "commandActions": [],
                    "aggregatedOutput": "output",
                    "exitCode": 1 if mode == "native_command_failed" else 0,
                }},
            })
            send({
                "method": "item/completed",
                "params": {"item": {"id": "answer-1", "type": "agentMessage", "text": "done"}},
            })
            send({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn-1", "status": "completed"}},
            })
        else:
            send({
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 8,
                            "cachedInputTokens": 2,
                            "outputTokens": 2,
                            "reasoningOutputTokens": 0,
                            "totalTokens": 10,
                        },
                        "total": {
                            "inputTokens": 8,
                            "cachedInputTokens": 2,
                            "outputTokens": 2,
                            "reasoningOutputTokens": 0,
                            "totalTokens": 10,
                        },
                    }
                },
            })
            recovered = state_path is not None and state_path.exists()
            recovered_new_tool = mode == "recover_new_tool" and recovered
            send({
                "id": 900,
                "method": "item/tool/call",
                "params": {
                    "callId": "call-new" if recovered_new_tool else (
                        "call-recovered" if recovered else "call-1"
                    ),
                    "tool": "write",
                    "arguments": {"value": 2 if recovered_new_tool else 1},
                },
            })
            if mode == "crash_before_result":
                print("fatal token=sk-super-secret-value", file=sys.stderr, flush=True)
                raise SystemExit(16)
    elif message.get("id") == 700 and "result" in message:
        if message["result"].get("decision") != "decline":
            raise SystemExit(19)
        send({
            "method": "item/completed",
            "params": {"item": {
                "id": "patch-1", "type": "fileChange", "status": "declined", "changes": []
            }},
        })
        send({
            "method": "item/completed",
            "params": {"item": {"id": "answer-1", "type": "agentMessage", "text": "done"}},
        })
        send({
            "method": "turn/completed",
            "params": {"turn": {"id": "turn-1", "status": "completed"}},
        })
    elif message.get("id") == 701 and "result" in message:
        if message["result"].get("decision") != "decline":
            raise SystemExit(20)
        send({
            "method": "item/completed",
            "params": {"item": {
                "id": "command-1",
                "type": "commandExecution",
                "status": "declined",
                "command": "network command",
                "cwd": ".",
                "commandActions": [],
            }},
        })
        send({
            "method": "item/completed",
            "params": {"item": {"id": "answer-1", "type": "agentMessage", "text": "done"}},
        })
        send({
            "method": "turn/completed",
            "params": {"turn": {"id": "turn-1", "status": "completed"}},
        })
    elif message.get("id") in {900, 901} and "result" in message:
        if mode == "crash_after_tool":
            print("fatal token=sk-super-secret-value", file=sys.stderr, flush=True)
            raise SystemExit(17)
        if mode in {"recover_once", "recover_stream", "recover_new_tool"} and state_path is not None:
            if not state_path.exists():
                if mode == "recover_stream":
                    send({"method": "item/agentMessage/delta", "params": {"delta": "partial"}})
                state_path.write_text("crashed", encoding="utf-8")
                print("one-time bridge crash", file=sys.stderr, flush=True)
                raise SystemExit(18)
        if mode == "repeat_normal" and message.get("id") == 900:
            send({
                "id": 901,
                "method": "item/tool/call",
                "params": {"callId": "call-2", "tool": "write", "arguments": {"value": 1}},
            })
            continue
        if mode == "image_after_tool" and state_path is not None:
            state_path.write_text("image", encoding="utf-8")
            send({
                "method": "item/started",
                "params": {
                    "item": {
                        "id": "image-1",
                        "type": "imageView",
                        "path": "reference.png",
                    }
                },
            })
            continue
        send({
            "method": "thread/tokenUsage/updated",
            "params": {
                "tokenUsage": {
                    "last": {
                        "inputTokens": 5,
                        "cachedInputTokens": 1,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 1,
                        "totalTokens": 8,
                    },
                    "total": {
                        "inputTokens": 13,
                        "cachedInputTokens": 3,
                        "outputTokens": 5,
                        "reasoningOutputTokens": 1,
                        "totalTokens": 18,
                    },
                }
            },
        })
        send({
            "method": "item/completed",
            "params": {"item": {"id": "answer-1", "type": "agentMessage", "text": "done"}},
        })
        send({
            "method": "turn/completed",
            "params": {"turn": {"id": "turn-1", "status": "completed"}},
        })
"""


def _fake_command(mode: str, state_path: Path | None = None) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-c",
        _FAKE_APP_SERVER,
        mode,
        str(state_path) if state_path is not None else "",
    ]


def _tool_schema() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "side effect",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
            },
        }
    ]


def _tool_schema_with_read_file() -> list[dict[str, object]]:
    return [
        *_tool_schema(),
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a workspace file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def test_convert_dynamic_tools_accepts_openai_and_flat_schemas() -> None:
    converted = _convert_dynamic_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "name": "todo_write",
                "description": "Update todos",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
    )
    assert converted[0]["type"] == "namespace"
    assert converted[0]["name"] == "nanobot"
    namespace_tools = converted[0]["tools"]
    assert namespace_tools[0]["name"] == "exec"
    assert namespace_tools[0]["inputSchema"]["required"] == ["command"]
    assert namespace_tools[1] == {
        "type": "function",
        "name": "todo_write",
        "description": "Update todos",
        "inputSchema": {"type": "object", "properties": {}},
    }


def test_convert_dynamic_tools_omits_empty_namespace() -> None:
    assert _convert_dynamic_tools(None) == []
    assert _convert_dynamic_tools([]) == []


def test_messages_become_instructions_history_and_latest_user_input() -> None:
    instructions, turn_input = _messages_to_app_server_input(
        [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Latest question"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
        ]
    )
    assert instructions == "System rules"
    assert "user: Earlier question" in turn_input[0]["text"]
    assert "assistant: Earlier answer" in turn_input[0]["text"]
    assert turn_input[1] == {"type": "text", "text": "Latest question"}
    assert turn_input[2] == {"type": "image", "url": "data:image/png;base64,AAA"}


def test_recovery_input_includes_completed_current_turn_history() -> None:
    _instructions, turn_input = _messages_to_app_server_input(
        [
            {"role": "user", "content": "write once"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"value":1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ],
        include_current_turn_history=True,
    )
    checkpoint = turn_input[-1]["text"]
    assert "Recovery checkpoint" in checkpoint
    assert "call-1" in checkpoint
    assert "tool [call-1]: ok" in checkpoint


def test_history_omits_content_read_from_disabled_skill() -> None:
    skill_path = r"C:\Users\test\.codex\plugins\browser\skills\control\SKILL.md"
    messages = [
        {"role": "user", "content": "test the browser"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-browser-skill",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": skill_path.replace("\\", "/")}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read-browser-skill",
            "name": "read_file",
            "content": "You MUST call mcp__node_repl__js now.",
        },
        {"role": "user", "content": "continue"},
    ]


    _instructions, turn_input = _messages_to_app_server_input(
        messages,
        disabled_skill_paths={skill_path},
    )

    transcript = turn_input[0]["text"]
    assert "mcp__node_repl__js" not in transcript
    assert "disabled because its required tools are unavailable" in transcript
    assert "read-browser-skill" in transcript


def test_tool_argument_canonicalization_ignores_json_key_order() -> None:
    assert _canonical_tool_arguments('{"b":2,"a":1}') == _canonical_tool_arguments({"a": 1, "b": 2})


def test_ledger_keeps_original_result_after_context_compaction(tmp_path: Path) -> None:
    ledger = _idempotency_ledger(
        ("session", "compacted-result"),
        root_override=tmp_path / "ledger",
    )
    call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "exec", "arguments": {"command": "git status"}},
            }
        ],
    }
    ledger.record_messages(
        [
            call,
            {"role": "tool", "tool_call_id": "call-1", "content": "original result"},
        ]
    )

    ledger.record_messages(
        [
            call,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "[ToolDigest td_123 status=ok]",
            },
        ]
    )

    assert ledger.entries[0]["content"] == "original result"


def test_ledger_rejects_reused_call_id_with_different_signature(tmp_path: Path) -> None:
    ledger = _idempotency_ledger(
        ("session", "conflicting-signature"),
        root_override=tmp_path / "ledger",
    )
    ledger.record_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "exec", "arguments": {"command": "git status"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
    )

    with pytest.raises(CodexIdempotencyLedgerError, match="Conflicting"):
        ledger.record_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "exec", "arguments": {"command": "git diff"}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            ]
        )

def test_recovery_replays_repeated_signature_in_original_order(tmp_path: Path) -> None:
    ledger = _idempotency_ledger(
        ("session", "ambiguous"),
        root_override=tmp_path / "ledger",
    )
    ledger.record_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "write", "arguments": '{"value":1}'},
                    },
                    {
                        "id": "call-2",
                        "function": {"name": "write", "arguments": {"value": 1}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first"},
            {"role": "tool", "tool_call_id": "call-2", "content": "second"},
        ]
    )
    ledger.begin_recovery()
    first_recovered_call = _tool_call_from_server_request(
        {
            "id": 60,
            "method": "item/tool/call",
            "params": {
                "callId": "new-call-id",
                "tool": "write",
                "arguments": {"value": 1},
            },
        }
    )
    second_recovered_call = _tool_call_from_server_request(
        {
            "id": 61,
            "method": "item/tool/call",
            "params": {
                "callId": "another-new-call-id",
                "tool": "write",
                "arguments": {"value": 1},
            },
        }
    )

    assert ledger.cached_result(first_recovered_call) == ("first", True)
    assert ledger.cached_result(second_recovered_call) == ("second", True)
    with pytest.raises(CodexIdempotencyLedgerError, match="repeated again"):
        ledger.cached_result(second_recovered_call)


def test_recovery_fails_safe_when_tool_order_diverges(tmp_path: Path) -> None:
    ledger = _idempotency_ledger(
        ("session", "order-diverged"),
        root_override=tmp_path / "ledger",
    )
    ledger.record_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "read", "arguments": {"path": "a"}},
                    },
                    {
                        "id": "call-2",
                        "function": {"name": "write", "arguments": {"path": "b"}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first"},
            {"role": "tool", "tool_call_id": "call-2", "content": "second"},
        ]
    )
    ledger.begin_recovery()
    out_of_order_call = _tool_call_from_server_request(
        {
            "id": 62,
            "method": "item/tool/call",
            "params": {
                "callId": "new-call-id",
                "tool": "write",
                "arguments": {"path": "b"},
            },
        }
    )

    with pytest.raises(CodexIdempotencyLedgerError, match="order diverged"):
        ledger.cached_result(out_of_order_call)


def test_current_turn_messages_excludes_older_tool_results(tmp_path: Path) -> None:
    messages = [
        {"role": "user", "content": "older"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "old-call",
                    "function": {"name": "write", "arguments": {"value": "old"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old-call", "content": "old result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "new-call",
                    "function": {"name": "write", "arguments": {"value": "new"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "new-call", "content": "new result"},
    ]

    turn_messages = _current_turn_messages(messages)

    assert turn_messages[0] == {"role": "user", "content": "current"}
    assert _tool_results_by_id(turn_messages) == {"new-call": "new result"}
    ledger = _idempotency_ledger(
        ("session", "current-turn-only"),
        root_override=tmp_path / "ledger",
    )
    ledger.record_messages(turn_messages)
    assert [entry["call_id"] for entry in ledger.entries] == ["new-call"]


def test_ledger_write_cleans_stale_json_and_temp_files(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    stale_json = ledger_dir / "stale.json"
    stale_tmp = ledger_dir / ".stale.tmp"
    for path in (stale_json, stale_tmp):
        path.write_text("stale", encoding="utf-8")
        os.utime(path, (1, 1))
    ledger = _idempotency_ledger(
        ("session", "cleanup"),
        root_override=ledger_dir,
    )
    ledger.record_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "write", "arguments": {"value": 1}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]
    )
    assert not stale_json.exists()
    assert not stale_tmp.exists()
    assert ledger.path.exists()


def test_dynamic_tool_request_and_results_keep_call_id() -> None:
    call = _tool_call_from_server_request(
        {
            "id": 60,
            "method": "item/tool/call",
            "params": {
                "callId": "call_123",
                "tool": "exec",
                "arguments": {"command": "Write-Output ok"},
            },
        }
    )
    results = _tool_results_by_id(
        [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "exec",
                "content": "ok",
            }
        ]
    )
    assert call.id == "call_123"
    assert call.name == "exec"
    assert call.arguments == {"command": "Write-Output ok"}
    assert results == {"call_123": "ok"}


def test_dynamic_tool_result_preserves_multimodal_content() -> None:
    content = [
        {"type": "text", "text": "chart"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
            "_meta": {"path": "chart.png"},
        },
    ]
    results = _tool_results_by_id(
        [{"role": "tool", "tool_call_id": "call-image", "content": content}]
    )

    assert results == {"call-image": content}
    assert _tool_result_content_items(results["call-image"]) == [
        {"type": "inputText", "text": "chart"},
        {"type": "inputImage", "imageUrl": "data:image/png;base64,AAAA"},
    ]


def test_ledger_restores_multimodal_tool_result(tmp_path: Path) -> None:
    ledger = _idempotency_ledger(
        ("session", "image-recovery"),
        root_override=tmp_path / "ledger",
    )
    content = [
        {"type": "text", "text": "chart"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    ledger.record_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-image",
                        "function": {"name": "read_file", "arguments": {"path": "chart.png"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-image", "content": content},
        ]
    )

    restored = _idempotency_ledger(
        ("session", "image-recovery"),
        root_override=tmp_path / "ledger",
    )
    cached = restored.cached_result(
        _tool_call_from_server_request(
            {
                "id": 60,
                "method": "item/tool/call",
                "params": {
                    "callId": "call-image",
                    "tool": "read_file",
                    "arguments": {"path": "chart.png"},
                },
            }
        )
    )
    assert cached == (content, True)


async def test_submit_dynamic_tool_result_sends_image_content_item() -> None:
    turn = _CodexAppServerTurn(["unused"])
    turn._pending_tools["call-image"] = 42
    sent: list[dict] = []

    async def capture(message: dict) -> None:
        sent.append(message)

    turn._send = capture  # type: ignore[method-assign]
    await turn.submit_tool_results(
        [
            {
                "role": "tool",
                "tool_call_id": "call-image",
                "content": [
                    {"type": "text", "text": "chart"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]
    )

    assert sent == [
        {
            "id": 42,
            "result": {
                "contentItems": [
                    {"type": "inputText", "text": "chart"},
                    {"type": "inputImage", "imageUrl": "data:image/png;base64,AAAA"},
                ],
                "success": True,
            },
        }
    ]


def test_usage_and_model_mapping() -> None:
    usage = _map_token_usage(
        {
            "tokenUsage": {
                "last": {
                    "inputTokens": 100,
                    "cachedInputTokens": 80,
                    "outputTokens": 12,
                    "reasoningOutputTokens": 3,
                    "totalTokens": 112,
                }
            }
        }
    )
    assert _strip_codex_model_prefix("openai-codex/gpt-5.6-sol") == "gpt-5.6-sol"
    assert _strip_codex_model_prefix("gpt-5.6-sol") == "gpt-5.6-sol"
    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "total_tokens": 112,
        "cached_tokens": 80,
        "reasoning_tokens": 3,
    }


def test_usage_mapping_prefers_thread_total_and_reports_only_delta() -> None:
    first = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "cached_tokens": 80,
    }
    second = _map_token_usage(
        {
            "tokenUsage": {
                "last": {
                    "inputTokens": 30,
                    "cachedInputTokens": 20,
                    "outputTokens": 5,
                    "reasoningOutputTokens": 1,
                    "totalTokens": 35,
                },
                "total": {
                    "inputTokens": 130,
                    "cachedInputTokens": 100,
                    "outputTokens": 15,
                    "reasoningOutputTokens": 1,
                    "totalTokens": 145,
                },
            }
        }
    )
    assert second["total_tokens"] == 145
    assert _usage_delta(second, first) == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
        "cached_tokens": 20,
        "reasoning_tokens": 1,
    }


def test_tool_choice_filters_dynamic_tools() -> None:
    tools = _tool_schema()
    assert _apply_tool_choice(tools, "none")[0] == []
    selected, instruction = _apply_tool_choice(
        tools,
        {"type": "function", "function": {"name": "write"}},
    )
    assert selected == tools
    assert "write" in (instruction or "")


def test_skill_dependency_scan_detects_missing_dynamic_tool(tmp_path: Path) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("Use `mcp__node_repl__js` to control the browser.", encoding="utf-8")
    skill = {"path": str(skill_file), "enabled": True}

    assert _skill_requires_unavailable_tools(skill, {"read_file"}) is True
    assert _skill_requires_unavailable_tools(skill, {"mcp__node_repl__js"}) is False

    declared = {
        "path": str(tmp_path / "missing.md"),
        "dependencies": {"tools": [{"type": "mcp", "value": "docs"}]},
    }
    assert _skill_requires_unavailable_tools(declared, {"read_file"}) is True
    assert _skill_requires_unavailable_tools(declared, {"mcp__docs__search"}) is False


async def test_app_server_hides_skill_with_unavailable_dynamic_tool(tmp_path: Path) -> None:
    skill_dir = tmp_path / "browser-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "Use `mcp__node_repl__js` to control the browser.",
        encoding="utf-8",
    )
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("capture_skills", skill_dir)

    response = await provider.chat(
        messages=[
            {"role": "user", "content": "test the browser"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "read-browser-skill",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": str(skill_dir / "SKILL.md")}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "read-browser-skill",
                "name": "read_file",
                "content": "You MUST call mcp__node_repl__js now.",
            },
            {"role": "user", "content": "continue"},
        ],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": "skill-filter"},
    )

    assert response.content == "done"
    params = json.loads((skill_dir / "params.json").read_text(encoding="utf-8"))
    assert params["config"]["skills"] == {
        "config": [{"path": str(skill_dir / "SKILL.md"), "enabled": False}]
    }
    turn = json.loads((skill_dir / "turn.json").read_text(encoding="utf-8"))
    rendered_input = json.dumps(turn["input"], ensure_ascii=False)
    assert "mcp__node_repl__js" not in rendered_input
    assert "disabled because its required tools are unavailable" in rendered_input


def test_explicit_proxy_is_passed_only_to_codex_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://environment-proxy")
    assert _codex_subprocess_env(None) is None
    env = _codex_subprocess_env("http://provider-proxy")
    assert env is not None
    assert env["HTTP_PROXY"] == "http://provider-proxy"
    assert env["HTTPS_PROXY"] == "http://provider-proxy"
    assert env["ALL_PROXY"] == "http://provider-proxy"


@pytest.mark.parametrize("access_mode", ["restricted", "full"])
async def test_app_server_always_uses_full_access_sandbox(
    tmp_path: Path,
    access_mode: str,
) -> None:
    state_path = tmp_path / f"{access_mode}.json"
    project = tmp_path / "project"
    project.mkdir()
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("capture_sandbox", state_path)
    scope = build_workspace_scope(project, access_mode, source_channel="cli")
    token = bind_workspace_scope(scope)
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": "inspect sibling projects"}],
            tools=_tool_schema(),
            request_context={"session_key": access_mode, "turn_id": access_mode},
        )
    finally:
        reset_workspace_scope(token)

    assert response.content == "done"
    params = json.loads(state_path.read_text(encoding="utf-8"))
    config = params["config"]
    assert params["cwd"] == str(project.resolve())
    assert params["sandbox"] == "danger-full-access"
    assert "runtimeWorkspaceRoots" not in params
    assert "sandbox_workspace_write" not in config


async def test_stale_cached_codex_command_is_resolved_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    stale_command = [str(tmp_path / "removed-codex.exe"), "app-server", "--stdio"]
    fresh_command = _fake_command("answer")
    provider._app_server_command = stale_command
    resolutions = 0

    def resolve_again() -> list[str]:
        nonlocal resolutions
        resolutions += 1
        return fresh_command

    monkeypatch.setattr(
        "nanobot.fork.providers.codex_app_server_provider._resolve_codex_app_server_command",
        resolve_again,
    )

    response = await provider.chat(
        messages=[{"role": "user", "content": "continue"}],
        tools=None,
        request_context={"session_key": "session", "turn_id": "stale-command"},
    )

    assert response.content == "done"
    assert response.provider_diagnostics["app_server_command_refreshes"] == 1
    assert provider._app_server_command == fresh_command
    assert resolutions == 1


async def test_stdio_protocol_continues_tool_result_and_usage_is_incremental(
    tmp_path: Path,
) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("complete")
    context = {"session_key": "session", "turn_id": "turn"}
    messages = [{"role": "user", "content": "write once"}]

    first = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert first.finish_reason == "tool_calls"
    assert first.usage["total_tokens"] == 10

    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "write",
                "content": "ok",
            },
        ]
    )
    second = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert second.content == "done"
    assert second.usage["total_tokens"] == 8
    assert first.usage["total_tokens"] + second.usage["total_tokens"] == 18
    assert provider._turns == {}


async def test_stdio_protocol_accepts_json_lines_larger_than_asyncio_default(
    tmp_path: Path,
) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("large_event")

    response = await provider.chat(
        messages=[{"role": "user", "content": "large response"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": "large-event"},
    )

    assert response.finish_reason == "stop"
    assert response.content == "x" * (128 * 1024)


@pytest.mark.parametrize("mode", ["crash_before_result", "crash_after_tool"])
async def test_failure_after_tool_execution_is_not_retried_or_replayed(
    mode: str, tmp_path: Path
) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command(mode)
    context = {"session_key": "session", "turn_id": "turn"}
    messages = [{"role": "user", "content": "write once"}]
    first = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "write",
                "content": "ok",
            },
        ]
    )

    failed = await provider.chat_with_retry(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert failed.finish_reason == "error"
    assert failed.error_should_retry is False
    assert failed.provider_diagnostics["retry_suppressed_after_tool_result"] is True
    assert "[REDACTED]" in (failed.content or "")
    assert "sk-super-secret-value" not in (failed.content or "")


async def test_bridge_recovers_once_and_replays_cached_result(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    marker = tmp_path / "crashed-once"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    provider._app_server_command = _fake_command("recover_once", marker)
    context = {"session_key": "session", "turn_id": "recover"}
    messages = [{"role": "user", "content": "write once"}]
    first = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "write",
                "content": "ok",
            },
        ]
    )

    recovered = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert recovered.content == "done"
    assert recovered.usage["total_tokens"] == 18
    assert recovered.provider_diagnostics["bridge_recovery_attempts"] == 1
    assert recovered.provider_diagnostics["idempotent_tool_replays"] == 1
    assert list(ledger_dir.glob("*.json")) == []


async def test_recovery_returns_a_genuinely_new_tool_to_runner(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    marker = tmp_path / "new-tool-crashed-once"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    provider._app_server_command = _fake_command("recover_new_tool", marker)
    context = {"session_key": "session", "turn_id": "recover-new"}
    messages = [{"role": "user", "content": "write twice"}]
    first = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first"},
        ]
    )
    new_call = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert new_call.finish_reason == "tool_calls"
    assert new_call.tool_calls[0].id == "call-new"
    assert new_call.tool_calls[0].arguments == {"value": 2}
    assert "idempotent_tool_replays" not in new_call.provider_diagnostics
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [new_call.tool_calls[0].to_openai_tool_call()],
            },
            {"role": "tool", "tool_call_id": "call-new", "content": "second"},
        ]
    )
    final = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert final.content == "done"
    assert list(ledger_dir.glob("*.json")) == []


async def test_same_signature_can_execute_twice_without_bridge_recovery(
    tmp_path: Path,
) -> None:
    ledger_dir = tmp_path / "ledger"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    provider._app_server_command = _fake_command("repeat_normal")
    context = {"session_key": "session", "turn_id": "normal-repeat"}
    messages = [{"role": "user", "content": "write twice"}]
    first = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first"},
        ]
    )
    second = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert second.tool_calls[0].id == "call-2"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [second.tool_calls[0].to_openai_tool_call()],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "second"},
        ]
    )
    final = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert final.content == "done"
    assert list(ledger_dir.glob("*.json")) == []


async def test_new_provider_restores_durable_result_after_process_restart(
    tmp_path: Path,
) -> None:
    ledger_dir = tmp_path / "ledger"
    context = {"session_key": "session", "turn_id": "restart"}
    messages = [{"role": "user", "content": "write once"}]
    first_provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    first_provider._app_server_command = _fake_command("crash_after_tool")
    first = await first_provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "write",
                "content": "persisted result",
            },
        ]
    )
    failed = await first_provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert failed.finish_reason == "error"
    assert len(list(ledger_dir.glob("*.json"))) == 1

    restarted = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    restarted._app_server_command = _fake_command("complete")
    restored = await restarted.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert restored.content == "done"
    assert restored.usage["total_tokens"] == 18
    assert restored.provider_diagnostics["restored_from_idempotency_ledger"] is True
    assert restored.provider_diagnostics["idempotent_tool_replays"] == 1
    assert list(ledger_dir.glob("*.json")) == []


async def test_stream_recovery_opens_a_new_output_segment(tmp_path: Path) -> None:
    marker = tmp_path / "stream-crashed-once"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test",
        idempotency_dir=tmp_path / "ledger",
    )
    provider._app_server_command = _fake_command("recover_stream", marker)
    context = {"session_key": "session", "turn_id": "stream-recover"}
    messages = [{"role": "user", "content": "write once"}]
    first = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "write",
                "content": "ok",
            },
        ]
    )
    deltas: list[str] = []
    recoveries: list[bool] = []

    async def on_delta(text: str) -> None:
        deltas.append(text)

    async def on_recover() -> None:
        recoveries.append(True)

    response = await provider.chat_stream_with_retry(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
        on_content_delta=on_delta,
        on_stream_recover=on_recover,
    )
    assert response.content == "done"
    assert deltas == ["partial", "done"]
    assert recoveries == [True]
    assert response.provider_diagnostics["idempotent_tool_replays"] == 1


async def test_corrupt_ledger_is_rebuilt_from_authoritative_messages(
    tmp_path: Path,
) -> None:
    ledger_dir = tmp_path / "ledger"
    context = {"session_key": "session", "turn_id": "corrupt"}
    ledger = _idempotency_ledger(
        (context["session_key"], context["turn_id"]),
        root_override=ledger_dir,
    )
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text("{broken", encoding="utf-8")
    messages = [
        {"role": "user", "content": "write once"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "write", "arguments": '{"value":1}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "write",
            "content": "ok",
        },
    ]
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    provider._app_server_command = _fake_command("complete")
    response = await provider.chat(
        messages=messages,
        tools=_tool_schema(),
        request_context=context,
    )
    assert response.content == "done"
    assert response.provider_diagnostics["idempotent_tool_replays"] == 1
    assert list(ledger_dir.glob("*.json")) == []


async def test_unsupported_native_codex_tool_event_is_rejected(tmp_path: Path) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("unsupported_native")
    response = await provider.chat(
        messages=[{"role": "user", "content": "inspect"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": "native"},
    )
    assert response.finish_reason == "error"
    assert response.error_code == "native_tool_blocked"
    assert response.error_should_retry is False


async def test_native_image_view_is_bridged_through_nanobot_read_file(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    provider._app_server_command = _fake_command("image_view")
    context = {"session_key": "session", "turn_id": "image-view"}
    messages = [{"role": "user", "content": "inspect the image"}]

    bridged = await provider.chat(
        messages=messages,
        tools=_tool_schema_with_read_file(),
        request_context=context,
    )

    assert bridged.finish_reason == "tool_calls"
    assert bridged.tool_calls[0].name == "read_file"
    assert bridged.tool_calls[0].arguments == {"path": "reference.png"}
    assert bridged.provider_diagnostics["native_image_view_bridged"] is True
    assert provider._turns == {}

    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [bridged.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": bridged.tool_calls[0].id,
                "content": [
                    {"type": "text", "text": "reference"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ]
    )
    completed = await provider.chat(
        messages=messages,
        tools=_tool_schema_with_read_file(),
        request_context=context,
    )

    assert completed.finish_reason == "stop"
    assert completed.content == "done"
    assert completed.provider_diagnostics["idempotent_tool_replays"] == 1
    assert list(ledger_dir.glob("*.json")) == []


async def test_native_image_view_recovery_skips_prior_tool_replay(tmp_path: Path) -> None:
    marker = tmp_path / "image-after-tool"
    ledger_dir = tmp_path / "ledger"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=ledger_dir
    )
    provider._app_server_command = _fake_command("image_after_tool", marker)
    context = {"session_key": "session", "turn_id": "image-after-tool"}
    messages = [{"role": "user", "content": "inspect after another tool"}]

    first = await provider.chat(
        messages=messages,
        tools=_tool_schema_with_read_file(),
        request_context=context,
    )
    assert first.tool_calls[0].name == "write"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [first.tool_calls[0].to_openai_tool_call()],
            },
            {"role": "tool", "tool_call_id": first.tool_calls[0].id, "content": "ok"},
        ]
    )

    bridged = await provider.chat(
        messages=messages,
        tools=_tool_schema_with_read_file(),
        request_context=context,
    )
    assert bridged.tool_calls[0].name == "read_file"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [bridged.tool_calls[0].to_openai_tool_call()],
            },
            {
                "role": "tool",
                "tool_call_id": bridged.tool_calls[0].id,
                "content": [
                    {"type": "text", "text": "reference"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ]
    )

    completed = await provider.chat(
        messages=messages,
        tools=_tool_schema_with_read_file(),
        request_context=context,
    )

    assert completed.finish_reason == "stop"
    assert completed.content == "done"
    assert completed.provider_diagnostics["idempotent_tool_replays"] == 1
    assert list(ledger_dir.glob("*.json")) == []


def test_native_image_view_bridge_fails_closed_without_path_or_read_file() -> None:
    bridge = _CodexAppServerTurn(["unused"])
    with pytest.raises(CodexAppServerError, match="read_file is unavailable"):
        bridge._bridge_image_view({"id": "image-1", "type": "imageView", "path": "a.png"})

    bridge._dynamic_tool_names = {"read_file"}
    with pytest.raises(CodexAppServerError, match="missing an image path"):
        bridge._bridge_image_view({"id": "image-1", "type": "imageView"})


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [("native", "completed"), ("native_failed", "failed")],
)
async def test_native_file_change_is_supported_and_reported(
    tmp_path: Path,
    mode: str,
    expected_status: str,
) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command(mode)

    response = await provider.chat(
        messages=[{"role": "user", "content": "modify the file"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": f"native-{mode}"},
    )

    assert response.finish_reason == "stop"
    assert response.content == "done"
    assert response.provider_diagnostics["native_file_changes"] == {
        "count": 1,
        "statuses": {expected_status: 1},
        "approval_requests_declined": 0,
    }


async def test_native_file_change_expansion_approval_is_declined(tmp_path: Path) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("native_approval")

    response = await provider.chat(
        messages=[{"role": "user", "content": "modify outside the workspace"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": "native-approval"},
    )

    assert response.finish_reason == "stop"
    assert response.provider_diagnostics["native_file_changes"] == {
        "count": 1,
        "statuses": {"declined": 1},
        "approval_requests_declined": 1,
    }


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [("native_command", "completed"), ("native_command_failed", "failed")],
)
async def test_native_command_execution_is_supported_and_reported(
    tmp_path: Path,
    mode: str,
    expected_status: str,
) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command(mode)

    response = await provider.chat(
        messages=[{"role": "user", "content": "inspect the workspace"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": f"command-{mode}"},
    )

    assert response.finish_reason == "stop"
    assert response.content == "done"
    assert response.provider_diagnostics["native_command_executions"] == {
        "count": 1,
        "statuses": {expected_status: 1},
        "approval_requests_declined": 0,
    }


async def test_native_command_expansion_approval_is_declined(tmp_path: Path) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("native_command_approval")

    response = await provider.chat(
        messages=[{"role": "user", "content": "run a network command"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": "command-approval"},
    )

    assert response.finish_reason == "stop"
    assert response.provider_diagnostics["native_command_executions"] == {
        "count": 1,
        "statuses": {"declined": 1},
        "approval_requests_declined": 1,
    }


async def test_unsupported_native_codex_tool_is_corrected_once_with_nanobot_tools(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "native-blocked-once"
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("unsupported_native_once", marker)

    response = await provider.chat(
        messages=[{"role": "user", "content": "modify the file"}],
        tools=_tool_schema(),
        request_context={"session_key": "session", "turn_id": "native-correction"},
    )

    assert response.finish_reason == "stop"
    assert response.content == "done"


async def test_rpc_timeout_closes_hung_subprocess() -> None:
    bridge = _CodexAppServerTurn(
        _fake_command("hang"),
        rpc_timeout_s=0.1,
        event_timeout_s=0.1,
    )
    with pytest.raises(CodexAppServerError, match="timed out") as exc_info:
        await bridge.start(
            model="gpt-test",
            reasoning_effort=None,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_choice=None,
        )
    assert exc_info.value.retryable is True
    process = bridge.process
    assert process is not None
    await bridge.close()
    assert bridge.process is None
    transport = getattr(process, "_transport", None)
    assert transport is None or transport.is_closing()
    assert transport is None or getattr(transport, "_loop", None) is None


async def test_provider_aclose_closes_active_app_server_bridges(tmp_path: Path) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    bridge = _CodexAppServerTurn(_fake_command("hang_turn"))
    await bridge.start(
        model="gpt-test",
        reasoning_effort=None,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        tool_choice=None,
    )
    provider._turns[("session", "turn")] = bridge

    await provider.aclose()

    assert bridge.process is None
    assert provider._turns == {}
    assert provider._turn_locks == {}


async def test_cancellation_closes_active_subprocess_and_turn_state(tmp_path: Path) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("hang_turn")
    task = asyncio.create_task(
        provider.chat(
            messages=[{"role": "user", "content": "wait"}],
            tools=None,
            request_context={"session_key": "session", "turn_id": "cancel"},
        )
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider._turns == {}


async def test_concurrent_turn_keys_use_independent_protocol_sessions(tmp_path: Path) -> None:
    provider = CodexAppServerProvider(
        default_model="openai-codex/gpt-test", idempotency_dir=tmp_path / "ledger"
    )
    provider._app_server_command = _fake_command("answer")

    async def run(turn_id: str):
        return await provider.chat(
            messages=[{"role": "user", "content": turn_id}],
            tools=None,
            request_context={"session_key": turn_id, "turn_id": turn_id},
        )

    first, second = await asyncio.gather(run("one"), run("two"))
    assert first.content == second.content == "done"
    assert provider._turns == {}


def test_factory_routes_openai_codex_to_fork_without_eager_binary_lookup() -> None:
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "openai-codex",
                    "model": "openai-codex/gpt-5.6-sol",
                }
            }
        }
    )
    provider = make_provider(config)
    assert isinstance(provider, CodexAppServerProvider)
    assert provider.__class__.__name__ == "OpenAICodexProvider"
    assert provider._app_server_command is None
