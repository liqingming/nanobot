from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.fork.agent.tools.lsp_client import LspClient, LspError


@pytest.mark.asyncio
async def test_server_configuration_request_gets_one_object_per_item(tmp_path: Path) -> None:
    client = LspClient([], tmp_path)
    sent: list[dict] = []

    async def capture(payload: dict) -> None:
        sent.append(payload)

    client._send = capture  # type: ignore[method-assign]
    await client._reply_to_server_request({
        "id": 7,
        "method": "workspace/configuration",
        "params": {"items": [{"section": "a"}, {"section": "b"}]},
    })
    assert sent == [{"jsonrpc": "2.0", "id": 7, "result": [{}, {}]}]


@pytest.mark.asyncio
async def test_unknown_server_request_returns_method_not_found(tmp_path: Path) -> None:
    client = LspClient([], tmp_path)
    sent: list[dict] = []

    async def capture(payload: dict) -> None:
        sent.append(payload)

    client._send = capture  # type: ignore[method-assign]
    await client._reply_to_server_request({"id": 9, "method": "custom/unknown"})
    assert sent[0]["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_apply_edit_is_rejected_by_read_only_client(tmp_path: Path) -> None:
    client = LspClient([], tmp_path)
    sent: list[dict] = []

    async def capture(payload: dict) -> None:
        sent.append(payload)

    client._send = capture  # type: ignore[method-assign]
    await client._reply_to_server_request({"id": 3, "method": "workspace/applyEdit"})
    assert sent[0]["result"] == {"applied": False, "failureReason": "read-only client"}


@pytest.mark.asyncio
async def test_request_send_failure_cleans_pending_future(tmp_path: Path) -> None:
    client = LspClient([], tmp_path)
    client.process = SimpleNamespace(returncode=None)

    async def fail(_payload: dict) -> None:
        raise BrokenPipeError("closed")

    client._send = fail  # type: ignore[method-assign]
    with pytest.raises(BrokenPipeError):
        await client.request("x", {})
    assert client._pending == {}


def test_fail_pending_sets_errors_and_clears_map(tmp_path: Path) -> None:
    client = LspClient([], tmp_path)

    async def run() -> None:
        future = asyncio.get_running_loop().create_future()
        client._pending[1] = future
        client._fail_pending(LspError("closed"))
        assert client._pending == {}
        with pytest.raises(LspError, match="closed"):
            await future

    asyncio.run(run())


@pytest.mark.asyncio
async def test_stdio_protocol_initialize_request_and_shutdown(tmp_path: Path) -> None:
    import sys
    import textwrap

    server = tmp_path / "fake_lsp.py"
    server.write_text(textwrap.dedent(r'''
        import json, sys
        def read():
            headers = {}
            while True:
                line = sys.stdin.buffer.readline()
                if not line:
                    return None
                if line in (b"\r\n", b"\n"):
                    break
                key, value = line.decode("ascii").split(":", 1)
                headers[key.lower()] = value.strip()
            return json.loads(sys.stdin.buffer.read(int(headers["content-length"])))
        def send(value):
            body = json.dumps(value, separators=(",", ":")).encode()
            sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            sys.stdout.buffer.flush()
        while True:
            msg = read()
            if msg is None:
                break
            method = msg.get("method")
            if method == "initialize":
                send({"jsonrpc":"2.0","id":msg["id"],"result":{"capabilities":{"definitionProvider":True}}})
            elif method == "shutdown":
                send({"jsonrpc":"2.0","id":msg["id"],"result":None})
            elif method == "exit":
                break
    '''), encoding="utf-8")
    client = LspClient([sys.executable, str(server)], tmp_path, timeout=3)
    await client.start(startup_timeout=3)
    assert client.capabilities["definitionProvider"] is True
    assert client.alive
    await client.close()
    assert not client.alive


@pytest.mark.asyncio
async def test_request_timeout_override_is_used_for_initialize_path(tmp_path: Path) -> None:
    client = LspClient([], tmp_path, timeout=0.01)
    client.process = SimpleNamespace(returncode=None)

    async def sent(_payload: dict) -> None:
        async def complete() -> None:
            await asyncio.sleep(0.03)
            next(iter(client._pending.values())).set_result({"ok": True})
        asyncio.create_task(complete())

    client._send = sent  # type: ignore[method-assign]
    result = await client.request("initialize", {}, timeout=0.1)
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_cancelled_start_force_closes_process_and_reader(
    tmp_path: Path, monkeypatch
) -> None:
    client = LspClient(["fake-server"], tmp_path)
    process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(), stdout=SimpleNamespace())
    process.kill = lambda: setattr(process, "returncode", -9)

    async def wait() -> int:
        return process.returncode

    async def create_process(*_args, **_kwargs):
        return process

    process.wait = wait
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    async def cancelled_request(*_args, **_kwargs):
        raise asyncio.CancelledError

    async def blocked_reader() -> None:
        await asyncio.sleep(60)

    client.request = cancelled_request  # type: ignore[method-assign]
    client._read_loop = blocked_reader  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await client.start(startup_timeout=1)
    assert client.process is None
    assert client._reader_task is None
