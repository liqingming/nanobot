"""IDE 桥接的快照、鉴权、话题隔离和生命周期回归测试。"""

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from nanobot.fork.cli.ide_bridge import IDEBridge
from nanobot.fork.cli.ide_context import MAX_SELECTION_BYTES, PendingIDEContext
from nanobot.fork.cli.ide_integration import IDEIntegration


def make_pending(tmp_path):
    pending = PendingIDEContext(tmp_path, Mock())
    pending.switch("cli:first")
    return pending


def payload(pending, **changes):
    return {
        "request_id": "one", "path": str(pending.workspace / "玩家.cs"),
        "start_line": 10, "end_line": 12, "text": "未保存的代码\n",
        "session_key": pending.session_key, "generation": pending.generation,
        **changes,
    }


def test_snapshot_consumed_once_and_commands_preserve_pending(tmp_path):
    pending = make_pending(tmp_path)
    data = payload(pending)
    assert pending.add(data)
    assert not pending.add(data)
    assert pending.consume("/model") == "/model"
    assert pending.consume(" ") == " "
    assert "玩家.cs:10–12" in pending.preview("分析")
    result = pending.consume("分析")
    assert result.startswith("分析")
    assert json.loads(result.split("\n")[-1])[0]["text"] == "未保存的代码\n"
    assert pending.consume("下个问题") == "下个问题"
    with pytest.raises(LookupError):
        pending.add(data)


def test_topic_switch_and_switch_back_reject_stale_picker(tmp_path):
    pending = make_pending(tmp_path)
    stale = payload(pending)
    pending.add(stale)
    pending.switch("cli:second")
    assert not pending.items
    pending.switch("cli:first")
    with pytest.raises(LookupError):
        pending.add(stale)


@pytest.mark.parametrize("changes", [
    {"path": "../secret.txt"}, {"path": "relative.cs"},
    {"start_line": 0}, {"start_line": True}, {"end_line": 1},
    {"text": ""}, {"text": "x" * (MAX_SELECTION_BYTES + 1)},
    {"path": "bad\npath"}, {"request_id": []},
])
def test_reject_invalid_selection(tmp_path, changes):
    pending = make_pending(tmp_path)
    with pytest.raises(ValueError):
        pending.add(payload(pending, **changes))


def test_reject_outside_and_symlink(tmp_path):
    pending = make_pending(tmp_path)
    with pytest.raises(ValueError):
        pending.add(payload(pending, path=str(tmp_path.parent / "outside.cs")))
    link = tmp_path / "link"
    try:
        link.symlink_to(tmp_path.parent, target_is_directory=True)
    except OSError:
        pytest.skip("当前 Windows 用户无符号链接权限")
    with pytest.raises(ValueError):
        pending.add(payload(pending, path=str(link / "outside.cs")))


def test_limits_remove_and_clear(tmp_path):
    pending = make_pending(tmp_path)
    for i in range(8):
        pending.add(payload(pending, request_id=str(i)))
    with pytest.raises(ValueError):
        pending.add(payload(pending, request_id="overflow"))
    pending.remove(2)
    pending.add(payload(pending, request_id="new"))
    with pytest.raises(ValueError):
        pending.remove(99)
    pending.clear()
    for i in range(4):
        pending.add(payload(pending, request_id=str(i), text="x" * MAX_SELECTION_BYTES))
    with pytest.raises(ValueError):
        pending.add(payload(pending, request_id="overflow"))
    pending.clear()
    assert pending.changed.call_args.args == ([],)


@pytest.mark.asyncio
async def test_http_auth_snapshot_conflict_and_cleanup(tmp_path):
    pending = make_pending(tmp_path)
    bridge = IDEBridge(pending, lambda: "当前话题", tmp_path / "registry")
    await bridge.start()
    manifest = bridge._manifest
    assert json.loads(manifest.read_text())["port"] == bridge.port
    headers = {"Authorization": f"Bearer {bridge.token}"}
    url = f"http://127.0.0.1:{bridge.port}"
    try:
        async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
            assert (await client.get("/v1/session")).status_code == 401
            response = await client.get("/v1/session", headers=headers)
            assert response.json()["session_key"] == "cli:first"
            assert response.json()["title"] == "当前话题"
            assert "token" not in response.json()
            for extra in ({"Origin": "http://evil.example"}, {"Host": "evil.example"}):
                assert (await client.get("/v1/session", headers=headers | extra)).status_code == 403
            assert (await client.post("/v1/context", headers=headers, json=payload(pending))).json()["added"]
            assert not (await client.post("/v1/context", headers=headers, json=payload(pending))).json()["added"]
            assert len(pending.items) == 1
            stale = payload(pending)
            pending.switch("cli:other")
            assert (await client.post("/v1/context", headers=headers, json=stale)).status_code == 409
            assert (await client.post("/v1/context", headers=headers, json=[])).status_code == 400
            assert (await client.post("/v1/context", headers=headers, content="x")).status_code == 415
            assert (await client.post("/v1/context", headers=headers, content="x" * 410000)).status_code == 413
            assert (await client.get("/unknown", headers=headers)).status_code == 404
    finally:
        await bridge.close()
    assert not manifest.exists()
    assert bridge.server is None
    assert not bridge._clients


@pytest.mark.asyncio
async def test_close_cancels_incomplete_connections_and_restarts(tmp_path):
    bridge = IDEBridge(make_pending(tmp_path), lambda: "", tmp_path / "registry")
    await bridge.start()
    first_token = bridge.token
    _, writer = await asyncio.open_connection("127.0.0.1", bridge.port)
    writer.write(b"GET /v1/session HTTP/1.1\r\n")
    await writer.drain()
    await asyncio.sleep(0)
    await bridge.close()
    writer.close()
    await writer.wait_closed()
    await bridge.start()
    assert bridge.token != first_token
    await bridge.close()


@pytest.mark.asyncio
async def test_integration_waits_for_submit_and_queue_snapshot_is_fixed(tmp_path):
    tui = Mock()
    integration = IDEIntegration(tmp_path, tui)
    integration.bridge.registry = tmp_path / "registry"
    integration.bridge._manifest = integration.bridge.registry / "test.json"
    integration.pending.switch("cli:first")
    await integration.command("/ide on")
    try:
        integration.pending.add(payload(integration.pending))
        tui.set_ide_context.assert_called_with(["玩家.cs:10–12"])
        tui.set_on_submit.assert_not_called()
        assert await integration.command("/ide")
        assert not await integration.command("解释代码")
        first_queued = integration.pending.consume("第一条问题")
        integration.pending.add(payload(integration.pending, text="第二份选区"))
        assert "第二份选区" not in first_queued
        assert "第二份选区" in integration.pending.consume("第二条问题")
        integration.pending.add(payload(integration.pending))
        await integration.command("/ide remove 1")
        assert not integration.pending.items
    finally:
        await integration.command("/ide off")
    assert integration.bridge.server is None


@pytest.mark.asyncio
async def test_two_instances_are_isolated(tmp_path):
    first = make_pending(tmp_path)
    second = make_pending(tmp_path)
    second.switch("cli:second")
    registry = tmp_path / "registry"
    a = IDEBridge(first, lambda: "A", registry)
    b = IDEBridge(second, lambda: "B", registry)
    await a.start()
    await b.start()
    try:
        assert a.port != b.port
        async with httpx.AsyncClient(trust_env=False) as client:
            url = f"http://127.0.0.1:{b.port}/v1/context"
            response = await client.post(url, json=payload(second),
                                        headers={"Authorization": f"Bearer {a.token}"})
            assert response.status_code == 401
            response = await client.post(url, json=payload(second),
                                        headers={"Authorization": f"Bearer {b.token}"})
            assert response.status_code == 200
        assert not first.items
        assert len(second.items) == 1
    finally:
        await a.close()
        await b.close()
    assert not list(registry.glob("*.json"))


@pytest.mark.asyncio
async def test_failed_start_does_not_remove_existing_manifest(tmp_path):
    bridge = IDEBridge(make_pending(tmp_path), lambda: "", tmp_path)
    bridge._manifest.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        await bridge.start()
    assert bridge._manifest.read_text() == "existing"
    assert bridge.server is None


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [
    "Host: duplicate\r\n",
    "Transfer-Encoding: chunked\r\n",
    "X-Long: " + "x" * 8200 + "\r\n",
])
async def test_malformed_http_is_rejected(tmp_path, extra):
    bridge = IDEBridge(make_pending(tmp_path), lambda: "", tmp_path / "registry")
    await bridge.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", bridge.port)
        writer.write((
            f"GET /v1/session HTTP/1.1\r\nHost: 127.0.0.1:{bridge.port}\r\n"
            f"Authorization: Bearer {bridge.token}\r\n{extra}\r\n"
        ).encode("ascii"))
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 2)
        assert response.startswith((b"HTTP/1.1 400", b"HTTP/1.1 403"))
        writer.close()
        await writer.wait_closed()
    finally:
        await bridge.close()
