"""Small asyncio JSON-RPC client for stdio Language Server Protocol servers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class LspError(RuntimeError):
    pass


class LspClient:
    def __init__(self, command: list[str], workspace: Path, *, timeout: float = 20.0) -> None:
        self.command = command
        self.workspace = workspace
        self.timeout = timeout
        self.process: asyncio.subprocess.Process | None = None
        self.capabilities: dict[str, Any] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self, *, startup_timeout: float = 60.0) -> None:
        if self.alive:
            return
        if not self.command:
            raise LspError("language server command is empty")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=str(self.workspace),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise LspError(f"failed to start language server: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            result = await asyncio.wait_for(
                self.request("initialize", {
                    "processId": None,
                    "rootUri": self.workspace.as_uri(),
                    "capabilities": {
                        "general": {"positionEncodings": ["utf-16"]},
                        "textDocument": {
                            "definition": {"linkSupport": True},
                            "references": {},
                            "implementation": {"linkSupport": True},
                            "callHierarchy": {"dynamicRegistration": False},
                        }
                    },
                    "workspaceFolders": [{"uri": self.workspace.as_uri(), "name": self.workspace.name}],
                }, timeout=startup_timeout),
                timeout=startup_timeout + 1,
            )
            self.capabilities = result.get("capabilities", {}) if isinstance(result, dict) else {}
            await self.notify("initialized", {})
        except BaseException:
            await self.close(force=True)
            raise

    async def request(
        self, method: str, params: Any, *, timeout: float | None = None
    ) -> Any:
        if not self.alive or self.process is None:
            raise LspError("language server is not running")
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({
                "jsonrpc": "2.0", "id": request_id, "method": method, "params": params
            })
            try:
                return await asyncio.wait_for(
                    future, timeout=self.timeout if timeout is None else timeout
                )
            except asyncio.TimeoutError as exc:
                raise LspError(f"language server request timed out: {method}") from exc
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Any) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.alive or self.process is None or self.process.stdin is None:
            raise LspError("language server is not running")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        packet = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        async with self._write_lock:
            self.process.stdin.write(packet)
            await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                headers: dict[str, str] = {}
                while True:
                    line = await self.process.stdout.readline()
                    if not line:
                        raise EOFError("language server closed stdout")
                    if line in {b"\r\n", b"\n"}:
                        break
                    key, _, value = line.decode("ascii", errors="replace").partition(":")
                    headers[key.lower().strip()] = value.strip()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                message = json.loads((await self.process.stdout.readexactly(length)).decode("utf-8"))
                request_id = message.get("id")
                if "method" in message and request_id is not None:
                    await self._reply_to_server_request(message)
                    continue
                if request_id is not None and request_id in self._pending:
                    future = self._pending[request_id]
                    if "error" in message:
                        future.set_exception(LspError(str(message["error"])))
                    else:
                        future.set_result(message.get("result"))
        except (asyncio.CancelledError, EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
            self._fail_pending(LspError(str(exc)))

    async def _reply_to_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if method == "workspace/configuration":
            items = (message.get("params") or {}).get("items") or []
            result: Any = [{} for _item in items]
            payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
        elif method in {"client/registerCapability", "client/unregisterCapability", "window/workDoneProgress/create", "workspace/codeLens/refresh", "workspace/semanticTokens/refresh", "workspace/inlayHint/refresh"}:
            payload = {"jsonrpc": "2.0", "id": request_id, "result": None}
        elif method == "workspace/applyEdit":
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"applied": False, "failureReason": "read-only client"},
            }
        else:
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        await self._send(payload)

    def _fail_pending(self, error: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def close(self, *, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None and not force:
            try:
                await self.request("shutdown", None)
                await self.notify("exit", None)
            except Exception:
                force = True
        if process.returncode is None:
            if force:
                process.kill()
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        reader = self._reader_task
        if reader is not None and not reader.done():
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
        self._reader_task = None
        self._fail_pending(LspError("language server closed"))
        self.process = None
