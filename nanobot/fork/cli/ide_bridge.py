"""仅监听 IPv4 回环的 IDE 接口，无网关依赖、无模型调用。"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import secrets
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Callable

from loguru import logger

from nanobot.fork.cli.ide_context import PendingIDEContext

MAX_BODY = 400 * 1024


class IDEBridge:
    def __init__(
        self, pending: PendingIDEContext, title: Callable[[], str],
        registry: Path | None = None,
    ) -> None:
        self.pending = pending
        self.title = title
        self.registry = registry or Path.home() / ".nanobot" / "ide-bridge"
        self.instance_id = uuid.uuid4().hex
        self.token = secrets.token_urlsafe(32)
        self.port = 0
        self.server: asyncio.Server | None = None
        self._clients: set[asyncio.Task] = set()
        self._manifest = self.registry / f"{self.instance_id}.json"
        self._manifest_owned = False

    async def start(self) -> None:
        if self.server is not None:
            return
        self.token = secrets.token_urlsafe(32)
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=8192)
        self.port = self.server.sockets[0].getsockname()[1]
        try:
            self.registry.mkdir(parents=True, exist_ok=True, mode=0o700)
            # 不写入项目目录；Unix 新文件仅当前用户可读写，Windows 继承用户目录 ACL。
            fd = os.open(self._manifest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            self._manifest_owned = True
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({
                    "version": 1, "instance_id": self.instance_id,
                    "workspace": str(self.pending.workspace), "port": self.port,
                    "token": self.token, "pid": os.getpid(),
                }, output)
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        for task in list(self._clients):
            task.cancel()
        await asyncio.gather(*self._clients, return_exceptions=True)
        if self._manifest_owned:
            try:
                self._manifest.unlink(missing_ok=True)
            except OSError as exc:
                # 登记清理失败不能阻断 CLI 的 Agent/MCP 正常退出。
                logger.warning("IDE 登记清理失败：{}", exc)
            else:
                self._manifest_owned = False

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if len(self._clients) >= 16:
            writer.close()
            return
        self._clients.add(task)
        try:
            async with asyncio.timeout(5):
                await self._request(reader, writer)
        except (TimeoutError, ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            self._clients.discard(task)
            writer.close()
            with contextlib.suppress(ConnectionError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), 1)

    async def _request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            if len(raw) > 8192:
                raise ValueError()
            lines = raw.decode("ascii").split("\r\n")
            method, route, version = lines[0].split(" ")
            if version != "HTTP/1.1":
                raise ValueError()
            headers = {}
            for line in lines[1:-2]:
                key, value = line.split(":", 1)
                key = key.lower()
                if key in headers:
                    raise ValueError()
                headers[key] = value.strip()
        except (ValueError, UnicodeError, asyncio.LimitOverrunError):
            await self._reply(writer, 400, {"error": "无效 HTTP 请求"})
            return
        # 不提供 CORS；拒绝浏览器来源以及非固定回环 Host，避免跨站/重绑定。
        if (headers.get("host") != f"127.0.0.1:{self.port}"
                or "origin" in headers or "transfer-encoding" in headers):
            await self._reply(writer, 403, {"error": "仅允许本机 IDE 客户端"})
            return
        if not hmac.compare_digest(headers.get("authorization", ""), f"Bearer {self.token}"):
            await self._reply(writer, 401, {"error": "无效凭证"})
            return
        if method == "GET" and route == "/v1/session":
            await self._reply(writer, 200, {
                "version": 1, "instance_id": self.instance_id,
                "workspace": str(self.pending.workspace),
                "session_key": self.pending.session_key,
                "generation": self.pending.generation,
                "title": self.title(), "pending": len(self.pending.items),
            })
            return
        if method != "POST" or route != "/v1/context":
            await self._reply(writer, 404, {"error": "接口不存在"})
            return
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            await self._reply(writer, 413, {"error": "无效请求大小"})
            return
        if headers.get("content-type", "").split(";")[0] != "application/json":
            await self._reply(writer, 415, {"error": "要求 application/json"})
            return
        try:
            payload = json.loads(await reader.readexactly(length))
            if not isinstance(payload, dict):
                raise ValueError("要求 JSON 对象")
            added = self.pending.add(payload)
        except LookupError as exc:
            await self._reply(writer, 409, {"error": str(exc)})
        except (ValueError, OSError, RecursionError) as exc:
            await self._reply(writer, 400, {"error": str(exc)})
        else:
            await self._reply(writer, 200, {"added": added, "pending": len(self.pending.items)})

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        writer.write((
            f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Connection: close\r\nCache-Control: no-store\r\n\r\n"
        ).encode("ascii") + body)
        await writer.drain()
