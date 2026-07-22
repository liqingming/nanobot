"""Workspace-scoped language-server lifecycle and LSP result normalization."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from nanobot.fork.agent.tools.code_intelligence_models import CodeLocation
from nanobot.fork.agent.tools.lsp_client import LspClient, LspError


class CodeIntelligenceManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._sessions: dict[str, LspClient] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._open_documents: dict[tuple[str, str], tuple[int, str]] = {}
        self._document_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _command(self, workspace: Path) -> list[str]:
        raw = self.config.get("command") or []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list) and raw:
            return [str(item).replace("{workspace}", str(workspace)) for item in raw]
        executable = shutil.which("csharp-ls")
        return [executable] if executable else []

    async def session(self, workspace: Path) -> LspClient:
        key = str(workspace.resolve()).casefold()
        existing = self._sessions.get(key)
        if existing is not None and existing.alive:
            return existing
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = self._sessions.get(key)
            if existing is not None and existing.alive:
                return existing
            if existing is not None:
                self._open_documents = {
                    doc_key: state
                    for doc_key, state in self._open_documents.items()
                    if doc_key[0] != key
                }
            command = self._command(workspace)
            if not command:
                raise LspError(
                    "C# language server unavailable. Install csharp-ls or configure "
                    "tools.codeIntelligence.command with a standard stdio LSP command."
                )
            client = LspClient(
                command,
                workspace,
                timeout=float(self.config.get("request_timeout_seconds", 20)),
            )
            try:
                await client.start(
                    startup_timeout=float(self.config.get("startup_timeout_seconds", 60))
                )
            except LspError:
                raise
            except Exception as exc:
                raise LspError(f"language server initialization failed: {exc}") from exc
            self._sessions[key] = client
            return client

    @staticmethod
    def text_document_position(
        path: Path, line: int, column: int, text: str = ""
    ) -> dict[str, Any]:
        line_index = max(0, line - 1)
        character_index = max(0, column - 1)
        if text:
            source_line = text.splitlines()[line_index] if line_index < len(text.splitlines()) else ""
            prefix = source_line[:character_index]
            character_index = len(prefix.encode("utf-16-le")) // 2
        return {
            "textDocument": {"uri": path.as_uri()},
            "position": {"line": line_index, "character": character_index},
        }

    async def _sync_document(
        self, client: LspClient, workspace: Path, path: Path, text: str
    ) -> None:
        session_key = str(workspace.resolve()).casefold()
        uri = path.as_uri()
        key = (session_key, uri)
        lock = self._document_locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = self._open_documents.get(key)
            if existing is None:
                await client.notify("textDocument/didOpen", {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "csharp",
                        "version": 1,
                        "text": text,
                    }
                })
                self._open_documents[key] = (1, text)
                return
            version, old_text = existing
            if old_text == text:
                return
            version += 1
            await client.notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            })
            self._open_documents[key] = (version, text)

    @staticmethod
    def supports(client: LspClient, action: str) -> bool:
        key = {
            "definition": "definitionProvider",
            "references": "referencesProvider",
            "implementations": "implementationProvider",
            "callers": "callHierarchyProvider",
            "callees": "callHierarchyProvider",
        }.get(action)
        return bool(key and key in client.capabilities and client.capabilities[key] is not False)

    async def query(
        self,
        workspace: Path,
        *,
        action: str,
        path: Path,
        line: int,
        column: int,
    ) -> tuple[list[CodeLocation], dict[str, Any]]:
        client = await self.session(workspace)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise LspError(f"cannot read source document: {exc}") from exc
        await self._sync_document(client, workspace, path, text)
        base = self.text_document_position(path, line, column, text)
        metadata = {"semantic_available": True, "degraded": False}
        if action == "references":
            result = await client.request(
                "textDocument/references",
                {**base, "context": {"includeDeclaration": False}},
            )
        elif action == "definition":
            result = await client.request("textDocument/definition", base)
        elif action == "implementations":
            result = await client.request("textDocument/implementation", base)
        elif action in {"callers", "callees"}:
            return await self._call_hierarchy(client, base, action)
        else:
            raise LspError(f"unsupported semantic action: {action}")
        return self._normalize_locations(result, workspace, kind=action.rstrip("s")), metadata

    async def _call_hierarchy(
        self, client: LspClient, base: dict[str, Any], action: str
    ) -> tuple[list[CodeLocation], dict[str, Any]]:
        if not self.supports(client, action):
            if action == "callers" and self.supports(client, "references"):
                result = await client.request(
                    "textDocument/references",
                    {**base, "context": {"includeDeclaration": False}},
                )
                return self._normalize_locations(result, client.workspace, kind="reference"), {
                    "semantic_available": True,
                    "degraded": True,
                    "relationship_source": "references_fallback",
                    "warning": (
                        "call hierarchy unsupported; callers degraded to references. "
                        "Results are symbol references, not proven call sites."
                    ),
                }
            raise LspError(f"language server does not support {action} call hierarchy")
        items = await client.request("textDocument/prepareCallHierarchy", base) or []
        if not items:
            return [], {"semantic_available": True, "degraded": False}
        method = "callHierarchy/incomingCalls" if action == "callers" else "callHierarchy/outgoingCalls"
        rows: list[CodeLocation] = []
        for item in items:
            calls = await client.request(method, {"item": item}) or []
            for call in calls:
                target = call.get("from") if action == "callers" else call.get("to")
                if isinstance(target, dict):
                    rows.extend(
                        self._normalize_locations(target, client.workspace, kind=action[:-1])
                    )
        rows = self._deduplicate_locations(rows)
        return rows, {
            "semantic_available": True,
            "degraded": False,
            "relationship_source": "lsp_call_hierarchy",
        }

    @staticmethod
    def _deduplicate_locations(rows: list[CodeLocation]) -> list[CodeLocation]:
        unique: list[CodeLocation] = []
        seen: set[tuple[str, int, int, str]] = set()
        for row in rows:
            key = (row.path.casefold(), row.line, row.column, row.kind)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        return unique

    @classmethod
    def _normalize_locations(
        cls, raw: Any, workspace: Path, *, kind: str
    ) -> list[CodeLocation]:
        if raw is None:
            return []
        items = raw if isinstance(raw, list) else [raw]
        locations: list[CodeLocation] = []
        seen: set[tuple[str, int, int]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri") or item.get("targetUri")
            region = item.get("range") or item.get("targetSelectionRange") or item.get("selectionRange")
            if not isinstance(uri, str) or not isinstance(region, dict):
                continue
            start = region.get("start", {})
            end = region.get("end", {})
            path = cls._uri_path(uri)
            if path is None:
                continue
            try:
                resolved = path.resolve()
                display = resolved.relative_to(workspace.resolve()).as_posix()
            except ValueError:
                continue
            start_line = int(start.get("line", 0))
            end_line = int(end.get("line", start_line))
            start_character = cls._utf16_to_character(resolved, start_line, int(start.get("character", 0)))
            end_character = cls._utf16_to_character(resolved, end_line, int(end.get("character", 0)))
            key = (display.casefold(), start_line, start_character)
            if key in seen:
                continue
            seen.add(key)
            locations.append(CodeLocation(
                path=display,
                line=key[1] + 1,
                column=key[2] + 1,
                end_line=end_line + 1,
                end_column=end_character + 1,
                kind=kind,
                container=item.get("name") if isinstance(item.get("name"), str) else None,
            ))
        return locations

    @staticmethod
    def _utf16_to_character(path: Path, line: int, offset: int) -> int:
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError):
            return offset
        if not (0 <= line < len(lines)):
            return offset
        units = 0
        for index, character in enumerate(lines[line]):
            next_units = units + len(character.encode("utf-16-le")) // 2
            if next_units > offset:
                return index
            units = next_units
        return len(lines[line])

    @staticmethod
    def _uri_path(uri: str) -> Path | None:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path)

    async def close_all(self) -> None:
        for (session_key, uri), _state in list(self._open_documents.items()):
            client = self._sessions.get(session_key)
            if client is not None and client.alive:
                try:
                    await client.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
                except Exception:
                    pass
        self._open_documents.clear()
        self._document_locks.clear()
        await asyncio.gather(
            *(client.close() for client in self._sessions.values()), return_exceptions=True
        )
        self._sessions.clear()
