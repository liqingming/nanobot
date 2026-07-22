"""Semantic-first code relationship search with bounded text fallbacks."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.registry import register_fork_tool
from nanobot.fork.agent.tools.code_intelligence_manager import CodeIntelligenceManager
from nanobot.fork.agent.tools.lsp_client import LspError
from nanobot.fork.agent.tools.macro_risk import assess_macro_risk, find_asmdef
from nanobot.fork.agent.tools.text_candidates import search_text_candidates
from nanobot.security.workspace_access import current_tool_workspace

_ACTIONS = ("definition", "references", "implementations", "callers", "callees", "candidates")
_MODES = ("semantic", "auto", "exhaustive")


class CodeSearchTool(Tool):
    def __init__(
        self,
        workspace: str | Path,
        *,
        config: dict[str, Any] | None = None,
        manager: CodeIntelligenceManager | None = None,
    ) -> None:
        self._workspace = Path(workspace)
        self._config = dict(config or {})
        self._manager = manager or CodeIntelligenceManager(self._config)

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return (
            "Semantic-first code navigation for definitions, references, implementations, callers, "
            "and callees. Use file+line+column to identify an exact symbol. mode=auto performs a "
            "bounded text-diff only for macro-sensitive code; candidates searches cross-language "
            "text and optional Unity resources. Candidate results are never claimed as bound semantic "
            "references. Prefer this over grep for static code relationships."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(_ACTIONS)},
                "file": {"type": "string", "description": "Source file containing the exact symbol."},
                "line": {"type": "integer", "minimum": 1},
                "column": {"type": "integer", "minimum": 1},
                "symbol": {"type": "string", "description": "Symbol text for fallback/candidate search."},
                "scope": {"type": "string", "description": "Optional workspace-relative fallback scope."},
                "mode": {"type": "string", "enum": list(_MODES)},
                "intent": {"type": "string", "description": "Optional user intent for risk assessment."},
                "include_resources": {
                    "type": "boolean",
                    "description": "Include Unity prefab/scene/asset/material/meta/shader candidates.",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["action"],
        }

    def _workspace_root(self) -> Path:
        access = current_tool_workspace(self._workspace)
        return (access.project_path or self._workspace).resolve()

    @staticmethod
    def _symbol_at(path: Path, line: int, column: int) -> str:
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError):
            return ""
        if not (1 <= line <= len(lines)):
            return ""
        text = lines[line - 1]
        index = min(max(0, column - 1), len(text))
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
            if match.start() <= index <= match.end():
                return match.group(0)
        return ""

    @staticmethod
    def _inside(root: Path, path: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    async def execute(
        self,
        action: str,
        file: str = "",
        line: int = 1,
        column: int = 1,
        symbol: str = "",
        scope: str = "",
        mode: str = "auto",
        intent: str = "",
        include_resources: bool = False,
        max_results: int = 50,
        **_kwargs: Any,
    ) -> str:
        root = self._workspace_root()
        max_results = max(1, min(int(max_results), 200))
        if action not in _ACTIONS or mode not in _MODES:
            return ToolResult.error("Error: unsupported code_search action or mode")
        source = (root / file).resolve() if file else None
        if source is not None and not self._inside(root, source):
            return ToolResult.error("Error: source file must stay inside the active workspace")
        explicit_search_root = (root / scope).resolve() if scope else None
        if explicit_search_root is not None and not self._inside(root, explicit_search_root):
            return ToolResult.error("Error: search scope must stay inside the active workspace")

        risk = assess_macro_risk(source, root, intent=intent) if source is not None else None
        if explicit_search_root is not None:
            search_root = explicit_search_root
        elif risk is not None and risk.suggested_scope == "module" and source is not None:
            asmdef = find_asmdef(source, root)
            search_root = asmdef.parent if asmdef is not None else source.parent
        else:
            search_root = root
        semantic = []
        metadata: dict[str, Any] = {"semantic_available": False, "degraded": False}
        warning: str | None = None
        if action != "candidates":
            if source is None or not source.is_file():
                return ToolResult.error("Error: semantic actions require an existing source file")
            try:
                semantic, metadata = await self._manager.query(
                    root, action=action, path=source, line=line, column=column
                )
            except LspError as exc:
                warning = str(exc)
                metadata = {"semantic_available": False, "degraded": True}
                if mode == "semantic":
                    return json.dumps({
                        "status": "unavailable",
                        "action": action,
                        **metadata,
                        "warning": warning,
                        "results": [],
                    }, ensure_ascii=False)

        if not symbol and source is not None:
            symbol = self._symbol_at(source, line, column)
        should_scan = action == "candidates" or mode == "exhaustive" or (
            mode == "auto" and (not metadata.get("semantic_available") or bool(risk and risk.macro_sensitive))
        )
        candidates: list[dict] = []
        candidate_total = 0
        if should_scan:
            if not symbol:
                warning = warning or "text fallback skipped because symbol was not provided"
            else:
                candidates, candidate_total = await asyncio.to_thread(
                    search_text_candidates,
                    root,
                    search_root,
                    symbol,
                    semantic=semantic,
                    include_resources=include_resources,
                    max_results=max_results,
                )
        limited = semantic[:max_results]
        payload = {
            "status": "ok" if metadata.get("semantic_available") or candidates else "degraded",
            "action": action,
            "mode_requested": mode,
            "mode_used": (
                "semantic+text-diff"
                if should_scan and metadata.get("semantic_available")
                else "text-fallback" if should_scan else "semantic"
            ),
            **metadata,
            "macro_sensitive": bool(risk and risk.macro_sensitive),
            "risk_score": risk.score if risk else 0,
            "risk_reasons": risk.reasons if risk else [],
            "semantic_results": [item.to_dict() for item in limited],
            "semantic_total": len(semantic),
            "text_candidates": candidates,
            "text_candidates_total": candidate_total,
            "truncated": len(semantic) > max_results or candidate_total > len(candidates),
        }
        if warning:
            payload["warning"] = warning
        return json.dumps(payload, ensure_ascii=False)

    async def aclose(self) -> None:
        await self._manager.close_all()

    def summarize_result(self, args: dict[str, Any], result: Any) -> str:
        try:
            data = json.loads(result)
        except Exception:
            return ""
        semantic = int(data.get("semantic_total", 0))
        candidates = int(data.get("text_candidates_total", 0))
        return f"{data.get('mode_used', data.get('status', 'done'))}: {semantic} semantic, {candidates} candidates"


def _factory(loop: Any) -> CodeSearchTool | None:
    config_model = getattr(loop.tools_config, "code_intelligence", None)
    config = config_model.model_dump() if hasattr(config_model, "model_dump") else config_model
    if not isinstance(config, dict):
        config = {}
    if config.get("enabled") is not True:
        return None
    return CodeSearchTool(loop.workspace, config=config)


register_fork_tool(_factory)
