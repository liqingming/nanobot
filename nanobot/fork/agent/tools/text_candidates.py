"""Bounded text and Unity-resource candidate search used by code_search."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from nanobot.fork.agent.tools.code_intelligence_models import CodeLocation

_TEXT_SUFFIXES = frozenset({
    ".cs", ".lua", ".xml", ".json", ".yaml", ".yml", ".uss", ".uxml",
})
_UNITY_SUFFIXES = frozenset({
    ".prefab", ".unity", ".asset", ".mat", ".meta", ".shader", ".shadergraph", ".asmdef",
})
_SKIP_DIRS = frozenset({".git", "library", "temp", "logs", "obj", "bin", "node_modules"})
_DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|elif|else|endif)\b\s*(.*)$")


def _condition_at(lines: list[str], line_index: int) -> list[str]:
    stack: list[str] = []
    for line in lines[: line_index + 1]:
        match = _DIRECTIVE_RE.match(line)
        if not match:
            continue
        command, expression = match.groups()
        if command == "if":
            stack.append(expression.strip() or "unknown")
        elif command == "elif" and stack:
            stack[-1] = expression.strip() or "unknown"
        elif command == "else" and stack:
            stack[-1] = f"!({stack[-1]})"
        elif command == "endif" and stack:
            stack.pop()
    return stack


def _iter_files(root: Path, suffixes: frozenset[str]) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.casefold() in suffixes:
            yield root
        return
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.casefold() not in suffixes:
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if any(part.casefold() in _SKIP_DIRS for part in relative_parts):
            continue
        yield path


def search_text_candidates(
    workspace: Path,
    root: Path,
    symbol: str,
    *,
    semantic: Iterable[CodeLocation] = (),
    include_resources: bool = False,
    max_results: int = 20,
) -> tuple[list[dict], int]:
    suffixes = _TEXT_SUFFIXES | (_UNITY_SUFFIXES if include_resources else frozenset())
    known = {(item.path.casefold(), item.line, item.column) for item in semantic}
    candidates: list[dict] = []
    total = 0
    pattern = re.compile(re.escape(symbol))
    workspace_resolved = workspace.resolve()
    for path in _iter_files(root, suffixes):
        try:
            resolved = path.resolve(strict=True)
            display = resolved.relative_to(workspace_resolved).as_posix()
        except (OSError, ValueError):
            continue
        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index, text in enumerate(lines):
            for match in pattern.finditer(text):
                location = (display.casefold(), index + 1, match.start() + 1)
                if location in known:
                    continue
                total += 1
                if len(candidates) >= max_results:
                    continue
                conditions = (
                    _condition_at(lines, index) if path.suffix.casefold() == ".cs" else []
                )
                suffix = path.suffix.casefold()
                if suffix in _UNITY_SUFFIXES:
                    classification = "unity_resource_candidate"
                elif suffix == ".cs":
                    classification = "csharp_text_candidate"
                elif suffix == ".lua":
                    classification = "cross_language_candidate"
                else:
                    classification = "configuration_candidate"
                candidates.append({
                    "path": display,
                    "line": index + 1,
                    "column": match.start() + 1,
                    "classification": classification,
                    "condition_stack": conditions,
                    "snippet": text.strip()[:240],
                })
    return candidates, total
