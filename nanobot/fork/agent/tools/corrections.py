"""Deterministic corrections for recurring tool-call mistakes.

Fork wrappers replace selected core tools after normal discovery. Corrections
are deliberately narrow: only transformations that preserve user intent are
applied without another model round trip.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import register_fork_tool
from nanobot.utils.atomic_write import replace_file_with_retry

_STORE_VERSION = 1
_CORE_FORMAT_PATTERN = r"(?:^|[;&|]\s*)format(?!=)\b"
_SAFE_FORMAT_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)format-(?:custom|hex|list|table|wide)\b",
    re.IGNORECASE,
)
_DISK_FORMAT_PATTERN = r"(?:^|[;&|]\s*)format(?:\.com|\.exe)?(?=\s|$)"
_BOUNDED_FIELDS = {
    "grep": frozenset({"context_before", "context_after"}),
}


_CORRECTION_CATALOG = (
    {
        "fingerprint": "exec:allow:powershell-format-cmdlet",
        "sequence": 1,
        "tool": "exec",
        "kind": "deny_false_positive_avoided",
        "observed_count": 2,
    },
    {
        "fingerprint": "grep:clamp:context_after",
        "sequence": 2,
        "tool": "grep",
        "kind": "parameter_clamp",
        "parameter": "context_after",
        "corrected_to": 20,
        "observed_count": 2,
    },
    {
        "fingerprint": "exec:avoid:blocking-codex-app-server-readline",
        "sequence": 3,
        "tool": "exec",
        "kind": "preventive_hint",
        "observed_count": 1,
        "hint": (
            "For Codex app-server probes, use the provider's async RPC bridge; "
            "do not block on raw subprocess stdout.readline()."
        ),
    },
    {
        "fingerprint": "exec:prefer-powershell-here-string-python",
        "sequence": 4,
        "tool": "exec",
        "kind": "preventive_hint",
        "observed_count": 1,
        "hint": (
            "For multiline Python in PowerShell, pipe a here-string to python - "
            "instead of nesting quoted source in python -c."
        ),
    },
)


class ToolCorrectionStore:
    """Small persistent counter for corrections applied before execution."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "memory" / "tool_corrections.json"
        self._lock = threading.Lock()
        self._entries = self._load()
        self._ensure_catalog()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return {}
        if not isinstance(payload, dict) or payload.get("version") != _STORE_VERSION:
            return {}
        entries = payload.get("corrections")
        return entries if isinstance(entries, dict) else {}

    def record(self, fingerprint: str, **fields: Any) -> None:
        with self._lock:
            old = self._entries.get(fingerprint, {})
            entry = dict(old)
            entry.update(
                {
                    key: value
                    for key, value in fields.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
            )
            entry["hits"] = int(old.get("hits", 0)) + 1
            entry["last_seen"] = int(time.time())
            self._entries[fingerprint] = entry
            self._write()

    def hints_for(self, tool: str) -> tuple[str, ...]:
        return tuple(
            str(entry["hint"])
            for entry in sorted(
                self._entries.values(),
                key=lambda entry: int(entry.get("sequence", 0)),
            )
            if entry.get("tool") == tool and entry.get("kind") == "preventive_hint"
        )

    def _ensure_catalog(self) -> None:
        changed = False
        for correction in _CORRECTION_CATALOG:
            fingerprint = str(correction["fingerprint"])
            if fingerprint in self._entries:
                continue
            self._entries[fingerprint] = {
                key: value for key, value in correction.items() if key != "fingerprint"
            }
            self._entries[fingerprint]["hits"] = 0
            changed = True
        if changed:
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"version": _STORE_VERSION, "corrections": self._entries},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        replace_file_with_retry(tmp, self.path)


class CorrectedTool(Tool):
    """Delegate to a core tool after applying safe fork corrections."""

    _plugin_discoverable = False

    def __init__(self, wrapped: Tool, store: ToolCorrectionStore) -> None:
        self._wrapped = wrapped
        self._store = store
        if wrapped.name == "exec":
            self._fix_exec_deny_pattern()

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        hints = self._store.hints_for(self.name)
        if not hints:
            return self._wrapped.description
        return f"{self._wrapped.description}\nKnown corrections: {' '.join(hints)}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    @property
    def read_only(self) -> bool:
        return self._wrapped.read_only

    @property
    def supports_read_only_calls(self) -> bool:
        return self._wrapped.supports_read_only_calls

    @property
    def exclusive(self) -> bool:
        return self._wrapped.exclusive

    @property
    def concurrency_safe(self) -> bool:
        return self._wrapped.concurrency_safe

    @property
    def execution_timeout_s(self) -> float | None:
        return self._wrapped.execution_timeout_s

    def is_read_only_call(self, params: Any) -> bool:
        return self._wrapped.is_read_only_call(params)

    def is_concurrency_safe_call(self, params: Any) -> bool:
        return self._wrapped.is_concurrency_safe_call(params)

    def set_context(self, ctx: Any) -> None:
        setter = getattr(self._wrapped, "set_context", None)
        if callable(setter):
            setter(ctx)

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        casted = self._wrapped.cast_params(params)
        if not isinstance(casted, dict):
            return casted
        corrected = dict(casted)
        properties = self.parameters.get("properties", {})
        for field in _BOUNDED_FIELDS.get(self.name, ()):
            value = corrected.get(field)
            schema = properties.get(field, {}) if isinstance(properties, dict) else {}
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            bounded = value
            if isinstance(schema.get("minimum"), (int, float)):
                bounded = max(bounded, schema["minimum"])
            if isinstance(schema.get("maximum"), (int, float)):
                bounded = min(bounded, schema["maximum"])
            if bounded == value:
                continue
            corrected[field] = bounded
            self._store.record(
                f"{self.name}:clamp:{field}",
                tool=self.name,
                kind="parameter_clamp",
                parameter=field,
                corrected_to=bounded,
            )
        return corrected

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        return self._wrapped.validate_params(params)

    def to_schema(self) -> dict[str, Any]:
        return self._wrapped.to_schema()

    def summarize_result(self, args: dict[str, Any], result: Any) -> str:
        return self._wrapped.summarize_result(args, result)

    async def execute(self, **kwargs: Any) -> Any:
        command = kwargs.get("command")
        if (
            self.name == "exec"
            and isinstance(command, str)
            and _SAFE_FORMAT_PATTERN.search(command)
        ):
            self._store.record(
                "exec:allow:powershell-format-cmdlet",
                tool="exec",
                kind="deny_false_positive_avoided",
            )
        return await self._wrapped.execute(**kwargs)

    def _fix_exec_deny_pattern(self) -> None:
        patterns = getattr(self._wrapped, "deny_patterns", None)
        if not isinstance(patterns, list):
            return
        self._wrapped.deny_patterns = [
            _DISK_FORMAT_PATTERN if pattern == _CORE_FORMAT_PATTERN else pattern
            for pattern in patterns
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _corrected_tool_factory(name: str):
    def factory(loop: Any) -> Tool | None:
        wrapped = loop.tools.get(name)
        if wrapped is None:
            return None
        store = getattr(loop, "_tool_correction_store", None)
        if store is None:
            store = ToolCorrectionStore(Path(loop.context.data_dir))
            loop._tool_correction_store = store
        return CorrectedTool(wrapped, store)

    factory.__name__ = f"correct_{name}_tool"
    return factory


register_fork_tool(_corrected_tool_factory("exec"))
register_fork_tool(_corrected_tool_factory("grep"))
