"""文件回执绑定实际读取字节；恢复核验不扩大已有读取能力。"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import ToolResult

_MAX_CHECK_BYTES = 8 * 1024 * 1024
_TOTAL_CHECK_BYTES = 16 * 1024 * 1024


class FileReadResult(ToolResult):
    """保持字符串回执兼容，运行时旁路字段不从回执文字中解析。"""

    def __new__(cls, content: str, source: dict[str, Any] | None = None):
        obj = super().__new__(cls, content)
        obj.source_evidence = source or {}
        return obj


def text_read_result(
    content: str, path: Path, raw: bytes, *, start: int, end: int, total: int,
) -> FileReadResult:
    # raw 就是本次解码、编号和截取的字节，不再重新读文件生成“读取时指纹”。
    return FileReadResult(content, {
        "schema": 1, "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
        "coverage": "returned_text_lines", "start_line": start, "end_line": end,
        "total_lines": total, "complete": start == 1 and end == total,
    })


def source_from_result(result: Any) -> dict[str, Any]:
    return dict(result.source_evidence) if isinstance(result, FileReadResult) else {}


def make_source_verifier(workspace: Any, tool: Any, metadata: Any):
    """只复核工作区内、现有 read_file 及请求策略共同允许的普通文件。"""
    from nanobot.agent.tools.filesystem import ReadFileTool

    if not isinstance(workspace, (str, Path)) or not isinstance(tool, ReadFileTool):
        return None
    policy = metadata.get("tool_policy", {}) if isinstance(metadata, dict) else {}
    if not isinstance(policy, dict):
        return None
    blocked_tools = policy.get("blocked_tool_names") or []
    if not isinstance(blocked_tools, list):
        return None
    if policy.get("disable_all_tools") is True or "read_file" in blocked_tools:
        return None
    try:
        root = Path(workspace).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    remaining = _TOTAL_CHECK_BYTES
    cache: dict[str, dict[str, Any]] = {}

    def check(source: dict[str, Any]) -> dict[str, Any]:
        nonlocal remaining
        unknown = {"status": "unverified", "reason": "invalid_source"}
        if (
            source.get("schema") != 1 or not isinstance(source.get("path"), str)
            or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64
        ):
            return unknown
        requested = source["path"]
        try:
            # 显式检查当前请求策略，不能依赖构建上下文时尚未绑定的 ContextVar。
            blocked = policy.get("blocked_read_file_paths") or []
            if not isinstance(blocked, list):
                return {"status": "unverified", "reason": "read_policy"}
            path = Path(requested)
            if not path.is_absolute():
                return unknown
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                return {"status": "unverified", "reason": "outside_workspace"}
            for item in blocked:
                denied = Path(str(item))
                denied = (denied if denied.is_absolute() else root / denied).resolve()
                if resolved == denied or resolved.is_relative_to(denied):
                    return {"status": "unverified", "reason": "read_policy"}
            # 复用工具的能力专属路径解析，不借自动核验绕过 restricted/sandbox 边界。
            if tool._resolve_read(requested) != resolved:
                return {"status": "unverified", "reason": "path_changed"}
            key = str(resolved)
            if key not in cache:
                before = resolved.stat()
                if not stat.S_ISREG(before.st_mode):
                    return {"status": "unverified", "reason": "not_regular_file"}
                if before.st_size > min(_MAX_CHECK_BYTES, remaining):
                    return {"status": "unverified", "reason": "verification_budget"}
                read_limit = min(_MAX_CHECK_BYTES, remaining)
                with resolved.open("rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode):
                        return {"status": "unverified", "reason": "not_regular_file"}
                    raw = stream.read(read_limit + 1)
                    after = os.fstat(stream.fileno())
                remaining = max(0, remaining - len(raw))
                if len(raw) > read_limit:
                    return {"status": "unverified", "reason": "verification_budget"}
                current = resolved.stat()
                def signature(s):
                    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)

                # Windows 3.12 的 stat/fstat 对 ctime 语义不同，只在同一 API 内比较。
                if not (
                    signature(before) == signature(opened) == signature(after) == signature(current)
                    and before.st_ctime_ns == current.st_ctime_ns
                    and opened.st_ctime_ns == after.st_ctime_ns
                ):
                    return {"status": "unverified", "reason": "changed_during_check"}
                cache[key] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw), "mtime_ns": current.st_mtime_ns,
                }
            observed = cache[key]
            return {
                "status": "unchanged_at_check" if observed["sha256"] == source["sha256"] else "changed",
                **observed,
                "scope": "source_bytes_only; not semantic validity or future freshness",
            }
        except FileNotFoundError:
            return {"status": "missing"}
        except (OSError, RuntimeError, ValueError):
            return {"status": "unverified", "reason": "unavailable_or_denied"}

    return check
