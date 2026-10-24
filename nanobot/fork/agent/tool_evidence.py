"""工具原始回执的内容寻址快照；持久化失败不能丢弃唯一原文。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from nanobot.utils.atomic_write import replace_file_with_retry


class EvidencePersistenceError(RuntimeError):
    """证据未可靠落盘，必须保留原回执并停止模型续跑。"""


def result_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(
        result, ensure_ascii=False, sort_keys=True, default=str
    )


def persist_tool_evidence(config: Any, result: Any) -> str | None:
    """只在运行时给定的缓存根中写入；路径不使用工具参数或回执内的路径。"""
    # SDK 测试/调用者可能没有真实路径；不能把 MagicMock 的 __fspath__ 当成目录。
    if isinstance(config.data_dir, (str, Path)):
        base = Path(config.data_dir).resolve()
        root = base / "sessions"
    elif isinstance(config.workspace, (str, Path)):
        base = Path(config.workspace).resolve()
        root = base / ".nanobot" / "tool-evidence"
    else:
        # 无存储配置的 SDK 调用仍保留内联原文，不伪称已经持久化。
        return None
    payload = result_text(result).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    session = hashlib.sha256((config.session_key or "default").encode("utf-8")).hexdigest()
    path = root / session / "evidence" / f"{digest}.txt"
    tmp = path.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
    try:
        root = root.resolve()
        if not root.is_relative_to(base):
            raise EvidencePersistenceError("证据根目录超出配置的存储范围。")
        # 创建目录前后都检查，避免已有 junction/symlink 引导根外写入。
        if not path.resolve().is_relative_to(root):
            raise EvidencePersistenceError("证据路径超出缓存根，拒绝写入。")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.resolve().is_relative_to(root):
            raise EvidencePersistenceError("证据路径超出缓存根，拒绝写入。")
        if path.exists():
            if path.read_bytes() != payload:
                raise EvidencePersistenceError("证据指纹冲突或文件损坏，拒绝覆盖原快照。")
        else:
            with tmp.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            replace_file_with_retry(tmp, path)
            if path.read_bytes() != payload:
                raise EvidencePersistenceError("证据落盘校验失败。")
            if os.name != "nt":
                fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        return str(path.resolve())
    except Exception as exc:
        if isinstance(exc, EvidencePersistenceError):
            raise
        raise EvidencePersistenceError("工具证据保存失败，原始回执保留；停止后续模型请求。") from exc
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def evidence_preview(result: Any, locator: str | None, max_chars: int) -> Any:
    if locator is None or max_chars <= 0:
        return result

    rendered = result_text(result)
    if not _contains_inline_binary(result) and len(rendered) <= max_chars:
        return result
    header = (
        "[tool output persisted]\n"
        f"Full output saved to: {locator}\n"
        f"Original serialized size: {len(rendered)} chars\n"
        "Historical tool-output snapshot; not proof of current file state.\nPreview:\n"
    )
    # Inline binary must never be replayed into model history.  The complete
    # receipt is already durable at ``locator``; retain only a bounded,
    # structural preview so later turns know that an image/artifact existed.
    preview = result_text(_replace_inline_binary(result))
    return header + preview[:max(0, min(1200, max_chars - len(header)))]


def _contains_inline_binary(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("data:") and ";base64," in value[:256]
    if isinstance(value, list):
        return any(_contains_inline_binary(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_inline_binary(item) for item in value.values())
    return False


def _replace_inline_binary(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:256]:
            mime = value[5:value.find(";")] or "application/octet-stream"
            return f"[inline base64 {mime} omitted; see persisted tool output]"
        return value
    if isinstance(value, list):
        return [_replace_inline_binary(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_inline_binary(item) for key, item in value.items()}
    return value
