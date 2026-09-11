"""接收入口原文与证据引用；完整性不是身份认证或授权语义。"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nanobot.bus.events import InboundMessage
from nanobot.fork.agent.tool_evidence import EvidencePersistenceError, persist_tool_evidence
from nanobot.session.automation_turns import automation_history_overrides
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.turn_continuation import should_persist_user_message


@dataclass(frozen=True)
class InputReceipt:
    receipt_id: str
    origin: str
    channel: str
    sender_id: str
    chat_id: str
    text: str
    media: tuple[str, ...]
    received_at: str


@dataclass
class ReceivedInput(InboundMessage):
    # 独立字段而非可由网络 JSON 填入的 metadata；内部 replace 保留原文而非重新盖章。
    receipt: InputReceipt | None = None


def capture_input(
    msg: InboundMessage, origin: str = "unverified", *, original_text: str | None = None,
    original: ReceivedInput | None = None,
) -> ReceivedInput:
    if isinstance(msg, ReceivedInput) and isinstance(msg.receipt, InputReceipt):
        return msg
    receipt = InputReceipt(
        uuid.uuid4().hex, origin, msg.channel, msg.sender_id, msg.chat_id,
        msg.content if original_text is None else original_text,
        tuple(msg.media), msg.timestamp.isoformat(),
    )
    if isinstance(original, ReceivedInput) and isinstance(original.receipt, InputReceipt):
        receipt = original.receipt
    return ReceivedInput(
        **{item.name: getattr(msg, item.name) for item in fields(InboundMessage)},
        receipt=receipt,
    )


def input_origin(msg: InboundMessage, receipt: InputReceipt) -> str:
    # 否定性内部标记优先，绝不从 metadata 的“用户已确认”反推正向授权。
    if not should_persist_user_message(msg.metadata):
        return "internal_continuation"
    _, automation = automation_history_overrides(msg.metadata)
    if automation or is_hidden_history_message(msg.metadata):
        return "automation"
    if msg.sender_id.startswith("system:") or msg.channel == "system":
        return "internal"
    if (msg.channel, msg.sender_id, msg.chat_id) != (
        receipt.channel, receipt.sender_id, receipt.chat_id,
    ):
        return "unverified"
    if msg.content != receipt.text or tuple(msg.media) != receipt.media:
        return "derived_input"
    return receipt.origin if receipt.origin in {"cli_interactive", "sdk"} else "unverified"


def build_input_evidence(msg: InboundMessage, *, data_dir: Any, session_key: str):
    from nanobot.agent.context_artifacts import EvidenceRef

    received = capture_input(msg)
    receipt = received.receipt
    assert receipt is not None
    source = {
        "schema": 1, "receipt_id": receipt.receipt_id, "session_key": session_key,
        "origin": input_origin(msg, receipt), "recorded_at_entry": receipt.origin,
        "channel": receipt.channel, "sender_id": receipt.sender_id, "chat_id": receipt.chat_id,
        "received_at": receipt.received_at,
        "text_sha256": hashlib.sha256(receipt.text.encode("utf-8")).hexdigest(),
        "media_scope": "locators_only; attachment bytes and extracted text are not user instructions",
    }
    payload = json.dumps(
        {"source": source, "text": receipt.text, "media": list(receipt.media)},
        ensure_ascii=False, sort_keys=True,
    )
    locator = persist_tool_evidence(
        SimpleNamespace(data_dir=data_dir, workspace=None, session_key=session_key), payload,
    )
    if locator is None:
        raise EvidencePersistenceError("原始输入没有可靠存储位置，停止而不伪造引用。")
    return EvidenceRef(
        evidence_id=f"input_{receipt.receipt_id}", tool_call_id="",
        kind="input_snapshot", locator=locator,
        sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        trust="runtime_receipt_not_authorization", input_source=source,
    )


def make_input_verifier(data_dir: Any, session_key: Any):
    # 只读运行时缓存中按会话和摘要计算的路径，metadata 不能指定任意文件。
    if not isinstance(data_dir, (str, Path)) or not isinstance(session_key, str) or not session_key:
        return None
    root = Path(data_dir).resolve()
    session_hash = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    remaining = 4 * 1024 * 1024

    def verify(evidence: Any) -> str:
        nonlocal remaining
        digest = evidence.sha256
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return "unverified_invalid_reference"
        path = root / "sessions" / session_hash / "evidence" / f"{digest}.txt"
        try:
            if not path.resolve().is_relative_to(root) or str(path.resolve()) != evidence.locator:
                return "unverified_reference_scope"
            limit = min(1024 * 1024, remaining)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                return "unverified_not_regular_file"
            if info.st_size > limit:
                return "unverified_budget"
            with path.open("rb") as stream:
                raw = stream.read(limit + 1)
            remaining = max(0, remaining - len(raw))
            if len(raw) > limit:
                return "unverified_budget"
            if hashlib.sha256(raw).hexdigest() != digest:
                return "mismatch"
            payload = json.loads(raw)
            source = payload["source"]
            if (
                source != evidence.input_source or source["session_key"] != session_key
                or evidence.evidence_id != f"input_{source['receipt_id']}"
                or source["text_sha256"] != hashlib.sha256(payload["text"].encode("utf-8")).hexdigest()
            ):
                return "mismatch"
            return "snapshot_matches_reference; authorization=unverified"
        except FileNotFoundError:
            return "missing"
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            return "unverified_unavailable"

    return verify
