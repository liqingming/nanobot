"""Append-only raw transcripts for compacted sessions."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptPage:
    """One reverse-chronological page, returned in normal display order."""

    messages: list[dict[str, Any]]
    before_offset: int | None
    has_older: bool


class TranscriptArchive:
    def __init__(self, sessions_dir: Path):
        self.root = sessions_dir / ".transcripts"
        self.known: dict[str, set[str]] = {}

    def path_for(self, key: str) -> Path:
        name = base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")
        return self.root / f"{name}.jsonl"

    def sync(self, key: str, messages: list[dict[str, Any]]) -> None:
        path = self.path_for(key)
        known = self.known.get(key)
        if known is None:
            known = set()
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if isinstance(row.get("id"), str):
                            known.add(row["id"])
            except (OSError, json.JSONDecodeError):
                pass
            self.known[key] = known
        missing = [
            m
            for m in messages
            if isinstance(m.get("_transcript_id"), str) and m["_transcript_id"] not in known
        ]
        if not missing:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for message in missing:
                f.write(
                    json.dumps(
                        {"id": message["_transcript_id"], "message": message}, ensure_ascii=False
                    )
                    + "\n"
                )
                known.add(message["_transcript_id"])

    def read(self, key: str) -> list[dict[str, Any]]:
        try:
            with open(self.path_for(key), encoding="utf-8") as f:
                return [
                    row["message"]
                    for line in f
                    if line.strip()
                    if isinstance((row := json.loads(line)).get("message"), dict)
                ]
        except (OSError, json.JSONDecodeError):
            return []

    def read_turn_page(
        self,
        key: str,
        *,
        before_offset: int | None = None,
        turn_limit: int = 30,
    ) -> TranscriptPage:
        """Read up to ``turn_limit`` complete user turns backwards from disk.

        ``before_offset`` is an opaque byte cursor returned by the previous call.
        Reading proceeds from the file tail in bounded chunks, so opening a large
        transcript does not deserialize the whole archive.
        """
        path = self.path_for(key)
        if turn_limit <= 0:
            return TranscriptPage([], before_offset, bool(before_offset))
        try:
            with open(path, "rb") as f:
                file_size = f.seek(0, 2)
                end = file_size if before_offset is None else max(0, min(before_offset, file_size))
                position = end
                carry = b""
                newest_first: list[tuple[int, dict[str, Any]]] = []
                user_turns = 0
                oldest_offset: int | None = None

                while position > 0 and user_turns < turn_limit:
                    chunk_size = min(64 * 1024, position)
                    position -= chunk_size
                    f.seek(position)
                    chunk = f.read(chunk_size) + carry
                    lines = chunk.split(b"\n")
                    carry = lines.pop(0) if position > 0 else b""
                    offsets: list[int] = []
                    cursor = position + (len(carry) if position > 0 else 0)
                    if position > 0:
                        cursor += 1
                    for raw in lines:
                        offsets.append(cursor)
                        cursor += len(raw) + 1
                    for line_offset, raw in reversed(list(zip(offsets, lines, strict=True))):
                        if not raw.strip() or line_offset >= end:
                            continue
                        try:
                            row = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        message = row.get("message") if isinstance(row, dict) else None
                        if not isinstance(message, dict):
                            continue
                        newest_first.append((line_offset, message))
                        oldest_offset = line_offset
                        if message.get("role") == "user":
                            user_turns += 1
                            if user_turns >= turn_limit:
                                break

                # A transcript with no user rows (for example proactive delivery)
                # is still a valid final page.
                messages = [message for _offset, message in reversed(newest_first)]
                has_older = bool(oldest_offset is not None and oldest_offset > 0)
                return TranscriptPage(
                    messages,
                    oldest_offset if has_older else None,
                    has_older,
                )
        except OSError:
            return TranscriptPage([], None, False)

    def delete(self, key: str) -> None:
        self.known.pop(key, None)
        self.path_for(key).unlink(missing_ok=True)
