"""Append-only raw transcripts for compacted sessions."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


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

    def delete(self, key: str) -> None:
        self.known.pop(key, None)
        self.path_for(key).unlink(missing_ok=True)
