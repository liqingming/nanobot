"""Per-topic memory factory — fork addition.

Each session_key (channel:chat_id) gets its own ``MemoryStore`` rooted at
``<data_dir>/memory/topics/<safe_key>/``, while SOUL.md / USER.md remain
shared with the global store through the same ``workspace`` argument.

This reuses upstream :class:`nanobot.agent.memory.MemoryStore` via the
``memory_dir_override`` hook so per-topic memory inherits all of upstream's
capabilities (atomic write, history.jsonl + cursor, Dream cursor, legacy
HISTORY.md migration, oversize entry guard, etc.) for free.
"""
from __future__ import annotations

from pathlib import Path

from nanobot.agent.memory import MemoryStore
from nanobot.utils.helpers import safe_filename


class TopicMemoryFactory:
    """Lazily build / cache per-topic ``MemoryStore`` instances.

    ``data_dir`` is the same root passed to the global ``MemoryStore`` —
    typically ``ContextBuilder.data_dir``.  Topic stores live under
    ``data_dir/memory/topics/<safe_key>/``.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._cache: dict[str, MemoryStore] = {}

    @staticmethod
    def _topic_dir(data_dir: Path, session_key: str) -> Path:
        safe = safe_filename(session_key.replace(":", "_"))
        return data_dir / "memory" / "topics" / safe

    def get(self, session_key: str) -> MemoryStore:
        cached = self._cache.get(session_key)
        if cached is not None:
            return cached
        store = MemoryStore(
            self.data_dir,
            memory_dir_override=self._topic_dir(self.data_dir, session_key),
        )
        self._cache[session_key] = store
        return store
