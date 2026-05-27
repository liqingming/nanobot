"""BM25-based search over HISTORY.md for cross-session recall."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool

# CJK Unicode ranges: CJK Unified Ideographs + Hiragana + Katakana
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ]+")
# Entry boundary: blank line separator
_ENTRY_SEP = re.compile(r"\n\n+")
# Valid entry: starts with a timestamp header
_ENTRY_HEADER = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]")
# RAW fallback entries written by MemoryStore when LLM consolidation fails
_RAW_ENTRY = re.compile(r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+\[RAW\]")

# ---------------------------------------------------------------------------
# Bilingual synonym table for cross-language query expansion.
# Keys are matched against the full text (lowercase); values are appended as
# extra tokens so BM25 can bridge "超时" ↔ "timeout" etc.
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, list[str]] = {
    # Errors / failures
    "超时": ["timeout", "timed"],
    "timeout": ["超时", "超时错误"],
    "错误": ["error", "err", "报错"],
    "error": ["错误", "报错"],
    "err": ["错误"],
    "报错": ["error", "错误"],
    "失败": ["failed", "failure", "fail"],
    "failed": ["失败", "报错"],
    "failure": ["失败"],
    "异常": ["exception", "error"],
    "exception": ["异常", "错误"],
    "崩溃": ["crash", "crashed"],
    "crash": ["崩溃"],
    # Connections
    "连接": ["connect", "connection"],
    "connect": ["连接"],
    "connection": ["连接", "连接失败"],
    # Actions
    "部署": ["deploy", "deployment"],
    "deploy": ["部署"],
    "deployment": ["部署"],
    "安装": ["install", "pip"],
    "install": ["安装"],
    "更新": ["update", "upgrade"],
    "update": ["更新"],
    "upgrade": ["更新", "升级"],
    "升级": ["upgrade", "update"],
    "创建": ["create", "新建"],
    "create": ["创建", "新建"],
    "新建": ["create", "创建"],
    "删除": ["delete", "remove"],
    "delete": ["删除", "移除"],
    "remove": ["删除", "移除"],
    "移除": ["remove", "delete"],
    # States
    "成功": ["success", "ok", "done"],
    "success": ["成功"],
    "完成": ["done", "complete", "finish"],
    "done": ["完成", "成功"],
    "complete": ["完成"],
    "取消": ["cancel", "abort"],
    "cancel": ["取消"],
    "abort": ["取消", "中断"],
    "中断": ["abort", "cancel"],
    # Tool operations
    "执行": ["exec", "execute", "run"],
    "exec": ["执行", "运行"],
    "execute": ["执行"],
    "run": ["运行", "执行"],
    "运行": ["run", "exec", "执行"],
    "搜索": ["search", "find"],
    "search": ["搜索", "查找"],
    "find": ["搜索", "查找"],
    "查找": ["find", "search"],
    "写入": ["write", "write_file"],
    "write": ["写入"],
    "读取": ["read", "read_file"],
    "read": ["读取"],
    "提交": ["commit", "push"],
    "commit": ["提交"],
    "push": ["推送", "提交"],
    "推送": ["push"],
    # Memory / learning
    "压缩": ["consolidate", "compress"],
    "consolidate": ["压缩", "合并"],
    "合并": ["consolidate", "merge"],
    "merge": ["合并"],
    "记忆": ["memory"],
    "memory": ["记忆"],
    "历史": ["history"],
    "history": ["历史"],
}


def _expand_synonyms(text: str) -> str:
    """Append synonym tokens to text before BM25 tokenization.

    Scans for known terms (case-insensitive) and appends their cross-language
    equivalents so BM25 can match e.g. "超时" when query is "timeout".
    """
    text_lower = text.lower()
    additions: list[str] = []
    for term, syns in _SYNONYMS.items():
        if term in text_lower:
            additions.extend(syns)
    if additions:
        return text + " " + " ".join(additions)
    return text


def _tokenize(text: str) -> list[str]:
    """Whitespace-split with CJK character-level expansion + synonym injection."""
    tokens: list[str] = []
    for word in _expand_synonyms(text).lower().split():
        word = word.strip(".,;:!?\"'()[]{}、，。！？")
        if not word:
            continue
        # Expand CJK runs into individual characters for better BM25 recall
        parts = _CJK_RE.split(word)
        cjk_spans = [m.group() for m in _CJK_RE.finditer(word)]
        merged: list[str] = []
        for i, part in enumerate(parts):
            if part:
                merged.append(part)
            if i < len(cjk_spans):
                merged.extend(list(cjk_spans[i]))
        tokens.extend(merged)
    return tokens or [""]


def _parse_entries(text: str) -> list[str]:
    """Split HISTORY.md into individual entries, dropping RAW fallback dumps."""
    raw = _ENTRY_SEP.split(text.strip())
    return [
        e.strip() for e in raw
        if e.strip()
        and _ENTRY_HEADER.match(e.strip())
        and not _RAW_ENTRY.match(e.strip())
    ]


class SearchHistoryTool(Tool):
    """BM25 search over HISTORY.md with bilingual synonym expansion."""

    # path → (mtime, entries, BM25Okapi) — single-slot cache per file
    _cache: dict[str, tuple[float, list[str], Any]] = {}
    # Fork(perf): merged-index cache — (signature, all_entries, BM25Okapi) where
    # signature = ((path, mtime), ...) over every history file. Reused across
    # searches so the combined BM25 isn't re-tokenized/rebuilt on every execute().
    _combined_cache: tuple[tuple, list[str], Any] | None = None

    def __init__(self, data_dir: Path) -> None:
        self._memory_dir = data_dir / "memory"

    @property
    def name(self) -> str:
        return "search_history"

    @property
    def description(self) -> str:
        return (
            "Search HISTORY.md for past events relevant to a query using BM25 ranking. "
            "More effective than grep for multi-word or semantically related queries. "
            "Supports bilingual queries — searching 'timeout' also matches '超时' entries "
            "and vice versa. Use this to check if a similar error, topic, or task appeared "
            "in a previous session."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for — a keyword, error description, or topic.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5, max 20).",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        }

    def _get_all_history_files(self) -> list[Path]:
        files = []
        global_hist = self._memory_dir / "HISTORY.md"
        if global_hist.exists():
            files.append(global_hist)
        topics_dir = self._memory_dir / "topics"
        if topics_dir.exists():
            for topic_dir in sorted(topics_dir.iterdir()):
                if topic_dir.is_dir():
                    h = topic_dir / "HISTORY.md"
                    if h.exists():
                        files.append(h)
        return files

    def _load_index(self) -> tuple[list[str], Any] | tuple[None, None]:
        """Return cached or freshly built BM25 index merging all history files."""
        from rank_bm25 import BM25Okapi

        history_files = self._get_all_history_files()
        if not history_files:
            return None, None

        # Fork(perf): reuse the merged index when no history file changed.
        signature = tuple((str(f), f.stat().st_mtime) for f in history_files)
        combined = type(self)._combined_cache
        if combined is not None and combined[0] == signature:
            return combined[1], combined[2]

        all_entries: list[str] = []
        for hist_file in history_files:
            path = str(hist_file)
            mtime = hist_file.stat().st_mtime
            cached = self._cache.get(path)
            if cached and cached[0] == mtime:
                file_entries = cached[1]
            else:
                text = hist_file.read_text(encoding="utf-8")
                file_entries = _parse_entries(text)
                if file_entries:
                    file_index = BM25Okapi([_tokenize(e) for e in file_entries])
                    self._cache[path] = (mtime, file_entries, file_index)
            all_entries.extend(file_entries)

        if not all_entries:
            return None, None

        combined_index = BM25Okapi([_tokenize(e) for e in all_entries])
        type(self)._combined_cache = (signature, all_entries, combined_index)
        return all_entries, combined_index

    async def execute(self, query: str, top_k: int = 5) -> str:
        top_k = min(max(1, top_k), 20)
        entries, index = self._load_index()

        if entries is None:
            return "HISTORY.md is empty or does not exist."

        scores = index.get_scores(_tokenize(query))
        ranked = sorted(
            ((score, i) for i, score in enumerate(scores)),
            reverse=True,
        )

        results: list[str] = []
        for score, idx in ranked[:top_k]:
            if score <= 0:
                break
            results.append(f"[score={score:.2f}]\n{entries[idx]}")

        if not results:
            return f"No relevant entries found for: {query!r}"

        header = f"Top {len(results)} results for {query!r} (out of {len(entries)} entries):\n\n"
        return header + "\n\n---\n\n".join(results)


# ── self-registration ────────────────────────────────────────────────────

from nanobot.agent.tools.registry import register_fork_tool  # noqa: E402


def _search_history_factory(loop):
    if not getattr(loop, "enable_learning", False):
        return None
    return SearchHistoryTool(data_dir=loop.context.data_dir)


register_fork_tool(_search_history_factory)
