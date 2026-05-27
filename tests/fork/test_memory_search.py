"""Tests for fork SearchHistoryTool, focused on the H3 merged-index cache.

The combined BM25 index used to be re-tokenized and rebuilt on every
``execute`` call. These tests pin the caching contract: the merged index is
reused while no history file changes, and rebuilt once any file's mtime moves.
"""

import os

import pytest

from nanobot.fork.agent.tools.memory_search import SearchHistoryTool


@pytest.fixture(autouse=True)
def _clear_caches():
    # Caches are class-level; isolate tests from each other.
    SearchHistoryTool._cache.clear()
    SearchHistoryTool._combined_cache = None
    yield
    SearchHistoryTool._cache.clear()
    SearchHistoryTool._combined_cache = None


def _write_history(data_dir, content):
    memory_dir = data_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    hist = memory_dir / "HISTORY.md"
    hist.write_text(content, encoding="utf-8")
    return hist


def test_merged_index_reused_when_unchanged(tmp_path):
    _write_history(
        tmp_path,
        "[2026-05-27 10:00] deployed the service successfully\n\n"
        "[2026-05-27 11:00] connection timeout error occurred\n",
    )
    tool = SearchHistoryTool(data_dir=tmp_path)

    entries1, idx1 = tool._load_index()
    entries2, idx2 = tool._load_index()

    assert idx1 is not None
    assert idx1 is idx2  # merged BM25 reused, not rebuilt
    assert entries1 is entries2


def test_merged_index_rebuilt_after_file_change(tmp_path):
    hist = _write_history(tmp_path, "[2026-05-27 10:00] first entry here\n")
    tool = SearchHistoryTool(data_dir=tmp_path)
    _, idx1 = tool._load_index()

    # Append an entry and force mtime forward so the signature differs.
    hist.write_text(
        "[2026-05-27 10:00] first entry here\n\n"
        "[2026-05-27 12:00] a brand new second entry\n",
        encoding="utf-8",
    )
    st = hist.stat()
    os.utime(hist, (st.st_atime, st.st_mtime + 10))

    _, idx2 = tool._load_index()
    assert idx2 is not idx1  # rebuilt after the file changed


async def test_execute_returns_ranked_results(tmp_path):
    _write_history(
        tmp_path,
        "[2026-05-27 10:00] deployed the service successfully\n\n"
        "[2026-05-27 11:00] connection timeout error occurred\n",
    )
    tool = SearchHistoryTool(data_dir=tmp_path)

    out = await tool.execute(query="timeout", top_k=5)
    assert "timeout" in out.lower()
