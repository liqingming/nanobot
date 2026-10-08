"""Compatibility with the fork's pre-namespace per-workspace session cache."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nanobot.config.paths import get_workspace_cache_dir
from nanobot.session.manager import JsonlSessionStore, SessionManager


@pytest.fixture
def cache_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    workspace = tmp_path / "project"
    root = get_workspace_cache_dir(workspace) / "sessions"
    root.mkdir()
    return workspace, root


def write_session(root, key="cli:old", text="original conversation"):
    path = root / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    path.write_text(
        json.dumps({"_type": "metadata", "key": key,
                    "created_at": "2026-09-01T00:00:00",
                    "updated_at": "2026-09-01T00:00:00", "metadata": {}}) + "\n"
        + json.dumps({"role": "user", "content": text}) + "\n", encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("legacy_name", [False, True])
def test_flat_cache_topics_and_artifacts_are_imported_once(cache_workspace, legacy_name):
    workspace, root = cache_workspace
    src = write_session(root)
    if legacy_name:
        src = src.rename(root / "cli_old.jsonl")
    original = src.read_bytes()
    artifacts = [".transcripts/old.jsonl", "cli_old/runtime.log",
                 "telegram_old/evidence/result.txt", "a" * 64 + "/receipt.txt"]
    for name in artifacts:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original evidence")
    foreign = root / ("b" * 32)
    foreign.mkdir()
    write_session(foreign, "cli:foreign")

    manager = SessionManager(workspace, sessions_root=root)
    assert [s["key"] for s in manager.list_sessions()] == ["cli:old"]
    assert manager.get_or_create("cli:old").messages[0]["content"] == "original conversation"
    assert src.read_bytes() == original
    assert manager._get_session_path("cli:old").read_bytes() == original
    for name in artifacts:
        assert (manager.sessions_dir / name).read_bytes() == (root / name).read_bytes()
    assert not (manager.sessions_dir / foreign.name).exists()
    assert manager.delete_session("cli:old")
    restarted = SessionManager(workspace, sessions_root=root)
    assert restarted.list_sessions() == []  # Retained originals must not resurrect a topic.
    assert src.read_bytes() == original


def test_import_does_not_overwrite_new_session(cache_workspace):
    workspace, root = cache_workspace
    manager = SessionManager(workspace, sessions_root=root)
    current = manager.get_or_create("cli:old")
    current.add_message("user", "new version")
    manager.save(current)
    src = write_session(root)
    restarted = SessionManager(workspace, sessions_root=root)
    assert restarted.get_or_create("cli:old").messages[0]["content"] == "new version"
    assert "original conversation" in src.read_text(encoding="utf-8")


def test_import_retries_after_copy_failure(cache_workspace):
    workspace, root = cache_workspace
    src = write_session(root)
    with patch.object(JsonlSessionStore, "_prepare_copy", side_effect=OSError("disk full")):
        failed = SessionManager(workspace, sessions_root=root)
    assert src.exists()
    assert not (failed.sessions_dir / ".legacy-cache-imported").exists()
    restarted = SessionManager(workspace, sessions_root=root)
    assert restarted.get_or_create("cli:old").messages
    assert (restarted.sessions_dir / ".legacy-cache-imported").exists()


def test_shared_custom_root_is_not_claimed_as_legacy_cache(cache_workspace):
    workspace, root = cache_workspace
    shared = root.parent / "shared"
    shared.mkdir()
    write_session(shared)
    manager = SessionManager(workspace, sessions_root=shared)
    assert manager.list_sessions() == []
    assert not (manager.sessions_dir / ".legacy-cache-imported").exists()


def test_linked_cache_entry_is_not_imported(cache_workspace, monkeypatch):
    workspace, root = cache_workspace
    src = write_session(root)
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == src or real_is_symlink(p))
    manager = SessionManager(workspace, sessions_root=root)
    assert manager.list_sessions() == []
    assert src.exists()


def test_cache_path_lookup_can_be_read_only(cache_workspace):
    workspace, root = cache_workspace
    other = workspace.parent / "other"
    path = get_workspace_cache_dir(other, create=False)
    assert not path.exists()
