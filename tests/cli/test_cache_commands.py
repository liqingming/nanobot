"""Workspace cache command behavior."""

from pathlib import Path

from nanobot.config.paths import get_workspace_cache_dir
from nanobot.fork.cli.cache import cache_migrate


def test_cache_migrate_moves_workspace_identity(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    old_workspace = tmp_path / "project-old"
    new_workspace = tmp_path / "project-new"
    old_workspace.mkdir()
    new_workspace.mkdir()
    old_cache = get_workspace_cache_dir(old_workspace)
    (old_cache / "workspace-id").write_text("a" * 32 + "\n", encoding="utf-8")
    expected_new_cache = get_workspace_cache_dir(new_workspace, create=False)

    cache_migrate(str(old_workspace), str(new_workspace))

    assert not old_cache.exists()
    assert (expected_new_cache / "workspace-id").read_text(encoding="utf-8").strip() == "a" * 32
