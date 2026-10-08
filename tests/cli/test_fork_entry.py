"""Protect fork routing through the upstream CLI entry point."""
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config


@pytest.mark.parametrize("backend", ["textual", "prompt_toolkit"])
def test_default_agent_preserves_configured_fork_tui(monkeypatch, tmp_path: Path, backend: str):
    config = Config()
    config.agents.defaults.tui_backend = backend
    seen = {}

    def launch(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("nanobot.cli.agent._load_runtime_config", lambda *_args: config)
    monkeypatch.setattr("nanobot.fork.cli.agent.agent", launch)
    result = CliRunner().invoke(app, ["agent", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert seen["workspace"] == str(tmp_path)
    assert seen["session_id"] == "cli:direct"
    assert seen["message"] is None
    assert config.agents.defaults.tui_backend == backend


def test_cache_commands_remain_available():
    result = CliRunner().invoke(app, ["cache", "--help"])
    assert result.exit_code == 0
    assert "migrate" in result.output
    assert "list" in result.output
