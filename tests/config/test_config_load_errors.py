import json

import pytest

from nanobot.config.loader import load_config
from nanobot.config.schema import ApiConfig


def test_load_config_missing_file_uses_defaults(tmp_path) -> None:
    config = load_config(tmp_path / "missing.json")

    assert config.agents.defaults.model


def test_load_config_invalid_json_fails_fast(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{broken json", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to load config"):
        load_config(config_path)


def test_load_config_invalid_schema_fails_fast(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"tools": {"exec": {"timeout": -1}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Failed to load config"):
        load_config(config_path)


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_api_config_requires_key_for_wildcard_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="api_key is not set"):
        ApiConfig(host=host)


def test_api_config_allows_wildcard_host_with_key() -> None:
    config = ApiConfig(host="0.0.0.0", api_key="secret")

    assert config.host == "0.0.0.0"
    assert config.api_key == "secret"


def test_tui_show_tool_preface_defaults_true_and_accepts_camel_case() -> None:
    from nanobot.config.schema import Config

    assert Config().agents.defaults.tui_show_tool_preface is True
    config = Config.model_validate(
        {"agents": {"defaults": {"tuiShowToolPreface": False}}}
    )
    assert config.agents.defaults.tui_show_tool_preface is False


def test_load_config_resolves_skill_roots_and_exec_paths_from_config_dir(tmp_path) -> None:
    config_dir = tmp_path / "deploy" / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "runtime.json"
    config_path.write_text(
        json.dumps({
            "agents": {"defaults": {"skillRoots": ["../../shared/skills"]}},
            "tools": {"exec": {"pathPrepend": "../../shared/cli"}},
        }),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.agents.defaults.skill_roots == [
        str((config_dir / "../../shared/skills").resolve())
    ]
    assert config.tools.exec.path_prepend == str(
        (config_dir / "../../shared/cli").resolve()
    )


def test_load_config_rejects_empty_skill_root(tmp_path) -> None:
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"agents": {"defaults": {"skillRoots": [" "]}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skillRoots entries must not be empty"):
        load_config(config_path)
