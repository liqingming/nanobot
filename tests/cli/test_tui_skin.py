from pathlib import Path

from nanobot.config.schema import AgentDefaults
from nanobot.fork.cli.tui_factory import create_tui


def test_tui_skin_config_defaults_off_and_accepts_camel_case() -> None:
    assert AgentDefaults().tui_skin_enabled is False
    assert AgentDefaults.model_validate({"tuiSkinEnabled": True}).tui_skin_enabled is True


def test_tui_factory_passes_skin_setting_to_textual_backend(tmp_path: Path) -> None:
    tui = create_tui(
        backend="textual",
        skin_enabled=True,
        history_file=str(tmp_path / "history"),
    )

    assert getattr(tui, "_skin_enabled") is True
