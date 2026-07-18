from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from nanobot.cli import terminal_skin
from nanobot.cli.terminal_skin import (
    SkinError,
    current_background,
    list_skin_images,
    resolve_skin,
    set_background,
)
from nanobot.config.schema import Config


def _settings(path: Path, image: str = r"C:\old\wallpaper.jpg") -> Path:
    path.write_text(
        "{\n"
        '    "profiles": {\n'
        '        "defaults": {\n'
        f'            "backgroundImage": {json.dumps(image)},\n'
        '            "opacity": 30\n'
        "        },\n"
        '        "list": [{"name": "PowerShell", "backgroundImage": "profile.jpg"}]\n'
        "    },\n"
        '    "theme": "unchanged"\n'
        "}\n",
        encoding="utf-8",
    )
    return path


def test_list_skin_images_uses_natural_order_and_supported_extensions(tmp_path: Path) -> None:
    for name in ("10.jpg", "2.JPG", "1.png", "ignore.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert [path.name for path in list_skin_images(tmp_path)] == ["1.png", "2.JPG", "10.jpg"]


def test_resolve_skin_supports_navigation_number_name_and_random(tmp_path: Path) -> None:
    images = [tmp_path / name for name in ("1.jpg", "2.jpg", "10.jpg")]

    assert resolve_skin("next", images, images[0]) == images[1]
    assert resolve_skin("next", images, None) == images[0]
    assert resolve_skin("prev", images, images[0]) == images[-1]
    assert resolve_skin("prev", images, None) == images[-1]
    assert resolve_skin("2", images) == images[1]
    assert resolve_skin("10.jpg", images) == images[2]
    assert resolve_skin("random", images, images[0], rng=random.Random(1)) != images[0]


def test_resolve_skin_rejects_ambiguous_or_missing_names(tmp_path: Path) -> None:
    images = [tmp_path / "night-one.jpg", tmp_path / "night-two.jpg"]

    with pytest.raises(SkinError, match="不唯一"):
        resolve_skin("night", images)
    with pytest.raises(SkinError, match="未找到"):
        resolve_skin("day", images)


def test_set_background_only_updates_defaults_and_keeps_first_backup(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "settings.json")
    first = tmp_path / "第一张.jpg"
    second = tmp_path / "second.png"
    first.write_bytes(b"image")
    second.write_bytes(b"image")
    original = settings.read_text(encoding="utf-8")

    _, backup = set_background(first, settings)
    first_text = settings.read_text(encoding="utf-8")
    parsed = json.loads(first_text)
    assert Path(parsed["profiles"]["defaults"]["backgroundImage"]) == first.resolve()
    assert parsed["profiles"]["list"][0]["backgroundImage"] == "profile.jpg"
    assert parsed["theme"] == "unchanged"
    assert backup.read_text(encoding="utf-8") == original

    set_background(second, settings)
    assert current_background(settings) == second.resolve()
    assert backup.read_text(encoding="utf-8") == original


def test_set_background_refuses_malformed_settings_without_backup(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{broken", encoding="utf-8")
    image = tmp_path / "1.jpg"
    image.write_bytes(b"image")

    with pytest.raises(json.JSONDecodeError):
        set_background(image, settings)
    assert not settings.with_name("settings.skin-backup.json").exists()


def test_skin_directory_config_uses_camel_case_and_expands_home(tmp_path: Path) -> None:
    config = Config.model_validate({"agents": {"defaults": {"tuiSkinDir": str(tmp_path)}}})

    assert config.agents.defaults.tui_skin_dir == str(tmp_path)
    assert config.model_dump(mode="json", by_alias=True)["agents"]["defaults"]["tuiSkinDir"] == str(tmp_path)


def test_main_uses_configured_skin_directory(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "configured.jpg"
    image.write_bytes(b"image")
    config = Config.model_validate({"agents": {"defaults": {"tuiSkinDir": str(tmp_path)}}})
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)
    monkeypatch.setattr(terminal_skin, "find_terminal_settings", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(terminal_skin, "current_background", lambda _settings: image)

    assert terminal_skin.main(["list"]) == 0
