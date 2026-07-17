"""Windows Terminal background image discovery and switching.

The updater intentionally preserves the user's settings.json formatting and only
replaces ``profiles.defaults.backgroundImage``.  The first mutation keeps an
untouched sibling backup for easy recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_DEFAULT_SKIN_DIR = Path.home() / "cmdSkins"
_STORE_SETTINGS = (
    Path.home()
    / "AppData"
    / "Local"
    / "Packages"
    / "Microsoft.WindowsTerminal_8wekyb3d8bbwe"
    / "LocalState"
    / "settings.json"
)
_PREVIEW_SETTINGS = (
    Path.home()
    / "AppData"
    / "Local"
    / "Packages"
    / "Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe"
    / "LocalState"
    / "settings.json"
)
_UNPACKAGED_SETTINGS = (
    Path.home() / "AppData" / "Local" / "Microsoft" / "Windows Terminal" / "settings.json"
)


class SkinError(RuntimeError):
    """A user-facing skin switch failure."""


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def list_skin_images(skin_dir: Path = _DEFAULT_SKIN_DIR) -> list[Path]:
    """Return supported images in natural filename order."""
    if not skin_dir.is_dir():
        raise SkinError(f"背景图目录不存在: {skin_dir}")
    return sorted(
        (path.resolve() for path in skin_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=_natural_key,
    )


def find_terminal_settings() -> Path:
    """Find the active Windows Terminal settings file."""
    override = os.environ.get("NANOBOT_TERMINAL_SETTINGS")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path.resolve()
        raise SkinError(f"指定的 Windows Terminal 配置不存在: {path}")
    candidates = [path for path in (_STORE_SETTINGS, _PREVIEW_SETTINGS, _UNPACKAGED_SETTINGS) if path.is_file()]
    if not candidates:
        raise SkinError("未找到 Windows Terminal settings.json")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _find_object(text: str, key: str, start: int = 0, end: int | None = None) -> tuple[int, int]:
    """Return the brace bounds of a named JSON object without reformatting JSON."""
    limit = len(text) if end is None else end
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', text[start:limit])
    if match is None:
        raise SkinError(f"Windows Terminal 配置缺少对象: {key}")
    opening = start + match.end() - 1
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, limit):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return opening, index + 1
    raise SkinError(f"Windows Terminal 配置中的 {key} 对象不完整")


def _defaults_bounds(text: str) -> tuple[int, int]:
    profiles_start, profiles_end = _find_object(text, "profiles")
    return _find_object(text, "defaults", profiles_start + 1, profiles_end - 1)


def current_background(settings_path: Path | None = None) -> Path | None:
    """Read the current defaults background image."""
    path = settings_path or find_terminal_settings()
    text = path.read_text(encoding="utf-8")
    start, end = _defaults_bounds(text)
    match = re.search(r'"backgroundImage"\s*:\s*("(?:\\.|[^"\\])*")', text[start:end])
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SkinError(f"backgroundImage 值无效: {exc}") from exc
    return Path(value).resolve() if value else None


def _updated_settings_text(text: str, image: Path) -> str:
    json.loads(text)  # Refuse to touch malformed settings.
    start, end = _defaults_bounds(text)
    defaults = text[start:end]
    encoded = json.dumps(str(image.resolve()), ensure_ascii=False)
    pattern = re.compile(r'("backgroundImage"\s*:\s*)"(?:\\.|[^"\\])*"')
    if pattern.search(defaults):
        updated_defaults = pattern.sub(lambda match: match.group(1) + encoded, defaults, count=1)
    else:
        line_start = text.rfind("\n", 0, start) + 1
        object_indent = text[line_start:start]
        child_indent = object_indent + "    "
        inner = defaults[1:-1]
        if inner.strip():
            updated_defaults = "{" + f'\n{child_indent}"backgroundImage": {encoded},' + inner
        else:
            updated_defaults = "{" + f'\n{child_indent}"backgroundImage": {encoded}\n{object_indent}' + "}"
    updated = text[:start] + updated_defaults + text[end:]
    json.loads(updated)
    return updated


def set_background(image: Path, settings_path: Path | None = None) -> tuple[Path, Path]:
    """Atomically update the Terminal background and return settings/backup paths."""
    image = image.expanduser().resolve()
    if not image.is_file() or image.suffix.lower() not in _IMAGE_SUFFIXES:
        raise SkinError(f"背景图不存在或格式不支持: {image}")
    settings = (settings_path or find_terminal_settings()).resolve()
    original = settings.read_text(encoding="utf-8")
    updated = _updated_settings_text(original, image)
    backup = settings.with_name("settings.skin-backup.json")
    if not backup.exists():
        shutil.copy2(settings, backup)
    if updated != original:
        fd, temp_name = tempfile.mkstemp(prefix="settings.skin-", suffix=".tmp", dir=settings.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, settings)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return settings, backup


def resolve_skin(
    selector: str,
    images: Sequence[Path],
    current: Path | None = None,
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> Path:
    """Resolve next/prev/random/index/name selectors to an image."""
    if not images:
        raise SkinError("背景图目录中没有支持的图片")
    value = selector.strip()
    lowered = value.casefold()
    current_index = next((i for i, path in enumerate(images) if current and path == current), -1)
    if lowered == "next":
        return images[(current_index + 1) % len(images)]
    if lowered == "prev":
        return images[-1] if current_index < 0 else images[(current_index - 1) % len(images)]
    if lowered == "random":
        choices = list(images)
        if len(choices) > 1 and current in choices:
            choices.remove(current)
        return (rng or random.SystemRandom()).choice(choices)
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(images):
            return images[index - 1]
        raise SkinError(f"编号超出范围: 1-{len(images)}")
    exact = [path for path in images if value.casefold() in {path.name.casefold(), path.stem.casefold()}]
    if len(exact) == 1:
        return exact[0]
    partial = [path for path in images if lowered in path.name.casefold()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise SkinError("文件名匹配不唯一: " + ", ".join(path.name for path in partial))
    raise SkinError(f"未找到背景图: {value}")


def format_skin_list(images: Sequence[Path], current: Path | None) -> str:
    lines = ["可用 Windows Terminal 背景图:"]
    for index, image in enumerate(images, 1):
        marker = " *" if current == image else ""
        lines.append(f"  {index:>2}. {image.name}{marker}")
    if current is not None and current not in images:
        lines.append(f"\n当前配置: {current}")
    lines.append("\n* 表示当前背景")
    return "\n".join(lines)


def switch_skin(selector: str, skin_dir: Path = _DEFAULT_SKIN_DIR) -> tuple[Path, Path, Path]:
    images = list_skin_images(skin_dir)
    settings = find_terminal_settings()
    selected = resolve_skin(selector, images, current_background(settings))
    settings, backup = set_background(selected, settings)
    return selected, settings, backup


def _interactive_selector(images: Sequence[Path], current: Path | None) -> str:
    print(format_skin_list(images, current))
    try:
        return input("请选择编号（直接回车取消）: ").strip()
    except EOFError:
        return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skin", description="切换 Windows Terminal 背景图")
    parser.add_argument("selector", nargs="?", help="list/next/prev/random/编号/文件名")
    parser.add_argument("--dir", dest="skin_dir", type=Path, default=_DEFAULT_SKIN_DIR)
    args = parser.parse_args(argv)
    try:
        images = list_skin_images(args.skin_dir)
        settings = find_terminal_settings()
        current = current_background(settings)
        selector = args.selector
        if selector is None:
            selector = _interactive_selector(images, current)
            if not selector:
                print("已取消。")
                return 0
        if selector.casefold() == "list":
            print(format_skin_list(images, current))
            return 0
        selected = resolve_skin(selector, images, current)
        _, backup = set_background(selected, settings)
        print(f"已切换背景图: {selected.name}")
        print(f"配置文件: {settings}")
        print(f"原始备份: {backup}")
        return 0
    except (SkinError, json.JSONDecodeError, OSError) as exc:
        print(f"切换失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
