"""Configuration loading utilities."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pydantic
from loguru import logger
from pydantic import BaseModel

from nanobot.config.schema import Config, _resolve_tool_config_refs

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None
_schema_refs_ready = False


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".nanobot" / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    global _schema_refs_ready
    if not _schema_refs_ready:
        _resolve_tool_config_refs()
        _schema_refs_ready = True

    path = config_path or get_config_path()

    config = Config()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            config = Config.model_validate(data)
            _resolve_config_paths(config, path.parent)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            raise ValueError(f"Failed to load config from {path}: {e}") from e

    _apply_network_access_policy(config)
    return config


def _apply_network_access_policy(config: Config) -> None:
    """Apply private-network access and SSRF exceptions to the shared guard."""
    from nanobot.security.network import (
        configure_private_network_access,
        configure_ssrf_whitelist,
    )

    configure_private_network_access(config.tools.allow_private_network_access)
    configure_ssrf_whitelist(config.tools.ssrf_whitelist)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)
    if config.providers.openai_codex.proxy is not None:
        data.setdefault("providers", {})["openaiCodex"] = {
            "proxy": config.providers.openai_codex.proxy,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_active_model_preset(name: str, config_path: Path | None = None) -> Path:
    """Persist only the active model preset without resolving config secrets."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model preset name must be a non-empty string")

    path = config_path or get_config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to update config at {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"Failed to update config at {path}: root must be an object")
        data = loaded

    agents = data.setdefault("agents", {})
    if not isinstance(agents, dict):
        raise ValueError(f"Failed to update config at {path}: agents must be an object")
    defaults = agents.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Failed to update config at {path}: agents.defaults must be an object")

    # Preserve the spelling already used by hand-written configs. Remove the
    # alternate spelling so two conflicting values can never coexist.
    key = "model_preset" if "model_preset" in defaults else "modelPreset"
    defaults[key] = name.strip()
    defaults.pop("modelPreset" if key == "model_preset" else "model_preset", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_config_env_vars(config: Config) -> Config:
    """Return *config* with ``${VAR}`` env-var references resolved.

    Walks in place so fields declared with ``exclude=True`` survive;
    returns the same instance when no references are present.
    Missing variables in optional channel configs disable only that channel.
    References elsewhere remain strict and raise ``ValueError`` when missing.
    """
    extras = config.channels.__pydantic_extra__ or {}
    updates: dict[str, Any] = {}
    for name, section in extras.items():
        try:
            updates[name] = _resolve_in_place(section)
        except ValueError as exc:
            resolved = _resolve_env_vars_or_empty(section)
            if not isinstance(resolved, dict):
                raise
            resolved["enabled"] = False
            updates[name] = resolved
            logger.warning("{} channel disabled: {}", name, exc)
    if updates:
        channels = config.channels.model_copy()
        channels.__pydantic_extra__ = {**extras, **updates}
        config = config.model_copy(update={"channels": channels})
    return _resolve_in_place(config)


def _resolve_in_place(obj: Any) -> Any:
    if isinstance(obj, str):
        new = _ENV_REF_PATTERN.sub(_env_replace, obj)
        return new if new != obj else obj
    if isinstance(obj, BaseModel):
        updates: dict[str, Any] = {}
        for name in type(obj).model_fields:
            old = getattr(obj, name)
            new = _resolve_in_place(old)
            if new is not old:
                updates[name] = new
        extras = obj.__pydantic_extra__
        new_extras: dict[str, Any] | None = None
        if extras:
            resolved = {k: _resolve_in_place(v) for k, v in extras.items()}
            if any(resolved[k] is not extras[k] for k in extras):
                new_extras = resolved
        if not updates and new_extras is None:
            return obj
        copy = obj.model_copy(update=updates) if updates else obj.model_copy()
        if new_extras is not None:
            copy.__pydantic_extra__ = new_extras
        return copy
    if isinstance(obj, dict):
        resolved = {k: _resolve_in_place(v) for k, v in obj.items()}
        return resolved if any(resolved[k] is not obj[k] for k in obj) else obj
    if isinstance(obj, list):
        resolved = [_resolve_in_place(v) for v in obj]
        return resolved if any(nv is not ov for nv, ov in zip(resolved, obj)) else obj
    return obj


def _resolve_env_vars_or_empty(obj: Any) -> Any:
    """Resolve env references, using an empty value for missing variables."""
    if isinstance(obj, str):
        return _ENV_REF_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), ""), obj
        )
    if isinstance(obj, dict):
        return {key: _resolve_env_vars_or_empty(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars_or_empty(value) for value in obj]
    return obj


def _resolve_env_vars(obj: object) -> object:
    """Recursively resolve ``${VAR}`` patterns in plain strings/dicts/lists."""
    if isinstance(obj, str):
        return _ENV_REF_PATTERN.sub(_env_replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _env_replace(match: re.Match[str]) -> str:
    name = match.group(1)
    value = os.environ.get(name)
    if value is None:
        raise ValueError(
            f"Environment variable '{name}' referenced in config is not set"
        )
    return value


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    agents = data.get("agents", {})
    defaults = agents.get("defaults", {}) if isinstance(agents, dict) else {}
    if isinstance(defaults, dict):
        had_legacy_max_messages = (
            "maxMessages" in defaults or "max_messages" in defaults
        )
        defaults.pop("maxMessages", None)
        defaults.pop("max_messages", None)
        if had_legacy_max_messages:
            # TODO(next version): Remove this legacy cleanup branch; the schema
            # will silently ignore this field once the warning grace period ends.
            logger.warning(
                "agents.defaults.maxMessages/max_messages is legacy and ignored; "
                "replay max messages is now an internal safety cap. Remove it from "
                "config. This compatibility warning will be removed in the next version."
            )

    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # Move tools.myEnabled / tools.mySet → tools.my.{enable, allowSet}.
    # The old flat keys shipped in the initial MyTool landing; wrapping them in a
    # sub-config keeps `web` / `exec` / `my` symmetric and gives room to grow.
    if "myEnabled" in tools or "mySet" in tools:
        my_cfg = tools.setdefault("my", {})
        if "myEnabled" in tools and "enable" not in my_cfg:
            my_cfg["enable"] = tools.pop("myEnabled")
        else:
            tools.pop("myEnabled", None)
        if "mySet" in tools and "allowSet" not in my_cfg:
            my_cfg["allowSet"] = tools.pop("mySet")
        else:
            tools.pop("mySet", None)

    return data


def _resolve_config_paths(config: Config, config_dir: Path) -> None:
    """Resolve config-relative Skill roots and executable search paths."""
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in config.agents.defaults.skill_roots:
        value = str(raw).strip()
        if not value:
            raise ValueError("agents.defaults.skillRoots entries must not be empty")
        normalized = str(_resolve_config_path(value, config_dir))
        key = normalized.casefold() if os.name == "nt" else normalized
        if key not in seen:
            seen.add(key)
            resolved.append(normalized)
    config.agents.defaults.skill_roots = resolved

    exec_config = config.tools.exec
    exec_config.path_prepend = _resolve_config_search_path(exec_config.path_prepend, config_dir)
    exec_config.path_append = _resolve_config_search_path(exec_config.path_append, config_dir)


def _resolve_config_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve(strict=False)


def _resolve_config_search_path(value: str, config_dir: Path) -> str:
    if not value or "${" in value:
        return value
    return os.pathsep.join(
        str(_resolve_config_path(part.strip(), config_dir))
        for part in value.split(os.pathsep)
        if part.strip()
    )
