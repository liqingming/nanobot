"""Register fork-bundled skills without changing the upstream skills tree."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from nanobot.agent.skills import SkillsLoader

_FORK_SKILLS_DIR = Path(__file__).parent / "builtin_skills"
_ORIGINAL_INIT = SkillsLoader.__init__


def install_fork_skill_root() -> None:
    """Prepend the fork skill root once to every new loader."""
    if getattr(SkillsLoader, "_fork_skill_root_installed", False):
        return

    @wraps(_ORIGINAL_INIT)
    def init_with_fork_skills(self: SkillsLoader, *args: Any, **kwargs: Any) -> None:
        roots = list(kwargs.pop("extra_skill_roots", None) or [])
        has_custom_builtin = len(args) >= 2 or kwargs.get("builtin_skills_dir") is not None
        fork_root = _FORK_SKILLS_DIR.resolve()
        if not has_custom_builtin and not any(
            Path(root).resolve() == fork_root for _source, root in roots
        ):
            roots.insert(0, ("fork-builtin", _FORK_SKILLS_DIR))
        _ORIGINAL_INIT(self, *args, extra_skill_roots=roots, **kwargs)

    SkillsLoader.__init__ = init_with_fork_skills
    SkillsLoader._fork_skill_root_installed = True


install_fork_skill_root()
