"""Factory: pick the right TUI backend at runtime.

Selection order:
1. ``NANOBOT_TUI`` environment variable (``prompt_toolkit`` / ``textual``)
2. Falls back to ``prompt_toolkit`` (the original backend)

Usage::

    from nanobot.fork.cli.tui_factory import create_tui
    tui = create_tui(render_markdown=True, history_file=..., model=...)
    await tui.run_async()
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.fork.cli.tui_base import TUIBase

_ENV_KEY = "NANOBOT_TUI"


def create_tui(
    render_markdown: bool = True,
    history_file: str | None = None,
    model: str | None = None,
    backend: str = "textual",
) -> "TUIBase":
    # Environment variable overrides config value
    backend = os.environ.get(_ENV_KEY, backend).strip().lower()

    if backend == "textual":
        from nanobot.fork.cli.tui_textual import TextualTUI
        return TextualTUI(
            render_markdown=render_markdown,
            history_file=history_file,
            model=model,
        )

    # Default: original prompt_toolkit backend
    from nanobot.fork.cli.tui import PromptTUI
    return PromptTUI(
        render_markdown=render_markdown,
        history_file=history_file,
        model=model,
    )
