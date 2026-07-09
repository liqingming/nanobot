"""Compatibility helpers for oauth-cli-kit versions with optional proxy support."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def call_with_optional_proxy(func: Callable[..., T], /, *args: Any, proxy: str | None = None, **kwargs: Any) -> T:
    """Call *func* with ``proxy`` only when its signature accepts it."""
    if proxy and _accepts_keyword(func, "proxy"):
        kwargs["proxy"] = proxy
    return func(*args, **kwargs)


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and param.name == name:
            return True
    return False