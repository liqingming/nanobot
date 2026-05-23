"""
nanobot - A lightweight AI agent framework
"""

__version__ = "0.1.4.post6"
__logo__ = "🐈"

from nanobot.nanobot import Nanobot, RunResult

# Fork bootstrap — triggers self-registration of fork modules (hook
# handlers, event-bus subscribers, path-resolver callables). When the
# fork tree is absent (bare upstream install / test isolation), this
# is a silent no-op and upstream behavior is preserved.
try:
    import nanobot.fork  # noqa: F401
except ImportError:
    pass

__all__ = ["Nanobot", "RunResult"]
