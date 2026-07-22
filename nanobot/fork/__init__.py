"""nanobot fork — fork-only code mirrored from the upstream package.

All files here are fork additions that have no upstream counterpart.
Directory layout mirrors ``nanobot/`` so the upstream module path is
obvious from the fork path (e.g. ``nanobot/fork/agent/learning.py``
extends ``nanobot/agent/``).

Importing this package triggers each fork submodule's import-time
self-registration: hook handlers, event-bus subscribers, path-resolver
callables, etc. Each fork module is responsible for its own
``paths.X = my_X`` / ``bus.on("event", my_handler)`` line at module
bottom, so behavior lives next to implementation — core files only
carry minimal patches (1-3 lines): a callable registry slot or an
``await bus.emit(...)`` call. All business logic stays here.

This module is empty for now — phases 1+ of the fork-mirror refactor
populate it as files migrate from the core tree.
"""

# Tool self-registration — importing each module triggers its
# ``register_fork_tool(...)`` call at the bottom of the file.
from nanobot.fork.agent.tools import (  # noqa: F401
    ask_user,
    code_search,
    memory_search,
    skill,
    todo,
)
