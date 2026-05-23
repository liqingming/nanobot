"""Fork extension points — small modules registering to core callable
slots or event-bus events. Layout deliberately *not* mirroring the
upstream tree because these are cross-cutting (e.g. a CLI extension
that hooks into both ``cli/commands.py`` and ``config/paths.py``).
"""
