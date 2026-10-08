"""Workspace cache management for the fork CLI."""
from pathlib import Path

import typer
from rich.console import Console

console = Console()
cache_app = typer.Typer(help="Manage nanobot workspace caches")


@cache_app.command("migrate")
def cache_migrate(
    old_path: str = typer.Argument(..., help="Old workspace directory path"),
    new_path: str = typer.Argument(..., help="New workspace directory path"),
):
    """Migrate workspace cache when a project directory is moved or renamed.

    Computes the cache directory names for both paths and renames the old
    cache to the new one, preserving all sessions, memory, and history.
    """
    import shutil

    from nanobot.config.paths import get_workspace_cache_dir

    old = Path(old_path).expanduser().resolve()
    new = Path(new_path).expanduser().resolve()

    old_cache = get_workspace_cache_dir(old, create=False)
    new_cache = get_workspace_cache_dir(new, create=False)

    if old_cache == new_cache:
        console.print("[yellow]Both paths resolve to the same cache directory — nothing to do.[/yellow]")
        raise typer.Exit(0)

    if not old_cache.exists():
        console.print(f"[red]No cache found for:[/red] {old}")
        console.print(f"[dim]Expected: {old_cache}[/dim]")
        raise typer.Exit(1)

    if new_cache.exists():
        console.print(f"[red]Cache already exists for new path:[/red] {new}")
        console.print(f"[dim]{new_cache}[/dim]")
        console.print("[dim]Remove it first if you want to overwrite.[/dim]")
        raise typer.Exit(1)

    shutil.move(str(old_cache), str(new_cache))
    console.print("[green]✓ Cache migrated[/green]")
    console.print(f"  [dim]{old_cache}[/dim]")
    console.print(f"  [dim]→ {new_cache}[/dim]")


@cache_app.command("list")
def cache_list():
    """List all workspace caches and their associated paths."""
    caches_root = Path.home() / ".nanobot" / "caches"
    if not caches_root.exists() or not any(caches_root.iterdir()):
        console.print("[dim]No workspace caches found.[/dim]")
        return

    console.print(f"[bold]Workspace caches[/bold] [dim]({caches_root})[/dim]\n")
    for d in sorted(caches_root.iterdir()):
        if not d.is_dir():
            continue
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1_048_576
        console.print(f"  [cyan]{d.name}[/cyan]  [dim]{size_mb:.1f} MB[/dim]")
