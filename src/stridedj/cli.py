"""Command-line entry point (`stridedj ...`). Thin: parses arguments and delegates to the composition root."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from stridedj.config import load_settings
from stridedj.doctor import CheckStatus, run_checks

app = typer.Typer(help="StrideDJ: plan a run, book the slot, match the music.", no_args_is_help=True)
console = Console()

_ICON = {
    CheckStatus.OK: "[green]✓[/]",
    CheckStatus.MISSING: "[red]✗[/]",
    CheckStatus.FAILED: "[red]✗[/]",
    CheckStatus.PENDING: "[yellow]…[/]",
}


@app.command()
def doctor(
    offline: Annotated[
        bool, typer.Option("--offline", help="Only check that values are present (no network).")
    ] = False,
) -> None:
    """Check every credential in .env. Prints ✓/✗ and hints, never the values."""
    settings = load_settings()
    results = asyncio.run(run_checks(settings, live=not offline))
    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("check")
    table.add_column("detail")
    for r in results:
        table.add_row(_ICON[r.status], r.name, r.detail)
    console.print(table)
    if any(r.status in (CheckStatus.MISSING, CheckStatus.FAILED) for r in results):
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed version."""
    from importlib.metadata import version as pkg_version

    console.print(pkg_version("stridedj"))


if __name__ == "__main__":  # pragma: no cover
    app()
