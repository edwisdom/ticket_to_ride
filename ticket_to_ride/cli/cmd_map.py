"""`ttr map` -- board data and the invariants it has to satisfy.

The point is not pretty output, it is a second pair of eyes on the generated constants: the
counts, the colour balance, the double routes, the connectivity -- as the engine sees them
(`ttr_rust.board_summary`). `--format ascii` is a
Phase 4 deliverable and needs the lat/lon table that does not exist yet.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ticket_to_ride.engine import board_names, board_summary


def _render(console: Console, data: dict[str, Any]) -> None:
    table = Table(title=f"{data['name']}  ({data['data_hash']})", title_style="bold")
    table.add_column("property", style="cyan")
    table.add_column("value", justify="right")
    for key, value in data.items():
        if key in ("name", "data_hash"):
            continue
        if isinstance(value, dict):
            rendered = "  ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, list):
            rendered = ", ".join(str(v) for v in value)
        else:
            rendered = str(value)
        table.add_row(key.replace("_", " "), rendered)
    console.print(table)


def map_command(
    name: Annotated[str, typer.Option("--map", help="Board to describe.")] = "usa",
    output: Annotated[str, typer.Option("--format", help="table | json")] = "table",
    all_maps: Annotated[bool, typer.Option("--all", help="Describe every board.")] = False,
) -> None:
    """Print a board's data and derived invariants."""
    try:
        boards = [board_summary(n) for n in (board_names() if all_maps else [name])]
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    if output == "json":
        typer.echo(json.dumps(boards if all_maps else boards[0], indent=2))
        return
    if output != "table":
        raise typer.BadParameter(
            f"unknown format {output!r}; use table or json "
            "(ascii needs the lat/lon table, which is Phase 4)"
        )
    console = Console()
    for data in boards:
        _render(console, data)
