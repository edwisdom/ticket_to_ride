"""`ttr map` -- board data and the invariants it has to satisfy.

The point is not pretty output, it is a second pair of eyes on the generated constants: the
counts, the colour balance, the double routes, the connectivity. `--format ascii` is a
Phase 4 deliverable and needs the lat/lon table that does not exist yet.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ticket_to_ride.data.board import BOARDS, NO_SIBLING, UNREACHABLE, Board, get_board
from ticket_to_ride.data.rawmap import GRAY


def summary(board: Board) -> dict[str, object]:
    """Everything worth asserting about a board, as plain data."""
    doubles = sum(1 for s in board.sibling if s != NO_SIBLING) // 2
    colors = Counter(board.seg_color)
    degrees = [len({nb for nb, _ in row}) for row in board.adjacency]
    diameter = max(max(row) for row in board.dist)
    return {
        "name": board.name,
        "data_hash": board.data_hash,
        "cities": board.n_cities,
        "pairs": board.n_pairs,
        "segments": board.n_segments,
        "spaces": board.total_spaces,
        "double_routes": doubles,
        "tickets": board.n_tickets,
        "colors": board.n_colors,
        "cards": board.deck_size,
        "locomotives": board.raw.locomotives,
        "trains_per_player": board.raw.trains_per_player,
        "players": [board.raw.min_players, board.raw.max_players],
        "max_route_length": board.max_len,
        "claim_buckets": len(board.buckets),
        "action_space": board.n_segments * board.n_card_types + 15,
        "gray_segments": colors[GRAY],
        "segments_per_color": {board.color_names[c]: colors[c] for c in range(board.n_colors)},
        "max_degree": max(degrees),
        "mean_degree": round(2 * board.n_pairs / board.n_cities, 2),
        "hubs": sorted(board.cities[c] for c, d in enumerate(degrees) if d == max(degrees)),
        "diameter": diameter,
        "connected": diameter < UNREACHABLE,
    }


def _render(console: Console, board: Board) -> None:
    data = summary(board)
    table = Table(title=f"{board.name}  ({data['data_hash']})", title_style="bold")
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
    boards = list(BOARDS.values()) if all_maps else [get_board(name)]
    if output == "json":
        payload = [summary(b) for b in boards]
        typer.echo(json.dumps(payload if all_maps else payload[0], indent=2))
        return
    if output != "table":
        raise typer.BadParameter(
            f"unknown format {output!r}; use table or json "
            "(ascii needs the lat/lon table, which is Phase 4)"
        )
    console = Console()
    for board in boards:
        _render(console, board)
