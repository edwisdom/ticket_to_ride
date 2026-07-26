"""`ttr bench` -- engine throughput, reported honestly.

The number that matters is microseconds per step, not games per second: games per second
mixes in how long a random game happens to be, which differs by map and seat count. Both
are printed so the Phase 2 target (>=50x, ~85-170k games/s batched) has a baseline to
measure against.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, NamedTuple

import typer
from rich.console import Console
from rich.table import Table

from ticket_to_ride.data.board import BOARDS
from ticket_to_ride.engine.config import RuleConfig
from ticket_to_ride.engine.rng import stream
from ticket_to_ride.engine.state import Game


class BenchResult(NamedTuple):
    map_name: str
    n_players: int
    games: int
    steps: int
    seconds: float

    @property
    def games_per_second(self) -> float:
        return self.games / self.seconds

    @property
    def microseconds_per_step(self) -> float:
        return self.seconds / self.steps * 1e6

    @property
    def steps_per_game(self) -> float:
        return self.steps / self.games


def random_playouts(map_name: str, n_players: int, games: int, warmup: int = 5) -> BenchResult:
    """Play `games` uniformly-random games and time them.

    `track_history` is off: copying the action list would dominate a clone, and no consumer
    of this number wants the history.
    """
    game = Game(RuleConfig(map_name=map_name, n_players=n_players, track_history=False))
    for seed in range(warmup):
        state = game.new_initial_state(seed)
        rng = stream(seed, "bench")
        while not state.is_terminal():
            state.step(state.sample_legal(rng))

    steps = 0
    start = time.perf_counter()
    for seed in range(games):
        state = game.new_initial_state(seed)
        rng = stream(seed, "bench")
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
            steps += 1
    elapsed = time.perf_counter() - start
    return BenchResult(map_name, n_players, games, steps, elapsed)


def bench_command(
    suite: Annotated[str, typer.Option("--suite", help="engine | all")] = "engine",
    games: Annotated[int, typer.Option("--games", help="Games per configuration.")] = 300,
    output: Annotated[str, typer.Option("--format", help="table | json")] = "table",
) -> None:
    """Measure engine throughput with uniformly random play."""
    if suite == "engine":
        configurations = [("usa", 2), ("usa", 4), ("mini", 2)]
    elif suite == "all":
        configurations = [
            (name, n)
            for name, board in BOARDS.items()
            for n in range(board.raw.min_players, board.raw.max_players + 1)
        ]
    else:
        raise typer.BadParameter(f"unknown suite {suite!r}; use engine or all")

    results = [random_playouts(m, n, games) for m, n in configurations]

    if output == "json":
        typer.echo(
            json.dumps(
                [
                    {
                        "map": r.map_name,
                        "players": r.n_players,
                        "games_per_second": round(r.games_per_second, 1),
                        "microseconds_per_step": round(r.microseconds_per_step, 2),
                        "steps_per_game": round(r.steps_per_game, 1),
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    table = Table(title=f"random playouts, {games} games per configuration")
    table.add_column("map", style="cyan")
    table.add_column("seats", justify="right")
    table.add_column("games/s", justify="right")
    table.add_column("us/step", justify="right")
    table.add_column("steps/game", justify="right")
    for r in results:
        table.add_row(
            r.map_name,
            str(r.n_players),
            f"{r.games_per_second:,.0f}",
            f"{r.microseconds_per_step:.2f}",
            f"{r.steps_per_game:.0f}",
        )
    Console().print(table)
