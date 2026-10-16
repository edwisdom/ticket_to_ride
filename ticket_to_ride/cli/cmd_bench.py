"""`ttr bench` -- engine throughput, reported honestly.

The number that matters is microseconds per step, not games per second: games per second
mixes in how long a random game happens to be, which differs by map and seat count.

Two figures are reported, and conflating them is how a speedup gets overstated:

* **rust-1** -- the engine single-threaded, driven entirely inside Rust. One FFI call for
  the whole batch, because a Python-driven loop pays a few microseconds per step in call
  overhead, which on a sub-microsecond engine is the only thing it would measure.
* **rust-N** -- the same across a rayon pool sized to the machine's performance cores.

Absolute figures depend on the machine and its thermal state, so `--repeat` takes a median
rather than a single sample, and a figure is only ever compared with one taken on the same
machine in the same session.
"""

from __future__ import annotations

import json
from typing import Annotated, NamedTuple

import ttr_rust
import typer
from rich.console import Console
from rich.table import Table

from ticket_to_ride.data.board import BOARDS


class BenchResult(NamedTuple):
    map_name: str
    n_players: int
    games: int
    steps: int
    seconds: float
    engine: str

    @property
    def games_per_second(self) -> float:
        return self.games / self.seconds

    @property
    def microseconds_per_step(self) -> float:
        return self.seconds / self.steps * 1e6

    @property
    def steps_per_game(self) -> float:
        return self.steps / self.games


def rust_playouts(map_name: str, n_players: int, games: int, threads: int = 1) -> BenchResult:
    """Play `games` uniformly random games entirely inside Rust and time them."""
    played, steps, seconds = ttr_rust.random_playouts(map_name, n_players, games, 0, threads)
    engine = "rust-1" if threads <= 1 else f"rust-{threads}"
    return BenchResult(map_name, n_players, played, steps, seconds, engine)


#: The scripted ladder, in order.
TIERS = ("h0", "h1", "h2", "h3", "h4")


class AgentBench(NamedTuple):
    """What one heuristic costs per decision."""

    map_name: str
    n_players: int
    tier: str
    decisions: int
    seconds: float

    @property
    def microseconds_per_decision(self) -> float:
        return self.seconds / self.decisions * 1e6


def agent_playouts(map_name: str, n_players: int, tier: str, blocks: int) -> AgentBench:
    """Self-play a tier and time its decisions.

    **Single-threaded, deliberately.** This number is a per-decision cost, and it is read
    as one: H3 is the ISMCTS rollout policy from Phase 5, so a rollout's sim budget is set
    by exactly this figure on one core. Dividing a parallel wall-clock by the decision count
    would report a throughput and call it a latency.
    """
    games, _ = ttr_rust.run_arena(
        map_name, n_players, [(tier, None, 1)], [0] * n_players, 0, blocks, 1
    )
    return AgentBench(map_name, n_players, tier, sum(games.decisions), games.seconds)


def median_result(runs: list[BenchResult]) -> BenchResult:
    """The median run by microseconds per step: one sample is at the mercy of whatever else
    the machine was doing."""
    return sorted(runs, key=lambda r: r.microseconds_per_step)[len(runs) // 2]


def _collect(
    configurations: list[tuple[str, int]], *, games: int, repeat: int, threads: int
) -> list[BenchResult]:
    """Both figures measured back to back on each configuration, medians taken.

    Interleaved by configuration so the two see the same thermal state.
    """
    results: list[BenchResult] = []
    for m, n in configurations:
        results.append(median_result([rust_playouts(m, n, games) for _ in range(repeat)]))
        if threads > 1:
            runs = [rust_playouts(m, n, games * 4, threads) for _ in range(repeat)]
            results.append(median_result(runs))
    return results


def bench_command(
    suite: Annotated[str, typer.Option("--suite", help="engine | agents | all")] = "engine",
    games: Annotated[
        int, typer.Option("--games", help="Single-threaded games per configuration.")
    ] = 15_000,
    repeat: Annotated[int, typer.Option("--repeat", help="Runs per figure; median wins.")] = 3,
    output: Annotated[str, typer.Option("--format", help="table | json")] = "table",
) -> None:
    """Measure engine throughput with uniformly random play."""
    if suite == "agents":
        _bench_agents(games=max(20, games // 500), repeat=repeat)
        return
    if suite == "engine":
        configurations = [("usa", 2), ("usa", 4), ("mini", 2)]
    elif suite == "all":
        configurations = [
            (name, n)
            for name, board in BOARDS.items()
            for n in range(board.raw.min_players, board.raw.max_players + 1)
        ]
    else:
        raise typer.BadParameter(f"unknown suite {suite!r}; use engine, agents or all")

    threads = ttr_rust.performance_threads()
    results = _collect(configurations, games=games, repeat=repeat, threads=threads)

    if output == "json":
        typer.echo(
            json.dumps(
                [
                    {
                        "map": r.map_name,
                        "players": r.n_players,
                        "engine": r.engine,
                        "games_per_second": round(r.games_per_second, 1),
                        "microseconds_per_step": round(r.microseconds_per_step, 4),
                        "steps_per_game": round(r.steps_per_game, 1),
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    table = Table(title=f"random playouts, median of {repeat} runs ({games} games per figure)")
    table.add_column("map", style="cyan")
    table.add_column("seats", justify="right")
    table.add_column("engine", style="magenta")
    table.add_column("games/s", justify="right")
    table.add_column("us/step", justify="right")
    table.add_column("steps/game", justify="right")
    for r in results:
        table.add_row(
            r.map_name,
            str(r.n_players),
            r.engine,
            f"{r.games_per_second:,.0f}",
            f"{r.microseconds_per_step:.3f}",
            f"{r.steps_per_game:.0f}",
        )
    Console().print(table)


def _bench_agents(*, games: int, repeat: int) -> None:
    """`ttr bench --suite agents`: microseconds per decision for each scripted tier.

    Why this has its own suite: **H3 doubles as the ISMCTS rollout policy** (PLAN.md §7,
    §8.3), so its per-decision cost is what sets the sim budget of every Phase 5 search. It
    is far more expensive than a step -- a plan rebuild is a Steiner solve -- and
    discovering that in Phase 5 would be the expensive version.
    """
    configurations = [("usa", 2), ("usa", 4), ("mini", 2)]
    table = Table(title=f"heuristic cost, self-play, single-threaded (median of {repeat})")
    table.add_column("map", style="cyan")
    table.add_column("seats", justify="right")
    for tier in TIERS:
        table.add_column(tier, justify="right")
    for map_name, n_players in configurations:
        row = [map_name, str(n_players)]
        for tier in TIERS:
            runs = [agent_playouts(map_name, n_players, tier, games) for _ in range(repeat)]
            median = sorted(runs, key=lambda r: r.microseconds_per_decision)[len(runs) // 2]
            row.append(f"{median.microseconds_per_decision:.2f}")
        table.add_row(*row)
    Console().print(table)
    Console().print(
        "[dim]microseconds per decision. H3 is the Phase 5 rollout policy, so its column "
        "is the one that bounds a search's sim count.[/]"
    )
