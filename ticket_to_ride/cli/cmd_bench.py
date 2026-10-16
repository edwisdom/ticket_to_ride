"""`ttr bench` -- engine throughput, reported honestly.

The number that matters is microseconds per step, not games per second: games per second
mixes in how long a random game happens to be, which differs by map and seat count.

Two figures are reported, and conflating them is how a speedup gets overstated:

* **rust-1** -- the engine single-threaded, driven entirely inside Rust. One FFI call for
  the whole batch, because a Python-driven loop pays a few microseconds per step in call
  overhead, which on a sub-microsecond engine is the only thing it would measure.
* **rust-N** -- the same across a rayon pool sized to the machine's performance cores.

`--suite vecenv` measures what training actually runs: the batched loop of
`observe_current -> legal_masks -> masked pick -> step -> auto_reset`, timed phase by
phase, with the policy replaced by a uniformly random legal pick in numpy.

Absolute figures depend on the machine and its thermal state, so `--repeat` takes a median
rather than a single sample, and a figure is only ever compared with one taken on the same
machine in the same session.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, NamedTuple

import numpy as np
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


class VecEnvBench(NamedTuple):
    """One timed run of the training loop's environment side, per phase."""

    map_name: str
    n_players: int
    n_envs: int
    env_steps: int
    observe: float
    mask: float
    pick: float
    step: float
    reset: float

    def per_step(self, seconds: float) -> float:
        """Microseconds of wall time per env-step."""
        return seconds / self.env_steps * 1e6

    @property
    def env_seconds(self) -> float:
        """Everything Rust does: observe, mask, step, reset."""
        return self.observe + self.mask + self.step + self.reset

    @property
    def env_steps_per_second(self) -> float:
        return self.env_steps / self.env_seconds

    @property
    def loop_steps_per_second(self) -> float:
        """Including the numpy action pick a policy would replace."""
        return self.env_steps / (self.env_seconds + self.pick)


def vecenv_loop(
    map_name: str, n_players: int, n_envs: int, seconds: float = 2.0, seed: int = 0
) -> VecEnvBench:
    """Time `observe_current -> legal_masks -> masked pick -> step -> auto_reset`.

    The training loop with the policy replaced by a uniformly random legal pick in numpy,
    so what is measured is the environment and the buffers it fills. Phases are timed
    separately because they scale differently: observe is the encoder across the rayon
    pool, mask and step are cheap, and the pick is Python-side work a policy replaces.
    """
    env = ttr_rust.VecEnv(map_name, n_players, n_envs, seed, track_history=False)
    obs = np.zeros((n_envs, env.obs_size), dtype=np.float32)
    mask = np.zeros((n_envs, env.n_actions), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    times = [0.0] * 5
    env_steps = 0
    clock = time.perf_counter
    start = clock()
    while clock() - start < seconds:
        t0 = clock()
        env.observe_current(obs)
        t1 = clock()
        env.legal_masks(mask)
        t2 = clock()
        keys = rng.random(mask.shape, dtype=np.float32)
        keys[mask == 0] = -1.0
        actions = keys.argmax(axis=1).tolist()
        t3 = clock()
        env.step(actions)
        t4 = clock()
        env.auto_reset()
        t5 = clock()
        for i, (a, b) in enumerate(((t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5))):
            times[i] += b - a
        env_steps += n_envs
    return VecEnvBench(map_name, n_players, n_envs, env_steps, *times)


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
    suite: Annotated[
        str, typer.Option("--suite", help="engine | agents | vecenv | all")
    ] = "engine",
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
    if suite == "vecenv":
        _bench_vecenv(output=output)
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
        raise typer.BadParameter(f"unknown suite {suite!r}; use engine, agents, vecenv or all")

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


def _bench_vecenv(*, output: str) -> None:
    """`ttr bench --suite vecenv`: the batched training loop, phase by phase.

    Microseconds of wall time per env-step. Observe runs across the rayon pool, so at large
    batches its figure is CPU per observation divided by the cores; at one environment it
    is the single-threaded cost.
    """
    rows = [vecenv_loop(m, n, envs) for m, n in (("usa", 2), ("mini", 2)) for envs in (1, 64, 1024)]
    if output == "json":
        typer.echo(
            json.dumps(
                [
                    {
                        "map": r.map_name,
                        "players": r.n_players,
                        "n_envs": r.n_envs,
                        **{
                            phase: round(r.per_step(getattr(r, phase)), 3)
                            for phase in ("observe", "mask", "pick", "step", "reset")
                        },
                        "env_steps_per_second": round(r.env_steps_per_second),
                        "loop_steps_per_second": round(r.loop_steps_per_second),
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        return
    table = Table(title="vecenv loop, us of wall time per env-step (random legal actions)")
    for column in ("map", "seats", "envs", "observe", "mask", "pick", "step", "reset"):
        table.add_column(column, justify="right")
    table.add_column("env steps/s", justify="right")
    table.add_column("loop steps/s", justify="right")
    for r in rows:
        table.add_row(
            r.map_name,
            str(r.n_players),
            str(r.n_envs),
            *(
                f"{r.per_step(getattr(r, phase)):.2f}"
                for phase in ("observe", "mask", "pick", "step", "reset")
            ),
            f"{r.env_steps_per_second:,.0f}",
            f"{r.loop_steps_per_second:,.0f}",
        )
    Console().print(table)
    Console().print(
        "[dim]env steps/s counts what Rust does (observe, mask, step, reset); loop steps/s "
        "adds the numpy pick a policy would replace.[/]"
    )
