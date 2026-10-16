"""Throughput benchmarks.

`make bench` saves a baseline; `make bench-check` fails on a >20% regression. Both are
local-only by design -- absolute microsecond figures are machine- and thermal-dependent,
and a CI runner's numbers would be noise pretending to be a gate.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from ticket_to_ride.cli.cmd_bench import rust_playouts, vecenv_loop

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.bench
@pytest.mark.parametrize(("map_name", "n_players"), [("usa", 2), ("usa", 4), ("mini", 2)])
def test_engine_throughput(
    benchmark: BenchmarkFixture, map_name: str, n_players: int, rust: ModuleType
) -> None:
    """Record the Rust engine's microseconds per step under pytest-benchmark."""
    result = rust_playouts(map_name, n_players, 20_000)
    benchmark.extra_info.update(
        {
            "engine": "rust-1",
            "map": map_name,
            "players": n_players,
            "microseconds_per_step": round(result.microseconds_per_step, 4),
            "games_per_second": round(result.games_per_second, 1),
            "steps_per_game": round(result.steps_per_game, 1),
        }
    )
    benchmark(lambda: rust_playouts(map_name, n_players, 2_000))


@pytest.mark.bench
def test_batched_throughput(benchmark: BenchmarkFixture, rust: ModuleType) -> None:
    threads = rust.performance_threads()
    result = rust_playouts("usa", 2, 40_000, threads)
    benchmark.extra_info.update(
        {
            "engine": f"rust-{threads}",
            "threads": threads,
            "games_per_second": round(result.games_per_second, 1),
            "microseconds_per_step": round(result.microseconds_per_step, 4),
        }
    )
    benchmark(lambda: rust_playouts("usa", 2, 8_000, threads))


@pytest.mark.bench
@pytest.mark.parametrize(("map_name", "n_envs"), [("usa", 1), ("usa", 1024), ("mini", 1024)])
def test_vecenv_throughput(benchmark: BenchmarkFixture, map_name: str, n_envs: int) -> None:
    """The training loop's environment side: observe, mask, pick, step, reset.

    `make bench-check` gates a 20% regression on these, phase figures recorded alongside
    so a regression can be located without re-running anything.
    """
    result = vecenv_loop(map_name, 2, n_envs, seconds=2.0)
    benchmark.extra_info.update(
        {
            "map": map_name,
            "n_envs": n_envs,
            **{
                f"{phase}_us": round(result.per_step(getattr(result, phase)), 3)
                for phase in ("observe", "mask", "pick", "step", "reset")
            },
            "env_steps_per_second": round(result.env_steps_per_second),
        }
    )
    benchmark(lambda: vecenv_loop(map_name, 2, n_envs, seconds=0.5))
