"""Throughput benchmarks and the regression floor.

`make bench` saves a baseline; `make bench-check` fails on a >20% regression. Both are
local-only by design -- absolute microsecond figures are machine- and thermal-dependent,
and a CI runner's numbers would be noise pretending to be a gate.

What *is* asserted unconditionally, because it is a ratio rather than an absolute and so
survives a slow machine: the Rust core must beat the Python oracle by a wide margin, on
figures taken **back to back in this same process**. The Python baseline has been observed
at 5.9, 6.5 and 7.5 us/step for identical code across sessions, so a ratio taken against a
number recorded on another day measures the weather, not the port.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from ticket_to_ride.cli.cmd_bench import median_result, random_playouts, rust_playouts

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

#: The Phase 2 exit criterion (PLAN.md §14). Asserted at a margin below the measured
#: figures so ordinary noise does not fail the build; see the module docstring in
#: `cmd_bench` for why the comparison is same-session.
MIN_SPEEDUP_SINGLE_THREAD = 35.0

#: Measured single-thread on an M2 Max: 46x (usa 2P), 43x (usa 4P), 52x (mini 2P). Recorded
#: so a future reader can tell "slower machine" from "the port regressed".
RECORDED_SINGLE_THREAD = {("usa", 2): 46.2, ("usa", 4): 42.8, ("mini", 2): 52.0}


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


@pytest.mark.slow
@pytest.mark.parametrize(("map_name", "n_players"), [("usa", 2), ("usa", 4), ("mini", 2)])
def test_rust_is_far_faster_than_the_oracle(
    map_name: str, n_players: int, rust: ModuleType
) -> None:
    """The exit criterion, as a same-session ratio.

    Marked slow rather than bench because it is a correctness-of-the-port claim, not a
    timing record: if the Rust engine ever stops being an order of magnitude faster,
    something has gone wrong that is worth failing a build over.
    """
    python = median_result([random_playouts(map_name, n_players, 200) for _ in range(3)])
    rust_1 = median_result([rust_playouts(map_name, n_players, 10_000) for _ in range(3)])
    speedup = python.microseconds_per_step / rust_1.microseconds_per_step
    recorded = RECORDED_SINGLE_THREAD[map_name, n_players]
    assert speedup >= MIN_SPEEDUP_SINGLE_THREAD, (
        f"{map_name} {n_players}P: rust is only {speedup:.1f}x the python oracle "
        f"({python.microseconds_per_step:.2f} vs {rust_1.microseconds_per_step:.3f} us/step); "
        f"{recorded:.0f}x was measured on an M2 Max. A slower machine moves both numbers, "
        "so a drop in the *ratio* means the port regressed."
    )
