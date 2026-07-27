"""Python oracle vs Rust core, compared at every step.

The Phase 2 exit criterion. Three tiers:

* **fast** (default run, and pre-commit): a few hundred games. Sized to run in a couple of
  seconds and, more importantly, *shaped* to cover the configurations that can actually
  fail -- see `FAST_CONFIGS` for why that is not "USA 2P and call it a day".
* **golden**: the 84 recorded games in `tests/golden/replays.bin`, replayed through Rust.
* **slow** (nightly): 100k seeds x {2,3,4,5}P, the §14 criterion. ~26 minutes single-core
  by measurement, a few minutes under `-n auto`.

Every tier compares `state_hash()`, `position_hash()`, `legal_actions()`, `is_terminal()`
and `current_player()` after **every** step, never only at the terminal.
"""

from __future__ import annotations

import os
import pathlib
from types import ModuleType

import pytest
from differential import Divergence, compare_game, compare_position, compare_replay

from ticket_to_ride.data.board import BOARDS
from ticket_to_ride.engine.config import RuleConfig
from ticket_to_ride.engine.replay import unpack
from ticket_to_ride.engine.state import Game

# Every (map, seat count) the engine supports.
ALL_CONFIGS = [
    (name, n)
    for name, board in BOARDS.items()
    for n in range(board.raw.min_players, board.raw.max_players + 1)
]

#: Which configurations the fast tier drives, and how many seeds each gets.
#:
#: **This shape was measured, not guessed.** Deleting the end-of-turn refill from the Rust
#: engine -- a real bug, and one Phase 1 only found by hitting it -- survives 60 seeds of
#: USA 2P, 3P and 4P undetected, because reaching it requires the deck *and* the discard to
#: run dry, which those configurations rarely do in a random game. It is caught at mini
#: seed 0 (3P and 4P) and USA 5P seed 8. So the fast tier weights the small map and the
#: full table heavily; a tier that only drove USA 2P would be green and worthless.
FAST_CONFIGS = [
    ("mini", 2, 40),
    ("mini", 3, 40),
    ("mini", 4, 40),
    ("usa", 5, 40),
    ("usa", 2, 20),
    ("usa", 3, 20),
    ("usa", 4, 20),
]

#: Seeds per (map, seats) in the nightly sweep, and the block size each xdist worker takes.
SLOW_SEEDS = int(os.environ.get("TTR_DIFFERENTIAL_SEEDS", "100000"))
SLOW_BLOCK = 2_000


@pytest.fixture(scope="session")
def rust_engine(rust: ModuleType) -> ModuleType:
    return rust


@pytest.mark.parametrize(("map_name", "n_players", "seeds"), FAST_CONFIGS)
def test_engines_agree_at_every_step(
    map_name: str, n_players: int, seeds: int, rust_engine: ModuleType
) -> None:
    for seed in range(seeds):
        compare_game(map_name, n_players, seed, rust=rust_engine)


def test_the_harness_can_actually_fail(rust_engine: ModuleType) -> None:
    """Guard the guard.

    A differential harness that cannot report a difference is worse than none: it
    manufactures confidence. Rather than mutating the engine, this drives two games from
    *different seeds* through the comparison and asserts it objects -- which exercises the
    same comparison and reporting path a real divergence would take.
    """
    py = Game(RuleConfig(map_name="usa", n_players=2)).new_initial_state(1)
    rs = rust_engine.Game("usa", 2).new_initial_state(2)
    with pytest.raises(Divergence) as caught:
        compare_position(py, rs, ("usa", 2, 1, 0, None))
    message = str(caught.value)
    assert "state_hash" in message
    # The whole point of the harness: it must say *where*, not merely *that*.
    assert "first differing field is" in message, message
    assert "reproduce: compare_game" in message, message


def test_golden_replays_reproduce_in_rust(rust_engine: ModuleType) -> None:
    """The 84 recorded games, replayed through Rust and compared step by step.

    Stronger than a random drive in one specific way: the action sequences were written by
    the Python engine before Rust existed, so they reach positions a fresh random sweep may
    never construct, and the recorded final hash is an independent third party rather than
    whatever the two engines happen to agree on today.
    """
    records = unpack(pathlib.Path("tests/golden/replays.bin").read_bytes())
    assert len(records) == 84, f"expected the 84-game corpus, found {len(records)}"
    for record in records:
        compare_replay(record, rust=rust_engine)


@pytest.mark.slow
@pytest.mark.parametrize(("map_name", "n_players"), ALL_CONFIGS)
@pytest.mark.parametrize("block", range(0, 100_000, SLOW_BLOCK))
def test_engines_agree_over_the_full_sweep(
    map_name: str, n_players: int, block: int, rust_engine: ModuleType
) -> None:
    """The §14 criterion: 100k seeds x every seat count, compared at every step.

    Parameterized in blocks so `-n auto` distributes it; measured at ~50k compared-steps/s
    single-core, so the whole sweep is ~26 minutes on one core and a few minutes on twelve.
    """
    if block >= SLOW_SEEDS:
        pytest.skip(f"TTR_DIFFERENTIAL_SEEDS={SLOW_SEEDS} stops before block {block}")
    for seed in range(block, min(block + SLOW_BLOCK, SLOW_SEEDS)):
        compare_game(map_name, n_players, seed, rust=rust_engine)
