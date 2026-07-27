"""Determinization: every sample must be consistent with what the observer can see.

The Phase 5 validator sweeps a million samples; this is the Phase 2 version, sized for
the fast suite. It exists now because `resample_from_infoset` is the one search hook that
constrains the **state layout** -- if the unseen-pool arithmetic did not close against the
fields the engine keeps, that is a second-port-pass problem, and PLAN.md §8.1 mitigation 2
exists to catch it while the layout is still free to change.

Checked from Python as well as from Rust because `assert_consistent` is what Phase 5's
validator will call, and a checker nobody exercises across the FFI boundary is a checker
that breaks the first time it is needed.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from ticket_to_ride.data.board import BOARDS

if TYPE_CHECKING:
    # The payoff of shipping a .pyi with py.typed: the Rust core is a typed module here,
    # so a wrong method name fails the build rather than the test run.
    from ttr_rust import State as RustState

ALL_CONFIGS = [
    (name, n)
    for name, board in BOARDS.items()
    for n in range(board.raw.min_players, board.raw.max_players + 1)
]

#: Sample here rather than only at the start: an opening position has almost nothing
#: hidden, so a start-only sweep would barely exercise the sampler.
DEPTHS = (0, 25, 90, 250)


def _advance(state: RustState, steps: int) -> RustState:
    for _ in range(steps):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.step(legal[len(legal) // 2])
    return state


@pytest.mark.parametrize(("map_name", "n_players"), ALL_CONFIGS)
def test_every_determinization_is_consistent(
    map_name: str, n_players: int, rust: ModuleType
) -> None:
    game = rust.Game(map_name, n_players)
    for seed in range(3):
        for depth in DEPTHS:
            state = _advance(game.new_initial_state(seed), depth)
            for observer in range(n_players):
                rng = rust.Rng.stream(seed, "determinize", observer)
                for _ in range(8):
                    sample = state.resample_from_infoset(observer, rng)
                    sample.assert_consistent(state, observer)
                    # A particle search will step must satisfy the engine's own laws too.
                    sample.validate()
                    assert sample.hand_of(observer) == state.hand_of(observer)


def test_determinizations_actually_vary(rust: ModuleType) -> None:
    """A sampler returning the true state every time passes every consistency check.

    So this is the guard on that: with a mid-game position and real hidden information,
    repeated draws must produce different positions.
    """
    state = _advance(rust.Game("usa", 3).new_initial_state(11), 80)
    rng = rust.Rng.stream(11, "vary")
    positions = {state.resample_from_infoset(0, rng).position_hash() for _ in range(25)}
    assert len(positions) > 1, "every determinization was identical"


def test_the_rng_advances_rather_than_being_copied(rust: ModuleType) -> None:
    """`Rng` must be taken by reference.

    PyO3 gives `Clone` pyclasses an automatic `FromPyObject`, which would let the stream be
    passed by value -- silently cloned at the boundary, leaving the caller's generator
    where it started and making every determinization identical. The binding opts out; this
    asserts the opt-out held.
    """
    state = _advance(rust.Game("usa", 2).new_initial_state(3), 60)
    rng = rust.Rng.stream(3, "advance")
    before = rng.state
    state.resample_from_infoset(0, rng)
    assert rng.state != before, "the sampler did not advance the caller's stream"


def test_an_inconsistent_sample_is_rejected(rust: ModuleType) -> None:
    """Guard the guard: a state from a different seed must not pass as a determinization."""
    game = rust.Game("usa", 2)
    state = _advance(game.new_initial_state(1), 60)
    impostor = _advance(game.new_initial_state(2), 60)
    with pytest.raises(RuntimeError):
        impostor.assert_consistent(state, 0)
