"""The golden replay corpus, and the wider determinism sweeps.

`tests/golden/replays.bin` holds 84 finished games -- random and greedy, across every map
and seat count -- each carrying the actions that produced it plus the state hash and scores
it must produce again. It is the strongest regression guard in the project: a rules change,
a draw-order change, a PRNG change or a scoring change shows up here as a concrete failing
game rather than as a subtly different win rate three weeks later.

The heavy sweeps are marked `slow` and run nightly. The corpus check is not: it is under a
second and it guards the property everything else depends on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ticket_to_ride.data.board import BOARDS
from ticket_to_ride.engine import Game, RuleConfig, final_scores
from ticket_to_ride.engine.replay import record, replay, unpack
from ticket_to_ride.engine.rng import stream

CONFIGURATIONS = [
    (name, n)
    for name, board in BOARDS.items()
    for n in range(board.raw.min_players, board.raw.max_players + 1)
]


@pytest.fixture(scope="session")
def golden(repo_root: Path) -> list:
    return unpack((repo_root / "tests" / "golden" / "replays.bin").read_bytes())


def test_the_corpus_covers_every_configuration(golden: list) -> None:
    seen = {(r.map_name, r.n_players) for r in golden}
    assert seen == set(CONFIGURATIONS)
    assert len(golden) >= 80


def test_every_golden_replay_still_reproduces(golden: list) -> None:
    for rec in golden:
        state = replay(rec)
        assert state.state_hash() == rec.final_hash
        assert tuple(final_scores(state)) == rec.final_scores


def test_the_corpus_is_not_stale(repo_root: Path) -> None:
    """Regenerating must produce byte-identical games."""
    result = subprocess.run(
        [sys.executable, "tools/gen_replays.py", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_corpus_reaches_the_interesting_states(golden: list) -> None:
    """A corpus that never triggers the endgame or a long route guards very little."""
    longest = 0
    total_actions = 0
    for rec in golden:
        state = replay(rec)
        total_actions += len(rec.actions)
        board = state.game.board
        for segment in range(board.n_segments):
            if state.seg_owner[segment] < 254:
                longest = max(longest, board.seg_len[segment])
    assert longest >= 5, "no long route is ever claimed in the corpus"
    assert total_actions > 5_000


# ---------------------------------------------------------------------------
# Wider sweeps
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("map_name", "n_players"), CONFIGURATIONS, ids=[f"{m}-{n}p" for m, n in CONFIGURATIONS]
)
def test_a_thousand_seeds_replay_bitwise(map_name: str, n_players: int) -> None:
    game = Game(RuleConfig(map_name=map_name, n_players=n_players))
    for seed in range(1000):
        state = game.new_initial_state(seed)
        rng = stream(seed, "policy")
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        rec = record(state)
        assert replay(rec).state_hash() == rec.final_hash


@pytest.mark.slow
@pytest.mark.parametrize(
    ("map_name", "n_players"), CONFIGURATIONS, ids=[f"{m}-{n}p" for m, n in CONFIGURATIONS]
)
def test_ten_thousand_random_games_hold_every_invariant(map_name: str, n_players: int) -> None:
    game = Game(RuleConfig(map_name=map_name, n_players=n_players))
    for seed in range(10_000):
        state = game.new_initial_state(seed)
        rng = stream(seed, "policy")
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        state.validate()
        assert sum(final_scores(state)) is not None
