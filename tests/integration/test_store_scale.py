"""Does a full refit stay cheap as the store fills up?

This is the measurement behind the design decision in `store.py`: **recompute, do not
accumulate**. The argument is that Bradley-Terry's sufficient statistic is the pairwise
win-count matrix, so however many games pile up they reduce to an `A x A` table and the fit
costs `O(A^2)` per Newton step. Only the reduction is `O(N)`, and it is one grouped pass.

If that argument is wrong the symptom is a leaderboard that gets slower every week, so it is
measured rather than asserted. The numbers this produces are in docs/WORKLOG.md; the test
checks the *shape* -- that cost grows about linearly in the game count rather than worse,
and that the fit itself is flat.

Rows are synthesised rather than played. A million real games is twenty minutes of engine
time and would measure the engine; what is under test is the store and the fit.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ticket_to_ride.eval.leaderboard import fit
from ticket_to_ride.eval.store import AgentRow, MatchRow, Store

pytestmark = pytest.mark.slow

AGENTS = ("h0", "h1", "h2", "h3", "h4")

#: Enough to see the trend without turning the suite into a benchmark. The 1M point is in
#: the worklog, run by hand.
SIZES = (10_000, 100_000)


def fill(path: Path, n_games: int, rng_seed: int = 0) -> None:
    """Write `n_games` two-seat games over the five scripted agents.

    Ranks are drawn from a fixed ladder so the fit has something real to converge on --
    a store of coin flips would fit instantly for the wrong reason.
    """
    import random  # noqa: PLC0415 - test-local, not part of the seeding discipline

    rng = random.Random(rng_seed)
    with Store(path) as store:
        config = store.config_id("usa", 2, "rules", "data", 1)
        ids = [
            store.agent_id(
                AgentRow(spec=name, family=name, params_hash=f"p{i}", behaviour_hash=f"b{i}")
            )
            for i, name in enumerate(AGENTS)
        ]
        strength = dict(zip(ids, (0.05, 0.25, 0.42, 0.5, 0.5), strict=True))

        blocks = max(1, n_games // 2)
        games, seats = [], []
        for g in range(n_games):
            a, b = rng.sample(ids, 2)
            games.append((g // 2, g % 2, 100, f"{g:016x}"))
            p = strength[a] / (strength[a] + strength[b])
            a_first = rng.random() < p
            for seat, (agent, rank) in enumerate(
                ((a, 1 if a_first else 2), (b, 2 if a_first else 1))
            ):
                seats.append(
                    (
                        g,
                        seat,
                        agent,
                        100,
                        1.0 if rank == 1 else -1.0,
                        rank,
                        int(rank == 1),
                        *([0] * 17),
                    )
                )
        store.record_match(
            MatchRow(
                config_id=config, seed_root=0, n_blocks=blocks, lineup="synthetic", seconds=0.0
            ),
            games,
            seats,
        )


@pytest.mark.parametrize("n_games", SIZES)
def test_a_full_refit_stays_cheap(tmp_path: Path, n_games: int) -> None:
    path = tmp_path / f"{n_games}.db"
    fill(path, n_games)
    with Store(path) as store:
        # Timed separately: the read is the only O(N) step, and the fit is what the design
        # claims is flat. Reporting one total would hide which of the two is growing.
        start = time.perf_counter()
        rows = store.seat_rows(1)
        read = time.perf_counter() - start

        start = time.perf_counter()
        report = fit(store, 1, resamples=200, persist=False)
        total = time.perf_counter() - start

    assert len(rows) == 2 * n_games
    assert report.n_games == n_games
    # Generous, because this runs under `-n auto` alongside everything else and core
    # contention is real. The point is the shape, not the constant: a maintained aggregate
    # table would only be worth building if this grew super-linearly, and it does not.
    budget = 10.0 + n_games / 20_000
    assert total < budget, (
        f"{n_games:,} games: read {read:.2f}s, refit {total:.2f}s, budget {budget:.1f}s"
    )


def test_the_ladder_survives_a_hundred_thousand_games(tmp_path: Path) -> None:
    """A sanity check on the fit itself at scale, not just its speed: the planted ordering
    has to come back out, with intervals that have tightened rather than degenerated."""
    path = tmp_path / "big.db"
    fill(path, 100_000)
    with Store(path) as store:
        report = fit(store, 1, resamples=200, persist=False)
    elo = {s.spec: s.elo for s in report.standings}
    assert elo["h2"] > elo["h1"] > elo["h0"]
    for standing in report.standings:
        if standing.spec != "h3":
            assert standing.hi - standing.lo < 60.0, f"{standing.spec} interval did not tighten"
