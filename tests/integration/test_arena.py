"""Phase 3's exit criteria for the evaluation harness (PLAN.md §14).

    10k paired games across 5 agents in seconds; leaderboard stable across two runs of the
    same seed root; seat win rate flat within CI; ratings accumulate across invocations;
    H3 > H2 > H1 > random by predicted margins; every heuristic exercises every action type
    on both maps.

Coverage and the ladder ordering are asserted on the Rust side, where they are cheap
(`crates/ttr-core/tests/heuristics.rs`). What lives here is everything that needs the store:
stability, accumulation, and the seat measurement.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ticket_to_ride.eval.arena import parse_spec, round_robin, run_round_robin
from ticket_to_ride.eval.leaderboard import NoAnchorError, fit
from ticket_to_ride.eval.store import Store

pytestmark = pytest.mark.slow

LADDER = ("h0", "h1", "h2", "h3", "h4")


@pytest.fixture(autouse=True)
def _needs_rust(rust: object) -> None:
    """Every test here plays real games, so the extension is required, not optional."""


def specs(names: tuple[str, ...] = LADDER, seed: int = 0) -> list:
    return [parse_spec(name, seed) for name in names]


def play(path: Path, *, blocks: int, seed_root: int = 0, names: tuple[str, ...] = LADDER) -> None:
    with Store(path) as store:
        run_round_robin(store, "usa", 2, specs(names), seed_root=seed_root, blocks=blocks)


# ---------------------------------------------------------------------------
# Stability and accumulation
# ---------------------------------------------------------------------------


def test_the_leaderboard_is_stable_across_two_runs_of_the_same_seed_root(tmp_path: Path) -> None:
    """Same seed root, same games, same ratings -- bit for bit, not within noise.

    Also compares the stored `final_hash` of every game. Two runs whose ratings agree but
    whose games differ would pass a ratings-only check while being irreproducible, which is
    the failure that is hardest to notice later.
    """
    reports, hashes = [], []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.db"
        play(path, blocks=20)
        with Store(path) as store:
            reports.append(fit(store, 1, resamples=100, seed=1))
            hashes.append(
                [
                    r["final_hash"]
                    for r in store.db.execute("SELECT final_hash FROM game ORDER BY game_id")
                ]
            )
    first, second = reports
    assert hashes[0] == hashes[1], "the same seed root produced different games"
    assert [s.spec for s in first.standings] == [s.spec for s in second.standings]
    for a, b in zip(first.standings, second.standings, strict=True):
        assert a.elo == pytest.approx(b.elo, abs=1e-9), f"{a.spec}: {a.elo} vs {b.elo}"
        assert (a.lo, a.hi) == pytest.approx((b.lo, b.hi), abs=1e-9)


def test_ratings_accumulate_across_invocations(tmp_path: Path) -> None:
    """The property agent identity exists for. A second arena run against the same store
    must *add* to each agent's record rather than mint new agents holding a slice each."""
    path = tmp_path / "results.db"
    play(path, blocks=10, seed_root=0)
    with Store(path) as store:
        first = fit(store, 1, resamples=100)
        agents_after_one = store.counts()["agent"]

    # Disjoint blocks, so this is genuinely new evidence rather than the same games again.
    play(path, blocks=10, seed_root=1000)
    with Store(path) as store:
        second = fit(store, 1, resamples=100)
        assert store.counts()["agent"] == agents_after_one, "a re-run minted new agents"
        assert store.counts()["rating_run"] == 2, "the refit overwrote the earlier snapshot"

    by_spec = {s.spec: s for s in second.standings}
    for standing in first.standings:
        assert by_spec[standing.spec].games > standing.games, standing.spec
    assert second.n_blocks > first.n_blocks


def test_re_running_the_same_seed_root_adds_games_without_changing_the_evidence(
    tmp_path: Path,
) -> None:
    """Block seeds are `seed_root + i`, so replaying a root replays the same decks.

    That is deliberate -- extending a run from 500 to 1000 blocks reuses the first 500 --
    but it means the *same* games can be recorded twice. They must not then read as twice
    as much evidence: the block count stays put, so the confidence interval does too.
    """
    path = tmp_path / "results.db"
    play(path, blocks=15, seed_root=0)
    with Store(path) as store:
        before = fit(store, 1, resamples=200, seed=3)
    play(path, blocks=15, seed_root=0)
    with Store(path) as store:
        after = fit(store, 1, resamples=200, seed=3)

    assert after.n_games == 2 * before.n_games
    assert after.n_blocks == before.n_blocks, "duplicate decks were counted as new blocks"
    # Per agent, and skipping the anchor, whose interval is zero-width by construction --
    # taking a `min` over all of them would compare 0 against 0 and pass vacuously.
    widths_before = {s.spec: s.hi - s.lo for s in before.standings if s.spec != "h3"}
    widths_after = {s.spec: s.hi - s.lo for s in after.standings if s.spec != "h3"}
    for spec, before_width in widths_before.items():
        assert widths_after[spec] > 0.6 * before_width, (
            f"{spec}: recording the same decks twice narrowed the interval from "
            f"{before_width:.0f} to {widths_after[spec]:.0f}; the bootstrap is counting "
            "correlated games as independent evidence"
        )


def test_a_leaderboard_without_the_anchor_is_refused(tmp_path: Path) -> None:
    """Every rating is a difference from H3. Anchoring on whoever happened to be present
    would produce a plausible table on an unlabelled scale, comparable to nothing."""
    path = tmp_path / "results.db"
    play(path, blocks=5, names=("h0", "h1", "h2"))
    with Store(path) as store, pytest.raises(NoAnchorError):
        fit(store, 1, resamples=50)


# ---------------------------------------------------------------------------
# What the numbers say
# ---------------------------------------------------------------------------


def test_the_ladder_is_monotone_and_the_anchor_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "results.db"
    play(path, blocks=60)
    with Store(path) as store:
        report = fit(store, 1, resamples=300)

    elo = {s.spec: s.elo for s in report.standings}
    assert elo["h3"] == 0.0
    assert elo["h3"] > elo["h2"] > elo["h1"] > elo["h0"]
    # The intervals must separate the tiers that are genuinely different, or the ladder is
    # an ordering of noise. H4 is deliberately excluded: measured, it sits within noise of
    # H3 rather than at the +130 Elo PLAN.md §11 predicted -- see docs/WORKLOG.md.
    by_spec = {s.spec: s for s in report.standings}
    assert by_spec["h2"].hi < by_spec["h3"].lo + 1e-9
    assert by_spec["h1"].hi < by_spec["h2"].lo
    assert by_spec["h0"].hi < by_spec["h1"].lo


def test_the_win_rate_matrix_measures_both_seatings(tmp_path: Path) -> None:
    """The published prior art reported one seating and its complement, so first-player
    advantage was confounded into every cell. The symmetry here is honest because every
    rotation really was played -- which the arena's own test asserts -- so what this checks
    is that each pair's games were split evenly between the seatings.
    """
    path = tmp_path / "results.db"
    play(path, blocks=20)
    with Store(path) as store:
        report = fit(store, 1, resamples=50)
        rows = store.db.execute(
            "SELECT s.agent_id, s.seat, COUNT(*) AS n FROM seat s GROUP BY s.agent_id, s.seat"
        ).fetchall()

    per_agent: dict[int, dict[int, int]] = {}
    for row in rows:
        per_agent.setdefault(int(row["agent_id"]), {})[int(row["seat"])] = int(row["n"])
    for agent, seats in per_agent.items():
        assert seats[0] == seats[1], f"agent {agent} played {seats}, so seat bias survives"
    assert report.cycle_fraction == 0.0, "the scripted ladder should be transitive"


def test_the_seat_effect_is_reported_with_an_interval(tmp_path: Path) -> None:
    """PLAN.md §11 calls a flat seat win rate the canary that mirroring works. Half right:
    rotation removes seat bias from the *agent* ratings by construction, while first-player
    advantage stays a real property of the game. So this asserts the interval is finite and
    the seats partition the games -- and reports the effect rather than legislating it.
    """
    path = tmp_path / "results.db"
    play(path, blocks=40)
    with Store(path) as store:
        report = fit(store, 1, resamples=300)

    assert len(report.seat_score) == 2
    total = sum(mean for mean, _, _ in report.seat_score)
    assert total == pytest.approx(1.0, abs=1e-9), "seat win shares must sum to one"
    for mean, lo, hi in report.seat_score:
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def test_ten_thousand_paired_games_across_five_agents_take_seconds(tmp_path: Path) -> None:
    """The §14 exit criterion, measured rather than quoted.

    Ten lineups over five agents at two seats; 500 blocks each is 1000 games per lineup and
    10,000 in total. The budget is generous because this runs alongside the rest of the
    suite under `-n auto`, where core contention is real -- the same effect that makes
    `make bench-check` report a 40-80% regression on a busy machine. The honest figure is
    in docs/WORKLOG.md, measured on an idle box.
    """
    lineups = round_robin(len(LADDER), 2)
    blocks = 10_000 // (len(lineups) * 2)
    path = tmp_path / "results.db"

    start = time.perf_counter()
    with Store(path) as store:
        summaries = run_round_robin(store, "usa", 2, specs(), seed_root=0, blocks=blocks)
        played = sum(s.games for s in summaries)
        engine = sum(s.seconds for s in summaries)
    elapsed = time.perf_counter() - start

    assert played == 10_000, played
    assert elapsed < 120.0, f"{played} games took {elapsed:.1f}s wall ({engine:.1f}s engine)"
