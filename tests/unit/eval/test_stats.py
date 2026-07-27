"""The statistics layer, checked against known answers and by simulation.

Two kinds of test here, and the second is the one that matters. Recovering planted ratings
proves the optimiser works. **Simulating the sequential test against known effect sizes and
counting how often it is wrong** proves the error rates are what the parameters claim --
which is the only property anyone actually relies on when using SPRT to decide whether a
change helped.
"""

from __future__ import annotations

import numpy as np
import pytest

from ticket_to_ride.eval.stats import (
    ELO_SCALE,
    block_pair_matrix,
    bootstrap_elo,
    bradley_terry,
    cycle_fraction,
    elo_to_score,
    pair_counts,
    seat_advantage,
    sprt,
    win_rate_matrix,
)


def synthetic_wins(true_elo: np.ndarray, games: int, seed: int = 0) -> np.ndarray:
    """Sample a win table from a Bradley-Terry model with the given ratings."""
    rng = np.random.default_rng(seed)
    n = len(true_elo)
    wins = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = elo_to_score(true_elo[i] - true_elo[j])
            w = rng.binomial(games, p)
            wins[i, j], wins[j, i] = w, games - w
    return wins


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------


def test_the_fit_recovers_planted_ratings() -> None:
    true = np.array([-400.0, -120.0, 0.0, 130.0])
    fitted = bradley_terry(synthetic_wins(true, games=20_000), anchor=2)
    assert np.allclose(fitted, true, atol=25), f"{fitted} vs {true}"


def test_the_anchor_is_exactly_zero_whichever_agent_it_is() -> None:
    """H3 is the permanent zero of the scale (PLAN.md §11), so this is not approximate."""
    wins = synthetic_wins(np.array([-300.0, 0.0, 200.0]), games=500)
    for anchor in range(3):
        fitted = bradley_terry(wins, anchor=anchor)
        assert fitted[anchor] == 0.0
    # Only differences are identified, so re-anchoring must be a pure shift.
    a = bradley_terry(wins, anchor=0)
    b = bradley_terry(wins, anchor=2)
    assert np.allclose(a - a[2], b, atol=1e-9)


def test_the_prior_keeps_an_undefeated_agent_finite() -> None:
    """Without it the MLE is genuinely +inf, and a 6-game newcomer lands at the edge of
    the float range instead of near the anchor."""
    wins = np.array([[0.0, 40.0], [0.0, 0.0]])
    fitted = bradley_terry(wins, anchor=1, prior_sd=400.0)
    assert np.isfinite(fitted).all()
    assert 200 < fitted[0] < 2000, fitted


def test_a_shorter_prior_pulls_a_thin_record_further_toward_the_anchor() -> None:
    wins = np.array([[0.0, 6.0], [0.0, 0.0]])
    loose = bradley_terry(wins, anchor=1, prior_sd=800.0)
    tight = bradley_terry(wins, anchor=1, prior_sd=100.0)
    assert tight[0] < loose[0]


def test_ties_count_half_a_win_to_each_side() -> None:
    """TTR produces real draws -- two seats can match on points, tickets and longest path.
    Dropping them would bias ratings toward whoever draws less often."""
    all_draws = np.array([[0.0, 50.0], [50.0, 0.0]])
    fitted = bradley_terry(all_draws, anchor=0)
    assert abs(fitted[1]) < 1e-6


def test_a_pair_that_never_met_does_not_break_the_fit() -> None:
    # A and B both beat C; A and B never played. The fit must stay finite and rank both
    # above C, inferring the missing comparison transitively rather than failing.
    wins = np.zeros((3, 3))
    wins[0, 2], wins[2, 0] = 80.0, 20.0
    wins[1, 2], wins[2, 1] = 80.0, 20.0
    fitted = bradley_terry(wins, anchor=2)
    assert np.isfinite(fitted).all()
    assert fitted[0] > 0 and fitted[1] > 0


# ---------------------------------------------------------------------------
# Block decomposition and the bootstrap
# ---------------------------------------------------------------------------


def test_a_game_decomposes_into_its_pairwise_comparisons() -> None:
    # One 3-player game, agents 0/1/2 finishing 1st/2nd/3rd.
    agents = np.array([[0, 1, 2]])
    ranks = np.array([[1, 2, 3]])
    blocks = block_pair_matrix(np.array([0]), agents, ranks, n_agents=3, n_blocks=1)
    wins = pair_counts(blocks, 3)
    assert wins[0, 1] == 1 and wins[1, 0] == 0
    assert wins[0, 2] == 1 and wins[2, 0] == 0
    assert wins[1, 2] == 1 and wins[2, 1] == 0
    # C(3, 2) comparisons, no more.
    assert wins.sum() == 3


def test_self_pairings_are_excluded() -> None:
    """Self-play says nothing about strength; counting it would only pad a diagonal cell
    nothing reads."""
    agents = np.array([[0, 0]])
    ranks = np.array([[1, 2]])
    wins = pair_counts(block_pair_matrix(np.array([0]), agents, ranks, 1, 1), 1)
    assert wins.sum() == 0


def test_a_tie_splits_the_comparison() -> None:
    agents = np.array([[0, 1]])
    ranks = np.array([[1, 1]])
    wins = pair_counts(block_pair_matrix(np.array([0]), agents, ranks, 2, 1), 2)
    assert wins[0, 1] == 0.5 and wins[1, 0] == 0.5


def test_the_bootstrap_resamples_blocks_not_games() -> None:
    """The rule from PLAN.md §11, as a measurement.

    Games inside a block share a deck and are correlated. Resampling games independently
    ignores that and produces intervals that are too narrow -- by roughly sqrt(P), which is
    exactly how you come to believe in improvements that are not there. Here every game in
    a block is made perfectly correlated, which is the extreme of the real situation, so a
    game-level bootstrap should look dramatically more confident than a block-level one.
    """
    rng = np.random.default_rng(7)
    n_blocks, per_block = 200, 4
    agents, ranks, block_of_game = [], [], []
    for b in range(n_blocks):
        winner = rng.integers(0, 2)
        for _ in range(per_block):
            agents.append([0, 1])
            ranks.append([1, 2] if winner == 0 else [2, 1])
            block_of_game.append(b)
    agents = np.array(agents)
    ranks = np.array(ranks)
    block_of_game = np.array(block_of_game)

    by_block = block_pair_matrix(block_of_game, agents, ranks, 2, n_blocks)
    lo_b, hi_b = bootstrap_elo(by_block, 2, 1, resamples=300, seed=1)

    # The same games, each pretending to be its own block.
    as_games = block_pair_matrix(np.arange(len(agents)), agents, ranks, 2, len(agents))
    lo_g, hi_g = bootstrap_elo(as_games, 2, 1, resamples=300, seed=1)

    width_block = hi_b[0] - lo_b[0]
    width_game = hi_g[0] - lo_g[0]
    assert width_block > 1.5 * width_game, (
        f"block CI {width_block:.1f} is not meaningfully wider than the game CI "
        f"{width_game:.1f}; the bootstrap is resampling the wrong unit"
    )


def test_the_interval_covers_the_planted_rating() -> None:
    rng = np.random.default_rng(3)
    n_blocks = 400
    agents, ranks, blocks = [], [], []
    p = elo_to_score(150.0)
    for b in range(n_blocks):
        for _ in range(2):
            first = rng.random() < p
            agents.append([0, 1])
            ranks.append([1, 2] if first else [2, 1])
            blocks.append(b)
    matrix = block_pair_matrix(np.array(blocks), np.array(agents), np.array(ranks), 2, n_blocks)
    elo = bradley_terry(pair_counts(matrix, 2), anchor=1)
    lo, hi = bootstrap_elo(matrix, 2, 1, resamples=400, seed=2)
    assert lo[0] < 150.0 < hi[0], f"150 outside [{lo[0]:.0f}, {hi[0]:.0f}] (point {elo[0]:.0f})"


# ---------------------------------------------------------------------------
# The matrix, cycles and seats
# ---------------------------------------------------------------------------


def test_a_clean_ladder_has_no_cycles_and_a_rock_paper_scissors_pool_is_all_cycle() -> None:
    ladder = synthetic_wins(np.array([-300.0, 0.0, 300.0]), games=2000)
    assert cycle_fraction(ladder) == 0.0

    rps = np.zeros((3, 3))
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        rps[a, b], rps[b, a] = 70.0, 30.0
    assert cycle_fraction(rps) == 1.0


def test_an_unplayed_pair_reads_as_nan_not_as_a_draw() -> None:
    wins = np.zeros((2, 2))
    rate = win_rate_matrix(wins)
    assert np.isnan(rate[0, 1]) and np.isnan(rate[1, 0])


def _seat_sample(
    rng: np.random.Generator, n_blocks: int, edge: float
) -> tuple[np.ndarray, np.ndarray]:
    blocks, scores = [], []
    for b in range(n_blocks):
        for _ in range(2):
            blocks.append(b)
            first_wins = rng.random() < edge
            scores.append([1.0, 0.0] if first_wins else [0.0, 1.0])
    return np.array(blocks), np.array(scores)


def test_seat_advantage_detects_a_planted_first_player_edge() -> None:
    """First-player advantage is a property of the game, not a harness bug, so this
    measures it with an interval rather than asserting it away."""
    blocks, scores = _seat_sample(np.random.default_rng(5), 2000, edge=0.55)
    out = seat_advantage(blocks, scores, resamples=400, seed=1)
    (mean0, lo0, hi0), (mean1, _, _) = out
    assert abs(mean0 + mean1 - 1.0) < 1e-9, "seat scores must partition the games"
    assert lo0 <= mean0 <= hi0
    assert lo0 > 0.5, f"a real 55% seat edge was not resolvable: [{lo0:.3f}, {hi0:.3f}]"


def test_the_seat_interval_covers_the_truth_at_about_its_nominal_rate() -> None:
    """A 95% interval covers the truth 95% of the time, so a single draw missing it is not
    a defect -- it is the 5%. Asserting coverage on one sample makes a test that fails one
    run in twenty for the right reason, which is worse than no test. This measures the rate
    over replicates, which is the property the interval actually claims.
    """
    rng = np.random.default_rng(17)
    replicates, covered = 120, 0
    for r in range(replicates):
        blocks, scores = _seat_sample(rng, 200, edge=0.55)
        (_, lo, hi), _ = seat_advantage(blocks, scores, resamples=200, seed=r)
        covered += lo < 0.55 < hi
    rate = covered / replicates
    assert 0.86 < rate < 1.0, f"nominal 95% interval covered {rate:.2f} of the time"


# ---------------------------------------------------------------------------
# SPRT, validated by simulation rather than by inspection
# ---------------------------------------------------------------------------


def run_sequential(scores: np.ndarray, stride: int = 5) -> str:
    """Feed blocks in order, stopping at the first decision."""
    for n in range(10, scores.size + 1, stride):
        result = sprt(scores[:n])
        if result.decided:
            return result.verdict
    return "continue"


@pytest.mark.parametrize(
    ("delta_elo", "wrong_verdict", "budget"),
    [
        # Nominal alpha = beta = 0.05. The measured rates at 3000 trials were 0.045 under
        # H0 and 0.072 under H1 -- beta runs a little above nominal, which is expected:
        # Wald's bounds ignore overshoot and the variance is estimated rather than known.
        # Recorded here rather than rounded away, and the budgets are set from it.
        (0.0, "h1", 0.09),
        (20.0, "h0", 0.12),
    ],
)
def test_the_error_rates_are_close_to_nominal(
    delta_elo: float, wrong_verdict: str, budget: float
) -> None:
    rng = np.random.default_rng(11)
    p = elo_to_score(delta_elo)
    trials, wrong = 400, 0
    for _ in range(trials):
        scores = np.clip(rng.normal(p, 0.30, size=4000), 0.0, 1.0)
        if run_sequential(scores) == wrong_verdict:
            wrong += 1
    rate = wrong / trials
    assert rate < budget, f"delta={delta_elo}: wrong {rate:.3f} of the time, budget {budget}"


def test_a_large_effect_stops_early() -> None:
    """The reason for using SPRT at all: it stops early on the many changes that do not
    help, and quickly on the few that obviously do."""
    rng = np.random.default_rng(13)
    scores = np.clip(rng.normal(elo_to_score(200.0), 0.30, size=4000), 0.0, 1.0)
    stopped = next(n for n in range(10, 4001, 5) if sprt(scores[:n]).decided)
    assert sprt(scores[:stopped]).verdict == "h1"
    assert stopped < 200, f"took {stopped} blocks to resolve a 200-Elo gap"


def test_it_never_decides_on_a_single_block() -> None:
    """Stopping needs a variance estimate, and one block has none. The wider rule is that
    the test stops on block boundaries at all: half a block is one seating, and reporting
    one seating is the mirrored fiction the whole design exists to avoid."""
    assert sprt(np.array([1.0])).verdict == "continue"
    assert sprt(np.array([])).verdict == "continue"


def test_identical_block_scores_do_not_manufacture_certainty() -> None:
    """A clean sweep of every block gives zero sample variance. Undefended, the LLR divides
    by zero and reports an infinitely significant result on a handful of games."""
    result = sprt(np.ones(6))
    assert np.isfinite(result.llr)


def test_elo_and_score_agree_at_the_scale_constant() -> None:
    assert elo_to_score(0.0) == pytest.approx(0.5)
    assert elo_to_score(ELO_SCALE) == pytest.approx(10.0 / 11.0)
