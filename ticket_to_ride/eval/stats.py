"""Ratings, confidence intervals and the sequential test. Pure numpy, no I/O.

PLAN.md §11, and the one rule in it that is easy to get wrong and fatal when you do:

    Games inside a seed block share a deck, so they are **not** independent. Aggregate to a
    per-block statistic first, then treat *blocks* as the i.i.d. units. Bootstrap CIs
    resample blocks, never games.

Treating paired games as independent inflates significance by about `sqrt(P)` and will make
you believe in improvements that are not there. Every function here takes a block index and
uses it; none of them accepts a flat list of games.

## Why a full refit is the right answer, at any scale

Bradley-Terry is an exponential family whose sufficient statistic is the **pairwise win
count matrix**. However many games accumulate, they reduce to an `A x A` table -- so a
"full recompute" costs `O(A^2)` per Newton step and is independent of the game count. The
only `O(N)` step is building the table, which is one grouped sum.

That is why there is no incremental Elo here. Incremental updates are order-dependent, drift
against the batch fit, and would buy nothing: the expensive object was already small.

The bootstrap *is* `O(blocks x B)`, which is why blocks are pre-reduced to a
`blocks x pairs` matrix once and each resample is a single matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Elo's scale constant: a 400-point gap is 10:1 odds.
ELO_SCALE = 400.0

#: Natural-log units per Elo point.
LOG10_OVER_SCALE = np.log(10.0) / ELO_SCALE

#: An even matchup, in score share. Also what an unplayed pair is treated as when looking
#: for cycles: no evidence either way must not read as a beat.
EVEN = 0.5

#: A cycle needs three agents.
MIN_CYCLE_AGENTS = 3

#: SPRT needs a variance estimate, so it needs at least two blocks.
MIN_SPRT_BLOCKS = 2


def elo_to_score(delta: float) -> float:
    """Expected score for a rating advantage of `delta` Elo."""
    return float(1.0 / (1.0 + 10.0 ** (-delta / ELO_SCALE)))


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------


def block_pair_matrix(
    block_of_game: np.ndarray,
    agent_by_seat: np.ndarray,
    rank_by_seat: np.ndarray,
    n_agents: int,
    n_blocks: int,
) -> np.ndarray:
    """Per-block pairwise scores, shape `(n_blocks, n_agents ** 2)`.

    Each `P`-player game is decomposed into its `C(P, 2)` pairwise comparisons by final
    rank, per PLAN.md §11. A tie contributes 0.5 to each side: Bradley-Terry has no tie
    term, and half a win is what Elo and BayesElo do. Stating it here rather than leaving it
    implicit, because TTR produces real draws -- two seats can match on points, tickets
    *and* longest path -- and silently dropping them would bias every rating toward whoever
    draws less often.

    Self-pairings are skipped. An agent playing itself carries no information about its
    strength and would only add `n/2` to both sides of a diagonal cell nothing reads.

    `agent_by_seat` and `rank_by_seat` are `(n_games, P)`: the arena returns one row per
    (game, seat) and every game in a match has the same seat count, so the rows reshape.
    """
    n_games, seats = agent_by_seat.shape
    if rank_by_seat.shape != agent_by_seat.shape:
        raise ValueError("agent and rank tables must have the same shape")
    if block_of_game.shape[0] != n_games:
        raise ValueError("one block index per game")

    cells = n_agents * n_agents
    flat = np.zeros(n_blocks * cells, dtype=np.float64)
    for x in range(seats):
        for y in range(x + 1, seats):
            ax, ay = agent_by_seat[:, x], agent_by_seat[:, y]
            rx, ry = rank_by_seat[:, x], rank_by_seat[:, y]
            keep = ax != ay
            if not keep.any():
                continue
            score_x = np.where(rx < ry, 1.0, np.where(rx > ry, 0.0, 0.5))[keep]
            base = block_of_game[keep] * cells
            # bincount rather than np.add.at: same scatter-add, roughly two orders of
            # magnitude faster, and this runs over every game in the store.
            flat += np.bincount(
                base + ax[keep] * n_agents + ay[keep],
                weights=score_x,
                minlength=n_blocks * cells,
            )
            flat += np.bincount(
                base + ay[keep] * n_agents + ax[keep],
                weights=1.0 - score_x,
                minlength=n_blocks * cells,
            )
    return flat.reshape(n_blocks, cells)


def pair_counts(blocks: np.ndarray, n_agents: int) -> np.ndarray:
    """Collapse a `blocks x pairs` matrix to the `A x A` sufficient statistic."""
    return blocks.sum(axis=0).reshape(n_agents, n_agents)


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rating:
    """One agent's fitted rating."""

    agent: str
    elo: float
    lo: float
    hi: float
    games: int
    score: float


def bradley_terry(
    wins: np.ndarray,
    anchor: int,
    prior_sd: float = 400.0,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> np.ndarray:
    """Bradley-Terry MLE by Newton's method. Returns Elo, with `anchor` at exactly 0.

    `wins[i, j]` is `i`'s score against `j` (wins plus half the draws).

    Two things keep this well-posed. A weak `N(0, prior_sd^2)` prior on Elo stops a
    six-game newcomer -- or an agent that has never lost -- landing at infinity, which the
    unpenalised MLE genuinely does. And the fit is then **shifted** so the anchor reads 0,
    rather than the anchor being constrained during optimisation: shifting is exact because
    the likelihood depends only on rating *differences*, and it keeps the prior doing the
    one job it is here for.

    `anchor` is H3 permanently (PLAN.md §11) so the scale stays comparable for the
    project's lifetime -- see `ttr_rust.behaviour_hash` for what actually holds it still.
    """
    n = wins.shape[0]
    if wins.shape != (n, n):
        raise ValueError("the win table must be square")
    if not 0 <= anchor < n:
        raise ValueError(f"anchor {anchor} is not one of the {n} agents")

    played = wins + wins.T
    np.fill_diagonal(played, 0.0)
    observed = wins.sum(axis=1)
    prior_precision = 1.0 / (prior_sd * LOG10_OVER_SCALE) ** 2

    theta = np.zeros(n, dtype=np.float64)
    for _ in range(max_iterations):
        diff = theta[:, None] - theta[None, :]
        # Expected score of i against j under the current ratings.
        p = 1.0 / (1.0 + np.exp(-diff))
        expected = (played * p).sum(axis=1)
        gradient = observed - expected - prior_precision * theta

        variance = played * p * (1.0 - p)
        hessian = np.diag(variance.sum(axis=1) + prior_precision) - variance
        step = np.linalg.solve(hessian, gradient)
        theta += step
        if np.max(np.abs(step)) < tolerance:
            break

    elo = theta / LOG10_OVER_SCALE
    return elo - elo[anchor]


def bootstrap_elo(
    blocks: np.ndarray,
    n_agents: int,
    anchor: int,
    *,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    prior_sd: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CIs for every agent's Elo, resampling **blocks**.

    Blocks, not games: games inside a block share a deck and are correlated, and resampling
    them independently would produce intervals that are far too narrow.

    Each resample is one matrix product against the pre-reduced `blocks x pairs` matrix,
    then a refit over the `A x A` sufficient statistic -- so cost scales with the number of
    resamples and the number of *agents*, not with the number of games.
    """
    n_blocks = blocks.shape[0]
    if n_blocks == 0 or resamples <= 0:
        # `resamples=0` is a legitimate ask on a large store -- point estimates only, no
        # intervals. NaN rather than zero, so a missing interval reads as missing instead
        # of as a suspiciously confident one.
        blank = np.full(n_agents, np.nan)
        return blank, blank
    rng = np.random.default_rng(seed)

    draws = np.empty((resamples, n_agents), dtype=np.float64)
    chunk = max(1, min(resamples, 1 + 4_000_000 // max(n_blocks, 1)))
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        # A multinomial over blocks is the same distribution as sampling n_blocks blocks
        # with replacement and counting them, and it turns the resample into a matmul.
        weights = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=size)
        totals = weights.astype(np.float64) @ blocks
        for row in range(size):
            table = totals[row].reshape(n_agents, n_agents)
            draws[done + row] = bradley_terry(table, anchor, prior_sd=prior_sd)
        done += size

    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# The win-rate matrix and what it shows that Elo hides
# ---------------------------------------------------------------------------


def win_rate_matrix(wins: np.ndarray) -> np.ndarray:
    """`rate[i, j]` = i's score share against j; NaN where they never met.

    Kept alongside the ratings because **Elo compresses a matrix to a vector** and hides
    rock-paper-scissors, which is exactly what a league needs to show.

    A note for whoever reads the output: if every off-diagonal pair sums to *exactly* 1.00
    that is expected here, because both seatings really were played. It is a red flag only
    when the matrix was produced by mirroring one seating, which is the flaw in the
    published prior art -- the arena's own test asserts each agent occupies each seat
    equally often, which is the property that makes the symmetry honest rather than assumed.
    """
    played = wins + wins.T
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(played > 0, wins / played, np.nan)
    np.fill_diagonal(rate, np.nan)
    return rate


def cycle_fraction(wins: np.ndarray) -> float:
    """Fraction of agent triples that form a beat-cycle `A > B > C > A`.

    The one number that says whether a league is progressing or spinning (PLAN.md §11).
    Zero for a clean ladder; rising as the pool learns to counter itself.
    """
    n = wins.shape[0]
    if n < MIN_CYCLE_AGENTS:
        return 0.0
    rate = win_rate_matrix(wins)
    beats = np.nan_to_num(rate, nan=EVEN) > EVEN
    cycles = total = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                total += 1
                if (beats[a, b] and beats[b, c] and beats[c, a]) or (
                    beats[a, c] and beats[c, b] and beats[b, a]
                ):
                    cycles += 1
    return cycles / total if total else 0.0


def seat_advantage(
    block_of_game: np.ndarray,
    seat_scores: np.ndarray,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[tuple[float, float, float]]:
    """Mean score per seat with a block-bootstrap CI: `[(mean, lo, hi), ...]`.

    **What this measures, and what it does not.** PLAN.md §11 calls a flat seat win rate the
    canary that mirroring works. That is half right and the half matters: with cyclic
    rotation each *agent* occupies each seat equally often, so agent ratings carry no seat
    bias by construction -- but the raw per-seat score can still be non-flat, because
    first-player advantage is a real property of Ticket to Ride and not a harness bug.

    So this reports the seat effect with an interval rather than asserting it away. A
    non-flat result is a finding about the game; the harness property is the one the arena
    tests directly, that every agent visits every seat the same number of times.
    """
    n_seats = seat_scores.shape[1]
    n_blocks = int(block_of_game.max()) + 1 if block_of_game.size else 0
    if n_blocks == 0:
        return [(float("nan"),) * 3] * n_seats

    counts = np.bincount(block_of_game, minlength=n_blocks).astype(np.float64)
    per_block = np.stack(
        [
            np.bincount(block_of_game, weights=seat_scores[:, s], minlength=n_blocks)
            for s in range(n_seats)
        ],
        axis=1,
    )
    safe = np.where(counts > 0, counts, 1.0)[:, None]
    per_block = per_block / safe

    mean = per_block.mean(axis=0)
    if resamples <= 0:
        # Point estimates only, as in `bootstrap_elo`. NaN bounds, so a missing interval
        # reads as missing rather than as a suspiciously narrow one.
        return [(float(mean[s]), float("nan"), float("nan")) for s in range(n_seats)]

    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, n_seats))
    for r in range(resamples):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        draws[r] = per_block[idx].mean(axis=0)
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    return [(float(mean[s]), float(lo[s]), float(hi[s])) for s in range(n_seats)]


# ---------------------------------------------------------------------------
# SPRT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SprtResult:
    """Where a sequential test stands."""

    llr: float
    lower: float
    upper: float
    #: `"h1"` (the candidate is better), `"h0"` (it is not), or `"continue"`.
    verdict: str
    blocks: int

    @property
    def decided(self) -> bool:
        return self.verdict != "continue"


def sprt(
    block_scores: np.ndarray,
    elo0: float = 0.0,
    elo1: float = 20.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> SprtResult:
    """Sequential probability ratio test on a stream of **per-block** scores.

    `block_scores[b]` is the candidate's mean score over block `b`, in `[0, 1]`.

    **Blocks, not games.** Stopping mid-block would break the pairing the block exists to
    provide: half a block is one seating, and one seating is exactly the mirrored fiction
    §11 forbids. The unit that goes into the likelihood ratio must be the unit that is
    i.i.d., which is the block.

    The nuisance-variance form (as used by chess engine testing): with block scores treated
    as approximately normal with unknown variance, the log-likelihood ratio between two
    hypothesised means reduces to a difference of squared deviations scaled by the sample
    variance. That avoids assuming a draw model, which matters here because the block score
    is an average over rotations and is not trinomial at all.

    Bounds are Wald's: accept H1 above `log((1 - beta) / alpha)`, H0 below
    `log(beta / (1 - alpha))`.
    """
    scores = np.asarray(block_scores, dtype=np.float64)
    lower = float(np.log(beta / (1.0 - alpha)))
    upper = float(np.log((1.0 - beta) / alpha))
    n = scores.size
    if n < MIN_SPRT_BLOCKS:
        return SprtResult(0.0, lower, upper, "continue", n)

    p0, p1 = elo_to_score(elo0), elo_to_score(elo1)
    # Population variance about the sample mean, floored so a run of identical block scores
    # -- every block a clean sweep, which happens against a much weaker opponent -- cannot
    # divide by zero and declare an infinitely significant result.
    variance = float(scores.var())
    variance = max(variance, 1e-9)
    llr = float(((scores - p0) ** 2 - (scores - p1) ** 2).sum() / (2.0 * variance))

    verdict = "continue"
    if llr >= upper:
        verdict = "h1"
    elif llr <= lower:
        verdict = "h0"
    return SprtResult(llr, lower, upper, verdict, n)
