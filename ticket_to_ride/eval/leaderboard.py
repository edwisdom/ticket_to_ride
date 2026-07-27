"""Fitting ratings from everything the store holds, and reporting what Elo hides.

A leaderboard is always a **full refit** over every game recorded for a configuration --
never an incremental update. See `store.py` for why that is cheap regardless of scale:
Bradley-Terry reduces the whole history to an `A x A` table, and the fit is `O(A^2)` per
Newton step from there.

Three things are reported alongside the ratings, because a rating vector on its own is not
enough to tell whether a pool is healthy:

* the **win-rate matrix**, both seatings measured -- Elo compresses a matrix to a vector and
  hides rock-paper-scissors;
* **`cycle_fraction`**, the single number that says whether a league is progressing or
  spinning;
* the **seat effect** with a block-bootstrap interval. Not asserted flat: cyclic rotation
  removes seat bias from *agent* ratings by construction, while first-player advantage
  remains a real property of the game and is worth measuring rather than assuming away.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ticket_to_ride.eval.stats import (
    block_pair_matrix,
    bootstrap_elo,
    bradley_terry,
    cycle_fraction,
    pair_counts,
    seat_advantage,
    win_rate_matrix,
)
from ticket_to_ride.eval.store import Store


@dataclass(frozen=True)
class Standing:
    """One row of the leaderboard."""

    agent_id: int
    spec: str
    behaviour_hash: str
    elo: float
    lo: float
    hi: float
    games: int
    score: float


@dataclass(frozen=True)
class Report:
    """A fitted leaderboard plus the diagnostics that keep it honest."""

    config_id: int
    map_name: str
    n_players: int
    standings: list[Standing]
    wins: np.ndarray
    win_rate: np.ndarray
    cycle_fraction: float
    seat_score: list[tuple[float, float, float]]
    n_games: int
    n_blocks: int
    anchor_agent_id: int
    rating_run_id: int | None = None

    @property
    def anchored(self) -> Standing | None:
        return next((s for s in self.standings if s.agent_id == self.anchor_agent_id), None)


class NoGamesError(RuntimeError):
    def __init__(self, config_id: int) -> None:
        super().__init__(f"config {config_id} has no recorded games")


class NoAnchorError(RuntimeError):
    """The anchor agent has not played in this configuration.

    Fatal on purpose. Without it the fit would be anchored on an arbitrary agent and the
    numbers would look fine while sitting on a different, unlabelled scale.
    """

    def __init__(self, family: str) -> None:
        super().__init__(
            f"no games recorded for the rating anchor ({family!r}) in this configuration. "
            "Every rating is a difference from the anchor, so a leaderboard without it "
            "would be on an unlabelled scale that cannot be compared to any other run."
        )


def fit(
    store: Store,
    config_id: int,
    *,
    anchor_family: str = "h3",
    prior_sd: float = 400.0,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    persist: bool = True,
) -> Report:
    """Refit every rating for one configuration from the full game table."""
    rows = store.seat_rows(config_id)
    if not rows:
        raise NoGamesError(config_id)
    agents = store.agents_in(config_id)
    index_of = {int(a["agent_id"]): i for i, a in enumerate(agents)}
    n_agents = len(agents)

    seats_per_game = 1 + max(int(r["seat"]) for r in rows)
    n_games = len(rows) // seats_per_game
    agent_by_seat = np.empty((n_games, seats_per_game), dtype=np.int64)
    rank_by_seat = np.empty((n_games, seats_per_game), dtype=np.int64)
    won_by_seat = np.zeros((n_games, seats_per_game), dtype=np.float64)
    block_seed = np.empty(n_games, dtype=np.int64)
    # `seat_rows` orders by (game_id, seat) and every game in a configuration has the same
    # seat count, so the flat rows tile the table exactly.
    for i, row in enumerate(rows):
        g, s = divmod(i, seats_per_game)
        agent_by_seat[g, s] = index_of[int(row["agent_id"])]
        rank_by_seat[g, s] = int(row["rank"])
        won_by_seat[g, s] = float(row["won"])
        if s == 0:
            block_seed[g] = int(row["block_seed"])

    # Blocks are identified by their seed. Two matches that used the same seed root share
    # decks, so their games belong to the *same* block for resampling purposes -- treating
    # them as distinct would quietly restore the independence assumption blocks exist to
    # remove.
    unique_blocks, block_of_game = np.unique(block_seed, return_inverse=True)
    n_blocks = len(unique_blocks)

    matrix = block_pair_matrix(block_of_game, agent_by_seat, rank_by_seat, n_agents, n_blocks)
    wins = pair_counts(matrix, n_agents)

    # The anchor is the agent whose spec is *exactly* the family name. A tuned variant is
    # written `h3@configs/agents/h3_v2.toml` and shares the family but not the constants;
    # anchoring on one would silently re-base the whole scale to whichever tuning happened
    # to be registered first, and every number would still look reasonable.
    anchor = next((i for i in range(n_agents) if str(agents[i]["spec"]) == anchor_family), None)
    if anchor is None:
        raise NoAnchorError(anchor_family)

    elo = bradley_terry(wins, anchor, prior_sd=prior_sd)
    lo, hi = bootstrap_elo(
        matrix, n_agents, anchor, resamples=resamples, alpha=alpha, seed=seed, prior_sd=prior_sd
    )
    played = (wins + wins.T).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(played > 0, wins.sum(axis=1) / played, np.nan)

    order = np.argsort(-elo)
    standings = [
        Standing(
            agent_id=int(agents[i]["agent_id"]),
            spec=str(agents[i]["spec"]),
            behaviour_hash=str(agents[i]["behaviour_hash"]),
            elo=float(elo[i]),
            lo=float(lo[i]),
            hi=float(hi[i]),
            games=int(played[i]),
            score=float(share[i]),
        )
        for i in order
    ]

    config = next(c for c in store.configs() if int(c["config_id"]) == config_id)
    report = Report(
        config_id=config_id,
        map_name=str(config["map_name"]),
        n_players=int(config["n_players"]),
        standings=standings,
        wins=wins,
        win_rate=win_rate_matrix(wins),
        cycle_fraction=cycle_fraction(wins),
        seat_score=seat_advantage(block_of_game, won_by_seat, resamples=resamples, seed=seed),
        n_games=n_games,
        n_blocks=n_blocks,
        anchor_agent_id=int(agents[anchor]["agent_id"]),
    )
    if not persist:
        return report

    run_id = store.record_ratings(
        config_id,
        report.anchor_agent_id,
        [(s.agent_id, s.elo, s.lo, s.hi, s.games, s.score) for s in standings],
        method="bradley-terry-newton",
        prior_sd=prior_sd,
        resamples=resamples,
        n_games=n_games,
        n_blocks=n_blocks,
        cycle_fraction=report.cycle_fraction,
    )
    return replace(report, rating_run_id=run_id)
