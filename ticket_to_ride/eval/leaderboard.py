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

Rated human games join the same fit (`fit_with`, PLAN.md §14.2): every game of a
configuration, arena and human alike, on the H3 = 0 scale. A human game is its own seed
block -- nobody else was dealt its deck -- and a human is identified by player id.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

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
from ticket_to_ride.eval.store import AgentRow, Store

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


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


# ---------------------------------------------------------------------------
# The fit, whatever the games came from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Games:
    """Finished games as a fit reads them, one row per game.

    `competitors[g, s]` indexes whoever sat in seat `s` of game `g`; `blocks[g]` is the
    game's seed block, the unit the bootstrap resamples.
    """

    competitors: np.ndarray
    ranks: np.ndarray
    won: np.ndarray
    blocks: np.ndarray
    n_blocks: int

    @property
    def n_games(self) -> int:
        return int(self.blocks.shape[0])


def tabulate[B: SupportsRichComparison](
    rows: Sequence[tuple[B, Sequence[int], Sequence[int], Sequence[float]]],
) -> Games:
    """`(block key, competitor per seat, rank per seat, won per seat)` rows into a table.

    Blocks are numbered in sorted key order, which for the arena's integer seeds is the
    order `np.unique` gives -- so the bootstrap draws the same resamples it always has.
    """
    keys = sorted({row[0] for row in rows})
    number = {key: i for i, key in enumerate(keys)}
    return Games(
        competitors=np.array([row[1] for row in rows], dtype=np.int64),
        ranks=np.array([row[2] for row in rows], dtype=np.int64),
        won=np.array([row[3] for row in rows], dtype=np.float64),
        blocks=np.array([number[row[0]] for row in rows], dtype=np.int64),
        n_blocks=len(keys),
    )


@dataclass(frozen=True)
class Fitted:
    """Ratings for every competitor in a table, indexed as the table indexes them."""

    matrix: np.ndarray
    wins: np.ndarray
    elo: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    played: np.ndarray
    share: np.ndarray
    #: Seed blocks each competitor played in. An interval resampled from fewer than two
    #: is a single point -- certainty the data does not have.
    blocks: np.ndarray


def rate(
    games: Games,
    n_competitors: int,
    anchor: int,
    *,
    prior_sd: float = 400.0,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Fitted:
    """Bradley-Terry over a table, anchored at `anchor`, with block-bootstrap intervals."""
    matrix = block_pair_matrix(
        games.blocks, games.competitors, games.ranks, n_competitors, games.n_blocks
    )
    wins = pair_counts(matrix, n_competitors)
    elo = bradley_terry(wins, anchor, prior_sd=prior_sd)
    lo, hi = bootstrap_elo(
        matrix,
        n_competitors,
        anchor,
        resamples=resamples,
        alpha=alpha,
        seed=seed,
        prior_sd=prior_sd,
    )
    played = (wins + wins.T).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(played > 0, wins.sum(axis=1) / played, np.nan)
    per_block = matrix.reshape(games.n_blocks, n_competitors, n_competitors)
    blocks = ((per_block.sum(axis=2) + per_block.sum(axis=1)) > 0).sum(axis=0)
    return Fitted(
        matrix=matrix,
        wins=wins,
        elo=elo,
        lo=lo,
        hi=hi,
        played=played,
        share=share,
        blocks=blocks,
    )


def _arena_rows[K: Hashable](
    store: Store, config_id: int, competitor: dict[K, int], key_of: dict[int, K]
) -> list[tuple[int, list[int], list[int], list[float]]]:
    """Every arena game of a configuration as `tabulate` rows.

    Blocks are keyed by their seed. Two matches that used the same seed root share decks,
    so their games belong to the *same* block for resampling purposes -- treating them as
    distinct would quietly restore the independence assumption blocks exist to remove.
    """
    rows = store.seat_rows(config_id)
    if not rows:
        return []
    seats_per_game = 1 + max(int(r["seat"]) for r in rows)
    out: list[tuple[int, list[int], list[int], list[float]]] = []
    # `seat_rows` orders by (game_id, seat) and every game in a configuration has the same
    # seat count, so the flat rows tile the table exactly.
    for start in range(0, len(rows), seats_per_game):
        game = rows[start : start + seats_per_game]
        out.append(
            (
                int(game[0]["block_seed"]),
                [competitor[key_of[int(r["agent_id"])]] for r in game],
                [int(r["rank"]) for r in game],
                [float(r["won"]) for r in game],
            )
        )
    return out


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
    agents = store.agents_in(config_id)
    key_of = {int(a["agent_id"]): int(a["agent_id"]) for a in agents}
    competitor = {int(a["agent_id"]): i for i, a in enumerate(agents)}
    rows = _arena_rows(store, config_id, competitor, key_of)
    if not rows:
        raise NoGamesError(config_id)
    games = tabulate(rows)
    n_agents = len(agents)

    # The anchor is the agent whose spec is *exactly* the family name. A tuned variant is
    # written `h3@configs/agents/h3_v2.toml` and shares the family but not the constants;
    # anchoring on one would silently re-base the whole scale to whichever tuning happened
    # to be registered first, and every number would still look reasonable.
    anchor = next((i for i in range(n_agents) if str(agents[i]["spec"]) == anchor_family), None)
    if anchor is None:
        raise NoAnchorError(anchor_family)

    fitted = rate(
        games, n_agents, anchor, prior_sd=prior_sd, resamples=resamples, alpha=alpha, seed=seed
    )
    order = np.argsort(-fitted.elo)
    standings = [
        Standing(
            agent_id=int(agents[i]["agent_id"]),
            spec=str(agents[i]["spec"]),
            behaviour_hash=str(agents[i]["behaviour_hash"]),
            elo=float(fitted.elo[i]),
            lo=float(fitted.lo[i]),
            hi=float(fitted.hi[i]),
            games=int(fitted.played[i]),
            score=float(fitted.share[i]),
        )
        for i in order
    ]

    config = next(c for c in store.configs() if int(c["config_id"]) == config_id)
    report = Report(
        config_id=config_id,
        map_name=str(config["map_name"]),
        n_players=int(config["n_players"]),
        standings=standings,
        wins=fitted.wins,
        win_rate=win_rate_matrix(fitted.wins),
        cycle_fraction=cycle_fraction(fitted.wins),
        seat_score=seat_advantage(games.blocks, games.won, resamples=resamples, seed=seed),
        n_games=games.n_games,
        n_blocks=games.n_blocks,
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
        n_games=games.n_games,
        n_blocks=games.n_blocks,
        cycle_fraction=report.cycle_fraction,
    )
    return replace(report, rating_run_id=run_id)


# ---------------------------------------------------------------------------
# Arena and human games together
# ---------------------------------------------------------------------------


#: A competitor in `fit_with`: an agent's identity (`AgentRow.key`), or a person (`human`).
type Competitor = tuple[str | None, ...]


def human(player_id: str) -> Competitor:
    """A person's competitor key."""
    return ("human", player_id)


@dataclass(frozen=True)
class ExternalGame:
    """A finished game recorded outside the arena -- a rated human game -- in fit terms.

    `key` names the game, which is its own seed block: nobody else was dealt its deck.
    `seats` holds each seat's competitor key, `ranks` its competition rank (1 is first).
    """

    key: str
    seats: tuple[Competitor, ...]
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class Rated:
    """One competitor's place in a combined fit."""

    key: Competitor
    #: The arena's spec for an agent it has recorded; `None` otherwise.
    spec: str | None
    elo: float
    lo: float
    hi: float
    games: int
    score: float
    blocks: int


@dataclass(frozen=True)
class Combined:
    standings: list[Rated]
    n_games: int
    n_blocks: int


def fit_with(
    store: Store | None,
    map_name: str,
    n_players: int,
    rules_hash: str,
    external: Sequence[ExternalGame],
    *,
    anchor: Competitor,
    prior_sd: float = 400.0,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Combined:
    """Every arena game of a configuration plus `external` games, on one scale.

    `anchor` is the competitor key of H3 itself (`AgentRow.key` of the spec `h3`); an
    arena agent and a bot in an external game with the same identity are the same
    competitor, which is what puts people and bots on one leaderboard. Nothing is
    persisted: the result is a view, refit whenever it is asked for.
    """
    competitor: dict[Competitor, int] = {}
    spec: dict[Competitor, str] = {}
    rows: list[tuple[tuple[str, int | str], list[int], list[int], list[float]]] = []
    config = None
    if store is not None:
        config = next(
            (
                c
                for c in store.configs()
                if (c["map_name"], int(c["n_players"]), c["rules_hash"])
                == (map_name, n_players, rules_hash)
            ),
            None,
        )
    if config is not None:
        assert store is not None
        config_id = int(config["config_id"])
        key_of: dict[int, Competitor] = {}
        # Arena agents first, in id order -- the order `fit` gives them.
        for agent in store.agents_in(config_id):
            row = AgentRow(
                spec=str(agent["spec"]),
                family=str(agent["family"]),
                params_hash=str(agent["params_hash"]),
                behaviour_hash=str(agent["behaviour_hash"]),
                checkpoint=agent["checkpoint"],
            )
            key_of[int(agent["agent_id"])] = row.key
            competitor.setdefault(row.key, len(competitor))
            spec.setdefault(row.key, row.spec)
        rows = [
            (("seed", block), seats, ranks, won)
            for block, seats, ranks, won in _arena_rows(store, config_id, competitor, key_of)
        ]
    for game in external:
        for key in game.seats:
            competitor.setdefault(key, len(competitor))
        rows.append(
            (
                ("game", game.key),
                [competitor[key] for key in game.seats],
                list(game.ranks),
                [float(rank == 1) for rank in game.ranks],
            )
        )
    if not rows:
        raise NoGamesError(-1)
    if anchor not in competitor:
        raise NoAnchorError(str(anchor[0]))

    games = tabulate(rows)
    fitted = rate(
        games,
        len(competitor),
        competitor[anchor],
        prior_sd=prior_sd,
        resamples=resamples,
        alpha=alpha,
        seed=seed,
    )
    keys = list(competitor)
    order = np.argsort(-fitted.elo, kind="stable")
    return Combined(
        standings=[
            Rated(
                key=keys[i],
                spec=spec.get(keys[i]),
                elo=float(fitted.elo[i]),
                lo=float(fitted.lo[i]),
                hi=float(fitted.hi[i]),
                games=int(fitted.played[i]),
                score=float(fitted.share[i]),
                blocks=int(fitted.blocks[i]),
            )
            for i in order
        ],
        n_games=games.n_games,
        n_blocks=games.n_blocks,
    )
