"""People and bots on one leaderboard (PLAN.md §14.2).

Finished rated human games join the arena's games in one Bradley-Terry fit on the H3 = 0
scale (`eval.leaderboard.fit_with`): each human game its own seed block, a person
identified by player id, a bot by its agent identity -- so the H3 a person beat is the same
competitor the arena measured. Nothing is stored: a rating is a view, refit whenever a
rated game finishes or the arena records more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ttr_rust

from ticket_to_ride.agents.registry import ANCHOR_FAMILY, AgentSpec, parse_spec
from ticket_to_ride.eval.leaderboard import (
    Combined,
    Competitor,
    ExternalGame,
    NoAnchorError,
    NoGamesError,
    Rated,
    fit_with,
    human,
)
from ticket_to_ride.eval.store import AgentRow, Store
from ticket_to_ride.ui.web import protocol as p
from ticket_to_ride.ui.web.games import N_PLAYERS, player_model
from ticket_to_ride.ui.web.store import PlayStore, bot_id

MAPS: tuple[p.MapName, ...] = ("usa", "mini")


@dataclass(frozen=True)
class Unrated:
    """Why there is no fit yet."""

    reason: str


class Ratings:
    """The combined fit per map, cached until its inputs change."""

    def __init__(
        self,
        store: PlayStore,
        results: Path | None,
        opponents: dict[str, AgentSpec],
        *,
        resamples: int = 1000,
    ) -> None:
        self.store = store
        self.results = results
        self.opponents = opponents
        self.resamples = resamples
        self.anchor = parse_spec(ANCHOR_FAMILY).identity().key
        self._cache: dict[str, tuple[object, Combined | Unrated]] = {}

    def _inputs(self) -> object:
        """Changes exactly when a refit could: a rated game finished, or the arena wrote.
        The results store is in WAL mode, so its `-wal` file moves before the main one."""
        stats: list[object] = [self.store.version()]
        if self.results is not None:
            for suffix in ("", "-wal"):
                path = self.results.with_name(self.results.name + suffix)
                stats.append(
                    (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
                )
        return tuple(stats)

    def fit(self, map_name: p.MapName) -> Combined | Unrated:
        inputs = self._inputs()
        cached = self._cache.get(map_name)
        if cached is not None and cached[0] == inputs:
            return cached[1]
        result = self._fit(map_name)
        self._cache[map_name] = (inputs, result)
        return result

    def _fit(self, map_name: p.MapName) -> Combined | Unrated:
        rules_hash = ttr_rust.Game(map_name, N_PLAYERS).rules_hash
        external = [
            ExternalGame(
                key=game.game_id,
                seats=tuple(self._competitor(seat.player_id) for seat in seats),
                ranks=tuple(int(seat.rank or 0) for seat in seats),
            )
            for game, seats in self.store.finished_rated(map_name, N_PLAYERS, rules_hash)
        ]
        arena = Store(self.results) if self.results is not None and self.results.exists() else None
        try:
            return fit_with(
                arena,
                map_name,
                N_PLAYERS,
                rules_hash,
                external,
                anchor=self.anchor,
                resamples=self.resamples,
            )
        except NoGamesError:
            return Unrated("no games have been recorded on this map yet")
        except NoAnchorError:
            return Unrated(f"nobody has played {ANCHOR_FAMILY.upper()}, the scale's zero, here yet")
        finally:
            if arena is not None:
                arena.close()

    def _competitor(self, player_id: str) -> Competitor:
        player = self.store.player(player_id)
        assert player is not None
        identity = player.identity
        return human(player_id) if identity is None else identity.key

    def _player(self, rated: Rated) -> p.Player:
        key = rated.key
        if key[0] == "human":
            row = self.store.player(str(key[1]))
            assert row is not None
            return player_model(row)
        family, params_hash, behaviour_hash, checkpoint = (
            str(key[0]),
            str(key[1]),
            str(key[2]),
            key[3],
        )
        spec = rated.spec or family
        identity = AgentRow(spec, family, params_hash, behaviour_hash, checkpoint)
        row = self.store.player(bot_id(identity))
        if row is not None:
            return player_model(row)
        return p.Player(id=bot_id(identity), kind="bot", name=_label(spec), spec=spec)

    # -- the endpoints --------------------------------------------------

    def leaderboard(self, map_name: p.MapName) -> p.Leaderboard:
        result = self.fit(map_name)
        if isinstance(result, Unrated):
            return p.Leaderboard(
                map=map_name,
                n_players=N_PLAYERS,
                anchor=ANCHOR_FAMILY,
                games=0,
                blocks=0,
                standings=[],
                note=result.reason,
            )
        return p.Leaderboard(
            map=map_name,
            n_players=N_PLAYERS,
            anchor=ANCHOR_FAMILY,
            games=result.n_games,
            blocks=result.n_blocks,
            standings=[
                p.Standing(player=self._player(r), rating=_rating(r)) for r in result.standings
            ],
            note=None,
        )

    def ratings_of(self, key: Competitor) -> list[p.MapRating]:
        out = []
        for map_name in MAPS:
            result = self.fit(map_name)
            if isinstance(result, Unrated):
                continue
            rated = next((r for r in result.standings if r.key == key), None)
            if rated is not None:
                out.append(p.MapRating(map=map_name, rating=_rating(rated)))
        return out

    def agents(self) -> list[p.Agent]:
        out = []
        for spec in self.opponents.values():
            identity = spec.identity()
            out.append(
                p.Agent(
                    spec=spec.text,
                    name=spec.name,
                    family=spec.family,
                    description=spec.description,
                    anchor=spec.is_anchor,
                    player=p.Player(
                        id=bot_id(identity), kind="bot", name=spec.name, spec=spec.text
                    ),
                    behaviour_hash=identity.behaviour_hash,
                    ratings=self.ratings_of(identity.key),
                )
            )
        return out

    def profile(self, player_id: str) -> p.Profile | None:
        row = self.store.player(player_id)
        if row is None:
            return None
        records: dict[tuple[p.MapName, str], dict[str, float]] = {}
        opponents: dict[str, p.Player] = {}
        results = []
        for game in self.store.games_of(player_id, limit=1000):
            if game.status != "finished" or game.mode != "play":
                continue
            seats = self.store.seats(game.game_id)
            mine = next(s for s in seats if s.player_id == player_id)
            theirs = next(s for s in seats if s.player_id != player_id)
            other = self.store.player(theirs.player_id)
            assert other is not None and mine.score is not None and theirs.score is not None
            opponents[other.player_id] = player_model(other)
            result = _result(mine.rank, theirs.rank)
            results.append(
                p.GameResult(
                    game_id=game.game_id,
                    map=game.map_name,
                    rated=game.rated,
                    opponent=opponents[other.player_id],
                    seat=mine.seat,
                    score=mine.score,
                    opponent_score=theirs.score,
                    result=result,
                    finished=game.finished_utc or "",
                )
            )
            tally = records.setdefault(
                (game.map_name, other.player_id),
                {"games": 0, "win": 0, "loss": 0, "draw": 0, "mine": 0.0, "theirs": 0.0},
            )
            tally["games"] += 1
            tally[result] += 1
            tally["mine"] += mine.score
            tally["theirs"] += theirs.score
        identity = row.identity
        return p.Profile(
            player=player_model(row),
            ratings=self.ratings_of(human(player_id) if identity is None else identity.key),
            record=[
                p.OpponentRecord(
                    map=map_name,
                    opponent=opponents[other],
                    games=int(t["games"]),
                    wins=int(t["win"]),
                    losses=int(t["loss"]),
                    draws=int(t["draw"]),
                    mean_score=t["mine"] / t["games"],
                    mean_opponent_score=t["theirs"] / t["games"],
                )
                for (map_name, other), t in sorted(records.items())
            ],
            games=results,
        )


#: Seed blocks a competitor needs before its interval means anything: resampling one block
#: always draws the same games, so the interval collapses to the point estimate.
MIN_INTERVAL_BLOCKS = 2


def _rating(rated: Rated) -> p.Rating:
    def bound(x: float) -> float | None:
        return x if math.isfinite(x) and rated.blocks >= MIN_INTERVAL_BLOCKS else None

    return p.Rating(
        elo=rated.elo,
        lo=bound(rated.lo),
        hi=bound(rated.hi),
        games=rated.games,
        score=rated.score if math.isfinite(rated.score) else None,
    )


def _result(mine: int | None, theirs: int | None) -> Literal["win", "loss", "draw"]:
    if mine == theirs:
        return "draw"
    return "win" if mine == 1 else "loss"


def _label(spec: str) -> str:
    try:
        return parse_spec(spec).name
    except ValueError, OSError:
        return spec
