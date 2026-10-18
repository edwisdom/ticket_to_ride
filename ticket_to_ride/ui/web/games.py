"""Live games: a cache over the play store.

**The store is the source of truth; memory is a cache** (PLAN.md §14.2). Every move is
appended to the store *before* the in-memory game advances, and a game not in memory is
rebuilt from `(seed, actions)`. A new game is built the same way, from its freshly
written row, so there is one path by which a live game comes to exist.

A rebuild also calls each bot at each of its past decisions and **requires it to
reproduce the recorded action**. That warms its plan cache and advances H0's stream, and it
makes every rebuild a replay check: a bot whose code changed mid-game is reported
(`replay_mismatch`), never papered over. So a restart, or one day a second server, loses
nothing and changes nothing.

Bots think in a worker thread, on a clone of the position, so a slow agent (Phase 5's
search) never blocks the event loop; moves stream to the client as they are made, and the
client paces the animation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anyio.to_thread
import ttr_rust

from ticket_to_ride.agents.registry import AgentSpec, Bot, parse_spec
from ticket_to_ride.ui.web import protocol as p
from ticket_to_ride.ui.web import views
from ticket_to_ride.ui.web.store import (
    ActionRow,
    FinalSeat,
    GameRow,
    PlayerRow,
    PlayStore,
    SeatRow,
    now,
)

#: Candidates kept per explanation. A keep has at most seven, so every keep shows whole.
TOP_K = 8

#: Seats in every game the server hosts today. The protocol is N-player throughout.
N_PLAYERS = 2

#: Finished reviews held in memory. Each is a replay with every bot decision explained.
REVIEW_CACHE = 32

log = logging.getLogger(__name__)


def server_seed() -> int:
    """A game's seed: a server secret, revealed when the game ends. 63 bits, so it fits the
    signed BIGINT both SQLite and Postgres store."""
    return secrets.randbits(63)


class GameError(Exception):
    """A request the game cannot honour, with the protocol's code for it."""

    def __init__(
        self, code: p.ErrorCode, message: str, *, status: int = 400, ply: int | None = None
    ) -> None:
        super().__init__(message)
        self.code: p.ErrorCode = code
        self.message = message
        self.status = status
        self.ply = ply

    def to_message(self) -> p.ErrorMessage:
        return p.ErrorMessage(type="error", code=self.code, message=self.message, ply=self.ply)


@dataclass
class Seat:
    player: PlayerRow
    spec: AgentSpec | None
    bot: Bot | None


def player_model(row: PlayerRow) -> p.Player:
    return p.Player(id=row.player_id, kind=row.kind, name=row.name, spec=row.spec)


def summary(
    row: GameRow, seats: Sequence[SeatRow], players: Sequence[PlayerRow], ply: int
) -> p.GameSummary:
    human = [s.seat for s, player in zip(seats, players, strict=True) if player.kind == "human"]
    return p.GameSummary(
        id=row.game_id,
        map=row.map_name,
        mode=row.mode,
        rated=row.rated,
        assist=row.assist,
        status=row.status,
        n_players=row.n_players,
        seats=[
            p.SeatInfo(seat=s.seat, player=player_model(player), score=s.score, rank=s.rank)
            for s, player in zip(seats, players, strict=True)
        ],
        human_seat=human[0] if row.mode == "play" and human else None,
        ply=ply,
        created=row.created_utc,
        finished=row.finished_utc,
    )


class LiveGame:
    """One game in memory: the engine state, its bots, and what has been sent about it.

    Rendered for the one viewer a game has -- the person's seat in a play game, everyone
    (omniscient) when spectating -- as each move is made, so a connection that falls
    behind catches up from `events` rather than re-reading a state that has moved on.
    """

    def __init__(
        self, row: GameRow, game: ttr_rust.Game, state: ttr_rust.State, seats: list[Seat]
    ) -> None:
        self.row = row
        self.game = game
        self.state = state
        self.seats = seats
        human = [i for i, seat in enumerate(seats) if seat.player.kind == "human"]
        self.viewer: int | None = human[0] if row.mode == "play" and human else None
        #: Spectators see each bot's reasoning live; rated play never does.
        self.explain_live = row.mode == "spectate"
        self.actions: list[int] = []
        self.played_by: list[int] = []
        self.log: list[p.LogEntry] = []
        #: Events for plies played since this game was loaded: `events[i]` is ply
        #: `events_from + i`. A connection's snapshot is never older than that.
        self.events: list[p.Event] = []
        self.events_from = 0
        self.over: p.Over | None = None
        self.final_seats: list[SeatRow] = []
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.runner: asyncio.Task[None] | None = None
        self.subscribers = 0
        self.turn_started = time.monotonic()

    @property
    def ply(self) -> int:
        return len(self.actions)

    @property
    def names(self) -> list[str]:
        return [seat.player.name for seat in self.seats]

    def notify(self) -> None:
        """Wake every connection waiting on this game. Each waits on the event it saw
        before sending, so a move made while it was sending is never missed."""
        self.changed.set()
        self.changed = asyncio.Event()

    def summary(self) -> p.GameSummary:
        seats = self.final_seats or [
            SeatRow(self.row.game_id, i, seat.player.player_id, *([None] * 8))
            for i, seat in enumerate(self.seats)
        ]
        return summary(self.row, seats, [seat.player for seat in self.seats], self.ply)

    def snapshot(self) -> p.Snapshot:
        return p.Snapshot(
            type="snapshot",
            game=self.summary(),
            view=views.view(self.state, self.viewer),
            legal=views.legal(self.game, self.state, self.viewer),
            log=list(self.log),
        )


class GameService:
    """Creates, rebuilds and advances games; every change goes through the store first."""

    def __init__(
        self,
        store: PlayStore,
        *,
        opponents: Sequence[str],
        git_sha: str | None = None,
        client_build: Callable[[], str | None] = lambda: None,
        seed: Callable[[], int] = server_seed,
        choose_seat: Callable[[int], int] = secrets.randbelow,
    ) -> None:
        self.store = store
        #: The agents this server will seat, by spec. A request may name only these: a
        #: spec can name a file, and a public server must not read files on request.
        self.opponents: dict[str, AgentSpec] = {text: parse_spec(text) for text in opponents}
        self.git_sha = git_sha
        self.client_build = client_build
        self.seed = seed
        self.choose_seat = choose_seat
        self._live: dict[str, LiveGame] = {}
        self._reviews: dict[str, p.Review] = {}

    async def close(self) -> None:
        for live in self._live.values():
            if live.runner is not None:
                live.runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await live.runner

    # -- creating and loading -------------------------------------------

    def opponent(self, text: str) -> AgentSpec:
        spec = self.opponents.get(text)
        if spec is None:
            raise GameError(
                "not_found", f"no agent {text!r} here; GET /api/agents lists them", status=404
            )
        return spec

    async def create(self, request: p.NewPlayGame | p.NewSpectateGame) -> LiveGame:
        """Write a new game to the store, load it from there, and let any bot to move play."""
        if isinstance(request, p.NewPlayGame):
            players = self._play_seats(request)
        else:
            players = [self._bot_player(self.opponent(text)) for text in request.agents]
        game = ttr_rust.Game(request.map, N_PLAYERS)
        row = GameRow(
            game_id=secrets.token_hex(8),
            map_name=request.map,
            n_players=N_PLAYERS,
            mode=request.mode,
            rated=isinstance(request, p.NewPlayGame) and request.rated,
            assist=p.ASSIST,
            status="active",
            seed=self.seed(),
            contract_version=ttr_rust.contract_version(),
            obs_version=ttr_rust.obs_version(),
            data_hash=game.data_hash,
            rules_hash=game.rules_hash,
            git_sha=self.git_sha,
            client_build=self.client_build(),
            created_utc=now(),
        )
        self.store.add_game(row, [player.player_id for player in players])
        return await self.live(row.game_id)

    def _play_seats(self, request: p.NewPlayGame) -> list[PlayerRow]:
        human = self.store.player(request.player_id)
        if human is None or human.kind != "human":
            raise GameError("not_found", f"no player {request.player_id!r}", status=404)
        bot = self._bot_player(self.opponent(request.opponent))
        if request.rated and request.seat is not None:
            raise GameError(
                "bad_message", "a rated game seats you at random: first-player advantage is real"
            )
        seat = self.choose_seat(N_PLAYERS) if request.seat is None else request.seat
        if not 0 <= seat < N_PLAYERS:
            raise GameError("bad_message", f"seat {seat} outside 0..{N_PLAYERS - 1}")
        players = [bot] * N_PLAYERS
        players[seat] = human
        return players

    def _bot_player(self, spec: AgentSpec) -> PlayerRow:
        return self.store.add_bot(spec.identity(), spec.name)

    async def live(self, game_id: str) -> LiveGame:
        """The game in memory, rebuilt from the store if it is not. Plays any bot whose
        turn it is -- a game left on a bot's move by a restart carries on."""
        live = self._live.get(game_id)
        if live is None:
            # No await between the lookup and the insert, so two connections cannot both
            # rebuild the same game.
            live = self._rebuild(game_id)
            self._live[game_id] = live
            if live.row.mode == "play":
                async with live.lock:
                    await self._advance(live)
        return live

    def _setup(self, row: GameRow) -> tuple[ttr_rust.Game, ttr_rust.State, list[Seat]]:
        """A fresh engine and fresh bots for `row`, refusing anything that would replay a
        different game than the one recorded."""
        game = ttr_rust.Game(row.map_name, row.n_players)
        recorded = (row.contract_version, row.rules_hash, row.data_hash)
        current = (ttr_rust.contract_version(), game.rules_hash, game.data_hash)
        if recorded != current:
            raise GameError(
                "unavailable",
                f"game {row.game_id} was played under contract/rules/board {recorded}, this "
                f"build is {current}: it cannot be replayed",
                status=409,
            )
        seats = []
        for seat_row in self.store.seats(row.game_id):
            player = self._player(seat_row.player_id)
            spec = bot = None
            if player.kind == "bot":
                spec, bot = self._seat_bot(player, row, seat_row.seat)
            seats.append(Seat(player=player, spec=spec, bot=bot))
        return game, game.new_initial_state(row.seed), seats

    def _seat_bot(self, player: PlayerRow, row: GameRow, seat: int) -> tuple[AgentSpec, Bot]:
        assert player.spec is not None and player.identity is not None
        spec = parse_spec(player.spec)
        current = spec.identity()
        if current.key != player.identity.key:
            raise GameError(
                "replay_mismatch",
                f"{player.name} is not the player it was when game {row.game_id} began: its "
                f"behaviour hash was {player.identity.behaviour_hash[:12]} and is "
                f"{current.behaviour_hash[:12]} now",
                status=409,
            )
        bot = spec.build()
        bot.begin_game(seat, row.seed)
        return spec, bot

    def _rebuild(self, game_id: str) -> LiveGame:
        row = self.store.game(game_id)
        if row is None:
            raise GameError("not_found", f"no game {game_id!r}", status=404)
        game, state, seats = self._setup(row)
        live = LiveGame(row, game, state, seats)
        for record in self.store.actions(game_id):
            seat = state.current_player()
            if record.seat != seat:
                raise _mismatch(row, record.ply, f"seat {record.seat} moved in seat {seat}'s turn")
            bot = seats[seat].bot
            if bot is not None:
                chosen = bot.act(state)
                if chosen != record.action:
                    raise _mismatch(
                        row,
                        record.ply,
                        f"{seats[seat].player.name} now plays "
                        f"{game.action_to_string(chosen)}, the store says "
                        f"{game.action_to_string(record.action)}",
                    )
            before = state.clone()
            try:
                state.step(record.action)
            except ttr_rust.IllegalAction as e:
                raise _mismatch(row, record.ply, f"the recorded action is illegal: {e}") from e
            live.actions.append(record.action)
            live.played_by.append(seat)
            live.log.append(
                views.entry(
                    game, before, record.action, state, viewer=live.viewer, names=live.names
                )
            )
        live.events_from = live.ply
        if state.is_terminal():
            live.final_seats = self.store.seats(game_id)
            live.over = self._over(live)
        return live

    # -- playing --------------------------------------------------------

    async def act(self, game_id: str, ply: int, action: int) -> p.Ack:
        """The person's move at `ply`. Idempotent on `ply`: the same action again is an
        acknowledged no-op, so a double click or a reconnect cannot play twice."""
        live = await self.live(game_id)
        if live.viewer is None:
            raise GameError("spectating", "nobody plays a spectated game", ply=ply)
        async with live.lock:
            if ply < live.ply:
                if live.actions[ply] == action and live.played_by[ply] == live.viewer:
                    return p.Ack(type="ack", ply=ply, action=action, duplicate=True)
                raise GameError("stale_ply", f"ply {ply} has already been played", ply=ply)
            if live.state.is_terminal():
                raise GameError("game_over", "the game is over", ply=ply)
            if ply > live.ply:
                raise GameError("future_ply", f"the game is at ply {live.ply}, not {ply}", ply=ply)
            if live.state.current_player() != live.viewer:
                raise GameError("not_your_turn", "it is not your turn", ply=ply)
            think_ms = round((time.monotonic() - live.turn_started) * 1000)
            self._apply(live, live.viewer, action, think_ms)
            await self._advance(live)
        return p.Ack(type="ack", ply=ply, action=action, duplicate=False)

    def watch(self, live: LiveGame) -> None:
        """A spectator arrived: start the bots if they are not already playing."""
        if live.row.mode == "spectate" and live.over is None and live.runner is None:
            live.runner = asyncio.create_task(self._run(live))

    async def _run(self, live: LiveGame) -> None:
        try:
            async with live.lock:
                await self._advance(live)
        except Exception:
            # Nobody awaits a spectated game's task, so say what stopped it -- and drop
            # the game from memory: the store holds every move made, and the next
            # connection's rebuild either carries on or reports the cause.
            log.exception("spectated game %s stopped", live.row.game_id)
            self._live.pop(live.row.game_id, None)

    async def _advance(self, live: LiveGame) -> None:
        """Play bot seats until a person is to move or the game ends. Call with the lock."""
        while not live.state.is_terminal():
            seat = live.state.current_player()
            spec, bot = live.seats[seat].spec, live.seats[seat].bot
            if bot is None or spec is None:
                break
            thinking = live.state.clone()
            started = time.perf_counter()
            explained = None
            if live.explain_live:
                explained = await anyio.to_thread.run_sync(bot.explain, thinking, TOP_K)
                action = explained.chosen
            else:
                action = await anyio.to_thread.run_sync(bot.act, thinking)
            think_ms = round((time.perf_counter() - started) * 1000)
            explanation = (
                None
                if explained is None
                else views.explanation(live.game, live.state, explained, spec.text)
            )
            self._apply(live, seat, action, think_ms, explanation)
        live.turn_started = time.monotonic()

    def _apply(
        self,
        live: LiveGame,
        seat: int,
        action: int,
        think_ms: int,
        explanation: p.Explanation | None = None,
    ) -> None:
        """Play one move: checked on a copy, written to the store, and only then made."""
        before = live.state
        after = before.clone()
        try:
            after.step(action)
        except ttr_rust.IllegalAction as e:
            raise GameError("illegal_action", str(e), ply=live.ply) from e
        record = ActionRow(
            ply=live.ply, seat=seat, action=action, server_utc=now(), think_ms=think_ms
        )
        final = _final(after) if after.is_terminal() else ()
        self.store.append(
            live.row.game_id,
            record,
            final=final,
            final_hash=f"{after.state_hash():016x}" if final else None,
        )

        live.state = after
        live.actions.append(action)
        live.played_by.append(seat)
        entry = views.entry(live.game, before, action, after, viewer=live.viewer, names=live.names)
        live.log.append(entry)
        live.events.append(
            p.Event(
                type="event",
                entry=entry,
                view=views.view(after, live.viewer),
                legal=views.legal(live.game, after, live.viewer),
                explanation=explanation,
                think_ms=think_ms,
            )
        )
        if final:
            finished = self.store.game(live.row.game_id)
            assert finished is not None
            live.row = finished
            live.final_seats = self.store.seats(live.row.game_id)
            live.over = self._over(live)
        live.notify()

    def _over(self, live: LiveGame) -> p.Over:
        return p.Over(
            type="over",
            game=live.summary(),
            view=views.view(live.state, None),
            scores=views.scores(live.state),
            seed=str(live.row.seed),
        )

    def release(self, live: LiveGame) -> None:
        """A connection left. A finished game nobody is watching leaves memory; the store
        still has all of it."""
        live.subscribers -= 1
        if live.over is not None and live.subscribers <= 0:
            self._live.pop(live.row.game_id, None)

    # -- reading --------------------------------------------------------

    def summary(self, game_id: str) -> p.GameSummary:
        live = self._live.get(game_id)
        if live is not None:
            return live.summary()
        row = self.store.game(game_id)
        if row is None:
            raise GameError("not_found", f"no game {game_id!r}", status=404)
        return self._stored_summary(row)

    def _stored_summary(self, row: GameRow) -> p.GameSummary:
        seats = self.store.seats(row.game_id)
        players = [self._player(s.player_id) for s in seats]
        return summary(row, seats, players, len(self.store.actions(row.game_id)))

    def _player(self, player_id: str) -> PlayerRow:
        player = self.store.player(player_id)
        if player is None:
            raise GameError("unavailable", f"a seat names unknown player {player_id!r}", status=500)
        return player

    def games_of(self, player_id: str) -> list[p.GameSummary]:
        return [self.summary(row.game_id) for row in self.store.games_of(player_id)]

    async def review(self, game_id: str) -> p.Review:
        """A finished game, every hidden thing revealed and each bot decision explained --
        recomputed by replaying it, never stored."""
        row = self.store.game(game_id)
        if row is None:
            raise GameError("not_found", f"no game {game_id!r}", status=404)
        if row.status != "finished":
            raise GameError(
                "unavailable",
                "a review reveals every hand: it opens once the game is over",
                status=409,
            )
        review = self._reviews.get(game_id)
        if review is None:
            review = await anyio.to_thread.run_sync(self._replay, row)
            if len(self._reviews) >= REVIEW_CACHE:
                self._reviews.pop(next(iter(self._reviews)))
            self._reviews[game_id] = review
        return review

    def _replay(self, row: GameRow) -> p.Review:
        game, state, seats = self._setup(row)
        names = [seat.player.name for seat in seats]
        initial = views.view(state, None)
        plies = []
        for record in self.store.actions(row.game_id):
            seat = seats[state.current_player()]
            explanation = None
            if seat.bot is not None and seat.spec is not None:
                explained = seat.bot.explain(state, TOP_K)
                if explained.chosen != record.action:
                    raise _mismatch(row, record.ply, f"{seat.player.name} now plays differently")
                explanation = views.explanation(game, state, explained, seat.spec.text)
            before = state.clone()
            state.step(record.action)
            plies.append(
                p.ReviewPly(
                    entry=views.entry(game, before, record.action, state, viewer=None, names=names),
                    view=views.view(state, None),
                    explanation=explanation,
                    think_ms=record.think_ms,
                )
            )
        return p.Review(
            game=self._stored_summary(row),
            seed=str(row.seed),
            initial=initial,
            plies=plies,
            scores=views.scores(state),
        )


def _final(state: ttr_rust.State) -> list[FinalSeat]:
    return [
        FinalSeat(
            seat=s.seat,
            score=s.total,
            routes=s.routes,
            tickets_made=s.tickets_made,
            tickets_missed=s.tickets_missed,
            ticket_points=s.ticket_points,
            longest_bonus=s.longest_bonus,
            longest_trail=s.longest_trail,
            rank=s.rank,
        )
        for s in views.scores(state)
    ]


def _mismatch(row: GameRow, ply: int, what: str) -> GameError:
    return GameError(
        "replay_mismatch",
        f"game {row.game_id} does not replay at ply {ply}: {what}",
        status=409,
        ply=ply,
    )
