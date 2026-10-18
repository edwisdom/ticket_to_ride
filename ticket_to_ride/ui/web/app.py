"""The FastAPI app: REST, the game WebSocket, and the built client.

docs/WEB_PROTOCOL.md describes the endpoints; `protocol.py` defines every payload. The
app holds no game logic: `GameService` owns games, `Ratings` the leaderboard, the engine
every rule.

Everything stateful -- the play store, the service -- is made in the lifespan, inside the
event loop that will serve it, so a fresh app (a restart, a test) starts clean over the
same database.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated

import anyio
import anyio.to_thread
import ttr_rust
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import ValidationError

from ticket_to_ride.agents.registry import TIERS
from ticket_to_ride.eval.arena import git_sha
from ticket_to_ride.ui.web import protocol as p
from ticket_to_ride.ui.web import views
from ticket_to_ride.ui.web.games import (
    GameError,
    GameService,
    LiveGame,
    player_model,
    server_seed,
)
from ticket_to_ride.ui.web.glossary import glossary
from ticket_to_ride.ui.web.ratings import MAPS, Ratings
from ticket_to_ride.ui.web.store import PlayStore

#: Where the client's build lands (web/ builds it with Vite; gitignored).
STATIC = Path(__file__).resolve().parent / "static"

#: Who the server seats unless told otherwise.
OPPONENTS = TIERS

PLACEHOLDER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ticket to Ride</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:36rem;margin:4rem auto;padding:0 1rem}
code{background:#f3efe4;padding:.1em .3em;border-radius:3px}</style></head>
<body><h1>The server is running</h1>
<p>The web client has not been built yet. Build it from <code>web/</code> (see the README) into
<code>ticket_to_ride/ui/web/static/</code>, then reload.</p>
<p>The API is up: try <a href="/api/agents"><code>/api/agents</code></a> or
<a href="/docs"><code>/docs</code></a>.</p></body></html>
"""


def client_build(static: Path) -> str | None:
    """Which client build is being served: a hash of its `index.html`, which names every
    hashed asset Vite emitted. Recorded with each game."""
    index = static / "index.html"
    if not index.exists():
        return None
    return hashlib.blake2b(index.read_bytes(), digest_size=6).hexdigest()


def create_app(
    *,
    db: Path = Path("runs/play.db"),
    results: Path | None = Path("runs/results.db"),
    opponents: Sequence[str] = OPPONENTS,
    static: Path = STATIC,
    seed: Callable[[], int] | None = None,
    choose_seat: Callable[[int], int] | None = None,
    resamples: int = 1000,
) -> FastAPI:
    """The server. `seed` and `choose_seat` default to server secrets; tests pin them."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = PlayStore(db)
        service = GameService(
            store,
            opponents=opponents,
            git_sha=git_sha(),
            client_build=lambda: client_build(static),
            seed=seed or server_seed,
            choose_seat=choose_seat or secrets.randbelow,
        )
        app.state.service = service
        app.state.ratings = Ratings(store, results, service.opponents, resamples=resamples)
        # Each agent's identity is a couple of hundred milliseconds of probe games, memoized;
        # paid here, off the event loop, rather than by the first game against it.
        await anyio.to_thread.run_sync(
            lambda: [spec.identity() for spec in service.opponents.values()]
        )
        try:
            yield
        finally:
            await service.close()
            store.close()

    app = FastAPI(
        title="Ticket to Ride",
        summary="Play and watch the heuristics; see how they reason.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.exception_handler(GameError)
    async def game_error(_: Request, e: GameError) -> JSONResponse:
        detail = p.ApiError(code=e.code, message=e.message)
        return JSONResponse({"detail": detail.model_dump()}, status_code=e.status)

    def service(request: Request) -> GameService:
        return request.app.state.service

    def ratings(request: Request) -> Ratings:
        return request.app.state.ratings

    # -- reference data -------------------------------------------------

    @app.get("/api/meta")
    async def meta() -> p.Meta:
        return p.Meta(
            contract_version=ttr_rust.contract_version(),
            obs_version=ttr_rust.obs_version(),
            assist=p.ASSIST,
            maps=list(MAPS),
            client_build=client_build(static),
        )

    @app.get("/api/boards/{map_name}")
    async def board(map_name: p.MapName) -> p.Board:
        return views.board(map_name)

    @app.get("/api/glossary")
    async def explain_terms() -> p.Glossary:
        return glossary()

    @app.get("/api/agents")
    async def agents(request: Request) -> list[p.Agent]:
        return await anyio.to_thread.run_sync(ratings(request).agents)

    @app.get("/api/leaderboard")
    async def leaderboard(
        request: Request, map_name: Annotated[p.MapName, Query(alias="map")]
    ) -> p.Leaderboard:
        return await anyio.to_thread.run_sync(ratings(request).leaderboard, map_name)

    # -- players --------------------------------------------------------

    @app.post("/api/players", status_code=201)
    async def new_player(request: Request, body: p.NewPlayer) -> p.Player:
        return player_model(service(request).store.add_human(body.name.strip()))

    @app.get("/api/players/{player_id}")
    async def profile(request: Request, player_id: str) -> p.Profile:
        found = await anyio.to_thread.run_sync(ratings(request).profile, player_id)
        if found is None:
            raise GameError("not_found", f"no player {player_id!r}", status=404)
        return found

    # -- games ----------------------------------------------------------

    @app.post("/api/games", status_code=201)
    async def new_game(request: Request, body: p.NewGame) -> p.GameSummary:
        live = await service(request).create(body.root)
        return live.summary()

    @app.get("/api/games")
    async def games(request: Request, player_id: str) -> list[p.GameSummary]:
        return service(request).games_of(player_id)

    @app.get("/api/games/{game_id}")
    async def game(request: Request, game_id: str) -> p.GameSummary:
        return service(request).summary(game_id)

    @app.get("/api/games/{game_id}/review")
    async def review(request: Request, game_id: str) -> p.Review:
        return await service(request).review(game_id)

    @app.websocket("/ws/games/{game_id}")
    async def socket(websocket: WebSocket, game_id: str) -> None:
        await play_socket(websocket, websocket.app.state.service, game_id)

    # -- the client -----------------------------------------------------

    @app.get("/{path:path}", include_in_schema=False)
    async def client(path: str) -> Response:
        if path.split("/", 1)[0] in {"api", "ws"}:
            detail = p.ApiError(code="not_found", message=f"no endpoint /{path}")
            return JSONResponse({"detail": detail.model_dump()}, status_code=404)
        index = static / "index.html"
        if not index.exists():
            return HTMLResponse(PLACEHOLDER)
        target = (static / path).resolve()
        if path and target.is_file() and target.is_relative_to(static.resolve()):
            return FileResponse(target)
        # A client-side route: the single-page app takes it from here.
        return FileResponse(index)

    return app


async def play_socket(websocket: WebSocket, service: GameService, game_id: str) -> None:
    """One connection to one game: a snapshot, then every move as it is made, and the
    person's `act`s answered with an `ack` or an `error`."""
    await websocket.accept()
    try:
        live = await service.live(game_id)
    except GameError as e:
        await websocket.send_text(e.to_message().model_dump_json())
        await websocket.close(code=1008)
        return

    live.subscribers += 1
    send_lock = asyncio.Lock()

    async def send(message: p.Model) -> None:
        async with send_lock:
            await websocket.send_text(message.model_dump_json())

    try:
        # Taken and sent before anything else runs, so the stream below starts exactly
        # where the snapshot ends.
        sent = live.ply
        await send(live.snapshot())
        service.watch(live)
        # anyio's task group, not asyncio's: Starlette ends a connection by cancelling it
        # through an anyio cancel scope, and asyncio.TaskGroup's own cancellation
        # bookkeeping turns that into a CancelledError escaping the handler.
        async with anyio.create_task_group() as group:
            group.start_soon(_stream, live, sent, send)
            await _listen(websocket, service, game_id, send)
            group.cancel_scope.cancel()
    except* WebSocketDisconnect, OSError:
        # The client went away -- cleanly, or mid-send (uvicorn raises an OSError then).
        pass
    finally:
        service.release(live)


type Send = Callable[[p.Model], Awaitable[None]]


async def _stream(live: LiveGame, sent: int, send: Send) -> None:
    """Every event after ply `sent`, then `over`."""
    over_sent = False
    while True:
        changed = live.changed
        while sent < live.ply:
            await send(live.events[sent - live.events_from])
            sent += 1
        if live.over is not None and not over_sent:
            await send(live.over)
            over_sent = True
        await changed.wait()


async def _listen(websocket: WebSocket, service: GameService, game_id: str, send: Send) -> None:
    """Client messages until the client goes away."""
    while True:
        raw = await websocket.receive_text()
        try:
            act = p.ClientMessage.model_validate_json(raw).root
        except ValidationError as e:
            error = p.ErrorMessage(
                type="error", code="bad_message", message=_first_error(e), ply=None
            )
            await send(error)
            continue
        try:
            reply: p.Model = await service.act(game_id, act.ply, act.action)
        except GameError as e:
            reply = e.to_message()
        await send(reply)


def _first_error(e: ValidationError) -> str:
    error = e.errors()[0]
    where = ".".join(str(part) for part in error["loc"])
    return f"{where}: {error['msg']}" if where else error["msg"]
