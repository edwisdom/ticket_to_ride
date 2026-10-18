"""Driving the game server as its client does: REST to set up, the game socket to play.

A helper module beside the root conftest, which puts `tests/` on the path.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ticket_to_ride.ui.web.app import create_app


class Seeds:
    """Game seeds for a test, in order. The server draws them from `secrets` otherwise."""

    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def __call__(self) -> int:
        return self.values.pop(0)


def make_app(
    root: Path,
    *,
    seeds: Callable[[], int] | None = None,
    seats: Callable[[int], int] | None = None,
    results: Path | None = None,
    static: Path | None = None,
) -> FastAPI:
    """A server over `root/play.db`. A new one over the same directory is a restart."""
    return create_app(
        db=root / "play.db",
        results=results,
        seed=seeds,
        choose_seat=seats,
        resamples=50,
        static=static if static is not None else root / "no-client-built",
    )


def new_player(client: TestClient, name: str = "Ann") -> str:
    response = client.post("/api/players", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def new_game(
    client: TestClient,
    player: str,
    map_name: str = "mini",
    opponent: str = "h3",
    *,
    rated: bool = True,
    seat: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "mode": "play",
        "map": map_name,
        "opponent": opponent,
        "player_id": player,
        "rated": rated,
    }
    if seat is not None:
        body["seat"] = seat
    response = client.post("/api/games", json=body)
    assert response.status_code == 201, response.text
    return response.json()


@dataclass
class Played:
    """What a connection saw."""

    #: Every ply's action id, from the snapshot's log and each event.
    actions: list[int] = field(default_factory=list)
    #: Every server message, as sent, in order.
    messages: list[str] = field(default_factory=list)
    over: dict[str, Any] | None = None
    acts: int = 0


def play_out(
    client: TestClient,
    game_id: str,
    rng: random.Random,
    *,
    stop_after: int | None = None,
) -> Played:
    """Play the person's seat with uniformly random legal moves until the game ends, or
    until `stop_after` moves have been acknowledged -- then hang up mid-game."""
    played = Played()
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:

        def receive() -> dict[str, Any]:
            text = ws.receive_text()
            played.messages.append(text)
            return json.loads(text)

        snapshot = receive()
        assert snapshot["type"] == "snapshot", snapshot
        played.actions = [entry["action"]["id"] for entry in snapshot["log"]]
        legal, ply = snapshot["legal"], snapshot["view"]["ply"]
        while snapshot["game"]["status"] == "active":
            if legal:
                if stop_after is not None and played.acts == stop_after:
                    break
                ws.send_json({"type": "act", "ply": ply, "action": rng.choice(legal)["id"]})
                legal = []
                played.acts += 1
            message = receive()
            if message["type"] == "event":
                played.actions.append(message["entry"]["action"]["id"])
                legal, ply = message["legal"], message["view"]["ply"]
            elif message["type"] == "over":
                played.over = message
                break
            else:
                assert message["type"] == "ack" and not message["duplicate"], message
        else:
            played.over = receive()
    return played
