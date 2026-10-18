"""Phase 4.2's exit criteria for play, through the API (PLAN.md §14.2).

Full games on both maps with a scripted person; a restart mid-game that rebuilds and
carries on identically, every bot reproducing its moves; duplicate, illegal and
out-of-turn moves answered clearly; no seed in any payload before the end; review only
once a game is over.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import ttr_rust
from fastapi.testclient import TestClient
from web_rig import Seeds, make_app, new_game, new_player, play_out

from ticket_to_ride.ui.web.store import ActionRow, PlayStore, now

#: A seed no payload could contain by accident: 19 digits, 16 hex characters.
SEED = 7_349_102_334_458_871_123


@pytest.mark.parametrize(("map_name", "opponent"), [("usa", "h3"), ("usa", "h4"), ("mini", "h3")])
def test_a_full_game_through_the_api(tmp_path: Path, map_name: str, opponent: str) -> None:
    with TestClient(make_app(tmp_path)) as client:
        player = new_player(client)
        game = new_game(client, player, map_name, opponent)
        played = play_out(client, game["id"], random.Random(map_name))
        assert played.over is not None
        scores = played.over["scores"]
        assert sorted(s["rank"] for s in scores) in ([1, 2], [1, 1])

        # The store holds the game exactly as it was played.
        summary = client.get(f"/api/games/{game['id']}").json()
        assert summary["status"] == "finished"
        assert summary["ply"] == len(played.actions)
        assert [s["score"] for s in summary["seats"]] == [s["total"] for s in scores]

        # And it replays: the same seed and moves give the same final position.
        state = ttr_rust.Game(map_name, 2).new_initial_state(int(played.over["seed"]))
        for action in played.actions:
            state.step(action)
        assert state.is_terminal()
        assert [s["total"] for s in scores] == state.final_scores()


def test_spectating_streams_every_move_with_its_reasoning(client: TestClient) -> None:
    response = client.post(
        "/api/games", json={"mode": "spectate", "map": "mini", "agents": ["h2", "h4"]}
    )
    assert response.status_code == 201, response.text
    game = response.json()
    assert game["human_seat"] is None
    played = play_out(client, game["id"], random.Random(0))
    events = [json.loads(m) for m in played.messages if '"type":"event"' in m]
    assert events, "nothing was streamed"
    for event in events:
        assert event["view"]["viewer"] is None, "a spectator's view is omniscient"
        assert all(seat["hand"] is not None for seat in event["view"]["seats"])
        explanation = event["explanation"]
        assert explanation is not None
        assert explanation["chosen"] == event["entry"]["action"]["id"]
        assert explanation["candidates"][0]["action"]["id"] == explanation["chosen"]
    assert played.over is not None


def test_a_restart_mid_game_rebuilds_and_carries_on_identically(tmp_path: Path) -> None:
    """The store is the truth and memory a cache: hang up, restart the server twice, and
    the game ends exactly as an uninterrupted one with the same seed and moves."""
    with TestClient(make_app(tmp_path / "whole", seeds=Seeds(SEED))) as client:
        game = new_game(client, new_player(client), "usa", "h4", rated=False, seat=1)
        whole = play_out(client, game["id"], random.Random(3))

    root, rng = tmp_path / "interrupted", random.Random(3)
    with TestClient(make_app(root, seeds=Seeds(SEED))) as client:
        game = new_game(client, new_player(client), "usa", "h4", rated=False, seat=1)
        play_out(client, game["id"], rng, stop_after=8)
    with TestClient(make_app(root)) as client:  # a new server over the same store
        play_out(client, game["id"], rng, stop_after=8)
    with TestClient(make_app(root)) as client:
        rest = play_out(client, game["id"], rng)

    assert whole.over is not None and rest.over is not None
    assert rest.actions == whole.actions
    assert rest.over["scores"] == whole.over["scores"]
    assert rest.over["view"] == whole.over["view"]


def test_a_game_left_on_a_bots_move_carries_on_after_a_restart(tmp_path: Path) -> None:
    """The server died between recording the person's move and the bot's reply: the
    rebuild finds the bot to move and plays it."""
    with TestClient(make_app(tmp_path, seeds=Seeds(SEED))) as client:
        game = new_game(client, new_player(client), "mini", "h3", rated=False, seat=0)
        game_id = game["id"]
    with PlayStore(tmp_path / "play.db") as store:
        state = ttr_rust.Game("mini", 2).new_initial_state(SEED)
        for row in store.actions(game_id):
            state.step(row.action)
        assert state.current_player() == 0
        move = state.legal_actions()[0]
        store.append(game_id, ActionRow(len(store.actions(game_id)), 0, move, now(), 0))
    with TestClient(make_app(tmp_path)) as client:
        with client.websocket_connect(f"/ws/games/{game_id}") as ws:
            snapshot = ws.receive_json()
        assert snapshot["log"][-1]["seat"] == 1, "the bot did not reply after the rebuild"
        assert snapshot["legal"], "the game should be waiting on the person again"


def test_a_bot_that_no_longer_plays_its_moves_is_reported(tmp_path: Path) -> None:
    """A rebuild re-asks every bot for every past decision; one that answers differently
    -- its code changed mid-game -- is an error, never papered over."""
    with TestClient(make_app(tmp_path)) as client:
        game_id = new_game(client, new_player(client), "mini", "h3", rated=False, seat=1)["id"]
    with PlayStore(tmp_path / "play.db") as store:
        first = store.actions(game_id)[0]
        assert first.seat == 0, "the bot moves first from seat 0"
        stored = store.game(game_id)
        assert stored is not None
        state = ttr_rust.Game("mini", 2).new_initial_state(stored.seed)
        other = next(a for a in state.legal_actions() if a != first.action)
        store.db.execute(
            "UPDATE action SET action = ? WHERE game_id = ? AND ply = 0", (other, game_id)
        )
    with TestClient(make_app(tmp_path)) as client:
        with client.websocket_connect(f"/ws/games/{game_id}") as ws:
            error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "replay_mismatch"
        assert "ply 0" in error["message"] and "H3 now plays" in error["message"]
        assert client.get(f"/api/games/{game_id}").status_code == 200


def test_acts_are_idempotent_on_ply_and_bad_ones_say_why(client: TestClient) -> None:
    game = new_game(client, new_player(client), "mini", "h3", rated=False, seat=0)
    with client.websocket_connect(f"/ws/games/{game['id']}") as ws:
        snapshot = ws.receive_json()
        seen = {"ply": snapshot["view"]["ply"], "legal": snapshot["legal"]}
        assert seen["ply"] == 0 and seen["legal"]

        def reply() -> dict:
            """The next ack or error, noting the move events streamed meanwhile."""
            while True:
                message = ws.receive_json()
                if message["type"] == "event":
                    seen.update(ply=message["view"]["ply"], legal=message["legal"])
                elif message["type"] in {"ack", "error"}:
                    return message

        def act(at: int, action: int) -> dict:
            ws.send_json({"type": "act", "ply": at, "action": action})
            return reply()

        opening = seen["legal"]
        first = opening[0]["id"]
        assert act(0, first) == {"type": "ack", "ply": 0, "action": first, "duplicate": False}
        # A double click: the same move for the same ply is acknowledged and changes nothing.
        assert act(0, first)["duplicate"] is True
        # A different move for a ply already played.
        assert act(0, opening[1]["id"])["code"] == "stale_ply"
        # A ply the game has not reached.
        assert act(50, first)["code"] == "future_ply"
        # Once the bot has replied it is the person's move again, at a later ply.
        while not seen["legal"]:
            message = ws.receive_json()
            if message["type"] == "event":
                seen.update(ply=message["view"]["ply"], legal=message["legal"])
        at = seen["ply"]
        # A move the engine refuses, with the engine's reason.
        refused = act(at, 0)
        assert refused["code"] == "illegal_action" and refused["ply"] == at
        assert refused["message"]
        # Not a move at all.
        ws.send_json({"type": "act", "ply": -1, "action": 0})
        assert reply()["code"] == "bad_message"
        ws.send_text("{not json")
        assert reply()["code"] == "bad_message"
        # None of that cost the person their turn: a legal move still plays.
        assert client.get(f"/api/games/{game['id']}").json()["ply"] == at
        assert act(at, seen["legal"][0]["id"])["duplicate"] is False


def test_nobody_acts_in_a_spectated_game_or_after_the_end(client: TestClient) -> None:
    spectated = client.post(
        "/api/games", json={"mode": "spectate", "map": "mini", "agents": ["h1", "h2"]}
    ).json()
    with client.websocket_connect(f"/ws/games/{spectated['id']}") as ws:
        ws.receive_json()
        ws.send_json({"type": "act", "ply": 0, "action": 0})
        while (message := ws.receive_json())["type"] != "error":
            pass
        assert message["code"] == "spectating"

    game = new_game(client, new_player(client), "mini", "h1", rated=False)
    played = play_out(client, game["id"], random.Random(1))
    with client.websocket_connect(f"/ws/games/{game['id']}") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        assert ws.receive_json()["type"] == "over"
        ws.send_json({"type": "act", "ply": len(played.actions), "action": 0})
        assert ws.receive_json()["code"] == "game_over"


def test_out_of_turn_is_refused_even_when_the_ply_matches(tmp_path: Path) -> None:
    """Defensive: with bots replying at once the person cannot normally reach the bot's
    turn at the current ply, so hold the bot back and try."""
    app = make_app(tmp_path)
    with TestClient(app) as client:
        service = app.state.service

        async def wait(live: object) -> None:
            return None

        service._advance = wait  # the bot never replies
        game = new_game(client, new_player(client), "mini", "h3", rated=False, seat=1)
        with client.websocket_connect(f"/ws/games/{game['id']}") as ws:
            snapshot = ws.receive_json()
            assert snapshot["view"]["current"] == 0 and not snapshot["legal"]
            ws.send_json({"type": "act", "ply": snapshot["view"]["ply"], "action": 0})
            assert ws.receive_json()["code"] == "not_your_turn"


def test_no_seed_in_any_payload_before_the_end(tmp_path: Path) -> None:
    """The seed fixes the deck order: one leak and every later card is known."""
    seed_texts = (str(SEED), f"{SEED:x}", f"{SEED:X}")
    with TestClient(make_app(tmp_path, seeds=Seeds(SEED))) as client:
        player = new_player(client)
        game = new_game(client, player, "usa", "h3")
        game_id = game["id"]
        paths = (
            f"/api/games/{game_id}",
            f"/api/games?player_id={player}",
            f"/api/games/{game_id}/review",
            f"/api/players/{player}",
            "/api/leaderboard?map=usa",
            "/api/agents",
        )
        payloads = [json.dumps(game), *(client.get(path).text for path in paths)]
        played = play_out(client, game_id, random.Random(9), stop_after=20)
        payloads += played.messages
        played = play_out(client, game_id, random.Random(9))
        payloads += played.messages[:-1]
        assert played.over is not None and json.loads(played.messages[-1])["type"] == "over"
        for payload in payloads:
            for text in seed_texts:
                assert text not in payload, f"the seed leaked before the end: {payload[:200]}"
        # Once it is over the seed is public, so the game can be replayed by anyone.
        assert played.over["seed"] == str(SEED)
        assert client.get(f"/api/games/{game_id}/review").json()["seed"] == str(SEED)


def test_review_waits_for_the_end_and_explains_every_bot_decision(client: TestClient) -> None:
    game = new_game(client, new_player(client), "usa", "h3", rated=False, seat=0)
    refused = client.get(f"/api/games/{game['id']}/review")
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "unavailable"

    played = play_out(client, game["id"], random.Random(4))
    review = client.get(f"/api/games/{game['id']}/review").json()
    assert [ply["entry"]["action"]["id"] for ply in review["plies"]] == played.actions
    assert review["initial"]["ply"] == 0 and review["initial"]["viewer"] is None
    for ply in review["plies"]:
        bot = ply["entry"]["seat"] == 1
        assert (ply["explanation"] is not None) == bot
        assert ply["view"]["viewer"] is None
        if bot:
            explanation = ply["explanation"]
            assert explanation["agent"] == "h3"
            assert explanation["candidates"][0]["action"]["id"] == ply["entry"]["action"]["id"]
            for candidate in explanation["candidates"]:
                if candidate["total"] is not None:
                    total = sum(term["value"] for term in candidate["terms"])
                    assert total == pytest.approx(candidate["total"])
    assert review["scores"] == played.over["scores"]  # ty: ignore[not-subscriptable]
    assert client.get("/api/games/nope/review").status_code == 404


def test_rated_games_seat_the_person_at_random(client: TestClient) -> None:
    player = new_player(client)
    seats = {new_game(client, player)["human_seat"] for _ in range(30)}
    assert seats == {0, 1}
    refused = client.post(
        "/api/games",
        json={"mode": "play", "map": "mini", "opponent": "h3", "player_id": player, "seat": 0},
    )
    assert refused.status_code == 400 and "random" in refused.json()["detail"]["message"]
    assert new_game(client, player, rated=False, seat=1)["human_seat"] == 1


def test_requests_naming_nothing_real_are_refused(client: TestClient) -> None:
    player = new_player(client)
    body = {"mode": "play", "map": "mini", "opponent": "h3@/etc/passwd", "player_id": player}
    assert client.post("/api/games", json=body).status_code == 404
    body = {"mode": "play", "map": "mini", "opponent": "h3", "player_id": "nobody"}
    assert client.post("/api/games", json=body).status_code == 404
    body = {"mode": "play", "map": "europe", "opponent": "h3", "player_id": player}
    assert client.post("/api/games", json=body).status_code == 422
    with client.websocket_connect("/ws/games/nope") as ws:
        assert ws.receive_json()["code"] == "not_found"
    assert client.get("/api/games/nope").status_code == 404
