"""Human games rated on the bots' scale (PLAN.md §14.2): one Bradley-Terry fit over arena
and rated human games, H3 = 0, each human game its own block, a person by player id."""

from __future__ import annotations

import random
from pathlib import Path

from fastapi.testclient import TestClient
from web_rig import make_app, new_game, new_player, play_out

from ticket_to_ride.agents.registry import parse_spec
from ticket_to_ride.eval.arena import run_round_robin
from ticket_to_ride.eval.store import Store


def _play(
    client: TestClient, player: str, games: int, *, opponent: str = "h3", rated: bool = True
) -> None:
    for i in range(games):
        game = new_game(client, player, "mini", opponent, rated=rated)
        assert play_out(client, game["id"], random.Random(i)).over is not None


def _standing(board: dict, player_id: str) -> dict:
    return next(s for s in board["standings"] if s["player"]["id"] == player_id)


def test_rated_human_games_sit_on_the_h3_scale(client: TestClient) -> None:
    empty = client.get("/api/leaderboard", params={"map": "mini"}).json()
    assert empty["standings"] == [] and empty["note"]

    ann = new_player(client, "Ann")
    _play(client, ann, 3)
    board = client.get("/api/leaderboard", params={"map": "mini"}).json()
    assert board["anchor"] == "h3" and board["games"] == 3 and board["blocks"] == 3
    h3 = next(s for s in board["standings"] if s["player"]["spec"] == "h3")
    assert h3["rating"]["elo"] == 0.0, "H3 is the zero of the scale"
    me = _standing(board, ann)
    assert me["player"] == {"id": ann, "kind": "human", "name": "Ann", "spec": None}
    # A random player against H3: three blocks, so an interval, and a long way below zero.
    assert me["rating"]["lo"] is not None and me["rating"]["hi"] is not None
    assert me["rating"]["lo"] <= me["rating"]["elo"] <= me["rating"]["hi"]
    assert me["rating"]["elo"] < 0

    # The profile shows the same rating, and the record behind it.
    profile = client.get(f"/api/players/{ann}").json()
    assert profile["ratings"] == [{"map": "mini", "rating": me["rating"]}]
    (record,) = profile["record"]
    assert record["opponent"]["spec"] == "h3" and record["games"] == 3
    assert record["wins"] + record["losses"] + record["draws"] == 3
    assert len(profile["games"]) == 3
    assert {g["result"] for g in profile["games"]} <= {"win", "loss", "draw"}
    assert client.get("/api/players/nobody").status_code == 404

    # The opponents list carries the same numbers.
    agents = {a["spec"]: a for a in client.get("/api/agents").json()}
    assert agents["h3"]["anchor"] and agents["h3"]["ratings"][0]["rating"]["elo"] == 0.0
    assert agents["h3"]["player"]["id"] == h3["player"]["id"]


def test_one_game_is_a_rating_without_an_interval(client: TestClient) -> None:
    """One block resampled is the same game every time: no interval, not a zero-width one."""
    ann = new_player(client)
    _play(client, ann, 1)
    me = _standing(client.get("/api/leaderboard", params={"map": "mini"}).json(), ann)
    assert me["rating"]["lo"] is None and me["rating"]["hi"] is None


def test_unrated_games_are_not_rated(client: TestClient) -> None:
    ann, bob = new_player(client, "Ann"), new_player(client, "Bob")
    _play(client, ann, 2)
    _play(client, bob, 2, rated=False)
    board = client.get("/api/leaderboard", params={"map": "mini"}).json()
    assert board["games"] == 2
    assert bob not in {s["player"]["id"] for s in board["standings"]}
    assert client.get(f"/api/players/{bob}").json()["ratings"] == []
    assert len(client.get(f"/api/players/{bob}").json()["games"]) == 2


def test_people_and_arena_bots_share_one_fit(tmp_path: Path) -> None:
    """The H2 a person plays is the H2 the arena measured: one identity, one rating."""
    results = tmp_path / "results.db"
    with Store(results) as store:
        specs = [parse_spec(text) for text in ("h2", "h3", "h4")]
        run_round_robin(store, "mini", 2, specs, blocks=20)
    with TestClient(make_app(tmp_path, results=results)) as client:
        arena_only = client.get("/api/leaderboard", params={"map": "mini"}).json()
        assert {s["player"]["spec"] for s in arena_only["standings"]} == {"h2", "h3", "h4"}
        assert arena_only["games"] == 20 * 2 * 3

        ann = new_player(client)
        _play(client, ann, 2, opponent="h2")
        board = client.get("/api/leaderboard", params={"map": "mini"}).json()
        assert board["games"] == arena_only["games"] + 2
        assert board["blocks"] == arena_only["blocks"] + 2, "each human game is its own block"
        specs = [s["player"]["spec"] for s in board["standings"]]
        assert sorted(filter(None, specs)) == ["h2", "h3", "h4"], "H2 appears once"
        assert _standing(board, ann)["rating"]["games"] == 2
        # Nothing else moved the arena's ratings much: the person played two games.
        before = {s["player"]["spec"]: s["rating"]["elo"] for s in arena_only["standings"]}
        after = {s["player"]["spec"]: s["rating"]["elo"] for s in board["standings"]}
        assert after["h3"] == before["h3"] == 0.0
        assert abs(after["h4"] - before["h4"]) < 5
        # Another map is another fit, with nothing in it yet.
        assert client.get("/api/leaderboard", params={"map": "usa"}).json()["standings"] == []
