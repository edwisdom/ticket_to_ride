"""The play store: versioned, WAL, append-only decisions, bots as identities."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ticket_to_ride.agents.registry import parse_spec
from ticket_to_ride.ui.web.store import ActionRow, GameRow, PlayStore, bot_id, now


def _game(store: PlayStore, players: list[str]) -> GameRow:
    row = GameRow(
        game_id="g1",
        map_name="mini",
        n_players=2,
        mode="play",
        rated=True,
        assist="parity-1",
        status="active",
        seed=2**63 - 1,
        contract_version=1,
        obs_version=2,
        data_hash="d",
        rules_hash="r",
        git_sha=None,
        client_build=None,
        created_utc=now(),
    )
    store.add_game(row, players)
    return row


def test_a_bot_is_one_player_per_identity(tmp_path: Path) -> None:
    with PlayStore(tmp_path / "play.db") as store:
        h3 = parse_spec("h3").identity()
        first = store.add_bot(h3, "H3")
        assert store.add_bot(h3, "H3") == first
        assert first.player_id == bot_id(h3)
        assert first.identity == h3
        assert store.add_bot(parse_spec("h4").identity(), "H4").player_id != first.player_id
        # Two names for one way of playing are one player.
        assert bot_id(parse_spec("random").identity()) == bot_id(parse_spec("h0").identity())


def test_a_decision_is_recorded_once_and_the_largest_seed_survives(tmp_path: Path) -> None:
    with PlayStore(tmp_path / "play.db") as store:
        person = store.add_human("Ann")
        bot = store.add_bot(parse_spec("h3").identity(), "H3")
        _game(store, [person.player_id, bot.player_id])
        stored = store.game("g1")
        assert stored is not None and stored.seed == 2**63 - 1
        store.append("g1", ActionRow(0, 0, 219, now(), 5))
        with pytest.raises(sqlite3.IntegrityError):
            store.append("g1", ActionRow(0, 0, 220, now(), 5))
        assert [a.action for a in store.actions("g1")] == [219]
        assert [g.game_id for g in store.games_of(person.player_id)] == ["g1"]


def test_wal_is_on_and_another_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "play.db"
    with PlayStore(path) as store:
        assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        store.db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(RuntimeError, match="schema version 99"):
        PlayStore(path)
