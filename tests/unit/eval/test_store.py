"""The results store: identity, accumulation, and the properties ACID is here for."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from ticket_to_ride.eval.store import SCHEMA_VERSION, AgentRow, MatchRow, Store


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "results.db") as s:
        yield s


def a_config(store: Store) -> int:
    return store.config_id("usa", 2, "rules", "data", 1)


def an_agent(spec: str = "h3", behaviour: str = "bbbb", params: str = "pppp") -> AgentRow:
    return AgentRow(
        spec=spec,
        family=spec.split("@", maxsplit=1)[0],
        params_hash=params,
        behaviour_hash=behaviour,
    )


def a_match(config_id: int) -> MatchRow:
    return MatchRow(config_id=config_id, seed_root=0, n_blocks=1, lineup="h3,h0", seconds=0.1)


def two_seat_rows(agent_a: int, agent_b: int) -> list[tuple]:
    """One game, agent_a first. `game_id` is the game *index*; the store assigns the real id."""
    base = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return [
        (0, 0, agent_a, 100, 1.0, 1, 1, *base),
        (0, 1, agent_b, 50, -1.0, 2, 0, *base),
    ]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_same_behaviour_is_the_same_agent(store: Store) -> None:
    """This is what makes ratings accumulate across invocations, days and commits."""
    first = store.agent_id(an_agent())
    second = store.agent_id(an_agent())
    assert first == second


def test_different_play_is_a_different_agent(store: Store) -> None:
    """Two builds that play differently are different players, even at the same commit."""
    assert store.agent_id(an_agent(behaviour="aaaa")) != store.agent_id(an_agent(behaviour="cccc"))


def test_a_commit_does_not_mint_a_new_agent(store: Store) -> None:
    """PLAN.md §11 suggests keying agents on `(checkpoint, config_hash, git_sha)`.

    With `git_sha` in the identity, a hundred commits that never touch the heuristics would
    leave a hundred "different" H3s, each holding a slice of the games and none with a
    usable rating -- shattering exactly the accumulation the scheme exists for. The sha is
    recorded on the *match*, which this asserts by writing two matches at different shas
    against one agent id.
    """
    config = a_config(store)
    agent = store.agent_id(an_agent())
    other = store.agent_id(an_agent(spec="h0", behaviour="zzzz"))
    for sha in ("aaaaaaa", "bbbbbbb"):
        match = a_match(config)
        match.git_sha = sha
        store.record_match(match, [(0, 0, 10, "00ff")], two_seat_rows(agent, other))
    assert store.counts()["agent"] == 2
    assert store.counts()["game"] == 2
    shas = {row["git_sha"] for row in store.db.execute("SELECT git_sha FROM match")}
    assert shas == {"aaaaaaa", "bbbbbbb"}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_a_failed_match_leaves_nothing_behind(store: Store) -> None:
    """The reason §11 asks for ACID: a crashed 100k-game arena must not corrupt prior
    results. A half-written match would be worse than a lost one, because it would be
    counted."""
    config = a_config(store)
    agent = store.agent_id(an_agent())
    other = store.agent_id(an_agent(spec="h0", behaviour="zzzz"))
    store.record_match(a_match(config), [(0, 0, 10, "00ff")], two_seat_rows(agent, other))
    before = store.counts()

    bad = two_seat_rows(agent, other)
    bad[1] = (0, 1, 999_999, 50, -1.0, 2, 0, *([0] * 17))  # no such agent
    with pytest.raises(sqlite3.IntegrityError):
        store.record_match(a_match(config), [(1, 0, 10, "0100")], bad)
    assert store.counts() == before, "a rolled-back match left rows behind"


def test_the_same_block_and_rotation_cannot_be_recorded_twice_in_a_match(store: Store) -> None:
    config = a_config(store)
    agent = store.agent_id(an_agent())
    other = store.agent_id(an_agent(spec="h0", behaviour="zzzz"))
    rows = [
        *two_seat_rows(agent, other),
        (1, 0, agent, 10, 1.0, 1, 1, *([0] * 17)),
        (1, 1, other, 5, -1.0, 2, 0, *([0] * 17)),
    ]
    with pytest.raises(sqlite3.IntegrityError):
        store.record_match(a_match(config), [(7, 0, 10, "aa"), (7, 0, 11, "bb")], rows)


def test_a_u64_state_hash_survives_the_round_trip(store: Store) -> None:
    """SQLite's INTEGER is *signed* 64-bit, so a state hash above 2^63 raises OverflowError
    on insert. Hex text stores the value the engine actually reports, rather than a
    two's-complement negative that matches nothing."""
    config = a_config(store)
    agent = store.agent_id(an_agent())
    other = store.agent_id(an_agent(spec="h0", behaviour="zzzz"))
    big = 0xFFFF_FFFF_FFFF_FFFF
    store.record_match(a_match(config), [(0, 0, 10, f"{big:016x}")], two_seat_rows(agent, other))
    stored = store.db.execute("SELECT final_hash FROM game").fetchone()["final_hash"]
    assert int(stored, 16) == big


def test_a_rating_run_is_appended_never_overwritten(store: Store) -> None:
    """A refit must not destroy 'the leaderboard as of last Tuesday'. The anchor each fit
    used is recorded with it, so a moved anchor shows up in the history rather than
    retroactively rewriting it."""
    config = a_config(store)
    agent = store.agent_id(an_agent())
    common = {
        "method": "bradley-terry-newton",
        "prior_sd": 400.0,
        "resamples": 100,
        "n_games": 10,
        "n_blocks": 5,
        "cycle_fraction": 0.0,
    }
    first = store.record_ratings(config, agent, [(agent, 0.0, 0.0, 0.0, 10, 0.5)], **common)
    second = store.record_ratings(config, agent, [(agent, 5.0, 1.0, 9.0, 20, 0.6)], **common)
    assert first != second
    assert store.counts()["rating_run"] == 2
    latest = store.latest_rating_run(config)
    assert latest is not None
    assert latest["rating_run_id"] == second
    assert store.ratings_of(first)[0]["elo"] == 0.0, "an earlier snapshot was rewritten"


# ---------------------------------------------------------------------------
# Schema guard
# ---------------------------------------------------------------------------


def test_a_database_from_another_schema_version_is_refused(tmp_path: Path) -> None:
    """Silently mixing versions would compute accumulated ratings over rows read under the
    wrong meaning -- which looks like a leaderboard, not like an error."""
    path = tmp_path / "old.db"
    with Store(path) as s:
        s.db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (SCHEMA_VERSION + 1,))
    with pytest.raises(RuntimeError, match="schema version"):
        Store(path)


def test_wal_is_on(tmp_path: Path) -> None:
    with Store(tmp_path / "r.db") as s:
        mode = s.db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
