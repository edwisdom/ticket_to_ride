"""Replays: bitwise, or loudly wrong. Reproducibility level L0.

Every rejection path here corresponds to a way a replay could silently become a *different*
game -- a different board, different rules, a different PRNG. Those are the failures worth
paying bytes for.
"""

from __future__ import annotations

import pytest

from ticket_to_ride.data.board import BOARDS
from ticket_to_ride.engine import Game, RuleConfig, final_scores
from ticket_to_ride.engine.replay import (
    MAGIC,
    Replay,
    ReplayError,
    decode,
    encode,
    record,
    replay,
)
from ticket_to_ride.engine.rng import stream

CONFIGURATIONS = [
    (name, n)
    for name, board in BOARDS.items()
    for n in range(board.raw.min_players, board.raw.max_players + 1)
]


def play(map_name: str = "usa", n_players: int = 2, seed: int = 1) -> tuple[Game, Replay]:
    game = Game(RuleConfig(map_name=map_name, n_players=n_players))
    state = game.new_initial_state(seed)
    rng = stream(seed, "policy")
    while not state.is_terminal():
        state.step(state.sample_legal(rng))
    return game, record(state)


# ---------------------------------------------------------------------------
# The core promise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("map_name", "n_players"), CONFIGURATIONS, ids=[f"{m}-{n}p" for m, n in CONFIGURATIONS]
)
def test_a_seed_replays_bitwise(map_name: str, n_players: int) -> None:
    for seed in range(8):
        _, rec = play(map_name, n_players, seed)
        state = replay(decode(encode(rec)))
        assert state.state_hash() == rec.final_hash
        assert tuple(final_scores(state)) == rec.final_scores
        assert tuple(state.history) == rec.actions


def test_round_trip_through_the_wire_format_is_exact() -> None:
    _, rec = play()
    assert decode(encode(rec)) == rec


def test_a_replay_is_around_half_a_kilobyte() -> None:
    """The budget the plan is sized against: ~150 MB per million games with zstd."""
    _, rec = play()
    blob = encode(rec)
    assert 200 < len(blob) < 700, len(blob)
    # Header, name, body, scores, actions -- and nothing else.
    assert len(blob) == 6 + len("usa") + 50 + 2 * rec.n_players + 2 * rec.n_actions


def test_replaying_leaves_the_history_intact() -> None:
    _, rec = play()
    state = replay(rec)
    assert list(state.history) == list(rec.actions)


# ---------------------------------------------------------------------------
# Everything that should fail loudly
# ---------------------------------------------------------------------------


def test_a_board_edit_invalidates_old_replays() -> None:
    _, rec = play()
    tampered = msgspec_replace(rec, data_hash="0" * 32)
    with pytest.raises(ReplayError, match="has changed since this replay"):
        replay(tampered)


def test_a_rule_change_invalidates_old_replays() -> None:
    _, rec = play()
    tampered = msgspec_replace(rec, rules_hash="0" * 32)
    with pytest.raises(ReplayError, match="rules have changed"):
        replay(tampered)


def test_a_contract_bump_invalidates_old_replays() -> None:
    """The PRNG or the draw procedure changed; the actions no longer mean the same game."""
    _, rec = play()
    tampered = msgspec_replace(rec, contract_version=rec.contract_version + 1)
    with pytest.raises(ReplayError, match="contract version"):
        replay(tampered)


def test_a_diverged_replay_is_caught_by_the_final_hash() -> None:
    _, rec = play()
    tampered = msgspec_replace(rec, final_hash=rec.final_hash ^ 1)
    with pytest.raises(ReplayError, match="diverged"):
        replay(tampered)


def test_a_diverged_replay_is_caught_by_the_scores() -> None:
    """A second, independent check: the state hash can match while scoring changed.

    `state_hash()` covers the position, not the interpretation of it -- a bug in ticket
    settlement or the longest-trail search would leave every hash intact.
    """
    _, rec = play()
    wrong = (rec.final_scores[0] + 1, *rec.final_scores[1:])
    with pytest.raises(ReplayError, match="scores"):
        replay(msgspec_replace(rec, final_scores=wrong))


def test_a_truncated_action_list_ends_before_the_game_does() -> None:
    _, rec = play()
    with pytest.raises(ReplayError, match="ended before the game"):
        replay(msgspec_replace(rec, actions=rec.actions[:-4]))


def test_a_replay_longer_than_the_game_is_rejected() -> None:
    _, rec = play()
    padded = (*rec.actions, rec.actions[-1])
    with pytest.raises(ReplayError, match="longer than the game"):
        replay(msgspec_replace(rec, actions=padded))


def test_an_unfinished_game_cannot_be_recorded() -> None:
    game = Game(RuleConfig(n_players=2))
    with pytest.raises(ReplayError, match="only a finished game"):
        record(game.new_initial_state(1))


def test_a_game_played_without_history_cannot_be_recorded() -> None:
    game = Game(RuleConfig(n_players=2, track_history=False))
    state = game.new_initial_state(1)
    rng = stream(1, "policy")
    while not state.is_terminal():
        state.step(state.sample_legal(rng))
    with pytest.raises(ReplayError, match="track_history off"):
        record(state)


# ---------------------------------------------------------------------------
# Wire-format robustness
# ---------------------------------------------------------------------------


def test_a_foreign_file_is_rejected() -> None:
    with pytest.raises(ReplayError, match="not a replay"):
        decode(b"PNG\x00\x02\x03" + b"\x00" * 80)


def test_a_truncated_file_is_rejected() -> None:
    with pytest.raises(ReplayError, match="truncated"):
        decode(MAGIC)


def test_trailing_bytes_are_rejected() -> None:
    _, rec = play()
    with pytest.raises(ReplayError, match="trailing bytes"):
        decode(encode(rec) + b"\x00")


def test_replay_accepts_an_explicit_matching_config() -> None:
    """Non-default rules have to be supplied, and are checked against `rules_hash`."""
    cfg = RuleConfig(n_players=2, turn_cap=500)
    game = Game(cfg)
    state = game.new_initial_state(3)
    rng = stream(3, "policy")
    while not state.is_terminal():
        state.step(state.sample_legal(rng))
    rec = record(state)

    with pytest.raises(ReplayError, match="rules have changed"):
        replay(rec)  # default config: different turn_cap, different rules_hash
    assert replay(rec, cfg).state_hash() == rec.final_hash


def msgspec_replace(rec: Replay, **changes: object) -> Replay:
    """`dataclasses.replace` for a frozen msgspec Struct."""
    fields = {name: getattr(rec, name) for name in Replay.__struct_fields__}
    fields.update(changes)
    return Replay(**fields)
