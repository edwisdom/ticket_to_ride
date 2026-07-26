"""The feature spec and the reference encoder.

This encoder is the oracle Phase 2's Rust encoder is checked against, so what matters here
is that the layout is exactly what the spec says, that every feature actually moves when the
thing it describes changes, and that it stays importable with no torch and no numpy.
"""

from __future__ import annotations

import subprocess
import sys
from array import array
from pathlib import Path

import pytest
from rig import cards, force, to_main

from ticket_to_ride.data.board import BOARDS, MINI, USA, Board
from ticket_to_ride.engine import CLOSED, Game, RuleConfig
from ticket_to_ride.engine.graph import dsu_union
from ticket_to_ride.engine.rng import stream
from ticket_to_ride.engine.state import State
from ticket_to_ride.rl.encode.observation import encode, observation_size
from ticket_to_ride.rl.encode.spec import (
    FEATURE_SPEC,
    OBS_VERSION,
    OPPONENT_SLOTS,
    ObsSpec,
    obs_spec,
)

ALL_BOARDS = pytest.mark.parametrize("board", list(BOARDS.values()), ids=list(BOARDS))


def played(seed: int = 1, steps: int = 60, n_players: int = 3) -> tuple[Game, State]:
    game = Game(RuleConfig(n_players=n_players))
    state = game.new_initial_state(seed)
    rng = stream(seed, "policy")
    for _ in range(steps):
        if state.is_terminal():
            break
        state.step(state.sample_legal(rng))
    return game, state


# ---------------------------------------------------------------------------
# The spec table
# ---------------------------------------------------------------------------


def test_generated_rust_spec_is_not_stale(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tools/gen_obs_spec.py", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@ALL_BOARDS
def test_blocks_tile_the_vector_exactly(board: Board) -> None:
    """No gaps, no overlaps: every float belongs to exactly one field of one block."""
    spec = obs_spec(board)
    cursor = 0
    for block in spec.blocks:
        assert block.offset == cursor, f"{block.name} starts at {block.offset}, not {cursor}"
        assert sum(f.width for f in block.fields) == block.stride
        field_cursor = 0
        for field in block.fields:
            assert field.offset == field_cursor
            field_cursor += field.width
        cursor += block.size
    assert cursor == spec.size


@ALL_BOARDS
def test_field_offsets_resolve_for_every_entity(board: Board) -> None:
    spec = obs_spec(board)
    for block in spec.blocks:
        for entity in range(block.count):
            for field in block.fields:
                offset = block.at(entity, field.name)
                assert 0 <= offset < spec.size
                assert offset + field.width <= spec.size


def test_spec_is_cached_per_board() -> None:
    assert obs_spec(USA) is obs_spec(USA)
    assert obs_spec(MINI) is not obs_spec(USA)
    assert "usa" in repr(obs_spec(USA))


def test_unknown_field_names_are_rejected() -> None:
    with pytest.raises(KeyError, match="no field"):
        obs_spec(USA).block("clock").at(0, "nonexistent")


def test_block_names_are_unique_and_documented() -> None:
    names = [b.name for b in FEATURE_SPEC]
    assert len(set(names)) == len(names)
    for block in FEATURE_SPEC:
        assert len({f.name for f in block.fields}) == len(block.fields)
        for field in block.fields:
            assert field.doc, f"{block.name}.{field.name} has no documentation"


def test_opponent_slots_are_fixed_so_one_network_plays_any_table() -> None:
    """A 2P and a 5P observation are the same width; unused slots carry `present` = 0."""
    assert obs_spec(USA).block("opponents").count == OPPONENT_SLOTS == 4
    assert (
        observation_size(Game(RuleConfig(n_players=2)).new_initial_state(1)) == obs_spec(USA).size
    )
    assert (
        observation_size(Game(RuleConfig(n_players=5)).new_initial_state(1)) == obs_spec(USA).size
    )


def test_the_spec_is_smaller_on_a_smaller_board() -> None:
    assert obs_spec(MINI).size < obs_spec(USA).size
    assert obs_spec(USA).size == 3355
    assert OBS_VERSION >= 1


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------


@ALL_BOARDS
def test_encoding_fills_a_buffer_of_the_right_size(board: Board) -> None:
    game = Game(RuleConfig(map_name=board.name, n_players=2))
    state = to_main(game.new_initial_state(1))
    out = encode(state, 0)
    assert len(out) == obs_spec(board).size
    assert all(-5.0 <= v <= 5.0 for v in out), "features should stay in a sane range"


def test_a_reused_buffer_is_cleared() -> None:
    _, state = played()
    buffer = array("f", [7.0] * observation_size(state))
    encode(state, 0, buffer)
    fresh = encode(state, 0)
    assert list(buffer) == list(fresh), "stale values leaked through the reused buffer"


def test_a_wrong_sized_buffer_is_rejected() -> None:
    _, state = played()
    with pytest.raises(ValueError, match="buffer holds"):
        encode(state, 0, array("f", [0.0] * 3))


def test_encoding_is_deterministic() -> None:
    _, state = played()
    assert list(encode(state, 0)) == list(encode(state, 0))


def test_every_seat_sees_a_different_observation() -> None:
    _, state = played()
    views = [tuple(encode(state, p)) for p in range(3)]
    assert len(set(views)) == 3, "the observation must be perspective-normalized"


def test_opponents_are_ordered_by_distance_in_turn_order() -> None:
    """Slot 0 is whoever acts next. Seat-relative, because who moves before you decides games."""
    game = Game(RuleConfig(n_players=4))
    state = to_main(game.new_initial_state(1))
    force(state, trains=[45, 40, 30, 20])
    spec = obs_spec(game.board)
    block = spec.block("opponents")
    supply = game.board.raw.trains_per_player

    def trains_seen(observer: int) -> list[int]:
        view = encode(state, observer)
        return [round(view[block.at(i, "trains")] * supply) for i in range(3)]

    assert trains_seen(0) == [40, 30, 20]
    assert trains_seen(2) == [20, 45, 40]


def test_unused_opponent_slots_are_absent_and_zero() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("opponents")
    view = encode(state, 0)
    assert view[block.at(0, "present")] == 1.0
    for slot in range(1, OPPONENT_SLOTS):
        assert view[block.at(slot, "present")] == 0.0
        assert view[block.at(slot, "trains")] == 0.0


# ---------------------------------------------------------------------------
# Individual features move when the thing they describe moves
# ---------------------------------------------------------------------------


def segment_named(board: Board, a: str, b: str) -> int:
    ia, ib = sorted((board.city_index[a], board.city_index[b]))
    return next(s for s in range(board.n_segments) if (board.seg_a[s], board.seg_b[s]) == (ia, ib))


def test_segment_owner_one_hot_tracks_claims() -> None:
    game = Game(RuleConfig(n_players=3))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("segment_dynamic")
    segment = segment_named(game.board, "Atlanta", "Miami")

    view = encode(state, 0)
    assert view[block.at(segment, "owner") + 0] == 1.0, "free"

    state.seg_owner[segment] = 0
    assert encode(state, 0)[block.at(segment, "owner") + 1] == 1.0, "mine"

    state.seg_owner[segment] = 1
    view = encode(state, 0)
    assert view[block.at(segment, "owner") + 2] == 1.0, "the seat that acts next"
    assert encode(state, 2)[block.at(segment, "owner") + 3] == 1.0, "two seats away"


def test_closed_is_distinct_from_owned() -> None:
    """A blocked double and an enemy's track are very different to a blocking heuristic."""
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("segment_dynamic")
    segment = segment_named(game.board, "Boston", "Montreal")
    state.seg_owner[segment] = CLOSED
    view = encode(state, 0)
    assert view[block.at(segment, "closed")] == 1.0
    assert view[block.at(segment, "owner") + 0] == 1.0, "closed is nobody's, not an opponent's"


def test_twin_locked_marks_the_other_half_of_my_double() -> None:
    game = Game(RuleConfig(n_players=4))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("segment_dynamic")
    segment = segment_named(game.board, "Boston", "Montreal")
    twin = game.board.sibling[segment]
    state.seg_owner[segment] = 0
    view = encode(state, 0)
    assert view[block.at(twin, "twin_locked")] == 1.0
    assert encode(state, 1)[block.at(twin, "twin_locked")] == 0.0


def test_can_afford_now_and_cards_short_track_the_hand() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("segment_dynamic")
    segment = segment_named(game.board, "Atlanta", "Miami")  # 5, blue
    blue = game.board.color_names.index("blue")

    force(state, hands=[cards(game.board, blue=5), [0] * game.board.n_card_types], cur=0)
    view = encode(state, 0)
    assert view[block.at(segment, "can_afford_now")] == 1.0
    assert view[block.at(segment, "cards_short")] == 0.0

    force(state, hands=[cards(game.board, blue=2), [0] * game.board.n_card_types], cur=0)
    view = encode(state, 0)
    assert view[block.at(segment, "can_afford_now")] == 0.0
    assert view[block.at(segment, "cards_short") + 0] == 1.0, "at least one card short"
    assert view[block.at(segment, "cards_short") + 2] == 1.0, "three short"
    assert view[block.at(segment, "cards_short") + 3] == 0.0
    del blue


def test_required_colour_is_a_feature_not_an_embedding() -> None:
    """The colour-symmetry regularizer permutes colours; it has to be able to reach this."""
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("segment_static")
    view = encode(state, 0)
    blue = game.board.color_names.index("blue")
    colored = segment_named(game.board, "Atlanta", "Miami")
    gray = segment_named(game.board, "Nashville", "Atlanta")
    assert view[block.at(colored, "required_color") + blue] == 1.0
    assert view[block.at(gray, "required_color") + game.board.n_colors] == 1.0, "gray goes last"


def test_remaining_cost_falls_as_the_route_gets_built() -> None:
    """The most important engineered feature in the project."""
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    board = game.board
    block = obs_spec(board).block("tickets")
    ticket = min(
        range(board.n_tickets), key=lambda t: board.dist[board.ticket_a[t]][board.ticket_b[t]]
    )
    force(state, tickets=[1 << ticket, 0], cur=0)

    before = encode(state, 0)[block.at(ticket, "remaining_cost")]
    assert before > 0
    assert encode(state, 0)[block.at(ticket, "held")] == 1.0

    here, target = board.ticket_a[ticket], board.ticket_b[ticket]
    nb, segment = min(
        board.adjacency[here], key=lambda e: board.seg_len[e[1]] + board.dist[e[0]][target]
    )
    del nb
    state.seg_owner[segment] = 0
    dsu_union(state.dsu[0], board.seg_a[segment], board.seg_b[segment])
    assert encode(state, 0)[block.at(ticket, "remaining_cost")] < before


def test_a_completed_ticket_reads_connected_at_zero_cost() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    board = game.board
    block = obs_spec(board).block("tickets")
    ticket = min(
        range(board.n_tickets), key=lambda t: board.dist[board.ticket_a[t]][board.ticket_b[t]]
    )
    force(state, tickets=[1 << ticket, 0], cur=0)

    here, target = board.ticket_a[ticket], board.ticket_b[ticket]
    while here != target:
        nb, segment = min(
            board.adjacency[here], key=lambda e: board.seg_len[e[1]] + board.dist[e[0]][target]
        )
        state.seg_owner[segment] = 0
        dsu_union(state.dsu[0], board.seg_a[segment], board.seg_b[segment])
        here = nb

    view = encode(state, 0)
    assert view[block.at(ticket, "connected")] == 1.0
    assert view[block.at(ticket, "remaining_cost")] == 0.0
    assert view[block.at(ticket, "is_dead")] == 0.0


def test_a_cut_ticket_reads_dead() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    board = game.board
    block = obs_spec(board).block("tickets")
    ticket = next(t for t in range(board.n_tickets) if board.cities[board.ticket_a[t]] == "Boston")
    force(state, tickets=[1 << ticket, 0], cur=0)
    for _, segment in board.adjacency[board.ticket_a[ticket]]:
        state.seg_owner[segment] = 1

    view = encode(state, 0)
    assert view[block.at(ticket, "is_dead")] == 1.0
    assert view[block.at(ticket, "remaining_cost")] == 0.0, "a dead ticket carries no cost"


def test_fragility_rises_when_a_route_hangs_on_one_edge() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    board = game.board
    block = obs_spec(board).block("tickets")
    ticket = next(
        t
        for t in range(board.n_tickets)
        if {board.cities[board.ticket_a[t]], board.cities[board.ticket_b[t]]}
        == {"Denver", "El Paso"}
    )
    force(state, tickets=[1 << ticket, 0], cur=0)
    open_board = encode(state, 0)[block.at(ticket, "fragility")]

    # Cut every alternative out of Denver except the Santa Fe corridor.
    denver = board.city_index["Denver"]
    for nb, segment in board.adjacency[denver]:
        if board.cities[nb] != "Santa Fe":
            state.seg_owner[segment] = 1
    squeezed = encode(state, 0)[block.at(ticket, "fragility")]
    assert squeezed > open_board, "losing the last corridor should read as fragile"


def test_faceup_one_hot_marks_empty_slots() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("faceup")
    counts = list(game.board.deck_composition_counts)
    force(state, hands=[list(counts), [0] * len(counts)], faceup=[], deck=[], cur=0)
    view = encode(state, 0)
    for slot in range(5):
        assert view[block.at(slot, "card") + game.board.n_card_types] == 1.0


def test_unseen_counts_reach_the_observation() -> None:
    game, state = played()
    block = obs_spec(game.board).block("piles")
    view = encode(state, 0)
    unseen = state.unseen_counts(0)
    for c in range(game.board.n_card_types):
        expected = unseen[c] / game.board.cards_per_type(c)
        assert abs(view[block.at(0, "unseen") + c] - expected) < 1e-6


def test_certain_knowledge_of_an_opponent_shows_up() -> None:
    """Face-up takes are public, so what an opponent certainly holds is not a guess."""
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("opponents")
    before = encode(state, 0)

    # Seat 1 takes a face-up card; seat 0 now knows one of its cards for certain.
    while state.current_player() != 1 or state.phase != 1:
        state.step(state.legal_actions()[0])
    card = state.faceup[0]
    state.step(game.space.draw(0))
    after = encode(state, 0)
    slot = block.at(0, "certain") + card
    assert after[slot] > before[slot]


def test_clock_block_tracks_the_endgame() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("clock")
    assert encode(state, 0)[block.at(0, "final_triggered")] == 0.0

    state.final_left = 2
    view = encode(state, 0)
    assert view[block.at(0, "final_triggered")] == 1.0
    assert view[block.at(0, "final_countdown")] == 1.0
    assert view[block.at(0, "seats") + 0] == 1.0, "two seats"


def test_score_rank_and_gap_to_leader() -> None:
    game = Game(RuleConfig(n_players=3))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("clock")
    state.score[:] = [10, 50, 30]
    assert encode(state, 1)[block.at(0, "score_rank")] == 1.0
    assert encode(state, 1)[block.at(0, "gap_to_leader")] == 0.0
    assert encode(state, 0)[block.at(0, "score_rank")] == 0.0
    assert abs(encode(state, 0)[block.at(0, "gap_to_leader")] - 0.4) < 1e-6


def test_steiner_block_reports_slack_and_exactness() -> None:
    game = Game(RuleConfig(n_players=2))
    state = to_main(game.new_initial_state(1))
    block = obs_spec(game.board).block("steiner")
    view = encode(state, 0)
    assert view[block.at(0, "exact")] == 1.0
    assert view[block.at(0, "remaining_cost")] > 0.0
    # Full train supply minus the tree cost, so slack starts positive.
    assert view[block.at(0, "cost_minus_trains")] < 1.0


@ALL_BOARDS
def test_encoding_survives_a_whole_game(board: Board) -> None:
    game = Game(RuleConfig(map_name=board.name, n_players=2))
    state = game.new_initial_state(4)
    rng = stream(4, "policy")
    buffer = array("f", bytes(4 * observation_size(state)))
    steps = 0
    while not state.is_terminal() and steps < 40:
        encode(state, state.current_player(), buffer)
        state.step(state.sample_legal(rng))
        steps += 1
    assert steps > 10


def test_the_encoder_imports_without_numpy_or_torch(repo_root: Path) -> None:
    """It is the differential oracle, and the harness runs with no torch installed."""
    script = (
        "import sys\n"
        "for banned in ('torch', 'numpy'):\n"
        "    sys.modules[banned] = None\n"
        "import ticket_to_ride.rl.encode.observation as m\n"
        "assert 'numpy' not in {k for k, v in sys.modules.items() if v is not None}\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spec_describe_is_printable() -> None:
    rows = obs_spec(USA).describe()
    assert rows[0][0] == "segment_static"
    assert sum(size for _, _, _, size in rows) == obs_spec(USA).size
    assert isinstance(obs_spec(USA), ObsSpec)
