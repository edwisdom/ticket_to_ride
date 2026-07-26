"""The query surface: accessors, the action-space helpers, and configuration errors.

Everything here is what Phase 2's PyO3 shim has to expose and Phase 5's search has to call,
so it is worth pinning now rather than discovering it is subtly wrong from inside Rust.
"""

from __future__ import annotations

import pytest
from rig import force, to_main

from ticket_to_ride.data.board import MINI, USA
from ticket_to_ride.engine import (
    ACTION_SPACE_VERSION,
    CONTRACT_VERSION,
    PHASE_MAIN,
    Game,
    IllegalAction,
    RuleConfig,
    action_space,
)
from ticket_to_ride.engine.graph import dsu_union
from ticket_to_ride.engine.rng import stream
from ticket_to_ride.engine.state import EMPTY_SLOT


def test_versions_are_present_and_separate() -> None:
    """Three independent version numbers; conflating them is how stale checkpoints load."""
    assert CONTRACT_VERSION >= 1
    assert ACTION_SPACE_VERSION >= 1


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


def test_num_distinct_actions_and_repr() -> None:
    g = Game(RuleConfig(n_players=2))
    assert g.num_distinct_actions == 915
    assert "usa" in repr(g) and "915" in repr(g)
    assert "usa" in repr(g.new_initial_state(1))


def test_default_config_is_a_two_player_usa_game() -> None:
    g = Game()
    assert (g.board.name, g.n_players) == ("usa", 2)


def test_seat_count_is_validated_against_the_map() -> None:
    with pytest.raises(ValueError, match="supports 2-5 players"):
        Game(RuleConfig(n_players=6))
    with pytest.raises(ValueError, match="supports 2-4 players"):
        Game(RuleConfig(map_name="mini", n_players=5))


def test_unimplemented_variants_say_so() -> None:
    with pytest.raises(NotImplementedError, match="wild_policy"):
        RuleConfig(wild_policy="explicit")
    with pytest.raises(NotImplementedError, match="chance_mode"):
        RuleConfig(chance_mode="explicit")
    with pytest.raises(ValueError, match="initial_hand"):
        RuleConfig(initial_hand=-1)


def test_rules_hash_moves_with_the_rules_but_not_with_bookkeeping() -> None:
    base = RuleConfig()
    assert base.rules_hash == RuleConfig().rules_hash
    assert base.rules_hash != RuleConfig(n_players=3).rules_hash
    assert base.rules_hash != RuleConfig(turn_cap=999).rules_hash
    assert base.rules_hash != RuleConfig(map_name="mini").rules_hash
    assert base.rules_hash == RuleConfig(track_history=False).rules_hash, (
        "history tracking does not change the game"
    )


def test_doubles_lock_by_seat_count() -> None:
    assert RuleConfig(n_players=2).doubles_locked_for_everyone
    assert RuleConfig(n_players=3).doubles_locked_for_everyone
    assert not RuleConfig(n_players=4).doubles_locked_for_everyone
    assert not RuleConfig(n_players=5).doubles_locked_for_everyone


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


def test_action_space_layout_matches_the_documented_ranges() -> None:
    space = action_space(USA)
    assert (space.claim_end, space.draw_base) == (900, 900)
    assert space.draw_tickets == 906
    assert [space.keep(m) for m in range(1, 8)] == list(range(907, 914))
    assert space.pass_action == 914
    assert space.n == 915


def test_action_space_is_cached_per_board() -> None:
    assert action_space(USA) is action_space(USA)
    assert action_space(MINI) is not action_space(USA)


def test_claim_ids_are_bijective_with_segment_and_pay() -> None:
    space = action_space(USA)
    seen = set()
    for segment in range(USA.n_segments):
        for pay in range(USA.n_card_types):
            action = space.claim(segment, pay)
            assert space.decode_claim(action) == (segment, pay)
            assert action not in seen
            seen.add(action)
    assert len(seen) == space.claim_end


def test_keep_masks_are_one_based() -> None:
    """Seven keep actions, not eight: keeping nothing is never legal."""
    space = action_space(USA)
    for bad in (0, 8, -1):
        with pytest.raises(ValueError, match=r"keep mask must be in 1\.\.7"):
            space.keep(bad)
    assert space.keep(1) - space.keep_base == 1


def test_decoders_round_trip() -> None:
    space = action_space(USA)
    assert space.decode_draw(space.draw(3)) == 3
    assert space.decode_keep(space.keep(5)) == 5
    assert space.is_claim(0)
    assert not space.is_claim(space.draw_base)


def test_action_names_cover_every_range() -> None:
    space = action_space(USA)
    assert space.to_string(space.claim(0, 0)).startswith("CLAIM")
    assert space.to_string(space.draw(2)) == "DRAW faceup[2]"
    assert space.to_string(space.draw(5)) == "DRAW blind"
    assert space.to_string(space.draw_tickets) == "DRAW_TICKETS"
    assert space.to_string(space.keep(0b101)) == "KEEP offer[0, 2]"
    assert space.to_string(space.pass_action) == "PASS"
    with pytest.raises(ValueError, match="outside"):
        space.to_string(space.n)


def test_claim_names_show_the_route_and_the_payment() -> None:
    space = action_space(USA)
    blue = USA.color_names.index("blue")
    segment = next(s for s in range(USA.n_segments) if USA.seg_color[s] == blue)
    name = space.to_string(space.claim(segment, USA.locomotive))
    assert "blue" in name and "pay loco" in name


def test_mini_action_space_is_smaller_and_still_consistent() -> None:
    space = action_space(MINI)
    assert space.n == MINI.n_segments * MINI.n_card_types + 15
    assert space.k == 7, "six colours plus the locomotive"


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------


def test_hand_accessors_agree_with_the_flat_array() -> None:
    g = Game(RuleConfig(n_players=3))
    s = to_main(g.new_initial_state(2))
    for p in range(3):
        assert list(s.hand_of(p)) == list(s.hand[p * s.game.board.n_card_types :][:9])
        assert s.hand_size(p) == sum(s.hand_of(p))
        assert sum(s.certain_of(p)) + s.unknown[p] == s.hand_size(p)


def test_ticket_queries() -> None:
    g = Game(RuleConfig(n_players=2))
    s = to_main(g.new_initial_state(2))
    held = s.tickets_of(0)
    assert len(held) == int(s.tickets[0]).bit_count()
    assert all(0 <= t < g.board.n_tickets for t in held)
    assert s.tickets_remaining() == s.tdeck_len
    # Nothing is claimed yet, so no ticket can be complete.
    assert not any(s.ticket_complete(0, t) for t in held)


def test_deck_counts_sum_to_the_undrawn_deck() -> None:
    g = Game(RuleConfig(n_players=2))
    s = to_main(g.new_initial_state(2))
    counts = s.deck_counts()
    assert sum(counts) == len(s.deck) - s.deck_pos
    for card_type, count in enumerate(counts):
        assert s.deck.count(card_type, s.deck_pos) == count


def test_unseen_counts_are_what_a_determinization_would_deal() -> None:
    """Everything the observer cannot account for: the deck plus opponents' blind draws."""
    g = Game(RuleConfig(n_players=3))
    s = to_main(g.new_initial_state(2))
    rng = stream(2, "policy")
    for _ in range(40):
        if s.is_terminal():
            break
        s.step(s.sample_legal(rng))

    for observer in range(3):
        unseen = s.unseen_counts(observer)
        assert all(c >= 0 for c in unseen), unseen
        hidden = sum(s.hand_size(p) - sum(s.certain_of(p)) for p in range(3) if p != observer)
        assert sum(unseen) == (len(s.deck) - s.deck_pos) + hidden


def test_history_is_recorded_when_tracking_is_on() -> None:
    g = Game(RuleConfig(n_players=2))
    s = to_main(g.new_initial_state(2))
    rng = stream(2, "policy")
    before = len(s.history_actions())
    s.step(s.sample_legal(rng))
    assert len(s.history_actions()) == before + 1


def test_history_is_skipped_when_tracking_is_off() -> None:
    """Copying the action list dominates a clone, so search and benchmarks turn it off."""
    g = Game(RuleConfig(n_players=2, track_history=False))
    s = to_main(g.new_initial_state(2))
    s.step(s.sample_legal(stream(2, "policy")))
    assert s.history_actions() == []
    assert s.clone().history_actions() == []


def test_stepping_a_finished_game_is_rejected() -> None:
    g = Game(RuleConfig(n_players=2))
    s = g.new_initial_state(2)
    rng = stream(2, "policy")
    while not s.is_terminal():
        s.step(s.sample_legal(rng))
    with pytest.raises(IllegalAction, match="game is over"):
        s.step(0)
    assert s.legal_actions() == []


def test_current_player_tracks_the_turn() -> None:
    g = Game(RuleConfig(n_players=3))
    s = to_main(g.new_initial_state(2))
    seats = []
    for _ in range(6):
        seats.append(s.current_player())
        while s.phase != PHASE_MAIN or s.current_player() == seats[-1]:
            if s.is_terminal():
                break
            s.step(s.sample_legal(stream(len(seats), "policy")))
    assert seats[:3] == [0, 1, 2]


def test_ticket_completion_follows_the_union_find() -> None:
    """No USA ticket connects two adjacent cities, so completion needs a whole path."""
    g = Game(RuleConfig(n_players=2))
    s = to_main(g.new_initial_state(2))
    board = g.board
    ticket = min(
        range(board.n_tickets), key=lambda t: board.dist[board.ticket_a[t]][board.ticket_b[t]]
    )
    force(s, tickets=[1 << ticket, 0], cur=0)
    assert not s.ticket_complete(0, ticket)

    # Walk a shortest path, claiming as we go.
    here = board.ticket_a[ticket]
    target = board.ticket_b[ticket]
    while here != target:
        nb, segment = min(
            board.adjacency[here],
            key=lambda edge: board.seg_len[edge[1]] + board.dist[edge[0]][target],
        )
        s.seg_owner[segment] = 0
        dsu_union(s.dsu[0], board.seg_a[segment], board.seg_b[segment])
        here = nb
    assert s.ticket_complete(0, ticket)


def test_faceup_reports_empty_slots_as_minus_one() -> None:
    g = Game(RuleConfig(n_players=2))
    s = to_main(g.new_initial_state(2))
    counts = list(g.board.deck_composition_counts)
    force(s, hands=[list(counts), [0] * len(counts)], faceup=[], deck=[], cur=0)
    assert s.faceup == [EMPTY_SLOT] * 5
