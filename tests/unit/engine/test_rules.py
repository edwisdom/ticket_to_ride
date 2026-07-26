"""The rules correctness checklist, one named test per item (PLAN.md §5.2).

Grouped as setup / drawing / claiming / tickets / end-and-scoring / degenerate. Most of
these positions occur once in thousands of random games, so they are rigged directly with
`tests/rig.py`, which keeps card conservation intact so `validate()` still applies.
"""

from __future__ import annotations

import pytest
from rig import cards, filler, force, spread_hands, to_main

from ticket_to_ride.data.board import MINI, NO_SIBLING, USA
from ticket_to_ride.engine import (
    CLOSED,
    FREE,
    PHASE_DRAW_SECOND,
    PHASE_INITIAL_TICKETS,
    PHASE_MAIN,
    PHASE_TERMINAL,
    PHASE_TICKET_KEEP,
    Game,
    IllegalAction,
    RuleConfig,
    final_scores,
    longest_trails,
    returns,
    score_breakdown,
    winners,
)
from ticket_to_ride.engine.graph import dsu_union
from ticket_to_ride.engine.rng import stream
from ticket_to_ride.engine.state import EMPTY_SLOT, NO_TICKET

LOCO = USA.locomotive


def game(n_players: int = 2, map_name: str = "usa", **kwargs: object) -> Game:
    return Game(RuleConfig(map_name=map_name, n_players=n_players, **kwargs))  # ty: ignore


def claim_action(g: Game, segment: int, pay: int) -> int:
    return g.space.claim(segment, pay)


def draw_action(g: Game, slot: int) -> int:
    return g.space.draw(slot)


def segment_named(board, a: str, b: str, *, which: int = 0) -> int:  # noqa: ANN001
    """The `which`-th segment between two cities, by name."""
    ia, ib = sorted((board.city_index[a], board.city_index[b]))
    found = [s for s in range(board.n_segments) if (board.seg_a[s], board.seg_b[s]) == (ia, ib)]
    return found[which]


# ===========================================================================
# Setup
# ===========================================================================


def test_hands_are_dealt_before_the_display() -> None:
    """Seat p holds exactly the deck slice [p*4, p*4+4); the display comes after all of them.

    Flipping the display first would deal every seat different cards, which is a difference
    no test downstream of setup could attribute to its real cause.
    """
    g = game(3)
    s = g.new_initial_state(7)
    k = g.board.n_card_types
    hand_size = g.cfg.initial_hand
    for p in range(3):
        expected = sorted(s.deck[p * hand_size : (p + 1) * hand_size])
        actual = sorted(c for c in range(k) for _ in range(s.hand[p * k + c]))
        assert actual == expected, f"seat {p} did not get its block off the top"
    assert s.deck_pos >= 3 * hand_size + 5


def test_display_holds_five_after_setup() -> None:
    s = game().new_initial_state(3)
    assert all(c != EMPTY_SLOT for c in s.faceup)


def test_initial_keep_has_exactly_four_legal_subsets() -> None:
    """3 dealt, keep >= 2: masks 011, 101, 110, 111 and nothing else."""
    g = game()
    s = g.new_initial_state(11)
    assert s.phase == PHASE_INITIAL_TICKETS
    masks = sorted(a - g.space.keep_base for a in s.legal_actions())
    assert masks == [0b011, 0b101, 0b110, 0b111]


def test_flush_check_runs_after_setup() -> None:
    """A setup display with 3+ locomotives must be flushed before anyone acts."""
    for seed in range(3000):
        s = game().new_initial_state(seed)
        assert s.faceup.count(LOCO) < 3, f"seed {seed} left 3 locomotives face-up at setup"


def test_initial_tickets_are_resolved_in_seat_order() -> None:
    g = game(4)
    s = g.new_initial_state(5)
    seen = []
    while s.phase == PHASE_INITIAL_TICKETS:
        seen.append(s.cur)
        s.step(s.legal_actions()[0])
    assert seen == [0, 1, 2, 3]
    assert s.phase == PHASE_MAIN
    assert (s.cur, s.turn) == (0, 0)


def test_returned_initial_tickets_go_to_the_bottom_in_seat_then_offer_order() -> None:
    g = game(2)
    s = g.new_initial_state(5)
    bottom_before = s.tdeck_len
    returned: list[int] = []
    while s.phase == PHASE_INITIAL_TICKETS:
        offer = s.offer[: s.offer_len]
        mask = 0b011  # keep the first two, return the third
        returned.append(offer[2])
        s.step(g.space.keep_base + mask)
    # The two returns sit at the bottom, seat 0's before seat 1's.
    tail = [s.tdeck[(s.tdeck_head + s.tdeck_len - 2 + i) % len(s.tdeck)] for i in range(2)]
    assert tail == returned
    # Two seats, three drawn and one returned each: a net two tickets leave the deck per seat.
    assert s.tdeck_len == g.board.n_tickets - 2 * 2
    assert bottom_before > s.tdeck_len


def test_every_seat_keeps_at_least_two_opening_tickets() -> None:
    g = game(3)
    s = to_main(g.new_initial_state(9))
    for p in range(3):
        assert int(s.tickets[p]).bit_count() >= 2


# ===========================================================================
# Drawing
# ===========================================================================


def test_face_up_locomotive_as_first_card_ends_the_draw() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[LOCO, 0, 1, 2, 3], deck=filler(g.board, 20), phase=PHASE_MAIN, cur=0)
    before = s.turn
    s.step(draw_action(g, 0))
    assert s.turn == before + 1, "taking a face-up locomotive is the whole turn"
    assert s.phase == PHASE_MAIN
    assert s.cur == 1
    s.validate()


def test_face_up_locomotive_is_never_legal_as_the_second_card() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[0, LOCO, 1, 2, 3], deck=filler(g.board, 20), phase=PHASE_MAIN, cur=0)
    s.step(draw_action(g, 0))
    assert s.phase == PHASE_DRAW_SECOND
    assert draw_action(g, 1) not in s.legal_actions()
    with pytest.raises(IllegalAction, match="locomotive cannot be taken as the second"):
        s.step(draw_action(g, 1))


def test_blind_locomotive_does_not_end_the_turn() -> None:
    """The rulebook is explicit: a locomotive off the top still counts as one card."""
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[0, 1, 2, 3, 4], deck=[LOCO, *filler(g.board, 20)], phase=PHASE_MAIN, cur=0)
    s.step(draw_action(g, 5))
    assert s.phase == PHASE_DRAW_SECOND
    assert s.hand[0 * g.board.n_card_types + LOCO] >= 1
    s.validate()


def test_taken_slot_is_refilled_before_the_second_draw() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[0, 1, 2, 3, 4], deck=[5, 6, 7, *filler(g.board, 20)], phase=PHASE_MAIN, cur=0)
    s.step(draw_action(g, 0))
    assert s.faceup[0] == 5, "the taken slot must be refilled from the deck immediately"
    assert s.phase == PHASE_DRAW_SECOND
    s.validate()


def test_the_refill_may_itself_be_a_locomotive_and_is_then_untakeable() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[0, 1, 2, 3, 4], deck=[LOCO, *filler(g.board, 20)], phase=PHASE_MAIN, cur=0)
    s.step(draw_action(g, 0))
    assert s.faceup[0] == LOCO
    assert draw_action(g, 0) not in s.legal_actions()
    s.validate()


def test_three_face_up_locomotives_flush_all_five() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    # Take slot 4; the refill makes three locomotives and must trigger the flush.
    force(s, faceup=[LOCO, LOCO, 0, 1, 2], deck=[LOCO, 3, 4, 5, 6, 7, *filler(g.board, 20)], cur=0)
    before = s.discard_total
    s.step(draw_action(g, 4))
    assert s.faceup.count(LOCO) < 3
    assert s.discard_total == before + 5, "all five face-up cards go to the discard"
    s.validate()


def test_flush_cascades() -> None:
    """The replacement five can themselves hold three locomotives."""
    g = game()
    s = to_main(g.new_initial_state(1))
    force(
        s,
        faceup=[LOCO, LOCO, 0, 1, 2],
        # Slot 4's refill is a locomotive (flush 1); the next five hold three more
        # (flush 2); the five after that are clean.
        deck=[LOCO, LOCO, LOCO, LOCO, 3, 4, 5, 6, 7, 0, 1],
        cur=0,
    )
    before = s.discard_total
    s.step(draw_action(g, 4))
    assert s.faceup.count(LOCO) < 3
    assert s.discard_total == before + 10, "two flushes should have discarded ten cards"
    s.validate()


def test_flush_is_blocked_when_too_few_non_locomotives_remain() -> None:
    """The hang guard. Without it an all-locomotive pool loops forever."""
    g = game()
    s = to_main(g.new_initial_state(1))
    counts = list(g.board.deck_composition_counts)
    mine = list(counts)
    mine[LOCO] = 0  # every locomotive is face-up or in the deck; every colour is in hand
    force(s, hands=[mine, [0] * len(counts)], faceup=[LOCO] * 5, deck=[LOCO] * 9, cur=0)
    assert s._nonloco_available() == 0
    before = list(s.faceup)
    s._flush_check()
    assert s.faceup == before, "the flush must not fire when it cannot produce a legal display"
    s.validate()


def test_flush_cascade_cap_bails_out_deterministically() -> None:
    """Second lock: even with the guard satisfied, the cascade is bounded."""
    g = game(2, flush_cascade_cap=2)
    s = to_main(g.new_initial_state(1))
    # Locomotives keep arriving, and the guard stays open because non-locomotives exist.
    force(s, faceup=[LOCO] * 5, deck=[LOCO] * 9 + [0, 1, 2], cur=0)
    before = s.discard_total
    s._flush_check()
    assert s.faceup.count(LOCO) >= 3, "the cap should have stopped the cascade mid-flight"
    assert s.discard_total == before + 2 * 5, "exactly two flushes ran"


def test_deck_exhausted_mid_refill_reshuffles_the_discard() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[0, 1, 2, 3, 4], deck=[5], cur=0)
    assert s.discard_total > 0
    s.step(draw_action(g, 0))  # consumes the last deck card refilling slot 0
    s.step(draw_action(g, 5))  # blind draw must now come from a reshuffle
    assert s.deck_pos <= len(s.deck)
    s.validate()


def test_reshuffle_is_lazy_not_eager() -> None:
    """`deck empty, discards available` and `both empty` must stay distinguishable."""
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[0, 1, 2, 3, 4], deck=[], cur=0)
    assert s.deck_pos == len(s.deck), "the cursor is spent"
    assert s.discard_total > 0
    assert len(s.deck) == 0, "no eager reshuffle happened"
    assert draw_action(g, 5) in s.legal_actions(), "a blind draw is still legal"


def test_deck_and_discard_empty_leaves_a_short_display() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    # Give both seats everything, leaving nothing to refill with.
    counts = list(g.board.deck_composition_counts)
    mine = list(counts)
    mine[0] -= 5  # the five face-up cards are the only ones not in a hand
    force(s, hands=[mine, [0] * len(counts)], faceup=[0] * 5, deck=[], cur=0)
    assert s.discard_total == 0
    s.step(draw_action(g, 0))
    assert s.faceup.count(EMPTY_SLOT) == 1, "no card exists to refill the taken slot"
    assert draw_action(g, 0) not in s.legal_actions()
    s.validate()


def test_drawing_is_entirely_illegal_when_every_pool_is_empty() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    counts = list(g.board.deck_composition_counts)
    force(s, hands=[counts, [0] * len(counts)], faceup=[], deck=[], cur=0)
    legal = s.legal_actions()
    assert all(a < g.space.draw_base or a >= g.space.draw_tickets for a in legal), (
        "no draw action may be legal with no cards anywhere"
    )
    with pytest.raises(IllegalAction, match="no cards left"):
        s.step(draw_action(g, 5))


def test_turn_ends_after_one_card_when_no_second_is_obtainable() -> None:
    """Never construct a node with zero legal actions."""
    g = game()
    s = to_main(g.new_initial_state(1))
    counts = list(g.board.deck_composition_counts)
    mine = list(counts)
    mine[0] -= 1
    force(s, hands=[mine, [0] * len(counts)], faceup=[0], deck=[], cur=0)
    before = s.turn
    s.step(draw_action(g, 0))
    assert s.turn == before + 1, "the turn ends rather than stalling in DRAW_SECOND"
    assert s.phase == PHASE_MAIN
    s.validate()


def test_a_short_display_refills_once_a_claim_replenishes_the_discard() -> None:
    """Otherwise an emptied display stays empty forever, since only a take refills it."""
    g = game()
    s = to_main(g.new_initial_state(1))
    counts = list(g.board.deck_composition_counts)
    force(s, hands=[list(counts), [0] * len(counts)], faceup=[], deck=[], cur=0)
    assert all(c == EMPTY_SLOT for c in s.faceup)
    segment = segment_named(g.board, "Nashville", "Atlanta")  # length 1, gray
    s.step(claim_action(g, segment, 0))
    assert sum(1 for c in s.faceup if c != EMPTY_SLOT) > 0, "the display should refill"
    s.validate()


# ===========================================================================
# Claiming
# ===========================================================================


def test_claim_pays_exactly_the_route_length_and_scores_the_table() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Atlanta", "Miami")  # 5, blue
    force(s, hands=[cards(g.board, blue=5), [0] * g.board.n_card_types], cur=0)
    s.step(claim_action(g, segment, g.board.color_names.index("blue")))
    assert s.seg_owner[segment] == 0
    assert s.trains[0] == g.board.raw.trains_per_player - 5
    assert s.score[0] == 10, "a length-5 route scores 10"
    assert sum(s.hand[: g.board.n_card_types]) == 0
    s.validate()


@pytest.mark.parametrize(("length", "points"), [(1, 1), (2, 2), (3, 4), (4, 7), (5, 10), (6, 15)])
def test_route_scoring_table(length: int, points: int) -> None:
    assert USA.route_points[length] == points


def test_gray_route_accepts_any_single_color() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Las Vegas", "Los Angeles")  # 2, gray
    force(s, hands=[cards(g.board, white=2), [0] * g.board.n_card_types], cur=0)
    white = g.board.color_names.index("white")
    assert claim_action(g, segment, white) in s.legal_actions()
    s.step(claim_action(g, segment, white))
    assert s.seg_owner[segment] == 0
    s.validate()


def test_gray_route_accepts_all_locomotives() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Las Vegas", "Los Angeles")
    force(s, hands=[cards(g.board, loco=2), [0] * g.board.n_card_types], cur=0)
    assert claim_action(g, segment, LOCO) in s.legal_actions()
    s.step(claim_action(g, segment, LOCO))
    assert s.seg_owner[segment] == 0
    s.validate()


def test_colored_route_rejects_the_wrong_color() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Atlanta", "Miami")  # blue
    force(s, hands=[cards(g.board, red=9), [0] * g.board.n_card_types], cur=0)
    red = g.board.color_names.index("red")
    assert claim_action(g, segment, red) not in s.legal_actions()
    with pytest.raises(IllegalAction, match="route needs blue"):
        s.step(claim_action(g, segment, red))


def test_locomotives_top_up_a_short_colour_run() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Atlanta", "Miami")  # 5, blue
    force(s, hands=[cards(g.board, blue=2, loco=3), [0] * g.board.n_card_types], cur=0)
    blue = g.board.color_names.index("blue")
    s.step(claim_action(g, segment, blue))
    assert s.discard[blue] == 2
    assert s.discard[LOCO] == 3
    s.validate()


def test_canonical_payment_spends_the_most_coloured_cards_possible() -> None:
    """Locomotives are strictly more flexible, so hoarding them weakly dominates."""
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Atlanta", "Miami")  # 5, blue
    force(s, hands=[cards(g.board, blue=9, loco=9), [0] * g.board.n_card_types], cur=0)
    blue = g.board.color_names.index("blue")
    s.step(claim_action(g, segment, blue))
    assert (s.discard[blue], s.discard[LOCO]) == (5, 0)


def test_pure_locomotive_hand_does_not_multiply_gray_pay_slots() -> None:
    """The `hand[c] >= 1` guard: without it, eight ids would denote the same payment."""
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Las Vegas", "Los Angeles")  # 2, gray
    force(s, hands=[cards(g.board, loco=9), [0] * g.board.n_card_types], cur=0)
    for_segment = [
        a for a in s.legal_actions() if a < g.space.claim_end and a // g.space.k == segment
    ]
    assert for_segment == [claim_action(g, segment, LOCO)]


def test_claim_requires_enough_trains() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Atlanta", "Miami")  # 5
    force(s, hands=[cards(g.board, blue=9), [0] * g.board.n_card_types], trains=[4, 45], cur=0)
    assert claim_action(g, segment, g.board.color_names.index("blue")) not in s.legal_actions()
    with pytest.raises(IllegalAction, match="trains left"):
        s.step(claim_action(g, segment, g.board.color_names.index("blue")))


def test_a_claimed_segment_cannot_be_claimed_again() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Nashville", "Atlanta")  # 1, gray
    force(s, hands=spread_hands(g.board, 2), cur=0)
    s.step(claim_action(g, segment, 0))
    assert claim_action(g, segment, 0) not in s.legal_actions()
    assert s.cur == 1
    with pytest.raises(IllegalAction, match="taken"):
        s.step(claim_action(g, segment, 1))


def test_paid_cards_are_publicly_discarded() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Atlanta", "Miami")
    force(s, hands=[cards(g.board, blue=5), [0] * g.board.n_card_types], cur=0)
    before = s.discard_total
    s.step(claim_action(g, segment, g.board.color_names.index("blue")))
    assert s.discard_total == before + 5
    s.validate()


def test_no_adjacency_requirement() -> None:
    """A player may claim any open route; connecting to their own network is optional."""
    g = game()
    s = to_main(g.new_initial_state(1))
    a = segment_named(g.board, "Nashville", "Atlanta")
    b = segment_named(g.board, "Vancouver", "Calgary")
    force(s, hands=spread_hands(g.board, 2), cur=0)
    s.step(claim_action(g, a, 0))
    while s.cur != 0 or s.phase != PHASE_MAIN:
        s.step(s.legal_actions()[0])
    assert claim_action(g, b, 0) in s.legal_actions(), "Vancouver-Calgary touches nothing of mine"


def test_one_route_per_turn() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, hands=spread_hands(g.board, 2), cur=0)
    s.step(claim_action(g, segment_named(g.board, "Nashville", "Atlanta"), 0))
    assert s.cur == 1, "the turn ends the moment a route is claimed"


# -- double routes -----------------------------------------------------------


def test_two_or_three_players_close_the_sibling_to_everyone() -> None:
    for n in (2, 3):
        g = game(n)
        s = to_main(g.new_initial_state(1))
        segment = segment_named(g.board, "Boston", "Montreal")  # gray || gray, length 2
        sibling = g.board.sibling[segment]
        assert sibling != NO_SIBLING
        force(s, hands=spread_hands(g.board, n), cur=0)
        s.step(claim_action(g, segment, 0))
        assert s.seg_owner[sibling] == CLOSED
        while s.cur != 1 or s.phase != PHASE_MAIN:
            s.step(s.legal_actions()[0])
        assert all(a // g.space.k != sibling for a in s.legal_actions() if a < g.space.claim_end)


def test_four_or_five_players_leave_the_sibling_open_to_others() -> None:
    for n in (4, 5):
        g = game(n)
        s = to_main(g.new_initial_state(1))
        segment = segment_named(g.board, "Boston", "Montreal")
        sibling = g.board.sibling[segment]
        force(s, hands=spread_hands(g.board, n), cur=0)
        s.step(claim_action(g, segment, 0))
        assert s.seg_owner[sibling] == FREE, "the twin stays open in a 4-5P game"
        while s.cur != 1 or s.phase != PHASE_MAIN:
            s.step(s.legal_actions()[0])
        assert claim_action(g, sibling, 1) in s.legal_actions()


def test_no_player_may_own_both_tracks_of_a_double() -> None:
    g = game(4)
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Boston", "Montreal")
    sibling = g.board.sibling[segment]
    force(s, hands=spread_hands(g.board, 4), cur=0)
    s.step(claim_action(g, segment, 0))
    while s.cur != 0 or s.phase != PHASE_MAIN:
        s.step(s.legal_actions()[0])
    assert claim_action(g, sibling, 0) not in s.legal_actions()
    with pytest.raises(IllegalAction, match="both tracks"):
        s.step(claim_action(g, sibling, 0))


# ===========================================================================
# Tickets
# ===========================================================================


def test_draw_tickets_offers_three_and_keeps_at_least_one() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    s.step(g.space.draw_tickets)
    assert s.phase == PHASE_TICKET_KEEP
    assert s.offer_len == 3
    masks = sorted(a - g.space.keep_base for a in s.legal_actions())
    assert masks == list(range(1, 8)), "every non-empty subset of three is legal"


def test_keeping_nothing_is_never_legal() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    s.step(g.space.draw_tickets)
    assert g.space.keep_base not in s.legal_actions()
    with pytest.raises(IllegalAction, match="only a ticket keep"):
        s.step(g.space.keep_base)


def test_draw_tickets_is_illegal_with_an_empty_deck() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    s.tdeck_len = 0
    s.tdeck[:] = [NO_TICKET] * len(s.tdeck)
    assert g.space.draw_tickets not in s.legal_actions()
    with pytest.raises(IllegalAction, match="ticket deck is empty"):
        s.step(g.space.draw_tickets)


def test_exactly_one_ticket_left_auto_resolves_the_forced_keep() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    keep = s.tdeck[s.tdeck_head]
    s.tdeck_len = 1
    for i in range(1, len(s.tdeck)):
        s.tdeck[(s.tdeck_head + i) % len(s.tdeck)] = NO_TICKET
    before = s.turn
    s.step(g.space.draw_tickets)
    assert s.phase == PHASE_MAIN, "a one-option keep is resolved, not asked"
    assert s.turn == before + 1
    assert s.tickets[0] >> keep & 1


def test_short_ticket_deck_deals_what_is_available() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    s.tdeck_len = 2
    for i in range(2, len(s.tdeck)):
        s.tdeck[(s.tdeck_head + i) % len(s.tdeck)] = NO_TICKET
    s.step(g.space.draw_tickets)
    assert s.offer_len == 2
    assert sorted(a - g.space.keep_base for a in s.legal_actions()) == [1, 2, 3]


def test_returned_tickets_go_to_the_bottom() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    s.step(g.space.draw_tickets)
    offer = s.offer[:3]
    s.step(g.space.keep_base + 0b001)  # keep the first, return two
    tail = [s.tdeck[(s.tdeck_head + s.tdeck_len - 2 + i) % len(s.tdeck)] for i in range(2)]
    assert tail == offer[1:], "returns land at the bottom in ascending offer index"


def test_there_is_no_ticket_hand_limit() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    for _ in range(6):
        while s.phase != PHASE_MAIN or s.cur != 0:
            s.step(s.legal_actions()[0])
        if g.space.draw_tickets not in s.legal_actions():
            break
        s.step(g.space.draw_tickets)
        if s.phase == PHASE_TICKET_KEEP:
            s.step(g.space.keep_base + 0b111)
    assert int(s.tickets[0]).bit_count() > 3
    s.validate()


# ===========================================================================
# End and scoring
# ===========================================================================


def test_end_trigger_fires_at_two_trains_at_end_of_turn() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Nashville", "Atlanta")  # length 1
    force(s, hands=spread_hands(g.board, 2), trains=[3, 45], cur=0)
    s.step(claim_action(g, segment, 0))
    assert s.trains[0] == 2
    assert s.final_left == 2, "the trigger sets exactly n_players more turns"


def test_end_trigger_fires_after_a_draw_turn_too() -> None:
    """A seat sitting on two trains that spends its turn drawing still triggers."""
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, faceup=[LOCO, 0, 1, 2, 3], deck=filler(g.board, 20), trains=[2, 45], cur=0)
    assert s.final_left == 255
    s.step(draw_action(g, 0))
    assert s.final_left == 2


def test_exactly_n_more_turns_follow_the_trigger() -> None:
    for n in (2, 3, 4):
        g = game(n)
        s = to_main(g.new_initial_state(1))
        force(s, trains=[2] + [45] * (n - 1), cur=0)
        rng = stream(1, "policy")
        s.step(s.sample_legal(rng))
        while s.phase != PHASE_MAIN:
            s.step(s.sample_legal(rng))
        assert s.final_left == n
        turns = 0
        while not s.is_terminal():
            start = s.turn
            s.step(s.sample_legal(rng))
            turns += s.turn - start
        assert turns == n, f"{n}P: expected {n} final turns, got {turns}"


def test_the_end_trigger_fires_only_once() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, trains=[2, 2], cur=0)
    rng = stream(1, "policy")
    s.step(s.sample_legal(rng))
    while s.phase != PHASE_MAIN:
        s.step(s.sample_legal(rng))
    assert s.final_left == 2
    s.step(s.sample_legal(rng))
    while s.phase != PHASE_MAIN and not s.is_terminal():
        s.step(s.sample_legal(rng))
    assert s.final_left == 1, "the second seat at two trains must not re-arm the countdown"


def test_zero_trains_is_legal() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    segment = segment_named(g.board, "Nashville", "Atlanta")  # length 1
    force(s, hands=spread_hands(g.board, 2), trains=[1, 45], cur=0)
    s.step(claim_action(g, segment, 0))
    assert s.trains[0] == 0
    assert not s.is_terminal(), "zero trains is legal, not an error"


def test_tickets_settle_plus_and_minus() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    made = next(
        t
        for t in range(g.board.n_tickets)
        if g.board.dist[g.board.ticket_a[t]][g.board.ticket_b[t]] <= 4
    )
    force(s, tickets=[1 << made, 0], cur=0)
    # Claim a path for the ticket by hand, straight into seg_owner + the union-find.
    a, b = g.board.ticket_a[made], g.board.ticket_b[made]
    _connect(s, 0, a, b)
    breakdown = score_breakdown(s)[0]
    assert breakdown.ticket_points == g.board.ticket_points[made]
    assert breakdown.tickets_made == 1

    s2 = to_main(g.new_initial_state(1))
    force(s2, tickets=[1 << made, 0], cur=0)
    assert score_breakdown(s2)[0].ticket_points == -g.board.ticket_points[made]


def _connect(s, player: int, a: int, b: int) -> None:  # noqa: ANN001
    """Hand the player a shortest path between two cities, bypassing the turn structure."""
    board = s.game.board
    frontier = [(a, [])]
    seen = {a}
    while frontier:
        city, path = frontier.pop(0)
        if city == b:
            for segment in path:
                s.seg_owner[segment] = player
                dsu_union(s.dsu[player], board.seg_a[segment], board.seg_b[segment])
                s.trains[player] -= board.seg_len[segment]
                s.score[player] += board.route_points[board.seg_len[segment]]
            return
        for nb, segment in board.adjacency[city]:
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, [*path, segment]))
    raise AssertionError("no path")


def test_longest_trail_is_measured_in_train_cars_not_routes() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    # A three-route chain: Nashville-Atlanta (1) + Atlanta-Miami (5) + Miami-Charleston (4).
    for pair in (("Nashville", "Atlanta"), ("Atlanta", "Miami"), ("Charleston", "Miami")):
        _own(s, 0, segment_named(g.board, *pair))
    assert longest_trails(s)[0] == 1 + 5 + 4


def test_longest_trail_allows_loops_and_repeated_cities() -> None:
    """It is a trail: each segment once, vertices may repeat."""
    g = game()
    s = to_main(g.new_initial_state(1))
    triangle = [
        ("Nashville", "Atlanta"),
        ("Atlanta", "Raleigh"),
        ("Raleigh", "Nashville"),
    ]
    for pair in triangle:
        _own(s, 0, segment_named(g.board, *pair))
    total = sum(g.board.seg_len[segment_named(g.board, *p)] for p in triangle)
    assert longest_trails(s)[0] == total, "an Eulerian circuit uses every edge"


def _own(s, player: int, segment: int) -> None:  # noqa: ANN001
    board = s.game.board
    s.seg_owner[segment] = player
    dsu_union(s.dsu[player], board.seg_a[segment], board.seg_b[segment])
    s.trains[player] -= board.seg_len[segment]
    s.score[player] += board.route_points[board.seg_len[segment]]


def test_the_longest_path_bonus_goes_to_the_single_longest() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    _own(s, 0, segment_named(g.board, "Atlanta", "Miami"))  # 5 cars
    _own(s, 1, segment_named(g.board, "Portland", "Salt Lake City"))  # 6 cars
    assert longest_trails(s) == [5, 6]
    assert [b.longest_bonus for b in score_breakdown(s)] == [0, 10]


def test_all_tied_players_score_the_longest_path_bonus() -> None:
    """The rulebook is explicit, and implementations routinely pick one arbitrary winner."""
    g = game(3)
    s = to_main(g.new_initial_state(1))
    _own(s, 0, segment_named(g.board, "Atlanta", "Miami"))  # 5
    _own(s, 1, segment_named(g.board, "Portland", "San Francisco"))  # 5
    _own(s, 2, segment_named(g.board, "Nashville", "Atlanta"))  # 1
    assert longest_trails(s) == [5, 5, 1]
    assert [b.longest_bonus for b in score_breakdown(s)] == [10, 10, 0]


def test_a_player_with_no_track_never_ties_for_the_bonus() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    assert longest_trails(s) == [0, 0]
    assert [b.longest_bonus for b in score_breakdown(s)] == [0, 0]


def test_equal_totals_break_on_completed_tickets() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    # A ticket seat 0 will complete, and the track that completes it.
    ticket = next(
        t
        for t in range(g.board.n_tickets)
        if g.board.dist[g.board.ticket_a[t]][g.board.ticket_b[t]] <= 4
    )
    force(s, tickets=[1 << ticket, 0])
    _connect(s, 0, g.board.ticket_a[ticket], g.board.ticket_b[ticket])
    made = score_breakdown(s)
    # Level the totals so only the tiebreak can separate them.
    s.score[1] = s.score[1] + made[0].total - made[1].total
    assert [b.total for b in score_breakdown(s)] == [made[0].total] * 2
    assert winners(s) == [0], "more completed tickets wins an equal-points game"


def test_equal_totals_and_tickets_break_on_the_longest_path_card() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, tickets=[0, 0])
    _own(s, 0, segment_named(g.board, "Atlanta", "Miami"))  # 5 cars, 10 route points
    # Seat 1 matches on points without any track, so seat 0 holds the only bonus.
    s.score[:] = [s.score[0], s.score[0] + g.board.raw.longest_bonus]
    breakdowns = score_breakdown(s)
    assert breakdowns[0].total == breakdowns[1].total
    assert (breakdowns[0].longest_bonus, breakdowns[1].longest_bonus) == (10, 0)
    assert winners(s) == [0], "the longest-path card breaks a total-and-tickets tie"


def test_a_true_draw_is_possible() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, tickets=[0, 0])
    s.score[:] = [42, 42]
    s.phase = PHASE_TERMINAL
    assert final_scores(s) == [42, 42]
    assert winners(s) == [0, 1]


def test_two_player_returns_are_win_draw_loss() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    force(s, tickets=[0, 0])
    s.score[:] = [50, 40]
    assert returns(s) == [1.0, -1.0]
    s.score[:] = [40, 40]
    assert returns(s) == [0.0, 0.0]


def test_multiplayer_returns_are_constant_sum() -> None:
    for n in (3, 4, 5):
        g = game(n)
        s = to_main(g.new_initial_state(1))
        force(s, tickets=[0] * n)
        s.score[:] = list(range(10, 10 + n))
        r = returns(s)
        assert abs(sum(r)) < 1e-9, f"{n}P returns must sum to zero, got {r}"
        assert r[n - 1] == max(r), "the highest score gets the highest return"


# ===========================================================================
# Degenerate positions
# ===========================================================================


def test_pass_is_legal_only_with_no_other_option() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    assert g.space.pass_action not in s.legal_actions()
    with pytest.raises(IllegalAction, match="only legal with no other option"):
        s.step(g.space.pass_action)


def test_a_stuck_player_gets_an_explicit_pass() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    counts = list(g.board.deck_composition_counts)
    force(s, hands=[[0] * len(counts), counts], faceup=[], deck=[], trains=[0, 45], cur=0)
    s.tdeck_len = 0
    s.tdeck[:] = [NO_TICKET] * len(s.tdeck)
    assert s.legal_actions() == [g.space.pass_action]


def test_all_players_passing_freezes_and_terminates() -> None:
    g = game()
    s = to_main(g.new_initial_state(1))
    counts = list(g.board.deck_composition_counts)
    half = [c // 2 for c in counts]
    rest = [c - h for c, h in zip(counts, half, strict=True)]
    force(s, hands=[half, rest], faceup=[], deck=[], trains=[0, 0], cur=0)
    s.tdeck_len = 0
    s.tdeck[:] = [NO_TICKET] * len(s.tdeck)
    s.step(g.space.pass_action)
    assert not s.is_terminal()
    s.step(g.space.pass_action)
    assert s.is_terminal(), "a full round of passes must terminate, not hang"


def test_turn_cap_terminates() -> None:
    g = game(2, turn_cap=5)
    s = to_main(g.new_initial_state(1))
    rng = stream(1, "policy")
    while not s.is_terminal():
        s.step(s.sample_legal(rng))
    assert s.turn <= 5


# ===========================================================================
# TTR-mini exercises the same rules on a different shape
# ===========================================================================


def test_mini_plays_a_full_game() -> None:
    g = game(2, "mini")
    s = g.new_initial_state(3)
    rng = stream(3, "policy")
    while not s.is_terminal():
        s.step(s.sample_legal(rng))
        s.validate()
    assert g.space.n == MINI.n_segments * MINI.n_card_types + 15


def test_mini_locomotive_is_not_card_type_eight() -> None:
    """The trap a hard-coded `LOCO = 8` would walk straight into."""
    assert MINI.locomotive == 6
    assert USA.locomotive == 8
