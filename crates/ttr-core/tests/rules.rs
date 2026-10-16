//! The rules correctness checklist, one named test per item (PLAN.md §5.2).
//!
//! The port of `tests/unit/engine/test_rules.py`, in the same groups: setup, drawing,
//! claiming, tickets, end and scoring, degenerate positions. Most of these positions occur
//! once in thousands of random games, so they are rigged directly with `common::force`,
//! which keeps card conservation intact so `validate()` still applies.
//!
//! Three checklist items already have an exact Rust twin and are not repeated here: the
//! locomotive is not card type 8 (`board::tests::the_locomotive_is_not_hard_coded_to_eight`),
//! a seat with no track never ties for the bonus
//! (`scoring::tests::a_seat_with_no_track_never_ties_for_the_bonus`), and TTR-mini plays a
//! full validated game (`random_games::random_games_finish_and_conserve_everything`, with
//! the action-space size in `actions::tests::mini_is_two_hundred_and_twenty_five`).

mod common;

use std::collections::VecDeque;

use common::{
    Rig, cards, color, filler, force, game, give, play_out, policy, segment_named, spread_hands,
    to_main,
};
use ttr_core::board::{Board, NO_SIBLING, get_board};
use ttr_core::config::{
    CLOSED, EMPTY_SLOT, FREE, NO_TICKET, NOT_TRIGGERED, PHASE_DRAW_SECOND, PHASE_INITIAL_TICKETS,
    PHASE_MAIN, PHASE_TERMINAL, PHASE_TICKET_KEEP, RuleConfig,
};
use ttr_core::rng::Pcg32;
use ttr_core::scoring::{final_scores, longest_trails, returns, score_breakdown, winners};
use ttr_core::state::{Game, State};

// ===========================================================================
// Helpers
// ===========================================================================

/// A USA game at `n` seats, past the opening ticket choices: where most rigs start.
fn usa_main(n: usize, seed: u64) -> (Game, State) {
    let g = game("usa", n);
    let mut s = g.new_initial_state(seed);
    to_main(&mut s);
    (g, s)
}

/// The claim id for a segment found by name, paid in `pay`.
fn claim(g: &Game, segment: usize, pay: u8) -> u16 {
    g.space.claim(segment as u16, u16::from(pay))
}

/// A rigged display. Slots are `i8` so that -1 can mean empty; rigs name card types.
fn faceup(slots: &[u8]) -> Option<Vec<i8>> {
    Some(slots.iter().map(|&c| c as i8).collect())
}

fn is_legal(s: &State, action: u16) -> bool {
    s.legal_actions().contains(&action)
}

/// The engine refuses `action` with a message containing `reason`: the Rust spelling of
/// `pytest.raises(IllegalAction, match=reason)`.
fn rejects(s: &mut State, action: u16, reason: &str) {
    let name = s.space.to_string(s.board, action);
    match s.step(action) {
        Ok(()) => panic!("{name} was accepted"),
        Err(e) => assert!(
            e.0.contains(reason),
            "{name} was refused with {:?}, expected {reason:?}",
            e.0
        ),
    }
}

fn valid(s: &State) {
    if let Err(e) = s.validate() {
        panic!("validate() failed: {e}");
    }
}

fn faceup_locos(s: &State) -> usize {
    let loco = s.board.locomotive as i8;
    s.faceup.iter().filter(|&&c| c == loco).count()
}

/// The legal keep masks, ascending. Only meaningful while a ticket offer is open.
fn keep_masks(s: &State) -> Vec<u16> {
    let base = s.space.keep_base();
    s.legal_actions().iter().map(|&a| a - base).collect()
}

/// Cut the ticket deck to its top `left` tickets, blanking the rest of the ring as the
/// engine does. The ring's capacity is the board's ticket count, not the array length.
fn leave_tickets(s: &mut State, left: u8) {
    let capacity = s.board.n_tickets;
    let head = usize::from(s.tdeck_head);
    s.tdeck_len = left;
    for i in usize::from(left)..capacity {
        s.tdeck[(head + i) % capacity] = NO_TICKET;
    }
}

/// The bottom `n` tickets of the ring, top to bottom.
fn ring_tail(s: &State, n: usize) -> Vec<u8> {
    let capacity = s.board.n_tickets;
    let end = usize::from(s.tdeck_head) + usize::from(s.tdeck_len);
    (0..n).map(|i| s.tdeck[(end - n + i) % capacity]).collect()
}

/// Spend the current seat's turn drawing tickets and keeping the first.
///
/// A ticket turn moves no cards, so the refill-and-flush check that ends it runs on exactly
/// the display and pools a test rigged. That makes it the public route to the flush check
/// the Python suite reached by calling the private `_flush_check()`.
fn ticket_turn(s: &mut State) {
    let space = s.space;
    s.step(space.draw_tickets).unwrap();
    if s.phase == PHASE_TICKET_KEEP {
        s.step(space.keep(0b001)).unwrap();
    }
}

/// Play the lowest legal action until it is `seat`'s turn in MAIN.
fn lowest_until_turn_of(s: &mut State, seat: u8) {
    while s.cur != seat || s.phase != PHASE_MAIN {
        s.step(s.legal_actions()[0]).unwrap();
    }
}

fn random_step(s: &mut State, rng: &mut Pcg32) {
    let action = s.sample_legal(rng).expect("a non-terminal state");
    s.step(action).expect("a sampled action is legal");
}

/// Give `player` a segment with the trains and route points a claim would cost and score,
/// bypassing the turn structure and the cards.
fn own(s: &mut State, player: u8, segment: usize) {
    let board = s.board;
    give(s, player, segment);
    let length = board.seg_len[segment];
    s.trains[usize::from(player)] -= length;
    s.score[usize::from(player)] += i16::from(board.route_points[usize::from(length)]);
}

/// Hand the player a fewest-routes path between two cities, bypassing the turn structure.
fn connect(s: &mut State, player: u8, a: u8, b: u8) {
    let board = s.board;
    let mut frontier = VecDeque::from([(a, Vec::new())]);
    let mut seen = vec![false; board.n_cities];
    seen[usize::from(a)] = true;
    while let Some((city, path)) = frontier.pop_front() {
        if city == b {
            for segment in path {
                own(s, player, segment);
            }
            return;
        }
        for &(next, segment) in &board.adjacency[usize::from(city)] {
            if !seen[usize::from(next)] {
                seen[usize::from(next)] = true;
                let mut longer = path.clone();
                longer.push(usize::from(segment));
                frontier.push_back((next, longer));
            }
        }
    }
    panic!("no path between cities {a} and {b}");
}

/// The first ticket whose cities are at most four cars apart: cheap to complete by hand.
fn short_ticket(board: &Board) -> usize {
    (0..board.n_tickets)
        .find(|&t| board.dist[usize::from(board.ticket_a[t])][usize::from(board.ticket_b[t])] <= 4)
        .expect("a ticket at most four cars long")
}

fn bonuses(s: &mut State) -> Vec<i16> {
    score_breakdown(s).iter().map(|b| b.longest_bonus).collect()
}

// ===========================================================================
// Setup
// ===========================================================================

/// Seat p holds exactly the deck slice [p*4, p*4+4); the display comes after all of them.
///
/// Flipping the display first would deal every seat different cards, a difference no test
/// downstream of setup could attribute to its real cause.
#[test]
fn hands_are_dealt_before_the_display() {
    let g = game("usa", 3);
    let s = g.new_initial_state(7);
    let k = g.board.n_card_types;
    let hand_size = usize::from(g.cfg.initial_hand);
    for p in 0..3 {
        let mut expected = s.deck[p * hand_size..(p + 1) * hand_size].to_vec();
        expected.sort_unstable();
        // Expanded in ascending card type, so already sorted.
        let actual: Vec<u8> = (0..k)
            .flat_map(|c| std::iter::repeat_n(c as u8, usize::from(s.hand[p * k + c])))
            .collect();
        assert_eq!(
            actual, expected,
            "seat {p} did not get its block off the top"
        );
    }
    assert!(usize::from(s.deck_pos) >= 3 * hand_size + 5);
}

#[test]
fn display_holds_five_after_setup() {
    let s = game("usa", 2).new_initial_state(3);
    assert!(s.faceup.iter().all(|&c| c != EMPTY_SLOT));
}

/// 3 dealt, keep >= 2: masks 011, 101, 110, 111 and nothing else.
#[test]
fn initial_keep_has_exactly_four_legal_subsets() {
    let s = game("usa", 2).new_initial_state(11);
    assert_eq!(s.phase, PHASE_INITIAL_TICKETS);
    assert_eq!(keep_masks(&s), [0b011, 0b101, 0b110, 0b111]);
}

/// A setup display with 3+ locomotives must be flushed before anyone acts.
#[test]
fn flush_check_runs_after_setup() {
    let g = game("usa", 2);
    for seed in 0..3000 {
        let s = g.new_initial_state(seed);
        assert!(
            faceup_locos(&s) < 3,
            "seed {seed} left 3 locomotives face-up at setup"
        );
    }
}

#[test]
fn initial_tickets_are_resolved_in_seat_order() {
    let mut s = game("usa", 4).new_initial_state(5);
    let mut seen = Vec::new();
    while s.phase == PHASE_INITIAL_TICKETS {
        seen.push(s.cur);
        s.step(s.legal_actions()[0]).unwrap();
    }
    assert_eq!(seen, [0, 1, 2, 3]);
    assert_eq!(s.phase, PHASE_MAIN);
    assert_eq!((s.cur, s.turn), (0, 0));
}

#[test]
fn returned_initial_tickets_go_to_the_bottom_in_seat_then_offer_order() {
    let g = game("usa", 2);
    let mut s = g.new_initial_state(5);
    let bottom_before = s.tdeck_len;
    let mut returned = Vec::new();
    while s.phase == PHASE_INITIAL_TICKETS {
        let offer = &s.offer[..usize::from(s.offer_len)];
        returned.push(offer[2]);
        s.step(g.space.keep(0b011)).unwrap(); // keep the first two, return the third
    }
    // The two returns sit at the bottom, seat 0's before seat 1's.
    assert_eq!(ring_tail(&s, 2), returned);
    // Two seats, three drawn and one returned each: a net two tickets leave the deck per seat.
    assert_eq!(usize::from(s.tdeck_len), g.board.n_tickets - 2 * 2);
    assert!(bottom_before > s.tdeck_len);
}

#[test]
fn every_seat_keeps_at_least_two_opening_tickets() {
    let (_, s) = usa_main(3, 9);
    for (p, tickets) in s.tickets[..3].iter().enumerate() {
        assert!(tickets.count_ones() >= 2, "seat {p}");
    }
}

// ===========================================================================
// Drawing
// ===========================================================================

#[test]
fn face_up_locomotive_as_first_card_ends_the_draw() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    force(
        &mut s,
        Rig {
            faceup: faceup(&[loco, 0, 1, 2, 3]),
            deck: Some(filler(g.board, 20)),
            phase: Some(PHASE_MAIN),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let before = s.turn;
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(
        s.turn,
        before + 1,
        "taking a face-up locomotive is the whole turn"
    );
    assert_eq!(s.phase, PHASE_MAIN);
    assert_eq!(s.cur, 1);
    valid(&s);
}

#[test]
fn face_up_locomotive_is_never_legal_as_the_second_card() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    force(
        &mut s,
        Rig {
            faceup: faceup(&[0, loco, 1, 2, 3]),
            deck: Some(filler(g.board, 20)),
            phase: Some(PHASE_MAIN),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(s.phase, PHASE_DRAW_SECOND);
    assert!(!is_legal(&s, g.space.draw(1)));
    rejects(
        &mut s,
        g.space.draw(1),
        "locomotive cannot be taken as the second",
    );
}

/// The rulebook is explicit: a locomotive off the top still counts as one card.
#[test]
fn blind_locomotive_does_not_end_the_turn() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    force(
        &mut s,
        Rig {
            faceup: faceup(&[0, 1, 2, 3, 4]),
            deck: Some([vec![loco], filler(g.board, 20)].concat()),
            phase: Some(PHASE_MAIN),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(g.space.draw(5)).unwrap();
    assert_eq!(s.phase, PHASE_DRAW_SECOND);
    assert!(s.hand_of(0)[usize::from(loco)] >= 1);
    valid(&s);
}

#[test]
fn taken_slot_is_refilled_before_the_second_draw() {
    let (g, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            faceup: faceup(&[0, 1, 2, 3, 4]),
            deck: Some([vec![5, 6, 7], filler(g.board, 20)].concat()),
            phase: Some(PHASE_MAIN),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(
        s.faceup[0], 5,
        "the taken slot must be refilled from the deck immediately"
    );
    assert_eq!(s.phase, PHASE_DRAW_SECOND);
    valid(&s);
}

#[test]
fn the_refill_may_itself_be_a_locomotive_and_is_then_untakeable() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    force(
        &mut s,
        Rig {
            faceup: faceup(&[0, 1, 2, 3, 4]),
            deck: Some([vec![loco], filler(g.board, 20)].concat()),
            phase: Some(PHASE_MAIN),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(s.faceup[0], loco as i8);
    assert!(!is_legal(&s, g.space.draw(0)));
    valid(&s);
}

#[test]
fn three_face_up_locomotives_flush_all_five() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    // Take slot 4; the refill makes three locomotives and must trigger the flush.
    force(
        &mut s,
        Rig {
            faceup: faceup(&[loco, loco, 0, 1, 2]),
            deck: Some([vec![loco, 3, 4, 5, 6, 7], filler(g.board, 20)].concat()),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let before = s.discard_total;
    s.step(g.space.draw(4)).unwrap();
    assert!(faceup_locos(&s) < 3);
    assert_eq!(
        s.discard_total,
        before + 5,
        "all five face-up cards go to the discard"
    );
    valid(&s);
}

/// The replacement five can themselves hold three locomotives.
#[test]
fn flush_cascades() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    force(
        &mut s,
        Rig {
            faceup: faceup(&[loco, loco, 0, 1, 2]),
            // Slot 4's refill is a locomotive (flush 1); the next five hold three more
            // (flush 2); the five after that are clean.
            deck: Some(vec![loco, loco, loco, loco, 3, 4, 5, 6, 7, 0, 1]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let before = s.discard_total;
    s.step(g.space.draw(4)).unwrap();
    assert!(faceup_locos(&s) < 3);
    assert_eq!(
        s.discard_total,
        before + 10,
        "two flushes should have discarded ten cards"
    );
    valid(&s);
}

/// The hang guard. Without it an all-locomotive pool loops forever.
///
/// The flush check is reached through a ticket turn rather than called directly (see
/// `ticket_turn`). The cap test below takes the same route and sees it flush, so this one
/// cannot pass merely because the route skipped the check.
///
/// Stronger than the Python original, which compared only the display: reflushing an
/// all-locomotive pool deals five locomotives straight back, so that check alone still
/// passes with the guard deleted. Unmoved pools are what "did not fire" means.
#[test]
fn flush_is_blocked_when_too_few_non_locomotives_remain() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    let counts = g.board.deck_composition_counts.clone();
    let mut mine = counts.clone();
    mine[usize::from(loco)] = 0; // every locomotive is face-up or in the deck; every colour is in hand
    force(
        &mut s,
        Rig {
            hands: Some(vec![mine, vec![0; counts.len()]]),
            faceup: faceup(&[loco; 5]),
            deck: Some(vec![loco; 9]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    // Not one non-locomotive is available to flush with: the discard is empty and the
    // undrawn deck is all locomotives.
    assert_eq!(s.discard_total, 0);
    let undrawn = &s.deck[usize::from(s.deck_pos)..usize::from(s.deck_len)];
    assert!(undrawn.iter().all(|&c| c == loco));
    let before = s.faceup;
    let pools = (s.deck_pos, s.discard_total);
    ticket_turn(&mut s);
    assert_eq!(
        s.faceup, before,
        "the flush must not fire when it cannot produce a legal display"
    );
    assert_eq!(
        (s.deck_pos, s.discard_total),
        pools,
        "nor deal or discard a single card"
    );
    valid(&s);
}

/// Second lock: even with the guard satisfied, the cascade is bounded.
///
/// Reached through a ticket turn, like the guard test above.
#[test]
fn flush_cascade_cap_bails_out_deterministically() {
    let g = Game::new(RuleConfig {
        flush_cascade_cap: 2,
        ..RuleConfig::new("usa", 2).unwrap()
    })
    .unwrap();
    let mut s = g.new_initial_state(1);
    to_main(&mut s);
    let loco = g.board.locomotive;
    // Locomotives keep arriving, and the guard stays open because non-locomotives exist.
    force(
        &mut s,
        Rig {
            faceup: faceup(&[loco; 5]),
            deck: Some([vec![loco; 9], vec![0, 1, 2]].concat()),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let before = s.discard_total;
    ticket_turn(&mut s);
    assert!(
        faceup_locos(&s) >= 3,
        "the cap should have stopped the cascade mid-flight"
    );
    assert_eq!(s.discard_total, before + 2 * 5, "exactly two flushes ran");
}

#[test]
fn deck_exhausted_mid_refill_reshuffles_the_discard() {
    let (g, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            faceup: faceup(&[0, 1, 2, 3, 4]),
            deck: Some(vec![5]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert!(s.discard_total > 0);
    s.step(g.space.draw(0)).unwrap(); // consumes the last deck card refilling slot 0
    s.step(g.space.draw(5)).unwrap(); // blind draw must now come from a reshuffle
    assert!(s.deck_pos <= s.deck_len);
    valid(&s);
}

/// `deck empty, discards available` and `both empty` must stay distinguishable.
#[test]
fn reshuffle_is_lazy_not_eager() {
    let (g, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            faceup: faceup(&[0, 1, 2, 3, 4]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert_eq!(s.deck_pos, s.deck_len, "the cursor is spent");
    assert!(s.discard_total > 0);
    assert_eq!(s.deck_len, 0, "no eager reshuffle happened");
    assert!(is_legal(&s, g.space.draw(5)), "a blind draw is still legal");
}

#[test]
fn deck_and_discard_empty_leaves_a_short_display() {
    let (g, mut s) = usa_main(2, 1);
    // Give both seats everything, leaving nothing to refill with.
    let counts = g.board.deck_composition_counts.clone();
    let mut mine = counts.clone();
    mine[0] -= 5; // the five face-up cards are the only ones not in a hand
    force(
        &mut s,
        Rig {
            hands: Some(vec![mine, vec![0; counts.len()]]),
            faceup: faceup(&[0; 5]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert_eq!(s.discard_total, 0);
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(
        s.faceup.iter().filter(|&&c| c == EMPTY_SLOT).count(),
        1,
        "no card exists to refill the taken slot"
    );
    assert!(!is_legal(&s, g.space.draw(0)));
    valid(&s);
}

#[test]
fn drawing_is_entirely_illegal_when_every_pool_is_empty() {
    let (g, mut s) = usa_main(2, 1);
    let counts = g.board.deck_composition_counts.clone();
    force(
        &mut s,
        Rig {
            hands: Some(vec![counts.clone(), vec![0; counts.len()]]),
            faceup: faceup(&[]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let draws = g.space.draw_base..g.space.draw_tickets;
    assert!(
        s.legal_actions().iter().all(|a| !draws.contains(a)),
        "no draw action may be legal with no cards anywhere"
    );
    rejects(&mut s, g.space.draw(5), "no cards left");
}

/// Never construct a node with zero legal actions.
#[test]
fn turn_ends_after_one_card_when_no_second_is_obtainable() {
    let (g, mut s) = usa_main(2, 1);
    let counts = g.board.deck_composition_counts.clone();
    let mut mine = counts.clone();
    mine[0] -= 1;
    force(
        &mut s,
        Rig {
            hands: Some(vec![mine, vec![0; counts.len()]]),
            faceup: faceup(&[0]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let before = s.turn;
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(
        s.turn,
        before + 1,
        "the turn ends rather than stalling in DRAW_SECOND"
    );
    assert_eq!(s.phase, PHASE_MAIN);
    valid(&s);
}

/// Otherwise an emptied display stays empty forever, since only a take refills it.
#[test]
fn a_short_display_refills_once_a_claim_replenishes_the_discard() {
    let (g, mut s) = usa_main(2, 1);
    let counts = g.board.deck_composition_counts.clone();
    force(
        &mut s,
        Rig {
            hands: Some(vec![counts.clone(), vec![0; counts.len()]]),
            faceup: faceup(&[]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert!(s.faceup.iter().all(|&c| c == EMPTY_SLOT));
    let segment = segment_named(g.board, "Nashville", "Atlanta", 0); // length 1, gray
    s.step(claim(&g, segment, 0)).unwrap();
    assert!(
        s.faceup.iter().any(|&c| c != EMPTY_SLOT),
        "the display should refill"
    );
    valid(&s);
}

// ===========================================================================
// Claiming
// ===========================================================================

#[test]
fn claim_pays_exactly_the_route_length_and_scores_the_table() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Atlanta", "Miami", 0); // 5, blue
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("blue", 5)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(claim(&g, segment, color(g.board, "blue"))).unwrap();
    assert_eq!(s.seg_owner[segment], 0);
    assert_eq!(s.trains[0], g.board.raw.trains_per_player - 5);
    assert_eq!(s.score[0], 10, "a length-5 route scores 10");
    assert_eq!(s.hand_size(0), 0);
    valid(&s);
}

#[test]
fn route_scoring_table() {
    let usa = get_board("usa").unwrap();
    for (length, points) in [(1, 1), (2, 2), (3, 4), (4, 7), (5, 10), (6, 15)] {
        assert_eq!(usa.route_points[length], points, "length {length}");
    }
}

#[test]
fn gray_route_accepts_any_single_color() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Las Vegas", "Los Angeles", 0); // 2, gray
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("white", 2)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let white = claim(&g, segment, color(g.board, "white"));
    assert!(is_legal(&s, white));
    s.step(white).unwrap();
    assert_eq!(s.seg_owner[segment], 0);
    valid(&s);
}

#[test]
fn gray_route_accepts_all_locomotives() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Las Vegas", "Los Angeles", 0);
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("loco", 2)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let locos = claim(&g, segment, g.board.locomotive);
    assert!(is_legal(&s, locos));
    s.step(locos).unwrap();
    assert_eq!(s.seg_owner[segment], 0);
    valid(&s);
}

#[test]
fn colored_route_rejects_the_wrong_color() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Atlanta", "Miami", 0); // blue
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("red", 9)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let red = claim(&g, segment, color(g.board, "red"));
    assert!(!is_legal(&s, red));
    rejects(&mut s, red, "route needs blue");
}

#[test]
fn locomotives_top_up_a_short_colour_run() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Atlanta", "Miami", 0); // 5, blue
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("blue", 2), ("loco", 3)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let blue = color(g.board, "blue");
    s.step(claim(&g, segment, blue)).unwrap();
    assert_eq!(s.discard[usize::from(blue)], 2);
    assert_eq!(s.discard[usize::from(g.board.locomotive)], 3);
    valid(&s);
}

/// Locomotives are strictly more flexible, so hoarding them weakly dominates.
#[test]
fn canonical_payment_spends_the_most_coloured_cards_possible() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Atlanta", "Miami", 0); // 5, blue
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("blue", 9), ("loco", 9)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let blue = color(g.board, "blue");
    s.step(claim(&g, segment, blue)).unwrap();
    assert_eq!(
        (
            s.discard[usize::from(blue)],
            s.discard[usize::from(g.board.locomotive)]
        ),
        (5, 0)
    );
}

/// The `hand[c] >= 1` guard: without it, eight ids would denote the same payment.
#[test]
fn pure_locomotive_hand_does_not_multiply_gray_pay_slots() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Las Vegas", "Los Angeles", 0); // 2, gray
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("loco", 9)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let for_segment: Vec<u16> = s
        .legal_actions()
        .into_iter()
        .filter(|&a| a < g.space.claim_end && usize::from(a / g.space.k) == segment)
        .collect();
    assert_eq!(for_segment, [claim(&g, segment, g.board.locomotive)]);
}

#[test]
fn claim_requires_enough_trains() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Atlanta", "Miami", 0); // 5
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("blue", 9)]),
                vec![0; g.board.n_card_types],
            ]),
            trains: Some(vec![4, 45]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let blue = claim(&g, segment, color(g.board, "blue"));
    assert!(!is_legal(&s, blue));
    rejects(&mut s, blue, "trains left");
}

#[test]
fn a_claimed_segment_cannot_be_claimed_again() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Nashville", "Atlanta", 0); // 1, gray
    force(
        &mut s,
        Rig {
            hands: Some(spread_hands(g.board, 2, 6)),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(claim(&g, segment, 0)).unwrap();
    assert!(!is_legal(&s, claim(&g, segment, 0)));
    assert_eq!(s.cur, 1);
    rejects(&mut s, claim(&g, segment, 1), "taken");
}

#[test]
fn paid_cards_are_publicly_discarded() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Atlanta", "Miami", 0);
    force(
        &mut s,
        Rig {
            hands: Some(vec![
                cards(g.board, &[("blue", 5)]),
                vec![0; g.board.n_card_types],
            ]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let before = s.discard_total;
    s.step(claim(&g, segment, color(g.board, "blue"))).unwrap();
    assert_eq!(s.discard_total, before + 5);
    valid(&s);
}

/// A player may claim any open route; connecting to their own network is optional.
#[test]
fn no_adjacency_requirement() {
    let (g, mut s) = usa_main(2, 1);
    let a = segment_named(g.board, "Nashville", "Atlanta", 0);
    let b = segment_named(g.board, "Vancouver", "Calgary", 0);
    force(
        &mut s,
        Rig {
            hands: Some(spread_hands(g.board, 2, 6)),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(claim(&g, a, 0)).unwrap();
    lowest_until_turn_of(&mut s, 0);
    assert!(
        is_legal(&s, claim(&g, b, 0)),
        "Vancouver-Calgary touches nothing of mine"
    );
}

#[test]
fn one_route_per_turn() {
    let (g, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            hands: Some(spread_hands(g.board, 2, 6)),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let segment = segment_named(g.board, "Nashville", "Atlanta", 0);
    s.step(claim(&g, segment, 0)).unwrap();
    assert_eq!(s.cur, 1, "the turn ends the moment a route is claimed");
}

// -- double routes -----------------------------------------------------------

#[test]
fn two_or_three_players_close_the_sibling_to_everyone() {
    for n in [2, 3] {
        let (g, mut s) = usa_main(n, 1);
        let segment = segment_named(g.board, "Boston", "Montreal", 0); // gray || gray, length 2
        let sibling = g.board.sibling[segment];
        assert_ne!(sibling, NO_SIBLING);
        force(
            &mut s,
            Rig {
                hands: Some(spread_hands(g.board, n, 6)),
                cur: Some(0),
                ..Rig::default()
            },
        );
        s.step(claim(&g, segment, 0)).unwrap();
        assert_eq!(s.seg_owner[usize::from(sibling)], CLOSED, "{n}P");
        lowest_until_turn_of(&mut s, 1);
        assert!(
            s.legal_actions()
                .iter()
                .filter(|&&a| a < g.space.claim_end)
                .all(|&a| a / g.space.k != sibling),
            "{n}P: the sibling must be closed to the other seats too"
        );
    }
}

#[test]
fn four_or_five_players_leave_the_sibling_open_to_others() {
    for n in [4, 5] {
        let (g, mut s) = usa_main(n, 1);
        let segment = segment_named(g.board, "Boston", "Montreal", 0);
        let sibling = usize::from(g.board.sibling[segment]);
        force(
            &mut s,
            Rig {
                hands: Some(spread_hands(g.board, n, 6)),
                cur: Some(0),
                ..Rig::default()
            },
        );
        s.step(claim(&g, segment, 0)).unwrap();
        assert_eq!(
            s.seg_owner[sibling], FREE,
            "{n}P: the twin stays open in a 4-5P game"
        );
        lowest_until_turn_of(&mut s, 1);
        assert!(is_legal(&s, claim(&g, sibling, 1)), "{n}P");
    }
}

#[test]
fn no_player_may_own_both_tracks_of_a_double() {
    let (g, mut s) = usa_main(4, 1);
    let segment = segment_named(g.board, "Boston", "Montreal", 0);
    let sibling = usize::from(g.board.sibling[segment]);
    force(
        &mut s,
        Rig {
            hands: Some(spread_hands(g.board, 4, 6)),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(claim(&g, segment, 0)).unwrap();
    lowest_until_turn_of(&mut s, 0);
    assert!(!is_legal(&s, claim(&g, sibling, 0)));
    rejects(&mut s, claim(&g, sibling, 0), "both tracks");
}

// ===========================================================================
// Tickets
// ===========================================================================

#[test]
fn draw_tickets_offers_three_and_keeps_at_least_one() {
    let (g, mut s) = usa_main(2, 1);
    s.step(g.space.draw_tickets).unwrap();
    assert_eq!(s.phase, PHASE_TICKET_KEEP);
    assert_eq!(s.offer_len, 3);
    assert_eq!(
        keep_masks(&s),
        (1..8).collect::<Vec<u16>>(),
        "every non-empty subset of three is legal"
    );
}

#[test]
fn keeping_nothing_is_never_legal() {
    let (g, mut s) = usa_main(2, 1);
    s.step(g.space.draw_tickets).unwrap();
    assert!(!is_legal(&s, g.space.keep_base()));
    rejects(&mut s, g.space.keep_base(), "only a ticket keep");
}

#[test]
fn draw_tickets_is_illegal_with_an_empty_deck() {
    let (g, mut s) = usa_main(2, 1);
    leave_tickets(&mut s, 0);
    assert!(!is_legal(&s, g.space.draw_tickets));
    rejects(&mut s, g.space.draw_tickets, "ticket deck is empty");
}

#[test]
fn exactly_one_ticket_left_auto_resolves_the_forced_keep() {
    let (g, mut s) = usa_main(2, 1);
    let keep = s.tdeck[usize::from(s.tdeck_head)];
    leave_tickets(&mut s, 1);
    let before = s.turn;
    s.step(g.space.draw_tickets).unwrap();
    assert_eq!(
        s.phase, PHASE_MAIN,
        "a one-option keep is resolved, not asked"
    );
    assert_eq!(s.turn, before + 1);
    assert_eq!(
        (s.tickets[0] >> keep) & 1,
        1,
        "seat 0 holds the last ticket"
    );
}

#[test]
fn short_ticket_deck_deals_what_is_available() {
    let (g, mut s) = usa_main(2, 1);
    leave_tickets(&mut s, 2);
    s.step(g.space.draw_tickets).unwrap();
    assert_eq!(s.offer_len, 2);
    assert_eq!(keep_masks(&s), [1, 2, 3]);
}

#[test]
fn returned_tickets_go_to_the_bottom() {
    let (g, mut s) = usa_main(2, 1);
    s.step(g.space.draw_tickets).unwrap();
    let offer = s.offer;
    s.step(g.space.keep(0b001)).unwrap(); // keep the first, return two
    assert_eq!(
        ring_tail(&s, 2),
        &offer[1..],
        "returns land at the bottom in ascending offer index"
    );
}

#[test]
fn there_is_no_ticket_hand_limit() {
    let (g, mut s) = usa_main(2, 1);
    for _ in 0..6 {
        lowest_until_turn_of(&mut s, 0);
        if !is_legal(&s, g.space.draw_tickets) {
            break;
        }
        s.step(g.space.draw_tickets).unwrap();
        if s.phase == PHASE_TICKET_KEEP {
            s.step(g.space.keep(0b111)).unwrap();
        }
    }
    assert!(s.tickets[0].count_ones() > 3);
    valid(&s);
}

// ===========================================================================
// End and scoring
// ===========================================================================

#[test]
fn end_trigger_fires_at_two_trains_at_end_of_turn() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Nashville", "Atlanta", 0); // length 1
    force(
        &mut s,
        Rig {
            hands: Some(spread_hands(g.board, 2, 6)),
            trains: Some(vec![3, 45]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(claim(&g, segment, 0)).unwrap();
    assert_eq!(s.trains[0], 2);
    assert_eq!(
        s.final_left, 2,
        "the trigger sets exactly n_players more turns"
    );
}

/// A seat sitting on two trains that spends its turn drawing still triggers.
#[test]
fn end_trigger_fires_after_a_draw_turn_too() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    force(
        &mut s,
        Rig {
            faceup: faceup(&[loco, 0, 1, 2, 3]),
            deck: Some(filler(g.board, 20)),
            trains: Some(vec![2, 45]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert_eq!(s.final_left, NOT_TRIGGERED);
    s.step(g.space.draw(0)).unwrap();
    assert_eq!(s.final_left, 2);
}

#[test]
fn exactly_n_more_turns_follow_the_trigger() {
    for n in [2, 3, 4] {
        let (_, mut s) = usa_main(n, 1);
        let mut trains = vec![45; n];
        trains[0] = 2;
        force(
            &mut s,
            Rig {
                trains: Some(trains),
                cur: Some(0),
                ..Rig::default()
            },
        );
        let mut rng = policy(1);
        random_step(&mut s, &mut rng);
        while s.phase != PHASE_MAIN {
            random_step(&mut s, &mut rng);
        }
        assert_eq!(usize::from(s.final_left), n);
        let mut turns = 0;
        while !s.is_terminal() {
            let start = s.turn;
            random_step(&mut s, &mut rng);
            turns += usize::from(s.turn - start);
        }
        assert_eq!(turns, n, "{n}P: expected {n} final turns, got {turns}");
    }
}

#[test]
fn the_end_trigger_fires_only_once() {
    let (_, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            trains: Some(vec![2, 2]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let mut rng = policy(1);
    random_step(&mut s, &mut rng);
    while s.phase != PHASE_MAIN {
        random_step(&mut s, &mut rng);
    }
    assert_eq!(s.final_left, 2);
    random_step(&mut s, &mut rng);
    while s.phase != PHASE_MAIN && !s.is_terminal() {
        random_step(&mut s, &mut rng);
    }
    assert_eq!(
        s.final_left, 1,
        "the second seat at two trains must not re-arm the countdown"
    );
}

#[test]
fn zero_trains_is_legal() {
    let (g, mut s) = usa_main(2, 1);
    let segment = segment_named(g.board, "Nashville", "Atlanta", 0); // length 1
    force(
        &mut s,
        Rig {
            hands: Some(spread_hands(g.board, 2, 6)),
            trains: Some(vec![1, 45]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    s.step(claim(&g, segment, 0)).unwrap();
    assert_eq!(s.trains[0], 0);
    assert!(!s.is_terminal(), "zero trains is legal, not an error");
}

#[test]
fn tickets_settle_plus_and_minus() {
    let (g, mut s) = usa_main(2, 1);
    let made = short_ticket(g.board);
    let value = i16::from(g.board.ticket_points[made]);
    force(
        &mut s,
        Rig {
            tickets: Some(vec![1 << made, 0]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    // Claim a path for the ticket by hand, straight into seg_owner + the union-find.
    connect(&mut s, 0, g.board.ticket_a[made], g.board.ticket_b[made]);
    let breakdown = score_breakdown(&mut s)[0];
    assert_eq!(breakdown.ticket_points, value);
    assert_eq!(breakdown.tickets_made, 1);

    let mut s2 = g.new_initial_state(1);
    to_main(&mut s2);
    force(
        &mut s2,
        Rig {
            tickets: Some(vec![1 << made, 0]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert_eq!(score_breakdown(&mut s2)[0].ticket_points, -value);
}

#[test]
fn longest_trail_is_measured_in_train_cars_not_routes() {
    let (g, mut s) = usa_main(2, 1);
    // A three-route chain: Nashville-Atlanta (1) + Atlanta-Miami (5) + Miami-Charleston (4).
    for (a, b) in [
        ("Nashville", "Atlanta"),
        ("Atlanta", "Miami"),
        ("Charleston", "Miami"),
    ] {
        own(&mut s, 0, segment_named(g.board, a, b, 0));
    }
    assert_eq!(longest_trails(&s)[0], 1 + 5 + 4);
}

/// It is a trail: each segment once, vertices may repeat.
#[test]
fn longest_trail_allows_loops_and_repeated_cities() {
    let (g, mut s) = usa_main(2, 1);
    let triangle = [
        ("Nashville", "Atlanta"),
        ("Atlanta", "Raleigh"),
        ("Raleigh", "Nashville"),
    ];
    for (a, b) in triangle {
        own(&mut s, 0, segment_named(g.board, a, b, 0));
    }
    let total: u16 = triangle
        .iter()
        .map(|&(a, b)| u16::from(g.board.seg_len[segment_named(g.board, a, b, 0)]))
        .sum();
    assert_eq!(
        longest_trails(&s)[0],
        total,
        "an Eulerian circuit uses every edge"
    );
}

#[test]
fn the_longest_path_bonus_goes_to_the_single_longest() {
    let (g, mut s) = usa_main(2, 1);
    own(&mut s, 0, segment_named(g.board, "Atlanta", "Miami", 0)); // 5 cars
    own(
        &mut s,
        1,
        segment_named(g.board, "Portland", "Salt Lake City", 0),
    ); // 6 cars
    assert_eq!(longest_trails(&s), [5, 6]);
    assert_eq!(bonuses(&mut s), [0, 10]);
}

/// The rulebook is explicit, and implementations routinely pick one arbitrary winner.
#[test]
fn all_tied_players_score_the_longest_path_bonus() {
    let (g, mut s) = usa_main(3, 1);
    own(&mut s, 0, segment_named(g.board, "Atlanta", "Miami", 0)); // 5
    own(
        &mut s,
        1,
        segment_named(g.board, "Portland", "San Francisco", 0),
    ); // 5
    own(&mut s, 2, segment_named(g.board, "Nashville", "Atlanta", 0)); // 1
    assert_eq!(longest_trails(&s), [5, 5, 1]);
    assert_eq!(bonuses(&mut s), [10, 10, 0]);
}

#[test]
fn equal_totals_break_on_completed_tickets() {
    let (g, mut s) = usa_main(2, 1);
    // A ticket seat 0 will complete, and the track that completes it.
    let ticket = short_ticket(g.board);
    force(
        &mut s,
        Rig {
            tickets: Some(vec![1 << ticket, 0]),
            ..Rig::default()
        },
    );
    connect(
        &mut s,
        0,
        g.board.ticket_a[ticket],
        g.board.ticket_b[ticket],
    );
    let made = score_breakdown(&mut s);
    // Level the totals so only the tiebreak can separate them.
    s.score[1] += made[0].total - made[1].total;
    let totals: Vec<i16> = score_breakdown(&mut s).iter().map(|b| b.total).collect();
    assert_eq!(totals, [made[0].total; 2]);
    assert_eq!(
        winners(&mut s),
        [0],
        "more completed tickets wins an equal-points game"
    );
}

#[test]
fn equal_totals_and_tickets_break_on_the_longest_path_card() {
    let (g, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            tickets: Some(vec![0, 0]),
            ..Rig::default()
        },
    );
    own(&mut s, 0, segment_named(g.board, "Atlanta", "Miami", 0)); // 5 cars, 10 route points
    // Seat 1 matches on points without any track, so seat 0 holds the only bonus.
    s.score[1] = s.score[0] + i16::from(g.board.raw.longest_bonus);
    let breakdowns = score_breakdown(&mut s);
    assert_eq!(breakdowns[0].total, breakdowns[1].total);
    assert_eq!(
        (breakdowns[0].longest_bonus, breakdowns[1].longest_bonus),
        (10, 0)
    );
    assert_eq!(
        winners(&mut s),
        [0],
        "the longest-path card breaks a total-and-tickets tie"
    );
}

#[test]
fn a_true_draw_is_possible() {
    let (_, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            tickets: Some(vec![0, 0]),
            ..Rig::default()
        },
    );
    s.score[..2].copy_from_slice(&[42, 42]);
    s.phase = PHASE_TERMINAL;
    assert_eq!(final_scores(&mut s), [42, 42]);
    assert_eq!(winners(&mut s), [0, 1]);
}

#[test]
fn two_player_returns_are_win_draw_loss() {
    let (_, mut s) = usa_main(2, 1);
    force(
        &mut s,
        Rig {
            tickets: Some(vec![0, 0]),
            ..Rig::default()
        },
    );
    s.score[..2].copy_from_slice(&[50, 40]);
    assert_eq!(returns(&mut s), [1.0, -1.0]);
    s.score[..2].copy_from_slice(&[40, 40]);
    assert_eq!(returns(&mut s), [0.0, 0.0]);
}

#[test]
fn multiplayer_returns_are_constant_sum() {
    for n in [3, 4, 5] {
        let (_, mut s) = usa_main(n, 1);
        force(
            &mut s,
            Rig {
                tickets: Some(vec![0; n]),
                ..Rig::default()
            },
        );
        for (p, score) in s.score[..n].iter_mut().enumerate() {
            *score = 10 + p as i16;
        }
        let r = returns(&mut s);
        let total: f64 = r.iter().sum();
        assert!(
            total.abs() < 1e-9,
            "{n}P returns must sum to zero, got {r:?}"
        );
        let best = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            r[n - 1],
            best,
            "{n}P: the highest score gets the highest return"
        );
    }
}

// ===========================================================================
// Degenerate positions
// ===========================================================================

#[test]
fn pass_is_legal_only_with_no_other_option() {
    let (g, mut s) = usa_main(2, 1);
    assert!(!is_legal(&s, g.space.pass_action));
    rejects(
        &mut s,
        g.space.pass_action,
        "only legal with no other option",
    );
}

#[test]
fn a_stuck_player_gets_an_explicit_pass() {
    let (g, mut s) = usa_main(2, 1);
    let counts = g.board.deck_composition_counts.clone();
    force(
        &mut s,
        Rig {
            hands: Some(vec![vec![0; counts.len()], counts]),
            faceup: faceup(&[]),
            deck: Some(vec![]),
            trains: Some(vec![0, 45]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    leave_tickets(&mut s, 0);
    assert_eq!(s.legal_actions(), [g.space.pass_action]);
}

#[test]
fn all_players_passing_freezes_and_terminates() {
    let (g, mut s) = usa_main(2, 1);
    let counts = &g.board.deck_composition_counts;
    let half: Vec<u8> = counts.iter().map(|&c| c / 2).collect();
    let rest: Vec<u8> = counts.iter().zip(&half).map(|(&c, &h)| c - h).collect();
    force(
        &mut s,
        Rig {
            hands: Some(vec![half, rest]),
            faceup: faceup(&[]),
            deck: Some(vec![]),
            trains: Some(vec![0, 0]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    leave_tickets(&mut s, 0);
    s.step(g.space.pass_action).unwrap();
    assert!(!s.is_terminal());
    s.step(g.space.pass_action).unwrap();
    assert!(
        s.is_terminal(),
        "a full round of passes must terminate, not hang"
    );
}

#[test]
fn turn_cap_terminates() {
    let g = Game::new(RuleConfig {
        turn_cap: 5,
        ..RuleConfig::new("usa", 2).unwrap()
    })
    .unwrap();
    let mut s = g.new_initial_state(1);
    to_main(&mut s);
    play_out(&mut s, &mut policy(1));
    assert!(s.turn <= 5);
}

// ===========================================================================
// Found by mutation testing (Phase 3.1)
// ===========================================================================

/// The flush guard counts the *discard's* locomotives as locomotives too.
///
/// The guard test above has an empty discard, so a guard that counted discarded
/// locomotives as available non-locomotives survived it. Here the deck is all locomotives
/// and so is the discard: still nothing to flush with.
#[test]
fn the_flush_guard_does_not_count_discarded_locomotives_as_flushable() {
    let (g, mut s) = usa_main(2, 1);
    let loco = g.board.locomotive;
    let counts = g.board.deck_composition_counts.clone();
    let mut mine = counts.clone();
    mine[usize::from(loco)] = 0;
    force(
        &mut s,
        Rig {
            hands: Some(vec![mine, vec![0; counts.len()]]),
            faceup: faceup(&[loco; 5]),
            deck: Some(vec![loco; 6]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    // The three locomotives left over went to the discard, and nothing else did.
    assert_eq!(s.discard[usize::from(loco)], 3);
    assert_eq!(s.discard_total, 3);
    let before = (s.faceup, s.deck_pos, s.discard_total);
    ticket_turn(&mut s);
    assert_eq!(
        (s.faceup, s.deck_pos, s.discard_total),
        before,
        "a flush fired with no non-locomotive anywhere to deal"
    );
    valid(&s);
}

/// An opening offer with exactly one legal keep is resolved without asking, and the turn
/// passes on (CONTRACT.md §2.2 step 5).
///
/// Unreachable with the real boards -- the ticket deck always covers a full table, which
/// `enough_tickets_for_a_full_table` enforces -- so it is rigged: one ticket left in the
/// ring when seat 0 decides. Seat 0 returns one, seat 1 is then offered exactly two with
/// "keep at least two" (one legal keep, so it is taken for it), and play reaches seat 2,
/// whose offer is empty.
#[test]
fn a_forced_opening_keep_is_taken_and_the_next_seat_is_dealt() {
    let g = game("usa", 3);
    let mut s = g.new_initial_state(4);
    assert_eq!((s.phase, s.cur), (PHASE_INITIAL_TICKETS, 0));
    let capacity = g.board.n_tickets;
    let head = usize::from(s.tdeck_head);
    let last = s.tdeck[head];
    for slot in 0..capacity {
        if slot != head {
            s.tdeck[slot] = NO_TICKET;
        }
    }
    s.tdeck_len = 1;
    let returned = s.offer[2];

    s.step(s.space.keep(0b011)).unwrap();

    let expected = (1u32 << last) | (1u32 << returned);
    assert_eq!(
        s.tickets[1], expected,
        "seat 1 kept both tickets it was dealt"
    );
    assert_eq!(s.cur, 2, "and play moved on to seat 2");
    assert_eq!((s.offer_len, s.tdeck_len), (0, 0));
}
