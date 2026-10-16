//! `State::validate` really rejects what it says it rejects.
//!
//! Every other test calls `validate()` on positions the engine produced and expects
//! `Ok` -- which mutation testing showed a `validate` that always returns `Ok(())` would
//! pass. A self-check that cannot fail checks nothing, so each invariant is broken here on
//! purpose, one at a time, and the specific complaint is asserted.

mod common;

use common::{Rig, filler, force, game, policy, to_main};
use ttr_core::config::{EMPTY_SLOT, NO_TICKET, PHASE_NAMES, PHASE_TICKET_KEEP};
use ttr_core::state::{IllegalAction, State};

/// A valid USA 3P position some way into a game.
fn mid_game() -> State {
    let mut state = game("usa", 3).new_initial_state(11);
    let mut rng = policy(11);
    for _ in 0..50 {
        let action = state.sample_legal(&mut rng).expect("not over");
        state.step(action).expect("legal");
    }
    state.validate().expect("the starting position is valid");
    state
}

fn rejects(state: &State, complaint: &str) {
    match state.validate() {
        Ok(()) => panic!("validate() accepted a position that breaks: {complaint}"),
        Err(e) => assert!(e.contains(complaint), "expected {complaint:?}, got {e:?}"),
    }
}

#[test]
fn a_drifted_discard_count_is_caught() {
    let mut state = mid_game();
    state.discard_total += 1;
    rejects(&state, "discard fast-count drifted");
}

#[test]
fn an_invented_card_is_caught() {
    let mut state = mid_game();
    state.hand[0] += 1;
    state.unknown[0] += 1;
    rejects(&state, "cards leaked");
}

#[test]
fn hidden_card_bookkeeping_that_does_not_add_up_is_caught() {
    let mut state = mid_game();
    state.unknown[1] += 1;
    rejects(&state, "seat 1: certain");
}

#[test]
fn knowing_more_than_a_seat_holds_is_caught() {
    let mut state = mid_game();
    let k = state.board.n_card_types;
    let seat = (0..3)
        .find(|&p| state.unknown[p] > 0)
        .expect("someone drew blind");
    let colour = (0..k)
        .find(|&c| state.hand[seat * k + c] == 0)
        .expect("a colour this seat holds none of");
    state.certain[seat * k + colour] += 1;
    state.unknown[seat] -= 1;
    rejects(&state, "publicly known to hold more");
}

#[test]
fn trains_that_were_never_spent_are_caught() {
    let mut state = mid_game();
    state.trains[2] -= 1;
    rejects(&state, "seat 2 trains");
}

#[test]
fn a_ticket_in_two_places_is_caught() {
    let mut state = mid_game();
    let in_deck = state.tdeck[usize::from(state.tdeck_head)];
    assert_ne!(in_deck, NO_TICKET);
    state.tickets[0] |= 1 << in_deck;
    rejects(&state, "tickets leaked");
}

#[test]
fn a_ring_that_disagrees_with_its_length_is_caught() {
    let mut state = mid_game();
    let n = state.board.n_tickets;
    let blank = (0..n)
        .find(|&i| state.tdeck[i] == NO_TICKET)
        .expect("dealt tickets leave blank slots");
    state.tdeck[blank] = 0;
    rejects(&state, "ring length disagrees");
}

#[test]
fn a_flush_that_should_have_fired_is_caught() {
    // Three locomotives face-up with plenty of other cards about, and no cascade bail-out
    // recorded: the assertion most engines lack.
    let mut state = game("usa", 2).new_initial_state(2);
    to_main(&mut state);
    let board = state.board;
    let loco = board.locomotive as i8;
    force(
        &mut state,
        Rig {
            faceup: Some(vec![loco, loco, loco, 0, 1]),
            deck: Some(filler(board, 20)),
            ..Rig::default()
        },
    );
    state.flush_capped = false;
    rejects(&state, "the flush should have fired");
    state.flush_capped = true;
    state
        .validate()
        .expect("a recorded bail-out makes it legitimate");
}

#[test]
fn a_current_player_outside_the_table_is_caught() {
    let mut state = mid_game();
    state.cur = 3;
    rejects(&state, "current player 3 outside 0..3");
}

#[test]
fn a_live_position_with_nothing_to_do_is_caught() {
    let mut state = mid_game();
    state.phase = PHASE_TICKET_KEEP;
    state.offer_len = 0;
    state.offer = [NO_TICKET; 3];
    rejects(&state, "no legal actions");
}

#[test]
fn phases_are_named_as_the_contract_spells_them() {
    let mut state = mid_game();
    for (phase, name) in PHASE_NAMES.iter().enumerate() {
        state.phase = phase as u8;
        assert_eq!(state.phase_name(), *name);
    }
    assert_eq!(
        PHASE_NAMES,
        [
            "INITIAL_TICKETS",
            "MAIN",
            "DRAW_SECOND",
            "TICKET_KEEP",
            "TERMINAL"
        ]
    );
}

#[test]
fn an_illegal_action_says_why() {
    // The message is the whole interface: a policy leaking past its mask is reported by it.
    let mut state = mid_game();
    let n = state.space.n;
    let err = state.step(n).expect_err("out of range");
    assert_eq!(err.to_string(), err.0);
    assert!(
        err.to_string().contains(&format!("action {n} outside")),
        "{err}"
    );
    assert_eq!(IllegalAction("x".into()).to_string(), "x");
    assert!(state.faceup.iter().any(|&c| c != EMPTY_SLOT));
}
