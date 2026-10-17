//! `state_hash()` and `position_hash()` against docs/CONTRACT.md §3, rebuilt from the text.
//!
//! The golden corpus pins both hashes at every step -- but only as comparisons with
//! recorded values. Mutation
//! testing found nothing in this crate stating what `position_hash` *is*: zeroing the dealt
//! deck prefix by shrinking the image instead of overwriting it survived every test here.
//! So the image is rebuilt field by field from the contract's table, by code that shares
//! nothing with `State::serialize`, and both hashes are recomputed from it.

mod common;

use common::{all_configs, game, policy};
use ttr_core::hashing::hash64;
use ttr_core::state::State;

/// CONTRACT.md §3.1, in order. `canonical` is §3.2: the RNG and the dealt deck prefix
/// zeroed, everything else identical.
fn image(state: &State, canonical: bool) -> Vec<u8> {
    let board = state.board;
    let n = state.n_players();
    let k = board.n_card_types;
    let mut out = vec![
        ttr_core::CONTRACT_VERSION,
        n as u8,
        state.phase,
        state.cur,
        state.draws_left,
        state.final_left,
        state.pass_streak,
    ];
    out.extend(state.turn.to_le_bytes());
    out.extend(&state.seg_owner[..board.n_segments]);
    out.extend(&state.hand[..n * k]);
    out.extend(&state.trains[..n]);
    for seat in 0..n {
        out.extend(state.score[seat].to_le_bytes());
    }
    for seat in 0..n {
        out.extend(state.tickets[seat].to_le_bytes());
    }
    out.extend(state.deck_pos.to_le_bytes());
    out.extend(state.deck_len.to_le_bytes());
    for i in 0..usize::from(state.deck_len) {
        let dealt = i < usize::from(state.deck_pos);
        out.push(if canonical && dealt { 0 } else { state.deck[i] });
    }
    out.extend(&state.discard[..k]);
    out.extend(state.faceup.iter().map(|&c| c as u8));
    out.push(state.tdeck_head);
    out.push(state.tdeck_len);
    out.extend(&state.tdeck[..board.n_tickets]);
    out.extend(&state.certain[..n * k]);
    out.extend(&state.unknown[..n]);
    out.push(state.offer_len);
    out.extend(state.offer);
    let (rng_state, rng_inc) = if canonical {
        (0, 0)
    } else {
        (state.rng.state, state.rng.inc)
    };
    out.extend(rng_state.to_le_bytes());
    out.extend(rng_inc.to_le_bytes());
    out
}

fn check(state: &State, context: &str) {
    for canonical in [false, true] {
        assert_eq!(
            state.serialize(canonical),
            image(state, canonical),
            "{context}: serialize({canonical}) is not the contract's image"
        );
    }
    assert_eq!(
        state.state_hash(),
        hash64(&image(state, false)),
        "{context}"
    );
    assert_eq!(
        state.position_hash(),
        hash64(&image(state, true)),
        "{context}"
    );
}

/// Every step of a few games on every configuration, reshuffles included: a reshuffle
/// shortens the deck and resets the cursor, which is exactly where a prefix bug would hide.
#[test]
fn both_hashes_are_the_contract_image_at_every_step() {
    let mut reshuffled = false;
    for (map, n) in all_configs() {
        for seed in 0..4 {
            let mut state = game(map, n).new_initial_state(seed);
            let full = state.deck_len;
            let mut rng = policy(seed);
            loop {
                check(
                    &state,
                    &format!("{map} {n}P seed {seed} turn {}", state.turn),
                );
                reshuffled |= state.deck_len != full;
                if state.is_terminal() {
                    break;
                }
                let action = state.sample_legal(&mut rng).expect("not over");
                state.step(action).expect("legal");
            }
        }
    }
    assert!(
        reshuffled,
        "no game reshuffled, so the shortened-deck case went unchecked"
    );
}

#[test]
fn the_position_hash_forgets_only_what_it_should() {
    let mut state = game("usa", 2).new_initial_state(9);
    let mut rng = policy(9);
    for _ in 0..30 {
        let action = state.sample_legal(&mut rng).expect("not over");
        state.step(action).expect("legal");
    }
    let before = state.position_hash();

    // The dealt prefix and the RNG are forgotten...
    let mut reordered = state.clone();
    reordered.deck.swap(0, 1);
    reordered.rng.state ^= 1;
    assert_eq!(reordered.position_hash(), before);
    assert_ne!(reordered.state_hash(), state.state_hash());

    // ...and nothing else is: the undrawn deck, and every field after it, still count.
    let mut undrawn = state.clone();
    let cursor = usize::from(undrawn.deck_pos);
    undrawn.deck.swap(cursor, cursor + 1);
    if undrawn.deck[cursor] != state.deck[cursor] {
        assert_ne!(undrawn.position_hash(), before);
    }
    let mut later = state.clone();
    later.unknown[1] += 1;
    assert_ne!(later.position_hash(), before);
}
