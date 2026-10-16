//! Helpers for forcing the engine into a specific, otherwise-rare position.
//!
//! The Rust successor to `tests/rig.py`. Rules edge cases are exactly the positions a
//! random game reaches once in ten thousand tries, so the tests build them directly.
//! [`force`] keeps **card** conservation intact, so [`State::validate`] stays usable on a
//! rigged state -- most of the point.
//!
//! The one invariant it cannot preserve is `trains[p] + track owned == trains_per_player`:
//! setting a seat's train count without inventing the claims that spent them is exactly what
//! several end-of-game tests need. Tests that pass `trains` must skip `validate()`.
//!
//! Each integration-test binary compiles this module separately and uses a different
//! subset, hence the blanket `dead_code` allowance.

#![allow(dead_code)]

use ttr_core::board::{Board, FACEUP_SLOTS};
use ttr_core::config::{EMPTY_SLOT, PHASE_MAIN, RuleConfig};
use ttr_core::graph::dsu_union;
use ttr_core::rng::{Part, Pcg32, stream};
use ttr_core::state::{Game, State};

pub fn game(map: &str, n_players: usize) -> Game {
    Game::new(RuleConfig::new(map, n_players).expect("a supported configuration"))
        .expect("a supported configuration")
}

/// Every `(map, seat count)` the engine supports.
pub fn all_configs() -> Vec<(&'static str, usize)> {
    ttr_core::board::all_boards()
        .iter()
        .flat_map(|b| {
            (usize::from(b.raw.min_players)..=usize::from(b.raw.max_players)).map(|n| (b.name, n))
        })
        .collect()
}

/// A seed's own stream for driving play, disjoint from every environment stream.
pub fn policy(seed: u64) -> Pcg32 {
    stream(seed, &[Part::Str("policy")])
}

/// Card-type counts from colour names: `cards(board, &[("blue", 3), ("loco", 1)])`.
///
/// `loco` is spelled out rather than assumed to be index 8 -- TTR-mini's locomotive is
/// card type 6.
pub fn cards(board: &Board, counts: &[(&str, u8)]) -> Vec<u8> {
    let mut out = vec![0u8; board.n_card_types];
    for &(name, count) in counts {
        let index = if name == "loco" {
            usize::from(board.locomotive)
        } else {
            color(board, name) as usize
        };
        out[index] = count;
    }
    out
}

/// A colour's index by name.
pub fn color(board: &Board, name: &str) -> u8 {
    board
        .raw
        .color_names
        .iter()
        .position(|&c| c == name)
        .unwrap_or_else(|| panic!("{} has no colour {name:?}", board.name)) as u8
}

/// A city's index by name.
pub fn city(board: &Board, name: &str) -> u8 {
    board
        .raw
        .cities
        .iter()
        .position(|&c| c == name)
        .unwrap_or_else(|| panic!("{} has no city {name:?}", board.name)) as u8
}

/// The `which`-th segment between two cities, by name.
pub fn segment_named(board: &Board, a: &str, b: &str, which: usize) -> usize {
    let (mut ia, mut ib) = (city(board, a), city(board, b));
    if ia > ib {
        std::mem::swap(&mut ia, &mut ib);
    }
    let found: Vec<usize> = (0..board.n_segments)
        .filter(|&s| (board.seg_a[s], board.seg_b[s]) == (ia, ib))
        .collect();
    *found
        .get(which)
        .unwrap_or_else(|| panic!("no segment #{which} between {a} and {b}"))
}

/// `n` cards that certainly exist: cycles the colours and never the locomotive.
///
/// A rigged deck of `[0; 20]` asks for twenty cards of one colour when only twelve are
/// printed, which [`force`] rejects.
pub fn filler(board: &Board, n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % board.n_colors) as u8).collect()
}

/// One distinct colour per seat, so a five-seat rig never over-subscribes a colour.
pub fn spread_hands(board: &Board, n: usize, count: u8) -> Vec<Vec<u8>> {
    (0..n)
        .map(|p| {
            (0..board.n_card_types)
                .map(|c| if c == p { count } else { 0 })
                .collect()
        })
        .collect()
}

/// Which parts of a position [`force`] overwrites. `None` leaves a field alone.
#[derive(Clone, Debug, Default)]
pub struct Rig {
    pub hands: Option<Vec<Vec<u8>>>,
    pub faceup: Option<Vec<i8>>,
    pub deck: Option<Vec<u8>>,
    pub trains: Option<Vec<u8>>,
    pub tickets: Option<Vec<u32>>,
    pub phase: Option<u8>,
    pub cur: Option<u8>,
}

/// Overwrite a state's guts, then place every unaccounted-for card so none is invented.
///
/// With an explicit `deck`, the leftovers become the discard pile; without one they become
/// the deck, in canonical order, and the discard is emptied. Either way the total is exact,
/// and a rig asking for more cards of a colour than the box holds panics rather than
/// silently truncating the position.
///
/// Every hidden card is marked blind-drawn (`certain` zero, `unknown` the hand size): the
/// weakest consistent information-set assignment, and enough to keep `validate()` happy.
pub fn force(state: &mut State, rig: Rig) {
    let board = state.board;
    let n = state.n_players();
    let k = board.n_card_types;

    if let Some(hands) = &rig.hands {
        for (p, counts) in hands.iter().enumerate() {
            assert_eq!(counts.len(), k, "a hand needs one count per card type");
            state.hand[p * k..(p + 1) * k].copy_from_slice(counts);
        }
    }
    for p in 0..n {
        let size: u32 = state.hand[p * k..(p + 1) * k]
            .iter()
            .map(|&c| u32::from(c))
            .sum();
        state.certain[p * k..(p + 1) * k].fill(0);
        state.unknown[p] = u8::try_from(size).expect("a hand fits in u8");
    }

    if let Some(faceup) = &rig.faceup {
        let mut slots = [EMPTY_SLOT; FACEUP_SLOTS];
        for (slot, &card) in slots.iter_mut().zip(faceup) {
            *slot = card;
        }
        state.faceup = slots;
    }
    if let Some(trains) = &rig.trains {
        state.trains[..trains.len()].copy_from_slice(trains);
    }
    if let Some(tickets) = &rig.tickets {
        state.tickets[..tickets.len()].copy_from_slice(tickets);
    }
    if let Some(phase) = rig.phase {
        state.phase = phase;
    }
    if let Some(cur) = rig.cur {
        state.cur = cur;
    }

    let mut leftover: Vec<i32> = board
        .deck_composition_counts
        .iter()
        .map(|&c| i32::from(c))
        .collect();
    for (c, count) in leftover.iter_mut().enumerate() {
        for p in 0..n {
            *count -= i32::from(state.hand[p * k + c]);
        }
    }
    for &card in &state.faceup {
        if card != EMPTY_SLOT {
            leftover[card as usize] -= 1;
        }
    }
    let explicit = rig.deck.clone();
    for &card in explicit.iter().flatten() {
        leftover[usize::from(card)] -= 1;
    }
    assert!(
        leftover.iter().all(|&c| c >= 0),
        "rigged position uses more cards than exist: {leftover:?}"
    );

    let deck: Vec<u8> = match explicit {
        Some(deck) => {
            for (c, &count) in leftover.iter().enumerate() {
                state.discard[c] = count as u8;
            }
            state.discard_total = leftover.iter().sum::<i32>() as u16;
            deck
        }
        None => {
            state.discard[..k].fill(0);
            state.discard_total = 0;
            leftover
                .iter()
                .enumerate()
                .flat_map(|(c, &count)| std::iter::repeat_n(c as u8, count as usize))
                .collect()
        }
    };
    state.deck[..deck.len()].copy_from_slice(&deck);
    state.deck_len = deck.len() as u16;
    state.deck_pos = 0;
}

/// Give `segment` to `player` the way a claim would -- owner and union-find -- without
/// spending cards or trains. Pair with `trains` in a [`Rig`] when the count matters.
pub fn give(state: &mut State, player: u8, segment: usize) {
    let board = state.board;
    state.seg_owner[segment] = player;
    let base = usize::from(player) * ttr_core::board::MAX_CITIES;
    dsu_union(
        &mut state.dsu[base..base + board.n_cities],
        board.seg_a[segment],
        board.seg_b[segment],
    );
    state.board_version += 1;
}

/// Play out the opening ticket choices with the lowest legal keep, reaching MAIN.
pub fn to_main(state: &mut State) {
    while state.phase != PHASE_MAIN {
        let action = state.legal_actions()[0];
        state.step(action).expect("the lowest keep is legal");
    }
}

/// Step randomly until `predicate` holds. Returns whether it ever did.
pub fn play_until(
    state: &mut State,
    predicate: impl Fn(&State) -> bool,
    rng: &mut Pcg32,
    limit: usize,
) -> bool {
    for _ in 0..limit {
        if predicate(state) {
            return true;
        }
        if state.is_terminal() {
            return false;
        }
        let action = state.sample_legal(rng).expect("a non-terminal state");
        state.step(action).expect("a sampled action is legal");
    }
    false
}

/// Play a game to the end with uniformly random legal actions from `rng`.
pub fn play_out(state: &mut State, rng: &mut Pcg32) {
    while !state.is_terminal() {
        let action = state.sample_legal(rng).expect("a non-terminal state");
        state.step(action).expect("a sampled action is legal");
    }
}
