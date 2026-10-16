//! The golden replay corpus, and the wider determinism sweeps.
//!
//! The port of `tests/integration/test_golden_replays.py`. `tests/golden/replays.bin` holds
//! 84 finished games -- random and greedy, across every map and seat count -- each carrying
//! the actions that produced it plus the state hash and scores it must produce again. The
//! Python engine wrote it, and it stays a frozen fixture once the Python engine is gone. It
//! is the strongest regression guard in the project: a rules change, a draw-order change, a
//! PRNG change or a scoring change shows up here as a concrete failing game rather than as a
//! subtly different win rate three weeks later.
//!
//! The heavy sweeps are `#[ignore]`d and run with `--release -- --ignored`. The corpus check
//! is not: it guards the property everything else depends on.
//!
//! Python's `test_the_corpus_is_not_stale` ran `tools/gen_replays.py --check`. Its port lives
//! with the Rust bin, in `crates/ttr-tools/src/bin/gen-replays.rs`: one test runs the same
//! check as `--check`, and another rebuilds the file byte for byte.

mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;

use common::{all_configs, game, play_out, policy};
use rayon::prelude::*;
use ttr_core::config::CLOSED;
use ttr_core::replay::{Replay, pack, record, replay, unpack};
use ttr_core::scoring::final_scores;

fn golden_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/replays.bin");
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn golden() -> Vec<Replay> {
    unpack(&golden_bytes()).expect("tests/golden/replays.bin parses")
}

fn label(rec: &Replay) -> String {
    format!("{} {}P seed {}", rec.map_name, rec.n_players, rec.seed)
}

#[test]
fn the_corpus_covers_every_configuration() {
    let golden = golden();
    let seen: BTreeSet<(String, usize)> = golden
        .iter()
        .map(|r| (r.map_name.clone(), usize::from(r.n_players)))
        .collect();
    let every: BTreeSet<(String, usize)> = all_configs()
        .into_iter()
        .map(|(map, n)| (map.to_string(), n))
        .collect();
    assert_eq!(seen, every);
    assert!(golden.len() >= 80, "{} games", golden.len());
}

#[test]
fn every_golden_replay_still_reproduces() {
    for rec in golden() {
        let mut state = replay(&rec, None).unwrap_or_else(|e| panic!("{}: {e}", label(&rec)));
        assert_eq!(state.state_hash(), rec.final_hash, "{}", label(&rec));
        assert_eq!(
            final_scores(&mut state),
            rec.final_scores,
            "{}",
            label(&rec)
        );
    }
}

/// A corpus that never triggers the endgame or a long route guards very little.
#[test]
fn the_corpus_reaches_the_interesting_states() {
    let mut longest = 0;
    let mut total_actions = 0;
    for rec in golden() {
        let state = replay(&rec, None).unwrap_or_else(|e| panic!("{}: {e}", label(&rec)));
        total_actions += rec.n_actions();
        let board = state.board;
        for segment in 0..board.n_segments {
            // Owned by a seat: neither free nor closed by the 2-3P double-route rule.
            if state.seg_owner[segment] < CLOSED {
                longest = longest.max(board.seg_len[segment]);
            }
        }
    }
    assert!(longest >= 5, "no long route is ever claimed in the corpus");
    assert!(total_actions > 5_000, "{total_actions} actions");
}

/// Rust only. The file was written by Python's `pack()`; reading it and packing it again
/// must give back the same bytes, or the two wire formats are not the same format.
#[test]
fn the_corpus_round_trips_byte_for_byte() {
    let bytes = golden_bytes();
    assert_eq!(pack(&unpack(&bytes).unwrap()).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// Wider sweeps
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow sweep (7k games); run with --release -- --ignored"]
fn a_thousand_seeds_replay_bitwise() {
    for (map, n) in all_configs() {
        let game = game(map, n);
        (0..1000u64).into_par_iter().for_each(|seed| {
            let mut state = game.new_initial_state(seed);
            play_out(&mut state, &mut policy(seed));
            let rec = record(&state).unwrap();
            let replayed =
                replay(&rec, None).unwrap_or_else(|e| panic!("{map} {n}P seed {seed}: {e}"));
            assert_eq!(
                replayed.state_hash(),
                rec.final_hash,
                "{map} {n}P seed {seed}"
            );
        });
    }
}

#[test]
#[ignore = "slow sweep (70k games); run with --release -- --ignored"]
fn ten_thousand_random_games_hold_every_invariant() {
    for (map, n) in all_configs() {
        let game = game(map, n);
        (0..10_000u64).into_par_iter().for_each(|seed| {
            let mut state = game.new_initial_state(seed);
            play_out(&mut state, &mut policy(seed));
            state
                .validate()
                .unwrap_or_else(|e| panic!("{map} {n}P seed {seed}: {e}"));
            assert_eq!(final_scores(&mut state).len(), n, "{map} {n}P seed {seed}");
        });
    }
}
