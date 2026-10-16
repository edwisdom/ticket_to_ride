//! Replays: bitwise, or loudly wrong. Reproducibility level L0.
//!
//! The port of `tests/unit/engine/test_replay.py`. Every rejection path here corresponds to
//! a way a replay could silently become a *different* game -- a different board, different
//! rules, a different PRNG. Those are the failures worth paying bytes for.
//!
//! The last section has no Python counterpart: it pins where `ttr_core::replay` is stricter
//! than replay.py, which let some malformed inputs escape as other exceptions.

mod common;

use std::fmt::Debug;

use common::{all_configs, game, play_out, policy};
use ttr_core::config::RuleConfig;
use ttr_core::replay::{MAGIC, Replay, ReplayError, decode, encode, pack, record, replay, unpack};
use ttr_core::scoring::final_scores;
use ttr_core::state::Game;

/// A random game on `seed`'s policy stream, recorded.
fn play(map: &str, n_players: usize, seed: u64) -> Replay {
    let mut state = game(map, n_players).new_initial_state(seed);
    play_out(&mut state, &mut policy(seed));
    record(&state).expect("a finished game with history")
}

/// The default: USA, two seats, seed 1.
fn usa() -> Replay {
    play("usa", 2, 1)
}

/// Assert `result` failed with a message containing `needle`, and hand the error back so
/// the caller can also check which check fired.
fn rejected<T: Debug>(result: Result<T, ReplayError>, needle: &str) -> ReplayError {
    let err = result.expect_err(&format!("expected a rejection mentioning {needle:?}"));
    assert!(
        err.to_string().contains(needle),
        "{needle:?} is not in {:?}",
        err.to_string()
    );
    err
}

// ---------------------------------------------------------------------------
// The core promise
// ---------------------------------------------------------------------------

/// Every configuration, eight seeds each, through the wire format and back.
#[test]
fn a_seed_replays_bitwise() {
    for (map, n) in all_configs() {
        for seed in 0..8 {
            let rec = play(map, n, seed);
            let wire = decode(&encode(&rec).unwrap()).unwrap();
            let mut state =
                replay(&wire, None).unwrap_or_else(|e| panic!("{map} {n}P seed {seed}: {e}"));
            assert_eq!(state.state_hash(), rec.final_hash, "{map} {n}P seed {seed}");
            assert_eq!(
                final_scores(&mut state),
                rec.final_scores,
                "{map} {n}P seed {seed}"
            );
            assert_eq!(state.history, rec.actions, "{map} {n}P seed {seed}");
        }
    }
}

#[test]
fn round_trip_through_the_wire_format_is_exact() {
    let rec = usa();
    assert_eq!(decode(&encode(&rec).unwrap()).unwrap(), rec);
}

/// The budget the plan is sized against: ~150 MB per million games with zstd.
#[test]
fn a_replay_is_around_half_a_kilobyte() {
    let rec = usa();
    let blob = encode(&rec).unwrap();
    assert!(200 < blob.len() && blob.len() < 700, "{}", blob.len());
    // Header, name, body, scores, actions -- and nothing else.
    assert_eq!(
        blob.len(),
        6 + "usa".len() + 50 + 2 * usize::from(rec.n_players) + 2 * rec.n_actions()
    );
}

#[test]
fn replaying_leaves_the_history_intact() {
    let rec = usa();
    let state = replay(&rec, None).unwrap();
    assert_eq!(state.history, rec.actions);
}

// ---------------------------------------------------------------------------
// Everything that should fail loudly
// ---------------------------------------------------------------------------

#[test]
fn a_board_edit_invalidates_old_replays() {
    let rec = usa();
    let tampered = Replay {
        data_hash: [0; 16],
        ..rec
    };
    let err = rejected(replay(&tampered, None), "has changed since this replay");
    assert!(matches!(err, ReplayError::DataHash { .. }), "{err:?}");
}

#[test]
fn a_rule_change_invalidates_old_replays() {
    let rec = usa();
    let tampered = Replay {
        rules_hash: [0; 16],
        ..rec
    };
    let err = rejected(replay(&tampered, None), "rules have changed");
    assert!(matches!(err, ReplayError::RulesHash { .. }), "{err:?}");
}

/// The PRNG or the draw procedure changed; the actions no longer mean the same game.
#[test]
fn a_contract_bump_invalidates_old_replays() {
    let rec = usa();
    let tampered = Replay {
        contract_version: rec.contract_version + 1,
        ..rec
    };
    let err = rejected(replay(&tampered, None), "contract version");
    assert!(
        matches!(err, ReplayError::ContractVersion { .. }),
        "{err:?}"
    );
}

#[test]
fn a_diverged_replay_is_caught_by_the_final_hash() {
    let rec = usa();
    let tampered = Replay {
        final_hash: rec.final_hash ^ 1,
        ..rec
    };
    let err = rejected(replay(&tampered, None), "diverged");
    assert!(matches!(err, ReplayError::FinalHash { .. }), "{err:?}");
}

/// A second, independent check: the state hash can match while scoring changed.
///
/// `state_hash()` covers the position, not the interpretation of it -- a bug in ticket
/// settlement or the longest-trail search would leave every hash intact.
#[test]
fn a_diverged_replay_is_caught_by_the_scores() {
    let rec = usa();
    let mut wrong = rec.final_scores.clone();
    wrong[0] += 1;
    let tampered = Replay {
        final_scores: wrong,
        ..rec
    };
    let err = rejected(replay(&tampered, None), "scores");
    assert!(matches!(err, ReplayError::FinalScores { .. }), "{err:?}");
}

#[test]
fn a_truncated_action_list_ends_before_the_game_does() {
    let rec = usa();
    let short = rec.actions[..rec.n_actions() - 4].to_vec();
    let tampered = Replay {
        actions: short,
        ..rec
    };
    let err = rejected(replay(&tampered, None), "ended before the game");
    assert_eq!(err, ReplayError::EndedEarly);
}

#[test]
fn a_replay_longer_than_the_game_is_rejected() {
    let rec = usa();
    let mut padded = rec.actions.clone();
    padded.push(*rec.actions.last().unwrap());
    let tampered = Replay {
        actions: padded,
        ..rec
    };
    let err = rejected(replay(&tampered, None), "longer than the game");
    assert!(matches!(err, ReplayError::TooLong { .. }), "{err:?}");
}

#[test]
fn an_unfinished_game_cannot_be_recorded() {
    let state = game("usa", 2).new_initial_state(1);
    let err = rejected(record(&state), "only a finished game");
    assert_eq!(err, ReplayError::Unfinished);
}

#[test]
fn a_game_played_without_history_cannot_be_recorded() {
    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new("usa", 2).unwrap()
    };
    let mut state = Game::new(cfg).unwrap().new_initial_state(1);
    play_out(&mut state, &mut policy(1));
    let err = rejected(record(&state), "track_history off");
    assert_eq!(err, ReplayError::NoHistory);
}

// ---------------------------------------------------------------------------
// Wire-format robustness
// ---------------------------------------------------------------------------

#[test]
fn a_foreign_file_is_rejected() {
    let mut foreign = b"PNG\x00\x02\x03".to_vec();
    foreign.extend([0u8; 80]);
    let err = rejected(decode(&foreign), "not a replay");
    assert!(matches!(err, ReplayError::NotAReplay(_)), "{err:?}");
}

#[test]
fn a_truncated_file_is_rejected() {
    let err = rejected(decode(MAGIC), "truncated");
    assert_eq!(err, ReplayError::Truncated);
}

#[test]
fn trailing_bytes_are_rejected() {
    let mut blob = encode(&usa()).unwrap();
    blob.push(0);
    let err = rejected(decode(&blob), "trailing bytes");
    assert_eq!(err, ReplayError::TrailingBytes(1));
}

/// Non-default rules have to be supplied, and are checked against `rules_hash`.
#[test]
fn replay_accepts_an_explicit_matching_config() {
    let cfg = RuleConfig {
        turn_cap: 500,
        ..RuleConfig::new("usa", 2).unwrap()
    };
    let mut state = Game::new(cfg.clone()).unwrap().new_initial_state(3);
    play_out(&mut state, &mut policy(3));
    let rec = record(&state).unwrap();

    // The default config: a different turn_cap, so a different rules_hash.
    rejected(replay(&rec, None), "rules have changed");
    assert_eq!(
        replay(&rec, Some(&cfg)).unwrap().state_hash(),
        rec.final_hash
    );
}

// ---------------------------------------------------------------------------
// Rust only: where this is stricter than replay.py
// ---------------------------------------------------------------------------

/// Python let the engine's `IllegalAction` escape `replay()`; here it is a replay error
/// naming the step, because a refused recorded action is a divergence like any other.
#[test]
fn a_recorded_action_the_engine_refuses_is_a_replay_error() {
    let rec = usa();
    let mut actions = rec.actions.clone();
    actions[10] = u16::MAX;
    let tampered = Replay { actions, ..rec };
    let err = rejected(replay(&tampered, None), "was refused");
    assert!(
        matches!(
            err,
            ReplayError::Illegal {
                index: 10,
                action: u16::MAX,
                ..
            }
        ),
        "{err:?}"
    );
}

/// Python raised `struct.error` for a record cut short past its header, and read a short
/// corpus prefix as a smaller number. Every truncation is a `ReplayError` here, and never a
/// panic.
#[test]
fn every_truncation_is_a_replay_error() {
    let rec = usa();
    let blob = encode(&rec).unwrap();
    for len in 0..blob.len() {
        assert_eq!(
            decode(&blob[..len]),
            Err(ReplayError::Truncated),
            "{len} bytes"
        );
    }
    let corpus = pack(&[rec.clone(), rec]).unwrap();
    for len in 0..corpus.len() {
        assert!(unpack(&corpus[..len]).is_err(), "{len} bytes");
    }
}

#[test]
fn trailing_bytes_after_a_corpus_are_rejected() {
    let mut corpus = pack(&[usa()]).unwrap();
    corpus.extend([0, 0]);
    let err = rejected(unpack(&corpus), "trailing bytes in pack");
    assert_eq!(err, ReplayError::TrailingPackBytes(2));
}

/// The two records the wire format cannot hold: Python refused the first itself and let
/// `struct.error` refuse the second.
#[test]
fn a_record_the_wire_format_cannot_hold_is_refused() {
    let rec = usa();
    let long_name = Replay {
        map_name: "x".repeat(256),
        ..rec.clone()
    };
    let err = rejected(encode(&long_name), "too long");
    assert!(matches!(err, ReplayError::NameTooLong(_)), "{err:?}");

    let extra_score = Replay {
        final_scores: vec![0; 3],
        ..rec
    };
    let err = rejected(encode(&extra_score), "final scores for 2 seats");
    assert!(matches!(err, ReplayError::ScoreCount { .. }), "{err:?}");
}

/// Python let `RuleConfig`'s own exception escape for a map or seat count it cannot build.
#[test]
fn an_unbuildable_configuration_is_a_replay_error() {
    let rec = usa();
    let elsewhere = Replay {
        map_name: "europe".into(),
        ..rec.clone()
    };
    let err = rejected(replay(&elsewhere, None), "unknown map");
    assert!(matches!(err, ReplayError::Config(_)), "{err:?}");

    let crowded = Replay {
        n_players: 9,
        ..rec
    };
    let err = rejected(replay(&crowded, None), "players");
    assert!(matches!(err, ReplayError::Config(_)), "{err:?}");
}
