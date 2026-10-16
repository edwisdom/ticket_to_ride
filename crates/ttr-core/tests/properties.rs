//! Properties that must hold in every game, on every map, at every player count.
//!
//! The port of `tests/property/test_engine_properties.py`, one test per Python test and
//! named after it without the `test_` prefix. The rules tests check named positions; these
//! check that nothing *else* went wrong on the way there, by sweeping seeds and asserting
//! invariants after every single step. Where a property is about the shape of the action
//! space rather than about play, `proptest` drives it instead of a seed sweep (Hypothesis,
//! in the Python suite).
//!
//! These are the Python suite's games, not merely games like them: the same seeds, the same
//! `"policy"` stream, and `sample_legal` draws from the same unsorted emission order. So a
//! seed named in a failure here is the game the Python test played, and measurements taken
//! on those games still hold -- the flush cascade cap first fires on 5P seed 15 at turn
//! 243, exactly as `state.rs` records.

mod common;

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use common::{all_configs, game, play_out, play_until, policy};
use proptest::prelude::*;
use ttr_core::board::NO_SIBLING;
use ttr_core::config::{CLOSED, EMPTY_SLOT, PHASE_TERMINAL};
use ttr_core::rng::{Part, Pcg32, stream};
use ttr_core::scoring::{final_scores, returns, score_breakdown, winners};
use ttr_core::state::State;

/// Enough seeds to exercise reshuffles and flushes without slowing the fast suite.
const SWEEP: u64 = 25;

/// One configuration's games, ready to be played out: `(seed, state, policy stream)`.
fn games(
    map: &str,
    n_players: usize,
    seeds: Range<u64>,
) -> impl Iterator<Item = (u64, State, Pcg32)> {
    let game = game(map, n_players);
    seeds.map(move |seed| (seed, game.new_initial_state(seed), policy(seed)))
}

/// Take one uniformly random legal step and return the action taken.
fn advance(state: &mut State, rng: &mut Pcg32) -> u16 {
    let action = state
        .sample_legal(rng)
        .unwrap_or_else(|| panic!("{state:?}: no legal action in a non-terminal state"));
    state
        .step(action)
        .unwrap_or_else(|e| panic!("{state:?}: sampled action {action} was rejected: {e}"));
    action
}

/// Where a sweep failed, for assertion messages.
fn at(map: &str, n_players: usize, seed: u64, state: &State) -> String {
    format!("{map} {n_players}P seed {seed}: {state:?}")
}

/// Everything a step could touch, hashed or not: the canonical image plus the fields the
/// hash deliberately leaves out (the union-find cache, the discard fast-count, the flush
/// flag, the board version) and the history.
fn fingerprint(state: &State) -> (Vec<u8>, Vec<u8>, u16, bool, u32, Vec<u16>) {
    (
        state.serialize(false),
        state.dsu.to_vec(),
        state.discard_total,
        state.flush_capped,
        state.board_version,
        state.history.clone(),
    )
}

// ---------------------------------------------------------------------------
// 1-5. Conservation and legality, asserted after every step
// ---------------------------------------------------------------------------

/// Cards, trains, tickets and the information-set bookkeeping, all the way down.
///
/// `validate()` is the single place these live, so one sweep covers conservation of the
/// deck, `certain + unknown == hand`, per-seat train accounting, ticket conservation across
/// hands, deck and offer, and the locomotive-flush assertion.
#[test]
fn every_invariant_holds_after_every_step() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..SWEEP) {
            if let Err(e) = state.validate() {
                panic!("{} at setup: {e}", at(map, n, seed, &state));
            }
            while !state.is_terminal() {
                let action = advance(&mut state, &mut rng);
                if let Err(e) = state.validate() {
                    panic!("{} after {action}: {e}", at(map, n, seed, &state));
                }
            }
        }
    }
}

#[test]
fn sampled_actions_are_always_legal() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..SWEEP) {
            while !state.is_terminal() {
                let legal = state.legal_actions();
                let action = state.sample_legal(&mut rng).expect("a non-terminal state");
                assert!(
                    legal.contains(&action),
                    "{}: sample_legal returned {action}, not in legal_actions()",
                    at(map, n, seed, &state)
                );
                state.step(action).expect("a legal action");
            }
        }
    }
}

/// Two engines' lists must compare directly, which needs a canonical order.
#[test]
fn legal_actions_are_sorted_and_unique() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..SWEEP) {
            while !state.is_terminal() {
                let legal = state.legal_actions();
                // Strictly ascending is sorted and unique in one comparison.
                assert!(
                    legal.windows(2).all(|w| w[0] < w[1]),
                    "{}: {legal:?}",
                    at(map, n, seed, &state)
                );
                advance(&mut state, &mut rng);
            }
        }
    }
}

/// The engine must never construct a node with nothing to do -- PASS exists for this.
#[test]
fn a_non_terminal_state_always_has_a_legal_action() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..SWEEP) {
            while !state.is_terminal() {
                assert!(
                    !state.legal_actions().is_empty(),
                    "{}: dead end",
                    at(map, n, seed, &state)
                );
                advance(&mut state, &mut rng);
            }
        }
    }
}

/// The mask is what the policy actually reads, so it must say exactly what the list says,
/// and a reused buffer must be cleared rather than accumulate the previous step's bits.
#[test]
fn legal_action_mask_agrees_with_the_list() {
    for (map, n) in all_configs() {
        let size = usize::from(game(map, n).space.n);
        for (seed, mut state, mut rng) in games(map, n, 0..5) {
            // Starts dirty (Python's started zeroed), so a mask that fails to clear its
            // buffer is caught on the first step rather than only from the second.
            let mut reused = vec![1u8; size];
            while !state.is_terminal() {
                let legal = state.legal_actions();
                let mut fresh = vec![0u8; size];
                state.legal_action_mask(&mut fresh);
                state.legal_action_mask(&mut reused);
                let here = at(map, n, seed, &state);
                assert_eq!(fresh, reused, "{here}: the reused buffer was not cleared");
                assert!(
                    fresh.iter().all(|&b| b <= 1),
                    "{here}: a mask entry is not 0/1"
                );
                let set: Vec<u16> = (0..size as u16)
                    .filter(|&a| fresh[usize::from(a)] == 1)
                    .collect();
                assert_eq!(set, legal, "{here}");
                advance(&mut state, &mut rng);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6-8. Board-level rules that must never be violated by any sequence of play
// ---------------------------------------------------------------------------

#[test]
fn no_player_ever_owns_both_tracks_of_a_double() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..SWEEP) {
            play_out(&mut state, &mut rng);
            let board = state.board;
            for (segment, &sibling) in board.sibling.iter().enumerate() {
                if sibling == NO_SIBLING {
                    continue;
                }
                let owner = state.seg_owner[segment];
                let twin = state.seg_owner[usize::from(sibling)];
                if owner < CLOSED && twin < CLOSED {
                    assert_ne!(
                        owner,
                        twin,
                        "{}: one seat owns both tracks of {}",
                        at(map, n, seed, &state),
                        board.segment_name(segment)
                    );
                }
            }
        }
    }
}

/// 2-3P closes a claimed double's sibling to everyone; 4-5P must never close anything, or
/// an agent trained there silently learns to avoid double routes.
#[test]
fn closed_siblings_appear_only_in_two_and_three_player_games() {
    for (map, n) in all_configs() {
        let locked = game(map, n).rules.doubles_locked;
        for (seed, mut state, mut rng) in games(map, n, 0..10) {
            play_out(&mut state, &mut rng);
            let board = state.board;
            let owners = &state.seg_owner[..board.n_segments];
            let closed = owners.iter().filter(|&&o| o == CLOSED).count();
            if locked {
                let claimed_doubles = (0..board.n_segments)
                    .filter(|&s| owners[s] < CLOSED && board.sibling[s] != NO_SIBLING)
                    .count();
                assert_eq!(closed, claimed_doubles, "{}", at(map, n, seed, &state));
            } else {
                assert_eq!(
                    closed,
                    0,
                    "{}: 4-5P must never close a sibling to everyone",
                    at(map, n, seed, &state)
                );
            }
        }
    }
}

#[test]
fn games_always_terminate_well_inside_the_cap() {
    for (map, n) in all_configs() {
        let cap = game(map, n).cfg.turn_cap;
        for (seed, mut state, mut rng) in games(map, n, 0..SWEEP) {
            play_out(&mut state, &mut rng);
            assert_eq!(state.phase, PHASE_TERMINAL);
            assert!(
                state.turn < cap,
                "{}: hit the belt-and-braces turn cap",
                at(map, n, seed, &state)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 9-12. Cloning and hashing
// ---------------------------------------------------------------------------

/// A clone shares nothing with its source: stepping either leaves the other intact.
#[test]
fn clone_is_fully_independent() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..5) {
            while !state.is_terminal() {
                let mut copy = state.clone();
                assert_eq!(copy.state_hash(), state.state_hash());
                let mut other = stream(1, &[Part::Str("other")]);
                let action = copy.sample_legal(&mut other).expect("the clone is live");
                copy.step(action).expect("a sampled action is legal");
                assert!(
                    state.state_hash() != copy.state_hash() || state.is_terminal(),
                    "{}: stepping the clone moved the original",
                    at(map, n, seed, &state)
                );
                advance(&mut state, &mut rng);
                // Mutating the original must not have touched the earlier clone's arrays.
                if let Err(e) = copy.validate() {
                    panic!("{}: the clone broke: {e}", at(map, n, seed, &state));
                }
            }
        }
    }
}

/// `clone_into` exists so search never allocates; it must be exactly `clone`.
///
/// Compared field for field here, not only by hash: "exactly" includes the union-find,
/// the counters the hash leaves out, and the history.
#[test]
fn clone_into_matches_clone() {
    for (map, n) in all_configs() {
        let mut arena = game(map, n).new_initial_state(0);
        for (seed, mut state, mut rng) in games(map, n, 0..5) {
            while !state.is_terminal() {
                state.clone_into(&mut arena);
                let here = at(map, n, seed, &state);
                assert_eq!(arena.state_hash(), state.state_hash(), "{here}");
                assert_eq!(arena.legal_actions(), state.legal_actions(), "{here}");
                assert_eq!(fingerprint(&arena), fingerprint(&state), "{here}");
                advance(&mut state, &mut rng);
            }
        }
    }
}

/// Two states differing only in which order dealt cards came out are one position.
#[test]
fn position_hash_ignores_the_rng_and_the_dealt_deck() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..5) {
            while !state.is_terminal() {
                let mut twin = state.clone();
                twin.rng = Pcg32::new(twin.rng.state ^ 0xDEAD_BEEF, twin.rng.inc);
                let cursor = usize::from(twin.deck_pos);
                twin.deck[..cursor].reverse();
                let here = at(map, n, seed, &state);
                assert_eq!(twin.position_hash(), state.position_hash(), "{here}");
                assert_ne!(twin.state_hash(), state.state_hash(), "{here}");
                advance(&mut state, &mut rng);
            }
        }
    }
}

#[test]
fn state_hash_separates_states_that_really_differ() {
    for (map, n) in all_configs() {
        let mut seen: HashMap<u64, u16> = HashMap::new();
        for (seed, mut state, mut rng) in games(map, n, 0..8) {
            while !state.is_terminal() {
                let turn = *seen.entry(state.state_hash()).or_insert(state.turn);
                assert_eq!(
                    turn,
                    state.turn,
                    "{}: two different positions collided",
                    at(map, n, seed, &state)
                );
                advance(&mut state, &mut rng);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 13-15. Determinism -- the property everything downstream depends on
// ---------------------------------------------------------------------------

#[test]
fn the_same_seed_replays_bitwise() {
    for (map, n) in all_configs() {
        for seed in 0..10 {
            let game = game(map, n);
            let mut first = game.new_initial_state(seed);
            let mut second = game.new_initial_state(seed);
            let (mut rng_a, mut rng_b) = (policy(seed), policy(seed));
            while !first.is_terminal() {
                let here = at(map, n, seed, &first);
                assert_eq!(first.state_hash(), second.state_hash(), "{here}");
                let action = first.sample_legal(&mut rng_a).expect("live");
                assert_eq!(Some(action), second.sample_legal(&mut rng_b), "{here}");
                first.step(action).expect("legal");
                second.step(action).expect("legal");
            }
            assert!(second.is_terminal());
            assert_eq!(first.state_hash(), second.state_hash());
            assert_eq!(final_scores(&mut first), final_scores(&mut second));
        }
    }
}

#[test]
fn different_seeds_deal_different_games() {
    for (map, n) in all_configs() {
        let game = game(map, n);
        let digests: HashSet<u64> = (0..50)
            .map(|seed| game.new_initial_state(seed).state_hash())
            .collect();
        assert_eq!(digests.len(), 50, "{map} {n}P");
    }
}

/// The precondition for paired evaluation.
///
/// Two games on the same seed whose agents act differently must still hold the *same*
/// shuffle. Sampling the deck lazily from card counts would break this silently, with no
/// error anywhere -- the variance-reduction scheme would simply stop working.
#[test]
fn environment_randomness_does_not_depend_on_agent_behaviour() {
    let game = game("usa", 2);
    let mut a = game.new_initial_state(4);
    let mut b = game.new_initial_state(4);
    let deck = |s: &State| s.deck[..usize::from(s.deck_len)].to_vec();
    assert_eq!(deck(&a), deck(&b));
    for _ in 0..12 {
        if a.is_terminal() || b.is_terminal() {
            break;
        }
        let first = a.legal_actions()[0];
        a.step(first).expect("legal");
        let last = *b.legal_actions().last().expect("live");
        b.step(last).expect("legal");
    }
    // Neither game reshuffled, so both still hold the permutation the seed produced.
    assert_eq!(
        deck(&a),
        deck(&b),
        "agent behaviour perturbed the environment's randomness"
    );
}

// ---------------------------------------------------------------------------
// 16-17. Scoring
// ---------------------------------------------------------------------------

#[test]
fn final_scores_match_the_breakdown() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..10) {
            play_out(&mut state, &mut rng);
            let here = at(map, n, seed, &state);
            let breakdowns = score_breakdown(&mut state);
            let totals: Vec<i16> = breakdowns.iter().map(|b| b.total).collect();
            assert_eq!(final_scores(&mut state), totals, "{here}");
            let board = state.board;
            for (p, b) in breakdowns.iter().enumerate() {
                assert_eq!(
                    b.total,
                    b.routes + b.ticket_points + b.longest_bonus,
                    "{here}"
                );
                assert_eq!(
                    u32::from(b.tickets_made) + u32::from(b.tickets_missed),
                    state.tickets[p].count_ones(),
                    "{here} seat {p}"
                );
                let routes: i16 = (0..board.n_segments)
                    .filter(|&s| usize::from(state.seg_owner[s]) == p)
                    .map(|s| i16::from(board.route_points[usize::from(board.seg_len[s])]))
                    .sum();
                assert_eq!(b.routes, routes, "{here} seat {p}");
            }
        }
    }
}

#[test]
fn returns_are_constant_sum_and_agree_with_the_winner() {
    for (map, n) in all_configs() {
        for (seed, mut state, mut rng) in games(map, n, 0..10) {
            play_out(&mut state, &mut rng);
            let here = at(map, n, seed, &state);
            let r = returns(&mut state);
            let total: f64 = r.iter().sum();
            assert!(
                total.abs() < 1e-9,
                "{here}: returns must be constant-sum, got {r:?}"
            );
            // The first seat holding the best return, as Python's `max(key=...)` picks it.
            let best = (0..n).fold(0, |best, p| if r[p] > r[best] { p } else { best });
            assert!(winners(&mut state).contains(&best), "{here}: {r:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// 18-19. The action space itself
// ---------------------------------------------------------------------------

/// The strongest statement the engine makes: an illegal action cannot get through.
///
/// A policy that leaks one past its mask is the classic silent RL bug, so `step()`
/// re-derives the preconditions of whatever it is handed rather than trusting the caller.
#[test]
fn step_rejects_every_illegal_action() {
    for (map, n) in all_configs() {
        let size = game(map, n).space.n;
        let stride = usize::from((size / 40).max(1));
        for (seed, mut state, mut rng) in games(map, n, 0..3) {
            while !state.is_terminal() {
                let legal: HashSet<u16> = state.legal_actions().into_iter().collect();
                for action in (0..size).step_by(stride) {
                    if legal.contains(&action) {
                        continue;
                    }
                    let mut probe = state.clone();
                    assert!(
                        probe.step(action).is_err(),
                        "{}: illegal {} got through",
                        at(map, n, seed, &state),
                        state.space.to_string(state.board, action)
                    );
                }
                advance(&mut state, &mut rng);
            }
        }
    }
}

/// Every step path validates before it mutates; history included, or a replay of a game
/// that survived a rejected action diverges.
///
/// Python compared the hash and the history; this compares everything the hash leaves out
/// as well, since "untouched" means untouched.
#[test]
fn a_rejected_action_leaves_the_state_untouched() {
    for (map, n) in all_configs() {
        let size = game(map, n).space.n;
        let stride = usize::from((size / 20).max(1));
        for (seed, mut state, mut rng) in games(map, n, 0..3) {
            while !state.is_terminal() {
                let legal: HashSet<u16> = state.legal_actions().into_iter().collect();
                let before = fingerprint(&state);
                let hash = state.state_hash();
                for action in (0..size).step_by(stride) {
                    if legal.contains(&action) {
                        continue;
                    }
                    let here = at(map, n, seed, &state);
                    assert!(state.step(action).is_err(), "{here}: {action} accepted");
                    assert_eq!(state.state_hash(), hash, "{here}: {action} moved the hash");
                    assert!(
                        fingerprint(&state) == before,
                        "{here}: rejected {action} still changed the state or the history"
                    );
                }
                advance(&mut state, &mut rng);
            }
        }
    }
}

/// Uniformity: over many streams every legal action must come up.
#[test]
fn sample_legal_covers_the_whole_legal_set() {
    let mut state = game("usa", 2).new_initial_state(1);
    while state.legal_actions().len() < 10 {
        let action = state.legal_actions()[0];
        state.step(action).expect("legal");
    }
    let legal = state.legal_actions();
    let mut drawn: HashMap<u16, u32> = HashMap::new();
    for i in 0..200 * legal.len() as u64 {
        let action = state
            .sample_legal(&mut stream(i, &[Part::Str("uniform")]))
            .expect("live");
        *drawn.entry(action).or_default() += 1;
    }
    let mut keys: Vec<u16> = drawn.keys().copied().collect();
    keys.sort_unstable();
    assert_eq!(keys, legal);
    let expected = 200.0;
    assert!(
        drawn
            .values()
            .all(|&n| expected * 0.6 < f64::from(n) && f64::from(n) < expected * 1.4),
        "{drawn:?}"
    );
}

/// There is nothing to sample at the end of a game, and stepping it is refused too.
#[test]
fn sample_legal_refuses_a_terminal_state() {
    let mut state = game("usa", 2).new_initial_state(1);
    let mut rng = policy(1);
    play_out(&mut state, &mut rng);
    assert_eq!(state.sample_legal(&mut rng), None);
    let err = state
        .step(0)
        .expect_err("a terminal state accepted an action");
    assert!(err.0.contains("over"), "{err}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]

    /// Every id at or past the end of the space is refused with a message naming the range.
    ///
    /// Python drew from -5000..=5000. Action ids are `u16` here, so the negative half is
    /// unrepresentable by construction; this draws from all of `u16` instead.
    #[test]
    fn out_of_range_actions_are_rejected(action in any::<u16>()) {
        let mut state = game("usa", 2).new_initial_state(1);
        if action >= state.space.n {
            let err = state.step(action).expect_err("an out-of-range action was accepted");
            prop_assert!(err.0.contains("outside"), "{}", err);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(120))]

    #[test]
    fn action_names_round_trip(action in 0u16..=914) {
        let game = game("usa", 2);
        let name = game.action_to_string(action);
        prop_assert!(!name.is_empty());
        if action < game.space.claim_end {
            let (segment, pay) = game.space.decode_claim(action);
            prop_assert_eq!(game.space.claim(segment, pay), action);
            prop_assert!(name.starts_with("CLAIM"), "{}", name);
        }
    }
}

// ---------------------------------------------------------------------------
// Reshuffles are rare; make sure at least one seed actually reaches one
// ---------------------------------------------------------------------------

/// A guard on the guard: an untested branch that never runs proves nothing.
///
/// Python watched for the deck object being replaced. The Rust deck is a fixed array, so
/// the equivalent is the live deck no longer being the permutation dealt at construction.
#[test]
fn the_sweep_actually_exercises_a_reshuffle() {
    let game = game("usa", 2);
    for seed in 0..SWEEP {
        let mut state = game.new_initial_state(seed);
        let (dealt, dealt_len) = (state.deck, state.deck_len);
        let reshuffled = move |s: &State| {
            s.deck_len != dealt_len
                || s.deck[..usize::from(dealt_len)] != dealt[..usize::from(dealt_len)]
        };
        if play_until(&mut state, reshuffled, &mut policy(seed), 100_000) {
            return;
        }
    }
    panic!("no seed below {SWEEP} reshuffled; the reshuffle path is untested");
}

/// The flush is a rules edge case that random play must reach, or its guard is untested.
///
/// Python's detector was looser than it read: it compared face-up slots against `FREE`
/// (the segment-owner sentinel, which a face-up slot never holds) where `EMPTY_SLOT` was
/// meant, and it accepted any step that grew the discard by five, which a five- or six-car
/// claim does on its own. This port uses the intended sentinel and ignores claims: only a
/// flush puts cards in the discard on any other step, so what is left is a real flush.
#[test]
fn the_sweep_actually_exercises_a_locomotive_flush() {
    let game = game("usa", 2);
    let loco = game.board.locomotive as i8;
    for seed in 0..200 {
        let mut state = game.new_initial_state(seed);
        let mut rng = policy(seed);
        while !state.is_terminal() {
            let before = i32::from(state.discard_total);
            let faceup_before = state.faceup;
            let action = advance(&mut state, &mut rng);
            let grew = i32::from(state.discard_total) - before;
            let full = !faceup_before.contains(&EMPTY_SLOT);
            let locos = faceup_before.iter().filter(|&&c| c == loco).count();
            if !state.space.is_claim(action) && grew >= 5 && full && locos >= 2 {
                return;
            }
        }
    }
    panic!("no seed below 200 triggered a locomotive flush");
}

/// The cap is load-bearing, not belt and braces.
///
/// Once most of the deck is in players' hands, the available pool can be small and
/// locomotive-heavy, and every reflush deals three more of them. Found by strengthening
/// `validate()`'s flush assertion, which until then passed vacuously.
#[test]
fn the_flush_cascade_cap_actually_fires_in_a_real_game() {
    let game = game("usa", 5);
    let loco = game.board.locomotive as i8;
    for seed in 0..SWEEP {
        let mut state = game.new_initial_state(seed);
        let mut rng = policy(seed);
        while !state.is_terminal() {
            advance(&mut state, &mut rng);
            if state.flush_capped {
                let locos = state.faceup.iter().filter(|&&c| c == loco).count();
                assert!(
                    locos >= 3,
                    "seed {seed}: a cascade bail-out with {locos} locomotives"
                );
                return;
            }
        }
    }
    panic!("no seed below {SWEEP} reached a cascade bail-out; the branch is untested");
}

/// It records how the last flush ended, not where the game is.
#[test]
fn flush_capped_is_transient_and_never_hashed() {
    let mut state = game("usa", 2).new_initial_state(1);
    let before = state.state_hash();
    state.flush_capped = !state.flush_capped;
    assert_eq!(state.state_hash(), before);
    assert_eq!(state.clone().flush_capped, state.flush_capped);
}
