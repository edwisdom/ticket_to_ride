//! Full random games on every map and seat count, validated after every single step.
//!
//! This is the cheap check that runs before the differential harness is even involved: it
//! catches panics, hangs and conservation-law breaks in Rust alone, so those failures are
//! attributed to the engine rather than showing up later as a mysterious hash mismatch.
//!
//! It deliberately validates after *every* step rather than at the terminal. A leaked card
//! or a drifted information-set count is a fact about one specific step, and finding it at
//! the end tells you only that it happened somewhere in 150 of them.

use std::collections::HashSet;

use ttr_core::config::{PHASE_TERMINAL, RuleConfig};
use ttr_core::rng::{Part, stream};
use ttr_core::state::{Game, State};

fn play(map: &str, n_players: usize, seed: u64) -> State {
    let game = Game::new(RuleConfig::new(map, n_players).expect("valid config")).unwrap();
    let mut state = game.new_initial_state(seed);
    let mut rng = stream(seed, &[Part::Str("test"), Part::Str("policy")]);
    let mut scratch = Vec::new();

    state
        .validate()
        .unwrap_or_else(|e| panic!("{map} {n_players}P seed {seed} at setup: {e}"));
    let mut steps = 0;
    while !state.is_terminal() {
        let action = state
            .sample_legal_into(&mut rng, &mut scratch)
            .unwrap_or_else(|| panic!("{map} {n_players}P seed {seed}: no legal action"));
        state
            .step(action)
            .unwrap_or_else(|e| panic!("{map} {n_players}P seed {seed} step {steps}: {e}"));
        state.validate().unwrap_or_else(|e| {
            panic!("{map} {n_players}P seed {seed} after step {steps} ({action}): {e}")
        });
        steps += 1;
        assert!(
            steps < 20_000,
            "{map} {n_players}P seed {seed} did not terminate"
        );
    }
    state
}

#[test]
fn random_games_finish_and_conserve_everything() {
    for (map, seats) in [("usa", 2..=5), ("mini", 2..=4)] {
        for n in seats {
            for seed in 0..12u64 {
                let state = play(map, n, seed);
                assert_eq!(state.phase, PHASE_TERMINAL);
                // Every seat had a turn, or the game ended before anyone played.
                assert!(state.turn > 0, "{map} {n}P seed {seed} ended on turn 0");
            }
        }
    }
}

#[test]
fn a_rejected_action_leaves_the_state_untouched() {
    // Including the history: appending before validating would leave a phantom entry, and
    // the replay of that game would then diverge. This is the trap GOTCHAS records.
    let game = Game::new(RuleConfig::new("usa", 2).unwrap()).unwrap();
    let mut state = game.new_initial_state(7);
    let legal: HashSet<u16> = state.legal_actions().into_iter().collect();

    for action in 0..game.space.n {
        if legal.contains(&action) {
            continue;
        }
        let before = state.state_hash();
        let history_before = state.history.clone();
        let err = state.step(action);
        assert!(err.is_err(), "illegal action {action} was accepted");
        assert_eq!(
            state.state_hash(),
            before,
            "action {action} mutated the state"
        );
        assert_eq!(
            state.history, history_before,
            "action {action} left a phantom history entry"
        );
    }
}

#[test]
fn environment_randomness_does_not_depend_on_agent_behaviour() {
    // The property paired evaluation rests on: two games on one seed whose agents diverge
    // still realize the same shuffle. Lazy count-based dealing would break this with no
    // error anywhere, which is why the deck is a pre-materialized permutation.
    let game = Game::new(RuleConfig::new("usa", 2).unwrap()).unwrap();
    let a = game.new_initial_state(99);
    let b = game.new_initial_state(99);
    assert_eq!(a.deck, b.deck);
    assert_eq!(a.state_hash(), b.state_hash());

    let mut diverged = game.new_initial_state(99);
    let mut rng = stream(99, &[Part::Str("other")]);
    let mut scratch = Vec::new();
    for _ in 0..20 {
        if diverged.is_terminal() {
            break;
        }
        let action = diverged.sample_legal_into(&mut rng, &mut scratch).unwrap();
        diverged.step(action).unwrap();
    }
    // The undealt tail of the permutation is untouched by anything the agents did.
    let pos = diverged.deck_pos as usize;
    if diverged.deck_len == a.deck_len {
        assert_eq!(
            diverged.deck[pos..],
            a.deck[pos..],
            "the shuffle moved under the agents"
        );
    }
}

#[test]
fn position_hash_ignores_the_dealt_order_and_the_rng() {
    let game = Game::new(RuleConfig::new("usa", 2).unwrap()).unwrap();
    let state = game.new_initial_state(3);

    let mut twin = state.clone();
    // Reordering the *consumed* prefix leaves the same cards in the same hands, on the
    // same table and in the same discard: the same position, a different game.
    let cursor = twin.deck_pos as usize;
    twin.deck[..cursor].reverse();
    assert_eq!(state.position_hash(), twin.position_hash());
    if cursor > 1 && state.deck[..cursor].windows(2).any(|w| w[0] != w[1]) {
        assert_ne!(
            state.state_hash(),
            twin.state_hash(),
            "state_hash must keep the order"
        );
    }

    // The reshuffle stream is future randomness, not position.
    let mut rerolled = state.clone();
    rerolled.rng = ttr_core::rng::Pcg32::new(12345, 6789);
    assert_eq!(state.position_hash(), rerolled.position_hash());
    assert_ne!(state.state_hash(), rerolled.state_hash());
}

#[test]
fn clone_into_reproduces_the_source_exactly() {
    let game = Game::new(RuleConfig::new("usa", 3).unwrap()).unwrap();
    let mut source = game.new_initial_state(11);
    let mut rng = stream(11, &[Part::Str("arena")]);
    let mut scratch = Vec::new();
    let mut arena = game.new_initial_state(999);

    for _ in 0..40 {
        if source.is_terminal() {
            break;
        }
        let action = source.sample_legal_into(&mut rng, &mut scratch).unwrap();
        source.step(action).unwrap();
        source.clone_into(&mut arena);
        assert_eq!(arena.state_hash(), source.state_hash());
        assert_eq!(arena.legal_actions(), source.legal_actions());
        assert_eq!(arena.history, source.history);
    }
}

#[test]
fn the_flush_guard_and_the_reshuffle_are_actually_reached() {
    // Two guards on the guards: a branch that never runs proves nothing. These mirror the
    // Python property tests, and if a future change makes either unreachable, that is
    // itself the news.
    let mut saw_reshuffle = false;
    let mut saw_flush = false;

    for n in 2..=5usize {
        for seed in 0..25u64 {
            let game = Game::new(RuleConfig::new("usa", n).unwrap()).unwrap();
            let mut state = game.new_initial_state(seed);
            let mut rng = stream(seed, &[Part::Str("guards")]);
            let mut scratch = Vec::new();
            let start_rng = state.rng;
            let loco = state.board.locomotive as i8;
            while !state.is_terminal() {
                if state.faceup.iter().filter(|&&c| c == loco).count() >= 2 {
                    saw_flush = true;
                }
                let action = state.sample_legal_into(&mut rng, &mut scratch).unwrap();
                state.step(action).unwrap();
            }
            // `rng` is advanced by reshuffles and nothing else.
            if state.rng != start_rng {
                saw_reshuffle = true;
            }
        }
    }

    assert!(
        saw_reshuffle,
        "no seed in the sweep exhausted the deck; the reshuffle is untested"
    );
    assert!(
        saw_flush,
        "no seed in the sweep put two locomotives face-up"
    );
}
