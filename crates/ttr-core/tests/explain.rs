//! `explain` is `act` with a recorder, never a second scorer (PLAN.md §14.2).
//!
//! The review screen shows each bot decision's candidates and the terms behind them, and a
//! review is only worth reading if those are the numbers the bot actually compared. So: an
//! agent driven by `explain` and an agent driven by `act` are run in lockstep through the
//! same games, and must choose the same action at every one of at least ten thousand
//! positions per map and tier, with the explanation's best candidate being that action.

use ttr_core::config::RuleConfig;
use ttr_core::heuristic::{Explanation, Heuristic, HeuristicParams, Tier};
use ttr_core::state::{Game, State};

/// Positions per (map, tier), as PLAN.md §14.2's 4.1 exit criterion asks.
const POSITIONS: usize = 10_000;

fn game(map: &str) -> Game {
    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new(map, 2).unwrap()
    };
    Game::new(cfg).unwrap()
}

/// Checks one explanation against the action `act` chose. Returns whether the decision
/// was a real one (more than one legal action).
fn check(state: &State, e: &Explanation, acted: u16, tier: Tier) -> bool {
    let what = format!("{} at turn {} ({:?})", tier.name(), state.turn, state);
    assert_eq!(
        e.chosen, acted,
        "explain chose differently from act: {what}"
    );

    // Every legal action is a candidate exactly once, and nothing else is.
    let mut seen: Vec<u16> = e.candidates.iter().map(|c| c.action).collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        state.legal_actions(),
        "candidates != legal actions: {what}"
    );
    if seen.len() == 1 {
        assert_eq!(e.rule, Some("forced"), "{what}");
        return false;
    }

    // Best first, ties to the lowest id -- the decision's own tiebreak -- so the argmax
    // is the first candidate, and it is the action played.
    for pair in e.candidates.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            a.total > b.total || (a.total == b.total && a.action < b.action),
            "candidates out of order: {what}"
        );
    }
    assert_eq!(e.candidates[0].action, acted, "argmax != act: {what}");

    // The breakdown is complete: the terms add up to the total the decision compared. A
    // term the scorer added and forgot to report would show up here.
    for c in &e.candidates {
        if !c.total.is_finite() {
            continue;
        }
        let sum: f64 = c.terms.iter().map(|&(_, v)| v).sum();
        assert!(
            (sum - c.total).abs() <= 1e-9 * c.total.abs().max(1.0),
            "{what}: action {} terms {:?} sum to {sum}, total {}",
            c.action,
            c.terms,
            c.total
        );
    }
    true
}

/// Lockstep games until `POSITIONS` real decisions of `tier` have been checked.
///
/// Alternates self-play, which stays on the trajectory the agent generates, with games
/// against H0, which reach the ragged positions it never would -- the probe set's
/// reasoning, for the same reason.
fn sweep(map: &str, tier: Tier) {
    let g = game(map);
    let params = HeuristicParams::default();
    let mut checked = 0;
    let mut seed = 0u64;
    while checked < POSITIONS {
        let mixed = seed % 2 == 1;
        let tiers = [tier, if mixed { Tier::H0 } else { tier }];
        let mut actors: Vec<Heuristic> = (0..2)
            .map(|seat| {
                let mut h = Heuristic::new(tiers[seat], params, 7);
                h.begin_game(seat, seed);
                h
            })
            .collect();
        let mut explainers = actors.clone();
        for (seat, h) in explainers.iter_mut().enumerate() {
            h.begin_game(seat, seed);
        }

        let mut state = g.new_initial_state(seed);
        let mut twin = state.clone();
        while !state.is_terminal() {
            let seat = state.current_player() as usize;
            let acted = actors[seat].act(&mut state);
            let e = explainers[seat].explain(&mut twin);
            if tiers[seat] == tier && check(&state, &e, acted, tier) {
                checked += 1;
            } else {
                assert_eq!(e.chosen, acted, "{map} {} seed {seed}", tiers[seat].name());
            }
            state.step(acted).unwrap();
            twin.step(acted).unwrap();
        }
        seed += 1;
    }
}

#[test]
fn explain_chooses_what_act_chooses_on_usa() {
    for tier in Tier::ALL[1..].iter().copied() {
        sweep("usa", tier);
    }
}

#[test]
fn explain_chooses_what_act_chooses_on_mini() {
    for tier in Tier::ALL[1..].iter().copied() {
        sweep("mini", tier);
    }
}

#[test]
fn explaining_h0_draws_exactly_what_acting_would() {
    // H0's explanation is the distribution it drew from, and taking it must advance the
    // stream exactly as `act` does -- or a game explained live would diverge from the same
    // game replayed.
    let g = game("usa");
    let params = HeuristicParams::default();
    let mut actor = Heuristic::new(Tier::H0, params, 3);
    let mut explainer = Heuristic::new(Tier::H0, params, 3);
    actor.begin_game(0, 11);
    explainer.begin_game(0, 11);
    let mut state = g.new_initial_state(11);
    let mut twin = state.clone();
    while !state.is_terminal() {
        let acted = actor.act(&mut state);
        let e = explainer.explain(&mut twin);
        assert_eq!(e.chosen, acted);
        if e.candidates.len() > 1 {
            assert_eq!(e.rule, Some("uniform"));
            let p = 1.0 / e.candidates.len() as f64;
            assert!(e.candidates.iter().all(|c| c.total == p));
        }
        state.step(acted).unwrap();
        twin.step(acted).unwrap();
    }
}

#[test]
fn truncation_keeps_the_best_and_the_chosen() {
    let g = game("usa");
    let params = HeuristicParams::default();
    let mut h0 = Heuristic::new(Tier::H0, params, 5);
    h0.begin_game(0, 2);
    let mut state = g.new_initial_state(2);
    // Into the main phase, where H0 has dozens of options.
    while state.phase != ttr_core::config::PHASE_MAIN {
        let a = state.legal_actions()[0];
        state.step(a).unwrap();
    }
    for _ in 0..40 {
        let mut e = h0.explain(&mut state);
        let chosen = e.chosen;
        e.truncate(3);
        assert!(e.candidates.len() <= 4);
        assert!(e.candidates.iter().any(|c| c.action == chosen));
        state.step(chosen).unwrap();
        if state.is_terminal() {
            break;
        }
    }
}
