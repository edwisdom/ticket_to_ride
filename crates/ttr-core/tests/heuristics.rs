//! Phase 3's exit criteria for the agent ladder, as tests.
//!
//! Three of them, and they fail for different reasons on purpose:
//!
//! * **Coverage** -- every heuristic reaches every action type on **both** maps. This is
//!   the published prior art's fatal bug turned into an assertion (PLAN.md §1, §14).
//! * **Ordering** -- H3 > H2 > H1 > H0. If the ladder is not monotone, everything Phase 5
//!   and 6 measure against it is built on sand (PLAN.md §13, de-risking ablation 2).
//! * **The anchor** -- H3's behaviour hash is a golden literal. H3 is the permanent Elo
//!   zero (PLAN.md §11), so a change to its constants *or* its code has to be a deliberate
//!   act, not something that happens during a refactor.

use ttr_core::config::RuleConfig;
use ttr_core::heuristic::{Heuristic, HeuristicParams, Tier, probe};
use ttr_core::scoring::returns;
use ttr_core::state::{Game, State};

/// Every (map, seat count) the engine supports.
fn all_configs() -> Vec<(&'static str, usize)> {
    let mut out = Vec::new();
    for board in ttr_core::board::all_boards() {
        for n in board.raw.min_players..=board.raw.max_players {
            out.push((board.name, n as usize));
        }
    }
    out
}

fn game(map: &str, n: usize) -> Game {
    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new(map, n).unwrap()
    };
    Game::new(cfg).unwrap()
}

/// Play one game and return the final state plus the action log.
fn play(game: &Game, tiers: &[Tier], params: &HeuristicParams, seed: u64) -> (State, Vec<u16>) {
    let mut agents: Vec<Heuristic> = tiers
        .iter()
        .enumerate()
        .map(|(seat, &tier)| {
            let mut agent = Heuristic::new(tier, *params, 4242);
            agent.begin_game(seat, seed);
            agent
        })
        .collect();
    let mut state = game.new_initial_state(seed);
    let mut log = Vec::new();
    while !state.is_terminal() {
        let seat = state.current_player() as usize;
        let action = agents[seat].act(&mut state);
        log.push(action);
        state
            .step(action)
            .expect("a heuristic played an illegal action");
    }
    (state, log)
}

/// Mean return for tier `a` over `seeds` seed blocks, every rotation played.
///
/// Cyclic rotations, so each agent sits in each seat equally often and first-player
/// advantage cancels instead of being folded into the result. Mirroring one seating and
/// reporting the complement is the flaw in the published prior art.
fn paired(map: &str, seats: usize, a: Tier, b: Tier, seeds: u64) -> f64 {
    let g = game(map, seats);
    let params = HeuristicParams::default();
    let (mut total, mut games) = (0.0, 0.0);
    for seed in 0..seeds {
        for rotation in 0..seats {
            let tiers: Vec<Tier> = (0..seats)
                .map(|i| if i == rotation { a } else { b })
                .collect();
            let (mut state, _) = play(&g, &tiers, &params, seed);
            total += returns(&mut state)[rotation];
            games += 1.0;
        }
    }
    total / games
}

// ---------------------------------------------------------------------------
// Legality and determinism
// ---------------------------------------------------------------------------

#[test]
fn every_heuristic_plays_only_legal_actions_everywhere() {
    // `play` panics on an illegal action, and `validate` catches a position the agent
    // drove into an inconsistent state. Every map, every seat count, every tier.
    let params = HeuristicParams::default();
    for (map, seats) in all_configs() {
        let g = game(map, seats);
        for tier in Tier::ALL {
            for seed in 0..3u64 {
                let tiers = vec![tier; seats];
                let (state, log) = play(&g, &tiers, &params, seed);
                state
                    .validate()
                    .unwrap_or_else(|e| panic!("{map} {seats}P {} seed {seed}: {e}", tier.name()));
                assert!(
                    !log.is_empty(),
                    "{map} {seats}P {} played nothing",
                    tier.name()
                );
            }
        }
    }
}

#[test]
fn h1_through_h4_are_deterministic_functions_of_the_position() {
    // Paired evaluation and the behaviour hash both rest on this. H0 is excluded: it is
    // the one agent that is *supposed* to be random, and it is reproducible from its seed
    // rather than deterministic in the position.
    let params = HeuristicParams::default();
    for (map, seats) in all_configs() {
        let g = game(map, seats);
        for tier in [Tier::H1, Tier::H2, Tier::H3, Tier::H4] {
            let tiers = vec![tier; seats];
            let (a, log_a) = play(&g, &tiers, &params, 11);
            let (b, log_b) = play(&g, &tiers, &params, 11);
            assert_eq!(log_a, log_b, "{map} {seats}P {}", tier.name());
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "{map} {seats}P {}",
                tier.name()
            );
        }
    }
}

#[test]
fn an_agents_stream_does_not_depend_on_the_seat_it_last_sat_in() {
    // `begin_game` re-seeds from (seat, game_id). Without it a seat swap would carry the
    // previous game's stream across, and the second half of a paired block would stop
    // being the mirror of the first.
    let g = game("usa", 2);
    let params = HeuristicParams::default();
    let mut fresh = Heuristic::new(Tier::H0, params, 7);
    fresh.begin_game(1, 99);
    let mut reused = Heuristic::new(Tier::H0, params, 7);
    reused.begin_game(0, 3);
    let mut state = g.new_initial_state(3);
    for _ in 0..20 {
        if state.is_terminal() {
            break;
        }
        let a = reused.act(&mut state);
        state.step(a).unwrap();
    }
    reused.begin_game(1, 99);

    let mut a = g.new_initial_state(99);
    let mut b = g.new_initial_state(99);
    while !a.is_terminal() {
        let x = fresh.act(&mut a);
        let y = reused.act(&mut b);
        assert_eq!(x, y, "a re-seated agent did not reproduce a fresh one");
        a.step(x).unwrap();
        b.step(y).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Coverage -- the prior art's fatal bug, as an assertion
// ---------------------------------------------------------------------------

#[test]
fn every_heuristic_exercises_every_action_type_on_both_maps() {
    // PLAN.md §14's Phase 3 criterion, and §1's process lesson. The paper's baselines were
    // lifted from a repo whose ticket appetite read `trains > 15`; on its 10-train map that
    // could never fire, so every "well-designed heuristic" it benchmarked against played
    // its opening tickets and never drew another. Nothing errored, and the entire published
    // ladder was measured against silently broken agents.
    //
    // This project reproduced the same symptom twice from the opposite direction -- a
    // budget-blind keep filter that took 22 cars of tickets onto a 20-train board, and a
    // ticket-draw model priced off the mean of the deck rather than the best of the three
    // actually offered. Both showed up here as `draw_tickets` at zero on TTR-mini.
    let params = HeuristicParams::default();
    for (map, seats) in all_configs() {
        for tier in [Tier::H1, Tier::H2, Tier::H3, Tier::H4] {
            let c = probe::coverage(map, seats, tier, &params, 0..40);
            assert!(
                c.missing().is_empty(),
                "{map} {seats}P {} never reached {:?} in 40 games ({c:?})",
                tier.name(),
                c.missing(),
            );
        }
    }
}

#[test]
fn h3_keeps_more_than_the_minimum_and_h2_never_does() {
    // The difference between the two tiers, stated as a measurement rather than left to
    // the reader. H2 minimizes added track, so keeping the minimum *is* its rule; H3 prices
    // the keep by expected settlement and sometimes takes a third ticket. If this ever
    // flipped, H3's EV filter would have stopped running with no other visible symptom.
    let params = HeuristicParams::default();
    for map in ["usa", "mini"] {
        let h2 = probe::coverage(map, 2, Tier::H2, &params, 0..40);
        let h3 = probe::coverage(map, 2, Tier::H3, &params, 0..40);
        assert_eq!(h2.keep_extra, 0, "{map}: H2 kept more than the minimum");
        assert!(
            h3.keep_extra > 0,
            "{map}: H3's EV filter never kept an extra ticket"
        );
    }
}

// ---------------------------------------------------------------------------
// Ladder ordering
// ---------------------------------------------------------------------------

#[test]
fn the_ladder_is_monotone_on_both_maps() {
    // De-risking ablation 2 from PLAN.md §13: "H3 > H2 > H1 > random -- if not, everything
    // downstream is built on sand."
    //
    // The margins are asserted loosely on purpose. What has to hold is the *ordering*; the
    // sizes are measured and reported by `ttr leaderboard`, and they do not match the
    // planning-time predictions (H2 is far stronger relative to H1 than §11 guessed, and
    // H4 far weaker relative to H3). Baking the predictions in here would turn a
    // measurement into an assumption.
    for (map, seeds) in [("usa", 30u64), ("mini", 40)] {
        let h1 = paired(map, 2, Tier::H1, Tier::H0, seeds);
        let h2 = paired(map, 2, Tier::H2, Tier::H1, seeds);
        let h3 = paired(map, 2, Tier::H3, Tier::H2, seeds);
        assert!(h1 > 0.4, "{map}: H1 beats random by only {h1:+.3}");
        assert!(h2 > 0.4, "{map}: H2 beats H1 by only {h2:+.3}");
        assert!(h3 > 0.1, "{map}: H3 beats H2 by only {h3:+.3}");
    }
}

#[test]
fn h4_is_not_worse_than_h3() {
    // Deliberately one-sided, and the reason is recorded rather than hidden. PLAN.md §11
    // predicts H4 at +130 Elo over H3 (~68% paired). Measured across 2P-5P on both maps it
    // is +-0.05 mean return -- inside noise at these sample sizes. Public-information
    // blocking against a strong non-adaptive planner is worth close to nothing here, and
    // the seat count does not change that; see docs/WORKLOG.md.
    //
    // So this asserts what is actually supported: H4 does not *regress*. An H4 that lost
    // materially would mean the threat pricing had broken again, which it has once.
    for map in ["usa", "mini"] {
        let h4 = paired(map, 2, Tier::H4, Tier::H3, 30);
        assert!(h4 > -0.25, "{map}: H4 lost to H3 by {h4:+.3}");
    }
}

// ---------------------------------------------------------------------------
// The anchor
// ---------------------------------------------------------------------------

/// H3's behaviour hash. **This is the Elo scale's zero.**
///
/// If this test fails, H3 plays differently than it did when the value was recorded --
/// because a constant moved, or because code it depends on did. Either way every rating
/// ever recorded against H3 has silently re-based, so the change has to be deliberate:
///
/// 1. Decide the new H3 is genuinely better, and register it as a **new agent** (`h3.v2`)
///    rated against the old anchor, which is the supported path.
/// 2. Or, if the anchor really must move, update this literal *and* record the re-basing
///    in docs/WORKLOG.md, because every stored rating predating it is now on a different
///    scale and the leaderboard cannot tell.
///
/// What it must never be is quietly refreshed to make the suite green.
const H3_ANCHOR: &str = "5b906156d4b1204be2d0fbd17700221d";

#[test]
fn the_h3_anchor_has_not_moved() {
    let got = probe::behaviour_hash(Tier::H3, &HeuristicParams::default());
    assert_eq!(
        got, H3_ANCHOR,
        "H3 plays differently than when the anchor was recorded. Read the comment on \
         H3_ANCHOR before touching this literal: it is the zero of every rating in the \
         results store."
    );
}

#[test]
fn each_tier_has_a_distinct_identity() {
    let params = HeuristicParams::default();
    let mut seen: Vec<String> = Vec::new();
    for tier in Tier::ALL {
        let h = probe::behaviour_hash(tier, &params);
        assert!(
            !seen.contains(&h),
            "{} shares an identity with another tier",
            tier.name()
        );
        seen.push(h);
    }
}
