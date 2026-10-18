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

use ttr_core::board::Board;
use ttr_core::config::{NOT_TRIGGERED, PHASE_MAIN, RuleConfig};
use ttr_core::heuristic::{Heuristic, HeuristicParams, Plan, Tier, probe};
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
fn every_tier_above_h0_is_a_deterministic_function_of_the_position() {
    // Paired evaluation and the behaviour hash both rest on this. H0 is excluded: it is
    // the one agent that is *supposed* to be random, and it is reproducible from its seed
    // rather than deterministic in the position.
    let params = HeuristicParams::default();
    for (map, seats) in all_configs() {
        let g = game(map, seats);
        for tier in Tier::ALL[1..].iter().copied() {
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
        for tier in Tier::ALL[1..].iter().copied() {
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
fn h4_beats_h3() {
    // PLAN.md §11 predicted H4 at +130 Elo over H3; its blocking alone measured level with
    // H3 at 2P and below it at 4P. What H4 gained instead is a plan that keeps what it
    // needs: off-plan claims pay for the trains and cards they take from it. Measured over
    // 10,000 games a configuration: +81 Elo at USA 2P, +43 on mini 2P, +77 at USA 4P
    // (docs/WORKLOG.md, "H4: off-plan claims pay for what they take").
    for map in ["usa", "mini"] {
        let h4 = paired(map, 2, Tier::H4, Tier::H3, 30);
        assert!(h4 > 0.0, "{map}: H4 lost to H3 by {h4:+.3}");
    }
}

#[test]
fn h5_beats_h4() {
    // H5 prices a ticket draw against its own network instead of the deck's mean ticket, and
    // drops H3's train gate, so it draws late, when its network already reaches the tickets.
    // Measured: +31 Elo over H4 at USA 2P and +35 at USA 4P, level on mini 2P, where the
    // network is rarely big enough to make a late ticket cheap (docs/WORKLOG.md, "H5").
    let h5 = paired("usa", 2, Tier::H5, Tier::H4, 30);
    assert!(h5 > 0.0, "usa: H5 lost to H4 by {h5:+.3}");
}

#[test]
fn h6_beats_h5() {
    // H6 chooses the network its tickets need by the points its routes score, not by its
    // cars alone. Measured: +177 Elo over H5 at USA 2P, +45 on mini 2P, +160 at USA 4P
    // (docs/WORKLOG.md, "H6").
    for map in ["usa", "mini"] {
        let h6 = paired(map, 2, Tier::H6, Tier::H5, 30);
        assert!(h6 > 0.0, "{map}: H6 lost to H5 by {h6:+.3}");
    }
}

// ---------------------------------------------------------------------------
// What H3 learned from losing 4-123
// ---------------------------------------------------------------------------

/// A USA 2P game H3 (seat 0) lost 4-123 to a person, and its first 87 moves. At this
/// point four of H3's five tickets hinge on joining the two halves of its network, and two
/// ways of doing it cost the same 11 cars.
const STALLED_SEED: u64 = 1_120_284_448_765_548_802;
#[rustfmt::skip]
const STALLED_PREFIX: [u16; 87] = [
    909, 911, 906, 911, 901, 903, 21, 905, 904, 688, 695, 901, 902, 901, 901, 901, 905, 905,
    532, 905, 905, 495, 905, 904, 901, 901, 902, 901, 901, 904, 905, 904, 904, 904, 382, 673,
    904, 640, 904, 905, 904, 158, 905, 905, 901, 901, 905, 905, 906, 910, 905, 900, 903, 903,
    573, 903, 905, 905, 191, 250, 905, 905, 221, 905, 905, 905, 905, 616, 905, 905, 905, 905,
    905, 905, 905, 905, 414, 603, 905, 905, 905, 905, 905, 905, 435, 905, 905,
];

/// A free-standing route between two cities, by name.
fn route(board: &Board, a: &str, b: &str) -> u16 {
    let city = |name: &str| board.raw.cities.iter().position(|&c| c == name).unwrap() as u8;
    let (a, b) = (city(a), city(b));
    (0..board.n_segments)
        .find(|&s| {
            (board.seg_a[s], board.seg_b[s]) == (a, b) || (board.seg_a[s], board.seg_b[s]) == (b, a)
        })
        .unwrap() as u16
}

#[test]
fn h3_plans_the_network_its_cards_can_pay_for() {
    let g = game("usa", 2);
    let board = g.board;
    let mut state = g.new_initial_state(STALLED_SEED);
    for &action in &STALLED_PREFIX {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 0);
    let yellow = route(board, "El Paso", "Oklahoma City");
    let orange = route(board, "Duluth", "Helena");
    let blue = route(board, "Oklahoma City", "Santa Fe");

    // By cars alone the networks tie, and the tie went to 6 orange + 5 yellow: nine cards
    // the draw pile did not hold. The old H3 drew blind for the rest of the game.
    let blind = Plan::build(&mut state, 0, 0.0);
    assert!(blind.contains(yellow) && blind.contains(orange));

    let params = HeuristicParams::default();
    let aware = Plan::build(&mut state, 0, params.plan_card_weight);
    assert_eq!(aware.steiner_cost, blind.steiner_cost);
    assert!(
        aware.contains(blue) && !aware.contains(yellow) && !aware.contains(orange),
        "H3 still plans the network its hand cannot pay for: {:?}",
        aware.segments
    );

    let mut h3 = Heuristic::new(Tier::H3, params, 0);
    h3.begin_game(0, STALLED_SEED);
    let action = h3.act(&mut state);
    assert!(action < g.space.claim_end, "H3 drew instead of building");
    let (segment, _) = g.space.decode_claim(action);
    assert!(aware.contains(segment), "H3 claimed off its plan");
}

#[test]
fn on_its_last_turn_h3_claims_whenever_it_can() {
    // Once the final round starts every seat has one turn left: a card drawn now is never
    // spent and a ticket drawn now can only cost. Valuing that turn like any other, the old
    // H3 drew on it whenever a blind card out-scored its best route -- and ended games
    // holding a dozen cards.
    let params = HeuristicParams::default();
    let mut last_turns = 0;
    for (map, seats) in all_configs() {
        let g = game(map, seats);
        for seed in 0..6u64 {
            let mut agents: Vec<Heuristic> = (0..seats)
                .map(|seat| {
                    let mut agent = Heuristic::new(Tier::H3, params, 1);
                    agent.begin_game(seat, seed);
                    agent
                })
                .collect();
            let mut state = g.new_initial_state(seed);
            while !state.is_terminal() {
                let seat = state.current_player() as usize;
                let last_turn = state.final_left != NOT_TRIGGERED && state.phase == PHASE_MAIN;
                let can_claim = state.legal_actions().iter().any(|&a| a < g.space.claim_end);
                let action = agents[seat].act(&mut state);
                if last_turn && can_claim {
                    last_turns += 1;
                    assert!(
                        action < g.space.claim_end,
                        "{map} {seats}P seed {seed}: H3 passed up a claim on its last turn"
                    );
                }
                state.step(action).unwrap();
            }
        }
    }
    assert!(
        last_turns >= 100,
        "only {last_turns} last turns with a claim to make"
    );
}

// ---------------------------------------------------------------------------
// The anchor
// ---------------------------------------------------------------------------

/// H3's behaviour hash. **This is what the Elo scale is anchored on.**
///
/// If this test fails, H3 plays differently than it did when the value was recorded --
/// because a constant moved, or because code it depends on did. Either way every rating
/// ever recorded against H3 has re-based, so the change has to be deliberate: measure the
/// new H3 against the old one in the arena, then update this literal *and* record the move
/// and the measurement in docs/WORKLOG.md, because every stored rating predating it is on a
/// different scale and the leaderboard cannot tell.
///
/// Moved once so far, on purpose: H3 was too weak to be the bar, so it gained card-aware
/// planning and final-turn scoring (worklog, "H3 moves: the plan follows the cards"). What
/// this literal must never be is quietly refreshed to make the suite green.
const H3_ANCHOR: &str = "1e84ef3c250ad2a947cb2024a60d60ad";

#[test]
fn the_h3_anchor_has_not_moved() {
    let got = probe::behaviour_hash(Tier::H3, &HeuristicParams::default());
    assert_eq!(
        got, H3_ANCHOR,
        "H3 plays differently than when the anchor was recorded. Read the comment on \
         H3_ANCHOR before touching this literal: it is the anchor of every rating in the \
         results store."
    );
}

/// Every tier's behaviour hash, H3's included.
///
/// Recorded before Phase 4.1 threaded the trace sink behind `explain` through the scorer
/// and re-wrote H1's cascade as a priority: both had to leave play exactly as it was, and
/// this is the check that they did. Unlike [`H3_ANCHOR`] these are not the rating scale's
/// zero -- a deliberate change to H0-H2 or H4 updates its literal here and is noted in the
/// worklog, since it makes a new agent in every store -- but none may move by accident.
const BEHAVIOUR_HASHES: [(Tier, &str); 7] = [
    (Tier::H0, "5c62d98faf15e624e3248a25c9a1f85a"),
    (Tier::H1, "2cd429112540266c66d055eab0040d79"),
    (Tier::H2, "f7411becf76b49abe316f7fe8a157de4"),
    (Tier::H3, H3_ANCHOR),
    // H4 moved with the anchor, and again when its off-plan claims began paying for what
    // they take from the plan.
    (Tier::H4, "3099a56f772c213325a2197df3857d69"),
    (Tier::H5, "8c917c3fb99338efb0307b871c1a52bf"),
    (Tier::H6, "0a0c63864a967ae1de97ab452264e482"),
];

#[test]
fn no_tier_plays_differently_by_accident() {
    let params = HeuristicParams::default();
    for (tier, expected) in BEHAVIOUR_HASHES {
        assert_eq!(
            probe::behaviour_hash(tier, &params),
            expected,
            "{} plays differently than when its hash was recorded",
            tier.name()
        );
    }
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

// ---------------------------------------------------------------------------
// What H4 learned from H3's self-play losses
// ---------------------------------------------------------------------------

/// H3 self-play, USA 2P, and seat 0's position after these moves: its two open tickets,
/// worth 39, need 12 more cars, it has 24 trains, and H3 spends 4 of them and four cards
/// its plan needs on Chicago-Omaha, which neither ticket uses. It spent 17 trains off its
/// plan from here and lost 39-137 with both tickets open.
const OFF_PLAN_SEED: u64 = 7_000_000;
#[rustfmt::skip]
const OFF_PLAN_PREFIX: [u16; 81] = [
    911, 913, 555, 906, 910, 900, 347, 695, 18, 900, 905, 900, 905, 905, 442, 171, 906, 908,
    906, 907, 900, 903, 900, 686, 813, 901, 902, 900, 901, 204, 902, 903, 903, 903, 561, 905,
    901, 905, 905, 532, 905, 905, 513, 905, 905, 905, 905, 905, 905, 365, 905, 905, 905, 905,
    905, 905, 607, 905, 905, 905, 905, 905, 905, 905, 905, 463, 905, 905, 158, 905, 905, 901,
    905, 437, 673, 905, 905, 905, 905, 905, 905,
];

/// H3 self-play, USA 2P: seat 0 holds five blacks, one short of the 6-black El Paso-Los
/// Angeles that finishes two tickets worth 37, and H3 spends all five on Montreal-Sault
/// St. Marie, off its plan. Both tickets stayed open; it lost 68-128.
const CARD_SPEND_SEED: u64 = 7_000_009;
#[rustfmt::skip]
const CARD_SPEND_PREFIX: [u16; 131] = [
    909, 911, 906, 908, 608, 161, 906, 911, 906, 910, 812, 228, 901, 900, 900, 900, 900, 901,
    901, 344, 204, 901, 902, 902, 447, 596, 904, 900, 900, 905, 900, 901, 248, 687, 900, 903,
    903, 900, 900, 745, 900, 901, 900, 901, 708, 544, 902, 905, 906, 911, 221, 513, 906, 907,
    901, 903, 903, 905, 364, 183, 904, 904, 904, 905, 905, 905, 906, 910, 905, 905, 902, 902,
    905, 905, 487, 905, 905, 902, 904, 310, 905, 905, 902, 904, 905, 905, 905, 905, 396, 109,
    905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 101, 640, 905, 905,
    905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 475, 905, 905, 905,
    905, 905, 905, 905, 905,
];

/// Play `prefix` from `seed`, then have H3 and H4 decide seat 0's move. H3 must claim
/// `a`-`b`, off its plan, which is the reviewed mistake; H4 must not, and its own pricing of
/// that claim must carry `term` -- what the claim takes from the plan.
fn h4_keeps_what_its_plan_needs(seed: u64, prefix: &[u16], a: &str, b: &str, term: &str) {
    let g = game("usa", 2);
    let mut state = g.new_initial_state(seed);
    for &action in prefix {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 0);
    let off_plan = route(g.board, a, b);
    let params = HeuristicParams::default();

    let mut h3 = Heuristic::new(Tier::H3, params, 0);
    h3.begin_game(0, seed);
    let action = h3.act(&mut state.clone());
    assert!(action < g.space.claim_end, "H3 no longer claims here");
    assert_eq!(
        g.space.decode_claim(action).0,
        off_plan,
        "H3 no longer claims {a}-{b}"
    );

    let mut h4 = Heuristic::new(Tier::H4, params, 0);
    h4.begin_game(0, seed);
    let e = h4.explain(&mut state);
    let claimed =
        |action: u16| action < g.space.claim_end && g.space.decode_claim(action).0 == off_plan;
    assert!(
        !claimed(e.chosen),
        "H4 spent its plan's trains and cards on {a}-{b}"
    );
    let priced = e
        .candidates
        .iter()
        .find(|c| claimed(c.action))
        .expect("the claim is still legal");
    let charge = priced
        .terms
        .iter()
        .find(|(name, _)| *name == term)
        .map(|&(_, value)| value);
    assert!(
        charge.is_some_and(|v| v < 0.0),
        "{a}-{b} carries no {term} charge: {:?}",
        priced.terms
    );
}

#[test]
fn h4_does_not_spend_the_trains_its_tickets_need() {
    h4_keeps_what_its_plan_needs(
        OFF_PLAN_SEED,
        &OFF_PLAN_PREFIX,
        "Chicago",
        "Omaha",
        "train_budget",
    );
}

#[test]
fn h4_does_not_spend_the_cards_its_tickets_need() {
    h4_keeps_what_its_plan_needs(
        CARD_SPEND_SEED,
        &CARD_SPEND_PREFIX,
        "Montreal",
        "Sault St. Marie",
        "plan_cards_spent",
    );
}

// ---------------------------------------------------------------------------
// What H5 learned: a ticket costs its distance from the network
// ---------------------------------------------------------------------------

/// H5 (seat 0) against H4, USA 2P: with 10 trains left, long past H3's gate, H5's network
/// already reaches enough of the tickets it might be dealt that an offer is worth drawing.
const LATE_TICKETS_SEED: u64 = 7_900_000;
#[rustfmt::skip]
const LATE_TICKETS_PREFIX: [u16; 119] = [
    912, 911, 870, 906, 912, 900, 905, 491, 768, 900, 900, 900, 595, 663, 900, 900, 900, 901,
    902, 748, 248, 901, 902, 905, 900, 900, 900, 900, 48, 607, 902, 905, 532, 902, 902, 905,
    905, 191, 118, 900, 902, 905, 900, 902, 905, 902, 902, 902, 902, 905, 902, 905, 43, 437,
    902, 905, 902, 905, 905, 905, 225, 905, 905, 906, 910, 905, 905, 695, 905, 905, 901, 901,
    905, 905, 901, 903, 873, 903, 905, 905, 171, 905, 905, 903, 905, 903, 616, 905, 905, 903,
    903, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 794, 905, 905,
    905, 905, 475, 905, 905, 478, 905, 905, 882, 905, 905,
];

#[test]
fn h5_draws_tickets_late_when_its_network_reaches_them() {
    let g = game("usa", 2);
    let mut state = g.new_initial_state(LATE_TICKETS_SEED);
    for &action in &LATE_TICKETS_PREFIX {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 0);
    assert_eq!(state.trains[0], 10);
    let params = HeuristicParams::default();

    // H3's gate closed at 25 trains, so H4 does something else.
    let mut h4 = Heuristic::new(Tier::H4, params, 0);
    h4.begin_game(0, LATE_TICKETS_SEED);
    assert_ne!(h4.act(&mut state.clone()), g.space.draw_tickets);

    let mut h5 = Heuristic::new(Tier::H5, params, 0);
    h5.begin_game(0, LATE_TICKETS_SEED);
    let e = h5.explain(&mut state);
    assert_eq!(e.chosen, g.space.draw_tickets, "H5 passed up a cheap offer");
    let draw = &e.candidates[0];
    let value = |name: &str, of: &[(&'static str, f64)]| {
        of.iter().find(|(n, _)| *n == name).map(|&(_, v)| v)
    };
    assert!(value("network_settlement", &draw.terms).is_some_and(|v| v > 0.0));
    assert!(value("tickets_ready", &draw.facts).is_some_and(|v| v >= 1.0));
}

/// H5 (seat 0) against H4, USA 2P, both seats on 7 trains: Chicago-Omaha takes H5's longest
/// trail from 31 cars to 35, past the rival's 31, with the game a few turns from its end.
const RACE_SEED: u64 = 8_400_039;
#[rustfmt::skip]
const RACE_PREFIX: [u16; 152] = [
    909, 909, 597, 906, 907, 905, 905, 906, 907, 905, 905, 906, 910, 247, 900, 902, 900, 904,
    905, 905, 905, 685, 605, 864, 906, 911, 905, 905, 901, 901, 661, 433, 905, 905, 901, 905,
    204, 158, 905, 905, 905, 905, 124, 905, 905, 905, 905, 905, 905, 905, 905, 905, 903, 905,
    905, 905, 905, 573, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905,
    578, 396, 905, 905, 362, 905, 905, 544, 905, 905, 515, 905, 905, 228, 905, 905, 905, 905,
    905, 905, 905, 905, 213, 221, 905, 905, 906, 909, 905, 905, 447, 905, 905, 906, 908, 875,
    905, 905, 905, 905, 905, 905, 905, 905, 106, 478, 905, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 905, 905, 905, 455, 132,
];

#[test]
fn h5_races_for_the_longest_route() {
    let g = game("usa", 2);
    let mut state = g.new_initial_state(RACE_SEED);
    for &action in &RACE_PREFIX {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 0);
    let race_route = route(g.board, "Chicago", "Omaha");
    let params = HeuristicParams::default();

    // H3's tiebreak alone does not see the race: H5 without it plays something else.
    let tiebreak = HeuristicParams {
        longest_race_weight: 0.0,
        ..params
    };
    let mut without = Heuristic::new(Tier::H5, tiebreak, 0);
    without.begin_game(0, RACE_SEED);
    let action = without.act(&mut state.clone());
    assert!(action >= g.space.claim_end || g.space.decode_claim(action).0 != race_route);

    let mut h5 = Heuristic::new(Tier::H5, params, 0);
    h5.begin_game(0, RACE_SEED);
    let e = h5.explain(&mut state);
    assert!(e.chosen < g.space.claim_end, "H5 did not claim");
    assert_eq!(
        g.space.decode_claim(e.chosen).0,
        race_route,
        "H5 let the race go"
    );
    let chosen = &e.candidates[0];
    let get = |name: &str, of: &[(&'static str, f64)]| {
        of.iter().find(|(n, _)| *n == name).map(|&(_, v)| v)
    };
    assert!(get("longest_race", &chosen.terms).is_some_and(|v| v > 0.0));
    let after = get("longest_after", &chosen.facts).unwrap();
    let rival = get("rival_longest", &chosen.facts).unwrap();
    assert!(
        after > rival,
        "the claim leaves H5 at {after} against {rival}"
    );
}

/// H5 (seat 0) against H4, USA 2P: every ticket H5 holds is done, it has 10 trains, and it
/// can pay for El Paso-Los Angeles, six cars for 15 points.
const FINISHED_SEED: u64 = 9_000_007;
#[rustfmt::skip]
const FINISHED_PREFIX: [u16; 156] = [
    909, 912, 598, 906, 910, 901, 901, 906, 908, 901, 901, 901, 591, 685, 901, 903, 903, 904,
    905, 347, 608, 900, 901, 905, 905, 445, 905, 905, 906, 910, 905, 905, 901, 905, 744, 902,
    904, 904, 905, 291, 905, 905, 904, 905, 905, 905, 204, 578, 905, 905, 502, 902, 902, 905,
    905, 340, 905, 905, 532, 905, 905, 906, 907, 905, 905, 900, 901, 901, 171, 730, 901, 905,
    513, 695, 905, 905, 814, 905, 905, 890, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 163, 905, 905, 905, 905, 905, 905, 905, 905, 158, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 905, 905, 905, 354, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 905, 905, 905, 905, 905, 905, 635, 905, 905,
];

#[test]
fn with_its_tickets_done_h5_spends_its_trains_on_points() {
    let g = game("usa", 2);
    let mut state = g.new_initial_state(FINISHED_SEED);
    for &action in &FINISHED_PREFIX {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 0);
    let six = route(g.board, "El Paso", "Los Angeles");
    let params = HeuristicParams::default();

    // Charged H3's rate for its trains, H5 drew tickets here instead.
    let charged = HeuristicParams {
        finished_train_cost: params.min_points_per_train,
        ..params
    };
    let mut before = Heuristic::new(Tier::H5, charged, 0);
    before.begin_game(0, FINISHED_SEED);
    assert_eq!(before.act(&mut state.clone()), g.space.draw_tickets);

    let mut h5 = Heuristic::new(Tier::H5, params, 0);
    h5.begin_game(0, FINISHED_SEED);
    let e = h5.explain(&mut state);
    assert!(e.context.contains(&("tickets_done", 1.0)));
    assert!(e.chosen < g.space.claim_end, "H5 did not claim");
    assert_eq!(g.space.decode_claim(e.chosen).0, six);
    assert!(
        !e.candidates[0]
            .terms
            .iter()
            .any(|(name, _)| *name == "train_cost"),
        "a finished H5 still charged its trains"
    );
}

/// H4 (seat 0) against an H5 without late card pricing, USA 2P. H5's tickets wait on
/// Helena-Seattle, six yellows with one in hand, and Portland-Salt Lake City, six blues. Its
/// hand pays for Calgary-Helena and Calgary-Seattle, gray, two cars longer than the yellow
/// route; it has 13 trains and the game is three-quarters done. It lost 56-134 with 24 cards
/// in hand.
const STALL_SEED: u64 = 9_500_119;
#[rustfmt::skip]
const STALL_LATE_PREFIX: [u16; 117] = [
    912, 911, 906, 908, 900, 605, 346, 23, 903, 903, 906, 908, 903, 904, 903, 903, 900, 901,
    900, 745, 329, 900, 901, 532, 588, 905, 905, 902, 900, 905, 905, 902, 903, 171, 628, 902,
    902, 904, 902, 48, 903, 905, 903, 905, 903, 903, 903, 279, 573, 903, 905, 905, 905, 191,
    905, 902, 906, 908, 747, 435, 814, 903, 904, 886, 448, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 708, 730, 905, 905, 831, 905, 905, 904, 905, 904, 905, 904, 903, 463, 904,
    904, 905, 905, 711, 905, 905, 904, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905, 905,
    905, 905, 905, 905, 903, 905, 784, 905, 905,
];

#[test]
fn late_in_the_game_h5_plans_the_network_its_hand_can_pay_for() {
    let g = game("usa", 2);
    let mut state = g.new_initial_state(STALL_SEED);
    for &action in &STALL_LATE_PREFIX {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 1);
    let calgary = route(g.board, "Calgary", "Helena");
    let params = HeuristicParams::default();

    let early_rules = HeuristicParams {
        late_card_progress: 2.0,
        ..params
    };
    let mut before = Heuristic::new(Tier::H5, early_rules, 0);
    before.begin_game(1, STALL_SEED);
    assert!(
        before.act(&mut state.clone()) >= g.space.claim_end,
        "it no longer draws here"
    );

    let mut h5 = Heuristic::new(Tier::H5, params, 0);
    h5.begin_game(1, STALL_SEED);
    let e = h5.explain(&mut state);
    assert!(
        e.chosen < g.space.claim_end,
        "H5 kept waiting on the yellows"
    );
    assert_eq!(g.space.decode_claim(e.chosen).0, calgary);
    assert!(e.candidates[0].facts.contains(&("on_plan", 1.0)));
}

// ---------------------------------------------------------------------------
// What H6 learned: a network is worth the points its routes score
// ---------------------------------------------------------------------------

/// H6 (seat 0) against H5, USA 2P, early in the game: H5's plan reaches Miami through
/// Charleston, H6's takes Atlanta-Miami, five blue cars for 10 points.
const POINTS_SEED: u64 = 10_200_003;
#[rustfmt::skip]
const POINTS_PREFIX: [u16; 23] = [
    909, 912, 490, 900, 24, 598, 900, 901, 900, 901, 901, 250, 903, 903, 905, 905, 903, 903,
    603, 904, 905, 905, 905,
];

#[test]
fn h6_plans_the_network_whose_routes_score() {
    let g = game("usa", 2);
    let mut state = g.new_initial_state(POINTS_SEED);
    for &action in &POINTS_PREFIX {
        state.step(action).unwrap();
    }
    assert_eq!(state.current_player(), 0);
    let miami = route(g.board, "Atlanta", "Miami");
    let params = HeuristicParams::default();
    let on_plan = |e: &ttr_core::heuristic::Explanation| {
        e.candidates
            .iter()
            .find(|c| c.action < g.space.claim_end && g.space.decode_claim(c.action).0 == miami)
            .and_then(|c| {
                c.facts
                    .iter()
                    .find(|(n, _)| *n == "on_plan")
                    .map(|&(_, v)| v)
            })
    };

    let mut h5 = Heuristic::new(Tier::H5, params, 0);
    h5.begin_game(0, POINTS_SEED);
    let e5 = h5.explain(&mut state.clone());
    assert_eq!(on_plan(&e5), Some(0.0), "H5 now plans Atlanta-Miami itself");

    let mut h6 = Heuristic::new(Tier::H6, params, 0);
    h6.begin_game(0, POINTS_SEED);
    let e6 = h6.explain(&mut state);
    assert_eq!(on_plan(&e6), Some(1.0));
    assert!(e6.chosen < g.space.claim_end);
    assert_eq!(g.space.decode_claim(e6.chosen).0, miami);
}
