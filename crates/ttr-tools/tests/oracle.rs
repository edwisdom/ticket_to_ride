//! The live crate against the frozen `oracle-v1` build, compared at **every** step.
//!
//! PLAN.md §14.1, Phase 3.1 step 2: the Python differential harness rewritten as
//! new-versus-frozen Rust. It keeps the one property that made the old harness worth
//! having -- a divergence names its step, its action and the first serialized field that
//! differs -- and costs nothing to maintain, because nobody edits a pinned commit.
//!
//! Compared after every step: `state_hash`, `position_hash`, the sorted legal actions,
//! `is_terminal`, `current_player`, and the acting seat's observation; every seat's
//! observation every `ALL_SEATS_EVERY` steps and at the terminal; and the whole terminal
//! settlement (breakdown, scores, winners, returns bitwise, longest trails).
//!
//! Two drives. **Random**, uniform over the sorted legal list from the seed's own stream,
//! which reaches the rules' corners. **Heuristic**, where live and frozen agents each pick
//! a move at every decision and must pick the same one -- the heuristics share the graph
//! code Phase 3.2 rewrites, and this is H3's behaviour hash asserted over thousands of
//! positions instead of ten.
//!
//! The fast tier runs under plain `cargo test`. The full sweep is `#[ignore]`d:
//! `cargo test --release -p ttr-tools --test oracle -- --ignored`.

use rayon::prelude::*;
use ttr_core::heuristic::{Heuristic, HeuristicParams, Tier};
use ttr_core::rng::{Part, stream};
use ttr_tools::corpus::{ObsMask, configs};
use ttr_tools::diff::{compare_observation, compare_position};
use ttr_tools::engine::{Engine, Live};

ttr_tools::engine_for!(
    /// The build frozen as `oracle-v1`.
    Oracle,
    ttr_oracle
);

/// Every seat's observation is compared this often, and at the terminal. The acting
/// seat's is compared at every step.
const ALL_SEATS_EVERY: usize = 7;

/// Which configurations the fast tier drives, and how many seeds each.
///
/// The shape of the Python harness's `FAST_CONFIGS`, for its measured reason: deleting the
/// end-of-turn refill survived 60 seeds of USA 2P-4P and was caught at mini seed 0 and USA
/// 5P seed 8, because reaching it needs the deck *and* the discard to run dry. The small
/// map and the full table are weighted accordingly.
const FAST: &[(&str, usize, u64)] = &[
    ("mini", 2, 30),
    ("mini", 3, 30),
    ("mini", 4, 30),
    ("usa", 5, 12),
    ("usa", 2, 8),
    ("usa", 3, 8),
    ("usa", 4, 8),
];

const TIERS: [Tier; 5] = [Tier::H3, Tier::H4, Tier::H2, Tier::H1, Tier::H0];

fn oracle_tier(tier: Tier) -> ttr_oracle::heuristic::Tier {
    use ttr_oracle::heuristic::Tier as T;
    match tier {
        Tier::H0 => T::H0,
        Tier::H1 => T::H1,
        Tier::H2 => T::H2,
        Tier::H3 => T::H3,
        Tier::H4 => T::H4,
    }
}

#[derive(Clone, Copy, Debug)]
enum Drive {
    Random,
    /// Seat `s` plays `TIERS[(s + offset) % 5]`.
    Heuristic(usize),
}

struct Seats {
    live: Vec<Heuristic>,
    oracle: Vec<ttr_oracle::heuristic::Heuristic>,
}

impl Seats {
    fn new(n: usize, offset: usize, seed: u64) -> Self {
        let tier = |seat: usize| TIERS[(seat + offset) % TIERS.len()];
        Self {
            live: (0..n)
                .map(|seat| {
                    let mut h = Heuristic::new(tier(seat), HeuristicParams::default(), 0);
                    h.begin_game(seat, seed);
                    h
                })
                .collect(),
            oracle: (0..n)
                .map(|seat| {
                    let mut h = ttr_oracle::heuristic::Heuristic::new(
                        oracle_tier(tier(seat)),
                        ttr_oracle::heuristic::HeuristicParams::default(),
                        0,
                    );
                    h.begin_game(seat, seed);
                    h
                })
                .collect(),
        }
    }
}

/// Play one game in both engines with identical actions, comparing at every step.
fn compare_game(map: &str, n: usize, seed: u64, drive: Drive) -> Result<usize, String> {
    let (live, oracle) = (Live::new(map, n), Oracle::new(map, n));
    let mask = ObsMask::for_map(map);
    let size = live.obs_size();
    if oracle.obs_size() != size {
        return Err(format!("observation sizes {size} vs {}", oracle.obs_size()));
    }
    let (mut buf_a, mut buf_b) = (vec![0.0f32; size], vec![0.0f32; size]);
    let mut a = live.initial(seed);
    let mut b = oracle.initial(seed);
    let mut policy = stream(seed, &[Part::Str("differential"), Part::Str("policy")]);
    let mut seats = match drive {
        Drive::Heuristic(offset) => Some(Seats::new(n, offset, seed)),
        Drive::Random => None,
    };

    let mut step = 0usize;
    let mut last: Option<u16> = None;
    let fail = |step: usize, last: Option<u16>, turn: u16, detail: String| {
        let at = match last {
            None => "at setup".to_string(),
            Some(action) => format!("after step {step} (action {action})"),
        };
        format!(
            "{map} {n}P seed {seed} {drive:?} diverged {at}, turn {turn}: {detail}\n  \
             reproduce: compare_game({map:?}, {n}, {seed}, Drive::{drive:?})"
        )
    };
    loop {
        compare_position(&live, &a, &oracle, &b).map_err(|e| fail(step, last, a.turn, e))?;
        let done = live.is_terminal(&a);
        let everyone = done || step % ALL_SEATS_EVERY == 0;
        let seats_to_check: Vec<usize> = if everyone {
            (0..n).collect()
        } else {
            vec![live.current_player(&a)]
        };
        for p in seats_to_check {
            compare_observation(
                &live, &mut a, &oracle, &mut b, p, &mask, &mut buf_a, &mut buf_b,
            )
            .map_err(|e| fail(step, last, a.turn, e))?;
        }
        if done {
            break;
        }

        let action = match seats.as_mut() {
            None => {
                let legal = live.legal(&a);
                legal[policy.below(legal.len() as u32) as usize]
            }
            Some(s) => {
                let seat = live.current_player(&a);
                let mine = s.live[seat].act(&mut a);
                let frozen = s.oracle[seat].act(&mut b);
                if mine != frozen {
                    return Err(fail(
                        step,
                        last,
                        a.turn,
                        format!("seat {seat}'s agent chose {mine}, the frozen one {frozen}"),
                    ));
                }
                mine
            }
        };
        live.step(&mut a, action);
        oracle.step(&mut b, action);
        step += 1;
        last = Some(action);
    }

    let (ta, tb) = (live.terminal(&mut a), oracle.terminal(&mut b));
    if ta != tb {
        return Err(fail(
            step,
            last,
            a.turn,
            format!("terminal settlement: {ta:?} vs {tb:?}"),
        ));
    }
    let bits = |t: &ttr_tools::engine::Terminal| -> Vec<u64> {
        t.returns.iter().map(|r| r.to_bits()).collect()
    };
    if bits(&ta) != bits(&tb) {
        return Err(fail(
            step,
            last,
            a.turn,
            "returns differ in their bits".into(),
        ));
    }
    Ok(step)
}

fn run(jobs: Vec<(&'static str, usize, u64, Drive)>) {
    let results: Vec<Result<usize, String>> = jobs
        .par_iter()
        .map(|&(map, n, seed, drive)| compare_game(map, n, seed, drive))
        .collect();
    let failures: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    assert!(
        failures.is_empty(),
        "{} of {} games diverged; first:\n{}",
        failures.len(),
        jobs.len(),
        failures[0]
    );
}

#[test]
fn the_live_engine_matches_oracle_v1_at_every_step() {
    let jobs = FAST
        .iter()
        .flat_map(|&(map, n, seeds)| (0..seeds).map(move |s| (map, n, s, Drive::Random)))
        .collect();
    run(jobs);
}

#[test]
fn every_heuristic_plays_exactly_as_its_frozen_self() {
    // Two games per configuration per lineup offset, so every tier sits in every seat
    // across the set.
    let jobs = configs()
        .into_iter()
        .flat_map(|(map, n)| {
            (0..TIERS.len()).flat_map(move |offset| {
                (0..2u64).map(move |s| (map, n, 100 + s, Drive::Heuristic(offset)))
            })
        })
        .collect();
    run(jobs);
}

#[test]
fn the_comparison_can_actually_fail() {
    // Guard the guard: a harness that cannot report a difference manufactures confidence.
    // Two different seeds must be reported, and reported *with the field*.
    let (live, oracle) = (Live::new("usa", 2), Oracle::new("usa", 2));
    let a = live.initial(1);
    let b = oracle.initial(2);
    let err = compare_position(&live, &a, &oracle, &b).expect_err("different seeds compared equal");
    assert!(err.contains("state_hash"), "{err}");
    assert!(err.contains("first differing field is"), "{err}");

    let mask = ObsMask::for_map("usa");
    let size = live.obs_size();
    let (mut a, mut b) = (live.initial(1), oracle.initial(2));
    let err = compare_observation(
        &live,
        &mut a,
        &oracle,
        &mut b,
        0,
        &mask,
        &mut vec![0.0; size],
        &mut vec![0.0; size],
    )
    .expect_err("different positions encoded identically");
    assert!(err.contains("slots differ"), "{err}");
}

#[test]
fn the_oracle_is_the_build_it_claims_to_be() {
    // The pin is a commit, so this cannot drift by accident -- but a wrong SHA pasted into
    // Cargo.toml would still compile. Both engines must report the versions the corpus was
    // signed under.
    assert_eq!(ttr_oracle::CONTRACT_VERSION, ttr_core::CONTRACT_VERSION);
    assert_eq!(ttr_oracle::obs::obs_version(), ttr_core::obs::obs_version());
}

#[test]
#[ignore = "the full sweep: run with --release -- --ignored"]
fn full_sweep_random() {
    let seeds: u64 = std::env::var("TTR_ORACLE_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let jobs = configs()
        .into_iter()
        .flat_map(|(map, n)| (0..seeds).map(move |s| (map, n, s, Drive::Random)))
        .collect();
    run(jobs);
}

#[test]
#[ignore = "the full sweep: run with --release -- --ignored"]
fn full_sweep_heuristic() {
    let jobs = configs()
        .into_iter()
        .flat_map(|(map, n)| {
            (0..TIERS.len()).flat_map(move |offset| {
                (0..100u64).map(move |s| (map, n, s, Drive::Heuristic(offset)))
            })
        })
        .collect();
    run(jobs);
}
