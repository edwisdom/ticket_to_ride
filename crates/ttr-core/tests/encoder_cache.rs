//! The encoder cache (PLAN.md §14.1, Phase 3.2): `encode_cached` writes exactly what
//! `encode` writes, at every position, and it actually hits.
//!
//! Two checks stack here. Each test compares the cached buffer with a from-scratch one bit
//! for bit, in any build. In debug builds -- the test profile -- the cache also recomputes
//! its stored graph work on every hit and asserts that it agrees, so a stale entry fails at
//! the lookup that served it, not just where it happens to move a float.
//!
//! The games mix configurations, uniformly random play and H3 self-play (fewer tickets,
//! longer networks), and every few positions encode for each seat rather than only the one
//! to act, as `observe_seat` does. One cache serves a run of consecutive games, so entries
//! left over from the previous game -- on another board, another seat count -- meet the
//! next one's positions, as they do in a `VecEnv` after `auto_reset`.

mod common;

use common::{all_configs, game, policy};
use rayon::prelude::*;
use ttr_core::heuristic::{Heuristic, HeuristicParams, Tier};
use ttr_core::obs::{EncoderCache, encode, encode_cached, observation_size};
use ttr_core::state::Game;

/// Every fifth game is H3 self-play; the rest are uniformly random.
const H3_EVERY: u64 = 5;

/// How often a position is encoded for every seat, not just the one to act.
const ALL_SEATS_EVERY: usize = 3;

/// Games per cache: one cache serves this many consecutive seeds.
const GAMES_PER_CACHE: u64 = 25;

#[derive(Default)]
struct Tally {
    games: u64,
    lookups: u64,
    hits: u64,
}

impl Tally {
    fn add(mut self, other: Self) -> Self {
        self.games += other.games;
        self.lookups += other.lookups;
        self.hits += other.hits;
        self
    }
}

/// Play `seeds`, cycling through every configuration, and compare every cached encoding
/// with a from-scratch one.
fn check(seeds: std::ops::Range<u64>) -> Tally {
    let configs = all_configs();
    let games: Vec<Game> = configs.iter().map(|&(map, n)| game(map, n)).collect();
    let starts: Vec<u64> = seeds.clone().step_by(GAMES_PER_CACHE as usize).collect();
    starts
        .into_par_iter()
        .map(|start| {
            let mut cache = EncoderCache::new();
            let mut tally = Tally::default();
            for seed in start..(start + GAMES_PER_CACHE).min(seeds.end) {
                play(&games[seed as usize % games.len()], seed, &mut cache);
                tally.games += 1;
            }
            tally.lookups = cache.hits() + cache.misses();
            tally.hits = cache.hits();
            tally
        })
        .reduce(Tally::default, Tally::add)
}

fn play(game: &Game, seed: u64, cache: &mut EncoderCache) {
    let mut state = game.new_initial_state(seed);
    let n = state.n_players();
    let mut rng = policy(seed);
    let mut agents: Option<Vec<Heuristic>> = (seed % H3_EVERY == 0).then(|| {
        (0..n)
            .map(|seat| {
                let mut agent = Heuristic::new(Tier::H3, HeuristicParams::default(), 0);
                agent.begin_game(seat, seed);
                agent
            })
            .collect()
    });
    let size = observation_size(&state);
    let (mut cached, mut fresh) = (vec![0.0f32; size], vec![0.0f32; size]);
    let mut position = 0;
    while !state.is_terminal() {
        let to_act = usize::from(state.current_player());
        let seats = if position % ALL_SEATS_EVERY == 0 {
            0..n
        } else {
            to_act..to_act + 1
        };
        for seat in seats {
            encode_cached(&state, seat, &mut cached, cache);
            encode(&state, seat, &mut fresh);
            if let Some(i) = (0..size).find(|&i| cached[i].to_bits() != fresh[i].to_bits()) {
                panic!(
                    "{} {n}P seed {seed}, position {position}, seat {seat}: slot {i} is {} \
                     cached and {} recomputed",
                    state.board.name, cached[i], fresh[i]
                );
            }
        }
        let action = match agents.as_mut() {
            Some(agents) => agents[to_act].act(&mut state),
            None => state.sample_legal(&mut rng).expect("not over"),
        };
        state.step(action).expect("a chosen action is legal");
        position += 1;
    }
}

fn assert_exercised(tally: &Tally) {
    // Measured hit rates are 45-64% by configuration and policy; well under that means the
    // key changed under the cache and it stopped doing anything.
    let rate = tally.hits as f64 / tally.lookups as f64;
    assert!(
        rate > 0.3,
        "{} hits of {} lookups over {} games",
        tally.hits,
        tally.lookups,
        tally.games
    );
}

#[test]
fn cached_encodings_equal_recomputed_ones() {
    let tally = check(0..150);
    assert_eq!(tally.games, 150);
    assert_exercised(&tally);
}

/// The 10k-game exercise PLAN.md §14.1 asks of every cache. Nightly, or
/// `cargo test -p ttr-core --test encoder_cache -- --ignored` (the debug-build assertion
/// inside the cache runs only without `--release`).
#[test]
#[ignore = "10k games; nightly"]
fn cached_encodings_equal_recomputed_ones_over_10k_games() {
    let tally = check(0..10_000);
    assert_eq!(tally.games, 10_000);
    assert_exercised(&tally);
}
