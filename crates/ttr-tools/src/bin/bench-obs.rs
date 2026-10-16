//! Time the observation encoder on recorded positions, single-threaded.
//!
//! ```text
//! cargo run --release -p ttr-tools --bin bench-obs -- [map] [seats] [games] [repeats] [random|h3]
//! samply record ./target/release/bench-obs usa 2 200 20
//! ```
//!
//! Plays `games` games -- uniformly random by default, or H3 self-play, whose positions
//! (fewer tickets, longer networks) are the ones training will mostly see -- keeps every
//! position, then encodes each one for the seat to act, `repeats` times over. Single-threaded on purpose: the figure is CPU per observation,
//! which is what a batched observe divides across cores. Positions are recorded first so
//! the timing measures the encoder and nothing else.

use std::time::Instant;

use ttr_core::config::RuleConfig;
use ttr_core::heuristic::{Heuristic, HeuristicParams, Tier};
use ttr_core::obs::{encode, observation_size};
use ttr_core::rng::{Part, stream};
use ttr_core::state::{Game, State};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let map = args.first().map_or("usa", String::as_str).to_string();
    let seats: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let games: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let repeats: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let h3 = args.get(4).is_some_and(|p| p == "h3");

    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new(&map, seats).expect("a supported configuration")
    };
    let game = Game::new(cfg).expect("a supported configuration");
    let mut positions: Vec<State> = Vec::new();
    for seed in 0..games {
        let mut state = game.new_initial_state(seed);
        let mut rng = stream(seed, &[Part::Str("bench-obs")]);
        let mut agents: Vec<Heuristic> = (0..seats)
            .map(|seat| {
                let mut h = Heuristic::new(Tier::H3, HeuristicParams::default(), 0);
                h.begin_game(seat, seed);
                h
            })
            .collect();
        loop {
            positions.push(state.clone());
            if state.is_terminal() {
                break;
            }
            let action = if h3 {
                agents[usize::from(state.current_player())].act(&mut state)
            } else {
                state.sample_legal(&mut rng).expect("not over")
            };
            state.step(action).expect("legal");
        }
    }

    let mut out = vec![0.0f32; observation_size(&positions[0])];
    let mut checksum = 0.0f64;
    let started = Instant::now();
    for _ in 0..repeats {
        for state in &mut positions {
            let seat = usize::from(state.current_player()).min(seats - 1);
            encode(state, seat, &mut out);
            checksum += out.iter().map(|&x| f64::from(x)).sum::<f64>();
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let n = positions.len() * repeats;
    println!(
        "{map} {seats}P {}: {} positions x {repeats}: {:.2} us/obs single-threaded (checksum {checksum:.6e})",
        if h3 { "h3" } else { "random" },
        positions.len(),
        elapsed / n as f64 * 1e6
    );
}
