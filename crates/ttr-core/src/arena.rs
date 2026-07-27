//! Paired match play, run entirely inside Rust.
//!
//! **Python hands down a whole schedule and gets back a table.** The same reasoning as
//! [`crate::vecenv`], one level up: a per-decision FFI call costs a few microseconds, and an
//! H3 decision costs about twenty-five, so a Python-driven arena loop would spend a fifth of
//! its time crossing the boundary and would not parallelise at all under the GIL. One call
//! plays every game across a rayon pool and returns two flat columnar tables that go
//! straight into SQLite.
//!
//! ## The seed block is the unit of work *and* the unit of analysis
//!
//! PLAN.md §11. A block fixes the deck permutation, the ticket permutation and the initial
//! deals; rotation `r` seats lineup entry `i` at seat `(i + r) % P`. Every rotation of a
//! block is played, so each agent sits in each seat equally often and first-player advantage
//! cancels rather than being folded into the result.
//!
//! **Both seatings are measured, never mirrored.** A win-rate matrix whose off-diagonal
//! pairs sum to exactly 1.00 reports one seating and its complement, with first-player
//! advantage confounded into every cell -- a real flaw in the published prior art (§1).
//!
//! The correlation this creates is the point, and it is also the trap: games inside a block
//! share a deck, so they are *not* independent. Everything downstream aggregates to the
//! block before computing a statistic, and the bootstrap resamples blocks. Treating paired
//! games as independent inflates significance by about sqrt(P).

use rayon::prelude::*;

use crate::heuristic::params::HeuristicParams;
use crate::heuristic::policy::{Heuristic, Tier};
use crate::heuristic::probe::Coverage;
use crate::scoring::{rank_key, returns, score_breakdown};
use crate::state::Game;

/// One competitor: a tier, its constants, and the root of its own random stream.
#[derive(Clone, Copy, Debug)]
pub struct AgentSpec {
    pub tier: Tier,
    pub params: HeuristicParams,
    pub seed: u64,
}

/// What one seat did in one game.
#[derive(Clone, Debug)]
pub struct SeatOutcome {
    /// Index into the arena's agent list -- *not* the seat, which is the position in
    /// [`GameOutcome::seats`]. Confusing the two is how a rotated result gets attributed to
    /// the wrong agent, which no assertion downstream would catch.
    pub agent: u16,
    pub score: i16,
    pub ret: f64,
    /// Competition ranking under the full rulebook tiebreak chain: ties share a rank and
    /// the next rank skips.
    pub rank: u8,
    pub won: bool,
    pub tickets_kept: u8,
    pub tickets_made: u8,
    pub ticket_points: i16,
    pub routes_claimed: u16,
    pub trains_left: u8,
    pub cards_left: u16,
    pub longest_trail: u16,
    pub longest_bonus: i16,
    pub coverage: Coverage,
}

/// One finished game.
#[derive(Clone, Debug)]
pub struct GameOutcome {
    pub block_seed: u64,
    pub rotation: u8,
    pub turns: u16,
    /// Decisions taken across every seat. The denominator for per-decision cost, which is
    /// what sets the sim budget of every Phase 5 search -- H3 is the ISMCTS rollout policy.
    pub decisions: u32,
    /// The final `state_hash`. Stored so a re-run is *verifiable* rather than merely
    /// repeatable: two runs of the same seed root that agree on ratings but disagree here
    /// played different games and happened to tie.
    pub final_hash: u64,
    pub seats: Vec<SeatOutcome>,
}

/// Play every rotation of every block.
///
/// `lineup` names which agents play, one entry per seat before rotation. `blocks` seed
/// blocks are played starting at `seed_root`, each in `n_players` rotations.
///
/// Block seeds are `seed_root + i`, deliberately: extending a run from 500 to 1000 blocks
/// replays the first 500 decks exactly, so results accumulate instead of being replaced.
pub fn run(
    game: &Game,
    agents: &[AgentSpec],
    lineup: &[u16],
    seed_root: u64,
    blocks: u64,
    threads: usize,
) -> Vec<GameOutcome> {
    let seats = game.rules.n_players as usize;
    assert_eq!(lineup.len(), seats, "the lineup must fill every seat");
    assert!(
        lineup.iter().all(|&a| (a as usize) < agents.len()),
        "a lineup entry names an agent that is not in the list"
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .expect("building a rayon pool");
    pool.install(|| {
        (0..blocks)
            .into_par_iter()
            .flat_map_iter(|b| {
                let block_seed = seed_root + b;
                (0..seats).map(move |r| play_one(game, agents, lineup, block_seed, r as u8))
            })
            .collect()
    })
}

fn play_one(
    game: &Game,
    agents: &[AgentSpec],
    lineup: &[u16],
    block_seed: u64,
    rotation: u8,
) -> GameOutcome {
    let seats = game.rules.n_players as usize;
    // Seat `s` is played by lineup entry `(s + seats - r) % seats`, the inverse of "entry i
    // sits at (i + r) % seats". Written as the inverse rather than by building a permutation
    // so there is one place to get it wrong instead of two.
    let at: Vec<u16> = (0..seats)
        .map(|s| lineup[(s + seats - rotation as usize) % seats])
        .collect();

    let mut players: Vec<Heuristic> = at
        .iter()
        .enumerate()
        .map(|(seat, &a)| {
            let spec = agents[a as usize];
            let mut h = Heuristic::new(spec.tier, spec.params, spec.seed);
            h.begin_game(seat, block_seed);
            h
        })
        .collect();

    let mut state = game.new_initial_state(block_seed);
    let mut coverage = vec![Coverage::default(); seats];
    let mut decisions = 0u32;
    while !state.is_terminal() {
        let seat = state.current_player() as usize;
        let action = players[seat].act(&mut state);
        decisions += 1;
        coverage[seat].record(&state, action);
        state
            .step(action)
            .expect("a heuristic played an illegal action");
    }

    let breakdown = score_breakdown(&mut state);
    let rets = returns(&mut state);
    let keys: Vec<_> = breakdown.iter().map(rank_key).collect();
    let board = state.board;

    let outcomes = (0..seats)
        .map(|s| {
            let b = breakdown[s];
            // Competition ranking: one plus the number of seats strictly ahead.
            let rank = 1 + keys.iter().filter(|k| **k > keys[s]).count() as u8;
            SeatOutcome {
                agent: at[s],
                score: b.total,
                ret: rets[s],
                rank,
                won: rank == 1,
                tickets_kept: b.tickets_made + b.tickets_missed,
                tickets_made: b.tickets_made,
                ticket_points: b.ticket_points,
                routes_claimed: (0..board.n_segments)
                    .filter(|&seg| state.seg_owner[seg] == s as u8)
                    .count() as u16,
                trains_left: state.trains[s],
                cards_left: state.hand_size(s) as u16,
                longest_trail: b.longest_trail,
                longest_bonus: b.longest_bonus,
                coverage: coverage[s],
            }
        })
        .collect();

    GameOutcome {
        block_seed,
        rotation,
        turns: state.turn,
        decisions,
        final_hash: state.state_hash(),
        seats: outcomes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;

    fn game(map: &str, n: usize) -> Game {
        Game::new(RuleConfig {
            track_history: false,
            ..RuleConfig::new(map, n).unwrap()
        })
        .unwrap()
    }

    fn specs(tiers: &[Tier]) -> Vec<AgentSpec> {
        tiers
            .iter()
            .map(|&tier| AgentSpec {
                tier,
                params: HeuristicParams::default(),
                seed: 99,
            })
            .collect()
    }

    #[test]
    fn every_rotation_of_every_block_is_played_once() {
        let g = game("mini", 3);
        let agents = specs(&[Tier::H1, Tier::H2, Tier::H3]);
        let out = run(&g, &agents, &[0, 1, 2], 100, 4, 2);
        assert_eq!(out.len(), 4 * 3);
        let mut seen: Vec<(u64, u8)> = out.iter().map(|g| (g.block_seed, g.rotation)).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 12, "a (block, rotation) was played twice");
    }

    #[test]
    fn each_agent_sits_in_each_seat_exactly_once_per_block() {
        // The property paired evaluation rests on. If it fails, first-player advantage does
        // not cancel and every rating carries it.
        let g = game("usa", 4);
        let agents = specs(&[Tier::H0, Tier::H1, Tier::H2, Tier::H3]);
        let out = run(&g, &agents, &[0, 1, 2, 3], 7, 1, 1);
        let mut seats_of = vec![Vec::new(); 4];
        for game_out in &out {
            for (seat, s) in game_out.seats.iter().enumerate() {
                seats_of[s.agent as usize].push(seat);
            }
        }
        for (agent, mut seats) in seats_of.into_iter().enumerate() {
            seats.sort_unstable();
            assert_eq!(
                seats,
                vec![0, 1, 2, 3],
                "agent {agent} did not visit every seat"
            );
        }
    }

    #[test]
    fn the_same_seed_root_reproduces_the_same_games() {
        // "Leaderboard stable across two runs of the same seed root" (PLAN.md §14) reduces
        // to this, and it must hold across thread counts too -- a result that depends on
        // how work was scheduled is not reproducible, it is merely repeatable on one box.
        let g = game("mini", 2);
        let agents = specs(&[Tier::H2, Tier::H3]);
        let a = run(&g, &agents, &[0, 1], 42, 6, 1);
        let b = run(&g, &agents, &[0, 1], 42, 6, 4);
        let key = |v: &Vec<GameOutcome>| {
            let mut k: Vec<(u64, u8, u64, i16)> = v
                .iter()
                .map(|g| (g.block_seed, g.rotation, g.final_hash, g.seats[0].score))
                .collect();
            k.sort_unstable();
            k
        };
        assert_eq!(key(&a), key(&b));
    }

    #[test]
    fn ranks_and_returns_agree_about_who_won() {
        let g = game("usa", 3);
        let agents = specs(&[Tier::H1, Tier::H2, Tier::H3]);
        for out in run(&g, &agents, &[0, 1, 2], 5, 3, 2) {
            let winners: Vec<usize> = (0..3).filter(|&s| out.seats[s].won).collect();
            assert!(!winners.is_empty(), "a finished game with no winner");
            let best = out
                .seats
                .iter()
                .map(|s| s.ret)
                .fold(f64::NEG_INFINITY, f64::max);
            for &w in &winners {
                assert_eq!(out.seats[w].rank, 1);
                assert!(
                    (out.seats[w].ret - best).abs() < 1e-12,
                    "a rank-1 seat did not have the best return"
                );
            }
        }
    }

    #[test]
    fn a_rotated_result_is_attributed_to_the_agent_not_the_seat() {
        // The bug this catches is silent and total: with the inverse permutation written
        // backwards, every rating is the *other* agent's. H0 against H3 is lopsided enough
        // that a swap is unmistakable.
        let g = game("usa", 2);
        let agents = specs(&[Tier::H0, Tier::H3]);
        let out = run(&g, &agents, &[0, 1], 0, 12, 2);
        let mut mean = [0.0; 2];
        for game_out in &out {
            for s in &game_out.seats {
                mean[s.agent as usize] += s.ret / 12.0 / 2.0;
            }
        }
        assert!(
            mean[1] > 0.8 && mean[0] < -0.8,
            "H3 scored {:+.2} and random scored {:+.2}; the rotation is inverted",
            mean[1],
            mean[0]
        );
    }
}
