//! Final scoring, rulebook tiebreaks, and the reward the agents actually optimize.
//!
//! Route points are banked as routes are claimed; this module settles the two things that
//! can only be known at the end -- ticket completion and the longest continuous path --
//! and turns the result into a return signal.

use crate::graph::longest_trail;
use crate::numeric::compensated_sum;
use crate::state::State;

/// Two seats make the game strictly zero-sum, so the return is a plain win/draw/loss.
const HEAD_TO_HEAD: usize = 2;

/// One seat's final score, itemized. `total` is what wins the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Breakdown {
    pub routes: i16,
    pub tickets_made: u8,
    pub tickets_missed: u8,
    pub ticket_points: i16,
    pub longest_bonus: i16,
    pub longest_trail: u16,
    pub completed: u8,
    pub total: i16,
}

/// Each seat's longest continuous path, in train cars.
pub fn longest_trails(state: &State) -> Vec<u16> {
    let n_segments = state.board.n_segments;
    (0..state.n_players())
        .map(|p| longest_trail(state.board, &state.seg_owner[..n_segments], p as u8))
        .collect()
}

/// Itemized final scores for every seat.
///
/// Ticket settlement is +/- the ticket's value on whether the two cities are connected *by
/// that seat's own network*. There is no hand limit and no cap on tickets, so a seat that
/// hoarded unreachable tickets can finish well below zero.
///
/// The longest-path bonus goes to **every tied seat**, not one of them -- the rulebook is
/// explicit and implementations routinely award it to a single arbitrary winner.
///
/// Takes `&mut State` because ticket completion is a union-find query and the forest uses
/// path halving. The DSU is a cache excluded from every hash, so the mutation is invisible
/// to anything that compares states.
pub fn score_breakdown(state: &mut State) -> Vec<Breakdown> {
    let board = state.board;
    let trails = longest_trails(state);
    let best_trail = trails.iter().copied().max().unwrap_or(0);

    let mut out = Vec::with_capacity(state.n_players());
    for (p, &trail) in trails.iter().enumerate() {
        let routes = state.score[p];
        let (mut made, mut missed, mut points) = (0u8, 0u8, 0i16);
        for ticket in state.tickets_of(p) {
            let value = i16::from(board.ticket_points[ticket as usize]);
            if state.ticket_complete(p, ticket as usize) {
                made += 1;
                points += value;
            } else {
                missed += 1;
                points -= value;
            }
        }
        // A seat with no track has a trail of 0; it must not tie for the bonus.
        let bonus = if trail == best_trail && best_trail > 0 {
            i16::from(board.raw.longest_bonus)
        } else {
            0
        };
        out.push(Breakdown {
            routes,
            tickets_made: made,
            tickets_missed: missed,
            ticket_points: points,
            longest_bonus: bonus,
            longest_trail: trail,
            completed: made,
            total: routes + points + bonus,
        });
    }
    out
}

pub fn final_scores(state: &mut State) -> Vec<i16> {
    score_breakdown(state).iter().map(|b| b.total).collect()
}

/// The full rulebook ordering, highest first: points, then tickets, then longest path.
///
/// A true draw is still possible -- two seats can match on all three -- and the engine
/// reports it rather than inventing a winner.
pub fn rank_key(b: &Breakdown) -> (i16, u8, i16) {
    (b.total, b.completed, b.longest_bonus)
}

/// Every seat that ties for first under the full tiebreak chain.
pub fn winners(state: &mut State) -> Vec<usize> {
    let keys: Vec<_> = score_breakdown(state).iter().map(rank_key).collect();
    let best = *keys.iter().max().expect("at least one seat");
    (0..keys.len()).filter(|&p| keys[p] == best).collect()
}

/// The reward. Terminal only, optimizing **win probability**, not score.
///
/// The one published attempt at this problem found its score-optimizing agents lost
/// head-to-head to its win-optimizing self-play agent, despite scoring more points. Raw
/// route points is worse still: it directly incentivizes long-route greed over ticket
/// completion and is the leading cause of the "agent never draws tickets" failure.
///
/// * **2P** -- `+1 / 0 / -1` with the full tiebreaks, so a true draw scores 0 for both.
/// * **3-5P** -- `0.75 * win + 0.25 * rank`, both terms recentred to sum to zero so
///   self-play stays well-behaved and MCTS value backup stays valid without a per-player
///   baseline. The rank term keeps a losing seat's gradient informative instead of flat.
pub fn returns(state: &mut State) -> Vec<f64> {
    let n = state.n_players();
    let keys: Vec<_> = score_breakdown(state).iter().map(rank_key).collect();
    let best = *keys.iter().max().expect("at least one seat");
    let top: Vec<usize> = (0..n).filter(|&p| keys[p] == best).collect();

    if n == HEAD_TO_HEAD {
        if top.len() == HEAD_TO_HEAD {
            return vec![0.0, 0.0];
        }
        return (0..n)
            .map(|p| if top.contains(&p) { 1.0 } else { -1.0 })
            .collect();
    }

    let win: Vec<f64> = (0..n)
        .map(|p| (if top.contains(&p) { 1.0 } else { 0.0 }) / top.len() as f64)
        .collect();

    // rank_norm: 1.0 for the best key, 0.0 for the worst, averaged over ties. A stable
    // sort, so equal keys keep ascending seat order exactly as Python's `sorted` does.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&p| keys[p]);
    let mut rank_of = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && keys[order[j + 1]] == keys[order[i]] {
            j += 1;
        }
        let shared = (i + j) as f64 / 2.0 / (n - 1) as f64;
        for &p in &order[i..=j] {
            rank_of[p] = shared;
        }
        i = j + 1;
    }

    // `compensated_sum`, not `iter().sum()`: CPython's builtin `sum()` compensates and
    // Rust's does not, and the two disagree in the last ULP on inputs as ordinary as
    // [0.0, 1.0, 2/3, 1/3]. See crate::numeric.
    let mean_win: f64 = compensated_sum(&win) / n as f64;
    let mean_rank: f64 = compensated_sum(&rank_of) / n as f64;
    (0..n)
        .map(|p| 0.75 * (win[p] - mean_win) + 0.25 * (rank_of[p] - mean_rank))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::rng::{Part, stream};
    use crate::state::Game;

    fn finished(map: &str, n: usize, seed: u64) -> State {
        let game = Game::new(RuleConfig::new(map, n).unwrap()).unwrap();
        let mut state = game.new_initial_state(seed);
        let mut rng = stream(seed, &[Part::Str("score")]);
        let mut scratch = Vec::new();
        while !state.is_terminal() {
            let action = state.sample_legal_into(&mut rng, &mut scratch).unwrap();
            state.step(action).unwrap();
        }
        state
    }

    #[test]
    fn breakdown_totals_are_self_consistent() {
        for seed in 0..10 {
            let mut state = finished("usa", 3, seed);
            for b in score_breakdown(&mut state) {
                assert_eq!(b.total, b.routes + b.ticket_points + b.longest_bonus);
                assert_eq!(b.completed, b.tickets_made);
            }
        }
    }

    #[test]
    fn the_longest_path_bonus_goes_to_every_tied_seat() {
        // The rulebook is explicit and implementations routinely award it to one arbitrary
        // winner. Constructed rather than hoped for: give two seats identical trails.
        for seed in 0..40u64 {
            let mut state = finished("mini", 2, seed);
            let trails = longest_trails(&state);
            if trails[0] == trails[1] && trails[0] > 0 {
                let breakdown = score_breakdown(&mut state);
                assert!(
                    breakdown.iter().all(|b| b.longest_bonus > 0),
                    "a tied longest trail must pay both seats"
                );
                return;
            }
        }
        panic!("no seed in the sweep produced a tied longest trail; the branch is untested");
    }

    #[test]
    fn a_seat_with_no_track_never_ties_for_the_bonus() {
        let game = Game::new(RuleConfig::new("usa", 2).unwrap()).unwrap();
        let mut state = game.new_initial_state(0);
        // Nobody has claimed anything, so every trail is 0 and nobody may take the bonus.
        assert!(
            score_breakdown(&mut state)
                .iter()
                .all(|b| b.longest_bonus == 0)
        );
    }

    #[test]
    fn two_player_returns_are_zero_sum_and_admit_a_draw() {
        for seed in 0..20 {
            let mut state = finished("usa", 2, seed);
            let r = returns(&mut state);
            assert!(
                (r[0] + r[1]).abs() < 1e-12,
                "2P returns must sum to zero: {r:?}"
            );
            assert!(r.iter().all(|v| [-1.0, 0.0, 1.0].contains(v)), "{r:?}");
        }
    }

    #[test]
    fn multiplayer_returns_are_constant_sum() {
        // What "constant-sum" buys: value backup in self-play stays valid with no
        // per-player baseline. Recentring both terms is what makes it true.
        for n in 3..=5 {
            for seed in 0..10 {
                let mut state = finished("usa", n, seed);
                let r = returns(&mut state);
                let total: f64 = r.iter().sum();
                assert!(
                    total.abs() < 1e-9,
                    "{n}P seed {seed} returns sum to {total}: {r:?}"
                );
            }
        }
    }

    #[test]
    fn the_winner_is_the_seat_with_the_best_rank_key() {
        for seed in 0..10 {
            let mut state = finished("usa", 4, seed);
            let breakdown = score_breakdown(&mut state);
            let champions = winners(&mut state);
            let best = champions
                .iter()
                .map(|&p| rank_key(&breakdown[p]))
                .max()
                .unwrap();
            for (p, b) in breakdown.iter().enumerate() {
                if champions.contains(&p) {
                    assert_eq!(rank_key(b), best);
                } else {
                    assert!(rank_key(b) < best);
                }
            }
        }
    }
}
