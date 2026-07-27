//! The behaviour hash: **what actually pins the Elo anchor**.
//!
//! PLAN.md §11 makes H3 the permanent zero of the rating scale. A rating recorded against
//! "H3" in March and one recorded in July are only comparable if H3 played the same way
//! both times, and nothing in a name or a parameter file guarantees that. Hashing the
//! *constants* catches an edited constant and misses everything else -- a reordered
//! tiebreak, a changed Steiner fallback threshold, a fixed bug in path reconstruction. Each
//! of those moves H3's play, and each would silently re-base every rating ever recorded.
//!
//! So the agent is identified by what it does: its action sequence over a fixed probe set,
//! blake2b-128'd, pinned as a golden literal in a test. Change a constant or change the
//! code -- if the play moves, the test fails and names the agent. You then move the anchor
//! deliberately, or you don't.
//!
//! ## What the probe set has to cover
//!
//! Phase 2 learned this the expensive way twice, and both lessons apply here:
//!
//! * **Not one map.** Deleting the end-of-turn refill survived 60 seeds of USA 2P/3P/4P and
//!   was caught at mini seed 0. A probe over the biggest board is not the widest probe.
//! * **Not openings.** Sampling only early positions leaves `remaining_cost`, `fragility`,
//!   `is_dead` and every endgame branch identically inert. These probes play games to the
//!   end, so the endgame re-weighting and the dead-ticket path are inside the hash.
//!
//! Two modes per configuration, because they exercise different halves. Self-play keeps the
//! agent on the trajectory it would actually generate; against random opponents it lands in
//! ragged positions it would never reach on its own -- blocked tickets, empty displays,
//! lopsided boards -- which is where the branches that never otherwise run live.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::actions::BLIND_SLOT;
use crate::config::RuleConfig;
use crate::hashing::{blake2b_128_state, hash128_hex};
use crate::heuristic::params::HeuristicParams;
use crate::heuristic::policy::{Heuristic, Tier};
use crate::state::Game;

/// `(map, seats, seed)` triples every behaviour hash is computed over. **Frozen**: changing
/// this list changes every agent's identity, which is the same event as moving the anchor.
pub fn probe_configs() -> &'static [(&'static str, usize, u64)] {
    &[
        ("usa", 2, 0),
        ("usa", 2, 1),
        ("usa", 2, 7),
        ("usa", 3, 3),
        ("usa", 4, 2),
        ("usa", 5, 8),
        ("mini", 2, 0),
        ("mini", 2, 5),
        ("mini", 3, 0),
        ("mini", 4, 4),
    ]
}

/// A stable identity for `(tier, params)`: hex blake2b-128 over the probe play.
///
/// Memoized, because the arena asks for it once per agent per invocation and the probe is
/// a couple of hundred milliseconds of real games.
pub fn behaviour_hash(tier: Tier, params: &HeuristicParams) -> String {
    static CACHE: OnceLock<Mutex<HashMap<(Tier, String), String>>> = OnceLock::new();
    let key = (tier, params.params_hash());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().expect("probe cache").get(&key) {
        return hit.clone();
    }
    let value = compute_behaviour_hash(tier, params);
    cache
        .lock()
        .expect("probe cache")
        .insert(key, value.clone());
    value
}

fn compute_behaviour_hash(tier: Tier, params: &HeuristicParams) -> String {
    let mut hasher = blake2b_128_state();
    hasher.update(tier.name().as_bytes());
    for &(map, n_players, seed) in probe_configs() {
        for mixed in [false, true] {
            let (actions, final_hash) = play_probe(map, n_players, seed, tier, params, mixed);
            hasher.update(&[0x1f]);
            for action in &actions {
                hasher.update(&action.to_le_bytes());
            }
            // The final position too, not just the moves: two agents can play the same
            // action ids into different games if the engine underneath them changed, and
            // an anchor that survives an engine change is not an anchor.
            hasher.update(&final_hash.to_le_bytes());
        }
    }
    hasher.finalize().to_hex().to_string()
}

/// Play one probe game. Returns the probed agent's own actions and the final `state_hash`.
///
/// `mixed` seats the agent at 0 against H0; otherwise every seat is the agent. Only the
/// probed agent's actions are recorded -- the opponents' are already implied by the final
/// hash, and recording them would make one agent's identity depend on another's move list
/// rather than on the positions it produced.
fn play_probe(
    map: &str,
    n_players: usize,
    seed: u64,
    tier: Tier,
    params: &HeuristicParams,
    mixed: bool,
) -> (Vec<u16>, u64) {
    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new(map, n_players).expect("probe configuration")
    };
    let game = Game::new(cfg).expect("probe configuration");
    let mut agents: Vec<Heuristic> = (0..n_players)
        .map(|seat| {
            let is_probe = !mixed || seat == 0;
            let t = if is_probe { tier } else { Tier::H0 };
            let mut agent = Heuristic::new(t, *params, seed);
            agent.begin_game(seat, seed);
            agent
        })
        .collect();

    let mut state = game.new_initial_state(seed);
    let mut actions = Vec::new();
    while !state.is_terminal() {
        let seat = state.current_player() as usize;
        let action = agents[seat].act(&mut state);
        if !mixed || seat == 0 {
            actions.push(action);
        }
        state
            .step(action)
            .expect("a heuristic played an illegal action");
    }
    (actions, state.state_hash())
}

// ---------------------------------------------------------------------------
// Action-type coverage
// ---------------------------------------------------------------------------

/// How often an agent reached each kind of action, per PLAN.md §14's Phase 3 criterion:
/// *every heuristic exercises every action type on both maps*.
///
/// This is the published prior art's fatal bug turned into a measurement. Its baselines
/// were lifted from a repository whose ticket appetite read `trains > 15`; on its own
/// 10-train map that condition could never fire, so every "well-designed heuristic" it
/// benchmarked against played its opening tickets and never drew another. Nothing errored.
/// The only thing that catches it is counting the action types an agent actually reaches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub claim: u32,
    /// Claims paid entirely in locomotives.
    pub claim_wild: u32,
    /// Claims on a segment that is half of a double route.
    pub claim_double: u32,
    pub draw_faceup: u32,
    pub draw_blind: u32,
    /// **The one the prior art never reached.**
    pub draw_tickets: u32,
    pub keep: u32,
    /// A keep of more than the minimum -- the choice, not the forced move.
    pub keep_extra: u32,
    pub pass: u32,
    pub games: u32,
    pub decisions: u32,
}

impl Coverage {
    /// Action types every heuristic must be able to reach, that this one did not.
    ///
    /// Three counters are recorded but deliberately **not** required, because requiring
    /// them would assert a strategy rather than detect a dead branch:
    ///
    /// * `pass` is legal only when nothing else is, so a competent agent may never see one.
    /// * `claim_wild` (paying a route entirely in locomotives) is a real choice to decline;
    ///   locomotives are the scarcest card and hoarding them is defensible play.
    /// * `keep_extra` (keeping more than the minimum offered) is H3's EV filter by
    ///   construction. H2 minimizes added track and so correctly always keeps the minimum;
    ///   demanding otherwise would be demanding H2 be H3.
    ///
    /// `draw_tickets` **is** required, on every map. That is the one the published prior
    /// art never reached.
    pub fn missing(&self) -> Vec<&'static str> {
        let checks: [(&str, u32); 5] = [
            ("claim", self.claim),
            ("claim_double", self.claim_double),
            ("draw_faceup", self.draw_faceup),
            ("draw_blind", self.draw_blind),
            ("draw_tickets", self.draw_tickets),
        ];
        checks
            .iter()
            .filter(|(_, n)| *n == 0)
            .map(|(name, _)| *name)
            .collect()
    }

    /// Fold one decision in. Public so the arena can collect the same counters
    /// per seat that the coverage sweep collects per agent -- one implementation,
    /// so an arena diagnostic and a coverage test can never disagree.
    pub fn record(&mut self, state: &crate::state::State, action: u16) {
        let space = state.space;
        let board = state.board;
        self.decisions += 1;
        if action < space.claim_end {
            let (segment, pay) = space.decode_claim(action);
            self.claim += 1;
            if pay as u8 == board.locomotive {
                self.claim_wild += 1;
            }
            if board.sibling[segment as usize] != crate::board::NO_SIBLING {
                self.claim_double += 1;
            }
        } else if action < space.draw_tickets {
            if action - space.draw_base == BLIND_SLOT {
                self.draw_blind += 1;
            } else {
                self.draw_faceup += 1;
            }
        } else if action == space.draw_tickets {
            self.draw_tickets += 1;
        } else if action < space.pass_action {
            self.keep += 1;
            let kept = (action - space.keep_base()).count_ones();
            let minimum = if state.phase == crate::config::PHASE_INITIAL_TICKETS {
                board.raw.initial_ticket_keep_min
            } else {
                board.raw.draw_ticket_keep_min
            };
            if kept > u32::from(minimum) {
                self.keep_extra += 1;
            }
        } else {
            self.pass += 1;
        }
    }
}

/// Play `seeds` self-play games and record which action types the agent reached.
pub fn coverage(
    map: &str,
    n_players: usize,
    tier: Tier,
    params: &HeuristicParams,
    seeds: impl IntoIterator<Item = u64>,
) -> Coverage {
    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new(map, n_players).expect("coverage configuration")
    };
    let game = Game::new(cfg).expect("coverage configuration");
    let mut total = Coverage::default();
    for seed in seeds {
        let mut agents: Vec<Heuristic> = (0..n_players)
            .map(|seat| {
                let mut agent = Heuristic::new(tier, *params, seed);
                agent.begin_game(seat, seed);
                agent
            })
            .collect();
        let mut state = game.new_initial_state(seed);
        while !state.is_terminal() {
            let seat = state.current_player() as usize;
            let action = agents[seat].act(&mut state);
            total.record(&state, action);
            state
                .step(action)
                .expect("a heuristic played an illegal action");
        }
        total.games += 1;
    }
    total
}

/// A short provenance string for a parameter set: `"<params_hash>"`. Recorded next to every
/// rating so a tuned agent is traceable to the numbers it ran with.
pub fn params_hash(params: &HeuristicParams) -> String {
    hash128_hex(params.canonical().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_stable_across_calls_and_distinct_per_tier() {
        let params = HeuristicParams::default();
        let a = compute_behaviour_hash(Tier::H2, &params);
        let b = compute_behaviour_hash(Tier::H2, &params);
        assert_eq!(a, b, "the probe is not deterministic");
        let mut seen = vec![a];
        for tier in [Tier::H0, Tier::H1, Tier::H3, Tier::H4] {
            let h = compute_behaviour_hash(tier, &params);
            assert!(
                !seen.contains(&h),
                "{} collides with another tier",
                tier.name()
            );
            seen.push(h);
        }
    }

    #[test]
    fn a_constant_that_changes_play_moves_the_hash() {
        // The property the anchor rests on, from the constants side.
        let base = HeuristicParams::default();
        let tuned = HeuristicParams {
            min_points_per_train: 0.2,
            ..base
        };
        assert_ne!(
            compute_behaviour_hash(Tier::H3, &base),
            compute_behaviour_hash(Tier::H3, &tuned)
        );
    }

    #[test]
    fn h0_ignores_the_params_because_it_has_none() {
        // The converse guard: a hash that moved for *every* parameter change, including
        // ones the tier cannot read, would be measuring the struct rather than the play --
        // and would force an anchor move on an H4-only retune.
        let base = HeuristicParams::default();
        let tuned = HeuristicParams {
            threat_weight: 0.0,
            ..base
        };
        assert_eq!(
            compute_behaviour_hash(Tier::H0, &base),
            compute_behaviour_hash(Tier::H0, &tuned)
        );
        assert_eq!(
            compute_behaviour_hash(Tier::H3, &base),
            compute_behaviour_hash(Tier::H3, &tuned),
            "an H4-only parameter moved H3's identity; the anchor would need re-basing for \
             a change H3 cannot even read"
        );
    }
}
