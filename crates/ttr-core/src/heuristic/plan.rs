//! The shared plan: "which unclaimed track do I still need, and what would losing it cost?"
//!
//! H2, H3 and H4 all reason about the same object, so it is computed once and cached rather
//! than three times. The cache key is `(board_version, ticket mask)`, which is exactly when
//! the answer can change: `State::board_version` is bumped by every claim by anyone, and a
//! seat's ticket mask by every keep. Nothing else moves the plan -- drawing cards changes
//! what you can *afford*, not what you need.
//!
//! That caching is not a micro-optimization. H3 is the ISMCTS rollout policy from Phase 5
//! (PLAN.md §7, §8.3), so its per-decision cost sets the sim budget of every later search.
//! A plan rebuild is a few dozen Dijkstras; a rollout that rebuilt per step would be two
//! orders of magnitude off. Between claims, every decision reuses one build.
//!
//! ## Why the plan is a Steiner *structure*, not per-ticket shortest paths
//!
//! Summing each ticket's shortest path double-counts shared trunk line: a player holding
//! Seattle-New York and Portland-New York looks like it needs two transcontinentals when it
//! needs one and a spur. [`crate::graph::steiner_cost_exact`] gives the honest number, and
//! the edge set here is built by the shortest-path heuristic -- attach the cheapest
//! unconnected terminal to the tree, then price the segments just added at zero so the next
//! attachment shares them. That sharing is the whole point, and it is what makes the
//! per-segment costs below mean "cost to *my plan*" rather than "cost to one ticket".

use crate::board::{Board, MAX_CARD_TYPES, MAX_CITIES, NO_SIBLING, UNREACHABLE};
use crate::config::{CLOSED, FREE};
use crate::graph::{edge_cost, steiner_cost_exact};
use crate::state::State;

/// No predecessor, in the Dijkstra scratch tables.
const NO_PRED: u8 = u8::MAX;

/// A per-seat plan over the board as it currently stands.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Cache key: `State::board_version` when this was built.
    pub board_version: u32,
    /// Cache key: the seat's ticket mask when this was built.
    pub tickets: u32,
    pub player: u8,

    /// Endpoints of held tickets that are not yet connected.
    pub terminals: Vec<u8>,
    /// Train cars still needed to connect every terminal, from
    /// [`crate::graph::steiner_cost_exact`].
    pub steiner_cost: u32,
    /// False when the terminal count forced the MST upper bound instead of the exact DP.
    pub exact: bool,
    /// Unclaimed segments on the plan, ascending.
    pub segments: Vec<u16>,
    /// Held tickets whose endpoints my network can no longer reach at any price.
    pub dead_tickets: Vec<u8>,
    /// Face value of the held tickets that are still incomplete and still reachable -- the
    /// points actually riding on this plan. Completed tickets are already banked and dead
    /// ones are already lost, so neither belongs in what a car of progress is worth.
    pub live_points: u32,
    /// Segments this seat owned when the plan was built. See [`Plan::survives`].
    own_segments: usize,
    /// Cars the plan still needs of each colour; `[locomotive]` counts gray demand, which
    /// any colour can satisfy.
    pub need: [u16; MAX_CARD_TYPES],

    /// Detour cost per plan segment, parallel to `segments`. `u16::MAX` marks "not computed
    /// yet" -- these are filled lazily, because a decision usually compares two or three
    /// candidates and pricing all twenty-five would dominate the turn.
    detour: Vec<u16>,
}

impl Plan {
    /// Build the plan for `player` from the position as it stands.
    pub fn build(state: &mut State, player: usize) -> Self {
        let board = state.board;
        let doubles_locked = state.rules.doubles_locked;
        let n_segments = board.n_segments;

        let mut terminals: Vec<u8> = Vec::new();
        let mut dead_tickets: Vec<u8> = Vec::new();
        let mut live_points = 0u32;
        for ticket in state.tickets_of(player) {
            if state.ticket_complete(player, ticket as usize) {
                continue;
            }
            let (a, b) = (
                board.ticket_a[ticket as usize],
                board.ticket_b[ticket as usize],
            );
            // A ticket whose endpoints are mutually unreachable is a sunk loss, not a plan
            // item. Leaving it in the terminal set would drag the Steiner cost to
            // UNREACHABLE and make every downstream comparison meaningless.
            let reach = remaining_costs_dense(
                board,
                &state.seg_owner[..n_segments],
                player as u8,
                a,
                doubles_locked,
            );
            if reach.dist[b as usize] >= UNREACHABLE {
                dead_tickets.push(ticket);
            } else {
                terminals.push(a);
                terminals.push(b);
                live_points += u32::from(board.ticket_points[ticket as usize]);
            }
        }
        terminals.sort_unstable();
        terminals.dedup();

        let (steiner_cost, exact) = steiner_cost_exact(
            board,
            &state.seg_owner[..n_segments],
            player as u8,
            &terminals,
            doubles_locked,
        );
        let segments = tree_segments(
            board,
            &state.seg_owner[..n_segments],
            player as u8,
            &terminals,
            doubles_locked,
        );

        let mut need = [0u16; MAX_CARD_TYPES];
        for &segment in &segments {
            let color = board.seg_color[segment as usize];
            let slot = if color == crate::board::GRAY {
                board.locomotive as usize
            } else {
                color as usize
            };
            need[slot] += u16::from(board.seg_len[segment as usize]);
        }

        Self {
            board_version: state.board_version,
            tickets: state.tickets[player],
            player: player as u8,
            terminals,
            steiner_cost,
            exact,
            detour: vec![u16::MAX; segments.len()],
            segments,
            dead_tickets,
            live_points,
            own_segments: own_segment_count(state, player as u8),
            need,
        }
    }

    /// Whether this plan is still the answer for `player` in `state`.
    ///
    /// The version check is the fast path; [`Plan::survives`] is the one that saves the
    /// work, turning an opponent claim that misses this plan into a cache refresh rather
    /// than a rebuild.
    pub fn is_fresh(&mut self, state: &State, player: usize) -> bool {
        if self.player != player as u8 || self.tickets != state.tickets[player] {
            return false;
        }
        if self.board_version == state.board_version {
            return true;
        }
        if !self.survives(state) {
            return false;
        }
        // The plan holds; the *detours* do not, because they measure routes around it and
        // the board just changed. Keeping them would price a contested segment at whatever
        // it cost before the route beside it was taken.
        self.board_version = state.board_version;
        self.detour.iter_mut().for_each(|d| *d = u16::MAX);
        true
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn contains(&self, segment: u16) -> bool {
        self.segments.binary_search(&segment).is_ok()
    }

    /// Extra train cars needed to get around `segment` if someone else takes it.
    ///
    /// The honest "how contested is it *for me*" number: a plan segment with a parallel
    /// route costs almost nothing to lose, one on a unique crossing costs the whole detour,
    /// and one whose loss disconnects its endpoints entirely reports [`UNREACHABLE`] --
    /// which is exactly the ticket-killing case. Zero for a segment not on the plan.
    ///
    /// **Priced by re-routing its own endpoints, not by re-solving the plan.** Deleting the
    /// segment and re-running Dreyfus-Wagner is the exact answer and costs a full Steiner
    /// solve per candidate; a single Dijkstra gives the same ranking, because a plan segment
    /// is on a cheapest path between two points of the tree by construction, so the detour
    /// around it *is* the extra cost the tree pays. Measured on the USA board this took H3
    /// from 76 to 26 microseconds a decision with no measurable change in strength -- and
    /// H3's per-decision cost is the sim budget of every Phase 5 search.
    ///
    /// Memoized per plan and computed on demand: a decision prices only the segments it can
    /// currently afford, which is a handful.
    pub fn detour_cost(&mut self, state: &State, segment: u16) -> u16 {
        let Ok(index) = self.segments.binary_search(&segment) else {
            return 0;
        };
        if self.detour[index] != u16::MAX {
            return self.detour[index];
        }
        let board = state.board;
        let n_segments = board.n_segments;
        let mut owner: Vec<u8> = state.seg_owner[..n_segments].to_vec();
        // CLOSED rather than an opponent id: the sibling rules must not fire off a
        // hypothetical claim. What is being priced is "this track is gone", nothing else.
        owner[segment as usize] = CLOSED;
        let (a, b) = (board.seg_a[segment as usize], board.seg_b[segment as usize]);
        let reach =
            remaining_costs_dense(board, &owner, self.player, a, state.rules.doubles_locked);
        let around = reach.dist[b as usize];
        let cost = around.saturating_sub(u16::from(board.seg_len[segment as usize]));
        self.detour[index] = cost;
        cost
    }

    /// Whether this plan survives a board change unchanged.
    ///
    /// Claims only ever *remove* options, never create cheaper ones. So if none of this
    /// plan's segments has been taken and this seat has claimed nothing itself, the plan is
    /// still available at the same price -- and since no cheaper tree can have appeared, a
    /// plan that was optimal still is. That turns most opponent claims into a version bump
    /// instead of a rebuild.
    ///
    /// The `own_segments` half is load-bearing: my *own* claim off the plan makes that
    /// track free to me and can genuinely lower my Steiner cost, so it forces a rebuild
    /// even though every plan segment survived.
    fn survives(&self, state: &State) -> bool {
        let n_segments = state.board.n_segments;
        if own_segment_count(state, self.player) != self.own_segments {
            return false;
        }
        self.segments
            .iter()
            .all(|&s| state.seg_owner[s as usize] == FREE)
            && n_segments == state.board.n_segments
    }
}

fn own_segment_count(state: &State, player: u8) -> usize {
    state.seg_owner[..state.board.n_segments]
        .iter()
        .filter(|&&o| o == player)
        .count()
}

/// A dense Dijkstra with predecessors. `MAX_CITIES` is 36, so the linear scan for the next
/// vertex beats a binary heap and, more usefully here, needs no allocation.
pub struct Reach {
    pub dist: [u16; MAX_CITIES],
    prev_city: [u8; MAX_CITIES],
    prev_seg: [u16; MAX_CITIES],
}

/// Cars `player` still needs to reach each city from `source`. Owned track is free, free
/// track costs its length, everyone else's is impassable.
pub fn remaining_costs_dense(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    source: u8,
    doubles_locked: bool,
) -> Reach {
    multi_source_costs(board, seg_owner, player, &[source], doubles_locked)
}

fn multi_source_costs(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    sources: &[u8],
    doubles_locked: bool,
) -> Reach {
    let n = board.n_cities;
    let mut reach = Reach {
        dist: [UNREACHABLE; MAX_CITIES],
        prev_city: [NO_PRED; MAX_CITIES],
        prev_seg: [NO_SIBLING; MAX_CITIES],
    };
    for &s in sources {
        reach.dist[s as usize] = 0;
    }
    let mut done = [false; MAX_CITIES];
    for _ in 0..n {
        let mut u = usize::MAX;
        let mut best = UNREACHABLE;
        for (v, (&settled, &d)) in done[..n].iter().zip(&reach.dist[..n]).enumerate() {
            if !settled && d < best {
                u = v;
                best = d;
            }
        }
        if u == usize::MAX {
            break;
        }
        done[u] = true;
        for &(nb, segment) in &board.adjacency[u] {
            let w = edge_cost(board, seg_owner, player, segment as usize, doubles_locked);
            if w >= UNREACHABLE {
                continue;
            }
            let nd = best + w;
            if nd < reach.dist[nb as usize] {
                reach.dist[nb as usize] = nd;
                reach.prev_city[nb as usize] = u as u8;
                reach.prev_seg[nb as usize] = segment;
            }
        }
    }
    reach
}

/// Unclaimed segments on a cheap structure connecting every terminal.
///
/// The shortest-path heuristic for Steiner trees: grow from one terminal, attach whichever
/// unconnected terminal is cheapest to reach, and price the newly added segments at zero so
/// later attachments share the trunk. That sharing is what stops two western tickets from
/// planning two separate transcontinentals.
///
/// The result is an edge set, not a cost -- [`crate::graph::steiner_cost_exact`] owns the
/// cost, and it is exact where this is a 2-approximation. Using the exact DP's own edge set
/// would need back-pointers threaded through its subset table, which is a great deal of
/// machinery for a set that is only ever used to *rank* candidate claims.
fn tree_segments(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    terminals: &[u8],
    doubles_locked: bool,
) -> Vec<u16> {
    if terminals.len() < 2 {
        return Vec::new();
    }
    let mut owner: Vec<u8> = seg_owner.to_vec();
    let mut tree: Vec<u8> = vec![terminals[0]];
    let mut pending: Vec<u8> = terminals[1..].to_vec();
    let mut chosen: Vec<u16> = Vec::new();

    while !pending.is_empty() {
        let reach = multi_source_costs(board, &owner, player, &tree, doubles_locked);
        // Nearest first, ties by city index so the plan is a function of the position and
        // not of iteration order -- the agents are deterministic and must stay that way.
        let Some((at, &target)) = pending
            .iter()
            .enumerate()
            .filter(|&(_, &t)| reach.dist[t as usize] < UNREACHABLE)
            .min_by_key(|&(_, &t)| (reach.dist[t as usize], t))
        else {
            break; // every remaining terminal is unreachable; the dead-ticket case
        };
        pending.remove(at);

        let mut city = target;
        while reach.dist[city as usize] > 0 {
            let segment = reach.prev_seg[city as usize];
            if owner[segment as usize] == FREE {
                chosen.push(segment);
                // Free for the rest of the build, which is what makes the trunk shared.
                owner[segment as usize] = player;
            }
            city = reach.prev_city[city as usize];
        }
        tree.push(target);
    }

    chosen.sort_unstable();
    chosen.dedup();
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::state::Game;

    fn game(map: &str, n: usize) -> Game {
        Game::new(RuleConfig::new(map, n).unwrap()).unwrap()
    }

    #[test]
    fn the_plan_connects_every_live_terminal() {
        for map in ["usa", "mini"] {
            let g = game(map, 2);
            for seed in 0..12u64 {
                let mut state = g.new_initial_state(seed);
                let plan = Plan::build(&mut state, 0);
                if plan.terminals.len() < 2 {
                    continue;
                }
                // Claiming exactly the plan must connect the terminals: pricing the plan's
                // segments as mine and re-solving has to leave nothing to pay for.
                let mut owner = state.seg_owner[..g.board.n_segments].to_vec();
                for &s in &plan.segments {
                    owner[s as usize] = 0;
                }
                let (rest, _) = steiner_cost_exact(
                    g.board,
                    &owner,
                    0,
                    &plan.terminals,
                    state.rules.doubles_locked,
                );
                assert_eq!(rest, 0, "{map} seed {seed}: the plan does not connect");
            }
        }
    }

    #[test]
    fn the_plan_is_within_the_steiner_bound_of_optimal() {
        // The edge set is a 2-approximation and the cost is exact, so the plan's own train
        // count must sit between them. Below the exact cost would mean the exact solver is
        // wrong; above twice it would mean the construction is not the heuristic it claims.
        for map in ["usa", "mini"] {
            let g = game(map, 2);
            for seed in 0..20u64 {
                let mut state = g.new_initial_state(seed);
                let plan = Plan::build(&mut state, 0);
                if plan.segments.is_empty() || !plan.exact {
                    continue;
                }
                let cars: u32 = plan
                    .segments
                    .iter()
                    .map(|&s| u32::from(g.board.seg_len[s as usize]))
                    .sum();
                assert!(
                    cars >= plan.steiner_cost,
                    "{map} seed {seed}: plan {cars} < exact {}",
                    plan.steiner_cost
                );
                assert!(
                    cars <= 2 * plan.steiner_cost,
                    "{map} seed {seed}: plan {cars} > 2x exact {}",
                    plan.steiner_cost
                );
            }
        }
    }

    #[test]
    fn a_draw_leaves_the_plan_fresh_but_taking_one_of_its_segments_does_not() {
        let g = game("usa", 2);
        let mut state = g.new_initial_state(4);
        while state.phase != crate::config::PHASE_MAIN {
            state.step(state.legal_actions()[0]).unwrap();
        }
        let mut plan = Plan::build(&mut state, 0);
        assert!(plan.is_fresh(&state, 0));
        assert!(!plan.segments.is_empty());

        // A card draw changes what I can afford, not what I need.
        state
            .step(state.space.draw(crate::actions::BLIND_SLOT))
            .unwrap();
        assert!(
            plan.is_fresh(&state, 0),
            "a draw invalidated the plan; the cache would rebuild every turn"
        );

        // An opponent claim that misses the plan must leave it fresh -- that is the whole
        // point of `survives`, since claims only ever remove options.
        let elsewhere = (0..g.board.n_segments as u16)
            .find(|s| !plan.contains(*s) && g.board.sibling[*s as usize] == NO_SIBLING)
            .expect("some segment is off the plan");
        state.seg_owner[elsewhere as usize] = 1;
        state.board_version += 1;
        assert!(
            plan.is_fresh(&state, 0),
            "an opponent claim away from my plan forced a rebuild"
        );

        // Taking a segment *on* the plan must not.
        let mine = plan.segments[0];
        state.seg_owner[mine as usize] = 1;
        state.board_version += 1;
        assert!(
            !plan.is_fresh(&state, 0),
            "a claim on the plan left it fresh"
        );
    }

    #[test]
    fn my_own_claim_rebuilds_even_when_every_plan_segment_survives() {
        // The subtle half of `survives`. An opponent's claim only removes options, so a
        // surviving plan is still optimal. *My* claim adds free track, which can genuinely
        // make a cheaper plan available -- so the segment count is checked too. Without
        // that clause this is a stale plan that looks valid.
        let g = game("usa", 2);
        let mut state = g.new_initial_state(4);
        while state.phase != crate::config::PHASE_MAIN {
            state.step(state.legal_actions()[0]).unwrap();
        }
        let mut plan = Plan::build(&mut state, 0);
        let elsewhere = (0..g.board.n_segments as u16)
            .find(|s| !plan.contains(*s))
            .expect("some segment is off the plan");
        state.seg_owner[elsewhere as usize] = 0;
        state.board_version += 1;
        assert!(!plan.is_fresh(&state, 0));
    }

    #[test]
    fn a_detour_is_zero_off_plan_and_positive_on_a_unique_crossing() {
        let g = game("usa", 2);
        let mut state = g.new_initial_state(9);
        let mut plan = Plan::build(&mut state, 0);
        if plan.segments.is_empty() {
            return;
        }
        // Off-plan segments cost nothing to lose.
        let off = (0..g.board.n_segments as u16)
            .find(|s| !plan.contains(*s))
            .expect("some segment is off plan");
        assert_eq!(plan.detour_cost(&state, off), 0);

        // At least one plan segment must be worth something, or "contested" is meaningless
        // on this board: a plan every one of whose edges has a free parallel alternative
        // would make the whole ranking a no-op.
        let priced: Vec<u16> = plan
            .segments
            .clone()
            .iter()
            .map(|&s| plan.detour_cost(&state, s))
            .collect();
        assert!(
            priced.iter().any(|&d| d > 0),
            "no plan segment costs anything to lose"
        );
    }

    #[test]
    fn dead_tickets_are_split_out_rather_than_poisoning_the_cost() {
        // Fence off a city entirely and give seat 0 a ticket into it. The ticket has to
        // land in `dead_tickets`; leaving it among the terminals would drag steiner_cost to
        // UNREACHABLE and make every candidate claim compare equal.
        let g = game("usa", 2);
        let mut state = g.new_initial_state(1);
        let board = g.board;
        let city = 1u8; // Boston
        for s in 0..board.n_segments {
            if board.seg_a[s] == city || board.seg_b[s] == city {
                state.seg_owner[s] = 1;
            }
        }
        let ticket = (0..board.n_tickets)
            .find(|&t| board.ticket_a[t] == city || board.ticket_b[t] == city)
            .expect("a ticket touching Boston");
        state.tickets[0] = 1 << ticket;
        let plan = Plan::build(&mut state, 0);
        assert_eq!(plan.dead_tickets, vec![ticket as u8]);
        assert!(plan.steiner_cost < u32::from(UNREACHABLE));
    }
}
