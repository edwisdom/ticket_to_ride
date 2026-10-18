//! H0-H4. One nested ladder, not five agents.
//!
//! PLAN.md §7 defines the tiers as strict extensions -- H3 is H2 plus four things, and H4
//! is H3 plus a plan that keeps what it needs and opponent modelling -- so they are written
//! that way: one scoring function whose terms switch on [`Tier`]. Five parallel implementations would let the tiers drift apart
//! in ways that look like strength differences, which is precisely what the ladder is
//! supposed to measure. H0 and H1 are the exceptions and are deliberately procedural: H0 is
//! the uniform floor, and H1 is a shallow rule cascade that exists to be beaten.
//!
//! **Everything is scored in one currency: points.** Route points are points; a train car
//! of progress toward a ticket is worth the ticket points it unlocks, discounted by how far
//! away completion still is; a card is worth the car it will buy. Mixing currencies -- say,
//! ranking claims in points and draws in "turns saved" -- is how a heuristic ends up with
//! an arbitrary conversion constant that nobody can tune, so there is exactly one.
//!
//! The one structural subtlety: **every train car costs
//! [`HeuristicParams::min_points_per_train`] to spend**, whether or not the claim is on the
//! plan. Trains are the game's real budget -- 45 of them, and the game ends when someone
//! runs out -- so a claim that pays less per car than that is worse than not claiming. That
//! single term is what makes the agents prefer dense routes, decline 1-car spurs, and stop
//! filling the board late; without it each tier needs its own endgame special case.
//!
//! **Every decision is explainable, by the same code that makes it.** `act` and `explain`
//! both run one `decide` -- `act` with a sink that records nothing, `explain` with
//! one that records each candidate's total, terms and inputs ([`crate::heuristic::trace`]).
//! The post-game review reads the second; it is never a re-derivation of the first.

use crate::actions::BLIND_SLOT;
use crate::board::{Board, GRAY, MAX_CARD_TYPES, NO_SIBLING};
use crate::config::{
    EMPTY_SLOT, FREE, NOT_TRIGGERED, PHASE_DRAW_SECOND, PHASE_INITIAL_TICKETS, PHASE_MAIN,
    PHASE_TICKET_KEEP,
};
use crate::graph::steiner_cost_exact;
use crate::heuristic::params::HeuristicParams;
use crate::heuristic::plan::Plan;
use crate::heuristic::trace::{Explanation, Recorder, Silent, Sink};
use crate::rng::{Part, Pcg32, stream};
use crate::state::State;

/// The agent ladder from PLAN.md §7. Ordered, and compared as such.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Tier {
    /// Uniform over legal actions. The floor every other agent must clear.
    H0 = 0,
    /// Greedy points: best affordable route, else the face-up card most represented in hand.
    H1 = 1,
    /// Ticket-path: Steiner plan over held tickets, most-contested segment first.
    H2 = 2,
    /// H2 + draw values, a card-aware plan, endgame and last-turn play, longest-path
    /// tiebreaks, ticket-keep EV, convex path cost, gray cannibalization. **The Elo anchor.**
    H3 = 3,
    /// H3 + a plan that keeps what it needs -- off-plan claims pay for the trains and cards
    /// they take from it -- and opponent modelling: threatened edges, threat pricing, double
    /// grabs, hoarding.
    H4 = 4,
}

impl Tier {
    pub fn name(self) -> &'static str {
        ["h0", "h1", "h2", "h3", "h4"][self as usize]
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "h0" | "random" => Some(Self::H0),
            "h1" => Some(Self::H1),
            "h2" => Some(Self::H2),
            "h3" => Some(Self::H3),
            "h4" => Some(Self::H4),
            _ => None,
        }
    }

    pub const ALL: [Tier; 5] = [Tier::H0, Tier::H1, Tier::H2, Tier::H3, Tier::H4];
}

// H1's rules, highest first. Within a rule an action's value ranks it, and every value
// stays below `H1_RULE_SCALE`, so a rule always decides before a value does.
const H1_CLAIM: f64 = 6.0;
const H1_TICKETS: f64 = 5.0;
const H1_FACEUP_LOCOMOTIVE: f64 = 4.0;
const H1_FACEUP_HELD: f64 = 3.0;
const H1_BLIND: f64 = 2.0;
const H1_FACEUP_UNHELD: f64 = 1.0;
const H1_UNWANTED: f64 = 0.0;
/// Above any route's points, any hand's count of one colour, and any offer's total cars.
const H1_RULE_SCALE: f64 = 100.0;

/// A scripted agent. Holds the plan cache and scratch buffers, so a decision allocates
/// nothing once it is warm.
pub struct Heuristic {
    pub tier: Tier,
    pub params: HeuristicParams,
    seed: u64,
    rng: Pcg32,
    plan: Plan,
    plan_ready: bool,
    /// Per-player track length per union-find root, cached on `board_version`. Used for the
    /// longest-chain tiebreak and for H4's threatened-edge primitive.
    nets: Vec<Vec<u16>>,
    /// Per-player count of free segments that would merge two of their components. §1's
    /// crowding rule needs it, and it moves only when someone claims.
    threats: Vec<u32>,
    nets_version: u32,
    nets_ready: bool,
    legal: Vec<u16>,
}

impl Clone for Heuristic {
    /// Clones the configuration, not the caches. A cloned agent is a fresh agent at the
    /// same settings; carrying a stale plan across into a different game would be a bug
    /// that only shows up as slightly wrong play.
    fn clone(&self) -> Self {
        Self::new(self.tier, self.params, self.seed)
    }
}

impl Heuristic {
    pub fn new(tier: Tier, params: HeuristicParams, seed: u64) -> Self {
        Self {
            tier,
            params,
            seed,
            rng: stream(seed, &[Part::Str("agent"), Part::Int(0), Part::Int(0)]),
            plan: Plan::default(),
            plan_ready: false,
            nets: Vec::new(),
            threats: Vec::new(),
            nets_version: 0,
            nets_ready: false,
            legal: Vec::new(),
        }
    }

    /// Re-seat and re-seed. The arena calls this once per game so a seat swap never carries
    /// an agent's stream -- or its plan cache -- across games.
    ///
    /// The stream name matches `ticket_to_ride.agents.base.Agent`: `("agent", seat,
    /// game_id)`, disjoint from the environment's `("env", ...)`. If the two shared a
    /// stream, an agent that sampled one extra action would shift every subsequent card
    /// draw and paired evaluation would silently stop being paired.
    pub fn begin_game(&mut self, seat: usize, game_id: u64) {
        self.rng = stream(
            self.seed,
            &[
                Part::Str("agent"),
                Part::Int(seat as u64),
                Part::Int(game_id),
            ],
        );
        self.plan_ready = false;
        self.nets_ready = false;
    }

    /// Pick an action for `state.current_player()`. Always legal.
    ///
    /// Takes `&mut State` for the same reason the observation encoder does: ticket
    /// completion is a union-find query and the forest uses path halving. The DSU is a
    /// cache excluded from every hash, so the mutation is invisible to anything that
    /// compares states.
    pub fn act(&mut self, state: &mut State) -> u16 {
        self.decide(state, &mut Silent)
    }

    /// [`Heuristic::act`], with its reasoning: every candidate the decision compared, its
    /// total, and the terms and inputs that made it (PLAN.md §14.2).
    ///
    /// **Not a second scorer.** This is `act` with a [`Recorder`] for its sink, so it returns
    /// the action `act` would have and leaves the agent as `act` would -- the same plan
    /// cache, H0's stream advanced by the same draw. Call it *instead of* `act` when the
    /// reasoning is wanted, never as well: for H0 that would draw twice.
    pub fn explain(&mut self, state: &mut State) -> Explanation {
        let seat = state.current_player();
        let mut recorder = Recorder::default();
        let chosen = self.decide(state, &mut recorder);
        recorder.finish(self.tier, seat, state.phase, chosen)
    }

    /// The one decision procedure behind `act` and `explain`. `sink` only records.
    fn decide<S: Sink>(&mut self, state: &mut State, sink: &mut S) -> u16 {
        let seat = state.current_player() as usize;
        self.legal.clear();
        state.legal_into(&mut self.legal);
        self.legal.sort_unstable();
        debug_assert!(!self.legal.is_empty(), "asked to act in a dead position");

        if self.legal.len() == 1 {
            sink.candidate(self.legal[0]);
            sink.rule("forced");
            sink.total(0.0);
            return self.legal[0];
        }
        if self.tier == Tier::H0 {
            let i = self.rng.below(self.legal.len() as u32) as usize;
            if S::LISTENING {
                // Nothing is compared, so the record is the distribution drawn from.
                let p = 1.0 / self.legal.len() as f64;
                for &action in &self.legal {
                    sink.candidate(action);
                    sink.rule("uniform");
                    sink.total(p);
                }
            }
            return self.legal[i];
        }

        match state.phase {
            PHASE_INITIAL_TICKETS | PHASE_TICKET_KEEP => self.act_keep(state, seat, sink),
            PHASE_MAIN | PHASE_DRAW_SECOND => {
                if self.tier == Tier::H1 {
                    self.act_h1(state, seat, sink)
                } else {
                    self.act_scored(state, seat, sink)
                }
            }
            _ => self.legal[0],
        }
    }

    // ------------------------------------------------------------------
    // Caches
    // ------------------------------------------------------------------

    fn refresh(&mut self, state: &mut State, seat: usize) {
        // H2 plans by cars alone; pricing the cards is one of H3's additions.
        let card_weight = if self.tier >= Tier::H3 {
            self.params.plan_card_weight
        } else {
            0.0
        };
        if !self.plan_ready || !self.plan.is_fresh(state, seat) {
            self.plan = Plan::build(state, seat, card_weight);
            self.plan_ready = true;
        } else {
            self.plan.follow_cards(state);
        }
        if !self.nets_ready || self.nets_version != state.board_version {
            let n = state.n_players();
            let n_cities = state.board.n_cities;
            self.nets.resize(n, Vec::new());
            for (p, net) in self.nets.iter_mut().enumerate() {
                net.clear();
                net.resize(n_cities, 0);
                for s in 0..state.board.n_segments {
                    if state.seg_owner[s] == p as u8 {
                        let root = state.dsu_root(p, state.board.seg_a[s]);
                        net[root as usize] += u16::from(state.board.seg_len[s]);
                    }
                }
            }
            // Only H4 reads the crowding counts, and the scan is O(segments) per claim.
            self.threats.clear();
            self.threats.resize(n, 0);
            if self.tier >= Tier::H4 {
                let nets = std::mem::take(&mut self.nets);
                for p in 0..n {
                    self.threats[p] = Self::count_threats(state, &nets, p);
                }
                self.nets = nets;
            }
            self.nets_version = state.board_version;
            self.nets_ready = true;
        }
    }

    /// Track length of `player`'s network component containing `city`.
    fn component_cars(&self, state: &mut State, player: usize, city: u8) -> u16 {
        let root = state.dsu_root(player, city);
        self.nets[player][root as usize]
    }

    // ------------------------------------------------------------------
    // H1 -- a shallow rule cascade, outside the points currency on purpose
    // ------------------------------------------------------------------

    /// H1's cascade -- the route worth the most points, else tickets if it wants them, else
    /// the face-up colour it holds most of -- as one priority per legal action, taking the
    /// highest with ties to the lowest id. A priority rather than a chain of early returns,
    /// so `explain` reports exactly what was compared; the rules and their order are the
    /// cascade's.
    fn act_h1<S: Sink>(&mut self, state: &State, seat: usize, sink: &mut S) -> u16 {
        let wants_tickets = self.h1_wants_tickets(state, seat);
        let mut best = self.legal[0];
        let mut best_priority = f64::NEG_INFINITY;
        for i in 0..self.legal.len() {
            let action = self.legal[i];
            sink.candidate(action);
            let priority = self.h1_priority(state, seat, action, wants_tickets, sink);
            sink.total(priority);
            if priority > best_priority {
                best = action;
                best_priority = priority;
            }
        }
        best
    }

    /// Which of H1's rules ranks `action`, and where within it: `rule * H1_RULE_SCALE +
    /// value`.
    fn h1_priority<S: Sink>(
        &self,
        state: &State,
        seat: usize,
        action: u16,
        wants_tickets: bool,
        sink: &mut S,
    ) -> f64 {
        let (space, board) = (state.space, state.board);
        let (rule, name, value_name, value) = if action < space.claim_end {
            let points = board.route_points[board.seg_len[(action / space.k) as usize] as usize];
            (
                H1_CLAIM,
                "claim_most_points",
                "route_points",
                f64::from(points),
            )
        } else if action == space.draw_tickets {
            if wants_tickets {
                (H1_TICKETS, "draw_tickets", "", 0.0)
            } else {
                (H1_UNWANTED, "tickets_unwanted", "", 0.0)
            }
        } else if action < space.draw_tickets {
            let slot = action - space.draw_base;
            if slot == BLIND_SLOT {
                // Holding none of every face-up colour: a blind draw beats starting a new one.
                (H1_BLIND, "blind_draw", "", 0.0)
            } else {
                // A legal face-up draw names a non-empty slot.
                let card = state.faceup[slot as usize];
                if card == board.locomotive as i8 && self.params.h1_take_faceup_locomotive {
                    (H1_FACEUP_LOCOMOTIVE, "faceup_locomotive", "", 0.0)
                } else {
                    let held = state.hand[seat * board.n_card_types + card as usize];
                    if held > 0 {
                        (
                            H1_FACEUP_HELD,
                            "faceup_most_held",
                            "cards_held",
                            f64::from(held),
                        )
                    } else {
                        (H1_FACEUP_UNHELD, "faceup_unheld", "", 0.0)
                    }
                }
            }
        } else {
            // KEEP is not legal here, and PASS only alone, which returned before scoring.
            (H1_UNWANTED, "pass", "", 0.0)
        };
        debug_assert!(
            value < H1_RULE_SCALE,
            "{name}: value {value} reaches the next rule"
        );
        sink.rule(name);
        sink.term("priority", rule * H1_RULE_SCALE);
        if !value_name.is_empty() {
            sink.term(value_name, value);
        }
        rule * H1_RULE_SCALE + value
    }

    fn h1_wants_tickets(&self, state: &State, seat: usize) -> bool {
        let space = state.space;
        if !self.legal.contains(&space.draw_tickets) {
            return false;
        }
        if state.tickets[seat].count_ones() >= u32::from(self.params.h1_max_tickets) {
            return false;
        }
        let supply = f64::from(state.board.raw.trains_per_player);
        f64::from(state.trains[seat]) >= self.params.ticket_draw_train_fraction * supply
    }

    // ------------------------------------------------------------------
    // H2-H4 -- the scored ladder
    // ------------------------------------------------------------------

    fn act_scored<S: Sink>(&mut self, state: &mut State, seat: usize, sink: &mut S) -> u16 {
        self.refresh(state, seat);
        let ctx = Ctx::build(state, seat, &self.params, &self.plan, self.tier);
        if S::LISTENING {
            ctx.report(&self.plan, state, sink);
        }
        let space = state.space;

        let mut best = self.legal[0];
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..self.legal.len() {
            let action = self.legal[i];
            sink.candidate(action);
            let score = if action < space.claim_end {
                self.score_claim(state, &ctx, action, sink)
            } else if action < space.draw_tickets {
                self.score_draw(state, &ctx, action, sink)
            } else if action == space.draw_tickets {
                self.score_draw_tickets(state, &ctx, sink)
            } else {
                // PASS, which is only legal when nothing else is -- and that case returned
                // early. Scored below everything so it can never win a real comparison.
                f64::NEG_INFINITY
            };
            sink.total(score);
            // Strictly greater: ties go to the lowest action id, which is what keeps every
            // heuristic a deterministic function of the position.
            if score > best_score {
                best = action;
                best_score = score;
            }
        }
        best
    }

    fn score_claim<S: Sink>(
        &mut self,
        state: &mut State,
        ctx: &Ctx,
        action: u16,
        sink: &mut S,
    ) -> f64 {
        let board = state.board;
        let p = &self.params;
        let (segment, pay) = state.space.decode_claim(action);
        let seg = segment as usize;
        let length = board.seg_len[seg];
        let cars = f64::from(length);

        // Every car spent has an opportunity cost. This is the term that makes the ladder
        // prefer dense routes and decline spurs, and it removes the need for a separate
        // endgame "maximize points per train" branch.
        let points = f64::from(board.route_points[length as usize]);
        if ctx.final_turn {
            return self.score_final_claim(state, ctx, segment, points, sink);
        }
        let train_cost = p.min_points_per_train * cars;
        let mut score = points - train_cost;
        sink.term("route_points", points);
        sink.term("train_cost", -train_cost);
        sink.fact("cars", cars);

        let on_plan = self.plan.contains(segment);
        sink.fact("on_plan", f64::from(u8::from(on_plan)));
        if on_plan {
            let plan_value = ctx.plan_car_value * cars;
            score += plan_value;
            sink.term("plan_value", plan_value);
            let detour = self.plan.detour_cost(state, segment);
            let contest = p.contest_weight * f64::from(detour);
            score += contest;
            sink.term("contest", contest);
            sink.fact("detour", f64::from(detour));
            // 2-3P: claiming either track of a double closes the sibling to everyone, so a
            // plan segment that is half of a live double is worth taking early.
            let sibling = board.sibling[seg];
            if state.rules.doubles_locked
                && sibling != NO_SIBLING
                && state.seg_owner[sibling as usize] == FREE
            {
                let urgency = if self.tier >= Tier::H4 {
                    p.double_grab_urgency
                } else {
                    1.0
                };
                let lockout = p.double_lockout_weight * urgency;
                score += lockout;
                sink.term("double_lockout", lockout);
            }
        }

        if self.tier >= Tier::H3 {
            score += self.h3_claim_terms(state, ctx, segment, pay, length, on_plan, sink);
        }
        if self.tier >= Tier::H4 {
            score += self.h4_claim_terms(state, ctx, segment, length, on_plan, sink);
            if !on_plan && !self.plan.is_empty() {
                score += self.off_plan_terms(state, ctx, seg, length, pay, sink);
            }
        }
        score
    }

    /// What an off-plan claim takes from the plan: trains the plan would have needed, and
    /// cards it would have spent.
    ///
    /// H3 charges every car the flat [`HeuristicParams::min_points_per_train`], which is
    /// right while the trains outnumber the plan and badly wrong once they do not. In H3
    /// self-play a quarter of seats made an off-plan claim that left fewer trains than
    /// their plan still needed; they won 30% of games against 57% for the rest, and 99% of
    /// the losers among them missed a ticket. One game: two tickets worth 39 points sat 11
    /// cars from done with 20 trains left, and H3 spent 17 of them on routes its tickets
    /// did not use, because each beat a blind draw on route points alone.
    ///
    /// So the trains are priced by the completion model the ticket decisions already use
    /// -- the settlement they cost the plan -- which is nearly nothing with trains to spare
    /// and dominant when they are short. The cards are priced as cards the plan must now
    /// draw again: another game spent five blacks on an off-plan route one black short of
    /// the 6-black route that would have finished 37 points of tickets. Gray
    /// cannibalization covered only gray routes.
    fn off_plan_terms<S: Sink>(
        &self,
        state: &State,
        ctx: &Ctx,
        seg: usize,
        length: u8,
        pay: u16,
        sink: &mut S,
    ) -> f64 {
        let p = &self.params;
        let mut extra = 0.0;
        if p.plan_budget_weight > 0.0 {
            let cost = f64::from(self.plan.steiner_cost);
            let trains = f64::from(state.trains[ctx.seat]);
            let before = p.completion_odds(cost, trains);
            let after = p.completion_odds(cost, trains - f64::from(length));
            let budget =
                p.plan_budget_weight * 2.0 * f64::from(self.plan.live_points) * (after - before);
            extra += budget;
            sink.term("train_budget", budget);
            sink.fact("completion_odds_after", after);
        }
        let payment = if p.plan_card_spend_weight > 0.0 {
            state.payment_for(seg, length, pay as u8).ok()
        } else {
            None
        };
        if let Some((colored, wilds)) = payment {
            let board = state.board;
            let loco = board.locomotive as usize;
            let mut hand = [0u8; MAX_CARD_TYPES];
            hand[..board.n_card_types].copy_from_slice(state.hand_of(ctx.seat));
            let before = plan_shortfall(&hand, &self.plan.need, board.n_colors, loco);
            hand[pay as usize] -= colored;
            hand[loco] -= wilds;
            let after = plan_shortfall(&hand, &self.plan.need, board.n_colors, loco);
            let lost = after.saturating_sub(before);
            let spent = p.plan_card_spend_weight * ctx.plan_car_value * f64::from(lost);
            extra -= spent;
            sink.term("plan_cards_spent", -spent);
            sink.fact("plan_cards_lost", f64::from(lost));
        }
        extra
    }

    /// A claim on this seat's last turn: its route points, plus every held ticket it
    /// completes, which swings from minus its value to plus. Nothing else -- no card, train
    /// or plan outlives the turn -- except the longest-route tiebreak, which is settled
    /// after it.
    fn score_final_claim<S: Sink>(
        &mut self,
        state: &mut State,
        ctx: &Ctx,
        segment: u16,
        points: f64,
        sink: &mut S,
    ) -> f64 {
        let board = state.board;
        let seg = segment as usize;
        let (a, b) = (board.seg_a[seg], board.seg_b[seg]);
        let (root_a, root_b) = (state.dsu_root(ctx.seat, a), state.dsu_root(ctx.seat, b));
        let mut finished = 0.0;
        for ticket in state.tickets_of(ctx.seat) {
            let t = ticket as usize;
            let (x, y) = (
                state.dsu_root(ctx.seat, board.ticket_a[t]),
                state.dsu_root(ctx.seat, board.ticket_b[t]),
            );
            // Open now, and this claim joins exactly the two pieces it spans.
            if x != y && ((x, y) == (root_a, root_b) || (x, y) == (root_b, root_a)) {
                finished += 2.0 * f64::from(board.ticket_points[t]);
            }
        }
        sink.term("route_points", points);
        sink.term("tickets_finished", finished);
        sink.fact("cars", f64::from(board.seg_len[seg]));
        let mut score = points + finished;
        let touching =
            self.component_cars(state, ctx.seat, a) + self.component_cars(state, ctx.seat, b);
        if touching > 0 {
            let chain = self.params.longest_chain_weight * f64::from(board.seg_len[seg]);
            score += chain;
            sink.term("longest_chain", chain);
        }
        score
    }

    /// H3's four claim-side additions: longest-chain tiebreak, gray cannibalization,
    /// locomotive thrift, and the endgame re-weighting of plan credit.
    #[allow(clippy::too_many_arguments)]
    fn h3_claim_terms<S: Sink>(
        &mut self,
        state: &mut State,
        ctx: &Ctx,
        segment: u16,
        pay: u16,
        length: u8,
        on_plan: bool,
        sink: &mut S,
    ) -> f64 {
        let board = state.board;
        let p = &self.params;
        let seg = segment as usize;
        let mut extra = 0.0;

        // The longest-path bonus is 10 points and often decisive, but it is a tiebreak, not
        // a plan: weight it by the cars this claim adds to a component that already exists.
        let (a, b) = (board.seg_a[seg], board.seg_b[seg]);
        let touching =
            self.component_cars(state, ctx.seat, a) + self.component_cars(state, ctx.seat, b);
        if touching > 0 {
            let chain = p.longest_chain_weight * f64::from(length);
            extra += chain;
            sink.term("longest_chain", chain);
        }

        // What this payment actually costs in flexibility.
        if let Ok((colored, wilds)) = state.payment_for(seg, length, pay as u8) {
            // Gray cannibalization: a gray route takes any colour, so paying it with one a
            // *coloured* plan segment still needs spends flexibility only that route can
            // use. Paying gray with the colour you are longest in is free; paying it with
            // your last four blues when the plan needs a 4-blue route is not.
            if board.seg_color[seg] == GRAY && (pay as u8) != board.locomotive {
                let wasted = u16::from(colored).min(ctx.deficit[pay as usize]);
                let cannibalized =
                    p.gray_cannibalization_penalty * ctx.plan_car_value * f64::from(wasted);
                extra -= cannibalized;
                sink.term("gray_cannibalization", -cannibalized);
                sink.fact("needed_cards_spent", f64::from(wasted));
            }
            // A locomotive substitutes for any colour, so spending one where a coloured
            // card would do throws away the difference.
            let thrift =
                (p.locomotive_value - 1.0).max(0.0) * ctx.plan_car_value * f64::from(wilds);
            extra -= thrift;
            sink.term("locomotive_thrift", -thrift);
            sink.fact("locomotives", f64::from(wilds));
        }

        // Endgame. Under `endgame_train_fraction` of the supply the binding question is
        // whether the plan can still be finished at all: cars sunk into a plan that cannot
        // complete score nothing at the end, while a completable one is worth finishing at
        // almost any price.
        if ctx.endgame && on_plan {
            let base = ctx.plan_car_value * f64::from(length);
            let urgency = if ctx.plan_completable {
                base * (p.endgame_ticket_urgency - 1.0)
            } else {
                -base
            };
            extra += urgency;
            sink.term("endgame", urgency);
        }
        extra
    }

    /// H4's claim-side additions: the threatened-edge primitive, and hoard-and-strike.
    fn h4_claim_terms<S: Sink>(
        &mut self,
        state: &mut State,
        ctx: &Ctx,
        segment: u16,
        length: u8,
        on_plan: bool,
        sink: &mut S,
    ) -> f64 {
        let p = self.params;
        // **Block with track you wanted anyway.** Denial priced on its own merits made H4
        // claim routes that served nobody but the denial: it took the longest-path bonus in
        // 50% of games against H3's 74% -- ten points, often decisive -- and scored 84 to
        // H3's 94 while losing the match. A blocking claim still costs trains, and trains
        // are the plan's budget. So the threat term is a *tiebreak among claims already
        // worth making*, not a reason to make one.
        let useful = on_plan
            || self.component_cars(state, ctx.seat, state.board.seg_a[segment as usize]) > 0
            || self.component_cars(state, ctx.seat, state.board.seg_b[segment as usize]) > 0;
        let threat = if useful {
            p.threat_weight * self.threat_value(state, ctx, segment, length)
        } else {
            0.0
        };
        sink.term("threat", threat);
        let mut extra = threat;

        // Hoard and strike, applied only where waiting is provably free: a plan segment
        // whose detour cost is zero has a parallel alternative, so losing it costs nothing
        // and the turn is better spent on cards for a segment that *is* contested. Gating
        // on a zero detour is what keeps this from deadlocking -- an agent that defers
        // claims it actually needs never claims anything.
        if on_plan && self.plan.detour_cost(state, segment) == 0 {
            let pending = self.max_unaffordable_detour(state, ctx);
            if f64::from(pending) >= p.hoard_safety_margin {
                // Divided by the rival count for the same reason the threat is: "waiting
                // costs nothing" is a two-player premise. A detour of zero means *a*
                // parallel route exists, and the more rivals there are the likelier one of
                // them takes it while you wait. Measured, holding everything else fixed:
                // hoarding alone cost -12.3 Elo at USA 5P and was neutral at 4P and 2P.
                let rivals = f64::from(state.rules.n_players.saturating_sub(1).max(1));
                let hoard = p.contest_weight * f64::from(pending) / rivals;
                extra -= hoard;
                sink.term("hoard", -hoard);
                sink.fact("pending_detour", f64::from(pending));
            }
        }
        extra
    }

    /// The threatened-edge primitive (PLAN.md §1), with §1's four pricing rules.
    ///
    /// The primitive: an unclaimed edge that would merge two currently-disconnected
    /// components of an opponent's network. Much sharper than "a route they might want" --
    /// it needs no model of their tickets, only their track, which is public.
    ///
    /// The pricing is the half that makes it usable, and leaving it out is measurable. All
    /// four rules are §1's:
    ///
    /// 1. **Doubles are not blockable in 4-5P** -- one action cannot close a pair, so the
    ///    sibling stays open and the "block" denies nothing. In 2-3P claiming either track
    ///    closes both, so there the threat is real.
    /// 2. **Crowding**: `(n_threats - 1) x k`. An opponent with six merging edges available
    ///    loses little when one is taken.
    /// 3. **Game progress**: blocking matters late. Early, the cars are worth more in your
    ///    own network -- which is exactly the mistake the ungated version made.
    /// 4. **Super-linear in the severance**, and on the **bottleneck** rather than the sum:
    ///    merging a 2-car stub into a 20-car trunk gains them two cars of trail, not
    ///    twenty-two. Summing them made denial worth ~12 points against a 15-point route
    ///    and H4 spent the whole game blocking -- it lost to H3 at 92.5%.
    fn threat_value(&self, state: &mut State, ctx: &Ctx, segment: u16, length: u8) -> f64 {
        let board = state.board;
        let p = &self.params;
        if !state.rules.doubles_locked && board.sibling[segment as usize] != NO_SIBLING {
            return 0.0; // rule 1
        }
        let (a, b) = (board.seg_a[segment as usize], board.seg_b[segment as usize]);
        let mut best = 0.0f64;
        for o in 0..state.n_players() {
            if o == ctx.seat {
                continue;
            }
            let (ra, rb) = (state.dsu_root(o, a), state.dsu_root(o, b));
            if ra == rb {
                continue; // already connected for them; taking it denies nothing
            }
            let bottleneck = self.nets[o][ra as usize].min(self.nets[o][rb as usize]);
            if bottleneck == 0 {
                continue; // one side is a bare city: there is no network to merge yet
            }
            if !ctx.opponent_could_pay(state, o, segment, length) {
                continue; // a threat they cannot execute is not a threat
            }
            let severed =
                (f64::from(bottleneck) + f64::from(length)).powf(p.threat_severance_exponent);
            let crowd = 1.0 + p.threat_crowding_penalty * f64::from(self.threats[o].max(1) - 1);
            best = best.max(severed / crowd);
        }
        // **Blocking has a free-rider problem, and it scales with the seat count.** The
        // cars I spend denying one opponent help every *other* opponent as much as they
        // help me, so the share of the denial I actually capture is about one over the
        // number of rivals. Measured before this term existed: H4 was -50 Elo against H3 at
        // USA 4P (95% CI [-58, -42]) while being +21 at TTR-mini 2P -- same code, same
        // constants, opposite signs. A blocking heuristic that ignores the seat count is
        // not mis-tuned; it is solving a two-player problem at every table.
        let rivals = f64::from(state.rules.n_players.saturating_sub(1).max(1));
        best * ctx.points_per_car * ctx.progress / rivals
    }

    /// How many free segments would merge two components of `player`'s network right now.
    ///
    /// Cached with the network table, because it is an O(segments) scan and the answer
    /// only moves when someone claims.
    fn count_threats(state: &mut State, nets: &[Vec<u16>], player: usize) -> u32 {
        let board = state.board;
        let mut n = 0;
        for s in 0..board.n_segments {
            if state.seg_owner[s] != FREE {
                continue;
            }
            if !state.rules.doubles_locked && board.sibling[s] != NO_SIBLING {
                continue;
            }
            let (ra, rb) = (
                state.dsu_root(player, board.seg_a[s]),
                state.dsu_root(player, board.seg_b[s]),
            );
            if ra != rb && nets[player][ra as usize] > 0 && nets[player][rb as usize] > 0 {
                n += 1;
            }
        }
        n
    }

    /// The largest detour among plan segments this seat cannot yet pay for.
    fn max_unaffordable_detour(&mut self, state: &mut State, ctx: &Ctx) -> u16 {
        let segments = self.plan.segments.clone();
        let mut worst = 0;
        for segment in segments {
            if ctx.affordable.contains(&segment) {
                continue;
            }
            worst = worst.max(self.plan.detour_cost(state, segment));
        }
        worst
    }

    // -- draws ---------------------------------------------------------

    fn score_draw<S: Sink>(&self, state: &State, ctx: &Ctx, action: u16, sink: &mut S) -> f64 {
        let board = state.board;
        if ctx.final_turn {
            sink.rule("final_turn");
            return 0.0;
        }
        let slot = action - state.space.draw_base;
        if slot == BLIND_SLOT {
            // Expected over what this seat cannot account for: the deck plus every
            // opponent's blind draws.
            let unseen = state.unseen_counts(ctx.seat);
            let total: i32 = unseen.iter().map(|&c| i32::from(c.max(0))).sum();
            sink.fact("unseen_cards", f64::from(total));
            if total == 0 {
                return 0.0;
            }
            let mut expected = 0.0;
            for (card, &count) in unseen.iter().enumerate() {
                if count > 0 {
                    expected += f64::from(count) / f64::from(total) * ctx.card_value(card as u8);
                }
            }
            sink.term("expected_card_value", expected);
            return expected;
        }
        let card = state.faceup[slot as usize];
        if card == EMPTY_SLOT {
            return f64::NEG_INFINITY;
        }
        let card = card as u8;
        let mut value = ctx.card_value(card);
        sink.term("card_value", value);
        if card != board.locomotive {
            sink.fact("plan_needs", f64::from(ctx.deficit[card as usize]));
        }
        // A known card of a colour the plan is short of beats the same card drawn blind:
        // taking it is certain, and certainty is what turns a plan into a claim.
        if self.tier >= Tier::H3 && card != board.locomotive && ctx.deficit[card as usize] > 0 {
            let bonus = self.params.faceup_needed_bonus * ctx.plan_car_value;
            value += bonus;
            sink.term("needed_bonus", bonus);
        }
        value
    }

    /// Whether to spend the turn on tickets, and what that is worth.
    ///
    /// One hard gate, then the shared completion model. The gate is the prior art's lesson
    /// stated as a fraction of the board's own supply rather than an absolute train count
    /// -- `trains > 15` carried onto a 10-train map is what invalidated its baselines. It
    /// says "do not take on new commitments in the back half of the game", and nothing
    /// else: whether a commitment is *affordable* is
    /// [`HeuristicParams::completion_odds`], the same function the keep filter uses,
    /// because keeping a ticket and drawing one are the same question asked twice.
    fn score_draw_tickets<S: Sink>(&self, state: &State, ctx: &Ctx, sink: &mut S) -> f64 {
        let p = &self.params;
        let supply = f64::from(state.board.raw.trains_per_player);
        let trains = f64::from(state.trains[ctx.seat]);
        if ctx.final_turn {
            sink.rule("final_turn");
            return f64::NEG_INFINITY;
        }
        if trains < p.ticket_draw_train_fraction * supply {
            sink.rule("ticket_gate");
            return f64::NEG_INFINITY;
        }
        let deal = f64::from(state.board.raw.draw_ticket_deal.max(1));
        let kept = f64::from(state.board.raw.draw_ticket_keep_min.max(1));
        // What the plan would cost after keeping the minimum from a typical offer.
        //
        // Cars, not points: a ticket's *cost* is the track it needs, and on the USA board
        // the rulebook's values run ~20% above the distance.
        //
        // And the **cheapest** of the offer, not an average one. You are dealt `deal` and
        // must keep only `keep_min`, so you take the one that best overlaps the plan you
        // already have. For a roughly even spread the expected minimum of n draws is
        // `2/(n+1)` of the mean -- derived from the board's own deal size rather than
        // fitted, so it moves with a map that deals a different number.
        let selectivity = 2.0 / (deal + 1.0);
        let after =
            f64::from(self.plan.steiner_cost as u16) + kept * ctx.mean_ticket_cars * selectivity;
        let odds = p.completion_odds(after, trains);
        let value = ctx.mean_remaining_ticket_points * (2.0 * odds - 1.0) * kept;
        sink.term("expected_settlement", value);
        sink.fact("plan_cost_after", after);
        sink.fact("completion_odds", odds);
        sink.fact("mean_ticket_points", ctx.mean_remaining_ticket_points);
        value
    }

    // ------------------------------------------------------------------
    // Keeping tickets
    // ------------------------------------------------------------------

    fn act_keep<S: Sink>(&mut self, state: &mut State, seat: usize, sink: &mut S) -> u16 {
        match self.tier {
            Tier::H1 => self.keep_h1(state, sink),
            Tier::H2 => self.keep_scored(state, seat, false, sink),
            _ => self.keep_scored(state, seat, true, sink),
        }
    }

    /// H1: keep the fewest tickets allowed, choosing the shortest routes -- fewest first,
    /// then fewest cars, then the lowest id, as one priority so `explain` shows the order.
    fn keep_h1<S: Sink>(&self, state: &State, sink: &mut S) -> u16 {
        let (board, space) = (state.board, state.space);
        let cost: Vec<u16> = state.offer[..state.offer_len as usize]
            .iter()
            .map(|&t| {
                board.dist[board.ticket_a[t as usize] as usize][board.ticket_b[t as usize] as usize]
            })
            .collect();
        let mut best = self.legal[0];
        let mut best_priority = f64::NEG_INFINITY;
        for &action in &self.legal {
            let mask = action - space.keep_base();
            let kept: Vec<usize> = (0..state.offer_len as usize)
                .filter(|i| mask >> i & 1 != 0)
                .collect();
            let cars = f64::from(kept.iter().map(|&i| cost[i]).sum::<u16>());
            debug_assert!(
                cars < H1_RULE_SCALE,
                "{cars} cars reach the next ticket count"
            );
            let fewest = -(kept.len() as f64) * H1_RULE_SCALE;
            let priority = fewest - cars;
            sink.candidate(action);
            sink.rule("keep_fewest_shortest");
            sink.term("tickets_kept", fewest);
            sink.term("route_cars", -cars);
            sink.total(priority);
            if priority > best_priority {
                best = action;
                best_priority = priority;
            }
        }
        best
    }

    /// H2 minimizes what the keep adds to the plan; H3+ price the keep as what it will
    /// actually settle for.
    ///
    /// **Marginal, not standalone**, in both cases. A ticket riding an already-planned trunk
    /// line is nearly free, and pricing it at its own shortest path would refuse exactly the
    /// tickets worth taking. That is why this re-solves the Steiner cost per candidate
    /// subset rather than summing per-ticket distances.
    ///
    /// **Deviation from PLAN.md §7, and the measurement behind it.** The plan specifies the
    /// H3 filter as `marginal_steiner_cost <= points`. That rule is blind to the train
    /// budget, and on TTR-mini -- where ticket points *are* shortest-path costs by
    /// construction, so marginal equals points for every ticket -- it accepts everything.
    /// Measured: 3 opening tickets costing 22 cars against a 20-train supply, in 59 of 60
    /// games, after which the agent correctly refused to draw another all game because the
    /// plan it had just accepted was already unaffordable. `draw_tickets` fired zero times
    /// across 30 mini games while the USA board drew freely, which is the prior art's
    /// symptom reached by the opposite route. Expected settlement against the train budget
    /// is the same idea with the missing constraint put back, and it costs one parameter
    /// fewer.
    fn keep_scored<S: Sink>(
        &mut self,
        state: &mut State,
        seat: usize,
        ev_filter: bool,
        sink: &mut S,
    ) -> u16 {
        self.refresh(state, seat);
        let board = state.board;
        let doubles_locked = state.rules.doubles_locked;
        let n_segments = board.n_segments;
        let base_terminals = self.plan.terminals.clone();
        let base_cost = self.plan.steiner_cost;
        let owner: Vec<u8> = state.seg_owner[..n_segments].to_vec();
        sink.context("plan_cost", f64::from(base_cost));
        sink.context("trains", f64::from(state.trains[seat]));

        let mut best = self.legal[0];
        let mut best_value = f64::NEG_INFINITY;
        for &action in &self.legal {
            sink.candidate(action);
            let mask = action - state.space.keep_base();
            let mut terminals = base_terminals.clone();
            let mut points = 0.0;
            let mut dead_penalty = 0.0;
            for i in 0..state.offer_len as usize {
                if mask >> i & 1 == 0 {
                    continue;
                }
                let ticket = state.offer[i] as usize;
                let (a, b) = (board.ticket_a[ticket], board.ticket_b[ticket]);
                let value = f64::from(board.ticket_points[ticket]);
                if board.dist[a as usize][b as usize] >= crate::board::UNREACHABLE {
                    dead_penalty += self.params.ticket_keep_dead_penalty * value;
                    continue;
                }
                terminals.push(a);
                terminals.push(b);
                points += value;
            }
            terminals.sort_unstable();
            terminals.dedup();
            let (cost, _) =
                steiner_cost_exact(board, &owner, seat as u8, &terminals, doubles_locked);
            let marginal = f64::from(cost.saturating_sub(base_cost) as u16);

            sink.fact("ticket_points", points);
            sink.fact("plan_cost_after", f64::from(cost as u16));
            sink.fact("added_cars", marginal);
            let value = if ev_filter {
                // What this ticket set settles for: `+points` each if completed, `-points`
                // each if not, weighted by whether the trains left can absorb the whole
                // resulting plan -- `cost`, not `marginal`, because the budget is shared
                // with everything already committed.
                let trains = f64::from(state.trains[seat]);
                let odds = self.params.completion_odds(f64::from(cost as u16), trains);
                let settlement = points * (2.0 * odds - 1.0);
                sink.term("expected_settlement", settlement);
                sink.fact("completion_odds", odds);
                settlement - dead_penalty
            } else {
                // H2: take on as little new track as possible, fewest tickets first.
                let count = f64::from(mask.count_ones());
                sink.term("added_track", -marginal);
                sink.term("tickets_kept", -count);
                -marginal - count - dead_penalty
            };
            sink.term("dead_tickets", -dead_penalty);
            sink.total(value);
            if value > best_value {
                best = action;
                best_value = value;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// Per-decision derived quantities
// ---------------------------------------------------------------------------

/// Everything the scoring terms share, computed once per decision.
struct Ctx {
    seat: usize,
    /// Cards of each colour the plan still needs beyond what this seat holds. Index
    /// `locomotive` is gray demand, which any colour can satisfy.
    deficit: [u16; MAX_CARD_TYPES],
    /// True when even the seat's surplus cannot cover the plan's gray demand, in which case
    /// every colour is worth its full plan value.
    gray_unmet: bool,
    locomotive: u8,
    /// Points a single train car of plan progress is worth.
    plan_car_value: f64,
    /// The board's own mean points per train car -- the exchange rate between cars and
    /// points, used to price threats without inventing a constant.
    points_per_car: f64,
    mean_remaining_ticket_points: f64,
    /// Mean shortest-path cost, in train cars, of the tickets still in the deck. Distinct
    /// from the points: on the USA board the rulebook's values run about 20% above the
    /// distance, and mixing the two silently mis-prices the ticket-draw decision.
    mean_ticket_cars: f64,
    /// How far through the game we are, from trains spent by the fastest seat. §1's threat
    /// pricing scales with it: blocking matters late, and early those cars are worth more
    /// in your own network.
    progress: f64,
    endgame: bool,
    plan_completable: bool,
    /// H3+: the final round is under way, so this turn is this seat's last. Nothing drawn
    /// now is ever spent, and only the points a claim scores this turn count.
    final_turn: bool,
    surplus_value: f64,
    locomotive_value: f64,
    /// Plan segments this seat can pay for right now.
    affordable: Vec<u16>,
}

impl Ctx {
    fn build(
        state: &State,
        seat: usize,
        params: &HeuristicParams,
        plan: &Plan,
        tier: Tier,
    ) -> Self {
        let board = state.board;
        let k = board.n_card_types;
        let hand = &state.hand[seat * k..(seat + 1) * k];
        let loco = board.locomotive;

        // What a car of plan progress is worth. The stake is doubled because a ticket
        // settles at +points when made and -points when missed, so completing one swings
        // twice its face value. The denominator is convex in the distance still to go: a
        // plan three cars from done is worth more per car than one twelve cars from done,
        // which is the paper's convex path cost expressed in this currency.
        let stake = f64::from(plan.live_points);
        let remaining = f64::from(plan.steiner_cost.max(1)).powf(params.convex_path_exponent);
        let plan_car_value = if plan.segments.is_empty() {
            0.0
        } else {
            2.0 * stake / remaining
        };

        let mut deficit = [0u16; MAX_CARD_TYPES];
        for c in 0..board.n_colors {
            deficit[c] = plan.need[c].saturating_sub(u16::from(hand[c]));
        }
        let colour_surplus: u16 = (0..board.n_colors)
            .map(|c| u16::from(hand[c]).saturating_sub(plan.need[c]))
            .sum();
        let gray_unmet = plan.need[loco as usize] > colour_surplus + u16::from(hand[loco as usize]);
        deficit[loco as usize] = plan.need[loco as usize]
            .saturating_sub(colour_surplus + u16::from(hand[loco as usize]));

        let total_points: u32 = (0..board.n_segments)
            .map(|s| u32::from(board.route_points[board.seg_len[s] as usize]))
            .sum();
        let points_per_car = f64::from(total_points) / f64::from(board.total_spaces);

        let (mut sum, mut cars, mut count) = (0u32, 0u32, 0u32);
        for &t in &state.tdeck[..board.n_tickets] {
            if t != crate::config::NO_TICKET {
                let t = t as usize;
                sum += u32::from(board.ticket_points[t]);
                cars +=
                    u32::from(board.dist[board.ticket_a[t] as usize][board.ticket_b[t] as usize]);
                count += 1;
            }
        }
        let (mean_remaining_ticket_points, mean_ticket_cars) = if count == 0 {
            (0.0, 0.0)
        } else {
            (
                f64::from(sum) / f64::from(count),
                f64::from(cars) / f64::from(count),
            )
        };

        let trains = u32::from(state.trains[seat]);
        let supply = f64::from(board.raw.trains_per_player);
        let affordable = plan
            .segments
            .iter()
            .copied()
            .filter(|&s| can_pay(state, seat, board, s))
            .collect();

        Self {
            seat,
            deficit,
            gray_unmet,
            locomotive: loco,
            plan_car_value,
            points_per_car,
            mean_remaining_ticket_points,
            mean_ticket_cars,
            // The *fastest* seat, not this one: the game ends when anybody runs out, so
            // how late it is is set by whoever is furthest along.
            progress: 1.0
                - f64::from(
                    state.trains[..state.n_players()]
                        .iter()
                        .copied()
                        .min()
                        .unwrap_or(0),
                ) / supply,
            endgame: f64::from(trains) < params.endgame_train_fraction * supply,
            plan_completable: plan.steiner_cost <= trains,
            // Every seat gets exactly one turn once the final round starts, so whoever is to
            // move is on their last. Knowing it is H3's; H2 plays its last turn like any other.
            final_turn: tier >= Tier::H3 && state.final_left != NOT_TRIGGERED,
            surplus_value: params.surplus_card_value,
            locomotive_value: params.locomotive_value,
            affordable,
        }
    }

    /// The decision-wide quantities behind every candidate's terms, for a trace.
    fn report<S: Sink>(&self, plan: &Plan, state: &State, sink: &mut S) {
        sink.context("plan_cost", f64::from(plan.steiner_cost));
        sink.context("plan_points", f64::from(plan.live_points));
        sink.context("plan_segments", plan.segments.len() as f64);
        sink.context("unreachable_tickets", plan.dead_tickets.len() as f64);
        sink.context("plan_car_value", self.plan_car_value);
        sink.context("trains", f64::from(state.trains[self.seat]));
        sink.context("endgame", f64::from(u8::from(self.endgame)));
        sink.context(
            "plan_completable",
            f64::from(u8::from(self.plan_completable)),
        );
        sink.context("progress", self.progress);
        sink.context("final_turn", f64::from(u8::from(self.final_turn)));
    }

    /// What one card of `card` is worth, in points.
    fn card_value(&self, card: u8) -> f64 {
        if self.final_turn {
            return 0.0; // nothing after this turn can spend it
        }
        if card == self.locomotive {
            return self.locomotive_value * self.plan_car_value;
        }
        if self.deficit[card as usize] > 0 || self.gray_unmet {
            self.plan_car_value
        } else {
            self.surplus_value * self.plan_car_value
        }
    }

    /// Could opponent `o` pay for `segment` within the threat horizon?
    ///
    /// Their certain cards are public; their blind draws are not, so those count as
    /// wildcards. A threat an opponent cannot execute is not a threat, and pricing one is
    /// how a blocking heuristic ends up spending its whole game denying routes nobody wants.
    fn opponent_could_pay(&self, state: &State, o: usize, segment: u16, length: u8) -> bool {
        let board = state.board;
        let k = board.n_card_types;
        let certain = &state.certain[o * k..(o + 1) * k];
        let loco = usize::from(self.locomotive);
        let colour = board.seg_color[segment as usize];
        let best_colour = if colour == GRAY {
            certain[..board.n_colors].iter().copied().max().unwrap_or(0)
        } else {
            certain[colour as usize]
        };
        let reachable =
            u16::from(best_colour) + u16::from(certain[loco]) + u16::from(state.unknown[o]);
        u16::from(state.trains[o]) >= u16::from(length) && reachable >= u16::from(length)
    }
}

/// Cards the plan still has to draw with `hand`: each colour's shortfall, plus the gray
/// demand the colour surplus cannot cover, less the locomotives, which cover either.
fn plan_shortfall(hand: &[u8], need: &[u16; MAX_CARD_TYPES], n_colors: usize, loco: usize) -> u16 {
    let (mut short, mut surplus) = (0u16, 0u16);
    for c in 0..n_colors {
        let (have, want) = (u16::from(hand[c]), need[c]);
        short += want.saturating_sub(have);
        surplus += have.saturating_sub(want);
    }
    let gray = need[loco].saturating_sub(surplus);
    (short + gray).saturating_sub(u16::from(hand[loco]))
}

/// Whether `seat` holds cards enough to claim `segment` right now.
fn can_pay(state: &State, seat: usize, board: &Board, segment: u16) -> bool {
    let k = board.n_card_types;
    let hand = &state.hand[seat * k..(seat + 1) * k];
    let length = board.seg_len[segment as usize];
    if state.trains[seat] < length {
        return false;
    }
    let wilds = hand[board.locomotive as usize];
    if wilds >= length {
        return true;
    }
    let colour = board.seg_color[segment as usize];
    if colour == GRAY {
        (0..board.n_colors).any(|c| hand[c] > 0 && hand[c] + wilds >= length)
    } else {
        let held = hand[colour as usize];
        held > 0 && held + wilds >= length
    }
}
