//! H0-H4. One nested ladder, not five agents.
//!
//! PLAN.md §7 defines the tiers as strict extensions -- H3 is H2 plus four things, and H4
//! is H3 plus opponent modeling -- so they are written that way: one scoring function whose
//! terms switch on [`Tier`]. Five parallel implementations would let the tiers drift apart
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

use crate::actions::BLIND_SLOT;
use crate::board::{Board, GRAY, MAX_CARD_TYPES, NO_SIBLING};
use crate::config::{
    EMPTY_SLOT, FREE, PHASE_DRAW_SECOND, PHASE_INITIAL_TICKETS, PHASE_MAIN, PHASE_TICKET_KEEP,
};
use crate::graph::steiner_cost_exact;
use crate::heuristic::params::HeuristicParams;
use crate::heuristic::plan::Plan;
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
    /// H2 + draw values, endgame, longest-path tiebreaks, ticket-keep EV, convex path cost,
    /// gray cannibalization. **The permanent Elo anchor.**
    H3 = 3,
    /// H3 + opponent modeling: threatened edges, threat pricing, double grabs, hoarding.
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
        let seat = state.current_player() as usize;
        self.legal.clear();
        state.legal_into(&mut self.legal);
        self.legal.sort_unstable();
        debug_assert!(!self.legal.is_empty(), "asked to act in a dead position");

        if self.legal.len() == 1 {
            return self.legal[0];
        }
        if self.tier == Tier::H0 {
            let i = self.rng.below(self.legal.len() as u32) as usize;
            return self.legal[i];
        }

        match state.phase {
            PHASE_INITIAL_TICKETS | PHASE_TICKET_KEEP => self.act_keep(state, seat),
            PHASE_MAIN | PHASE_DRAW_SECOND => {
                if self.tier == Tier::H1 {
                    self.act_h1(state, seat)
                } else {
                    self.act_scored(state, seat)
                }
            }
            _ => self.legal[0],
        }
    }

    // ------------------------------------------------------------------
    // Caches
    // ------------------------------------------------------------------

    fn refresh(&mut self, state: &mut State, seat: usize) {
        if !self.plan_ready || !self.plan.is_fresh(state, seat) {
            self.plan = Plan::build(state, seat);
            self.plan_ready = true;
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
    // H1 -- a shallow rule cascade, kept procedural on purpose
    // ------------------------------------------------------------------

    fn act_h1(&mut self, state: &State, seat: usize) -> u16 {
        let space = state.space;
        let board = state.board;
        if state.phase == PHASE_MAIN {
            // Highest route points, ties by lowest action id so play stays replayable.
            let mut best: Option<(u8, u16)> = None;
            for &action in &self.legal {
                if action >= space.claim_end {
                    break; // claims sort first
                }
                let points =
                    board.route_points[board.seg_len[(action / space.k) as usize] as usize];
                if best.is_none_or(|(p, _)| points > p) {
                    best = Some((points, action));
                }
            }
            if let Some((_, action)) = best {
                return action;
            }
            if self.h1_wants_tickets(state, seat) {
                return space.draw_tickets;
            }
        }
        self.h1_draw(state, seat)
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

    fn h1_draw(&self, state: &State, seat: usize) -> u16 {
        let (space, board) = (state.space, state.board);
        let base = seat * board.n_card_types;
        let mut best: Option<(u8, u16)> = None;
        let mut blind: Option<u16> = None;
        for &action in &self.legal {
            if !(space.draw_base..space.draw_tickets).contains(&action) {
                continue;
            }
            let slot = action - space.draw_base;
            if slot == BLIND_SLOT {
                blind = Some(action);
                continue;
            }
            let card = state.faceup[slot as usize];
            if card == EMPTY_SLOT {
                continue;
            }
            if card == board.locomotive as i8 && self.params.h1_take_faceup_locomotive {
                return action;
            }
            let held = state.hand[base + card as usize];
            if best.is_none_or(|(h, _)| held > h) {
                best = Some((held, action));
            }
        }
        match best {
            // Holding none of every face-up colour: a blind draw beats starting a new one.
            Some((0, _)) | None => blind.or_else(|| best.map(|(_, a)| a)),
            Some((_, action)) => Some(action),
        }
        .unwrap_or(self.legal[0])
    }

    // ------------------------------------------------------------------
    // H2-H4 -- the scored ladder
    // ------------------------------------------------------------------

    fn act_scored(&mut self, state: &mut State, seat: usize) -> u16 {
        self.refresh(state, seat);
        let ctx = Ctx::build(state, seat, &self.params, &self.plan);
        let space = state.space;

        let mut best = self.legal[0];
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..self.legal.len() {
            let action = self.legal[i];
            let score = if action < space.claim_end {
                self.score_claim(state, &ctx, action)
            } else if action < space.draw_tickets {
                self.score_draw(state, &ctx, action)
            } else if action == space.draw_tickets {
                self.score_draw_tickets(state, &ctx)
            } else {
                // PASS, which is only legal when nothing else is -- and that case returned
                // early. Scored below everything so it can never win a real comparison.
                f64::NEG_INFINITY
            };
            // Strictly greater: ties go to the lowest action id, which is what keeps every
            // heuristic a deterministic function of the position.
            if score > best_score {
                best = action;
                best_score = score;
            }
        }
        best
    }

    fn score_claim(&mut self, state: &mut State, ctx: &Ctx, action: u16) -> f64 {
        let board = state.board;
        let p = &self.params;
        let (segment, pay) = state.space.decode_claim(action);
        let seg = segment as usize;
        let length = board.seg_len[seg];
        let cars = f64::from(length);

        // Every car spent has an opportunity cost. This is the term that makes the ladder
        // prefer dense routes and decline spurs, and it removes the need for a separate
        // endgame "maximize points per train" branch.
        let mut score =
            f64::from(board.route_points[length as usize]) - p.min_points_per_train * cars;

        let on_plan = self.plan.contains(segment);
        if on_plan {
            let plan_value = ctx.plan_car_value * cars;
            score += plan_value;
            score += p.contest_weight * f64::from(self.plan.detour_cost(state, segment));
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
                score += p.double_lockout_weight * urgency;
            }
        }

        if self.tier >= Tier::H3 {
            score += self.h3_claim_terms(state, ctx, segment, pay, length, on_plan);
        }
        if self.tier >= Tier::H4 {
            score += self.h4_claim_terms(state, ctx, segment, length, on_plan);
        }
        score
    }

    /// H3's four claim-side additions: longest-chain tiebreak, gray cannibalization,
    /// locomotive thrift, and the endgame re-weighting of plan credit.
    fn h3_claim_terms(
        &mut self,
        state: &mut State,
        ctx: &Ctx,
        segment: u16,
        pay: u16,
        length: u8,
        on_plan: bool,
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
            extra += p.longest_chain_weight * f64::from(length);
        }

        // What this payment actually costs in flexibility.
        if let Ok((colored, wilds)) = state.payment_for(seg, length, pay as u8) {
            // Gray cannibalization: a gray route takes any colour, so paying it with one a
            // *coloured* plan segment still needs spends flexibility only that route can
            // use. Paying gray with the colour you are longest in is free; paying it with
            // your last four blues when the plan needs a 4-blue route is not.
            if board.seg_color[seg] == GRAY && (pay as u8) != board.locomotive {
                let wasted = u16::from(colored).min(ctx.deficit[pay as usize]);
                extra -= p.gray_cannibalization_penalty * ctx.plan_car_value * f64::from(wasted);
            }
            // A locomotive substitutes for any colour, so spending one where a coloured
            // card would do throws away the difference.
            extra -= (p.locomotive_value - 1.0).max(0.0) * ctx.plan_car_value * f64::from(wilds);
        }

        // Endgame. Under `endgame_train_fraction` of the supply the binding question is
        // whether the plan can still be finished at all: cars sunk into a plan that cannot
        // complete score nothing at the end, while a completable one is worth finishing at
        // almost any price.
        if ctx.endgame && on_plan {
            let base = ctx.plan_car_value * f64::from(length);
            extra += if ctx.plan_completable {
                base * (p.endgame_ticket_urgency - 1.0)
            } else {
                -base
            };
        }
        extra
    }

    /// H4's claim-side additions: the threatened-edge primitive, and hoard-and-strike.
    fn h4_claim_terms(
        &mut self,
        state: &mut State,
        ctx: &Ctx,
        segment: u16,
        length: u8,
        on_plan: bool,
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
        let mut extra = if useful {
            p.threat_weight * self.threat_value(state, ctx, segment, length)
        } else {
            0.0
        };

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
                extra -= p.contest_weight * f64::from(pending) / rivals;
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

    fn score_draw(&self, state: &State, ctx: &Ctx, action: u16) -> f64 {
        let board = state.board;
        let slot = action - state.space.draw_base;
        if slot == BLIND_SLOT {
            // Expected over what this seat cannot account for: the deck plus every
            // opponent's blind draws.
            let unseen = state.unseen_counts(ctx.seat);
            let total: i32 = unseen.iter().map(|&c| i32::from(c.max(0))).sum();
            if total == 0 {
                return 0.0;
            }
            let mut expected = 0.0;
            for (card, &count) in unseen.iter().enumerate() {
                if count > 0 {
                    expected += f64::from(count) / f64::from(total) * ctx.card_value(card as u8);
                }
            }
            return expected;
        }
        let card = state.faceup[slot as usize];
        if card == EMPTY_SLOT {
            return f64::NEG_INFINITY;
        }
        let card = card as u8;
        let mut value = ctx.card_value(card);
        // A known card of a colour the plan is short of beats the same card drawn blind:
        // taking it is certain, and certainty is what turns a plan into a claim.
        if self.tier >= Tier::H3 && card != board.locomotive && ctx.deficit[card as usize] > 0 {
            value += self.params.faceup_needed_bonus * ctx.plan_car_value;
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
    fn score_draw_tickets(&self, state: &State, ctx: &Ctx) -> f64 {
        let p = &self.params;
        let supply = f64::from(state.board.raw.trains_per_player);
        let trains = f64::from(state.trains[ctx.seat]);
        if trains < p.ticket_draw_train_fraction * supply {
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
        ctx.mean_remaining_ticket_points * (2.0 * odds - 1.0) * kept
    }

    // ------------------------------------------------------------------
    // Keeping tickets
    // ------------------------------------------------------------------

    fn act_keep(&mut self, state: &mut State, seat: usize) -> u16 {
        match self.tier {
            Tier::H1 => self.keep_h1(state),
            Tier::H2 => self.keep_scored(state, seat, false),
            _ => self.keep_scored(state, seat, true),
        }
    }

    /// H1: keep the fewest tickets allowed, choosing the shortest routes.
    fn keep_h1(&self, state: &State) -> u16 {
        let (board, space) = (state.board, state.space);
        let cost: Vec<u16> = state.offer[..state.offer_len as usize]
            .iter()
            .map(|&t| {
                board.dist[board.ticket_a[t as usize] as usize][board.ticket_b[t as usize] as usize]
            })
            .collect();
        *self
            .legal
            .iter()
            .min_by_key(|&&action| {
                let mask = action - space.keep_base();
                let kept: Vec<usize> = (0..state.offer_len as usize)
                    .filter(|i| mask >> i & 1 != 0)
                    .collect();
                (
                    kept.len(),
                    kept.iter().map(|&i| cost[i]).sum::<u16>(),
                    action,
                )
            })
            .expect("a legal keep")
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
    fn keep_scored(&mut self, state: &mut State, seat: usize, ev_filter: bool) -> u16 {
        self.refresh(state, seat);
        let board = state.board;
        let doubles_locked = state.rules.doubles_locked;
        let n_segments = board.n_segments;
        let base_terminals = self.plan.terminals.clone();
        let base_cost = self.plan.steiner_cost;
        let owner: Vec<u8> = state.seg_owner[..n_segments].to_vec();

        let mut best = self.legal[0];
        let mut best_value = f64::NEG_INFINITY;
        for &action in &self.legal {
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

            let value = if ev_filter {
                // What this ticket set settles for: `+points` each if completed, `-points`
                // each if not, weighted by whether the trains left can absorb the whole
                // resulting plan -- `cost`, not `marginal`, because the budget is shared
                // with everything already committed.
                let trains = f64::from(state.trains[seat]);
                let odds = self.params.completion_odds(f64::from(cost as u16), trains);
                points * (2.0 * odds - 1.0) - dead_penalty
            } else {
                // H2: take on as little new track as possible, fewest tickets first.
                -marginal - f64::from(mask.count_ones()) - dead_penalty
            };
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
    surplus_value: f64,
    locomotive_value: f64,
    /// Plan segments this seat can pay for right now.
    affordable: Vec<u16>,
}

impl Ctx {
    fn build(state: &State, seat: usize, params: &HeuristicParams, plan: &Plan) -> Self {
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
            surplus_value: params.surplus_card_value,
            locomotive_value: params.locomotive_value,
            affordable,
        }
    }

    /// What one card of `card` is worth, in points.
    fn card_value(&self, card: u8) -> f64 {
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
