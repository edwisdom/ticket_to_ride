//! The heuristics' constants, in one struct, **never inline in the decision code**.
//!
//! This exists because of a specific failure in the one published attempt at this problem
//! (PLAN.md §1). Its heuristic baselines were lifted from a public repository whose ticket
//! appetite was written as `trains > 15`. On its own 10-train map that condition can never
//! be true, so every "well-designed heuristic" it benchmarked against played its opening
//! tickets and never drew another -- and the paper's entire ladder was measured against
//! agents that were silently broken.
//!
//! The lesson is not "put the constants in a struct". It is **express every threshold as a
//! fraction of a quantity the board itself defines**, so carrying the same numbers to a
//! smaller map changes the threshold with it. `trains >= 0.55 * trains_per_player`
//! transfers from a 45-train board to a 20-train board; `trains >= 15` does not. Where a
//! field below is an absolute count, there is a comment saying why that is the right
//! dimension for it.
//!
//! Two hashes come out of this, and they answer different questions:
//!
//! * [`HeuristicParams::params_hash`] -- *were these numbers changed?* Provenance. Recorded
//!   next to every rating.
//! * The **behaviour hash** in [`crate::heuristic::probe`] -- *does this agent still play
//!   the same way?* Identity. That is the one the Elo anchor is pinned by, because a params
//!   hash pins the numbers and leaves the code around them free to drift.

use std::fmt::Write as _;

use crate::hashing::hash128_hex;
use crate::heuristic::policy::Tier;

/// Bumped when a field is added, removed or renamed -- i.e. when a stored parameter set
/// from an older build can no longer be interpreted. Editing a *value* does not bump this;
/// that shows up in [`HeuristicParams::params_hash`] instead.
///
/// Version 3 added the first fields only some tiers read, and with them the rule in
/// [`HeuristicParams::canonical`]: a tier's canonical text carries version 2's fields and
/// only the later fields it reads, so H0-H3 hash exactly as they did under version 2.
pub const PARAMS_VERSION: u32 = 3;

/// The version whose field list every tier's canonical text carries, and its prefix.
const BASE_VERSION: u32 = 2;

/// Every constant H1-H4 consult.
///
/// One struct rather than one per tier, because the tiers share machinery and splitting the
/// constants would mean keeping several copies of "how much is a train worth" in step. The
/// objection -- that a field added for H4 then moves H3's `params_hash` -- is met in
/// [`HeuristicParams::canonical`]: a field added after version 2 reaches only the hashes of
/// the tiers that read it. That matters because `params_hash` is part of an agent's key in
/// both stores, so moving H3's would orphan every game and rating recorded against it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeuristicParams {
    // -- ticket appetite (H1+) ------------------------------------------
    /// Draw more tickets only while at least this fraction of the train supply is left.
    ///
    /// **This is the field the published prior art got wrong.** Keep it a fraction.
    pub ticket_draw_train_fraction: f64,
    /// H1 only: never hold more than this many tickets.
    ///
    /// Absolute, and deliberately so. The prior art's trap was a threshold denominated in
    /// *trains* carried to a board with fewer trains; a ticket count is not that quantity.
    /// H2+ replace this cap entirely with the Steiner-cost budget below, which is relative.
    pub h1_max_tickets: u8,
    /// H1 only: take a face-up locomotive when one is showing.
    pub h1_take_faceup_locomotive: bool,

    // -- the plan (H2+) -------------------------------------------------
    /// The fraction of remaining trains a plan may consume before it is even money to
    /// complete. Governs **both** keeping tickets and drawing them, because they are the
    /// same decision asked twice, and a board where the two disagree is a board where the
    /// agent takes on commitments it then refuses to extend.
    pub plan_utilization_target: f64,
    /// How sharply completion odds fall as a plan passes [`Self::plan_utilization_target`].
    /// Higher is more of a cliff; 1 is nearly linear.
    pub completion_sharpness: f64,
    /// H2+: with no incomplete tickets left, keep claiming while trains remain -- but only
    /// routes paying at least this many points per train. Below it, drawing tickets or
    /// cards is worth more than filling the board with 1-car spurs.
    pub min_points_per_train: f64,
    /// Weight on "losing this segment would raise my plan's cost", when choosing between
    /// affordable segments that are all on the plan.
    pub contest_weight: f64,
    /// Extra weight on the last free track of a double route while doubles lock for
    /// everyone (2-3P). Claiming one there denies the pair outright.
    pub double_lockout_weight: f64,

    // -- H3: valuing a draw ---------------------------------------------
    /// How many cards a turn spent drawing is worth. Two, less a haircut for the turns that
    /// only get one (a face-up locomotive, or an empty pool).
    pub cards_per_draw_turn: f64,
    /// Value of a face-up card of a colour the plan needs, relative to a blind draw.
    pub faceup_needed_bonus: f64,
    /// Value of a face-up locomotive, in cards. Worth more than one because it substitutes
    /// for any colour, and worth less than two because taking it ends the turn.
    pub locomotive_value: f64,
    /// A colour is worth stockpiling past the plan's needs at this discount, so a heuristic
    /// with a fully-paid plan still prefers useful cards to arbitrary ones.
    pub surplus_card_value: f64,

    // -- H3: a plan the cards can pay for --------------------------------
    /// Train cars one blind draw is worth when choosing *which* network to plan. 0 plans by
    /// train cars alone, as H2 does.
    ///
    /// The cheapest network by cars is rarely unique, and the cards decide which one gets
    /// built. A colour-blind plan breaks the tie by city order and then waits on whatever
    /// colours that happens to need: one H3 game spent its last ten turns drawing toward a
    /// 6-orange and a 5-yellow route while the draw pile held fewer of those cards than it
    /// needed, and an equally short network it could pay for -- two face-up cards away --
    /// went unbuilt. Pricing each segment's missing cards in draws, by what is face up and
    /// what is still unseen, steers the plan to the network the hand can finish.
    ///
    /// Small on purpose: at the default a car outweighs dozens of draws, so in practice the
    /// cards break ties between equally short networks. Heavier weights trade route length
    /// for payability and measured worse (docs/WORKLOG.md, "H3 moves").
    pub plan_card_weight: f64,

    // -- H3: the endgame ------------------------------------------------
    /// Below this fraction of the train supply, switch to maximizing points per train.
    ///
    /// The rulebook's own trigger is 2 trains; this is the *approach* to it, where spending
    /// trains on plan segments you can no longer finish is worse than banking points.
    pub endgame_train_fraction: f64,
    /// In the endgame, a ticket that can still be completed is worth its face value times
    /// this. Above 1 because a completed ticket swings twice its points (it stops being a
    /// penalty and starts being a bonus).
    pub endgame_ticket_urgency: f64,

    // -- H3: the paper's two scoring ideas ------------------------------
    /// Convex path cost: a plan segment `c` trains from completion is discounted by
    /// `c^(-convex_path_exponent)`. Superlinear in the remaining distance, so a nearly-done
    /// connection outranks a long-shot one instead of merely tying it.
    pub convex_path_exponent: f64,
    /// Penalty for paying a gray route with a colour that some *coloured* route on the plan
    /// still needs. Gray cannibalization: gray routes accept anything, so paying them with
    /// a scarce colour spends flexibility that only the coloured route can use.
    pub gray_cannibalization_penalty: f64,
    /// Weight on "this claim extends my longest chain" when otherwise-equal claims tie. The
    /// bonus is 10 points and is often decisive; it is still a tiebreak, not a plan.
    pub longest_chain_weight: f64,

    // -- H3: keeping tickets --------------------------------------------
    /// A ticket whose endpoints are already unreachable through my own network is worth
    /// `-points` and is never kept unless the keep minimum forces it.
    pub ticket_keep_dead_penalty: f64,

    // -- H4: playing against someone ------------------------------------
    /// Weight on the threatened-edge primitive: an unclaimed edge that would merge two
    /// components of an opponent's network, scored by the network length it would join.
    /// A much sharper notion of blocking than "a route they might want" (PLAN.md §1).
    pub threat_weight: f64,
    /// Only price a threat when the opponent could actually pay for it: their known cards
    /// plus this many turns of drawing.
    pub threat_horizon_turns: f64,
    /// Super-linearity in how much network a block severs (§1's threat-pricing rule 4):
    /// cutting a 6-car component off a network is worth more than three times cutting a
    /// 2-car one, because the pieces are harder to reconnect the larger they are.
    pub threat_severance_exponent: f64,
    /// The `k` in §1's `(n_threats - 1) x k`: when an opponent has many merging edges
    /// available, denying any one of them is not decisive, so each is worth less.
    pub threat_crowding_penalty: f64,
    /// Hoard-and-strike: hold a claim back if waiting a turn would let a *more* contested
    /// segment be taken instead, provided the held one is at least this safe (its cost to
    /// me if lost, relative to the alternative's).
    pub hoard_safety_margin: f64,
    /// In 2-3P, grab a contested double-route half this many turns earlier than value alone
    /// would justify, because claiming it closes the pair to everyone.
    pub double_grab_urgency: f64,

    // -- H4: what an off-plan claim takes from the plan -------------------
    /// Weight on the trains an off-plan claim takes from the plan, priced as the expected
    /// ticket settlement they cost under [`Self::completion_odds`]. 0 leaves only H3's flat
    /// [`Self::min_points_per_train`].
    pub plan_budget_weight: f64,
    /// Weight on the cards an off-plan claim spends that the plan would have used, at the
    /// value of a car of plan progress each. 0 leaves only H3's gray cannibalization.
    pub plan_card_spend_weight: f64,

    // -- H5: tickets priced against the network ----------------------------
    /// Price a ticket draw against this seat's own network: each ticket it might be dealt
    /// at the cars from its network and plan to the ticket's cities, the offer at its
    /// expected best keep, and no train gate. Off is H3's mean-ticket model with its gate.
    pub ticket_draw_network: bool,

    // -- H5: the longest-route race ---------------------------------------
    /// Weight on the longest-route race: a claim's change in this seat's exact longest
    /// trail against the longest rival's, priced as odds on the bonus. It replaces
    /// [`Self::longest_chain_weight`]'s tiebreak, which knew neither trail. 0 = the tiebreak.
    pub longest_race_weight: f64,
    /// How fast those odds blur with the trains still to be placed: a logistic in the lead,
    /// at a scale of `1 + spread × (my trains + the rival's)` cars.
    pub longest_race_spread: f64,
    /// What a train car costs a claim once every held ticket is done, in place of
    /// [`Self::min_points_per_train`]. That charge keeps trains for the plan; with no plan
    /// left a train's only use is points, and one left unplaced when the game ends is
    /// worth nothing.
    pub finished_train_cost: f64,
}

impl Default for HeuristicParams {
    /// **The anchor's values.** Changing any of these moves H3, the anchor of every rating --
    /// the golden test in `tests/heuristics.rs` fails and says so. Experiment with a modified
    /// copy registered as its own spec (`h3@...`), rated against H3; edit these only to move
    /// the anchor on purpose, with the measurement recorded in docs/WORKLOG.md.
    fn default() -> Self {
        Self {
            ticket_draw_train_fraction: 0.55,
            h1_max_tickets: 4,
            h1_take_faceup_locomotive: true,

            plan_utilization_target: 0.75,
            completion_sharpness: 3.0,
            min_points_per_train: 1.4,
            contest_weight: 0.8,
            double_lockout_weight: 1.5,

            cards_per_draw_turn: 1.85,
            faceup_needed_bonus: 0.45,
            locomotive_value: 1.6,
            surplus_card_value: 0.15,

            plan_card_weight: 0.01,

            endgame_train_fraction: 0.18,
            endgame_ticket_urgency: 2.0,

            convex_path_exponent: 1.35,
            gray_cannibalization_penalty: 0.5,
            longest_chain_weight: 0.35,

            ticket_keep_dead_penalty: 2.0,

            threat_weight: 0.7,
            threat_horizon_turns: 2.0,
            threat_severance_exponent: 1.3,
            threat_crowding_penalty: 0.5,
            hoard_safety_margin: 1.25,
            double_grab_urgency: 2.0,

            plan_budget_weight: 1.0,
            plan_card_spend_weight: 1.0,

            ticket_draw_network: true,

            longest_race_weight: 1.0,
            longest_race_spread: 0.4,
            finished_train_cost: 0.0,
        }
    }
}

impl HeuristicParams {
    /// `tier`'s parameters as `name=value` text, the input to [`Self::params_hash`]: version
    /// 2's fields, then each later field `tier` reads, in declaration order.
    ///
    /// **Version 2's fields are every tier's**, H4's threat terms included, because that is
    /// what every hash recorded before version 3 covered; a field added since reaches only
    /// the tiers that read it. So adding a field for one tier leaves every other tier's
    /// hash -- and its key in the results store and the play store -- where it was.
    ///
    /// Floats use `{:?}`, which is the shortest representation that round-trips, so the
    /// text is a faithful image of the bits rather than a rounded rendering of them. Only
    /// this crate computes the hash -- Python reads it across the FFI -- so nothing depends
    /// on CPython formatting the same way.
    pub fn canonical(&self, tier: Tier) -> String {
        let mut out = format!("v{BASE_VERSION}");
        for (name, value, reader) in self.fields() {
            if reader.is_none_or(|r| tier >= r) {
                write!(out, "|{name}={value}").expect("String write");
            }
        }
        out
    }

    /// Every field as `(name, value, reader)`, in declaration order, the value formatted as
    /// [`Self::canonical`] writes it. `reader` is the lowest tier that reads a field added
    /// after version 2, and `None` for version 2's own.
    pub fn fields(&self) -> Vec<(&'static str, String, Option<Tier>)> {
        let mut out = Vec::new();
        macro_rules! field {
            ($name:ident) => {
                out.push((stringify!($name), format!("{:?}", self.$name), None))
            };
            ($name:ident, $reader:expr) => {
                out.push((
                    stringify!($name),
                    format!("{:?}", self.$name),
                    Some($reader),
                ))
            };
        }
        field!(ticket_draw_train_fraction);
        field!(h1_max_tickets);
        field!(h1_take_faceup_locomotive);
        field!(plan_utilization_target);
        field!(completion_sharpness);
        field!(min_points_per_train);
        field!(contest_weight);
        field!(double_lockout_weight);
        field!(cards_per_draw_turn);
        field!(faceup_needed_bonus);
        field!(locomotive_value);
        field!(surplus_card_value);
        field!(plan_card_weight);
        field!(endgame_train_fraction);
        field!(endgame_ticket_urgency);
        field!(convex_path_exponent);
        field!(gray_cannibalization_penalty);
        field!(longest_chain_weight);
        field!(ticket_keep_dead_penalty);
        field!(threat_weight);
        field!(threat_horizon_turns);
        field!(threat_severance_exponent);
        field!(threat_crowding_penalty);
        field!(hoard_safety_margin);
        field!(double_grab_urgency);
        field!(plan_budget_weight, Tier::H4);
        field!(plan_card_spend_weight, Tier::H4);
        field!(ticket_draw_network, Tier::H5);
        field!(longest_race_weight, Tier::H5);
        field!(longest_race_spread, Tier::H5);
        field!(finished_train_cost, Tier::H5);
        out
    }

    /// blake2b-128 over [`HeuristicParams::canonical`] for `tier`. Provenance, not identity.
    pub fn params_hash(&self, tier: Tier) -> String {
        hash128_hex(self.canonical(tier).as_bytes())
    }

    /// Odds that a plan costing `cars` train cars gets completed with `trains` left.
    ///
    /// **The one model behind both ticket decisions.** PLAN.md §7 specifies H3's keep
    /// filter as `marginal_steiner_cost <= points`; that rule is budget-blind, and on
    /// TTR-mini -- where ticket points *are* shortest-path costs, so marginal always equals
    /// points -- it accepts every ticket. Measured: H3 kept 3 opening tickets costing 22
    /// train cars against a 20-train supply, on 59 of 60 games, and then correctly refused
    /// to draw another for the rest of the game because the plan it had just taken on was
    /// already unaffordable. `draw_tickets` fired **zero** times across 30 mini games while
    /// USA drew freely.
    ///
    /// That is the published prior art's failure with the sign flipped: not a threshold
    /// that can never fire, but one that always does. The fix is to price a ticket by
    /// whether the *train budget* can absorb it, which is the quantity that actually
    /// differs between a 45-train board and a 20-train board.
    ///
    /// Even money at `plan_utilization_target`; approaching 1 as the plan shrinks and 0 as
    /// it outgrows the trains left. A settled ticket pays `+points` and a missed one
    /// `-points`, so expected value is `points * (2 * odds - 1)` and the break-even point
    /// is exactly `odds = 0.5`.
    pub fn completion_odds(&self, cars: f64, trains: f64) -> f64 {
        if cars <= 0.0 {
            return 1.0;
        }
        if trains <= 0.0 {
            return 0.0;
        }
        let utilization = cars / trains;
        let ratio = utilization / self.plan_utilization_target.max(f64::EPSILON);
        1.0 / (1.0 + ratio.powf(self.completion_sharpness))
    }

    /// Reject values that would make a heuristic degenerate rather than merely differently
    /// tuned.
    ///
    /// Deliberately loose: this is not a taste filter on tuning, it is a guard against the
    /// settings that produce a *silently* broken agent -- a negative exponent inverting the
    /// convex cost, or a fraction above 1 that can never be satisfied on any board, which
    /// is the prior art's bug expressed in the new units.
    pub fn validate(&self) -> Result<(), String> {
        let fractions = [
            (
                "ticket_draw_train_fraction",
                self.ticket_draw_train_fraction,
            ),
            ("plan_utilization_target", self.plan_utilization_target),
            ("endgame_train_fraction", self.endgame_train_fraction),
        ];
        for (name, value) in fractions {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "{name}={value} is a fraction of a board quantity and must lie in \
                     0..=1; outside it the condition can never fire, which is exactly the \
                     failure that invalidated the published prior art's baselines"
                ));
            }
        }
        if self.convex_path_exponent <= 0.0 {
            return Err(format!(
                "convex_path_exponent={} must be positive; at or below zero the discount \
                 stops being convex and prefers *longer* remaining paths",
                self.convex_path_exponent
            ));
        }
        if self.cards_per_draw_turn <= 0.0 {
            return Err("cards_per_draw_turn must be positive".into());
        }
        if self.h1_max_tickets == 0 {
            return Err("h1_max_tickets=0 stops H1 ever keeping a ticket".into());
        }
        if self.plan_utilization_target <= 0.0 {
            return Err("plan_utilization_target must be positive".into());
        }
        if self.completion_sharpness <= 0.0 {
            return Err("completion_sharpness must be positive".into());
        }
        if !(self.plan_card_weight >= 0.0 && self.plan_card_weight.is_finite()) {
            return Err(format!(
                "plan_card_weight={} must be finite and at least 0 (0 = colour-blind)",
                self.plan_card_weight
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_validate() {
        HeuristicParams::default().validate().unwrap();
    }

    #[test]
    fn every_field_reaches_the_hash() {
        // The failure this catches is a field added to the struct and forgotten in
        // `fields()`: the parameter would then be tunable and invisible, so two agents
        // that play differently would record the same provenance. Mutating each field in
        // turn is the only check that scales with the struct. The top tier reads every
        // field, so its hash is the one every field must reach.
        let top = *Tier::ALL.last().expect("a tier");
        let base = HeuristicParams::default();
        let baseline = base.params_hash(top);
        let mut mutants = vec![
            HeuristicParams {
                ticket_draw_train_fraction: 0.5,
                ..base
            },
            HeuristicParams {
                h1_max_tickets: 3,
                ..base
            },
            HeuristicParams {
                h1_take_faceup_locomotive: false,
                ..base
            },
            HeuristicParams {
                plan_utilization_target: 0.5,
                ..base
            },
            HeuristicParams {
                completion_sharpness: 2.0,
                ..base
            },
            HeuristicParams {
                min_points_per_train: 1.0,
                ..base
            },
            HeuristicParams {
                contest_weight: 0.0,
                ..base
            },
            HeuristicParams {
                double_lockout_weight: 1.0,
                ..base
            },
            HeuristicParams {
                cards_per_draw_turn: 1.5,
                ..base
            },
            HeuristicParams {
                faceup_needed_bonus: 0.0,
                ..base
            },
            HeuristicParams {
                locomotive_value: 1.0,
                ..base
            },
            HeuristicParams {
                surplus_card_value: 0.0,
                ..base
            },
            HeuristicParams {
                plan_card_weight: 0.0,
                ..base
            },
            HeuristicParams {
                endgame_train_fraction: 0.1,
                ..base
            },
            HeuristicParams {
                endgame_ticket_urgency: 1.0,
                ..base
            },
            HeuristicParams {
                convex_path_exponent: 1.0,
                ..base
            },
            HeuristicParams {
                gray_cannibalization_penalty: 0.0,
                ..base
            },
            HeuristicParams {
                longest_chain_weight: 0.0,
                ..base
            },
            HeuristicParams {
                ticket_keep_dead_penalty: 1.0,
                ..base
            },
            HeuristicParams {
                threat_weight: 0.0,
                ..base
            },
            HeuristicParams {
                threat_horizon_turns: 1.0,
                ..base
            },
            HeuristicParams {
                threat_severance_exponent: 1.0,
                ..base
            },
            HeuristicParams {
                threat_crowding_penalty: 0.0,
                ..base
            },
            HeuristicParams {
                hoard_safety_margin: 1.0,
                ..base
            },
            HeuristicParams {
                double_grab_urgency: 1.0,
                ..base
            },
            HeuristicParams {
                plan_budget_weight: 0.5,
                ..base
            },
            HeuristicParams {
                plan_card_spend_weight: 0.5,
                ..base
            },
            HeuristicParams {
                ticket_draw_network: false,
                ..base
            },
            HeuristicParams {
                longest_race_weight: 0.5,
                ..base
            },
            HeuristicParams {
                longest_race_spread: 0.5,
                ..base
            },
            HeuristicParams {
                finished_train_cost: 1.0,
                ..base
            },
        ];
        // One per field, and every one distinct: a `fields()` that dropped a field would
        // collide with the baseline, and one that wrote a field twice would still be
        // distinct, so the count check is what catches a stale list here.
        let mut hashes: Vec<String> = mutants.iter().map(|m| m.params_hash(top)).collect();
        for (mutant, hash) in mutants.iter_mut().zip(&hashes) {
            assert_ne!(*hash, baseline, "{mutant:?} did not move params_hash");
        }
        hashes.sort();
        hashes.dedup();
        assert_eq!(hashes.len(), mutants.len(), "two mutants hash alike");
        assert_eq!(
            mutants.len(),
            base.fields().len(),
            "the mutant list has drifted from the field list; add the new field to both"
        );
    }

    /// H3's `params_hash` under version 2, as stored with every game and rating against it.
    const H3_PARAMS_HASH: &str = "17df54a531efc8228c222931713e53eb";

    #[test]
    fn fields_added_for_one_tier_leave_the_others_hashes_alone() {
        // `params_hash` is part of an agent's key in the results store and the play store,
        // so a field added for H4 that reached H3's hash would orphan every game and rating
        // recorded against the anchor: the server would refuse to replay them.
        let base = HeuristicParams::default();
        for tier in [Tier::H0, Tier::H1, Tier::H2, Tier::H3] {
            assert_eq!(base.params_hash(tier), H3_PARAMS_HASH, "{}", tier.name());
        }
        let h4_only = HeuristicParams {
            plan_budget_weight: 0.25,
            ..base
        };
        assert_eq!(h4_only.params_hash(Tier::H3), H3_PARAMS_HASH);
        assert_ne!(h4_only.params_hash(Tier::H4), base.params_hash(Tier::H4));
    }

    #[test]
    fn fractions_outside_the_unit_interval_are_refused() {
        // The prior art's bug in the new units: a threshold that can never be satisfied.
        let bad = HeuristicParams {
            ticket_draw_train_fraction: 1.5,
            ..HeuristicParams::default()
        };
        let message = bad.validate().unwrap_err();
        assert!(message.contains("prior art"), "{message}");
    }

    #[test]
    fn a_non_convex_exponent_is_refused() {
        let bad = HeuristicParams {
            convex_path_exponent: 0.0,
            ..HeuristicParams::default()
        };
        assert!(bad.validate().is_err());
    }
}
