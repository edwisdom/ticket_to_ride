//! What a heuristic was thinking: the sink its one scorer reports to.
//!
//! PLAN.md §14.2 wants every bot decision explainable -- each candidate's total and the
//! terms that made it -- for the post-game review, and the explanation must be the *same
//! computation* as the play. A second scorer written for display would drift from the real
//! one in exactly the cases worth reviewing, so there is none: the scorer takes a [`Sink`]
//! and reports to it as it goes.
//!
//! * [`Silent`] is what [`Heuristic::act`](crate::heuristic::Heuristic::act) passes. Every
//!   method is an empty default, so after monomorphization the calls and the values built
//!   only for them compile away, and play costs what it did.
//! * [`Recorder`] is what [`Heuristic::explain`](crate::heuristic::Heuristic::explain)
//!   passes. It only writes down what the scorer computed; the totals it records are the
//!   numbers the decision compared, not a re-addition of the terms.
//!
//! **A sink only records.** Nothing in the scorer may branch on a sink beyond
//! [`Sink::LISTENING`] guarding work whose sole purpose is the trace, and that work must not
//! touch a cache the decision reads. That is what makes `explain` a drop-in for `act`: same
//! action, same plan cache, same random stream.

use crate::heuristic::policy::Tier;

/// Where the scorer reports. Every method defaults to doing nothing.
pub trait Sink {
    /// Whether anything is recording. Guards work done only for the trace.
    const LISTENING: bool;

    /// A new candidate action is being scored; the calls up to the next `candidate` are
    /// about it.
    fn candidate(&mut self, _action: u16) {}

    /// One additive term of the current candidate's score, in points.
    fn term(&mut self, _name: &'static str, _value: f64) {}

    /// An input behind the current candidate's terms, in its own unit: a detour in cars,
    /// completion odds, a flag. Not part of the total.
    fn fact(&mut self, _name: &'static str, _value: f64) {}

    /// The rule that ranks the current candidate (H1's cascade, a gate, a forced move).
    fn rule(&mut self, _rule: &'static str) {}

    /// The current candidate's total: exactly the number the decision compared.
    fn total(&mut self, _value: f64) {}

    /// A quantity of the whole decision rather than of one candidate.
    fn context(&mut self, _name: &'static str, _value: f64) {}
}

/// The sink `act` uses: records nothing, costs nothing.
pub struct Silent;

impl Sink for Silent {
    const LISTENING: bool = false;
}

/// One scored action.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub action: u16,
    /// What the decision compared. `-inf` for an action ruled out outright.
    pub total: f64,
    pub rule: Option<&'static str>,
    /// Additive terms, in points, in the order the scorer added them. Exact zeros are left
    /// out. They sum to `total` up to floating-point rounding; `total` is authoritative.
    pub terms: Vec<(&'static str, f64)>,
    pub facts: Vec<(&'static str, f64)>,
}

/// One decision, explained.
#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    pub tier: Tier,
    pub seat: u8,
    pub phase: u8,
    /// The action the decision returned.
    pub chosen: u16,
    /// The chosen candidate's rule, if it had one.
    pub rule: Option<&'static str>,
    pub context: Vec<(&'static str, f64)>,
    /// Every candidate the decision compared, best first: total descending, ties to the
    /// lowest action id -- the decision's own tiebreak, so for H1-H4 the first is `chosen`.
    /// H0 compares nothing; its candidates are the uniform distribution it drew from.
    pub candidates: Vec<Candidate>,
}

impl Explanation {
    /// Keep the best `k` candidates, and the chosen one wherever it ranked.
    ///
    /// Only H0 can choose outside the top: its candidates all tie.
    pub fn truncate(&mut self, k: usize) {
        if self.candidates.len() <= k {
            return;
        }
        let chosen = self.candidates.iter().position(|c| c.action == self.chosen);
        let keep_chosen = chosen
            .filter(|&i| i >= k)
            .map(|i| self.candidates[i].clone());
        self.candidates.truncate(k);
        self.candidates.extend(keep_chosen);
    }
}

/// The sink `explain` uses.
#[derive(Default)]
pub struct Recorder {
    context: Vec<(&'static str, f64)>,
    candidates: Vec<Candidate>,
}

impl Recorder {
    fn current(&mut self) -> &mut Candidate {
        self.candidates
            .last_mut()
            .expect("the scorer reported a term outside a candidate")
    }

    /// Close the record of a decision that returned `chosen`.
    pub fn finish(mut self, tier: Tier, seat: u8, phase: u8, chosen: u16) -> Explanation {
        self.candidates
            .sort_by(|a, b| b.total.total_cmp(&a.total).then(a.action.cmp(&b.action)));
        let rule = self
            .candidates
            .iter()
            .find(|c| c.action == chosen)
            .and_then(|c| c.rule);
        Explanation {
            tier,
            seat,
            phase,
            chosen,
            rule,
            context: self.context,
            candidates: self.candidates,
        }
    }
}

impl Sink for Recorder {
    const LISTENING: bool = true;

    fn candidate(&mut self, action: u16) {
        self.candidates.push(Candidate {
            action,
            total: f64::NEG_INFINITY,
            rule: None,
            terms: Vec::new(),
            facts: Vec::new(),
        });
    }

    fn term(&mut self, name: &'static str, value: f64) {
        if value != 0.0 {
            self.current().terms.push((name, value));
        }
    }

    fn fact(&mut self, name: &'static str, value: f64) {
        self.current().facts.push((name, value));
    }

    fn rule(&mut self, rule: &'static str) {
        self.current().rule = Some(rule);
    }

    fn total(&mut self, value: f64) {
        self.current().total = value;
    }

    fn context(&mut self, name: &'static str, value: f64) {
        self.context.push((name, value));
    }
}
