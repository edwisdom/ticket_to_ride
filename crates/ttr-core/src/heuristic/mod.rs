//! The scripted agent ladder, H0-H4, and the machinery that keeps H3 a usable Elo anchor.
//!
//! These live in Rust rather than Python for one reason (PLAN.md §7, §8.3): **H3 doubles as
//! the ISMCTS rollout policy in Phase 5**, and a rollout that crosses the FFI boundary once
//! per step defeats the entire batching design. Building the agents after the port means
//! writing them once instead of writing them in Python and porting them later.
//!
//! ## H3 is the Elo anchor, and a name is not a freeze
//!
//! PLAN.md §11 anchors the rating scale on H3, fixed at Elo 1500 (`eval.stats.ANCHOR_ELO`).
//! That only works if H3 is *the same player* across every invocation, months apart, until
//! it is moved on purpose. Putting its constants in [`params::HeuristicParams`] is
//! necessary and not sufficient: it pins the numbers and leaves the code around them free. A tiebreak reordered, a change to
//! [`crate::graph::MAX_STEINER_TERMINALS`], a fix in the plan's path reconstruction -- each
//! changes what H3 plays with `params_hash` unmoved, and every rating ever recorded against
//! it silently re-bases.
//!
//! So identity is a **behaviour hash** ([`probe`]): H3's actual action sequence over a fixed
//! probe set, pinned as a golden literal. Constants *or* code, if the play changes the test
//! says so. `params_hash` is provenance recorded alongside; the behaviour hash is identity.
//!
//! The corollary is how tuning works: an experiment is a **new agent** (`h3@...`) rated
//! *against* the anchor, and H3 itself changes only as a recorded event -- measured against
//! its predecessor, the golden hash updated, the re-basing noted in docs/WORKLOG.md. That is
//! what lets ratings accumulate across months rather than quietly re-basing. It has
//! happened once: the first H3 was too weak to be the bar, so it learned to plan the network
//! its cards can pay for and to spend its last turn on points.

pub mod params;
pub mod plan;
pub mod policy;
pub mod probe;
pub mod trace;

pub use params::{HeuristicParams, PARAMS_VERSION};
pub use plan::Plan;
pub use policy::{Heuristic, Tier};
pub use probe::{behaviour_hash, probe_configs};
pub use trace::{Candidate, Explanation};
