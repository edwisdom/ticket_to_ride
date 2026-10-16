//! The Ticket to Ride engine core.
//!
//! The only implementation. It began as a port of a Python reference engine, was verified
//! byte-identical to it over 700k games compared at every step, and replaced it in Phase
//! 3.1 (PLAN.md §14.1). `docs/CONTRACT.md` is normative for the three things that are
//! frozen bit for bit: the PRNG, the draw procedure, and the byte image behind
//! `state_hash()`.
//!
//! **How it is checked without a second implementation:** the golden corpus in
//! `tests/golden/` (signed off by the Python engine before it was deleted), a per-step
//! differential against `oracle-v1` -- this crate frozen at the last Python-verified
//! commit -- in `crates/ttr-tools`, the naive in-crate references (`naive_graph`,
//! `naive_legal`), and mutation testing over the rules, scoring, graph and encoder.
//!
//! Two consequences of the contract worth stating:
//!
//! * Anything the contract freezes is transcribed, not reinvented. Fisher-Yates runs
//!   descending; bounded draws reject rather than multiply-shift; the locomotive is card
//!   type `n_colors`, never a hard-coded 8. Each of these has an equally correct
//!   alternative that produces different -- still perfectly legal -- trajectories, and
//!   every recorded replay and the corpus would stop reproducing.
//! * Anything the contract explicitly leaves free (§5: adjacency, buckets, distance
//!   tables, the union-find) may be reorganised for speed without ceremony, because none
//!   of it reaches the hash.
//!
//! No PyO3 here. The bindings live in `crates/ttr-py`, so this crate builds and tests
//! with a plain `cargo test` and nothing about it depends on a Python interpreter.

pub mod actions;
pub mod arena;
pub mod board;
pub mod board_gen;
pub mod config;
pub mod graph;
pub mod hashing;
pub mod heuristic;
pub mod numeric;
pub mod obs;
pub mod obs_spec;
pub mod replay;
pub mod resample;
pub mod rng;
pub mod scoring;
pub mod state;
pub mod vecenv;

// Obviously-correct slow paths that the fast ones are tested against. Test-only: they
// exist to disagree with the engine, never to run in it.
#[cfg(test)]
mod naive_graph;
#[cfg(test)]
mod naive_legal;
#[cfg(test)]
mod naive_obs;

/// The frozen-contract version (docs/CONTRACT.md).
///
/// Every replay, the contract vectors and the golden corpus record it, and refuse to
/// replay under a different one: a bump is a deliberate, versioned event that invalidates
/// them all, never a side effect.
pub const CONTRACT_VERSION: u8 = 1;
