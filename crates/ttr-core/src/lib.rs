//! The Ticket to Ride engine core.
//!
//! `docs/CONTRACT.md` is normative for the three things that are frozen bit for bit: the
//! PRNG, the draw procedure, and the byte image behind `state_hash()`. Anything it freezes is
//! transcribed, not reinvented -- Fisher-Yates runs descending, bounded draws reject rather
//! than multiply-shift -- because an equally correct alternative plays different games and
//! the golden corpus would stop reproducing. Anything it leaves free (adjacency, distance
//! tables, the union-find) may be reorganised for speed, because none of it reaches the hash.
//!
//! **How it is checked:** the golden corpus (`crates/ttr-tools`, games first recorded and
//! checked by the retired Python engine), the naive in-crate references below, explicit
//! tests, and mutation testing over the rules, scoring, graph and encoder (`.cargo/`).
//!
//! No PyO3 here: the bindings live in `crates/ttr-py`, so this crate builds and tests with a
//! plain `cargo test`.

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

/// The frozen-contract version (docs/CONTRACT.md). The golden corpus records it; a bump is
/// a deliberate, versioned event that re-signs the corpus, never a side effect.
pub const CONTRACT_VERSION: u8 = 1;
