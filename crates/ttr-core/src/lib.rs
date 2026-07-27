//! The Ticket to Ride engine core.
//!
//! This crate is a port, not a redesign. [`ticket_to_ride.engine`] in Python is the
//! permanent oracle, and `docs/CONTRACT.md` is normative for the three things that must
//! agree bit for bit: the PRNG, the draw procedure, and the byte image behind
//! `state_hash()`. **Where this code and the Python engine disagree, the contract decides
//! which one is wrong** -- neither implementation is automatically right.
//!
//! Two consequences worth stating, because they are the difference between a port that
//! can be trusted and one that merely passes today:
//!
//! * Anything the contract freezes is transcribed, not reinvented. Fisher-Yates runs
//!   descending; bounded draws reject rather than multiply-shift; the locomotive is card
//!   type `n_colors`, never a hard-coded 8. Each of these has an equally correct
//!   alternative that produces different -- still perfectly legal -- trajectories, which
//!   is precisely what destroys a differential-testing oracle.
//! * Anything the contract explicitly leaves free (§5: adjacency, buckets, distance
//!   tables, the union-find) may be reorganised for speed without ceremony, because none
//!   of it reaches the hash.
//!
//! No PyO3 here. The bindings live in `crates/ttr-py`, so this crate builds and tests
//! with a plain `cargo test` and nothing about it depends on a Python interpreter.
//!
//! [`ticket_to_ride.engine`]: https://github.com/edwisdom/ticket_to_ride

pub mod actions;
pub mod board;
pub mod board_gen;
pub mod config;
pub mod graph;
pub mod hashing;
pub mod numeric;
pub mod obs;
pub mod obs_spec_gen;
pub mod rng;
pub mod scoring;
pub mod state;
pub mod vecenv;

/// The frozen-contract version, kept in lockstep with
/// `ticket_to_ride.engine.contract.CONTRACT_VERSION`.
///
/// A test asserts the two agree. If they ever drift, every replay recorded under the old
/// value is unreplayable and the differential harness is comparing two different games --
/// so this is a hard failure rather than a warning.
pub const CONTRACT_VERSION: u8 = 1;
