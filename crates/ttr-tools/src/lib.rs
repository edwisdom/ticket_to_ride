//! Generators and checkers for the project's golden artifacts.
//!
//! Never shipped: nothing in `ttr-core`, the PyO3 shim or the Python package depends on
//! this crate. It exists so the engine crate stays an engine, and so the code that decides
//! whether the engine still plays the recorded games sits apart from the engine itself.

pub mod corpus;
pub mod diff;
pub mod engine;
