//! PyO3 bindings for `ttr-core`.
//!
//! Deliberately thin: this crate converts types and nothing else. Every rule, every draw,
//! every hash lives in `ttr-core`, which is testable with a plain `cargo test` and has no
//! idea Python exists. Logic that leaks in here would be logic the standalone Rust tests
//! never see.
//!
//! The module is `ttr_rust`, a distribution of its own rather than a submodule of
//! `ticket_to_ride`. That separation is load-bearing: `ticket_to_ride.engine` is the
//! permanent differential-testing oracle, and an oracle that imports the implementation it
//! validates is not an oracle. `tests/unit/engine/test_import_boundary.py` enforces it.

use pyo3::prelude::*;

/// The frozen-contract version this build implements. Python asserts it matches
/// `ticket_to_ride.engine.contract.CONTRACT_VERSION`.
#[pyfunction]
fn contract_version() -> u8 {
    ttr_core::CONTRACT_VERSION
}

#[pymodule]
fn ttr_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contract_version, m)?)?;
    Ok(())
}
