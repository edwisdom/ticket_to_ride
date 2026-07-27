//! PyO3 bindings for `ttr-core`.
//!
//! Deliberately thin: this crate converts types and nothing else. Every rule, every draw,
//! every hash lives in `ttr-core`, which is testable with a plain `cargo test` and has no
//! idea Python exists. Logic that leaked in here would be logic the standalone Rust tests
//! never see -- and the differential harness would then be comparing the Python engine
//! against a mixture of the two languages rather than against Rust.
//!
//! The module is `ttr_rust`, a distribution of its own rather than a submodule of
//! `ticket_to_ride`. That separation is load-bearing: `ticket_to_ride.engine` is the
//! permanent differential-testing oracle, and an oracle that imports the implementation it
//! validates is not an oracle. `tests/unit/engine/test_import_boundary.py` enforces it.

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use ttr_core::config::RuleConfig;
use ttr_core::state::{Game, State};

create_exception!(
    ttr_rust,
    IllegalAction,
    pyo3::exceptions::PyException,
    "An action that is not legal in the current state. Mirrors \
     `ticket_to_ride.engine.state.IllegalAction`."
);

/// Immutable per-configuration setup: the board, the action space, the rule constants.
#[pyclass(name = "Game", module = "ttr_rust", frozen)]
pub struct PyGame {
    inner: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (map_name="usa", n_players=2, *, track_history=true))]
    fn new(map_name: &str, n_players: usize, track_history: bool) -> PyResult<Self> {
        let cfg = RuleConfig {
            map_name: map_name.to_string(),
            n_players,
            track_history,
            ..RuleConfig::default()
        };
        cfg.validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = Game::new(cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_distinct_actions(&self) -> u16 {
        self.inner.num_distinct_actions()
    }

    #[getter]
    fn n_players(&self) -> usize {
        self.inner.cfg.n_players
    }

    #[getter]
    fn map_name(&self) -> &str {
        self.inner.board.name
    }

    #[getter]
    fn data_hash(&self) -> &str {
        self.inner.board.data_hash
    }

    #[getter]
    fn rules_hash(&self) -> String {
        self.inner.cfg.rules_hash()
    }

    fn new_initial_state(&self, seed: u64) -> PyState {
        PyState {
            inner: self.inner.new_initial_state(seed),
        }
    }

    fn action_to_string(&self, action: u16) -> PyResult<String> {
        if action >= self.inner.space.n {
            return Err(PyValueError::new_err(format!(
                "action {action} outside 0..{}",
                self.inner.space.n - 1
            )));
        }
        Ok(self.inner.action_to_string(action))
    }

    fn __repr__(&self) -> String {
        format!(
            "<ttr_rust.Game {} {}P actions={}>",
            self.inner.board.name, self.inner.cfg.n_players, self.inner.space.n
        )
    }
}

/// One position. Cloneable, hashable, and steppable.
#[pyclass(name = "State", module = "ttr_rust")]
pub struct PyState {
    inner: State,
}

#[pymethods]
impl PyState {
    fn step(&mut self, action: u16) -> PyResult<()> {
        self.inner
            .step(action)
            .map_err(|e| IllegalAction::new_err(e.0))
    }

    /// Sorted ascending, so the two engines' lists compare directly.
    fn legal_actions(&self) -> Vec<u16> {
        self.inner.legal_actions()
    }

    /// A 0/1 mask over the whole action space, as bytes.
    fn legal_action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut mask = vec![0u8; self.inner.space.n as usize];
        self.inner.legal_action_mask(&mut mask);
        PyBytes::new(py, &mask)
    }

    fn state_hash(&self) -> u64 {
        self.inner.state_hash()
    }

    fn position_hash(&self) -> u64 {
        self.inner.position_hash()
    }

    /// The canonical byte image behind the hashes -- CONTRACT.md §3.1.
    ///
    /// Exposed because comparing *bytes* localises a mismatch to a field, while comparing
    /// hashes only says the states differ. The harness reaches for this the moment a hash
    /// disagrees.
    #[pyo3(signature = (canonical=false))]
    fn serialize<'py>(&self, py: Python<'py>, canonical: bool) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.serialize(canonical))
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn current_player(&self) -> u8 {
        self.inner.current_player()
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn clone_into(&self, dst: &mut PyState) {
        self.inner.clone_into(&mut dst.inner);
    }

    /// Assert every conservation law. Raises rather than returning, matching Python's
    /// `assert`-based `validate()`.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(PyRuntimeError::new_err)
    }

    fn history(&self) -> Vec<u16> {
        self.inner.history.clone()
    }

    // -- scoring -----------------------------------------------------------
    //
    // `&mut self` because ticket completion is a union-find query and the forest uses path
    // halving. The DSU is a cache excluded from every hash, so the mutation is invisible to
    // anything that compares states -- but it is real, so the borrow has to be honest.

    fn final_scores(&mut self) -> Vec<i16> {
        ttr_core::scoring::final_scores(&mut self.inner)
    }

    /// One tuple per seat, in the field order of `ticket_to_ride.engine.scoring.Breakdown`.
    #[allow(clippy::type_complexity)]
    fn score_breakdown(&mut self) -> Vec<(i16, u8, u8, i16, i16, u16, u8, i16)> {
        ttr_core::scoring::score_breakdown(&mut self.inner)
            .into_iter()
            .map(|b| {
                (
                    b.routes,
                    b.tickets_made,
                    b.tickets_missed,
                    b.ticket_points,
                    b.longest_bonus,
                    b.longest_trail,
                    b.completed,
                    b.total,
                )
            })
            .collect()
    }

    fn returns(&mut self) -> Vec<f64> {
        ttr_core::scoring::returns(&mut self.inner)
    }

    fn winners(&mut self) -> Vec<usize> {
        ttr_core::scoring::winners(&mut self.inner)
    }

    fn longest_trails(&self) -> Vec<u16> {
        ttr_core::scoring::longest_trails(&self.inner)
    }

    // -- observation -------------------------------------------------------

    #[getter]
    fn observation_size(&self) -> usize {
        ttr_core::obs::observation_size(&self.inner)
    }

    /// Encode from `player`'s point of view and return the floats.
    ///
    /// Returns a list rather than writing into a caller buffer: this exists for the
    /// differential harness, which compares against a pure-Python oracle that has no
    /// numpy. The zero-copy path that training actually uses is `VecEnv::observe`.
    fn observation(&mut self, player: usize) -> PyResult<Vec<f32>> {
        self.check_seat(player)?;
        let mut out = vec![0.0f32; ttr_core::obs::observation_size(&self.inner)];
        ttr_core::obs::encode(&mut self.inner, player, &mut out);
        Ok(out)
    }

    fn hand_of(&self, player: usize) -> PyResult<Vec<u8>> {
        self.check_seat(player)?;
        Ok(self.inner.hand_of(player).to_vec())
    }

    fn certain_of(&self, player: usize) -> PyResult<Vec<u8>> {
        self.check_seat(player)?;
        Ok(self.inner.certain_of(player).to_vec())
    }

    fn tickets_of(&self, player: usize) -> PyResult<Vec<u8>> {
        self.check_seat(player)?;
        Ok(self.inner.tickets_of(player))
    }

    fn deck_counts(&self) -> Vec<u8> {
        self.inner.deck_counts()
    }

    fn unseen_counts(&self, observer: usize) -> PyResult<Vec<i16>> {
        self.check_seat(observer)?;
        Ok(self.inner.unseen_counts(observer))
    }

    #[getter]
    fn phase(&self) -> u8 {
        self.inner.phase
    }

    #[getter]
    fn phase_name(&self) -> &str {
        self.inner.phase_name()
    }

    #[getter]
    fn turn(&self) -> u16 {
        self.inner.turn
    }

    #[getter]
    fn cur(&self) -> u8 {
        self.inner.cur
    }

    #[getter]
    fn draws_left(&self) -> u8 {
        self.inner.draws_left
    }

    #[getter]
    fn final_left(&self) -> u8 {
        self.inner.final_left
    }

    #[getter]
    fn pass_streak(&self) -> u8 {
        self.inner.pass_streak
    }

    #[getter]
    fn trains(&self) -> Vec<u8> {
        self.inner.trains[..self.inner.n_players()].to_vec()
    }

    #[getter]
    fn score(&self) -> Vec<i16> {
        self.inner.score[..self.inner.n_players()].to_vec()
    }

    #[getter]
    fn faceup(&self) -> Vec<i8> {
        self.inner.faceup.to_vec()
    }

    #[getter]
    fn seg_owner(&self) -> Vec<u8> {
        self.inner.seg_owner[..self.inner.board.n_segments].to_vec()
    }

    #[getter]
    fn deck_pos(&self) -> u16 {
        self.inner.deck_pos
    }

    #[getter]
    fn tickets_remaining(&self) -> u8 {
        self.inner.tickets_remaining()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

impl PyState {
    fn check_seat(&self, player: usize) -> PyResult<()> {
        let n = self.inner.n_players();
        if player >= n {
            return Err(PyValueError::new_err(format!(
                "seat {player} outside 0..{n}"
            )));
        }
        Ok(())
    }
}

/// The frozen-contract version this build implements. Python asserts it matches
/// `ticket_to_ride.engine.contract.CONTRACT_VERSION`.
#[pyfunction]
fn contract_version() -> u8 {
    ttr_core::CONTRACT_VERSION
}

/// The observation-layout version. Baked into every checkpoint, so the two encoders must
/// report the same one or a checkpoint could load against a layout it was not trained on.
#[pyfunction]
fn obs_version() -> u32 {
    ttr_core::obs::obs_version()
}

#[pymodule]
fn ttr_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contract_version, m)?)?;
    m.add_function(wrap_pyfunction!(obs_version, m)?)?;
    m.add_class::<PyGame>()?;
    m.add_class::<PyState>()?;
    m.add("IllegalAction", m.py().get_type::<IllegalAction>())?;
    Ok(())
}
