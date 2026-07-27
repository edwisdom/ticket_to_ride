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

use pyo3::buffer::{Element, PyBuffer};
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use ttr_core::config::RuleConfig;
use ttr_core::resample::{assert_consistent, resample_from_infoset};
use ttr_core::rng::Pcg32;
use ttr_core::state::{Game, State};
use ttr_core::vecenv::VecEnv;

/// Borrow a caller-provided buffer as a mutable slice, so batched writes copy nothing.
///
/// Accepts anything exporting the buffer protocol -- a `bytearray`, an `array.array`, a
/// numpy array, a torch tensor's storage -- without this crate depending on any of them.
///
/// # Safety
///
/// The raw slice is sound because every precondition is checked here and the rest is the
/// buffer protocol's own guarantee:
///
/// * `PyBuffer::get` holds a `Py_buffer` view for as long as `buf` lives, which is what
///   stops the exporter from resizing or freeing the memory underneath us.
/// * Non-contiguous and read-only buffers are rejected, so the elements really are
///   `want` consecutive `T`s that may be written.
/// * Nothing inside the call re-enters Python, so no interpreter code can observe the
///   slice while it exists.
///
/// The caller's side of the contract -- not touching the buffer from another thread during
/// the call -- is the same one numpy and torch extensions have always required.
///
/// Takes `&mut PyBuffer` rather than `&PyBuffer` on purpose. Deriving a `&mut [T]` from a
/// shared reference would let two calls hand out aliasing mutable slices of the same
/// memory; requiring unique access makes that impossible to write rather than merely
/// impolite. (clippy::mut_from_ref catches exactly this.)
unsafe fn writable_slice<'a, T: Element>(
    buf: &'a mut PyBuffer<T>,
    want: usize,
    what: &str,
) -> PyResult<&'a mut [T]> {
    if buf.readonly() {
        return Err(PyBufferError::new_err(format!(
            "{what} buffer is read-only"
        )));
    }
    if !buf.is_c_contiguous() {
        return Err(PyBufferError::new_err(format!(
            "{what} buffer is not C-contiguous"
        )));
    }
    if buf.item_count() != want {
        return Err(PyValueError::new_err(format!(
            "{what} buffer holds {} items, need {want}",
            buf.item_count()
        )));
    }
    // SAFETY: see the doc comment above -- checked contiguous, writable, correctly sized,
    // and pinned by the live buffer view.
    Ok(unsafe { std::slice::from_raw_parts_mut(buf.buf_ptr().cast::<T>(), want) })
}

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

/// A PCG-XSH-RR 64/32 stream. **Frozen** -- see docs/CONTRACT.md §1.
///
/// Exposed so a Python-side driver can hold and advance a stream across many
/// determinizations rather than reseeding per call, which would make consecutive samples
/// correlated in a way that is invisible until the search is quietly worse.
// `skip_from_py_object` is not lint appeasement. PyO3 gives `Clone` pyclasses an automatic
// `FromPyObject`, which would let `Rng` be taken *by value* -- silently cloning the stream
// at the call boundary, so the caller's generator never advanced. Determinizations would
// then be identical every time, which is exactly the failure this class exists to prevent.
// Methods take `&mut PyRng`, so the conversion is never wanted.
#[pyclass(name = "Rng", module = "ttr_rust", skip_from_py_object)]
#[derive(Clone)]
pub struct PyRng {
    inner: Pcg32,
}

#[pymethods]
impl PyRng {
    #[new]
    fn new(state: u64, inc: u64) -> Self {
        Self {
            inner: Pcg32::new(state, inc),
        }
    }

    /// `pcg32_srandom_r(initstate, initseq)`.
    #[staticmethod]
    fn seeded(initstate: u64, initseq: u64) -> Self {
        Self {
            inner: Pcg32::seeded(initstate, initseq),
        }
    }

    /// A named derived stream, matching `ticket_to_ride.engine.rng.stream`.
    ///
    /// Parts are text or integers and the two encode differently, so `1` and `"1"` name
    /// different streams -- exactly as the contract vectors pin down.
    #[staticmethod]
    #[pyo3(signature = (root, *parts))]
    fn stream(root: u64, parts: Vec<StreamPart>) -> Self {
        let owned: Vec<ttr_core::rng::Part> = parts
            .iter()
            .map(|p| match p {
                StreamPart::Text(s) => ttr_core::rng::Part::Str(s.as_str()),
                StreamPart::Int(i) => ttr_core::rng::Part::Int(*i),
            })
            .collect();
        Self {
            inner: ttr_core::rng::stream(root, &owned),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn below(&mut self, bound: u32) -> PyResult<u32> {
        if bound == 0 {
            return Err(PyValueError::new_err("bound must be positive"));
        }
        Ok(self.inner.below(bound))
    }

    #[getter]
    fn state(&self) -> u64 {
        self.inner.state
    }

    #[getter]
    fn inc(&self) -> u64 {
        self.inner.inc
    }

    fn __repr__(&self) -> String {
        format!(
            "Rng(state=0x{:016x}, inc=0x{:016x})",
            self.inner.state, self.inner.inc
        )
    }
}

/// One component of a derived stream name.
#[derive(FromPyObject)]
enum StreamPart {
    #[pyo3(transparent, annotation = "int")]
    Int(u64),
    #[pyo3(transparent, annotation = "str")]
    Text(String),
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

    // -- search hooks ------------------------------------------------------

    /// Sample a state consistent with everything `observer` can see (PLAN.md §5.5).
    ///
    /// Opponents' blind-drawn cards and the undrawn deck are resampled; the observer's
    /// hand, the discard, the display, ticket holdings and every public counter are
    /// reproduced exactly. Two documented approximations -- reshuffle-boundary correlation
    /// and the ticket ring's return order -- are described in `ttr_core::resample`.
    fn resample_from_infoset(&self, observer: usize, rng: &mut PyRng) -> PyResult<Self> {
        self.check_seat(observer)?;
        Ok(Self {
            inner: resample_from_infoset(&self.inner, observer, &mut rng.inner),
        })
    }

    /// Check a determinization against the public state it claims to match.
    ///
    /// `self` is the sample, `public` the state it was drawn from. Raises on the first
    /// broken conservation law; release builds do not check automatically, so a caller
    /// running a validator sweep calls this itself.
    fn assert_consistent(&self, public: &PyState, observer: usize) -> PyResult<()> {
        assert_consistent(&self.inner, &public.inner, observer)
            .map_err(|e| PyRuntimeError::new_err(e.0))
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

/// A batch of independent games stepped together.
///
/// Every buffer-writing method takes a **writable buffer** (anything supporting the buffer
/// protocol: a `bytearray`, a numpy array, a torch tensor's storage) and fills it in place.
/// Python must never loop over environments -- that loop is the whole cost the port exists
/// to remove (PLAN.md §8.3).
#[pyclass(name = "VecEnv", module = "ttr_rust")]
pub struct PyVecEnv {
    inner: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (map_name="usa", n_players=2, n_envs=1, base_seed=0, *, track_history=false))]
    fn new(
        map_name: &str,
        n_players: usize,
        n_envs: usize,
        base_seed: u64,
        track_history: bool,
    ) -> PyResult<Self> {
        // History off by default here and on by default for a single `Game`: copying the
        // action list dominates a clone, and no batched consumer wants it.
        let cfg = RuleConfig {
            map_name: map_name.to_string(),
            n_players,
            track_history,
            ..RuleConfig::default()
        };
        let game = Game::new(cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: VecEnv::new(game, n_envs, base_seed),
        })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn obs_size(&self) -> usize {
        self.inner.obs_size()
    }

    #[getter]
    fn n_actions(&self) -> usize {
        self.inner.n_actions()
    }

    #[getter]
    fn seeds(&self) -> Vec<u64> {
        self.inner.seeds().to_vec()
    }

    fn step(&mut self, actions: Vec<u16>) -> PyResult<()> {
        self.inner
            .step(&actions)
            .map_err(|(index, e)| IllegalAction::new_err(format!("env {index}: {}", e.0)))
    }

    fn auto_reset(&mut self) -> usize {
        self.inner.auto_reset()
    }

    /// Fill `out` with every environment's observation for its acting seat.
    ///
    /// `out` is `n_envs * obs_size` float32s, environment-major, written in place.
    fn observe_current(&mut self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut buf = PyBuffer::<f32>::get(out)?;
        let want = self.inner.len() * self.inner.obs_size();
        // SAFETY: `writable_slice` checks writability, contiguity and length; `buf` keeps
        // the view alive for the whole call.
        let slice = unsafe { writable_slice(&mut buf, want, "observation")? };
        // The GIL goes back while encoding: this is the expensive half of a training step
        // and it runs across the rayon pool with no interpreter involvement.
        py.detach(|| self.inner.observe_current(slice));
        Ok(())
    }

    /// Fill `out` with every environment's observation from one fixed seat's view.
    fn observe_seat(
        &mut self,
        py: Python<'_>,
        player: usize,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if player >= self.inner.states()[0].n_players() {
            return Err(PyValueError::new_err(format!(
                "seat {player} is out of range"
            )));
        }
        let mut buf = PyBuffer::<f32>::get(out)?;
        let want = self.inner.len() * self.inner.obs_size();
        // SAFETY: as above.
        let slice = unsafe { writable_slice(&mut buf, want, "observation")? };
        py.detach(|| self.inner.observe_seat(player, slice));
        Ok(())
    }

    /// Fill `out` with every environment's legal-action mask: `n_envs * n_actions` bytes.
    fn legal_masks(&self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut buf = PyBuffer::<u8>::get(out)?;
        let want = self.inner.len() * self.inner.n_actions();
        // SAFETY: as above.
        let slice = unsafe { writable_slice(&mut buf, want, "mask")? };
        py.detach(|| self.inner.legal_masks(slice));
        Ok(())
    }

    fn current_players(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.inner.len()];
        self.inner.current_players(&mut out);
        out
    }

    fn terminal_flags(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.inner.len()];
        self.inner.terminal_flags(&mut out);
        out
    }

    fn state_hashes(&self) -> Vec<u64> {
        self.inner.states().iter().map(State::state_hash).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<ttr_rust.VecEnv {} envs, obs={}, actions={}>",
            self.inner.len(),
            self.inner.obs_size(),
            self.inner.n_actions()
        )
    }
}

/// Play `games` uniformly-random games entirely inside Rust.
///
/// One FFI call for the whole batch, on purpose: a Python-driven loop pays a few
/// microseconds per step in call overhead, which on a sub-microsecond engine is the only
/// thing it would measure. Returns `(games, steps, seconds)`.
#[pyfunction]
#[pyo3(signature = (map_name, n_players, games, base_seed=0, threads=1))]
fn random_playouts(
    py: Python<'_>,
    map_name: &str,
    n_players: usize,
    games: u64,
    base_seed: u64,
    threads: usize,
) -> PyResult<(u64, u64, f64)> {
    let cfg = RuleConfig {
        map_name: map_name.to_string(),
        n_players,
        track_history: false,
        ..RuleConfig::default()
    };
    let game = Game::new(cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Release the GIL: this is pure Rust and may run for seconds across several threads.
    py.detach(|| {
        let start = std::time::Instant::now();
        let stats = if threads <= 1 {
            ttr_core::vecenv::random_playouts(&game, games, base_seed)
        } else {
            ttr_core::vecenv::random_playouts_parallel(&game, games, base_seed, threads)
        };
        Ok((stats.games, stats.steps, start.elapsed().as_secs_f64()))
    })
}

/// The worker-thread count batched work uses. See `ttr_core::vecenv` for why this is a
/// pool *size* rather than the P-core pinning PLAN.md §8.3 asks for.
#[pyfunction]
fn performance_threads() -> usize {
    ttr_core::vecenv::performance_threads()
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
    m.add_function(wrap_pyfunction!(random_playouts, m)?)?;
    m.add_function(wrap_pyfunction!(performance_threads, m)?)?;
    m.add_class::<PyGame>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyRng>()?;
    m.add("IllegalAction", m.py().get_type::<IllegalAction>())?;
    Ok(())
}
