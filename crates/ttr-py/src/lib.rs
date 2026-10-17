//! PyO3 bindings for `ttr-core`.
//!
//! Deliberately thin: this crate converts types and nothing else. Every rule, every draw,
//! every hash lives in `ttr-core`, which is testable with a plain `cargo test` and has no
//! idea Python exists. Logic that leaked in here would be logic the Rust suite and the golden
//! corpus never see.
//!
//! The module is `ttr_rust`, a uv workspace member that `ticket_to_ride` depends on and
//! re-exports as `ticket_to_ride.engine`. `uv sync` builds it in the release profile.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use ttr_core::arena::{self, AgentSpec};
use ttr_core::config::RuleConfig;
use ttr_core::heuristic::{HeuristicParams, Tier};
use ttr_core::resample::{assert_consistent, resample_from_infoset};
use ttr_core::rng::Pcg32;
use ttr_core::state::{Game, State};
use ttr_core::vecenv::Seat;
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
    /// Returns a list rather than writing into a caller buffer: a convenience for tests and
    /// interactive use. The zero-copy path that training uses is `VecEnv::observe_*`.
    fn observation(&self, player: usize) -> PyResult<Vec<f32>> {
        self.check_seat(player)?;
        let mut out = vec![0.0f32; ttr_core::obs::observation_size(&self.inner)];
        ttr_core::obs::encode(&self.inner, player, &mut out);
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

    #[getter]
    fn n_players(&self) -> usize {
        self.inner.states()[0].n_players()
    }

    #[getter]
    fn n_tickets(&self) -> usize {
        self.inner.states()[0].board.n_tickets
    }

    /// Give environment `env`'s seats to `"learner"` or a heuristic (`"h0"`..`"h4"`).
    /// Scripted seats then move inside Rust, so every env is either finished or waiting on
    /// a learner seat after each call.
    fn assign(&mut self, env: usize, seats: Vec<String>) -> PyResult<()> {
        if env >= self.inner.len() {
            return Err(PyValueError::new_err(format!("env {env} is out of range")));
        }
        if seats.len() != self.n_players() {
            return Err(PyValueError::new_err(format!(
                "{} seats given for a {}-player game",
                seats.len(),
                self.n_players()
            )));
        }
        let parsed = seats
            .iter()
            .map(|name| match name.as_str() {
                "learner" => Ok(Seat::Learner),
                other => Tier::parse(other).map(Seat::Scripted).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown seat {other:?}: learner or h0-h4"))
                }),
            })
            .collect::<PyResult<Vec<Seat>>>()?;
        self.inner.assign(env, &parsed);
        Ok(())
    }

    /// Fill `out` (`n_envs * n_players` float32s) with finished games' returns, zero for
    /// the rest. Read before `auto_reset`.
    fn returns(&mut self, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut buf = PyBuffer::<f32>::get(out)?;
        let want = self.inner.len() * self.n_players();
        // SAFETY: as above.
        let slice = unsafe { writable_slice(&mut buf, want, "returns")? };
        self.inner.returns(slice);
        Ok(())
    }

    /// Finished games' final scores, zero for the rest: `n_envs * n_players`.
    fn final_scores(&mut self) -> Vec<i16> {
        let mut out = vec![0i16; self.inner.len() * self.n_players()];
        self.inner.final_scores(&mut out);
        out
    }

    /// Fill `out` (`n_envs * n_players * n_tickets` bytes) with every seat's tickets:
    /// 0 not held, 1 open, 2 complete. The auxiliary targets.
    fn ticket_status(&mut self, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut buf = PyBuffer::<u8>::get(out)?;
        let want = self.inner.len() * self.n_players() * self.n_tickets();
        // SAFETY: as above.
        let slice = unsafe { writable_slice(&mut buf, want, "ticket status")? };
        self.inner.ticket_status(slice);
        Ok(())
    }

    /// `(env, seed, actions)` for every finished game: enough to replay it (§9). Actions
    /// are recorded only with `track_history=True`; otherwise the list is empty.
    fn finished_games(&self) -> Vec<(usize, u64, Vec<u16>)> {
        self.inner
            .states()
            .iter()
            .zip(self.inner.seeds())
            .enumerate()
            .filter(|(_, (state, _))| state.is_terminal())
            .map(|(i, (state, &seed))| (i, seed, state.history.clone()))
            .collect()
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

// ---------------------------------------------------------------------------
// Heuristics and the arena
// ---------------------------------------------------------------------------

/// Build a parameter set from a dict of overrides on the anchor's defaults.
///
/// Unknown keys are an **error**, not ignored. A silently dropped override is a tuning run
/// that reports a change it never applied, and the params hash would agree with the
/// baseline while the experimenter believes otherwise.
fn params_from(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<HeuristicParams> {
    let mut p = HeuristicParams::default();
    let Some(d) = overrides else {
        return Ok(p);
    };
    for (key, value) in d.iter() {
        let name: String = key.extract()?;
        match name.as_str() {
            "ticket_draw_train_fraction" => p.ticket_draw_train_fraction = value.extract()?,
            "h1_max_tickets" => p.h1_max_tickets = value.extract()?,
            "h1_take_faceup_locomotive" => p.h1_take_faceup_locomotive = value.extract()?,
            "plan_utilization_target" => p.plan_utilization_target = value.extract()?,
            "completion_sharpness" => p.completion_sharpness = value.extract()?,
            "min_points_per_train" => p.min_points_per_train = value.extract()?,
            "contest_weight" => p.contest_weight = value.extract()?,
            "double_lockout_weight" => p.double_lockout_weight = value.extract()?,
            "cards_per_draw_turn" => p.cards_per_draw_turn = value.extract()?,
            "faceup_needed_bonus" => p.faceup_needed_bonus = value.extract()?,
            "locomotive_value" => p.locomotive_value = value.extract()?,
            "surplus_card_value" => p.surplus_card_value = value.extract()?,
            "endgame_train_fraction" => p.endgame_train_fraction = value.extract()?,
            "endgame_ticket_urgency" => p.endgame_ticket_urgency = value.extract()?,
            "convex_path_exponent" => p.convex_path_exponent = value.extract()?,
            "gray_cannibalization_penalty" => p.gray_cannibalization_penalty = value.extract()?,
            "longest_chain_weight" => p.longest_chain_weight = value.extract()?,
            "ticket_keep_dead_penalty" => p.ticket_keep_dead_penalty = value.extract()?,
            "threat_weight" => p.threat_weight = value.extract()?,
            "threat_horizon_turns" => p.threat_horizon_turns = value.extract()?,
            "threat_severance_exponent" => p.threat_severance_exponent = value.extract()?,
            "threat_crowding_penalty" => p.threat_crowding_penalty = value.extract()?,
            "hoard_safety_margin" => p.hoard_safety_margin = value.extract()?,
            "double_grab_urgency" => p.double_grab_urgency = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown heuristic parameter {other:?}"
                )));
            }
        }
    }
    p.validate().map_err(PyValueError::new_err)?;
    Ok(p)
}

/// The anchor's defaults, as a dict. The one place Python reads the constants from, so a
/// tuning script cannot drift from the values the agents actually run.
#[pyfunction]
fn default_params(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let p = HeuristicParams::default();
    let d = PyDict::new(py);
    for entry in p.canonical().split('|').skip(1) {
        let (name, value) = entry.split_once('=').expect("canonical is name=value");
        // Round-trip through the canonical text rather than listing the fields again here:
        // a field added to the struct then appears automatically instead of being silently
        // missing from the dict Python tunes with.
        if let Ok(b) = value.parse::<bool>() {
            d.set_item(name, b)?;
        } else if let Ok(i) = value.parse::<i64>() {
            d.set_item(name, i)?;
        } else {
            d.set_item(
                name,
                value
                    .parse::<f64>()
                    .map_err(|e| PyRuntimeError::new_err(format!("canonical field {name}: {e}")))?,
            )?;
        }
    }
    Ok(d.unbind())
}

/// blake2b-128 over a parameter set. **Provenance, not identity** -- see `behaviour_hash`.
#[pyfunction]
#[pyo3(signature = (overrides=None))]
fn params_hash(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
    Ok(params_from(overrides)?.params_hash())
}

/// A content address for an agent: what it actually plays, over a frozen probe set.
///
/// **This is what pins the Elo anchor.** A parameter hash catches an edited constant and
/// misses a changed tiebreak, a changed Steiner fallback, a fixed bug -- each of which moves
/// H3's play and silently re-bases every rating recorded against it. Two builds that play
/// identically are the same player; two that play differently are different players even at
/// the same commit, which is also why `git_sha` is provenance on a match rather than part of
/// an agent's identity.
#[pyfunction]
#[pyo3(signature = (tier, overrides=None))]
fn behaviour_hash(
    py: Python<'_>,
    tier: &str,
    overrides: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let t =
        Tier::parse(tier).ok_or_else(|| PyValueError::new_err(format!("unknown tier {tier:?}")))?;
    let p = params_from(overrides)?;
    Ok(py.detach(|| ttr_core::heuristic::behaviour_hash(t, &p)))
}

/// Per-seat outcomes for one arena run, columnar. One row per (game, seat).
#[pyclass(name = "ArenaSeats", module = "ttr_rust", frozen, get_all)]
pub struct PyArenaSeats {
    pub game: Vec<u32>,
    pub seat: Vec<u8>,
    pub agent: Vec<u16>,
    pub score: Vec<i16>,
    pub ret: Vec<f64>,
    pub rank: Vec<u8>,
    pub won: Vec<u8>,
    pub tickets_kept: Vec<u8>,
    pub tickets_made: Vec<u8>,
    pub ticket_points: Vec<i16>,
    pub routes_claimed: Vec<u16>,
    pub trains_left: Vec<u8>,
    pub cards_left: Vec<u16>,
    pub longest_trail: Vec<u16>,
    pub longest_bonus: Vec<i16>,
    pub n_claim: Vec<u32>,
    pub n_claim_wild: Vec<u32>,
    pub n_claim_double: Vec<u32>,
    pub n_draw_faceup: Vec<u32>,
    pub n_draw_blind: Vec<u32>,
    pub n_draw_tickets: Vec<u32>,
    pub n_keep: Vec<u32>,
    pub n_keep_extra: Vec<u32>,
    pub n_pass: Vec<u32>,
}

/// Per-game outcomes for one arena run, columnar. One row per game.
#[pyclass(name = "ArenaGames", module = "ttr_rust", frozen, get_all)]
pub struct PyArenaGames {
    pub block_seed: Vec<u64>,
    pub rotation: Vec<u8>,
    pub turns: Vec<u16>,
    /// Decisions across every seat. The denominator for microseconds-per-decision, which
    /// is what sets the sim budget of every Phase 5 search.
    pub decisions: Vec<u32>,
    pub final_hash: Vec<u64>,
    pub seconds: f64,
}

/// Play every rotation of `blocks` seed blocks and return the two tables.
///
/// One call for the whole schedule. A Python loop calling into Rust per decision would pay
/// a few microseconds of call overhead against a ~25 microsecond H3 decision and would not
/// parallelise at all under the GIL; this releases it and spreads blocks across the pool.
///
/// `agents` is a list of `(tier, overrides_or_None, seed)`; `lineup` names which agent fills
/// each seat before rotation.
#[pyfunction]
#[pyo3(signature = (map_name, n_players, agents, lineup, seed_root=0, blocks=100, threads=0))]
#[allow(clippy::too_many_arguments)]
fn run_arena(
    py: Python<'_>,
    map_name: &str,
    n_players: usize,
    agents: Vec<(String, Option<Py<PyDict>>, u64)>,
    lineup: Vec<u16>,
    seed_root: u64,
    blocks: u64,
    threads: usize,
) -> PyResult<(PyArenaGames, PyArenaSeats)> {
    let cfg = RuleConfig {
        map_name: map_name.to_string(),
        n_players,
        track_history: false,
        ..RuleConfig::default()
    };
    let game = Game::new(cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut specs = Vec::with_capacity(agents.len());
    for (tier, overrides, seed) in &agents {
        let t = Tier::parse(tier)
            .ok_or_else(|| PyValueError::new_err(format!("unknown tier {tier:?}")))?;
        let bound = overrides.as_ref().map(|d| d.bind(py).clone());
        specs.push(AgentSpec {
            tier: t,
            params: params_from(bound.as_ref())?,
            seed: *seed,
        });
    }
    if lineup.len() != n_players {
        return Err(PyValueError::new_err(format!(
            "lineup has {} entries for {n_players} seats",
            lineup.len()
        )));
    }
    if let Some(bad) = lineup.iter().find(|&&a| a as usize >= specs.len()) {
        return Err(PyValueError::new_err(format!(
            "lineup names agent {bad}, but only {} were given",
            specs.len()
        )));
    }
    let threads = if threads == 0 {
        ttr_core::vecenv::performance_threads()
    } else {
        threads
    };

    let (out, seconds) = py.detach(|| {
        let start = std::time::Instant::now();
        let out = arena::run(&game, &specs, &lineup, seed_root, blocks, threads);
        (out, start.elapsed().as_secs_f64())
    });

    let rows = out.len() * n_players;
    let mut games = PyArenaGames {
        block_seed: Vec::with_capacity(out.len()),
        rotation: Vec::with_capacity(out.len()),
        turns: Vec::with_capacity(out.len()),
        decisions: Vec::with_capacity(out.len()),
        final_hash: Vec::with_capacity(out.len()),
        seconds,
    };
    macro_rules! seats_vecs {
        ($($f:ident),* $(,)?) => { PyArenaSeats { $($f: Vec::with_capacity(rows)),* } };
    }
    let mut seats = seats_vecs!(
        game,
        seat,
        agent,
        score,
        ret,
        rank,
        won,
        tickets_kept,
        tickets_made,
        ticket_points,
        routes_claimed,
        trains_left,
        cards_left,
        longest_trail,
        longest_bonus,
        n_claim,
        n_claim_wild,
        n_claim_double,
        n_draw_faceup,
        n_draw_blind,
        n_draw_tickets,
        n_keep,
        n_keep_extra,
        n_pass,
    );

    for (g, outcome) in out.iter().enumerate() {
        games.block_seed.push(outcome.block_seed);
        games.rotation.push(outcome.rotation);
        games.turns.push(outcome.turns);
        games.decisions.push(outcome.decisions);
        games.final_hash.push(outcome.final_hash);
        for (s, row) in outcome.seats.iter().enumerate() {
            seats.game.push(g as u32);
            seats.seat.push(s as u8);
            seats.agent.push(row.agent);
            seats.score.push(row.score);
            seats.ret.push(row.ret);
            seats.rank.push(row.rank);
            seats.won.push(u8::from(row.won));
            seats.tickets_kept.push(row.tickets_kept);
            seats.tickets_made.push(row.tickets_made);
            seats.ticket_points.push(row.ticket_points);
            seats.routes_claimed.push(row.routes_claimed);
            seats.trains_left.push(row.trains_left);
            seats.cards_left.push(row.cards_left);
            seats.longest_trail.push(row.longest_trail);
            seats.longest_bonus.push(row.longest_bonus);
            let c = row.coverage;
            seats.n_claim.push(c.claim);
            seats.n_claim_wild.push(c.claim_wild);
            seats.n_claim_double.push(c.claim_double);
            seats.n_draw_faceup.push(c.draw_faceup);
            seats.n_draw_blind.push(c.draw_blind);
            seats.n_draw_tickets.push(c.draw_tickets);
            seats.n_keep.push(c.keep);
            seats.n_keep_extra.push(c.keep_extra);
            seats.n_pass.push(c.pass);
        }
    }
    Ok((games, seats))
}

/// The frozen-contract version this build implements (docs/CONTRACT.md).
#[pyfunction]
fn contract_version() -> u8 {
    ttr_core::CONTRACT_VERSION
}

/// The observation-layout version. Baked into every checkpoint, so a checkpoint can never
/// load against a layout it was not trained on.
#[pyfunction]
fn obs_version() -> u32 {
    ttr_core::obs::obs_version()
}

/// One observation block: name, entity count, floats per entity, absolute offset, fields.
type BlockRow = (&'static str, usize, usize, usize, Vec<FieldRow>);
/// One field: name, offset within an entity, width, kind (`"flag"` or `"ratio"`), doc.
type FieldRow = (&'static str, usize, usize, &'static str, &'static str);

/// The observation layout for a board, resolved from the feature table in
/// `ttr_core::obs_spec` -- the one place it is declared. Python reads it here rather than
/// keeping a copy: a network that splits the vector into entity blocks needs the offsets,
/// and two copies of a layout is how they drift.
#[pyfunction]
#[pyo3(signature = (map_name="usa"))]
fn obs_spec(map_name: &str) -> PyResult<Vec<BlockRow>> {
    if ttr_core::board::get_board(map_name).is_none() {
        return Err(PyValueError::new_err(format!("unknown map {map_name:?}")));
    }
    let spec = ttr_core::obs::spec_for(map_name);
    Ok(spec
        .blocks
        .iter()
        .map(|b| {
            let fields = b
                .fields
                .iter()
                .map(|f| {
                    let kind = match f.kind {
                        ttr_core::obs_spec::Kind::Flag => "flag",
                        ttr_core::obs_spec::Kind::Ratio => "ratio",
                    };
                    (f.name, f.offset, f.width, kind, f.doc)
                })
                .collect();
            (b.name, b.count, b.stride, b.offset, fields)
        })
        .collect())
}

/// Every board's name, in data order.
#[pyfunction]
fn board_names() -> Vec<&'static str> {
    ttr_core::board::all_boards()
        .iter()
        .map(|b| b.name)
        .collect()
}

/// A board's data and the invariants worth asserting about it, as plain values -- what
/// `ttr map` prints.
#[pyfunction]
#[pyo3(signature = (map_name="usa"))]
fn board_summary(py: Python<'_>, map_name: &str) -> PyResult<Py<PyDict>> {
    use ttr_core::board::{GRAY, NO_SIBLING, UNREACHABLE};
    let board = ttr_core::board::get_board(map_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown map {map_name:?}")))?;
    let degrees: Vec<usize> = board
        .adjacency
        .iter()
        .map(|row| {
            let mut neighbours: Vec<u8> = row.iter().map(|&(city, _)| city).collect();
            neighbours.sort_unstable();
            neighbours.dedup();
            neighbours.len()
        })
        .collect();
    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    let mut hubs: Vec<&str> = (0..board.n_cities)
        .filter(|&c| degrees[c] == max_degree)
        .map(|c| board.raw.cities[c])
        .collect();
    hubs.sort_unstable();
    let diameter = board.dist.iter().flatten().copied().max().unwrap_or(0);
    let per_color = PyDict::new(py);
    for c in 0..board.n_colors {
        let n = board
            .seg_color
            .iter()
            .filter(|&&x| usize::from(x) == c)
            .count();
        per_color.set_item(board.raw.color_names[c], n)?;
    }
    let mean_degree = (200.0 * board.n_pairs as f64 / board.n_cities as f64).round() / 100.0;

    let d = PyDict::new(py);
    d.set_item("name", board.name)?;
    d.set_item("data_hash", board.data_hash)?;
    d.set_item("cities", board.n_cities)?;
    d.set_item("pairs", board.n_pairs)?;
    d.set_item("segments", board.n_segments)?;
    d.set_item("spaces", board.total_spaces)?;
    d.set_item(
        "double_routes",
        board.sibling.iter().filter(|&&s| s != NO_SIBLING).count() / 2,
    )?;
    d.set_item("tickets", board.n_tickets)?;
    d.set_item("colors", board.n_colors)?;
    d.set_item("cards", board.deck_size)?;
    d.set_item("locomotives", board.raw.locomotives)?;
    d.set_item("trains_per_player", board.raw.trains_per_player)?;
    // As usize: PyO3 hands any `[u8; N]` back as `bytes`.
    let players = [board.raw.min_players, board.raw.max_players].map(usize::from);
    d.set_item("players", players)?;
    d.set_item("max_route_length", board.max_len)?;
    d.set_item("claim_buckets", board.buckets.len())?;
    d.set_item("action_space", ttr_core::actions::ActionSpace::new(board).n)?;
    d.set_item(
        "gray_segments",
        board.seg_color.iter().filter(|&&c| c == GRAY).count(),
    )?;
    d.set_item("segments_per_color", per_color)?;
    d.set_item("max_degree", max_degree)?;
    d.set_item("mean_degree", mean_degree)?;
    d.set_item("hubs", hubs)?;
    d.set_item("diameter", diameter)?;
    d.set_item("connected", diameter < UNREACHABLE)?;
    Ok(d.unbind())
}

#[pymodule]
fn ttr_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contract_version, m)?)?;
    m.add_function(wrap_pyfunction!(obs_version, m)?)?;
    m.add_function(wrap_pyfunction!(obs_spec, m)?)?;
    m.add_function(wrap_pyfunction!(random_playouts, m)?)?;
    m.add_function(wrap_pyfunction!(performance_threads, m)?)?;
    m.add_function(wrap_pyfunction!(default_params, m)?)?;
    m.add_function(wrap_pyfunction!(params_hash, m)?)?;
    m.add_function(wrap_pyfunction!(behaviour_hash, m)?)?;
    m.add_function(wrap_pyfunction!(run_arena, m)?)?;
    m.add_function(wrap_pyfunction!(board_names, m)?)?;
    m.add_function(wrap_pyfunction!(board_summary, m)?)?;
    m.add_class::<PyArenaGames>()?;
    m.add_class::<PyArenaSeats>()?;
    m.add_class::<PyGame>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyRng>()?;
    m.add("IllegalAction", m.py().get_type::<IllegalAction>())?;
    m.add("CONTRACT_VERSION", ttr_core::CONTRACT_VERSION)?;
    Ok(())
}
