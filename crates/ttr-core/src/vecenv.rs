//! Batched environments, and the rollout entry points benchmarks measure.
//!
//! **The FFI boundary is where the performance is, not the Rust itself** (PLAN.md §8.3).
//! A per-step call into Rust is call-bound at a few microseconds, which throws away most
//! of what the port bought; the fix is that Python never loops over environments. It hands
//! down one slice of actions and gets back one contiguous buffer of observations, masks or
//! flags, written in place.
//!
//! Every `*_into` method writes into a caller-provided buffer rather than allocating and
//! returning, so the training loop can point them straight at a torch-shared tensor and
//! copy nothing.
//!
//! ## Thread count, and why this is not "pin to the P-cores"
//!
//! PLAN.md §8.3 says to pin rayon to the 8 performance cores: E-core stragglers hurt
//! batch-synchronised self-play more than their throughput helps. The intent is right and
//! the mechanism is not available -- **macOS exposes no thread-affinity API**. There is no
//! equivalent of `sched_setaffinity`; the only lever is a QoS class, which is a hint the
//! scheduler may ignore. What is achievable is *sizing* the pool to the performance-core
//! count, which keeps the batch from being split across cores that finish at wildly
//! different times. That is what [`performance_threads`] does, and the difference between
//! "pinned" and "sized" is recorded here rather than left as an unmet claim.

use rayon::prelude::*;

use crate::obs;
use crate::rng::{Part, Pcg32, stream};
use crate::state::{Game, IllegalAction, State};

/// How many worker threads to use for batched work.
///
/// On Apple silicon this is the performance-core count read from `sysctl`; elsewhere it is
/// the CPU count less two, leaving room for the Python process and the allocator. Falls
/// back to 1 rather than panicking if neither can be determined -- a slow batch beats a
/// crashed worker.
pub fn performance_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl_usize("hw.perflevel0.logicalcpu") {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(1))
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
fn sysctl_usize(name: &str) -> Option<usize> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

/// A batch of independent games stepped together.
///
/// Every environment runs the same board and seat count, which is what lets the
/// observation and mask buffers be plain rectangles.
pub struct VecEnv {
    game: Game,
    states: Vec<State>,
    /// The seed each environment was last reset with. Auto-reset advances it by
    /// `n_envs`, so no two environments ever replay each other's game.
    seeds: Vec<u64>,
    next_seed: u64,
    obs_size: usize,
    n_actions: usize,
}

impl VecEnv {
    pub fn new(game: Game, n_envs: usize, base_seed: u64) -> Self {
        assert!(n_envs > 0, "a VecEnv needs at least one environment");
        let states: Vec<State> = (0..n_envs)
            .map(|i| game.new_initial_state(base_seed + i as u64))
            .collect();
        let obs_size = obs::observation_size(&states[0]);
        Self {
            n_actions: game.space.n as usize,
            obs_size,
            seeds: (0..n_envs).map(|i| base_seed + i as u64).collect(),
            next_seed: base_seed + n_envs as u64,
            states,
            game,
        }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn obs_size(&self) -> usize {
        self.obs_size
    }

    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    pub fn states(&self) -> &[State] {
        &self.states
    }

    pub fn states_mut(&mut self) -> &mut [State] {
        &mut self.states
    }

    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    /// Step every environment. Terminal environments are skipped, not errors.
    ///
    /// Returns the first illegal action encountered, with the environment index, rather
    /// than panicking: a policy leaking past its mask is a bug worth a clean report.
    pub fn step(&mut self, actions: &[u16]) -> Result<(), (usize, IllegalAction)> {
        assert_eq!(
            actions.len(),
            self.states.len(),
            "one action per environment"
        );
        // Sequential: a step is ~1 microsecond, well under rayon's per-item overhead, and
        // the error has to name an index. The parallel win is in `observe`, which is two
        // orders of magnitude more work per environment.
        for (i, (state, &action)) in self.states.iter_mut().zip(actions).enumerate() {
            if state.is_terminal() {
                continue;
            }
            state.step(action).map_err(|e| (i, e))?;
        }
        Ok(())
    }

    /// Reset only the environments that have finished, each onto a fresh unused seed.
    ///
    /// Returns how many were reset. Seeds advance monotonically rather than being reused,
    /// so a long run never silently replays the same game twice.
    pub fn auto_reset(&mut self) -> usize {
        let mut reset = 0;
        for i in 0..self.states.len() {
            if !self.states[i].is_terminal() {
                continue;
            }
            let seed = self.next_seed;
            self.next_seed += 1;
            self.seeds[i] = seed;
            self.states[i] = self.game.new_initial_state(seed);
            reset += 1;
        }
        reset
    }

    /// Write every environment's observation for the acting seat into `out`.
    ///
    /// `out` is `len() * obs_size()` floats, environment-major. Parallel because the
    /// encoder is the expensive part of a training step by a wide margin.
    pub fn observe_current(&mut self, out: &mut [f32]) {
        assert_eq!(out.len(), self.states.len() * self.obs_size);
        self.states
            .par_iter_mut()
            .zip(out.par_chunks_mut(self.obs_size))
            .for_each(|(state, slot)| {
                let player = state.current_player() as usize;
                obs::encode(state, player, slot);
            });
    }

    /// Write every environment's observation from a fixed seat's point of view.
    pub fn observe_seat(&mut self, player: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.states.len() * self.obs_size);
        self.states
            .par_iter_mut()
            .zip(out.par_chunks_mut(self.obs_size))
            .for_each(|(state, slot)| obs::encode(state, player, slot));
    }

    /// Write every environment's legal-action mask into `out`, `len() * n_actions()` bytes.
    pub fn legal_masks(&self, out: &mut [u8]) {
        assert_eq!(out.len(), self.states.len() * self.n_actions);
        self.states
            .par_iter()
            .zip(out.par_chunks_mut(self.n_actions))
            .for_each(|(state, slot)| {
                if state.is_terminal() {
                    slot.fill(0);
                } else {
                    state.legal_action_mask(slot);
                }
            });
    }

    pub fn current_players(&self, out: &mut [u8]) {
        assert_eq!(out.len(), self.states.len());
        for (slot, state) in out.iter_mut().zip(&self.states) {
            *slot = state.current_player();
        }
    }

    pub fn terminal_flags(&self, out: &mut [u8]) {
        assert_eq!(out.len(), self.states.len());
        for (slot, state) in out.iter_mut().zip(&self.states) {
            *slot = u8::from(state.is_terminal());
        }
    }
}

// ---------------------------------------------------------------------------
// Rollouts -- what the benchmarks measure
// ---------------------------------------------------------------------------

/// What a batch of random playouts cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RolloutStats {
    pub games: u64,
    pub steps: u64,
}

/// Play `games` uniformly-random games entirely inside Rust, single-threaded.
///
/// The point of running the whole rollout in one call is to measure **the engine** rather
/// than the FFI boundary. A Python-driven loop pays a few microseconds per step in call
/// overhead, which on a sub-microsecond engine is the only thing the number would measure.
/// Both figures are worth having and `ttr bench` reports them separately.
pub fn random_playouts(game: &Game, games: u64, base_seed: u64) -> RolloutStats {
    let mut steps = 0u64;
    let mut scratch = Vec::new();
    for i in 0..games {
        let seed = base_seed + i;
        let mut state = game.new_initial_state(seed);
        let mut rng = policy_stream(seed);
        while !state.is_terminal() {
            let Some(action) = state.sample_legal_into(&mut rng, &mut scratch) else {
                break;
            };
            state.step(action).expect("sampled from the legal set");
            steps += 1;
        }
    }
    RolloutStats { games, steps }
}

/// [`random_playouts`] spread over a rayon pool sized to the performance cores.
pub fn random_playouts_parallel(
    game: &Game,
    games: u64,
    base_seed: u64,
    threads: usize,
) -> RolloutStats {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .expect("building a rayon pool");
    pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|i| random_playouts(game, 1, base_seed + i))
            .reduce(RolloutStats::default, |a, b| RolloutStats {
                games: a.games + b.games,
                steps: a.steps + b.steps,
            })
    })
}

/// The policy stream a benchmark rollout draws from.
///
/// Named `"bench"` to match `ticket_to_ride.cli.cmd_bench`, so the two engines play the
/// *same* random games and the comparison is like for like rather than one engine having
/// drawn an easier set of positions.
fn policy_stream(seed: u64) -> Pcg32 {
    stream(seed, &[Part::Str("bench")])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;

    fn game(map: &str, n: usize) -> Game {
        Game::new(RuleConfig::new(map, n).unwrap()).unwrap()
    }

    #[test]
    fn a_batch_steps_every_environment() {
        let g = game("usa", 2);
        let mut env = VecEnv::new(g, 8, 100);
        let mut mask = vec![0u8; env.len() * env.n_actions()];
        env.legal_masks(&mut mask);

        // Take each environment's lowest legal action.
        let actions: Vec<u16> = (0..env.len())
            .map(|i| {
                let row = &mask[i * env.n_actions()..(i + 1) * env.n_actions()];
                row.iter().position(|&m| m == 1).expect("a legal action") as u16
            })
            .collect();
        let before: Vec<u64> = env.states().iter().map(|s| s.state_hash()).collect();
        env.step(&actions).expect("legal actions");
        let after: Vec<u64> = env.states().iter().map(|s| s.state_hash()).collect();
        assert!(
            before.iter().zip(&after).all(|(a, b)| a != b),
            "some environment did not advance"
        );
    }

    #[test]
    fn the_batch_matches_one_env_stepped_alone() {
        // The whole reason to batch is throughput, not different behaviour. Environment i
        // of a batch must be bit-identical to the same seed run on its own.
        let g = game("mini", 3);
        let mut env = VecEnv::new(g.clone(), 4, 7);
        let mut singles: Vec<State> = (0..4).map(|i| g.new_initial_state(7 + i)).collect();

        let mut rng = policy_stream(999);
        let mut scratch = Vec::new();
        for _ in 0..30 {
            let actions: Vec<u16> = env
                .states()
                .iter()
                .map(|s| {
                    if s.is_terminal() {
                        0
                    } else {
                        s.sample_legal_into(&mut rng, &mut scratch).unwrap()
                    }
                })
                .collect();
            env.step(&actions).unwrap();
            for (single, &action) in singles.iter_mut().zip(&actions) {
                if !single.is_terminal() {
                    single.step(action).unwrap();
                }
            }
            for (i, (a, b)) in env.states().iter().zip(&singles).enumerate() {
                assert_eq!(a.state_hash(), b.state_hash(), "env {i} diverged from solo");
            }
        }
    }

    #[test]
    fn observations_are_written_in_place_and_match_the_single_encoder() {
        let g = game("usa", 2);
        let mut env = VecEnv::new(g, 5, 42);
        let mut buffer = vec![0.0f32; env.len() * env.obs_size()];
        env.observe_current(&mut buffer);

        for i in 0..env.len() {
            let mut want = vec![0.0f32; env.obs_size()];
            let player = env.states()[i].current_player() as usize;
            obs::encode(&mut env.states_mut()[i], player, &mut want);
            let got = &buffer[i * env.obs_size()..(i + 1) * env.obs_size()];
            assert_eq!(got, &want[..], "env {i}");
        }
    }

    #[test]
    fn a_terminal_environment_has_an_empty_mask_and_is_not_stepped() {
        let g = game("mini", 2);
        let mut env = VecEnv::new(g, 2, 3);
        // Drive env 0 to the end.
        let mut rng = policy_stream(3);
        let mut scratch = Vec::new();
        while !env.states()[0].is_terminal() {
            let a = env.states()[0]
                .sample_legal_into(&mut rng, &mut scratch)
                .unwrap();
            let b = if env.states()[1].is_terminal() {
                0
            } else {
                env.states()[1]
                    .sample_legal_into(&mut rng, &mut scratch)
                    .unwrap()
            };
            env.step(&[a, b]).unwrap();
        }
        let mut mask = vec![1u8; env.len() * env.n_actions()];
        env.legal_masks(&mut mask);
        assert!(mask[..env.n_actions()].iter().all(|&m| m == 0));

        // Stepping a finished environment is a no-op, not an error. Env 1 is still live,
        // so it needs a genuinely legal action -- handing every environment a placeholder
        // would be testing that the *batch* tolerates an illegal action, which it must not.
        let hash = env.states()[0].state_hash();
        let live = if env.states()[1].is_terminal() {
            0
        } else {
            env.states()[1]
                .sample_legal_into(&mut rng, &mut scratch)
                .unwrap()
        };
        env.step(&[0, live]).unwrap();
        assert_eq!(
            env.states()[0].state_hash(),
            hash,
            "a finished environment was stepped"
        );
    }

    #[test]
    fn auto_reset_gives_every_environment_a_fresh_unused_seed() {
        let g = game("mini", 2);
        let mut env = VecEnv::new(g, 3, 0);
        let original: Vec<u64> = env.seeds().to_vec();
        let mut rng = policy_stream(11);
        let mut scratch = Vec::new();
        while !env.states()[0].is_terminal() {
            let actions: Vec<u16> = env
                .states()
                .iter()
                .map(|s| {
                    if s.is_terminal() {
                        0
                    } else {
                        s.sample_legal_into(&mut rng, &mut scratch).unwrap()
                    }
                })
                .collect();
            env.step(&actions).unwrap();
        }
        assert!(env.auto_reset() >= 1);
        for (i, seed) in env.seeds().iter().enumerate() {
            if env.states()[i].turn == 0 && *seed != original[i] {
                assert!(!original.contains(seed), "a reset reused seed {seed}");
            }
        }
    }

    #[test]
    fn rollouts_are_deterministic_and_seed_addressable() {
        let g = game("usa", 2);
        let a = random_playouts(&g, 3, 0);
        let b = random_playouts(&g, 3, 0);
        assert_eq!(a, b);
        assert!(a.games == 3 && a.steps > 0);
        // Parallel and serial must agree on totals: the seeds are the same games.
        let p = random_playouts_parallel(&g, 3, 0, 2);
        assert_eq!(a, p, "parallel rollouts played different games");
    }
}
