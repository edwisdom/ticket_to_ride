//! The golden corpus: recorded games every build must still play the same way.
//!
//! One file per `(map, seat count)`, `tests/golden/corpus/<map>-<n>p.bin`. The games were
//! first recorded by the retired Python engine and checked by it, so the files carry that
//! verification forward: a mismatch means the engine now plays a different game, and the
//! only fixes are to repair the regression or to bump a version and re-sign (`gen-corpus`).
//!
//! Three sections:
//!
//! * **Random drive**, seeds `0..RANDOM_SEEDS`. Each step draws from the *sorted* legal list
//!   with `stream(seed, "corpus", "policy")`, so the drive needs no stored actions. Each game
//!   keeps its step count, final scores, winners and a **hash chain** (below).
//! * **Observation digests** for the first `OBS_SEEDS` of those games: every seat's
//!   observation at every `SAMPLE_EVERY`-th position and at the end.
//! * **Heuristic games**: H3 self-play with the actions stored, a chain and an observation
//!   digest each -- positions a competent player reaches, which random play rarely does.
//!   Storing the actions means a re-tuned H3 cannot break the check.
//!
//! **The chain** is blake2b-64 over, for every position from setup to the end: `state_hash`
//! and `position_hash` (u64 each), the legal-action count (u16) and the sorted legal actions
//! (u16 each); then, once, the settlement: per seat the eight `Breakdown` fields at their
//! natural widths and the return's IEEE-754 bits, then the winners mask. All little-endian.

use std::ops::Range;
use std::path::PathBuf;

use blake2b_simd::{Params, State as Hasher};
use rayon::prelude::*;
use ttr_core::board::all_boards;
use ttr_core::config::RuleConfig;
use ttr_core::heuristic::{Heuristic, HeuristicParams, Tier};
use ttr_core::obs::{encode, observation_size};
use ttr_core::rng::{Part, stream};
use ttr_core::scoring;
use ttr_core::state::{Game, State};

pub const MAGIC: &[u8; 4] = b"TTRG";
/// 2: one observation digest per game (1 split each into discrete and continuous halves).
pub const FORMAT_VERSION: u8 = 2;
pub const RANDOM_SEEDS: u32 = 10_000;
pub const OBS_SEEDS: u32 = 500;
pub const HEURISTIC_SEEDS: u32 = 200;
/// Observations are digested at every `SAMPLE_EVERY`-th position, and at the end.
pub const SAMPLE_EVERY: u8 = 7;

/// A game that has not ended by now is a hang, not a long game.
const MAX_STEPS: usize = 20_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub contract_version: u8,
    pub obs_version: u32,
    pub map: String,
    pub n_players: u8,
    pub data_hash: [u8; 16],
    pub rules_hash: [u8; 16],
    pub sample_every: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomGame {
    pub steps: u16,
    pub chain: u64,
    pub scores: Vec<i16>,
    pub winners: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedGame {
    pub actions: Vec<u16>,
    pub chain: u64,
    pub scores: Vec<i16>,
    pub winners: u8,
    pub obs: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corpus {
    pub header: Header,
    /// Seed `i` is entry `i`.
    pub random: Vec<RandomGame>,
    /// Observation digests for `random[..random_obs.len()]`.
    pub random_obs: Vec<u64>,
    /// H3 self-play; seed `i` is entry `i`.
    pub heuristic: Vec<RecordedGame>,
}

/// Every `(map, seat count)` the engine supports, in file order.
pub fn configs() -> Vec<(&'static str, usize)> {
    all_boards()
        .iter()
        .flat_map(|b| {
            (usize::from(b.raw.min_players)..=usize::from(b.raw.max_players)).map(|n| (b.name, n))
        })
        .collect()
}

pub fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/corpus")
}

pub fn corpus_path(map: &str, n_players: usize) -> PathBuf {
    corpus_dir().join(format!("{map}-{n_players}p.bin"))
}

pub fn game(map: &str, n_players: usize) -> Game {
    let cfg = RuleConfig::new(map, n_players).unwrap_or_else(|e| panic!("{map} {n_players}P: {e}"));
    Game::new(cfg).unwrap_or_else(|e| panic!("{map} {n_players}P: {e}"))
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// blake2b with an 8-byte digest, as Python's `hashlib.blake2b(digest_size=8)` wrote it.
fn hasher() -> Hasher {
    Params::new().hash_length(8).to_state()
}

fn finish(h: &Hasher) -> u64 {
    let digest = h.finalize();
    u64::from_le_bytes(digest.as_bytes().try_into().expect("an 8-byte digest"))
}

fn fold_position(state: &State, legal: &[u16], chain: &mut Hasher) {
    chain.update(&state.state_hash().to_le_bytes());
    chain.update(&state.position_hash().to_le_bytes());
    chain.update(&(legal.len() as u16).to_le_bytes());
    for action in legal {
        chain.update(&action.to_le_bytes());
    }
}

/// How a finished game ended: final scores, and the seats tied for first as a bitmask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ending {
    pub scores: Vec<i16>,
    pub winners: u8,
}

/// Fold the settlement into the chain, and return how the game ended.
fn fold_terminal(state: &mut State, chain: &mut Hasher) -> Ending {
    let breakdown = scoring::score_breakdown(state);
    for (b, ret) in breakdown.iter().zip(scoring::returns(state)) {
        chain.update(&b.routes.to_le_bytes());
        chain.update(&[b.tickets_made, b.tickets_missed]);
        chain.update(&b.ticket_points.to_le_bytes());
        chain.update(&b.longest_bonus.to_le_bytes());
        chain.update(&b.longest_trail.to_le_bytes());
        chain.update(&[b.completed]);
        chain.update(&b.total.to_le_bytes());
        chain.update(&ret.to_bits().to_le_bytes());
    }
    let winners = scoring::winners(state)
        .iter()
        .fold(0u8, |mask, &p| mask | 1 << p);
    chain.update(&[winners]);
    Ending {
        scores: breakdown.iter().map(|b| b.total).collect(),
        winners,
    }
}

/// Every seat's observation, seat 0 first, as little-endian f32s.
fn fold_observations(state: &State, buffer: &mut [f32], digest: &mut Hasher) {
    for player in 0..state.n_players() {
        encode(state, player, buffer);
        for value in buffer.iter() {
            digest.update(&value.to_le_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// Drives
// ---------------------------------------------------------------------------

/// One game, folded.
#[derive(Clone, Debug)]
pub struct Trace {
    pub actions: Vec<u16>,
    pub chain: u64,
    pub ending: Ending,
    pub obs: Option<u64>,
}

/// Play one game from `seed`, folding every position into the chain, and every sampled
/// position's observations into a digest if `observe`.
///
/// `choose` is handed the state, the sorted legal list and the step index, and must return
/// a legal action -- an illegal one is reported, never stepped.
pub fn trace(
    game: &Game,
    seed: u64,
    observe: bool,
    mut choose: impl FnMut(&mut State, &[u16], usize) -> Result<u16, String>,
) -> Result<Trace, String> {
    let mut state = game.new_initial_state(seed);
    let mut chain = hasher();
    let mut digest = observe.then(hasher);
    let mut buffer = vec![0.0f32; observation_size(&state)];
    let mut actions = Vec::new();
    let mut step = 0usize;
    loop {
        let legal = state.legal_actions();
        fold_position(&state, &legal, &mut chain);
        let done = state.is_terminal();
        let sampled = step % usize::from(SAMPLE_EVERY) == 0 || done;
        if let (Some(d), true) = (digest.as_mut(), sampled) {
            fold_observations(&state, &mut buffer, d);
        }
        if done {
            break;
        }
        let action = choose(&mut state, &legal, step)?;
        if legal.binary_search(&action).is_err() {
            return Err(format!("step {step}: action {action} is not legal here"));
        }
        state
            .step(action)
            .map_err(|e| format!("step {step}: action {action} refused: {e}"))?;
        actions.push(action);
        step += 1;
        if step > MAX_STEPS {
            return Err(format!("no terminal after {MAX_STEPS} steps"));
        }
    }
    let ending = fold_terminal(&mut state, &mut chain);
    Ok(Trace {
        actions,
        chain: finish(&chain),
        ending,
        obs: digest.as_ref().map(finish),
    })
}

/// The random drive: uniform over the sorted legal list, from the seed's own stream.
pub fn random_trace(game: &Game, seed: u64, observe: bool) -> Result<Trace, String> {
    let mut policy = stream(seed, &[Part::Str("corpus"), Part::Str("policy")]);
    trace(game, seed, observe, |_, legal, _| {
        Ok(legal[policy.below(legal.len() as u32) as usize])
    })
}

/// Replay recorded actions. Running out early, or finishing with actions left, is an error.
pub fn replay_trace(
    game: &Game,
    seed: u64,
    actions: &[u16],
    observe: bool,
) -> Result<Trace, String> {
    let traced = trace(game, seed, observe, |_, _, step| {
        actions
            .get(step)
            .copied()
            .ok_or_else(|| format!("the game outlived its {} recorded actions", actions.len()))
    })?;
    if traced.actions.len() != actions.len() {
        return Err(format!(
            "the game ended after {} of {} recorded actions",
            traced.actions.len(),
            actions.len()
        ));
    }
    Ok(traced)
}

/// H3 self-play, every seat's agent rooted at seed 0.
pub fn heuristic_trace(game: &Game, seed: u64, observe: bool) -> Result<Trace, String> {
    let mut agents: Vec<Heuristic> = (0..usize::from(game.rules.n_players))
        .map(|seat| {
            let mut agent = Heuristic::new(Tier::H3, HeuristicParams::default(), 0);
            agent.begin_game(seat, seed);
            agent
        })
        .collect();
    trace(game, seed, observe, |state, _, _| {
        let seat = usize::from(state.current_player());
        Ok(agents[seat].act(state))
    })
}

// ---------------------------------------------------------------------------
// Generation and checking
// ---------------------------------------------------------------------------

fn hex16(hex: &str) -> [u8; 16] {
    assert_eq!(hex.len(), 32, "expected 32 hex digits, got {hex:?}");
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("a hex digest");
    }
    out
}

/// The header this build would write for a configuration.
pub fn current_header(map: &str, n_players: usize) -> Header {
    let game = game(map, n_players);
    Header {
        contract_version: ttr_core::CONTRACT_VERSION,
        obs_version: ttr_core::obs::obs_version(),
        map: map.to_string(),
        n_players: n_players as u8,
        data_hash: hex16(game.board.data_hash),
        rules_hash: hex16(&game.cfg.rules_hash()),
        sample_every: SAMPLE_EVERY,
    }
}

/// Generate one configuration's corpus with this build.
pub fn generate(map: &str, n_players: usize) -> Corpus {
    let game = game(map, n_players);
    let label = format!("{map} {n_players}P");
    let random: Vec<Trace> = (0..RANDOM_SEEDS)
        .into_par_iter()
        .map(|seed| {
            random_trace(&game, u64::from(seed), seed < OBS_SEEDS)
                .unwrap_or_else(|e| panic!("{label} random seed {seed}: {e}"))
        })
        .collect();
    let heuristic: Vec<Trace> = (0..HEURISTIC_SEEDS)
        .into_par_iter()
        .map(|seed| {
            heuristic_trace(&game, u64::from(seed), true)
                .unwrap_or_else(|e| panic!("{label} heuristic seed {seed}: {e}"))
        })
        .collect();
    Corpus {
        header: current_header(map, n_players),
        random_obs: random[..OBS_SEEDS as usize]
            .iter()
            .map(|t| t.obs.expect("observed"))
            .collect(),
        random: random
            .iter()
            .map(|t| RandomGame {
                steps: u16::try_from(t.actions.len()).expect("a game fits in u16 steps"),
                chain: t.chain,
                scores: t.ending.scores.clone(),
                winners: t.ending.winners,
            })
            .collect(),
        heuristic: heuristic
            .into_iter()
            .map(|t| RecordedGame {
                chain: t.chain,
                scores: t.ending.scores,
                winners: t.ending.winners,
                obs: t.obs.expect("observed"),
                actions: t.actions,
            })
            .collect(),
    }
}

/// Which games a check re-derives. The fast tier takes prefixes; the slow tier takes all.
#[derive(Clone, Debug)]
pub struct Scope {
    pub random: Range<u32>,
    pub obs: Range<u32>,
    pub heuristic: Range<u32>,
    /// Compare observation digests at all. Off for `gen-corpus --check --play-only`, which
    /// checks what must still hold across an `OBS_VERSION` bump.
    pub observations: bool,
}

impl Scope {
    pub fn all() -> Self {
        Self::prefix(RANDOM_SEEDS, OBS_SEEDS, HEURISTIC_SEEDS)
    }

    pub fn prefix(random: u32, obs: u32, heuristic: u32) -> Self {
        Self {
            random: 0..random,
            obs: 0..obs,
            heuristic: 0..heuristic,
            observations: true,
        }
    }

    /// The same games, observation digests left out.
    pub fn without_observations(self) -> Self {
        Self {
            observations: false,
            ..self
        }
    }
}

/// Re-derive the games in `scope` and compare. Returns games checked.
///
/// Errors name the section, the seed and the first quantity that differs, lowest seed
/// first, so the report is the same however rayon schedules the work.
pub fn check(corpus: &Corpus, scope: &Scope) -> Result<usize, String> {
    let h = &corpus.header;
    let game = game(&h.map, usize::from(h.n_players));
    let label = format!("{} {}P", h.map, h.n_players);

    let random: Vec<Result<(), String>> = scope
        .random
        .clone()
        .into_par_iter()
        .map(|seed| {
            let want = &corpus.random[seed as usize];
            let here = format!("{label} random seed {seed}");
            let observe = scope.observations && scope.obs.contains(&seed);
            let got = random_trace(&game, u64::from(seed), observe)
                .map_err(|e| format!("{here}: {e}"))?;
            same(&here, "steps", got.actions.len(), usize::from(want.steps))?;
            same(&here, "scores", &got.ending.scores, &want.scores)?;
            same(&here, "winners", got.ending.winners, want.winners)?;
            same(&here, "chain", got.chain, want.chain)?;
            if let Some(obs) = got.obs {
                same(
                    &here,
                    "observation digest",
                    obs,
                    corpus.random_obs[seed as usize],
                )?;
            }
            Ok(())
        })
        .collect();
    random.into_iter().collect::<Result<(), String>>()?;

    let heuristic: Vec<Result<(), String>> = scope
        .heuristic
        .clone()
        .into_par_iter()
        .map(|seed| {
            let want = &corpus.heuristic[seed as usize];
            let here = format!("{label} heuristic seed {seed}");
            let got = replay_trace(&game, u64::from(seed), &want.actions, scope.observations)
                .map_err(|e| format!("{here}: {e}"))?;
            same(&here, "scores", &got.ending.scores, &want.scores)?;
            same(&here, "winners", got.ending.winners, want.winners)?;
            same(&here, "chain", got.chain, want.chain)?;
            if let Some(obs) = got.obs {
                same(&here, "observation digest", obs, want.obs)?;
            }
            Ok(())
        })
        .collect();
    heuristic.into_iter().collect::<Result<(), String>>()?;

    Ok(scope.random.len() + scope.heuristic.len())
}

fn same<T: PartialEq + std::fmt::Debug>(
    here: &str,
    what: &str,
    got: T,
    want: T,
) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{here}: {what} {got:?}, recorded {want:?}"))
    }
}

// ---------------------------------------------------------------------------
// Wire format: little-endian, no padding, every count before the thing it counts.
// ---------------------------------------------------------------------------

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at + n;
        if end > self.data.len() {
            return Err(format!("truncated at byte {} (wanted {n} more)", self.at));
        }
        let out = &self.data[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().expect("N bytes"))
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        self.array().map(u16::from_le_bytes)
    }

    fn i16(&mut self) -> Result<i16, String> {
        self.array().map(i16::from_le_bytes)
    }

    fn u32(&mut self) -> Result<u32, String> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, String> {
        self.array().map(u64::from_le_bytes)
    }

    fn scores(&mut self, n: usize) -> Result<Vec<i16>, String> {
        (0..n).map(|_| self.i16()).collect()
    }
}

impl Corpus {
    pub fn to_bytes(&self) -> Vec<u8> {
        let h = &self.header;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(FORMAT_VERSION);
        out.push(h.contract_version);
        out.extend_from_slice(&h.obs_version.to_le_bytes());
        out.push(h.n_players);
        out.push(u8::try_from(h.map.len()).expect("a short map name"));
        out.extend_from_slice(h.map.as_bytes());
        out.extend_from_slice(&h.data_hash);
        out.extend_from_slice(&h.rules_hash);
        out.push(h.sample_every);
        for len in [
            self.random.len(),
            self.random_obs.len(),
            self.heuristic.len(),
        ] {
            out.extend_from_slice(&(len as u32).to_le_bytes());
        }
        let scores = |out: &mut Vec<u8>, scores: &[i16]| {
            assert_eq!(scores.len(), usize::from(h.n_players));
            for s in scores {
                out.extend_from_slice(&s.to_le_bytes());
            }
        };
        for g in &self.random {
            out.extend_from_slice(&g.steps.to_le_bytes());
            out.extend_from_slice(&g.chain.to_le_bytes());
            scores(&mut out, &g.scores);
            out.push(g.winners);
        }
        for d in &self.random_obs {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for g in &self.heuristic {
            let len = u16::try_from(g.actions.len()).expect("a game fits in u16 steps");
            out.extend_from_slice(&len.to_le_bytes());
            for a in &g.actions {
                out.extend_from_slice(&a.to_le_bytes());
            }
            out.extend_from_slice(&g.chain.to_le_bytes());
            scores(&mut out, &g.scores);
            out.push(g.winners);
            out.extend_from_slice(&g.obs.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader { data, at: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a golden corpus file (bad magic)".into());
        }
        let format = r.u8()?;
        if format != FORMAT_VERSION {
            return Err(format!(
                "corpus format {format}, this reader is {FORMAT_VERSION}"
            ));
        }
        let contract_version = r.u8()?;
        let obs_version = r.u32()?;
        let n_players = r.u8()?;
        let name_len = usize::from(r.u8()?);
        let map = String::from_utf8(r.take(name_len)?.to_vec()).map_err(|e| e.to_string())?;
        let header = Header {
            contract_version,
            obs_version,
            map,
            n_players,
            data_hash: r.array()?,
            rules_hash: r.array()?,
            sample_every: r.u8()?,
        };
        let (n_random, n_obs, n_heuristic) = (r.u32()?, r.u32()?, r.u32()?);
        let n = usize::from(n_players);
        let random = (0..n_random)
            .map(|_| {
                Ok(RandomGame {
                    steps: r.u16()?,
                    chain: r.u64()?,
                    scores: r.scores(n)?,
                    winners: r.u8()?,
                })
            })
            .collect::<Result<_, String>>()?;
        let random_obs = (0..n_obs).map(|_| r.u64()).collect::<Result<_, _>>()?;
        let heuristic = (0..n_heuristic)
            .map(|_| {
                let len = usize::from(r.u16()?);
                Ok(RecordedGame {
                    actions: (0..len).map(|_| r.u16()).collect::<Result<_, _>>()?,
                    chain: r.u64()?,
                    scores: r.scores(n)?,
                    winners: r.u8()?,
                    obs: r.u64()?,
                })
            })
            .collect::<Result<_, String>>()?;
        if r.at != data.len() {
            return Err(format!("{} trailing bytes", data.len() - r.at));
        }
        Ok(Self {
            header,
            random,
            random_obs,
            heuristic,
        })
    }

    pub fn load(map: &str, n_players: usize) -> Result<Self, String> {
        let path = corpus_path(map, n_players);
        let data = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&data).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_format_round_trips() {
        let corpus = Corpus {
            header: current_header("mini", 3),
            random: vec![RandomGame {
                steps: 9,
                chain: 0x0123_4567_89ab_cdef,
                scores: vec![-4, 0, 31],
                winners: 0b100,
            }],
            random_obs: vec![u64::MAX],
            heuristic: vec![RecordedGame {
                actions: vec![3, 1, 4, 1, 5],
                chain: 7,
                scores: vec![1, 2, 3],
                winners: 0b111,
                obs: 9,
            }],
        };
        let bytes = corpus.to_bytes();
        assert_eq!(Corpus::from_bytes(&bytes).unwrap(), corpus);
        assert!(Corpus::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(Corpus::from_bytes(&longer).is_err());
    }

    #[test]
    fn a_trace_is_a_pure_function_of_the_seed() {
        let game = game("mini", 2);
        let a = random_trace(&game, 3, true).unwrap();
        let b = random_trace(&game, 3, true).unwrap();
        assert_eq!((a.chain, a.obs), (b.chain, b.obs));
        assert_ne!(a.chain, random_trace(&game, 4, true).unwrap().chain);
        let replayed = replay_trace(&game, 3, &a.actions, true).unwrap();
        assert_eq!((replayed.chain, replayed.obs), (a.chain, a.obs));
    }

    #[test]
    fn a_replay_that_does_not_fit_the_game_is_refused() {
        let game = game("mini", 2);
        let a = random_trace(&game, 3, false).unwrap();
        assert!(replay_trace(&game, 3, &a.actions[..a.actions.len() - 1], false).is_err());
        let mut longer = a.actions.clone();
        longer.push(0);
        assert!(replay_trace(&game, 3, &longer, false).is_err());
    }
}
