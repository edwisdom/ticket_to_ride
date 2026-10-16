//! Game replays: ~360 bytes per game, and bitwise or they fail loudly.
//!
//! The port of `ticket_to_ride/engine/replay.py`, wire format byte for byte:
//! `tests/golden/replays.bin` was written by the Python engine, and it parses, replays and
//! re-packs identically here.
//!
//! A replay is `(contract version, map, seats, seed, DATA_HASH, rules_hash, actions, final
//! state hash, final scores)`. Everything except the actions is there to make a stale replay
//! *fail* rather than quietly replay a different game -- a board edit moves `DATA_HASH`, a
//! rule change moves `rules_hash`, and a change to the PRNG or the draw procedure moves
//! `CONTRACT_VERSION`.
//!
//! This is reproducibility level **L0**: bitwise, always, and tested. See
//! docs/reproducibility.md for the three levels and why conflating L1 and L2 costs a day
//! chasing ghosts.
//!
//! ## Wire format
//!
//! One record, fixed field order, little-endian, no padding:
//!
//! | Bytes | Field |
//! | --- | --- |
//! | 3 | [`MAGIC`], `TTR` |
//! | 1 | contract version |
//! | 1 | seat count `n` |
//! | 1 | map-name length `m` |
//! | `m` | map name, UTF-8 |
//! | 2 | action count `a` |
//! | 8 | seed |
//! | 16 | `DATA_HASH` |
//! | 16 | `rules_hash` |
//! | 8 | final `state_hash()` |
//! | `2n` | final scores, signed |
//! | `2a` | actions |
//!
//! A corpus ([`pack`]) is a u32 record count, then each record behind a u32 byte length.
//!
//! ## Where this is stricter than Python
//!
//! Every malformed input is a [`ReplayError`]. Python let some through as other exceptions:
//! a record cut short past its header raised `struct.error`, a non-UTF-8 map name
//! `UnicodeDecodeError`, a recorded action the engine refuses `IllegalAction`, and a
//! configuration the engine does not support whatever `RuleConfig` raised. A well-formed
//! file reads identically either way.

use std::fmt;

use crate::CONTRACT_VERSION;
use crate::config::{ChanceMode, ConfigError, RuleConfig, WildPolicy};
use crate::scoring::final_scores;
use crate::state::{Game, State};

/// File magic. Present so a truncated or unrelated file is rejected immediately.
pub const MAGIC: &[u8; 3] = b"TTR";

/// The map name is length-prefixed with a single byte.
pub const MAX_NAME: usize = 0xFF;

/// Magic, contract version, seat count, map-name length.
pub const HEADER_SIZE: usize = 6;

/// Action count (u16), seed (u64), `DATA_HASH` and `rules_hash` (16 bytes each), final
/// state hash (u64).
pub const BODY_SIZE: usize = 2 + 8 + 16 + 16 + 8;

/// A replay that cannot be trusted: wrong board, wrong rules, a diverged outcome, or bytes
/// that are not a replay at all.
///
/// One variant per check, so a caller can tell *which* one fired; [`fmt::Display`] gives
/// the Python engine's messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    // -- recording ----------------------------------------------------------
    /// [`record`] was handed a game that has not finished.
    Unfinished,
    /// [`record`] was handed a game played with `track_history` off.
    NoHistory,

    // -- the wire format ----------------------------------------------------
    /// The map name does not fit its one-byte length prefix.
    NameTooLong(String),
    /// More actions than the u16 action count can hold.
    TooManyActions(usize),
    /// A record whose score list does not have one entry per seat.
    ScoreCount { n_players: u8, scores: usize },
    /// The bytes end before the record does.
    Truncated,
    /// The first three bytes are not [`MAGIC`].
    NotAReplay([u8; 3]),
    /// The map name is not UTF-8.
    MapName,
    /// Bytes left over after a complete record.
    TrailingBytes(usize),
    /// The bytes end before the corpus does.
    TruncatedPack,
    /// Bytes left over after the last record of a corpus.
    TrailingPackBytes(usize),

    // -- replaying, in the order [`replay`] checks --------------------------
    /// Recorded under a different PRNG, draw procedure or serialization.
    ContractVersion { recorded: u8 },
    /// The recorded configuration is one this engine cannot build.
    Config(ConfigError),
    /// The board has changed since the replay was recorded.
    DataHash {
        map: String,
        recorded: [u8; 16],
        current: [u8; 16],
    },
    /// The rules have changed since the replay was recorded.
    RulesHash {
        recorded: [u8; 16],
        current: [u8; 16],
    },
    /// The game ended with recorded actions still to play.
    TooLong { index: usize, n_actions: usize },
    /// The engine refused a recorded action.
    Illegal {
        index: usize,
        action: u16,
        reason: String,
    },
    /// The recorded actions ran out before the game ended.
    EndedEarly,
    /// The terminal position is not the recorded one.
    FinalHash { got: u64, recorded: u64 },
    /// The terminal position matches but scores differently.
    FinalScores { got: Vec<i16>, recorded: Vec<i16> },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unfinished => write!(f, "only a finished game can be recorded"),
            Self::NoHistory => write!(
                f,
                "this game was played with track_history off; nothing to record"
            ),
            Self::NameTooLong(name) => write!(f, "map name '{name}' is too long"),
            Self::TooManyActions(n) => write!(f, "{n} actions do not fit the u16 action count"),
            Self::ScoreCount { n_players, scores } => {
                write!(f, "{scores} final scores for {n_players} seats")
            }
            Self::Truncated => write!(f, "truncated replay"),
            Self::NotAReplay(magic) => {
                write!(f, "not a replay: magic b'{}'", magic.escape_ascii())
            }
            Self::MapName => write!(f, "map name is not UTF-8"),
            Self::TrailingBytes(n) => write!(f, "trailing bytes: {n}"),
            Self::TruncatedPack => write!(f, "truncated pack"),
            Self::TrailingPackBytes(n) => write!(f, "trailing bytes in pack: {n}"),
            Self::ContractVersion { recorded } => write!(
                f,
                "replay was recorded under contract version {recorded}, this engine is \
                 version {CONTRACT_VERSION}; see docs/CONTRACT.md"
            ),
            Self::Config(e) => write!(f, "cannot rebuild the recorded configuration: {e}"),
            Self::DataHash {
                map,
                recorded,
                current,
            } => write!(
                f,
                "board '{map}' has changed since this replay was recorded ({} != {})",
                hex(recorded),
                hex(current)
            ),
            Self::RulesHash { recorded, current } => write!(
                f,
                "rules have changed since this replay was recorded ({} != {})",
                hex(recorded),
                hex(current)
            ),
            Self::TooLong { index, n_actions } => {
                write!(f, "replay is longer than the game: {index} of {n_actions}")
            }
            Self::Illegal {
                index,
                action,
                reason,
            } => write!(
                f,
                "replay diverged: recorded action {action} at step {index} was refused: {reason}"
            ),
            Self::EndedEarly => write!(f, "replay ended before the game did"),
            Self::FinalHash { got, recorded } => {
                write!(
                    f,
                    "replay diverged: final hash {got:016x} != {recorded:016x}"
                )
            }
            Self::FinalScores { got, recorded } => {
                write!(f, "replay diverged on scores: {got:?} != {recorded:?}")
            }
        }
    }
}

impl std::error::Error for ReplayError {}

/// One recorded game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replay {
    pub map_name: String,
    pub n_players: u8,
    pub seed: u64,
    pub actions: Vec<u16>,
    /// Board identity. A map edit invalidates every replay taken on the old board.
    pub data_hash: [u8; 16],
    /// Rule identity, covering everything in [`RuleConfig`] that affects play.
    pub rules_hash: [u8; 16],
    /// `state_hash()` of the terminal position. The thing a replay actually proves.
    pub final_hash: u64,
    pub final_scores: Vec<i16>,
    pub contract_version: u8,
}

impl Replay {
    pub fn n_actions(&self) -> usize {
        self.actions.len()
    }
}

/// Capture a finished game. Requires `track_history`, which is on by default.
///
/// There is no action-space check: Python refused a space larger than a u16, and here
/// `ActionSpace::n` and every action already *are* u16s.
pub fn record(state: &State) -> Result<Replay, ReplayError> {
    if !state.is_terminal() {
        return Err(ReplayError::Unfinished);
    }
    if !state.rules.track_history {
        return Err(ReplayError::NoHistory);
    }
    let board = state.board;
    Ok(Replay {
        map_name: board.name.to_string(),
        n_players: state.rules.n_players,
        seed: state.seed,
        actions: state.history.clone(),
        data_hash: unhex16(board.data_hash),
        rules_hash: unhex16(&rule_config_of(state).rules_hash()),
        final_hash: state.state_hash(),
        // Settling tickets path-halves the union-find, so it needs a scratch copy. The DSU
        // is a cache outside every hash; the copy only keeps `record` from taking `&mut`.
        final_scores: final_scores(&mut state.clone()),
        contract_version: CONTRACT_VERSION,
    })
}

/// The configuration `state` was built from, for its `rules_hash`.
///
/// `State` carries its board and the play-affecting scalars, not the `RuleConfig` itself.
/// The two policies it does not carry have exactly one accepted value each --
/// [`RuleConfig::validate`], which `Game::new` runs, refuses the rest -- so the rebuild is
/// exact. Every field is spelled out rather than defaulted, so a new rule field fails to
/// compile here instead of silently hashing as its default.
fn rule_config_of(state: &State) -> RuleConfig {
    let rules = &state.rules;
    RuleConfig {
        map_name: state.board.name.to_string(),
        n_players: usize::from(rules.n_players),
        wild_policy: WildPolicy::Canonical,
        chance_mode: ChanceMode::Sampled,
        initial_hand: rules.initial_hand,
        turn_cap: rules.turn_cap,
        flush_cascade_cap: rules.flush_cascade_cap,
        track_history: rules.track_history,
    }
}

/// Pack to the wire format. Fixed field order, little-endian, no padding.
pub fn encode(rec: &Replay) -> Result<Vec<u8>, ReplayError> {
    let name = rec.map_name.as_bytes();
    if name.len() > MAX_NAME {
        return Err(ReplayError::NameTooLong(rec.map_name.clone()));
    }
    let n_actions = u16::try_from(rec.actions.len())
        .map_err(|_| ReplayError::TooManyActions(rec.actions.len()))?;
    if rec.final_scores.len() != usize::from(rec.n_players) {
        return Err(ReplayError::ScoreCount {
            n_players: rec.n_players,
            scores: rec.final_scores.len(),
        });
    }

    let mut out = Vec::with_capacity(
        HEADER_SIZE + name.len() + BODY_SIZE + 2 * rec.final_scores.len() + 2 * rec.actions.len(),
    );
    out.extend_from_slice(MAGIC);
    out.push(rec.contract_version);
    out.push(rec.n_players);
    out.push(name.len() as u8);
    out.extend_from_slice(name);
    out.extend_from_slice(&n_actions.to_le_bytes());
    out.extend_from_slice(&rec.seed.to_le_bytes());
    out.extend_from_slice(&rec.data_hash);
    out.extend_from_slice(&rec.rules_hash);
    out.extend_from_slice(&rec.final_hash.to_le_bytes());
    for score in &rec.final_scores {
        out.extend_from_slice(&score.to_le_bytes());
    }
    for action in &rec.actions {
        out.extend_from_slice(&action.to_le_bytes());
    }
    Ok(out)
}

/// Unpack the wire format, rejecting anything that is not one of ours.
pub fn decode(data: &[u8]) -> Result<Replay, ReplayError> {
    let mut r = Reader::new(data, ReplayError::Truncated);
    let header: [u8; HEADER_SIZE] = r.array()?;
    let magic = [header[0], header[1], header[2]];
    if &magic != MAGIC {
        return Err(ReplayError::NotAReplay(magic));
    }
    let (contract_version, n_players, name_len) = (header[3], header[4], usize::from(header[5]));

    let map_name = std::str::from_utf8(r.take(name_len)?)
        .map_err(|_| ReplayError::MapName)?
        .to_string();
    let n_actions = usize::from(r.u16()?);
    let seed = r.u64()?;
    let data_hash = r.hash16()?;
    let rules_hash = r.hash16()?;
    let final_hash = r.u64()?;
    let final_scores = (0..n_players)
        .map(|_| r.i16())
        .collect::<Result<Vec<_>, _>>()?;
    let actions = (0..n_actions)
        .map(|_| r.u16())
        .collect::<Result<Vec<_>, _>>()?;
    if r.at != data.len() {
        return Err(ReplayError::TrailingBytes(data.len() - r.at));
    }

    Ok(Replay {
        map_name,
        n_players,
        seed,
        actions,
        data_hash,
        rules_hash,
        final_hash,
        final_scores,
        contract_version,
    })
}

/// Concatenate replays with a u32 length prefix each, for a corpus in one file.
pub fn pack(records: &[Replay]) -> Result<Vec<u8>, ReplayError> {
    let count = u32::try_from(records.len()).expect("a corpus of fewer than 2^32 records");
    let mut out = count.to_le_bytes().to_vec();
    for rec in records {
        let blob = encode(rec)?;
        // A name, a u8 of scores and a u16 of actions: always well inside a u32.
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&blob);
    }
    Ok(out)
}

/// Split a corpus written by [`pack`] back into its records.
pub fn unpack(data: &[u8]) -> Result<Vec<Replay>, ReplayError> {
    let mut r = Reader::new(data, ReplayError::TruncatedPack);
    let count = r.u32()?;
    // Not `with_capacity(count)`: the count is untrusted until the records behind it parse.
    let mut records = Vec::new();
    for _ in 0..count {
        let size = r.u32()? as usize;
        records.push(decode(r.take(size)?)?);
    }
    if r.at != data.len() {
        return Err(ReplayError::TrailingPackBytes(data.len() - r.at));
    }
    Ok(records)
}

/// Re-run a recorded game and prove it reproduced.
///
/// Every check here exists because its failure mode is otherwise silent. A replay taken on
/// a different board, under different rules, or under a different PRNG will happily
/// produce *a* game -- just not the one that was recorded.
///
/// `cfg` is for games played under non-default rules: without it the default
/// configuration for the recorded map and seat count is rebuilt, and either way the result
/// is checked against the recorded `rules_hash`.
pub fn replay(rec: &Replay, cfg: Option<&RuleConfig>) -> Result<State, ReplayError> {
    if rec.contract_version != CONTRACT_VERSION {
        return Err(ReplayError::ContractVersion {
            recorded: rec.contract_version,
        });
    }

    let cfg = match cfg {
        Some(cfg) => cfg.clone(),
        None => RuleConfig::new(&rec.map_name, usize::from(rec.n_players))
            .map_err(ReplayError::Config)?,
    };
    let game = Game::new(cfg).map_err(ReplayError::Config)?;

    let data_hash = unhex16(game.board.data_hash);
    if data_hash != rec.data_hash {
        return Err(ReplayError::DataHash {
            map: rec.map_name.clone(),
            recorded: rec.data_hash,
            current: data_hash,
        });
    }
    let rules_hash = unhex16(&game.cfg.rules_hash());
    if rules_hash != rec.rules_hash {
        return Err(ReplayError::RulesHash {
            recorded: rec.rules_hash,
            current: rules_hash,
        });
    }

    let mut state = game.new_initial_state(rec.seed);
    for (index, &action) in rec.actions.iter().enumerate() {
        if state.is_terminal() {
            return Err(ReplayError::TooLong {
                index,
                n_actions: rec.n_actions(),
            });
        }
        state.step(action).map_err(|e| ReplayError::Illegal {
            index,
            action,
            reason: e.0,
        })?;
    }

    if !state.is_terminal() {
        return Err(ReplayError::EndedEarly);
    }
    let final_hash = state.state_hash();
    if final_hash != rec.final_hash {
        return Err(ReplayError::FinalHash {
            got: final_hash,
            recorded: rec.final_hash,
        });
    }
    let scores = final_scores(&mut state);
    if scores != rec.final_scores {
        return Err(ReplayError::FinalScores {
            got: scores,
            recorded: rec.final_scores.clone(),
        });
    }
    Ok(state)
}

// ---------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------

/// Lowercase hex, the form board and rule ids take everywhere else.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A 32-digit hex id as bytes. Only ever handed the engine's own ids -- `board.data_hash`
/// and `rules_hash()` -- so a malformed one is a bug, not bad input.
fn unhex16(hex: &str) -> [u8; 16] {
    assert_eq!(hex.len(), 32, "expected 32 hex digits, got {hex:?}");
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .unwrap_or_else(|_| panic!("expected hex digits, got {hex:?}"));
    }
    out
}

/// A cursor over untrusted bytes. Running off the end is `short`, never a panic.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    short: ReplayError,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8], short: ReplayError) -> Self {
        Self { data, at: 0, short }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ReplayError> {
        match self.at.checked_add(n) {
            Some(end) if end <= self.data.len() => {
                let out = &self.data[self.at..end];
                self.at = end;
                Ok(out)
            }
            _ => Err(self.short.clone()),
        }
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ReplayError> {
        Ok(self
            .take(N)?
            .try_into()
            .expect("take returns exactly N bytes"))
    }

    fn u16(&mut self) -> Result<u16, ReplayError> {
        self.array().map(u16::from_le_bytes)
    }

    fn i16(&mut self) -> Result<i16, ReplayError> {
        self.array().map(i16::from_le_bytes)
    }

    fn u32(&mut self) -> Result<u32, ReplayError> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, ReplayError> {
        self.array().map(u64::from_le_bytes)
    }

    fn hash16(&mut self) -> Result<[u8; 16], ReplayError> {
        self.array()
    }
}
