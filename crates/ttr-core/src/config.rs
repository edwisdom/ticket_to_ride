//! Rule configuration and the small integer constants the whole engine speaks in.
//!
//! [`RuleConfig`] is everything that changes the game but is not board data. It hashes to
//! `rules_hash`, which every replay carries, so a rule change makes old replays fail
//! loudly instead of quietly replaying a different game.

use std::fmt;

use crate::board::{Board, get_board};
use crate::hashing::hash128_hex;

// -- seg_owner sentinels ----------------------------------------------------

/// Nobody owns this segment and it can still be claimed.
pub const FREE: u8 = 255;

/// 2-3P only: the sibling track of a claimed double route, closed to *everyone*.
///
/// A distinct sentinel rather than a fake owner, so scoring and the observation can tell
/// "blocked" from "someone's track" -- very different things to a blocking heuristic.
pub const CLOSED: u8 = 254;

// -- phases -----------------------------------------------------------------

pub const PHASE_INITIAL_TICKETS: u8 = 0;
pub const PHASE_MAIN: u8 = 1;
pub const PHASE_DRAW_SECOND: u8 = 2;
pub const PHASE_TICKET_KEEP: u8 = 3;
pub const PHASE_TERMINAL: u8 = 4;

pub const PHASE_NAMES: [&str; 5] = [
    "INITIAL_TICKETS",
    "MAIN",
    "DRAW_SECOND",
    "TICKET_KEEP",
    "TERMINAL",
];

// -- other magic numbers ----------------------------------------------------

/// Three or more locomotives face-up triggers the flush.
pub const FLUSH_LOCOS: usize = 3;

/// End-of-game trigger: a seat finishing its turn on this many trains or fewer.
pub const END_TRIGGER_TRAINS: u8 = 2;

/// `final_left` when the end has not been triggered.
pub const NOT_TRIGGERED: u8 = 255;

/// At or below this many seats, one claimed track of a double closes its sibling to all.
pub const DOUBLES_LOCKED_MAX_PLAYERS: usize = 3;

/// Vacated slot in the ticket ring, and "no ticket" in a pending offer.
///
/// Keeping vacated slots at a fixed value is what makes the ring's byte image canonical
/// and therefore hashable: a stale leftover id would let two identical positions hash
/// differently. Same class of bug as padding bytes in a struct hash.
pub const NO_TICKET: u8 = 255;

/// Empty face-up slot. Serialized as a signed byte, so 255 on the wire.
pub const EMPTY_SLOT: i8 = -1;

/// `tickets` is a per-seat bitmask in a u32.
pub const MAX_TICKET_BITS: usize = 32;

// -- policies ---------------------------------------------------------------

/// How a claim's payment is chosen from the action id.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WildPolicy {
    /// Always pay the most coloured cards possible, collapsing the naive `S x K x L`
    /// payment space to `S x K`. PLAN.md §5.3 has the dominance argument.
    #[default]
    Canonical,
    /// Kept as a Phase 5 ablation hook. Not implemented.
    Explicit,
}

/// Whether the engine resolves chance itself or exposes it as nodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ChanceMode {
    /// The engine resolves flips and reshuffles internally, so PPO and MCTS never see
    /// chance nodes. This is what the frozen contract specifies.
    #[default]
    Sampled,
    /// Chance exposed for CFR-style solvers. **Not implemented, and deferred past Phase 2
    /// deliberately** -- see [`ConfigError::NotImplemented`] and docs/WORKLOG.md.
    Explicit,
}

/// A configuration that cannot be honoured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    UnknownMap(String),
    PlayerCount {
        map: String,
        low: u8,
        high: u8,
        got: usize,
    },
    TooManyTickets {
        map: String,
        tickets: usize,
    },
    NotImplemented(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMap(name) => write!(f, "unknown map {name:?}"),
            Self::PlayerCount {
                map,
                low,
                high,
                got,
            } => write!(f, "{map} supports {low}-{high} players, got {got}"),
            Self::TooManyTickets { map, tickets } => write!(
                f,
                "{map} has {tickets} tickets; the per-seat bitmask holds {MAX_TICKET_BITS}"
            ),
            Self::NotImplemented(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Everything that changes the game but is not board data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleConfig {
    pub map_name: String,
    pub n_players: usize,
    pub wild_policy: WildPolicy,
    pub chance_mode: ChanceMode,
    pub initial_hand: u8,
    /// Belt and braces. The analytical 4P bound is ~335 turns; hitting 1000 means a bug.
    pub turn_cap: u16,
    /// Cascade limit for the locomotive flush. The `nonloco >= 3` guard already makes an
    /// infinite cascade impossible; this is the second lock on the most common TTR hang.
    pub flush_cascade_cap: u8,
    /// Off for search and benchmarks -- copying the action list dominates a clone.
    pub track_history: bool,
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            map_name: "usa".to_string(),
            n_players: 2,
            wild_policy: WildPolicy::default(),
            chance_mode: ChanceMode::default(),
            initial_hand: 4,
            turn_cap: 1000,
            flush_cascade_cap: 10,
            track_history: true,
        }
    }
}

impl RuleConfig {
    pub fn new(map_name: &str, n_players: usize) -> Result<Self, ConfigError> {
        let cfg = Self {
            map_name: map_name.to_string(),
            n_players,
            ..Self::default()
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject anything the engine cannot honour, with the same messages Python gives.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let board = get_board(&self.map_name)
            .ok_or_else(|| ConfigError::UnknownMap(self.map_name.clone()))?;
        let (low, high) = (board.raw.min_players, board.raw.max_players);
        if self.n_players < low as usize || self.n_players > high as usize {
            return Err(ConfigError::PlayerCount {
                map: self.map_name.clone(),
                low,
                high,
                got: self.n_players,
            });
        }
        if board.n_tickets > MAX_TICKET_BITS {
            return Err(ConfigError::TooManyTickets {
                map: self.map_name.clone(),
                tickets: board.n_tickets,
            });
        }
        if self.wild_policy != WildPolicy::Canonical {
            return Err(ConfigError::NotImplemented(
                "wild_policy=explicit (Phase 5 ablation)",
            ));
        }
        if self.chance_mode != ChanceMode::Sampled {
            // Deferred deliberately, and not merely unwritten. docs/CONTRACT.md §2.1 says
            // the `deck_counts()` view "is never the source of a draw", which is exactly
            // what an explicit chance node would make it; and §3.1's serialization has no
            // field for a pending chance event. Implementing it means adding serialized
            // state, i.e. a CONTRACT_VERSION bump that invalidates every golden replay --
            // during the phase whose whole job is to be checked against them. It lands in
            // Python first, then here. See docs/WORKLOG.md for the amended §14 criterion.
            return Err(ConfigError::NotImplemented(
                "chance_mode=explicit is unspecified by docs/CONTRACT.md and deferred to \
                 Phase 5; see docs/WORKLOG.md",
            ));
        }
        Ok(())
    }

    pub fn board(&self) -> &'static Board {
        get_board(&self.map_name).expect("validated at construction")
    }

    /// 2-3P: claiming either track of a double closes the sibling to *all* players.
    ///
    /// 4-5P: the sibling stays open to others, but one player may never own both. Getting
    /// this backwards is the single most common TTR implementation bug, and it is
    /// invisible in play -- an agent trained against a wrongly-locked board simply learns
    /// to avoid double routes and nothing looks broken.
    pub fn doubles_locked_for_everyone(&self) -> bool {
        self.n_players <= DOUBLES_LOCKED_MAX_PLAYERS
    }

    /// blake2b-128 over the rules that affect play. `track_history` is excluded.
    ///
    /// Byte-identical to Python's, because every replay carries it and a mismatch has to
    /// mean "the rules changed", not "the two engines spell their config differently".
    pub fn rules_hash(&self) -> String {
        let wild = match self.wild_policy {
            WildPolicy::Canonical => "canonical",
            WildPolicy::Explicit => "explicit",
        };
        let chance = match self.chance_mode {
            ChanceMode::Sampled => "sampled",
            ChanceMode::Explicit => "explicit",
        };
        let canonical = format!(
            "map={}|players={}|wild={}|chance={}|hand={}|turn_cap={}|flush_cap={}",
            self.map_name,
            self.n_players,
            wild,
            chance,
            self.initial_hand,
            self.turn_cap,
            self.flush_cascade_cap,
        );
        hash128_hex(canonical.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_two_player_usa() {
        let cfg = RuleConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.map_name, "usa");
        assert_eq!(cfg.n_players, 2);
    }

    #[test]
    fn doubles_lock_at_three_seats_and_not_at_four() {
        for n in 2..=3 {
            assert!(
                RuleConfig::new("usa", n)
                    .unwrap()
                    .doubles_locked_for_everyone()
            );
        }
        for n in 4..=5 {
            assert!(
                !RuleConfig::new("usa", n)
                    .unwrap()
                    .doubles_locked_for_everyone()
            );
        }
    }

    #[test]
    fn seat_counts_outside_the_map_are_rejected() {
        assert!(matches!(
            RuleConfig::new("usa", 1),
            Err(ConfigError::PlayerCount { .. })
        ));
        // mini caps at 4 seats.
        assert!(matches!(
            RuleConfig::new("mini", 5),
            Err(ConfigError::PlayerCount { .. })
        ));
        assert!(RuleConfig::new("mini", 4).is_ok());
    }

    #[test]
    fn unknown_maps_are_rejected() {
        assert!(matches!(
            RuleConfig::new("europe", 2),
            Err(ConfigError::UnknownMap(_))
        ));
    }

    #[test]
    fn explicit_chance_mode_is_refused_with_its_reason() {
        let cfg = RuleConfig {
            chance_mode: ChanceMode::Explicit,
            ..RuleConfig::default()
        };
        match cfg.validate() {
            Err(ConfigError::NotImplemented(msg)) => {
                assert!(
                    msg.contains("CONTRACT.md"),
                    "the refusal must say why: {msg}"
                );
            }
            other => panic!("explicit chance mode was accepted: {other:?}"),
        }
    }

    #[test]
    fn rules_hash_moves_with_every_rule_that_affects_play() {
        let base = RuleConfig::default();
        let baseline = base.rules_hash();
        let variants = [
            RuleConfig {
                n_players: 3,
                ..base.clone()
            },
            RuleConfig {
                map_name: "mini".into(),
                ..base.clone()
            },
            RuleConfig {
                initial_hand: 5,
                ..base.clone()
            },
            RuleConfig {
                turn_cap: 999,
                ..base.clone()
            },
            RuleConfig {
                flush_cascade_cap: 9,
                ..base.clone()
            },
        ];
        for v in &variants {
            assert_ne!(v.rules_hash(), baseline, "{v:?} did not move the hash");
        }
        // ...and not with the one that does not.
        assert_eq!(
            RuleConfig {
                track_history: false,
                ..base.clone()
            }
            .rules_hash(),
            baseline,
            "track_history must not affect rules_hash: it changes no rule, and including \
             it would make benchmark replays unreplayable by the default config"
        );
    }
}
