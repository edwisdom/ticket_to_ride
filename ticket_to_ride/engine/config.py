"""Rule configuration and the small integer constants the whole engine speaks in.

`RuleConfig` is everything that changes the game but is not board data. It hashes to
`rules_hash`, which every replay carries, so a rule change makes old replays fail loudly
instead of silently replaying a different game.
"""

from __future__ import annotations

from typing import Final

import msgspec

from ticket_to_ride.data.board import Board, get_board
from ticket_to_ride.engine.hashing import hash128

# -- seg_owner sentinels ----------------------------------------------------

#: Nobody owns this segment and it can still be claimed.
FREE: Final = 255

#: 2-3P only: the sibling track of a claimed double route, closed to *everyone*. A
#: distinct sentinel rather than a fake owner, so scoring and the observation can tell
#: "blocked" from "someone's track" -- they are very different for a blocking heuristic.
CLOSED: Final = 254

# -- phases -----------------------------------------------------------------

PHASE_INITIAL_TICKETS: Final = 0
PHASE_MAIN: Final = 1
PHASE_DRAW_SECOND: Final = 2
PHASE_TICKET_KEEP: Final = 3
PHASE_TERMINAL: Final = 4

PHASE_NAMES: Final = ("INITIAL_TICKETS", "MAIN", "DRAW_SECOND", "TICKET_KEEP", "TERMINAL")

# -- other magic numbers ----------------------------------------------------

#: The face-up display is five cards in every published edition.
FACEUP_SLOTS: Final = 5

#: Three or more locomotives face-up triggers the flush.
FLUSH_LOCOS: Final = 3

#: End-of-game trigger: a seat finishing its turn on this many trains or fewer.
END_TRIGGER_TRAINS: Final = 2

#: `final_left` when the end has not been triggered.
NOT_TRIGGERED: Final = 255

MAX_PLAYERS: Final = 5

#: At or below this many seats, one claimed track of a double closes its sibling to all.
DOUBLES_LOCKED_MAX_PLAYERS: Final = 3


class RuleConfig(msgspec.Struct, frozen=True):
    """Everything that changes the game but is not board data.

    A `msgspec.Struct` rather than a dataclass so it decodes straight from TOML with type
    validation, and stays cheap enough to sit in the engine's hot path.
    """

    map_name: str = "usa"
    n_players: int = 2

    #: `canonical` collapses the 100x9x7 payment space to 100x9 by always paying the most
    #: colored cards possible (PLAN.md §5.3 has the dominance argument). `explicit` is kept
    #: as an ablation hook and is not implemented yet.
    wild_policy: str = "canonical"

    #: `sampled` resolves flips and reshuffles inside the engine so PPO/MCTS never see
    #: chance nodes. `explicit` exposes them for CFR-style solvers; not implemented yet.
    chance_mode: str = "sampled"

    initial_hand: int = 4

    #: Belt and braces. The analytical 4P bound is ~335 turns; hitting 1000 means a bug.
    turn_cap: int = 1000

    #: Cascade limit for the locomotive flush. The `nonloco >= 3` guard already makes an
    #: infinite cascade impossible; this is the second lock on the most common TTR hang.
    flush_cascade_cap: int = 10

    #: Off for search and benchmarks -- copying the action list dominates a 0.5 us clone.
    track_history: bool = True

    def __post_init__(self) -> None:
        board = get_board(self.map_name)
        low, high = board.raw.min_players, board.raw.max_players
        if not low <= self.n_players <= high:
            raise ValueError(f"{self.map_name} supports {low}-{high} players, got {self.n_players}")
        if self.wild_policy != "canonical":
            raise NotImplementedError(f"wild_policy={self.wild_policy!r} (Phase 5 ablation)")
        if self.chance_mode != "sampled":
            raise NotImplementedError(f"chance_mode={self.chance_mode!r} (Phase 5)")
        if self.initial_hand < 0:
            raise ValueError("initial_hand must be non-negative")

    @property
    def board(self) -> Board:
        return get_board(self.map_name)

    @property
    def doubles_locked_for_everyone(self) -> bool:
        """2-3P: claiming either track of a double closes the sibling to *all* players.

        4-5P: the sibling stays open to others, but one player may never own both. Getting
        this backwards is the single most common TTR implementation bug, and it is
        invisible in play -- an agent trained against a wrongly-locked board simply learns
        to avoid double routes and nothing looks broken.
        """
        return self.n_players <= DOUBLES_LOCKED_MAX_PLAYERS

    @property
    def rules_hash(self) -> str:
        """blake2b-128 over the rules that affect play. `track_history` is excluded."""
        canonical = (
            f"map={self.map_name}|players={self.n_players}|wild={self.wild_policy}"
            f"|chance={self.chance_mode}|hand={self.initial_hand}"
            f"|turn_cap={self.turn_cap}|flush_cap={self.flush_cascade_cap}"
        )
        return hash128(canonical.encode())
