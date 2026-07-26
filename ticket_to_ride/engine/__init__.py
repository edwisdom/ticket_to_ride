"""The Ticket to Ride rules engine.

**Imports without torch or numpy, and holds zero global mutable state.** Ten self-play
workers each importing torch costs ~1.5 s and ~300 MB RSS; a torch-free engine starts in
~50 ms, and CI runs the whole unit and property suite with no torch and no CUDA download.
That boundary is enforced three ways: torch is an optional extra, ruff bans module-level
torch imports here, and a test imports the engine in a subprocess with an import hook that
raises.

The frozen parts -- PRNG, draw procedure, `state_hash()` -- are specified in
docs/CONTRACT.md and pinned by test vectors. Nothing here may change them without a
`CONTRACT_VERSION` bump.
"""

from __future__ import annotations

from ticket_to_ride.engine.actions import ACTION_SPACE_VERSION, ActionSpace, action_space
from ticket_to_ride.engine.config import (
    CLOSED,
    FREE,
    PHASE_DRAW_SECOND,
    PHASE_INITIAL_TICKETS,
    PHASE_MAIN,
    PHASE_NAMES,
    PHASE_TERMINAL,
    PHASE_TICKET_KEEP,
    RuleConfig,
)
from ticket_to_ride.engine.contract import CONTRACT_VERSION
from ticket_to_ride.engine.scoring import (
    Breakdown,
    final_scores,
    longest_trails,
    returns,
    score_breakdown,
    winners,
)
from ticket_to_ride.engine.state import Game, IllegalAction, State

__all__ = [
    "ACTION_SPACE_VERSION",
    "CLOSED",
    "CONTRACT_VERSION",
    "FREE",
    "PHASE_DRAW_SECOND",
    "PHASE_INITIAL_TICKETS",
    "PHASE_MAIN",
    "PHASE_NAMES",
    "PHASE_TERMINAL",
    "PHASE_TICKET_KEEP",
    "ActionSpace",
    "Breakdown",
    "Game",
    "IllegalAction",
    "RuleConfig",
    "State",
    "action_space",
    "final_scores",
    "longest_trails",
    "returns",
    "score_breakdown",
    "winners",
]
