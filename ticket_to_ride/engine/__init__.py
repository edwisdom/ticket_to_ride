"""The Ticket to Ride rules engine: a thin re-export of the Rust core, `ttr_rust`.

The engine is `crates/ttr-core`; this package only gives it a stable Python import path, so
agent, CLI and UI code reads `from ticket_to_ride.engine import Game` whatever the
extension module is called. There is no Python engine -- it was retired in Phase 3.1 once
the Rust build could stand as its own oracle (docs/PLAN.md §14.1) -- and nothing here may
grow game logic: a second implementation is exactly what the retirement removed.

The frozen parts -- PRNG, draw procedure, `state_hash()` -- are specified in
docs/CONTRACT.md and pinned by test vectors and the golden corpus in `tests/golden/`.
"""

from __future__ import annotations

from ttr_rust import (
    CONTRACT_VERSION,
    Game,
    IllegalAction,
    Rng,
    State,
    VecEnv,
    obs_spec,
    obs_version,
)

#: The observation-layout version, baked into every checkpoint.
OBS_VERSION = obs_version()

__all__ = [
    "CONTRACT_VERSION",
    "OBS_VERSION",
    "Game",
    "IllegalAction",
    "Rng",
    "State",
    "VecEnv",
    "obs_spec",
    "obs_version",
]
