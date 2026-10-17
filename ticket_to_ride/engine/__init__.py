"""The Ticket to Ride rules engine: a thin re-export of the Rust core, `ttr_rust`.

The engine is `crates/ttr-core`; this package only gives it a stable Python import path, so
agent, CLI and UI code reads `from ticket_to_ride.engine import Game` whatever the
extension module is called. Nothing here may grow game logic or board data: one
implementation, in Rust.
"""

from __future__ import annotations

from ttr_rust import (
    CONTRACT_VERSION,
    Game,
    IllegalAction,
    Rng,
    State,
    VecEnv,
    board_names,
    board_summary,
    obs_spec,
    obs_version,
)

#: The observation-layout version, baked into every checkpoint.
OBS_VERSION = obs_version()


def configurations() -> list[tuple[str, int]]:
    """Every `(map, seat count)` the engine supports."""
    out = []
    for name in board_names():
        low, high = board_summary(name)["players"]
        out.extend((name, n) for n in range(low, high + 1))
    return out


__all__ = [
    "CONTRACT_VERSION",
    "OBS_VERSION",
    "Game",
    "IllegalAction",
    "Rng",
    "State",
    "VecEnv",
    "board_names",
    "board_summary",
    "configurations",
    "obs_spec",
    "obs_version",
]
