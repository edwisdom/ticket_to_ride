"""The agent protocol, and the seeding discipline every agent inherits.

One `seed` in the config; every stream **derived, never drawn**. An agent's randomness comes
from `("agent", seat, game_id)`, which is disjoint from the environment's `("env", ...)`.
That separation is not tidiness: if the two shared a stream, an agent that sampled one
extra action would shift every subsequent card draw, and paired evaluation would silently
stop being paired.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ticket_to_ride.engine.rng import Pcg32, stream
from ticket_to_ride.engine.state import State


class Agent(ABC):
    """Something that picks a legal action.

    Subclasses implement `act`. `begin_game` re-seats and re-seeds; the arena calls it once
    per game so a seat swap does not carry an agent's stream across games.
    """

    #: Short identifier used by the registry, the arena and the leaderboard.
    name: str = "agent"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.seat = 0
        self.rng: Pcg32 = stream(seed, "agent", 0, 0)

    def begin_game(self, seat: int, game_id: int = 0) -> None:
        self.seat = seat
        self.rng = stream(self.seed, "agent", seat, game_id)

    @abstractmethod
    def act(self, state: State) -> int:
        """Return a legal action for `state.current_player()`."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} seed={self.seed}>"
