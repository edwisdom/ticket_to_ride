"""Name-to-agent resolution: `"random"`, `"h1"`, `"flatmc:64"`.

This is the seam that keeps `eval/` from importing `rl/`. Neural agents will arrive here as
specs like `ppo:runs/x/best.pt` and lazily import torch *inside the factory*, so the arena
stays a general tournament tool that starts in ~50 ms.
"""

from __future__ import annotations

from collections.abc import Callable

from ticket_to_ride.agents.base import Agent
from ticket_to_ride.agents.throwaway import FlatMonteCarlo, GreedyAgent, RandomAgent

#: `name -> (seed, argument) -> Agent`. The argument is whatever followed the colon.
_FACTORIES: dict[str, Callable[[int, str | None], Agent]] = {
    "random": lambda seed, _: RandomAgent(seed),
    "h0": lambda seed, _: RandomAgent(seed),
    "h1": lambda seed, _: GreedyAgent(seed),
    "flatmc": lambda seed, arg: FlatMonteCarlo(seed, simulations=int(arg) if arg else 32),
}


def available() -> list[str]:
    return sorted(_FACTORIES)


def make_agent(spec: str, seed: int = 0) -> Agent:
    """Build an agent from a spec string.

    The `name:argument` shape is deliberate: it survives a command line, a TOML config and
    a leaderboard row unchanged, so an agent is identified the same way everywhere.
    """
    name, _, argument = spec.partition(":")
    factory = _FACTORIES.get(name)
    if factory is None:
        raise KeyError(f"unknown agent {name!r}; available: {available()}")
    agent = factory(seed, argument or None)
    agent.name = spec
    return agent
