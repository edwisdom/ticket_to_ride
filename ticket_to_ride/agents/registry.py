"""Agents by name: the one place a spec string becomes a player.

A **spec** names an agent the same way on a command line, in a config file, in the arena's
store, on the web server and on a leaderboard row:

* `h0` ... `h4` -- the Rust heuristics (`random` is an alias for `h0`);
* `h3@configs/agents/h3_v2.toml` -- a tier with parameter overrides, a `[params]` table or
  a bare one. A tuned agent is a *different spec*, never an edit to the anchor's constants:
  `h3` stays the zero of the rating scale and `h3@...` is a new competitor rated against it;
* `ppo:runs/x/best.pt` -- reserved for Phase 6's learned agents. Parsed now so the seam
  exists; building one imports torch lazily, so nothing that only needs heuristics (the
  arena, the web server, `ttr --help`) ever pays for it (PLAN.md §2, boundary 3).

An agent's identity is what it plays, not what it is called: `(family, params_hash,
behaviour_hash, checkpoint)`, the key of the results store (`eval/store.py`). So one bot is
one identity everywhere -- in the arena, across the table from a human, on the combined
leaderboard.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import ttr_rust

from ticket_to_ride.eval.store import AgentRow

#: Tiers `ttr_rust` knows.
TIERS = ("h0", "h1", "h2", "h3", "h4")

#: Other names for a tier.
ALIASES = {"random": "h0"}

#: The permanent Elo anchor (PLAN.md §11). Its identity is held by
#: `ttr_rust.behaviour_hash`, not by this string.
ANCHOR_FAMILY = "h3"

#: Phase 6's learned agents: `ppo:<checkpoint>`.
LEARNED = "ppo"

#: What each tier is, in a sentence, for anyone choosing an opponent.
DESCRIPTIONS = {
    "h0": "Uniformly random legal moves: the floor every other agent must clear.",
    "h1": "Greedy: claims the route worth the most points it can afford, otherwise takes "
    "the face-up colour it holds most of.",
    "h2": "Ticket planner: plans the cheapest network for its tickets and claims its most "
    "contested segment first.",
    "h3": "H2 plus card and ticket values, endgame play, longest-route tiebreaks and "
    "ticket keeping by expected value. The rating scale's zero.",
    "h4": "H3 plus opponent modelling: blocks routes that would join an opponent's network "
    "and holds claims that are safe to wait on.",
}


class Bot(Protocol):
    """What the web server needs of an agent: decide, and say why.

    `ttr_rust.Heuristic` is one. A learned agent will be another, its `explain` carrying
    policy probabilities and a value estimate in the same slot (PLAN.md §14.2).
    """

    def begin_game(self, seat: int, game_id: int) -> None: ...
    def act(self, state: ttr_rust.State) -> int: ...
    def explain(self, state: ttr_rust.State, top_k: int = 8) -> ttr_rust.Explanation: ...


@dataclass(frozen=True)
class AgentSpec:
    """A resolved agent specification."""

    #: The spec string exactly as written, so a leaderboard row round-trips to a re-run.
    text: str
    family: str
    overrides: dict[str, Any] = field(default_factory=dict)
    #: A learned agent's weights; `None` for a heuristic.
    checkpoint: str | None = None
    #: The root of the agent's own random stream. Only H0 draws from it.
    seed: int = 0

    @property
    def is_anchor(self) -> bool:
        return self.family == ANCHOR_FAMILY and not self.overrides

    @property
    def name(self) -> str:
        """What a person reads: `H3`, or `H3 (h3_v2)` for a tuned variant."""
        base = self.family.upper()
        if self.checkpoint:
            return f"{base} ({Path(self.checkpoint).stem})"
        if self.overrides:
            return f"{base} ({Path(self.text.partition('@')[2]).stem})"
        return base

    @property
    def description(self) -> str:
        text = DESCRIPTIONS.get(self.family, "")
        return f"{text} With tuned parameters." if self.overrides and text else text

    def identity(self) -> AgentRow:
        """The content address: what it plays, not where it came from."""
        if self.family == LEARNED:
            raise _not_yet(self)
        overrides = self.overrides or None
        return AgentRow(
            spec=self.text,
            family=self.family,
            params_hash=ttr_rust.params_hash(overrides),
            behaviour_hash=_behaviour_hash(self.family, _frozen(self.overrides)),
        )

    def build(self) -> Bot:
        """A fresh agent, ready for `begin_game`."""
        if self.family == LEARNED:
            return _load_learned(self)
        return ttr_rust.Heuristic(self.family, self.overrides or None, self.seed)


def parse_spec(text: str, seed: int = 0) -> AgentSpec:
    """`h3`, `h3@path.toml`, or `ppo:path`."""
    if text.startswith(f"{LEARNED}:"):
        checkpoint = text.partition(":")[2]
        if not checkpoint:
            raise ValueError(f"{text!r} names no checkpoint; write {LEARNED}:path/to/weights.pt")
        return AgentSpec(text=text, family=LEARNED, checkpoint=checkpoint, seed=seed)
    name, _, path = text.partition("@")
    name = ALIASES.get(name, name)
    if name not in TIERS:
        raise ValueError(
            f"unknown agent {name!r}; available: {sorted((*TIERS, *ALIASES))} "
            f"or {LEARNED}:<checkpoint>"
        )
    overrides: dict[str, Any] = {}
    if path:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        overrides = dict(data.get("params", data))
    return AgentSpec(text=text, family=name, overrides=overrides, seed=seed)


def _frozen(overrides: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(overrides.items()))


@cache
def _behaviour_hash(family: str, overrides: tuple[tuple[str, Any], ...]) -> str:
    """Memoized here as well as in Rust: the probe is a couple of hundred milliseconds of
    real games, and the server asks on every game it creates or rebuilds."""
    return ttr_rust.behaviour_hash(family, dict(overrides) or None)


def _load_learned(spec: AgentSpec) -> Bot:
    """Phase 6. torch is imported here and nowhere above, so loading the registry -- and
    everything that only plays heuristics -- never imports it."""
    raise _not_yet(spec)


def _not_yet(spec: AgentSpec) -> NotImplementedError:
    return NotImplementedError(
        f"{spec.text}: learned agents arrive in Phase 6; this build plays the heuristics h0-h4"
    )
