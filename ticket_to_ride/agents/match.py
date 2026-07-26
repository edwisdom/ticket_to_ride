"""Running a game between agents. The minimum `eval/` will be built on in Phase 3.

Deliberately thin: it seats agents, plays, and returns the result. Pairing, ratings, SPRT
and the SQLite store are Phase 3 and belong in `eval/`, not here.
"""

from __future__ import annotations

from typing import NamedTuple

from ticket_to_ride.agents.base import Agent
from ticket_to_ride.engine.scoring import final_scores, returns, winners
from ticket_to_ride.engine.state import Game, State


class Result(NamedTuple):
    """One finished game, from the seating that was actually played."""

    seed: int
    #: Agent names in seat order. Reporting a seating is what stops a win-rate matrix from
    #: being a mirrored fiction with first-player advantage confounded into every cell.
    seating: tuple[str, ...]
    scores: tuple[int, ...]
    returns: tuple[float, ...]
    winners: tuple[int, ...]
    turns: int
    state: State


def play_game(game: Game, agents: list[Agent], seed: int) -> Result:
    """Play one game to the end. Agents are seated in the order given."""
    if len(agents) != game.n_players:
        raise ValueError(f"{game.n_players} seats, {len(agents)} agents")
    for seat, agent in enumerate(agents):
        agent.begin_game(seat, seed)

    state = game.new_initial_state(seed)
    while not state.is_terminal():
        state.step(agents[state.current_player()].act(state))

    return Result(
        seed=seed,
        seating=tuple(a.name for a in agents),
        scores=tuple(final_scores(state)),
        returns=tuple(returns(state)),
        winners=tuple(winners(state)),
        turns=state.turn,
        state=state,
    )


def head_to_head(game: Game, agents: list[Agent], seeds: range) -> list[float]:
    """Mean return per agent over `seeds`, playing **both seatings** of every seed.

    Measuring both seatings rather than mirroring one is not a detail. A matrix whose
    off-diagonal pairs sum to exactly 1.00 reports one seating and its complement, with
    first-player advantage folded invisibly into every cell -- a real flaw in the published
    prior art. The cyclic rotation below seats every agent in every seat exactly once per
    seed, so seat bias cancels.
    """
    totals = [0.0] * len(agents)
    games = 0
    for seed in seeds:
        for rotation in range(len(agents)):
            order = [(i + rotation) % len(agents) for i in range(len(agents))]
            result = play_game(game, [agents[i] for i in order], seed)
            for seat, i in enumerate(order):
                totals[i] += result.returns[seat]
            games += 1
    return [t / games for t in totals]
