"""Throwaway H0/H1 and a flat Monte-Carlo stub. **Not deliverables -- API exercise.**

PLAN.md §8.1 mitigation 1. The Rust port is Phase 2, which means the engine API gets frozen
before agents, search or RL have used it in anger. These three exist to put `clone()`,
`clone_into()`, `legal_actions()` and `step()` under search-like load *now*, while changing
the interface is still free. The real H1-H4 are written once, in Rust, in Phase 3 -- H3
doubles as the ISMCTS rollout policy and a rollout that crosses the FFI boundary per step
would defeat the entire batching design.

What the exercise already surfaced is recorded in docs/WORKLOG.md.

**Every constant here is a fraction of the map's own train supply, never an absolute.** The
one published attempt at this problem had its heuristic baselines invalidated by exactly
that: thresholds hard-coded at 15 trains, carried onto a 10-train map, where the condition
could never fire -- so every "well-designed heuristic" played its opening tickets and never
drew another.
"""

from __future__ import annotations

import msgspec

from ticket_to_ride.agents.base import Agent
from ticket_to_ride.engine.actions import BLIND_SLOT
from ticket_to_ride.engine.config import PHASE_INITIAL_TICKETS, PHASE_MAIN, PHASE_TICKET_KEEP
from ticket_to_ride.engine.scoring import returns
from ticket_to_ride.engine.state import EMPTY_SLOT, State


class RandomAgent(Agent):
    """H0. Uniform over legal actions -- the floor every other agent must clear."""

    name = "random"

    def act(self, state: State) -> int:
        return state.sample_legal(self.rng)


class GreedyConfig(msgspec.Struct, frozen=True):
    """H1's constants, all relative to the map.

    `ticket_train_fraction` is the one that matters: "draw more tickets only while at least
    this fraction of my trains is left" transfers from a 45-train board to a 20-train board,
    where `trains > 15` does not.
    """

    ticket_train_fraction: float = 0.55
    max_tickets: int = 4
    #: Prefer face-up locomotives; they are worth two cards of anything.
    take_faceup_locomotive: bool = True


class GreedyAgent(Agent):
    """H1. Claim the best-scoring route if you can, otherwise collect cards.

    Deliberately shallow -- no connectivity reasoning, no blocking, no ticket valuation.
    It exists to be beaten, and to prove the action space is usable end to end.
    """

    name = "h1"

    def __init__(self, seed: int = 0, config: GreedyConfig | None = None) -> None:
        super().__init__(seed)
        self.config = config or GreedyConfig()

    def act(self, state: State) -> int:
        if state.phase in (PHASE_INITIAL_TICKETS, PHASE_TICKET_KEEP):
            return self._keep(state)
        legal = state.legal_actions()
        if state.phase == PHASE_MAIN:
            claim = self._best_claim(state, legal)
            if claim is not None:
                return claim
            tickets = self._maybe_draw_tickets(state, legal)
            if tickets is not None:
                return tickets
        draw = self._best_draw(state, legal)
        return draw if draw is not None else legal[0]

    # -- the three decisions -----------------------------------------------

    def _best_claim(self, state: State, legal: list[int]) -> int | None:
        """Highest route points, ties broken by the lowest action id so it stays replayable."""
        board, space = state.game.board, state.game.space
        best, best_points = None, 0
        for action in legal:
            if action >= space.claim_end:
                break  # claims come first and legal_actions() is sorted
            points = board.route_points[board.seg_len[action // space.k]]
            if points > best_points:
                best, best_points = action, points
        return best

    def _maybe_draw_tickets(self, state: State, legal: list[int]) -> int | None:
        space = state.game.space
        if space.draw_tickets not in legal:
            return None
        if int(state.tickets[self.seat]).bit_count() >= self.config.max_tickets:
            return None
        supply = state.game.board.raw.trains_per_player
        if state.trains[self.seat] < self.config.ticket_train_fraction * supply:
            return None
        return space.draw_tickets

    def _best_draw(self, state: State, legal: list[int]) -> int | None:
        """A face-up locomotive, else the colour already held most, else blind."""
        space, board = state.game.space, state.game.board
        draws = [a for a in legal if space.draw_base <= a < space.draw_tickets]
        if not draws:
            return None

        base = self.seat * board.n_card_types
        best, best_score = None, -1
        for action in draws:
            slot = action - space.draw_base
            if slot == BLIND_SLOT:
                continue
            card = state.faceup[slot]
            if card == EMPTY_SLOT:
                continue
            if card == board.locomotive and self.config.take_faceup_locomotive:
                return action
            score = state.hand[base + card]
            if score > best_score:
                best, best_score = action, score
        # Holding none of every face-up colour: a blind draw beats starting a new one.
        if best is None or best_score == 0:
            blind = space.draw_base + BLIND_SLOT
            if blind in draws:
                return blind
        return best if best is not None else draws[0]

    def _keep(self, state: State) -> int:
        """Keep the fewest tickets allowed, choosing the ones with the shortest routes."""
        board, space = state.game.board, state.game.space
        legal = state.legal_actions()
        offer = state.offer[: state.offer_len]
        cost = [board.dist[board.ticket_a[t]][board.ticket_b[t]] for t in offer]

        def total(action: int) -> tuple[int, int, int]:
            mask = action - space.keep_base
            kept = [i for i in range(state.offer_len) if mask >> i & 1]
            return (len(kept), sum(cost[i] for i in kept), action)

        return min(legal, key=total)


class FlatMonteCarlo(Agent):
    """A flat Monte-Carlo stub: score each legal action by random rollouts, take the best.

    **A stub, and cheating.** It rolls out from the true state rather than from a
    determinization, so it sees the opponents' hands and the deck order. That is exactly
    what Phase 5's `resample_from_infoset` fixes. Its job right now is to hammer
    `clone_into()` and `legal_actions()` at search rates and surface interface gaps -- and
    it already earns its place as the de-risking ablation in PLAN.md §13: if flat MC with
    random rollouts does not beat a random agent, the rollout plumbing is broken.
    """

    name = "flatmc"

    def __init__(self, seed: int = 0, simulations: int = 32, rollout_cap: int = 400) -> None:
        super().__init__(seed)
        self.simulations = simulations
        self.rollout_cap = rollout_cap
        self._arena: State | None = None

    def act(self, state: State) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]

        # One reusable scratch state for every rollout, so search never allocates.
        if self._arena is None or self._arena.game is not state.game:
            self._arena = state.clone()
        arena = self._arena

        seat = state.current_player()
        budget = max(1, self.simulations // len(legal))
        best, best_value = legal[0], -float("inf")
        for action in legal:
            total = 0.0
            for _ in range(budget):
                state.clone_into(arena)
                arena.step(action)
                total += self._rollout(arena, seat)
            value = total / budget
            if value > best_value:
                best, best_value = action, value
        return best

    def _rollout(self, state: State, seat: int) -> float:
        for _ in range(self.rollout_cap):
            if state.is_terminal():
                break
            state.step(state.sample_legal(self.rng))
        if not state.is_terminal():
            # A capped rollout is not a terminal position; score it as a draw rather than
            # pretending the current standing is a result.
            return 0.0
        return returns(state)[seat]
