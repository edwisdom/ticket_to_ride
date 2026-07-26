"""Final scoring, rulebook tiebreaks, and the reward the agents actually optimize.

Route points are already banked as they are claimed; this module settles the two things
that can only be known at the end -- ticket completion and the longest continuous path --
and turns the result into a return signal.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from ticket_to_ride.engine.graph import longest_trail
from ticket_to_ride.engine.state import State

#: Two seats make the game strictly zero-sum, so the return is the plain win/draw/loss.
HEAD_TO_HEAD: Final = 2


class Breakdown(NamedTuple):
    """One seat's final score, itemized. `total` is what wins the game."""

    routes: int
    tickets_made: int
    tickets_missed: int
    ticket_points: int
    longest_bonus: int
    longest_trail: int
    completed: int
    total: int


def longest_trails(state: State) -> list[int]:
    """Each seat's longest continuous path, in train cars."""
    return [
        longest_trail(state.game.board, state.seg_owner, p) for p in range(state.game.n_players)
    ]


def score_breakdown(state: State) -> list[Breakdown]:
    """Itemized final scores for every seat.

    Ticket settlement is ±the ticket's value on whether the two cities are connected *by
    that seat's own network*. There is no hand limit and no cap on tickets, so a seat that
    hoarded unreachable tickets can finish well below zero.

    The longest-path bonus goes to **every tied seat**, not one of them -- the rulebook is
    explicit and implementations routinely award it to a single arbitrary winner.
    """
    board = state.game.board
    trails = longest_trails(state)
    best_trail = max(trails) if trails else 0

    out: list[Breakdown] = []
    for p in range(state.game.n_players):
        routes = state.score[p]
        made = missed = points = 0
        for ticket in state.tickets_of(p):
            value = board.ticket_points[ticket]
            if state.ticket_complete(p, ticket):
                made += 1
                points += value
            else:
                missed += 1
                points -= value
        # A seat with no track has a trail of 0; it must not tie for the bonus.
        bonus = board.raw.longest_bonus if trails[p] == best_trail and best_trail > 0 else 0
        out.append(
            Breakdown(
                routes=routes,
                tickets_made=made,
                tickets_missed=missed,
                ticket_points=points,
                longest_bonus=bonus,
                longest_trail=trails[p],
                completed=made,
                total=routes + points + bonus,
            )
        )
    return out


def final_scores(state: State) -> list[int]:
    return [b.total for b in score_breakdown(state)]


def rank_key(breakdown: Breakdown) -> tuple[int, int, int]:
    """The full rulebook ordering, highest first: points, then tickets, then longest path.

    A true draw is still possible -- two seats can match on all three -- and the engine
    reports it rather than inventing a winner.
    """
    return (breakdown.total, breakdown.completed, breakdown.longest_bonus)


def winners(state: State) -> list[int]:
    """Every seat that ties for first under the full tiebreak chain."""
    breakdowns = score_breakdown(state)
    keys = [rank_key(b) for b in breakdowns]
    best = max(keys)
    return [p for p, key in enumerate(keys) if key == best]


def returns(state: State) -> list[float]:
    """The reward. Terminal only, optimizing **win probability**, not score.

    The one published attempt at this problem found its score-optimizing agents lost
    head-to-head to its win-optimizing self-play agent, despite scoring more points. Raw
    route points as a reward is worse still: it directly incentivizes long-route greed over
    ticket completion and is the leading cause of the "agent never draws tickets" failure.

    * **2P** -- `+1 / 0 / -1` with the full tiebreaks, so a true draw scores 0 for both.
    * **3-5P** -- `0.75 * win + 0.25 * rank`, both terms constant-sum so self-play stays
      well-behaved and MCTS value backup stays valid. The rank term is what keeps a losing
      seat's gradient informative instead of flat.
    """
    n = state.game.n_players
    breakdowns = score_breakdown(state)
    keys = [rank_key(b) for b in breakdowns]
    best = max(keys)
    top = [p for p, key in enumerate(keys) if key == best]

    if n == HEAD_TO_HEAD:
        if len(top) == HEAD_TO_HEAD:
            return [0.0, 0.0]
        return [1.0 if p in top else -1.0 for p in range(n)]

    win = [(1.0 if p in top else 0.0) / len(top) for p in range(n)]
    # rank_norm: 1.0 for the best key, 0.0 for the worst, averaged over ties.
    order = sorted(range(n), key=lambda p: keys[p])
    rank_of: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        shared = (i + j) / 2 / (n - 1)
        for p in order[i : j + 1]:
            rank_of[p] = shared
        i = j + 1

    # Both terms are recentred so the vector sums to zero, which is what "constant-sum"
    # buys: value backup in self-play stays valid without a per-player baseline.
    mean_win = sum(win) / n
    mean_rank = sum(rank_of) / n
    return [0.75 * (win[p] - mean_win) + 0.25 * (rank_of[p] - mean_rank) for p in range(n)]
