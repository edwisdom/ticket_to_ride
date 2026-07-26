"""Helpers for forcing an engine into a specific, otherwise-rare position.

Rules edge cases are exactly the positions a random game reaches once in ten thousand
tries, so the tests build them directly. `force` keeps **card** conservation intact so
`State.validate()` stays usable on a rigged state, which is most of the point.

The one invariant it cannot preserve is `trains[p] + track owned == trains_per_player`:
setting a seat's train count without also inventing the claims that spent them is exactly
what several end-of-game tests need. Tests that pass `trains=` must therefore skip
`validate()`.
"""

from __future__ import annotations

from ticket_to_ride.data.board import Board
from ticket_to_ride.engine.config import PHASE_MAIN
from ticket_to_ride.engine.state import EMPTY_SLOT, State


def cards(board: Board, **counts: int) -> list[int]:
    """Card-type counts from colour names: `cards(USA, blue=3, loco=1)`.

    `loco` is spelled out rather than assumed to be index 8 -- TTR-mini's locomotive is
    card type 6.
    """
    out = [0] * board.n_card_types
    for name, count in counts.items():
        index = board.locomotive if name == "loco" else board.color_names.index(name)
        out[index] = count
    return out


def filler(board: Board, n: int) -> list[int]:
    """`n` cards that certainly exist: cycles the colours and never the locomotive.

    A rigged deck of `[0] * 20` asks for twenty cards of one colour when only twelve are
    printed, which `force` rejects.
    """
    return [i % board.n_colors for i in range(n)]


def spread_hands(board: Board, n: int, count: int = 6) -> list[list[int]]:
    """One distinct colour per seat, so a five-seat rig never over-subscribes a colour."""
    return [[count if c == p else 0 for c in range(board.n_card_types)] for p in range(n)]


def force(  # noqa: PLR0912 - one branch per optional field; splitting it just hides the list
    state: State,
    *,
    hands: list[list[int]] | None = None,
    faceup: list[int] | None = None,
    deck: list[int] | None = None,
    trains: list[int] | None = None,
    tickets: list[int] | None = None,
    phase: int | None = None,
    cur: int | None = None,
) -> None:
    """Overwrite a state's guts, then place every unaccounted-for card so none is invented.

    With an explicit `deck`, the leftovers become the discard pile; without one they become
    the deck, in canonical order, and the discard is emptied. Either way the total is exact,
    and a rig that asks for more cards of a colour than the box contains is an error rather
    than a silently truncated position.

    Every hidden card is marked as blind-drawn (`certain` zero, `unknown` the hand size),
    which is the weakest consistent information-set assignment and keeps `validate()` happy.
    """
    board, n = state.game.board, state.game.n_players
    k = board.n_card_types

    if hands is not None:
        for p, counts in enumerate(hands):
            state.hand[p * k : (p + 1) * k] = bytes(counts)
    for p in range(n):
        size = sum(state.hand[p * k : (p + 1) * k])
        state.certain[p * k : (p + 1) * k] = bytes(k)
        state.unknown[p] = size

    if faceup is not None:
        state.faceup[:] = [*faceup, *([EMPTY_SLOT] * 5)][:5]
    if deck is not None:
        state.deck = bytes(deck)
    else:
        state.deck = b""
    state.deck_pos = 0
    if trains is not None:
        state.trains[:] = bytes(trains)
    if tickets is not None:
        state.tickets[:] = tickets
    if phase is not None:
        state.phase = phase
    if cur is not None:
        state.cur = cur

    leftover = list(board.deck_composition_counts)
    for c in range(k):
        leftover[c] -= sum(state.hand[p * k + c] for p in range(n))
    for card in state.faceup:
        if card != EMPTY_SLOT:
            leftover[card] -= 1
    for card in state.deck:
        leftover[card] -= 1
    if any(count < 0 for count in leftover):
        raise ValueError(f"rigged position uses more cards than exist: {leftover}")

    if deck is None:
        state.deck = bytes(c for c, count in enumerate(leftover) for _ in range(count))
        state.discard[:] = bytes(k)
        state.discard_total = 0
    else:
        state.discard[:] = bytes(leftover)
        state.discard_total = sum(leftover)


def to_main(state: State) -> State:
    """Play out the opening ticket choices with the lowest legal keep, reaching MAIN."""
    while state.phase != PHASE_MAIN:
        state.step(state.legal_actions()[0])
    return state


def play_until(state: State, predicate, rng, limit: int = 100_000) -> bool:  # noqa: ANN001
    """Step randomly until `predicate(state)` holds. Returns whether it ever did."""
    for _ in range(limit):
        if predicate(state):
            return True
        if state.is_terminal():
            return False
        state.step(state.sample_legal(rng))
    return False
