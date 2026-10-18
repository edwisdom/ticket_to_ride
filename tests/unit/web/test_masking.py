"""Information-set masking is tested, not assumed (PLAN.md §14.2).

**The exact form of §12's "perfect test":** on real positions, a seat's view of a state
equals its view of `resample_from_infoset(seat)` of that state -- which keeps everything the
seat can see and redeals everything it cannot (opponents' blind-drawn cards, the deck). A
view that read any hidden card would differ.

The sampler keeps ticket holdings and the ticket deck as they are (`ttr_core::resample`),
so it cannot catch an opponent's tickets or offer leaking; those are asserted directly.
"""

from __future__ import annotations

import json

import pytest
import ttr_rust

from ticket_to_ride.ui.web import views

#: Per map, PLAN.md §14.2's floor; each is checked from both seats.
POSITIONS = 1000


def _positions(map_name: str, count: int):  # noqa: ANN202 -- a generator of tuples
    """Positions from H3 against H3 and against H0 -- the trajectory the agents generate
    and the ragged one they do not."""
    game = ttr_rust.Game(map_name, 2)
    seen = 0
    for seed in range(10_000):
        tiers = ("h3", "h3" if seed % 2 else "h0")
        bots = [ttr_rust.Heuristic(tier, seed=seed) for tier in tiers]
        for seat, bot in enumerate(bots):
            bot.begin_game(seat, seed)
        state = game.new_initial_state(seed)
        while not state.is_terminal():
            yield game, state, seed, seen
            seen += 1
            if seen == count:
                return
            state.step(bots[state.current_player()].act(state))


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_a_seat_sees_only_its_information_set(map_name: str) -> None:
    redealt = 0
    for game, state, seed, index in _positions(map_name, POSITIONS):
        for seat in range(2):
            rng = ttr_rust.Rng.stream(seed, "masking", index, seat)
            sample = state.resample_from_infoset(seat, rng)
            sample.assert_consistent(state, seat)
            assert views.view(sample, seat) == views.view(state, seat), (
                f"{map_name} seed {seed} position {index}: seat {seat}'s view reads hidden state"
            )
            assert views.legal(game, sample, seat) == views.legal(game, state, seat)
            # Guard the guard: the resample must actually have redealt something, or the
            # comparison above proves nothing.
            redealt += views.view(sample, None) != views.view(state, None)
    assert redealt > POSITIONS, f"only {redealt} of {2 * POSITIONS} resamples changed anything"


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_an_opponents_hand_tickets_and_offer_never_appear(map_name: str) -> None:
    for _, state, _, _ in _positions(map_name, 300):
        for seat in range(2):
            view = views.view(state, seat)
            opponent = view.seats[1 - seat]
            assert opponent.hand is None and opponent.tickets is None
            assert view.deck_cards is None, "the deck's composition is not a seat's to see"
            if state.current_player() != seat:
                assert view.offer is None, "an opponent's offer leaked"


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_the_log_tells_a_seat_only_what_it_saw(map_name: str) -> None:
    """A blind draw's card and a keep's tickets are the actor's alone: the same move,
    logged for the other seat, must not name them -- and the text must not either."""
    game = ttr_rust.Game(map_name, 2)
    info = views.board_info(map_name)
    names = ["One", "Two"]
    hidden_kinds = 0
    for _, state, _, _ in _positions(map_name, 600):
        actor = state.current_player()
        for action in state.legal_actions()[:3] + state.legal_actions()[-3:]:
            after = state.clone()
            after.step(action)
            mine = views.entry(game, state, action, after, viewer=actor, names=names)
            theirs = views.entry(game, state, action, after, viewer=1 - actor, names=names)
            match theirs.action.kind:
                case "draw" if theirs.action.slot is None:
                    hidden_kinds += 1
                    assert theirs.action.card is None
                    assert "from the deck" in theirs.text and " a card " in theirs.text
                    assert mine.action.card is not None  # ty: ignore[unresolved-attribute]
                case "keep" | "draw_tickets":
                    hidden_kinds += 1
                    assert theirs.action.tickets is None
                    for ticket in range(len(info.tickets)):
                        assert info.ticket(ticket) not in theirs.text
                case _:
                    assert (
                        json.loads(theirs.model_dump_json())["action"]
                        == json.loads(mine.model_dump_json())["action"]
                    )
    assert hidden_kinds > 100
