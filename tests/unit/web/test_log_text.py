"""The plain-English log the client shows verbatim."""

from __future__ import annotations

import re

import pytest
import ttr_rust

from ticket_to_ride.ui.web import views


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_every_card_takes_the_right_article(map_name: str) -> None:
    info = views.board_info(map_name)
    for card in range(len(info.colors) + 1):
        name = info.card(card)
        assert info.a_card(card) == ("an " if name[0] in "aeiou" else "a ") + name
    assert "an orange" in {info.a_card(c) for c in range(len(info.colors))}


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_no_line_of_a_real_game_says_a_before_a_vowel(map_name: str) -> None:
    """ "H4 drew a orange from the deck" was the 4.3 client's catch; this reads every line
    of a few games, as each seat and omnisciently, for that and its relatives."""
    game = ttr_rust.Game(map_name, 2)
    names = ["H3", "H4"]
    article = re.compile(r"\ba [aeiou]", re.IGNORECASE)
    lines = 0
    for seed in range(8):
        bots = [ttr_rust.Heuristic(tier) for tier in ("h3", "h4")]
        for seat, bot in enumerate(bots):
            bot.begin_game(seat, seed)
        state = game.new_initial_state(seed)
        while not state.is_terminal():
            before = state.clone()
            action = bots[state.current_player()].act(state)
            state.step(action)
            for viewer in (None, 0, 1):
                text = views.entry(game, before, action, state, viewer=viewer, names=names).text
                assert not article.search(text), text
                lines += 1
    assert lines > 1000
