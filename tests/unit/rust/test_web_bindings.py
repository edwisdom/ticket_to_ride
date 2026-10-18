"""The engine bindings the web server stands on (PLAN.md §14.2, Phase 4.1).

The heavy checks are Rust's: `crates/ttr-core/tests/explain.rs` runs `explain` against
`act` over ten thousand positions per map and tier. These pin the Python surface -- types,
errors, and that each accessor says what the engine means.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from ttr_rust import State


def _to_main(state: State) -> None:
    while state.phase_name != "MAIN":
        state.step(state.legal_actions()[0])


@pytest.mark.parametrize("map_name", ["usa", "mini"])
@pytest.mark.parametrize("tier", ["h1", "h2", "h3", "h4"])
def test_explain_returns_what_act_plays(rust: ModuleType, map_name: str, tier: str) -> None:
    game = rust.Game(map_name, 2)
    state, twin = game.new_initial_state(5), game.new_initial_state(5)
    actors = [rust.Heuristic(tier, seed=1), rust.Heuristic(tier, seed=1)]
    explainers = [rust.Heuristic(tier, seed=1), rust.Heuristic(tier, seed=1)]
    for seat in range(2):
        actors[seat].begin_game(seat, 5)
        explainers[seat].begin_game(seat, 5)
    while not state.is_terminal():
        seat = state.current_player()
        acted = actors[seat].act(state)
        explanation = explainers[seat].explain(twin, top_k=3)
        assert explanation.chosen == acted
        assert explanation.candidates[0].action == acted
        assert explanation.seat == seat
        assert len(explanation.candidates) <= 3
        assert acted in state.legal_actions()
        state.step(acted)
        twin.step(acted)


def test_h0_keeps_its_chosen_action_when_truncated(rust: ModuleType) -> None:
    game = rust.Game("usa", 2)
    state = game.new_initial_state(8)
    _to_main(state)
    h0 = rust.Heuristic("h0", seed=4)
    h0.begin_game(state.current_player(), 8)
    explanation = h0.explain(state, top_k=1)
    assert explanation.rule == "uniform"
    assert explanation.chosen in [c.action for c in explanation.candidates]
    assert len(explanation.candidates) <= 2


def test_a_candidate_breaks_its_total_into_terms(rust: ModuleType) -> None:
    game = rust.Game("usa", 2)
    state = game.new_initial_state(3)
    _to_main(state)
    h3 = rust.Heuristic("h3")
    h3.begin_game(state.current_player(), 3)
    explanation = h3.explain(state, top_k=50)
    names = {name for c in explanation.candidates for name, _ in c.terms}
    assert "card_value" in names or "expected_card_value" in names
    assert dict(explanation.context)["plan_cost"] > 0
    for candidate in explanation.candidates:
        if candidate.total != float("-inf"):
            assert sum(v for _, v in candidate.terms) == pytest.approx(candidate.total)


def test_h1_names_the_rule_that_fired(rust: ModuleType) -> None:
    game = rust.Game("mini", 2)
    state = game.new_initial_state(2)
    _to_main(state)
    h1 = rust.Heuristic("h1")
    h1.begin_game(state.current_player(), 2)
    explanation = h1.explain(state)
    assert explanation.rule in {
        "claim_most_points",
        "draw_tickets",
        "faceup_locomotive",
        "faceup_most_held",
        "blind_draw",
    }


def test_bad_agents_and_finished_games_are_refused(rust: ModuleType) -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        rust.Heuristic("h9")
    with pytest.raises(ValueError, match="unknown heuristic parameter"):
        rust.Heuristic("h3", {"no_such_constant": 1.0})
    game = rust.Game("mini", 2)
    state = game.new_initial_state(1)
    rng = rust.Rng.stream(1, "finish")
    while not state.is_terminal():
        legal = state.legal_actions()
        state.step(legal[rng.below(len(legal))])
    with pytest.raises(ValueError, match="game is over"):
        rust.Heuristic("h3").act(state)


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_every_action_decodes_to_what_its_name_says(rust: ModuleType, map_name: str) -> None:
    game = rust.Game(map_name, 2)
    kinds = {
        "CLAIM": "claim",
        "DRAW": "draw",
        "DRAW_TICKETS": "draw_tickets",
        "KEEP": "keep",
        "PASS": "pass",
    }
    for action in range(game.num_distinct_actions):
        decoded = game.decode_action(action)
        assert decoded.kind == kinds[game.action_to_string(action).split()[0]]
        if decoded.kind == "draw":
            assert decoded.blind == (decoded.slot is None)
        if decoded.kind == "keep":
            assert decoded.mask is not None
            assert decoded.offer_indices == [i for i in range(3) if decoded.mask >> i & 1]
    with pytest.raises(ValueError, match="outside"):
        game.decode_action(game.num_distinct_actions)


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_board_data_is_the_engine_order(rust: ModuleType, map_name: str) -> None:
    """Each segment's cities and length are what the engine's own action names say, and
    siblings pair up -- the board the client draws is the board the engine plays."""
    board = rust.board_data(map_name)
    game = rust.Game(map_name, 2)
    cards = len(board["colors"]) + 1
    assert board["locomotive"] == len(board["colors"])
    assert len(board["card_counts"]) == cards
    assert board["action_space"] == game.num_distinct_actions
    for index, (a, b, length, colour, sibling) in enumerate(board["segments"]):
        name = game.action_to_string(index * cards)
        assert name.startswith(f"CLAIM {board['cities'][a]}-{board['cities'][b]}[{length}")
        assert colour is None or 0 <= colour < len(board["colors"])
        if sibling is not None:
            assert board["segments"][sibling][4] == index
            assert board["segments"][sibling][:2] == (a, b)
    for a, b, points in board["tickets"]:
        assert 0 <= a < len(board["cities"]) and 0 <= b < len(board["cities"]) and points > 0


def test_mini_s_toml_is_not_the_engine_order(rust: ModuleType, repo_root: Path) -> None:
    """Why `board_data` exists: reading the TOML would draw a different board."""
    toml = tomllib.loads((repo_root / "ticket_to_ride/data/maps/mini.toml").read_text())
    board = rust.board_data("mini")
    cities = board["cities"]
    engine = [(cities[a], cities[b], length) for a, b, length, _, _ in board["segments"]]
    authored = [(a, b, length) for a, b, length, _ in toml["segments"]]
    assert sorted(map(sorted_pair, engine)) == sorted(map(sorted_pair, authored))
    assert engine != authored


def sorted_pair(segment: tuple[str, str, int]) -> tuple[str, str, int]:
    a, b, length = segment
    return (min(a, b), max(a, b), length)


def test_the_new_state_accessors(rust: ModuleType) -> None:
    game = rust.Game("usa", 2)
    state = game.new_initial_state(7)
    # The opening keep: three tickets on offer, and they are no one's yet.
    assert len(state.offer()) == 3
    assert not set(state.offer()) & set(state.tickets_of(0))
    _to_main(state)
    assert state.offer() == []
    assert state.discard() == [0] * 9
    progress = state.ticket_progress(0)
    assert [t for t, _, _ in progress] == list(state.tickets_of(0))
    assert all(
        not connected and remaining and remaining > 0 for _, connected, remaining in progress
    )
    with pytest.raises(ValueError, match="seat 2"):
        state.ticket_progress(2)

    claims = [a for a in state.legal_actions() if game.decode_action(a).kind == "claim"]
    for action in claims:
        decoded = game.decode_action(action)
        colored, locomotives = state.claim_payment(action)
        length = rust.board_data("usa")["segments"][decoded.segment][2]
        assert colored + locomotives == length
    with pytest.raises(ValueError, match="not a claim"):
        state.claim_payment(game.num_distinct_actions - 1)


def test_ranks_agree_with_the_winners(rust: ModuleType) -> None:
    game = rust.Game("mini", 3)
    for seed in range(5):
        state = game.new_initial_state(seed)
        rng = rust.Rng.stream(seed, "ranks")
        while not state.is_terminal():
            legal = state.legal_actions()
            state.step(legal[rng.below(len(legal))])
        ranks = state.ranks()
        assert [seat for seat, rank in enumerate(ranks) if rank == 1] == state.winners()
