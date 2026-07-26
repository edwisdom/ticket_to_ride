"""The throwaway agents: legality, determinism, and the strength ordering.

These agents are API exercise, not deliverables (PLAN.md §8.1). What is worth asserting is
exactly what makes them useful as an exercise: they never emit an illegal action, they are
reproducible from a seed, their constants scale to a different map, and they beat what they
are supposed to beat -- because "flat Monte Carlo does not beat random" is the de-risking
signal that the rollout plumbing is broken.
"""

from __future__ import annotations

import pytest

from ticket_to_ride.agents.base import Agent
from ticket_to_ride.agents.match import head_to_head, play_game
from ticket_to_ride.agents.registry import available, make_agent
from ticket_to_ride.agents.throwaway import FlatMonteCarlo, GreedyAgent, GreedyConfig
from ticket_to_ride.engine import Game, RuleConfig
from ticket_to_ride.engine.actions import BLIND_SLOT
from ticket_to_ride.engine.config import PHASE_MAIN

SPECS = ["random", "h1", "flatmc:4"]


def make_game(map_name: str = "usa", n_players: int = 2) -> Game:
    return Game(RuleConfig(map_name=map_name, n_players=n_players))


# ---------------------------------------------------------------------------
# Legality and determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_agents_only_ever_emit_legal_actions(spec: str, map_name: str) -> None:
    game = make_game(map_name)
    agents = [make_agent(spec, 1), make_agent("random", 2)]
    for seed in range(3):
        for agent in agents:
            agent.begin_game(agents.index(agent), seed)
        state = game.new_initial_state(seed)
        while not state.is_terminal():
            action = agents[state.current_player()].act(state)
            assert action in state.legal_actions(), f"{spec} produced illegal action {action}"
            state.step(action)


@pytest.mark.parametrize("spec", SPECS)
def test_agents_are_reproducible_from_a_seed(spec: str) -> None:
    game = make_game()
    first = play_game(game, [make_agent(spec, 1), make_agent("random", 2)], 7)
    second = play_game(game, [make_agent(spec, 1), make_agent("random", 2)], 7)
    assert first.state.history == second.state.history
    assert first.scores == second.scores


@pytest.mark.parametrize("spec", SPECS)
def test_a_different_agent_seed_changes_play(spec: str) -> None:
    game = make_game()
    a = play_game(game, [make_agent(spec, 1), make_agent("random", 2)], 7)
    b = play_game(game, [make_agent(spec, 99), make_agent("random", 2)], 7)
    if spec == "h1":
        pytest.skip("h1 is deterministic by construction; its seed only reseats it")
    assert a.state.history != b.state.history


def test_seating_reseeds_the_agent() -> None:
    """A seat swap must not carry an agent's stream across games."""
    agent = make_agent("random", 5)
    agent.begin_game(0, 1)
    first = agent.rng.next_u32()
    agent.begin_game(1, 1)
    assert agent.rng.next_u32() != first
    agent.begin_game(0, 1)
    assert agent.rng.next_u32() == first


# ---------------------------------------------------------------------------
# Strength ordering -- the de-risking ablations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("map_name", ["usa", "mini"])
def test_h1_beats_random_on_every_map(map_name: str) -> None:
    game = make_game(map_name)
    scores = head_to_head(game, [make_agent("h1", 1), make_agent("random", 2)], range(30))
    assert scores[0] > 0.5, f"{map_name}: h1 only scored {scores[0]:.2f} against random"


def test_flat_monte_carlo_beats_random() -> None:
    """If policy improvement does not happen here, the rollout code is broken."""
    game = make_game("mini")
    scores = head_to_head(game, [make_agent("flatmc:16", 1), make_agent("random", 2)], range(8))
    assert scores[0] > 0.3, f"flat MC only scored {scores[0]:.2f} against random"


def test_h1_beats_random_in_a_multiplayer_game() -> None:
    game = make_game("usa", 4)
    agents = [make_agent("h1", 1), *(make_agent("random", 10 + i) for i in range(3))]
    scores = head_to_head(game, agents, range(8))
    assert scores[0] == max(scores)


# ---------------------------------------------------------------------------
# The paper's bug: constants must scale to the map
# ---------------------------------------------------------------------------


def test_h1_exercises_every_action_type_on_every_map() -> None:
    """The prior art's baselines were invalidated by a threshold that could never fire.

    Its heuristics drew tickets only above 15 trains, on a 10-train map -- so every
    "well-designed heuristic" played its opening tickets and never drew another, and nothing
    in the results looked wrong. This asserts H1 actually claims, draws cards *and* draws
    tickets on both boards.
    """
    for map_name in ("usa", "mini"):
        game = make_game(map_name)
        space = game.space
        used = {"claim": False, "draw": False, "tickets": False, "keep": False}
        for seed in range(12):
            agents: list[Agent] = [make_agent("h1", 1), make_agent("h1", 2)]
            result = play_game(game, agents, seed)
            for action in result.state.history:
                if action < space.claim_end:
                    used["claim"] = True
                elif action < space.draw_tickets:
                    used["draw"] = True
                elif action == space.draw_tickets:
                    used["tickets"] = True
                elif action < space.pass_action:
                    used["keep"] = True
        missing = [name for name, seen in used.items() if not seen]
        assert not missing, f"{map_name}: h1 never used {missing}"


def test_the_ticket_threshold_is_a_fraction_not_an_absolute() -> None:
    """The same config has to mean the same thing on a 45-train and a 20-train board."""
    config = GreedyConfig(ticket_train_fraction=0.9, max_tickets=99)
    for map_name in ("usa", "mini"):
        game = make_game(map_name)
        agent = GreedyAgent(1, config)
        agent.begin_game(0, 0)
        state = game.new_initial_state(1)
        supply = game.board.raw.trains_per_player
        # Fresh out of setup, every seat is at full supply, so the gate must be open.
        assert state.trains[0] == supply
        assert state.trains[0] >= config.ticket_train_fraction * supply


def test_greedy_claims_the_highest_scoring_route_available() -> None:
    game = make_game()
    agent = GreedyAgent(1)
    agent.begin_game(0, 0)
    state = game.new_initial_state(1)
    while state.phase != PHASE_MAIN:
        state.step(state.legal_actions()[0])

    board, space = game.board, game.space
    action = agent.act(state)
    if action < space.claim_end:
        chosen = board.route_points[board.seg_len[action // space.k]]
        best = max(
            board.route_points[board.seg_len[a // space.k]]
            for a in state.legal_actions()
            if a < space.claim_end
        )
        assert chosen == best


def test_greedy_prefers_a_face_up_locomotive() -> None:
    game = make_game()
    agent = GreedyAgent(1)
    agent.begin_game(0, 0)
    state = game.new_initial_state(1)
    while state.phase != PHASE_MAIN:
        state.step(state.legal_actions()[0])
    state.faceup[:] = [0, 1, game.board.locomotive, 2, 3]
    state.trains[0] = 0  # no claim is affordable, so it must draw

    action = agent.act(state)
    assert action == game.space.draw(2)


def test_greedy_falls_back_to_a_blind_draw() -> None:
    game = make_game()
    agent = GreedyAgent(1, GreedyConfig(take_faceup_locomotive=False))
    agent.begin_game(0, 0)
    state = game.new_initial_state(1)
    while state.phase != PHASE_MAIN:
        state.step(state.legal_actions()[0])
    state.trains[0] = 0
    state.hand[: game.board.n_card_types] = bytes(game.board.n_card_types)

    action = agent.act(state)
    assert action == game.space.draw(BLIND_SLOT), "holding nothing, blind beats a new colour"


# ---------------------------------------------------------------------------
# Registry and match plumbing
# ---------------------------------------------------------------------------


def test_registry_resolves_and_names_agents() -> None:
    assert "random" in available() and "h1" in available()
    agent = make_agent("flatmc:8", 3)
    assert agent.name == "flatmc:8"
    assert isinstance(agent, FlatMonteCarlo)
    assert agent.simulations == 8, "the spec argument reaches the constructor"
    assert "flatmc" in repr(agent)


def test_registry_rejects_unknown_specs() -> None:
    with pytest.raises(KeyError, match="unknown agent"):
        make_agent("alphazero")


def test_match_rejects_a_wrong_sized_table() -> None:
    with pytest.raises(ValueError, match="3 seats, 2 agents"):
        play_game(make_game("usa", 3), [make_agent("random", 1), make_agent("random", 2)], 0)


def test_result_records_the_seating_that_was_played() -> None:
    """Never report a mirrored matrix; a result has to say which seating produced it."""
    game = make_game()
    result = play_game(game, [make_agent("h1", 1), make_agent("random", 2)], 3)
    assert result.seating == ("h1", "random")
    assert len(result.scores) == 2
    assert abs(sum(result.returns)) < 1e-9
    assert result.winners
    assert result.turns > 0


def test_head_to_head_plays_both_seatings() -> None:
    """Two identical agents must come out level, which only holds if both seats are played."""
    game = make_game()
    scores = head_to_head(game, [make_agent("random", 1), make_agent("random", 2)], range(4))
    assert abs(sum(scores)) < 1e-9
