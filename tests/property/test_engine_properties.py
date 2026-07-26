"""Properties that must hold in every game, on every map, at every player count.

The rules tests in `unit/engine/test_rules.py` check named positions. These check that
nothing *else* went wrong on the way there, by sweeping seeds and asserting invariants
after every single step. Where a property is about the shape of the action space rather
than about play, hypothesis drives it instead of a seed sweep.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rig import play_until

from ticket_to_ride.data.board import BOARDS, NO_SIBLING
from ticket_to_ride.engine import (
    CLOSED,
    FREE,
    PHASE_TERMINAL,
    Game,
    IllegalAction,
    RuleConfig,
    final_scores,
    returns,
    score_breakdown,
    winners,
)
from ticket_to_ride.engine.rng import Pcg32, stream
from ticket_to_ride.engine.state import State

#: Every (map, player count) the engine claims to support.
CONFIGURATIONS = [
    (name, n)
    for name, board in BOARDS.items()
    for n in range(board.raw.min_players, board.raw.max_players + 1)
]
ALL_CONFIGS = pytest.mark.parametrize(
    ("map_name", "n_players"), CONFIGURATIONS, ids=[f"{m}-{n}p" for m, n in CONFIGURATIONS]
)

#: Enough seeds to exercise reshuffles and flushes without slowing the fast suite.
SWEEP = 25


def games(map_name: str, n_players: int, seeds: range) -> Iterator[tuple[Game, State, Pcg32]]:
    """Yield `(state, rng)` pairs ready to be played out."""
    game = Game(RuleConfig(map_name=map_name, n_players=n_players))
    for seed in seeds:
        yield game, game.new_initial_state(seed), stream(seed, "policy")


# ---------------------------------------------------------------------------
# 1-5. Conservation and legality, asserted after every step
# ---------------------------------------------------------------------------


@ALL_CONFIGS
def test_every_invariant_holds_after_every_step(map_name: str, n_players: int) -> None:
    """Cards, trains, tickets and the information-set bookkeeping, all the way down.

    `validate()` is the single place these live, so one sweep covers conservation of the
    deck, `certain + unknown == hand`, per-seat train accounting, ticket conservation
    across hands and deck and offer, and the locomotive-flush assertion.
    """
    for _, state, rng in games(map_name, n_players, range(SWEEP)):
        state.validate()
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
            state.validate()


@ALL_CONFIGS
def test_sampled_actions_are_always_legal(map_name: str, n_players: int) -> None:
    for _, state, rng in games(map_name, n_players, range(SWEEP)):
        while not state.is_terminal():
            legal = state.legal_actions()
            action = state.sample_legal(rng)
            assert action in legal, f"sample_legal returned {action}, not in legal_actions()"
            state.step(action)


@ALL_CONFIGS
def test_legal_actions_are_sorted_and_unique(map_name: str, n_players: int) -> None:
    """Two engines' lists must compare directly, which needs a canonical order."""
    for _, state, rng in games(map_name, n_players, range(SWEEP)):
        while not state.is_terminal():
            legal = state.legal_actions()
            assert legal == sorted(legal)
            assert len(set(legal)) == len(legal)
            state.step(state.sample_legal(rng))


@ALL_CONFIGS
def test_a_non_terminal_state_always_has_a_legal_action(map_name: str, n_players: int) -> None:
    """The engine must never construct a node with nothing to do -- PASS exists for this."""
    for _, state, rng in games(map_name, n_players, range(SWEEP)):
        while not state.is_terminal():
            assert state.legal_actions(), f"dead end at turn {state.turn}"
            state.step(state.sample_legal(rng))


@ALL_CONFIGS
def test_legal_action_mask_agrees_with_the_list(map_name: str, n_players: int) -> None:
    for game, state, rng in games(map_name, n_players, range(5)):
        buffer = bytearray(game.space.n)
        while not state.is_terminal():
            legal = state.legal_actions()
            fresh = state.legal_action_mask()
            reused = state.legal_action_mask(buffer)
            assert fresh == reused, "the reused buffer was not cleared"
            assert [i for i, bit in enumerate(fresh) if bit] == legal
            state.step(state.sample_legal(rng))


# ---------------------------------------------------------------------------
# 6-8. Board-level rules that must never be violated by any sequence of play
# ---------------------------------------------------------------------------


@ALL_CONFIGS
def test_no_player_ever_owns_both_tracks_of_a_double(map_name: str, n_players: int) -> None:
    for game, state, rng in games(map_name, n_players, range(SWEEP)):
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        for segment, sibling in enumerate(game.board.sibling):
            if sibling == NO_SIBLING:
                continue
            owner, twin = state.seg_owner[segment], state.seg_owner[sibling]
            if owner < CLOSED and twin < CLOSED:
                assert owner != twin, "one seat owns both tracks of a double route"


@ALL_CONFIGS
def test_closed_siblings_appear_only_in_two_and_three_player_games(
    map_name: str, n_players: int
) -> None:
    for game, state, rng in games(map_name, n_players, range(10)):
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        closed = state.seg_owner.count(CLOSED)
        if game.doubles_locked:
            claimed_doubles = sum(
                1
                for s in range(game.board.n_segments)
                if state.seg_owner[s] < CLOSED and game.board.sibling[s] != NO_SIBLING
            )
            assert closed == claimed_doubles
        else:
            assert closed == 0, "4-5P must never close a sibling to everyone"


@ALL_CONFIGS
def test_games_always_terminate_well_inside_the_cap(map_name: str, n_players: int) -> None:
    for game, state, rng in games(map_name, n_players, range(SWEEP)):
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        assert state.phase == PHASE_TERMINAL
        assert state.turn < game.cfg.turn_cap, "hit the belt-and-braces turn cap"


# ---------------------------------------------------------------------------
# 9-12. Cloning and hashing
# ---------------------------------------------------------------------------


@ALL_CONFIGS
def test_clone_is_fully_independent(map_name: str, n_players: int) -> None:
    for _, state, rng in games(map_name, n_players, range(5)):
        while not state.is_terminal():
            copy = state.clone()
            assert copy.state_hash() == state.state_hash()
            copy.step(copy.sample_legal(stream(1, "other")))
            assert state.state_hash() != copy.state_hash() or state.is_terminal()
            state.step(state.sample_legal(rng))
            # Mutating the original must not have touched the earlier clone's arrays.
            copy.validate()


@ALL_CONFIGS
def test_clone_into_matches_clone(map_name: str, n_players: int) -> None:
    """`clone_into` exists so search never allocates; it must be exactly `clone`."""
    game = Game(RuleConfig(map_name=map_name, n_players=n_players))
    arena = game.new_initial_state(0)
    for _, state, rng in games(map_name, n_players, range(5)):
        while not state.is_terminal():
            state.clone_into(arena)
            assert arena.state_hash() == state.state_hash()
            assert arena.legal_actions() == state.legal_actions()
            state.step(state.sample_legal(rng))


@ALL_CONFIGS
def test_position_hash_ignores_the_rng_and_the_dealt_deck(map_name: str, n_players: int) -> None:
    """Two states differing only in which order dealt cards came out are one position."""
    for _, state, rng in games(map_name, n_players, range(5)):
        while not state.is_terminal():
            twin = state.clone()
            twin.rng = Pcg32(twin.rng.state ^ 0xDEAD_BEEF, twin.rng.inc)
            twin.deck = bytes(reversed(twin.deck[: twin.deck_pos])) + twin.deck[twin.deck_pos :]
            assert twin.position_hash() == state.position_hash()
            assert twin.state_hash() != state.state_hash()
            state.step(state.sample_legal(rng))


@ALL_CONFIGS
def test_state_hash_separates_states_that_really_differ(map_name: str, n_players: int) -> None:
    seen: dict[int, int] = {}
    for _, state, rng in games(map_name, n_players, range(8)):
        while not state.is_terminal():
            digest = state.state_hash()
            turn = seen.setdefault(digest, state.turn)
            assert turn == state.turn, "two different positions collided"
            state.step(state.sample_legal(rng))


# ---------------------------------------------------------------------------
# 13-15. Determinism -- the property everything downstream depends on
# ---------------------------------------------------------------------------


@ALL_CONFIGS
def test_the_same_seed_replays_bitwise(map_name: str, n_players: int) -> None:
    for seed in range(10):
        game = Game(RuleConfig(map_name=map_name, n_players=n_players))
        first, second = game.new_initial_state(seed), game.new_initial_state(seed)
        rng_a, rng_b = stream(seed, "policy"), stream(seed, "policy")
        while not first.is_terminal():
            assert first.state_hash() == second.state_hash()
            action = first.sample_legal(rng_a)
            assert action == second.sample_legal(rng_b)
            first.step(action)
            second.step(action)
        assert second.is_terminal()
        assert first.state_hash() == second.state_hash()
        assert final_scores(first) == final_scores(second)


@ALL_CONFIGS
def test_different_seeds_deal_different_games(map_name: str, n_players: int) -> None:
    game = Game(RuleConfig(map_name=map_name, n_players=n_players))
    digests = {game.new_initial_state(seed).state_hash() for seed in range(50)}
    assert len(digests) == 50


def test_environment_randomness_does_not_depend_on_agent_behaviour() -> None:
    """The precondition for paired evaluation.

    Two games on the same seed whose agents act differently must still hold the *same*
    shuffle. Sampling the deck lazily from card counts would break this silently, with no
    error anywhere -- the variance-reduction scheme would simply stop working.
    """
    game = Game(RuleConfig(n_players=2))
    a, b = game.new_initial_state(4), game.new_initial_state(4)
    assert a.deck == b.deck
    for _ in range(12):
        if a.is_terminal() or b.is_terminal():
            break
        a.step(a.legal_actions()[0])
        b.step(b.legal_actions()[-1])
    # Neither game reshuffled, so both still hold the permutation the seed produced.
    assert a.deck == b.deck, "agent behaviour perturbed the environment's randomness"


# ---------------------------------------------------------------------------
# 16-17. Scoring
# ---------------------------------------------------------------------------


@ALL_CONFIGS
def test_final_scores_match_the_breakdown(map_name: str, n_players: int) -> None:
    for _, state, rng in games(map_name, n_players, range(10)):
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        breakdowns = score_breakdown(state)
        assert final_scores(state) == [b.total for b in breakdowns]
        for p, b in enumerate(breakdowns):
            assert b.total == b.routes + b.ticket_points + b.longest_bonus
            assert b.tickets_made + b.tickets_missed == int(state.tickets[p]).bit_count()
            assert b.routes == sum(
                state.game.board.route_points[state.game.board.seg_len[s]]
                for s in range(state.game.board.n_segments)
                if state.seg_owner[s] == p
            )


@ALL_CONFIGS
def test_returns_are_constant_sum_and_agree_with_the_winner(map_name: str, n_players: int) -> None:
    for _, state, rng in games(map_name, n_players, range(10)):
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
        r = returns(state)
        assert abs(sum(r)) < 1e-9, f"returns must be constant-sum, got {r}"
        top = winners(state)
        assert max(range(n_players), key=lambda p: r[p]) in top


# ---------------------------------------------------------------------------
# 18-19. The action space itself
# ---------------------------------------------------------------------------


@ALL_CONFIGS
def test_step_rejects_every_illegal_action(map_name: str, n_players: int) -> None:
    """The strongest statement the engine makes: an illegal action cannot get through.

    A policy that leaks one past its mask is the classic silent RL bug, so `step()`
    re-derives the preconditions of whatever it is handed rather than trusting the caller.
    """
    for game, state, rng in games(map_name, n_players, range(3)):
        stride = max(1, game.space.n // 40)
        while not state.is_terminal():
            legal = set(state.legal_actions())
            for action in range(0, game.space.n, stride):
                if action in legal:
                    continue
                probe = state.clone()
                with pytest.raises(IllegalAction):
                    probe.step(action)
            state.step(state.sample_legal(rng))


@ALL_CONFIGS
def test_a_rejected_action_leaves_the_state_untouched(map_name: str, n_players: int) -> None:
    for game, state, rng in games(map_name, n_players, range(3)):
        while not state.is_terminal():
            legal = set(state.legal_actions())
            before = state.state_hash()
            history = list(state.history)
            for action in range(0, game.space.n, max(1, game.space.n // 20)):
                if action in legal:
                    continue
                with pytest.raises(IllegalAction):
                    state.step(action)
                assert state.state_hash() == before
                assert state.history == history, "a rejected action was recorded"
            state.step(state.sample_legal(rng))


def test_sample_legal_covers_the_whole_legal_set() -> None:
    """Uniformity: over many streams every legal action must come up."""
    game = Game(RuleConfig(n_players=2))
    state = game.new_initial_state(1)
    while len(state.legal_actions()) < 10:
        state.step(state.legal_actions()[0])
    legal = state.legal_actions()
    drawn = Counter(state.sample_legal(stream(i, "uniform")) for i in range(200 * len(legal)))
    assert set(drawn) == set(legal)
    expected = 200
    assert all(expected * 0.6 < n < expected * 1.4 for n in drawn.values()), drawn


def test_sample_legal_refuses_a_terminal_state() -> None:
    game = Game(RuleConfig(n_players=2))
    state = game.new_initial_state(1)
    rng = stream(1, "policy")
    while not state.is_terminal():
        state.step(state.sample_legal(rng))
    with pytest.raises(IllegalAction, match="no legal actions"):
        state.sample_legal(rng)


@given(st.integers(min_value=-5000, max_value=5000))
@settings(max_examples=60, deadline=None)
def test_out_of_range_actions_are_rejected(action: int) -> None:
    game = Game(RuleConfig(n_players=2))
    state = game.new_initial_state(1)
    if 0 <= action < game.space.n:
        return
    with pytest.raises(IllegalAction, match="outside"):
        state.step(action)


@given(st.integers(min_value=0, max_value=914))
@settings(max_examples=120, deadline=None)
def test_action_names_round_trip(action: int) -> None:
    game = Game(RuleConfig(n_players=2))
    name = game.action_to_string(action)
    assert name
    if action < game.space.claim_end:
        segment, pay = game.space.decode_claim(action)
        assert game.space.claim(segment, pay) == action
        assert name.startswith("CLAIM")


# ---------------------------------------------------------------------------
# Reshuffles are rare; make sure at least one seed actually reaches one
# ---------------------------------------------------------------------------


def _reshuffle_detector(original: bytes) -> Callable[[State], bool]:
    """True once the state's deck object is no longer the one dealt at construction."""

    def changed(state: State) -> bool:
        return state.deck is not original

    return changed


def test_the_sweep_actually_exercises_a_reshuffle() -> None:
    """A guard on the guard: an untested branch that never runs proves nothing."""
    game = Game(RuleConfig(n_players=2))
    for seed in range(SWEEP):
        state = game.new_initial_state(seed)
        reshuffled = _reshuffle_detector(state.deck)
        if play_until(state, reshuffled, stream(seed, "policy")):
            return
    pytest.fail(f"no seed below {SWEEP} reshuffled; the reshuffle path is untested")


def test_the_sweep_actually_exercises_a_locomotive_flush() -> None:
    game = Game(RuleConfig(n_players=2))
    for seed in range(200):
        state = game.new_initial_state(seed)
        rng = stream(seed, "policy")
        while not state.is_terminal():
            before = state.discard_total
            faceup_before = list(state.faceup)
            state.step(state.sample_legal(rng))
            if state.discard_total - before >= 5 and faceup_before.count(FREE) == 0:
                locos = faceup_before.count(game.board.locomotive)
                if locos >= 2:
                    return
    pytest.fail("no seed below 200 triggered a locomotive flush")


def test_the_flush_cascade_cap_actually_fires_in_a_real_game() -> None:
    """The cap is load-bearing, not belt and braces.

    Once most of the deck is in players' hands, the available pool can be small and
    locomotive-heavy, and every reflush deals three more of them. Found by strengthening
    `validate()`'s flush assertion, which until then passed vacuously.
    """
    game = Game(RuleConfig(n_players=5))
    for seed in range(SWEEP):
        state = game.new_initial_state(seed)
        rng = stream(seed, "policy")
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
            if state.flush_capped:
                assert state.faceup.count(game.board.locomotive) >= 3
                return
    pytest.fail(f"no seed below {SWEEP} reached a cascade bail-out; the branch is untested")


def test_flush_capped_is_transient_and_never_hashed() -> None:
    """It records how the last flush ended, not where the game is."""
    game = Game(RuleConfig(n_players=2))
    state = game.new_initial_state(1)
    before = state.state_hash()
    state.flush_capped = not state.flush_capped
    assert state.state_hash() == before
    assert state.clone().flush_capped == state.flush_capped
