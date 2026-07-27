"""The Rust observation encoder against the pure-Python oracle, bitwise on f32.

The other half of the Phase 2 exit criterion. Compared **exactly**, not within a
tolerance: both encoders compute in doubles and narrow on store, so agreement is a
property of how they are written rather than of the inputs, and a tolerance would hide
precisely the kind of formula difference this exists to catch.

Sampled at many points in each game rather than only at the start. Most of the encoder's
interesting features -- `remaining_cost`, `fragility`, `on_my_steiner_tree`,
`extends_my_chain`, `is_dead` -- are identically zero or trivially uniform on an opening
position, so a start-of-game-only comparison would exercise almost none of the code that
can actually be wrong.
"""

from __future__ import annotations

from types import ModuleType

import pytest
from differential import RustGame

from ticket_to_ride.data.board import BOARDS
from ticket_to_ride.engine.config import RuleConfig
from ticket_to_ride.engine.rng import stream
from ticket_to_ride.engine.state import Game
from ticket_to_ride.rl.encode.observation import encode, observation_size
from ticket_to_ride.rl.encode.spec import OBS_VERSION, ObsSpec, obs_spec

ALL_CONFIGS = [
    (name, n)
    for name, board in BOARDS.items()
    for n in range(board.raw.min_players, board.raw.max_players + 1)
]

#: How often to compare, in steps. Coprime with nothing in particular -- it just has to
#: not line up with the turn structure, so the samples land in every phase.
SAMPLE_EVERY = 7


def _describe(spec_obj: ObsSpec, index: int) -> str:
    """Name the block and field a differing slot falls in."""
    for block in spec_obj.blocks:
        if not block.offset <= index < block.offset + block.size:
            continue
        within = index - block.offset
        entity, inside = divmod(within, block.stride)
        for field in block.fields:
            if field.offset <= inside < field.offset + field.width:
                return (
                    f"{block.name}[{entity}].{field.name}[{inside - field.offset}]"
                    if block.count > 1
                    else f"{block.name}.{field.name}[{inside - field.offset}]"
                )
    return "<past every block>"


@pytest.mark.parametrize(("map_name", "n_players"), ALL_CONFIGS)
def test_rust_encoder_matches_the_python_oracle(
    map_name: str, n_players: int, rust: ModuleType
) -> None:
    board = BOARDS[map_name]
    spec_obj = obs_spec(board)
    for seed in range(4):
        game: RustGame = rust.Game(map_name, n_players)
        py = Game(RuleConfig(map_name=map_name, n_players=n_players)).new_initial_state(seed)
        rs = game.new_initial_state(seed)
        policy = stream(seed, "obs", "policy")

        step = 0
        while True:
            if step % SAMPLE_EVERY == 0 or py.is_terminal():
                for player in range(n_players):
                    want = list(encode(py, player))
                    got = rs.observation(player)
                    if want != got:
                        bad = [
                            (i, a, b)
                            for i, (a, b) in enumerate(zip(want, got, strict=True))
                            if a != b
                        ]
                        first = bad[0]
                        pytest.fail(
                            f"{map_name} {n_players}P seed {seed} step {step} seat {player}: "
                            f"{len(bad)} of {len(want)} slots differ; first is slot "
                            f"{first[0]} ({_describe(spec_obj, first[0])}) "
                            f"python {first[1]!r} rust {first[2]!r}"
                        )
            if py.is_terminal():
                break
            legal = py.legal_actions()
            action = legal[policy.below(len(legal))]
            py.step(action)
            rs.step(action)
            step += 1


@pytest.mark.parametrize(("map_name", "n_players"), ALL_CONFIGS)
def test_the_two_encoders_agree_on_size(map_name: str, n_players: int, rust: ModuleType) -> None:
    py = Game(RuleConfig(map_name=map_name, n_players=n_players)).new_initial_state(0)
    rs = rust.Game(map_name, n_players).new_initial_state(0)
    assert observation_size(py) == rs.observation_size


def test_the_comparison_can_actually_fail(rust: ModuleType) -> None:
    """Guard the guard: two different positions must not encode identically."""
    py = Game(RuleConfig(map_name="usa", n_players=2)).new_initial_state(1)
    rs = rust.Game("usa", 2).new_initial_state(2)
    assert list(encode(py, 0)) != rs.observation(0)


def test_obs_version_is_recorded_alongside_the_layout(rust: ModuleType) -> None:
    """`OBS_VERSION` is baked into every checkpoint, so the two sides must agree on it.

    A checkpoint trained against one layout and loaded against another plays garbage with
    no error; the version is what turns that into a hard failure, and it is only useful if
    both encoders report the same one.
    """
    assert OBS_VERSION == 1
    assert rust.obs_version() == OBS_VERSION
