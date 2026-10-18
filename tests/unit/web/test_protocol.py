"""The protocol's checked-in schema, its glossary, and its agreement with the engine."""

from __future__ import annotations

import subprocess
import sys
import typing
from pathlib import Path

import pytest
import ttr_rust

from ticket_to_ride.agents.registry import TIERS
from ticket_to_ride.ui.web import glossary
from ticket_to_ride.ui.web import protocol as p


def test_the_checked_in_schema_is_not_stale(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "gen_protocol.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_protocol_names_the_engines_maps() -> None:
    assert set(typing.get_args(p.MapName.__value__)) == set(ttr_rust.board_names())


def test_every_name_a_bot_explains_with_is_in_the_glossary() -> None:
    """The review shows each term, input and rule by name; one the glossary lacks is a
    number nobody can read. Every tier, both maps, every decision of several games."""
    names: dict[str, set[str]] = {"terms": set(), "facts": set(), "context": set(), "rules": set()}
    for map_name in ttr_rust.board_names():
        game = ttr_rust.Game(map_name, 2)
        for tier in TIERS:
            for seed in range(4):
                bots = [ttr_rust.Heuristic(tier if s == 0 else "h0", seed=seed) for s in range(2)]
                for seat, bot in enumerate(bots):
                    bot.begin_game(seat, seed)
                state = game.new_initial_state(seed)
                while not state.is_terminal():
                    e = bots[state.current_player()].explain(state, top_k=100)
                    names["context"] |= {name for name, _ in e.context}
                    for c in e.candidates:
                        names["terms"] |= {name for name, _ in c.terms}
                        names["facts"] |= {name for name, _ in c.facts}
                        names["rules"] |= {c.rule} if c.rule else set()
                    state.step(e.chosen)
    listed = {
        "terms": glossary.TERMS,
        "facts": glossary.FACTS,
        "context": glossary.CONTEXT,
        "rules": glossary.RULES,
    }
    for kind, seen in names.items():
        assert seen <= set(listed[kind]), (
            f"{kind} with no glossary entry: {seen - set(listed[kind])}"
        )
        assert seen, f"no {kind} were emitted at all"


@pytest.mark.parametrize("model", p.TOP_LEVEL)
def test_every_top_level_model_is_in_the_schema(model: type) -> None:
    schema = p.json_schema()
    assert model.__name__ in schema["$defs"]
    assert schema["properties"][model.__name__] == {"$ref": f"#/$defs/{model.__name__}"}
