"""Agent specs: the one place a name becomes a player (agents/registry.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import ttr_rust

from ticket_to_ride.agents.registry import ANCHOR_FAMILY, parse_spec


def test_a_bare_name_is_the_anchor_and_a_tuned_variant_is_not(tmp_path: Path) -> None:
    """The mechanism by which ratings accumulate: `h3` stays the zero of the scale and a
    retune is a *new competitor* rated against it, never an edit to the anchor."""
    assert parse_spec("h3").is_anchor
    tuned = tmp_path / "h3_v2.toml"
    tuned.write_text("[params]\nmin_points_per_train = 1.0\n", encoding="utf-8")
    variant = parse_spec(f"h3@{tuned}")
    assert variant.family == ANCHOR_FAMILY
    assert not variant.is_anchor
    assert variant.overrides == {"min_points_per_train": 1.0}


def test_a_bare_table_works_as_well_as_a_params_section(tmp_path: Path) -> None:
    path = tmp_path / "flat.toml"
    path.write_text("threat_weight = 0.0\n", encoding="utf-8")
    assert parse_spec(f"h4@{path}").overrides == {"threat_weight": 0.0}


def test_random_is_an_alias_for_h0() -> None:
    assert parse_spec("random").family == "h0"


def test_an_unknown_agent_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown agent 'h9'"):
        parse_spec("h9")
    with pytest.raises(ValueError, match="names no checkpoint"):
        parse_spec("ppo:")


def test_the_spec_text_round_trips() -> None:
    """A leaderboard row has to be re-runnable, so the spec is kept verbatim rather than
    rebuilt from its parts."""
    for text in ("h0", "h3", "h4"):
        assert parse_spec(text).text == text


def test_identity_is_the_results_store_key() -> None:
    """One bot, one identity everywhere: what the arena stores is what the server seats."""
    row = parse_spec("h3").identity()
    assert (row.spec, row.family, row.checkpoint) == ("h3", "h3", None)
    assert row.params_hash == ttr_rust.params_hash(None)
    assert row.behaviour_hash == ttr_rust.behaviour_hash("h3")
    assert parse_spec("random").identity().behaviour_hash == ttr_rust.behaviour_hash("h0")


def test_a_tuned_variant_is_a_different_player(tmp_path: Path) -> None:
    tuned = tmp_path / "h3_v2.toml"
    tuned.write_text("[params]\nmin_points_per_train = 1.0\n", encoding="utf-8")
    variant = parse_spec(f"h3@{tuned}")
    assert variant.identity().behaviour_hash != parse_spec("h3").identity().behaviour_hash
    assert variant.name == "H3 (h3_v2)"
    assert parse_spec("h4").name == "H4"


def test_a_spec_builds_the_agent_it_names() -> None:
    game = ttr_rust.Game("mini", 2)
    state = game.new_initial_state(1)
    bot = parse_spec("h2").build()
    bot.begin_game(0, 1)
    assert bot.act(state) in state.legal_actions()


def test_learned_agents_have_a_seam_and_nothing_behind_it_yet() -> None:
    """`ppo:` parses, so a spec can name one; building or identifying it says Phase 6."""
    spec = parse_spec("ppo:runs/x/best.pt")
    assert (spec.family, spec.checkpoint, spec.name) == ("ppo", "runs/x/best.pt", "PPO (best)")
    with pytest.raises(NotImplementedError, match="Phase 6"):
        spec.build()
    with pytest.raises(NotImplementedError, match="Phase 6"):
        spec.identity()


def test_the_registry_does_not_import_torch() -> None:
    """The seam is lazy: the arena and the web server load the registry and never torch."""
    code = "import sys, ticket_to_ride.agents.registry; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
