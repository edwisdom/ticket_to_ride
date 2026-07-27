"""Agent specs and lineup scheduling. No games played, so no Rust needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from ticket_to_ride.eval.arena import ANCHOR_FAMILY, parse_spec, round_robin


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


def test_the_spec_text_round_trips() -> None:
    """A leaderboard row has to be re-runnable, so the spec is kept verbatim rather than
    rebuilt from its parts."""
    for text in ("h0", "h3", "h4"):
        assert parse_spec(text).text == text


def test_a_round_robin_lists_each_group_once() -> None:
    """Rotations are the arena's job, so a lineup is an unordered choice -- listing (a, b)
    and (b, a) separately would double the work and measure nothing new."""
    assert round_robin(5, 2) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    assert len(round_robin(5, 4)) == 5
    assert len(round_robin(4, 4)) == 1


def test_too_few_agents_for_the_seats_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        round_robin(2, 4)
