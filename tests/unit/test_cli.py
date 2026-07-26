"""The `ttr` command line.

Also the place the engine's throughput floor is asserted, since `ttr bench` is what
produces the number and Phase 2's ">=50x" target needs a baseline that is measured rather
than remembered.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ticket_to_ride.cli.app import app
from ticket_to_ride.cli.cmd_bench import random_playouts
from ticket_to_ride.cli.cmd_map import summary
from ticket_to_ride.data.board import MINI, USA

runner = CliRunner()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "map", "bench"):
        assert command in result.stdout


def test_version_still_works_as_a_subcommand() -> None:
    """A Typer app with one command collapses; the empty callback is what prevents it."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_no_arguments_shows_help_rather_than_failing_silently() -> None:
    result = runner.invoke(app, [])
    assert "Commands" in result.stdout


# ---------------------------------------------------------------------------
# ttr map
# ---------------------------------------------------------------------------


def test_map_prints_a_table() -> None:
    result = runner.invoke(app, ["map"])
    assert result.exit_code == 0
    assert USA.data_hash in result.stdout
    assert "Denver" in result.stdout


def test_map_json_matches_the_board() -> None:
    result = runner.invoke(app, ["map", "--map", "mini", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["name"] == "mini"
    assert data["data_hash"] == MINI.data_hash
    assert data["segments"] == MINI.n_segments
    assert data["connected"] is True


def test_map_all_covers_every_board() -> None:
    result = runner.invoke(app, ["map", "--all", "--format", "json"])
    assert result.exit_code == 0
    assert {row["name"] for row in json.loads(result.stdout)} == {"usa", "mini"}


def test_map_rejects_an_unknown_board() -> None:
    result = runner.invoke(app, ["map", "--map", "europe"])
    assert result.exit_code != 0


def test_map_says_ascii_is_not_ready_yet() -> None:
    result = runner.invoke(app, ["map", "--format", "ascii"])
    assert result.exit_code != 0
    assert "Phase 4" in str(result.exception) + result.output


def test_summary_reports_the_invariants_the_engine_is_sized_against() -> None:
    data = summary(USA)
    assert (data["cities"], data["pairs"], data["segments"], data["spaces"]) == (36, 78, 100, 309)
    assert data["double_routes"] == 22
    assert data["action_space"] == 915
    assert data["hubs"] == ["Denver", "Helena", "Pittsburgh"]
    assert data["gray_segments"] == 44
    per_color = data["segments_per_color"]
    assert isinstance(per_color, dict)
    assert set(per_color.values()) == {7}


# ---------------------------------------------------------------------------
# ttr bench
# ---------------------------------------------------------------------------


def test_bench_runs_and_reports_both_numbers() -> None:
    result = runner.invoke(app, ["bench", "--games", "3", "--format", "json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert {row["map"] for row in rows} == {"usa", "mini"}
    for row in rows:
        assert row["games_per_second"] > 0
        assert row["microseconds_per_step"] > 0


def test_bench_all_covers_every_configuration() -> None:
    result = runner.invoke(app, ["bench", "--suite", "all", "--games", "1", "--format", "json"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)) == 4 + 3  # usa 2-5P, mini 2-4P


def test_bench_rejects_an_unknown_suite() -> None:
    result = runner.invoke(app, ["bench", "--suite", "nonsense"])
    assert result.exit_code != 0


def test_bench_table_output() -> None:
    result = runner.invoke(app, ["bench", "--games", "2"])
    assert result.exit_code == 0
    assert "us/step" in result.stdout


# ---------------------------------------------------------------------------
# The throughput floor
# ---------------------------------------------------------------------------


@pytest.mark.bench
def test_engine_throughput_floor() -> None:
    """The Phase 1 exit criterion, measured rather than assumed.

    PLAN.md §14 asks for >=2000 random playouts/s/core. TTR-mini clears it; the USA map
    does not, at ~1000 full games/s (~6 us/step) in pure CPython. Profiling shows no
    remaining hotspot -- claim legality is about a third and the rest is spread across a
    dozen small functions -- so the gap is what the Rust core in Phase 2 exists to close.
    The floors here are set below the measured values so a real regression fails while
    ordinary machine-to-machine variation does not.
    """
    usa = random_playouts("usa", 2, 200)
    mini = random_playouts("mini", 2, 200)
    assert mini.games_per_second > 2000, f"mini: {mini.games_per_second:.0f} games/s"
    assert usa.games_per_second > 600, f"usa 2P: {usa.games_per_second:.0f} games/s"
    assert usa.microseconds_per_step < 12, f"usa 2P: {usa.microseconds_per_step:.1f} us/step"
