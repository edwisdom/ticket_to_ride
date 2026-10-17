"""The `ttr` command line.

Also the place the engine's throughput floor is asserted, since `ttr bench` is what
produces the number and Phase 2's ">=50x" target needs a baseline that is measured rather
than remembered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ticket_to_ride.cli.app import app
from ticket_to_ride.cli.cmd_cloud import job_id
from ticket_to_ride.cli.cmd_map import summary
from ticket_to_ride.data.board import MINI, USA

runner = CliRunner()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "map", "bench", "arena", "leaderboard", "cloud"):
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


def test_bench_runs_and_reports_its_numbers() -> None:
    result = runner.invoke(app, ["bench", "--games", "50", "--repeat", "1", "--format", "json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert {row["map"] for row in rows} == {"usa", "mini"}
    for row in rows:
        assert row["games_per_second"] > 0
        assert row["microseconds_per_step"] > 0
        assert row["engine"].startswith("rust-")


def test_bench_all_covers_every_configuration() -> None:
    """Every (map, seats) pair appears, however many figures each is measured with."""
    result = runner.invoke(
        app, ["bench", "--suite", "all", "--games", "5", "--repeat", "1", "--format", "json"]
    )
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    covered = {(row["map"], row["players"]) for row in rows}
    assert covered == {("usa", n) for n in range(2, 6)} | {("mini", n) for n in range(2, 5)}


@pytest.mark.slow
def test_bench_vecenv_reports_every_phase() -> None:
    result = runner.invoke(app, ["bench", "--suite", "vecenv", "--format", "json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert {(r["map"], r["n_envs"]) for r in rows} == {
        (m, n) for m in ("usa", "mini") for n in (1, 64, 1024)
    }
    for row in rows:
        assert row["observe"] > 0
        assert row["env_steps_per_second"] >= row["loop_steps_per_second"] > 0


def test_bench_rejects_an_unknown_suite() -> None:
    result = runner.invoke(app, ["bench", "--suite", "nonsense"])
    assert result.exit_code != 0


def test_bench_table_output() -> None:
    result = runner.invoke(app, ["bench", "--games", "20", "--repeat", "1"])
    assert result.exit_code == 0
    assert "us/step" in result.stdout


# ---------------------------------------------------------------------------
# arena and leaderboard
# ---------------------------------------------------------------------------


def test_a_leaderboard_over_an_empty_store_says_so(tmp_path: Path) -> None:
    """Rather than fitting ratings over nothing and printing an empty, plausible table."""
    result = runner.invoke(app, ["leaderboard", "--db", str(tmp_path / "empty.db")])
    assert result.exit_code != 0
    assert "ttr arena" in result.output


def test_an_arena_with_fewer_agents_than_seats_is_refused(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["arena", "--agents", "h3", "--players", "2", "--db", str(tmp_path / "x.db")]
    )
    assert result.exit_code != 0
    assert "cannot fill" in result.output


def test_sprt_needs_two_agents_and_two_seats(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["arena", "--sprt", "h3", "--players", "2", "--db", str(tmp_path / "x.db")],
    )
    assert result.exit_code != 0
    assert "challenger,incumbent" in result.output


# ---------------------------------------------------------------------------
# ttr cloud -- everything short of calling Modal
# ---------------------------------------------------------------------------


def test_cloud_lists_its_commands_without_importing_modal() -> None:
    result = runner.invoke(app, ["cloud", "--help"])
    assert result.exit_code == 0
    for command in ("bench", "arena", "run", "pull"):
        assert command in result.stdout


def test_cloud_without_the_cloud_group_says_how_to_get_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "modal", None)  # `import modal` now raises ImportError
    result = runner.invoke(app, ["cloud", "pull"])
    assert result.exit_code == 2
    assert "uv sync --group cloud" in result.output


def test_cloud_run_needs_a_command() -> None:
    result = runner.invoke(app, ["cloud", "run"])
    assert result.exit_code != 0


def test_cloud_job_ids_sort_by_time_and_name_their_command() -> None:
    first, second = job_id("arena"), job_id("arena")
    assert first != second
    stamp, command, suffix = first.rsplit("-", 2)[0], first.split("-")[2], first.rsplit("-", 1)[1]
    assert len(stamp) == len("20261016-201530")
    assert command == "arena"
    assert len(suffix) == 4
