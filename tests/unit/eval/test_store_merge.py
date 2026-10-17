"""Folding one results store into another: how cloud jobs' arenas come home (Phase 3.3)."""

from __future__ import annotations

from pathlib import Path

from ticket_to_ride.cli.cmd_cloud import same_games
from ticket_to_ride.eval.arena import agent_row, parse_spec, run_round_robin
from ticket_to_ride.eval.store import Store


def an_arena(path: Path, agents: str, *, seed_root: int, map_name: str = "mini") -> None:
    specs = [parse_spec(text) for text in agents.split(",")]
    with Store(path) as store:
        run_round_robin(store, map_name, 2, specs, seed_root=seed_root, blocks=3, threads=1)


def test_a_merged_store_holds_the_same_games(tmp_path: Path) -> None:
    an_arena(tmp_path / "job.db", "h0,h1,h3", seed_root=11)
    with Store(tmp_path / "local.db") as local, Store(tmp_path / "job.db") as job:
        assert local.merge(job) == 3
        assert local.outcomes() == job.outcomes()
        assert local.counts()["game"] == job.counts()["game"] == 3 * 3 * 2


def test_agents_and_configs_are_matched_by_content_not_by_id(tmp_path: Path) -> None:
    """The local store already knows these agents, under other ids: the merged games must
    count towards the same players, not mint new ones."""
    an_arena(tmp_path / "job.db", "h0,h1", seed_root=5)
    an_arena(tmp_path / "local.db", "h1,h3,h0", seed_root=900)
    with Store(tmp_path / "local.db") as local, Store(tmp_path / "job.db") as job:
        agents_before = local.counts()["agent"]
        configs_before = local.counts()["config"]
        h0 = local.agent_id(agent_row(parse_spec("h0")))
        assert h0 != job.agent_id(agent_row(parse_spec("h0")))
        local.merge(job)
        assert local.counts()["agent"] == agents_before
        assert local.counts()["config"] == configs_before
        assert set(job.outcomes()) <= set(local.outcomes())


def test_same_games_finds_the_first_difference(tmp_path: Path) -> None:
    an_arena(tmp_path / "a.db", "h0,h1", seed_root=3)
    an_arena(tmp_path / "b.db", "h0,h1", seed_root=3)
    an_arena(tmp_path / "c.db", "h0,h1", seed_root=4)
    assert same_games(tmp_path / "a.db", tmp_path / "b.db") == (6, None)
    games, difference = same_games(tmp_path / "a.db", tmp_path / "c.db")
    assert games == 6
    assert difference is not None
