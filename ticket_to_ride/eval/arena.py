"""Running matches and putting the results in the store.

Thin on purpose. Every game is played inside Rust (`ttr_rust.run_arena`), every statistic
lives in `stats.py`, every write lives in `store.py`, and every agent spec is parsed by
`agents/registry.py`; this module schedules and translates.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from itertools import combinations

import ttr_rust

from ticket_to_ride.agents.registry import AgentSpec
from ticket_to_ride.eval.store import MatchRow, Store


def round_robin(n_agents: int, n_players: int) -> list[tuple[int, ...]]:
    """Lineups covering every distinct group of `n_players` agents, each once.

    Rotations within a lineup are the arena's job, so a lineup is an unordered choice --
    listing `(a, b)` and `(b, a)` separately would double the work and measure nothing new,
    since every rotation of `(a, b)` is played either way.
    """
    if n_agents < n_players:
        raise ValueError(f"{n_agents} agents cannot fill {n_players} seats")
    return list(combinations(range(n_agents), n_players))


def git_sha() -> str | None:
    """Recorded on the match as provenance. Deliberately **not** part of agent identity."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True)
class MatchSummary:
    """What one lineup produced."""

    lineup: tuple[int, ...]
    match_id: int
    games: int
    seconds: float


def run_match(
    store: Store,
    map_name: str,
    n_players: int,
    specs: list[AgentSpec],
    lineup: tuple[int, ...],
    *,
    seed_root: int = 0,
    blocks: int = 100,
    threads: int = 0,
    agent_ids: dict[int, int] | None = None,
) -> MatchSummary:
    """Play one lineup and record it.

    Every rotation of every block is played inside Rust and returned as two columnar
    tables; the only per-row Python here is the translation into the store.
    """
    game = ttr_rust.Game(map_name, n_players)
    config_id = store.config_id(
        map_name,
        n_players,
        game.rules_hash,
        game.data_hash,
        ttr_rust.contract_version(),
    )
    ids = agent_ids if agent_ids is not None else {}
    for index in lineup:
        if index not in ids:
            ids[index] = store.agent_id(specs[index].identity())

    # Only the agents in this lineup are handed to Rust, so the returned `agent` column
    # indexes *this* list; `local` maps it back to the caller's numbering.
    local = list(lineup)
    payload = [(specs[i].family, specs[i].overrides or None, specs[i].seed) for i in local]
    games, seats = ttr_rust.run_arena(
        map_name,
        n_players,
        payload,
        list(range(n_players)),
        seed_root,
        blocks,
        threads,
    )

    match = MatchRow(
        config_id=config_id,
        seed_root=seed_root,
        n_blocks=blocks,
        lineup=",".join(specs[i].text for i in local),
        seconds=games.seconds,
        git_sha=git_sha(),
        ttr_version=_version(),
    )
    # `final_hash` goes in as hex: SQLite INTEGER is signed 64-bit and a u64 state hash
    # above 2^63 raises OverflowError on insert. See the schema comment.
    game_rows = [
        (seed, rot, turns, f"{h:016x}")
        for seed, rot, turns, h in zip(
            games.block_seed, games.rotation, games.turns, games.final_hash, strict=True
        )
    ]
    columns = (
        seats.game,
        seats.seat,
        [ids[local[a]] for a in seats.agent],
        seats.score,
        seats.ret,
        seats.rank,
        seats.won,
        seats.tickets_kept,
        seats.tickets_made,
        seats.ticket_points,
        seats.routes_claimed,
        seats.trains_left,
        seats.cards_left,
        seats.longest_trail,
        seats.longest_bonus,
        seats.n_claim,
        seats.n_claim_wild,
        seats.n_claim_double,
        seats.n_draw_faceup,
        seats.n_draw_blind,
        seats.n_draw_tickets,
        seats.n_keep,
        seats.n_keep_extra,
        seats.n_pass,
    )
    match_id = store.record_match(match, game_rows, zip(*columns, strict=True))
    return MatchSummary(lineup, match_id, len(game_rows), games.seconds)


def run_round_robin(
    store: Store,
    map_name: str,
    n_players: int,
    specs: list[AgentSpec],
    *,
    seed_root: int = 0,
    blocks: int = 100,
    threads: int = 0,
) -> list[MatchSummary]:
    """Every lineup of `n_players` drawn from `specs`, each for `blocks` seed blocks.

    Agent ids are resolved once and shared, so the behaviour probe -- a couple of hundred
    milliseconds of real games -- runs once per agent rather than once per lineup.
    """
    agent_ids: dict[int, int] = {}
    return [
        run_match(
            store,
            map_name,
            n_players,
            specs,
            lineup,
            seed_root=seed_root,
            blocks=blocks,
            threads=threads,
            agent_ids=agent_ids,
        )
        for lineup in round_robin(len(specs), n_players)
    ]


def _version() -> str | None:
    from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
    from importlib.metadata import version as _v  # noqa: PLC0415

    try:
        return _v("ticket_to_ride")
    except PackageNotFoundError:
        return None
