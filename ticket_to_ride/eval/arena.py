"""Running matches and putting the results in the store.

Thin on purpose. Every game is played inside Rust (`ttr_rust.run_arena`), every statistic
lives in `stats.py`, and every write lives in `store.py`; this module schedules and
translates. The one thing it owns is the **agent spec** -- the short string that names an
agent identically on a command line, in a config file and in a leaderboard row.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tomllib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from ticket_to_ride.eval.store import AgentRow, MatchRow, Store

#: Tiers `ttr_rust` knows. Neural agents arrive here in Phase 6 as `ppo:runs/x/best.pt`,
#: resolved by a factory that imports torch lazily so the arena still starts in ~50 ms.
TIERS = ("h0", "h1", "h2", "h3", "h4", "random")

#: The permanent Elo anchor (PLAN.md §11). Its identity is held by
#: `ttr_rust.behaviour_hash`, not by this string.
ANCHOR_FAMILY = "h3"


class RustMissingError(RuntimeError):
    """The Rust extension is not built."""

    def __init__(self) -> None:
        super().__init__("ttr_rust is not built; run `make rust`")


def rust_available() -> bool:
    return importlib.util.find_spec("ttr_rust") is not None


@dataclass(frozen=True)
class Spec:
    """A resolved agent specification."""

    #: The spec string exactly as written, so a leaderboard row round-trips to a re-run.
    text: str
    family: str
    overrides: dict[str, Any]
    seed: int

    @property
    def is_anchor(self) -> bool:
        return self.family == ANCHOR_FAMILY and not self.overrides


def parse_spec(text: str, seed: int = 0) -> Spec:
    """`h3`, or `h3@configs/agents/h3_v2.toml` for a tuned variant.

    A tuned agent is written as a **different spec**, never by editing the anchor's
    constants in place. That is the whole mechanism by which ratings accumulate: `h3` stays
    the zero of the scale and `h3@...` is a new competitor rated against it.
    """
    name, _, path = text.partition("@")
    if name == "random":
        name = "h0"
    if name not in TIERS:
        raise ValueError(f"unknown agent {name!r}; available: {sorted(TIERS)}")
    overrides: dict[str, Any] = {}
    if path:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        overrides = dict(data.get("params", data))
    return Spec(text=text, family=name, overrides=overrides, seed=seed)


def agent_row(spec: Spec) -> AgentRow:
    """The store identity for a spec: what it plays, not where it came from."""
    import ttr_rust  # noqa: PLC0415 - optional; built by `make rust`, not `uv sync`

    overrides = spec.overrides or None
    return AgentRow(
        spec=spec.text,
        family=spec.family,
        params_hash=ttr_rust.params_hash(overrides),
        behaviour_hash=ttr_rust.behaviour_hash(spec.family, overrides),
    )


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
    specs: list[Spec],
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
    if not rust_available():
        raise RustMissingError
    import ttr_rust  # noqa: PLC0415

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
            ids[index] = store.agent_id(agent_row(specs[index]))

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
    specs: list[Spec],
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
