"""The results store: SQLite, WAL, append-only.

PLAN.md §11 wants ACID here for a concrete reason -- a crashed 100k-game arena must not
corrupt prior results -- and wants agents content-addressed so ratings accumulate across
months. Both shape the schema below.

## Agents are addressed by what they play, not by where they came from

§11 suggests `(checkpoint_path, config_hash, git_sha)`. **The `git_sha` must not be in the
identity.** Every commit would mint a new agent id and shatter exactly the accumulation the
scheme exists for: a hundred commits that never touch the heuristics would leave a hundred
"different" H3s, each with a slice of the games and none with a usable rating.

So identity is `(family, params_hash, behaviour_hash, checkpoint)`, where the behaviour hash
is the agent's own action sequence over a frozen probe set. Two builds that play identically
*are* the same player; two that play differently are different players even at the same
commit. `git_sha` is recorded on the match, as provenance.

## Ratings are recomputed, not accumulated

There is no incremental Elo and no maintained aggregate table. Bradley-Terry's sufficient
statistic is the pairwise win-count matrix, so however many games pile up they reduce to an
`A x A` table by one grouped query; the fit is then `O(A^2)` per Newton step regardless.
Maintaining a `pair_stat` table would be a cache in front of a query whose cost is a
`GROUP BY` -- measured in `tests/integration/test_store_scale.py` rather than assumed.

Rating rows are **append-only snapshots**. A refit writes a new `rating_run` rather than
overwriting: the leaderboard as of any past date stays recoverable, and the anchor each fit
used is recorded, so a moved anchor shows up in the history instead of retroactively
rewriting it.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Bumped when the schema changes in a way an older database cannot be read as.
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Ratings are never comparable across configurations, so the configuration is a first-class
-- row and everything downstream is scoped by it.
CREATE TABLE IF NOT EXISTS config (
    config_id        INTEGER PRIMARY KEY,
    map_name         TEXT NOT NULL,
    n_players        INTEGER NOT NULL,
    rules_hash       TEXT NOT NULL,
    data_hash        TEXT NOT NULL,
    contract_version INTEGER NOT NULL,
    UNIQUE (map_name, n_players, rules_hash)
);

-- Content-addressed by behaviour. See the module docstring for why git_sha is absent.
CREATE TABLE IF NOT EXISTS agent (
    agent_id       INTEGER PRIMARY KEY,
    spec           TEXT NOT NULL,
    family         TEXT NOT NULL,
    params_hash    TEXT NOT NULL,
    behaviour_hash TEXT NOT NULL,
    checkpoint     TEXT,
    created_utc    TEXT NOT NULL,
    UNIQUE (family, params_hash, behaviour_hash, checkpoint)
);

CREATE TABLE IF NOT EXISTS match (
    match_id     INTEGER PRIMARY KEY,
    config_id    INTEGER NOT NULL REFERENCES config(config_id),
    seed_root    INTEGER NOT NULL,
    n_blocks     INTEGER NOT NULL,
    lineup       TEXT NOT NULL,
    started_utc  TEXT NOT NULL,
    seconds      REAL NOT NULL,
    engine       TEXT NOT NULL,
    git_sha      TEXT,
    ttr_version  TEXT
);

CREATE TABLE IF NOT EXISTS game (
    game_id     INTEGER PRIMARY KEY,
    match_id    INTEGER NOT NULL REFERENCES match(match_id),
    block_seed  INTEGER NOT NULL,
    rotation    INTEGER NOT NULL,
    turns       INTEGER NOT NULL,
    -- The final state_hash, as 16 hex characters. Stored so a re-run is *verifiable* and
    -- not merely repeatable: two runs that agree on ratings but disagree here played
    -- different games and happened to tie.
    --
    -- TEXT, not INTEGER: SQLite's INTEGER is *signed* 64-bit, and a u64 hash above 2^63
    -- raises OverflowError on insert. Storing the two's-complement view would work and
    -- would print as a negative number that matches nothing the engine ever reports.
    final_hash  TEXT NOT NULL,
    UNIQUE (match_id, block_seed, rotation)
);

CREATE TABLE IF NOT EXISTS seat (
    game_id         INTEGER NOT NULL REFERENCES game(game_id),
    seat            INTEGER NOT NULL,
    agent_id        INTEGER NOT NULL REFERENCES agent(agent_id),
    score           INTEGER NOT NULL,
    ret             REAL NOT NULL,
    rank            INTEGER NOT NULL,
    won             INTEGER NOT NULL,
    tickets_kept    INTEGER NOT NULL,
    tickets_made    INTEGER NOT NULL,
    ticket_points   INTEGER NOT NULL,
    routes_claimed  INTEGER NOT NULL,
    trains_left     INTEGER NOT NULL,
    cards_left      INTEGER NOT NULL,
    longest_trail   INTEGER NOT NULL,
    longest_bonus   INTEGER NOT NULL,
    n_claim         INTEGER NOT NULL,
    n_claim_wild    INTEGER NOT NULL,
    n_claim_double  INTEGER NOT NULL,
    n_draw_faceup   INTEGER NOT NULL,
    n_draw_blind    INTEGER NOT NULL,
    n_draw_tickets  INTEGER NOT NULL,
    n_keep          INTEGER NOT NULL,
    n_keep_extra    INTEGER NOT NULL,
    n_pass          INTEGER NOT NULL,
    PRIMARY KEY (game_id, seat)
);

-- Append-only snapshots. A refit adds a run; it never rewrites one.
CREATE TABLE IF NOT EXISTS rating_run (
    rating_run_id   INTEGER PRIMARY KEY,
    config_id       INTEGER NOT NULL REFERENCES config(config_id),
    computed_utc    TEXT NOT NULL,
    anchor_agent_id INTEGER NOT NULL REFERENCES agent(agent_id),
    method          TEXT NOT NULL,
    prior_sd        REAL NOT NULL,
    resamples       INTEGER NOT NULL,
    n_games         INTEGER NOT NULL,
    n_blocks        INTEGER NOT NULL,
    cycle_fraction  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rating (
    rating_run_id INTEGER NOT NULL REFERENCES rating_run(rating_run_id),
    agent_id      INTEGER NOT NULL REFERENCES agent(agent_id),
    elo           REAL NOT NULL,
    lo            REAL NOT NULL,
    hi            REAL NOT NULL,
    games         INTEGER NOT NULL,
    score         REAL NOT NULL,
    PRIMARY KEY (rating_run_id, agent_id)
);

CREATE INDEX IF NOT EXISTS game_by_match ON game(match_id);
CREATE INDEX IF NOT EXISTS seat_by_game ON seat(game_id);
CREATE INDEX IF NOT EXISTS seat_by_agent ON seat(agent_id);
CREATE INDEX IF NOT EXISTS match_by_config ON match(config_id);
CREATE INDEX IF NOT EXISTS rating_by_run ON rating(rating_run_id);
"""

#: `seat` columns in insert order, so the writer and the schema cannot drift apart.
SEAT_COLUMNS = (
    "game_id",
    "seat",
    "agent_id",
    "score",
    "ret",
    "rank",
    "won",
    "tickets_kept",
    "tickets_made",
    "ticket_points",
    "routes_claimed",
    "trains_left",
    "cards_left",
    "longest_trail",
    "longest_bonus",
    "n_claim",
    "n_claim_wild",
    "n_claim_double",
    "n_draw_faceup",
    "n_draw_blind",
    "n_draw_tickets",
    "n_keep",
    "n_keep_extra",
    "n_pass",
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class AgentRow:
    """An agent's content address plus the label humans read."""

    spec: str
    family: str
    params_hash: str
    behaviour_hash: str
    checkpoint: str | None = None


@dataclass
class MatchRow:
    """Provenance for one arena invocation. `git_sha` lives here, not on the agent."""

    config_id: int
    seed_root: int
    n_blocks: int
    lineup: str
    seconds: float
    engine: str = "rust"
    git_sha: str | None = None
    ttr_version: str | None = None
    started_utc: str = field(default_factory=_now)


class Store:
    """A results database. Every write happens here, in the parent process.

    Workers return plain records (PLAN.md §11) -- the arena runs inside Rust and hands back
    columnar tables, so there is exactly one writer and no cross-process locking to reason
    about.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        # WAL for the crash property §11 asks for: a reader never blocks the writer, and a
        # process killed mid-arena leaves the last committed transaction intact rather than
        # a half-written page.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        try:
            self.db.executescript(SCHEMA)
            self._check_version()
        except Exception:
            # Close before propagating. A connection left for the garbage collector is a
            # ResourceWarning at finalization, which under `filterwarnings = ["error"]`
            # surfaces as an unraisable exception in a *different* test than the one that
            # leaked it -- the failure is real, and it points at the wrong place.
            self.db.close()
            raise

    def _check_version(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return
        found = int(row["value"])
        if found != SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} was written by schema version {found}, this build is "
                f"{SCHEMA_VERSION}. Refusing to mix them: the accumulated ratings would be "
                "computed over rows read under the wrong meaning."
            )

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- identities ----------------------------------------------------

    def config_id(
        self,
        map_name: str,
        n_players: int,
        rules_hash: str,
        data_hash: str,
        contract_version: int,
    ) -> int:
        row = self.db.execute(
            "SELECT config_id FROM config WHERE map_name=? AND n_players=? AND rules_hash=?",
            (map_name, n_players, rules_hash),
        ).fetchone()
        if row is not None:
            return int(row["config_id"])
        cur = self.db.execute(
            "INSERT INTO config(map_name, n_players, rules_hash, data_hash, contract_version)"
            " VALUES (?, ?, ?, ?, ?)",
            (map_name, n_players, rules_hash, data_hash, contract_version),
        )
        return int(cur.lastrowid or 0)

    def agent_id(self, agent: AgentRow) -> int:
        """Find or create an agent by its content address.

        Re-registering the same behaviour returns the same id, which is what makes ratings
        accumulate across invocations, across days and across commits.
        """
        row = self.db.execute(
            "SELECT agent_id FROM agent WHERE family=? AND params_hash=? AND behaviour_hash=?"
            " AND checkpoint IS ?",
            (agent.family, agent.params_hash, agent.behaviour_hash, agent.checkpoint),
        ).fetchone()
        if row is not None:
            return int(row["agent_id"])
        cur = self.db.execute(
            "INSERT INTO agent(spec, family, params_hash, behaviour_hash, checkpoint,"
            " created_utc) VALUES (?, ?, ?, ?, ?, ?)",
            (
                agent.spec,
                agent.family,
                agent.params_hash,
                agent.behaviour_hash,
                agent.checkpoint,
                _now(),
            ),
        )
        return int(cur.lastrowid or 0)

    # -- writes --------------------------------------------------------

    def record_match(
        self,
        match: MatchRow,
        games: Sequence[tuple[int, int, int, str]],
        seats: Iterable[Sequence[Any]],
    ) -> int:
        """Insert one match, its games and its seat rows, in a single transaction.

        `games` is `(block_seed, rotation, turns, final_hash_hex)` per game and `seats` yields
        rows matching `SEAT_COLUMNS` with `game_id` replaced by the game's *index* -- the
        real id is assigned here, since it is a database concern the arena knows nothing
        about.

        One transaction, so an interrupted arena leaves either the whole match or none of
        it. A half-written match would be worse than a lost one: it would be counted.
        """
        self.db.execute("BEGIN")
        try:
            cur = self.db.execute(
                "INSERT INTO match(config_id, seed_root, n_blocks, lineup, started_utc,"
                " seconds, engine, git_sha, ttr_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    match.config_id,
                    match.seed_root,
                    match.n_blocks,
                    match.lineup,
                    match.started_utc,
                    match.seconds,
                    match.engine,
                    match.git_sha,
                    match.ttr_version,
                ),
            )
            match_id = int(cur.lastrowid or 0)
            self.db.executemany(
                "INSERT INTO game(match_id, block_seed, rotation, turns, final_hash)"
                " VALUES (?, ?, ?, ?, ?)",
                [(match_id, *g) for g in games],
            )
            first = self.db.execute(
                "SELECT MIN(game_id) AS lo FROM game WHERE match_id=?", (match_id,)
            ).fetchone()["lo"]
            placeholders = ", ".join("?" * len(SEAT_COLUMNS))
            self.db.executemany(
                f"INSERT INTO seat({', '.join(SEAT_COLUMNS)}) VALUES ({placeholders})",
                ((first + row[0], *row[1:]) for row in seats),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return match_id

    def record_ratings(
        self,
        config_id: int,
        anchor_agent_id: int,
        rows: Sequence[tuple[int, float, float, float, int, float]],
        *,
        method: str,
        prior_sd: float,
        resamples: int,
        n_games: int,
        n_blocks: int,
        cycle_fraction: float,
    ) -> int:
        """Append a rating snapshot. Never overwrites an earlier one."""
        self.db.execute("BEGIN")
        try:
            cur = self.db.execute(
                "INSERT INTO rating_run(config_id, computed_utc, anchor_agent_id, method,"
                " prior_sd, resamples, n_games, n_blocks, cycle_fraction)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    config_id,
                    _now(),
                    anchor_agent_id,
                    method,
                    prior_sd,
                    resamples,
                    n_games,
                    n_blocks,
                    cycle_fraction,
                ),
            )
            run_id = int(cur.lastrowid or 0)
            self.db.executemany(
                "INSERT INTO rating(rating_run_id, agent_id, elo, lo, hi, games, score)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(run_id, *row) for row in rows],
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return run_id

    # -- reads ---------------------------------------------------------

    def seat_rows(self, config_id: int) -> list[sqlite3.Row]:
        """Every seat row for a configuration, ordered so games stay contiguous.

        The single `O(N)` step in a refit. Everything after this works on the `A x A`
        sufficient statistic.
        """
        return self.db.execute(
            "SELECT g.game_id, g.block_seed, s.seat, s.agent_id, s.rank, s.ret, s.won"
            " FROM seat s JOIN game g ON g.game_id = s.game_id"
            " JOIN match m ON m.match_id = g.match_id"
            " WHERE m.config_id = ?"
            " ORDER BY g.game_id, s.seat",
            (config_id,),
        ).fetchall()

    def agents_in(self, config_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT DISTINCT a.* FROM agent a JOIN seat s ON s.agent_id = a.agent_id"
            " JOIN game g ON g.game_id = s.game_id JOIN match m ON m.match_id = g.match_id"
            " WHERE m.config_id = ? ORDER BY a.agent_id",
            (config_id,),
        ).fetchall()

    def configs(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM config ORDER BY config_id").fetchall()

    def latest_rating_run(self, config_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM rating_run WHERE config_id=? ORDER BY rating_run_id DESC LIMIT 1",
            (config_id,),
        ).fetchone()

    def ratings_of(self, rating_run_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT r.*, a.spec, a.behaviour_hash FROM rating r"
            " JOIN agent a ON a.agent_id = r.agent_id"
            " WHERE r.rating_run_id = ? ORDER BY r.elo DESC",
            (rating_run_id,),
        ).fetchall()

    def counts(self) -> dict[str, int]:
        return {
            table: int(self.db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in ("config", "agent", "match", "game", "seat", "rating_run", "rating")
        }
