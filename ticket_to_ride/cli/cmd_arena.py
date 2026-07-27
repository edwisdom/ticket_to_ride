"""`ttr arena` and `ttr leaderboard`.

`arena` plays and records; `leaderboard` refits and reports. They are separate commands
because the split is real: play accumulates into the store over months, and a rating is a
*view* of everything in it -- recomputed in full every time, never updated in place.
"""

from __future__ import annotations

import json
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from ticket_to_ride.eval.arena import (
    RustMissingError,
    agent_row,
    parse_spec,
    round_robin,
    run_match,
    run_round_robin,
)
from ticket_to_ride.eval.leaderboard import Report, fit
from ticket_to_ride.eval.stats import elo_to_score, sprt
from ticket_to_ride.eval.store import Store

DEFAULT_DB = "runs/results.db"

#: An even seat split. A CI straddling it means no first-player effect was resolvable.
EVEN = 0.5

#: SPRT here is a head-to-head decision, so it needs exactly two seats.
HEAD_TO_HEAD = 2


def _specs(agents: str, seed: int) -> list:
    return [parse_spec(text.strip(), seed) for text in agents.split(",") if text.strip()]


def arena_command(
    agents: Annotated[
        str, typer.Option("--agents", help="Comma-separated specs, e.g. 'h0,h1,h2,h3,h4'.")
    ] = "h0,h1,h2,h3,h4",
    map_name: Annotated[str, typer.Option("--map")] = "usa",
    n_players: Annotated[int, typer.Option("--players")] = 2,
    blocks: Annotated[
        int, typer.Option("--blocks", help="Seed blocks per lineup; each is N rotations.")
    ] = 500,
    seed_root: Annotated[
        int, typer.Option("--seed-root", help="Block seeds are seed-root + i.")
    ] = 0,
    database: Annotated[str, typer.Option("--db")] = DEFAULT_DB,
    threads: Annotated[int, typer.Option("--threads", help="0 = performance cores.")] = 0,
    agent_seed: Annotated[int, typer.Option("--agent-seed")] = 0,
    sprt_pair: Annotated[
        str | None,
        typer.Option("--sprt", help="'challenger,incumbent': stop early once decided."),
    ] = None,
    elo1: Annotated[float, typer.Option("--elo1", help="SPRT alternative, in Elo.")] = 20.0,
    batch: Annotated[int, typer.Option("--sprt-batch", help="Blocks between SPRT checks.")] = 50,
) -> None:
    """Play a round robin and record it.

    Every rotation of every seed block is played, so both seatings are measured rather than
    one being mirrored -- the flaw in the published prior art, where every off-diagonal pair
    summed to exactly 1.00 and first-player advantage was confounded into every cell.
    """
    console = Console()
    specs = _specs(agents, agent_seed)
    if len(specs) < n_players:
        raise typer.BadParameter(f"{len(specs)} agents cannot fill {n_players} seats")

    try:
        with Store(database) as store:
            if sprt_pair:
                _run_sprt(
                    console,
                    store,
                    map_name=map_name,
                    n_players=n_players,
                    pair=sprt_pair,
                    agent_seed=agent_seed,
                    blocks=blocks,
                    seed_root=seed_root,
                    threads=threads,
                    elo1=elo1,
                    batch=batch,
                    database=database,
                )
                return
            summaries = run_round_robin(
                store,
                map_name,
                n_players,
                specs,
                seed_root=seed_root,
                blocks=blocks,
                threads=threads,
            )
    except RustMissingError as exc:
        raise typer.BadParameter(str(exc)) from exc

    games = sum(s.games for s in summaries)
    engine = sum(s.seconds for s in summaries)
    console.print(
        f"[green]{games:,} games[/] over {len(summaries)} lineups "
        f"({round_robin(len(specs), n_players).__len__()} distinct groups), "
        f"{engine:.2f}s in the engine -> [cyan]{database}[/]"
    )
    console.print("Run [bold]ttr leaderboard[/] to refit ratings over everything recorded.")


def _run_sprt(
    console: Console,
    store: Store,
    *,
    map_name: str,
    n_players: int,
    pair: str,
    agent_seed: int,
    blocks: int,
    seed_root: int,
    threads: int,
    elo1: float,
    batch: int,
    database: str,
) -> None:
    """Play `challenger` against `incumbent` until the sequential test decides.

    **Checked on block boundaries only.** Half a block is one seating, and stopping there
    would report exactly the mirrored result the block structure exists to prevent.
    """
    names = [p.strip() for p in pair.split(",")]
    if len(names) != HEAD_TO_HEAD or n_players != HEAD_TO_HEAD:
        raise typer.BadParameter("--sprt takes 'challenger,incumbent' and needs --players 2")
    specs = [parse_spec(n, agent_seed) for n in names]

    challenger = store.agent_id(agent_row(specs[0]))
    scores: list[float] = []
    done = 0
    while done < blocks:
        step = min(batch, blocks - done)
        summary = run_match(
            store,
            map_name,
            n_players,
            specs,
            (0, 1),
            seed_root=seed_root + done,
            blocks=step,
            threads=threads,
        )
        scores.extend(_block_scores(store, summary.match_id, challenger))
        done += step
        result = sprt(np.array(scores), elo1=elo1)
        console.print(
            f"  {done:>5} blocks  LLR {result.llr:+7.2f}  "
            f"bounds [{result.lower:.2f}, {result.upper:.2f}]  {result.verdict}"
        )
        if result.decided:
            break

    verdict = sprt(np.array(scores), elo1=elo1)
    observed = float(np.mean(scores)) if scores else float("nan")
    message = {
        "h1": f"[green]{names[0]} is better than {names[1]} by at least {elo1:+.0f} Elo[/]",
        "h0": f"[yellow]{names[0]} is not {elo1:+.0f} Elo better than {names[1]}[/]",
        "continue": f"[yellow]undecided after {done} blocks[/]",
    }[verdict.verdict]
    console.print(f"{message}  (observed score {observed:.3f}) -> [cyan]{database}[/]")


def _block_scores(store: Store, match_id: int, challenger: int) -> list[float]:
    """Per-block mean score for `challenger` in one match, in block order.

    Read back through `agent_id` rather than reconstructed from the seat and the rotation.
    Both would work, and only one of them stays correct if the arena ever returns games in
    a different order -- and a silently transposed score here would make SPRT confidently
    decide the wrong way round.
    """
    rows = store.db.execute(
        "SELECT g.block_seed AS block, g.game_id, s.rank, s.agent_id FROM seat s"
        " JOIN game g ON g.game_id = s.game_id WHERE g.match_id = ?"
        " ORDER BY g.block_seed, g.game_id, s.seat",
        (match_id,),
    ).fetchall()
    per_game: dict[int, list] = {}
    for row in rows:
        per_game.setdefault(int(row["game_id"]), []).append(row)

    per_block: dict[int, list[float]] = {}
    for seats in per_game.values():
        best = min(int(r["rank"]) for r in seats)
        winners = [r for r in seats if int(r["rank"]) == best]
        share = (
            (1.0 / len(winners)) if any(int(r["agent_id"]) == challenger for r in winners) else 0.0
        )
        per_block.setdefault(int(seats[0]["block"]), []).append(share)
    return [float(np.mean(v)) for _, v in sorted(per_block.items())]


def leaderboard_command(
    database: Annotated[str, typer.Option("--db")] = DEFAULT_DB,
    map_name: Annotated[str, typer.Option("--map")] = "usa",
    n_players: Annotated[int, typer.Option("--players")] = 2,
    resamples: Annotated[
        int, typer.Option("--resamples", help="Bootstrap resamples over blocks.")
    ] = 1000,
    output: Annotated[str, typer.Option("--format", help="table | json")] = "table",
    matrix: Annotated[bool, typer.Option("--matrix/--no-matrix")] = True,
    persist: Annotated[bool, typer.Option("--persist/--no-persist")] = True,
) -> None:
    """Refit ratings over every game recorded for a configuration.

    A full refit, always. Bradley-Terry's sufficient statistic is the pairwise win-count
    matrix, so the cost is set by the number of *agents*, not the number of games -- an
    incremental rating would be order-dependent and would buy nothing.
    """
    console = Console()
    with Store(database) as store:
        config = next(
            (
                c
                for c in store.configs()
                if c["map_name"] == map_name and int(c["n_players"]) == n_players
            ),
            None,
        )
        if config is None:
            raise typer.BadParameter(
                f"no games recorded for {map_name} {n_players}P in {database}; run `ttr arena`"
            )
        report = fit(store, int(config["config_id"]), resamples=resamples, persist=persist)

    if output == "json":
        typer.echo(
            json.dumps(
                {
                    "map": report.map_name,
                    "players": report.n_players,
                    "games": report.n_games,
                    "blocks": report.n_blocks,
                    "cycle_fraction": report.cycle_fraction,
                    "seat_win_rate": [
                        {"seat": i, "mean": m, "lo": lo, "hi": hi}
                        for i, (m, lo, hi) in enumerate(report.seat_score)
                    ],
                    "standings": [
                        {
                            "spec": s.spec,
                            "elo": round(s.elo, 1),
                            "lo": round(s.lo, 1),
                            "hi": round(s.hi, 1),
                            "games": s.games,
                            "score": round(s.score, 4),
                            "behaviour_hash": s.behaviour_hash,
                        }
                        for s in report.standings
                    ],
                },
                indent=2,
            )
        )
        return

    table = Table(
        title=f"{report.map_name} {report.n_players}P -- {report.n_games:,} games in "
        f"{report.n_blocks:,} seed blocks, anchored on h3"
    )
    table.add_column("agent", style="cyan")
    table.add_column("elo", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("games", justify="right")
    table.add_column("score", justify="right")
    table.add_column("behaviour", style="dim")
    for standing in report.standings:
        table.add_row(
            standing.spec,
            f"{standing.elo:+.0f}",
            f"[{standing.lo:+.0f}, {standing.hi:+.0f}]",
            f"{standing.games:,}",
            f"{standing.score:.3f}",
            standing.behaviour_hash[:12],
        )
    console.print(table)

    if matrix and len(report.standings) > 1:
        _print_matrix(console, report)

    seats = ", ".join(
        f"seat {i} {m:.3f} [{lo:.3f}, {hi:.3f}]" for i, (m, lo, hi) in enumerate(report.seat_score)
    )
    flat = all(lo <= EVEN <= hi for _, lo, hi in report.seat_score)
    console.print(
        f"seat win rate: {seats}  "
        + (
            "[green](flat within CI)[/]"
            if flat
            else "[yellow](a real first-player effect, not a harness bug -- rotation "
            "already removes it from the agent ratings)[/]"
        )
    )
    console.print(
        f"cycle_fraction: {report.cycle_fraction:.2f}  "
        + (
            "[green](a clean ladder)[/]"
            if report.cycle_fraction == 0
            else "[yellow](non-transitive)[/]"
        )
    )


def _print_matrix(console: Console, report: Report) -> None:
    """The win-rate matrix, both seatings measured.

    Kept next to the ratings because Elo compresses a matrix to a vector and hides
    rock-paper-scissors, which is exactly what a league needs to show.
    """
    standings = report.standings
    order = {s.agent_id: i for i, s in enumerate(standings)}
    index = sorted(order, key=lambda a: order[a])
    rate = report.win_rate
    # `win_rate` is indexed by the fit's agent order, which is agent_id order; the table is
    # printed in Elo order. Conflating the two silently transposes the matrix.
    lookup = {a: i for i, a in enumerate(sorted(order))}

    table = Table(title="win rate (row vs column), both seatings measured")
    table.add_column("", style="cyan")
    for a in index:
        table.add_column(next(s.spec for s in standings if s.agent_id == a), justify="right")
    for a in index:
        cells = []
        for b in index:
            value = rate[lookup[a], lookup[b]]
            cells.append("--" if np.isnan(value) else f"{value:.3f}")
        table.add_row(next(s.spec for s in standings if s.agent_id == a), *cells)
    console.print(table)
    console.print(
        f"[dim]expected score at the fitted gap: "
        f"{elo_to_score(standings[0].elo - standings[-1].elo):.3f} "
        f"({standings[0].spec} vs {standings[-1].spec})[/]"
    )
