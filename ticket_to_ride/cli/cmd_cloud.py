"""`ttr cloud`: run `ttr` commands on Modal, and bring the results home.

PLAN.md §14.1, Phase 3.3. Needs the `cloud` dependency group (`uv sync --group cloud`) and
Modal credentials (`modal token new`, which keeps them in `~/.modal.toml` -- never in this
repository, never in the image). The app is `infra/modal_app.py`, run ephemerally per
command: there is nothing to deploy and nothing left running afterwards.

    ttr cloud bench --suite engine --games 2000
    ttr cloud arena --agents h1,h2,h3 --blocks 100 --check-local
    ttr cloud run --cpu 16 -- arena --agents h3,h4 --blocks 2000
    ttr cloud pull

Every job writes its own directory on the `ttr-runs` Volume. A command pulls its own job
when it finishes; `ttr cloud pull` fetches any others. Pulled jobs land in `runs/cloud/`,
and an arena's `results.db` is merged into the local store (`Store.merge`), once.

`modal` is imported inside the commands, never at module level, so `ttr --help` stays
instant and everything else works without the `cloud` group installed.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import secrets
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import typer
from rich.console import Console

from ticket_to_ride.cli.cmd_arena import DEFAULT_DB
from ticket_to_ride.eval.store import Store

APP_FILE = Path(__file__).resolve().parents[2] / "infra" / "modal_app.py"
VOLUME = "ttr-runs"
PULL_DIR = "runs/cloud"
#: Written next to a pulled job once every file is down, and once its arena is merged.
PULLED, MERGED = ".pulled", ".merged"

PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}

cloud_app = typer.Typer(
    name="cloud",
    help="Run ttr commands on Modal and pull the results (needs `uv sync --group cloud`).",
    no_args_is_help=True,
)

Cpu = Annotated[float | None, typer.Option("--cpu", help="Physical cores to reserve (default 8).")]
Gpu = Annotated[str | None, typer.Option("--gpu", help="A Modal GPU type, e.g. 'T4'.")]
Into = Annotated[str, typer.Option("--into", help="Where pulled jobs land.")]
Database = Annotated[str, typer.Option("--db", help="The local store arenas merge into.")]


def _modal() -> ModuleType:
    try:
        import modal  # noqa: PLC0415 -- optional, and slow to import
    except ImportError:
        typer.echo("ttr cloud needs the cloud dependency group: uv sync --group cloud", err=True)
        raise typer.Exit(2) from None
    return modal


def _load_app() -> ModuleType:
    """`infra/modal_app.py`, which lives outside the package, imported by path."""
    spec = importlib.util.spec_from_file_location("modal_app", APP_FILE)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(APP_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_app"] = module
    spec.loader.exec_module(module)
    return module


def job_id(command: str) -> str:
    """Sortable by time, labelled by command, unique: `20261016-201530-arena-3f9a`."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{command}-{secrets.token_hex(2)}"


def submit(argv: list[str], *, cpu: float | None, gpu: str | None) -> dict[str, Any]:
    """Run `ttr <argv>` as one job on Modal and return its record."""
    modal = _modal()
    module = _load_app()
    job = job_id(argv[0] if argv else "run")
    with modal.enable_output(), module.app.run():
        function = module.run
        if cpu is not None or gpu is not None:
            function = function.with_options(cpu=(cpu, cpu) if cpu else None, gpu=gpu)
        return function.remote(argv, job)


def _report(console: Console, result: dict[str, Any]) -> None:
    if result["stdout"]:
        console.print(result["stdout"], end="", markup=False, highlight=False)
    if result["stderr"]:
        console.print(result["stderr"], end="", markup=False, highlight=False, style="red")
    where = result["where"]
    console.print(
        f"[cyan]{result['job']}[/]: exit {result['returncode']} in {result['seconds']:.1f}s on "
        f"{where['cpu_model']} -- {where['process_cpu_count']} CPUs visible, quota "
        f"{where['cgroup_cpu_max']!r}, engine threads {where['performance_threads']}"
    )


def pull_jobs(into: Path, database: Path, only: str | None = None) -> list[str]:
    """Download jobs not yet pulled into `into/<job>/`, then merge any unmerged arena.

    Both steps leave a marker when they finish, so an interrupted pull resumes and nothing
    is merged twice.
    """
    modal = _modal()
    from modal.types import FileEntryType  # noqa: PLC0415 -- optional dependency

    volume = modal.Volume.from_name(VOLUME)
    if only is not None:
        jobs = [only]
    else:
        jobs = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    fetched = []
    for job in jobs:
        local = into / job
        if not (local / PULLED).exists():
            for entry in volume.listdir(job, recursive=True):
                if entry.type != FileEntryType.FILE:
                    continue
                target = into / entry.path.strip("/")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as out:
                    for chunk in volume.read_file(entry.path):
                        out.write(chunk)
            local.mkdir(parents=True, exist_ok=True)
            (local / PULLED).touch()
            fetched.append(job)
        results = local / "results.db"
        if results.exists() and not (local / MERGED).exists():
            with Store(database) as store, Store(results) as theirs:
                store.merge(theirs)
            (local / MERGED).touch()
    return fetched


def same_games(ours: Path, theirs: Path) -> tuple[int, str | None]:
    """How many games two stores hold, and the first difference between them, if any."""
    with Store(ours) as a, Store(theirs) as b:
        rows_a, rows_b = a.outcomes(), b.outcomes()
    games = len({row[:6] for row in rows_a})
    if rows_a == rows_b:
        return games, None
    for row_a, row_b in zip(rows_a, rows_b, strict=False):
        if row_a != row_b:
            return games, f"{row_a} != {row_b}"
    return games, f"{len(rows_a)} seat rows against {len(rows_b)}"


@cloud_app.command(context_settings=PASSTHROUGH)
def bench(ctx: typer.Context, cpu: Cpu = None, gpu: Gpu = None, into: Into = PULL_DIR) -> None:
    """`ttr bench …` on Modal. Arguments after the options go to `ttr bench`."""
    console = Console()
    result = submit(["bench", *ctx.args], cpu=cpu, gpu=gpu)
    _report(console, result)
    pull_jobs(Path(into), Path(DEFAULT_DB), only=result["job"])
    if result["returncode"] != 0:
        raise typer.Exit(1)


@cloud_app.command(context_settings=PASSTHROUGH)
def arena(
    ctx: typer.Context,
    cpu: Cpu = None,
    into: Into = PULL_DIR,
    database: Database = DEFAULT_DB,
    check_local: Annotated[
        bool,
        typer.Option(
            "--check-local", help="Replay the arena here and require game-for-game equality."
        ),
    ] = False,
) -> None:
    """`ttr arena …` on Modal; the job's games are merged into the local store.

    `--check-local` is PLAN.md's cross-platform check: the same arena run locally (into a
    scratch database, so no game is counted twice) must reproduce every game exactly --
    final state hash, scores and ranks, seat by seat.
    """
    if "--db" in ctx.args:
        raise typer.BadParameter("a cloud arena writes its own results.db; it is merged on pull")
    console = Console()
    argv = ["arena", *ctx.args, "--db", "results.db"]
    result = submit(argv, cpu=cpu, gpu=None)
    _report(console, result)
    if result["returncode"] != 0:
        raise typer.Exit(1)
    pull_jobs(Path(into), Path(database), only=result["job"])
    if not check_local:
        return

    from ticket_to_ride.cli.app import app  # noqa: PLC0415 -- the app imports this module

    cloud_db = Path(into) / result["job"] / "results.db"
    with tempfile.TemporaryDirectory() as scratch:
        local_db = Path(scratch) / "local.db"
        typer.main.get_command(app)([*argv[:-1], str(local_db)], standalone_mode=False)
        games, difference = same_games(cloud_db, local_db)
    if difference is not None:
        console.print(f"[red]cloud and local games differ[/]: {difference}")
        raise typer.Exit(1)
    console.print(f"[green]{games:,} games identical to a local run[/], seat by seat")


@cloud_app.command(context_settings=PASSTHROUGH)
def run(ctx: typer.Context, cpu: Cpu = None, gpu: Gpu = None, into: Into = PULL_DIR) -> None:
    """Any `ttr` command on Modal: `ttr cloud run -- <command> <args>`."""
    if not ctx.args:
        raise typer.BadParameter("give a ttr command after --")
    console = Console()
    result = submit(list(ctx.args), cpu=cpu, gpu=gpu)
    _report(console, result)
    pull_jobs(Path(into), Path(DEFAULT_DB), only=result["job"])
    if result["returncode"] != 0:
        raise typer.Exit(1)


@cloud_app.command()
def pull(into: Into = PULL_DIR, database: Database = DEFAULT_DB) -> None:
    """Fetch every job not yet pulled; merge their arenas into the local store."""
    fetched = pull_jobs(Path(into), Path(database))
    typer.echo(
        f"{len(fetched)} new job(s) in {into}" + (f": {', '.join(fetched)}" if fetched else "")
    )
