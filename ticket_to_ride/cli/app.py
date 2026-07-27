"""`ttr` command line entry point.

Commands are added as their phases land. Keeping this file free of heavy imports is
deliberate: `ttr --help` should be instant, and nothing here may pull in torch.
"""

from __future__ import annotations

from importlib.metadata import version as _version

import typer

from ticket_to_ride.cli.cmd_arena import arena_command, leaderboard_command
from ticket_to_ride.cli.cmd_bench import bench_command
from ticket_to_ride.cli.cmd_map import map_command

app = typer.Typer(
    name="ttr",
    help="Ticket to Ride engine, agents, and self-play RL.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Forces Typer into multi-command mode.

    Without a callback, a Typer app holding exactly one command collapses into a
    single-command app and `ttr version` fails as an unexpected argument. Removing this
    would silently break every subcommand the moment the app is down to one.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(_version("ticket_to_ride"))


app.command("map")(map_command)
app.command("bench")(bench_command)
app.command("arena")(arena_command)
app.command("leaderboard")(leaderboard_command)


def main() -> None:
    app()
