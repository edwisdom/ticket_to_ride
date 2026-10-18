"""`ttr serve`: the game server and the web client (PLAN.md §14.2).

FastAPI and uvicorn are the `web` extra and are imported only when the command runs, so
`ttr --help` stays instant and the rest of the CLI works without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ticket_to_ride.agents.registry import TIERS


def serve_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    database: Annotated[str, typer.Option("--db", help="The play store.")] = "runs/play.db",
    results: Annotated[
        str, typer.Option("--results-db", help="Arena games, rated alongside human ones.")
    ] = "runs/results.db",
    agents: Annotated[
        str, typer.Option("--agents", help="Comma-separated specs the server will seat.")
    ] = ",".join(TIERS),
) -> None:
    """Serve the game: play the heuristics, watch them, review finished games.

    The API is under /api and /ws; the built client, when there is one, everywhere else.
    """
    try:
        import uvicorn  # noqa: PLC0415 -- the web extra, loaded only here

        from ticket_to_ride.ui.web.app import create_app  # noqa: PLC0415
    except ModuleNotFoundError as e:
        typer.echo(f"`ttr serve` needs the web extra ({e.name} is missing): uv sync --extra web")
        raise typer.Exit(1) from e

    opponents = [text.strip() for text in agents.split(",") if text.strip()]
    app = create_app(db=Path(database), results=Path(results), opponents=opponents)
    typer.echo(f"serving on http://{host}:{port}  (play store: {database})")
    uvicorn.run(app, host=host, port=port, log_level="info")
