"""`ttr serve`: the API always, the built client when there is one, a placeholder before."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner
from web_rig import make_app

from ticket_to_ride.cli.app import app as cli


def test_without_a_build_every_page_is_the_placeholder(client: TestClient) -> None:
    for path in ("/", "/lobby", "/games/abc"):
        response = client.get(path)
        assert response.status_code == 200
        assert "has not been built yet" in response.text
    assert client.get("/api/meta").json()["client_build"] is None


def test_a_build_is_served_with_the_app_taking_its_own_routes(tmp_path: Path) -> None:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><div id=app></div>", encoding="utf-8")
    (static / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not for the web", encoding="utf-8")
    with TestClient(make_app(tmp_path, static=static)) as client:
        assert client.get("/").text.startswith("<!doctype html>")
        assert client.get("/assets/app.js").text == "console.log(1)"
        # A client-side route falls through to the app.
        assert client.get("/review/abc").text.startswith("<!doctype html>")
        # Nothing outside the build directory.
        assert "not for the web" not in client.get("/../secret.txt").text
        assert "not for the web" not in client.get("/%2e%2e/secret.txt").text
        # An unknown API path is a JSON 404, not the app.
        missing = client.get("/api/nope")
        assert missing.status_code == 404 and missing.json()["detail"]["code"] == "not_found"
        build = client.get("/api/meta").json()["client_build"]
        assert build is not None and len(build) == 12


def test_the_board_is_the_engines_and_carries_the_layout_when_built(client: TestClient) -> None:
    board = client.get("/api/boards/mini").json()
    assert len(board["segments"]) == 30 and board["locomotive"] == len(board["colors"])
    assert client.get("/api/boards/europe").status_code == 422
    glossary = client.get("/api/glossary").json()
    assert "route_points" in glossary["terms"]


def test_serve_is_a_command_and_the_cli_does_not_load_the_web_stack() -> None:
    result = CliRunner().invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0 and "--results-db" in result.output
    code = "import sys, ticket_to_ride.cli.app; sys.exit('fastapi' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
