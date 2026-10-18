"""Fixtures for the web server's tests; the driver itself is `tests/web_rig.py`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from web_rig import make_app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(make_app(tmp_path)) as c:
        yield c
