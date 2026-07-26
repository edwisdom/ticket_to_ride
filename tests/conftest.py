"""Shared test fixtures.

Seeding discipline note: nothing in the test suite may rely on the global `random` or
`np.random` state. Every fixture that needs randomness takes an explicit seed, which is
the same rule the engine itself follows (see docs/reproducibility.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def rules_md() -> str:
    return (REPO_ROOT / "RULES.md").read_text(encoding="utf-8")


@pytest.fixture
def seed() -> int:
    """A fixed seed. Tests that want several should derive, not draw."""
    return 20260726
