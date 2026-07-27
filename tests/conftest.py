"""Shared test fixtures.

Seeding discipline note: nothing in the test suite may rely on the global `random` or
`np.random` state. Every fixture that needs randomness takes an explicit seed, which is
the same rule the engine itself follows (see docs/reproducibility.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# `tests/rig.py` is a helper module, not a test file, so pytest never puts its directory on
# the path by itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


# ---------------------------------------------------------------------------
# The Rust core
# ---------------------------------------------------------------------------
#
# `ttr_rust` is built by `make rust`, not by `uv sync`, so the pure-Python edit loop and
# the torch-free CI job never need a Rust toolchain. Tests that need it therefore have to
# cope with it being absent.
#
# Skipping is right locally and **wrong in CI**: a differential harness that silently
# skips is worse than no harness, because it reports green while comparing nothing. So the
# skip is conditional -- `TTR_REQUIRE_RUST=1` turns it into a hard failure, and CI sets it.


def _require_rust() -> bool:
    return os.environ.get("TTR_REQUIRE_RUST", "").lower() not in ("", "0", "false", "no")


def rust_or_skip() -> ModuleType:
    """Import `ttr_rust`, or skip -- unless `TTR_REQUIRE_RUST` says a skip is a failure."""
    try:
        # Deferred on purpose: a top-level import would make collection of every module
        # that imports this one fail outright when the extension is not built, which is
        # the exact situation this function exists to handle gracefully.
        import ttr_rust  # noqa: PLC0415
    except ImportError as exc:
        message = f"ttr_rust is not built ({exc}); run `make rust`"
        if _require_rust():
            pytest.fail(f"TTR_REQUIRE_RUST is set but {message}")
        pytest.skip(message)
    return ttr_rust


@pytest.fixture(scope="session")
def rust() -> ModuleType:
    return rust_or_skip()
