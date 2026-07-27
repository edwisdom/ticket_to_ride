"""Boundary 1: `ticket_to_ride.engine` must import without torch, numpy, or `ttr_rust`.

This is the authoritative check. Ruff's TID253 rule catches module-level `import torch`
at lint time, but only this test catches a transitive import pulled in through some
other module, and only this test covers numpy (which is legitimate everywhere *except*
the engine, so it cannot be banned globally by lint).

Why torch and numpy: ten self-play worker processes each importing torch costs ~1.5s and
~300MB RSS apiece. A torch-free engine starts in ~50ms, and CI can run the entire
unit + property suite with no torch installed at all.

Why `ttr_rust`: the Python engine is the **permanent differential-testing oracle** for the
Rust core (PLAN.md §8). An oracle that imports the implementation it validates is not an
oracle -- a shared bug would agree with itself and the harness would report green. The two
are separate distributions precisely so this boundary is enforceable, and it is enforced
here rather than trusted.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Installed as a meta_path finder so it fires on the *first* import attempt, before the
# module is ever found on disk -- this works whether or not the package is installed.
_GUARD = """
import sys

BANNED = {"torch", "numpy", "ttr_rust"}


class Guard:
    def find_module(self, fullname, path=None):
        self.find_spec(fullname, path)
        return None

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BANNED:
            raise AssertionError(
                f"ticket_to_ride.engine must not import {root!r} "
                f"(triggered by {fullname!r})"
            )
        return None


sys.meta_path.insert(0, Guard())
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_GUARD + body)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_engine_imports_without_torch_or_numpy() -> None:
    result = _run("import ticket_to_ride.engine\n")
    assert result.returncode == 0, f"engine import pulled in a banned module:\n{result.stderr}"


def test_the_oracle_does_not_import_the_implementation_it_validates() -> None:
    """`ttr_rust` is in BANNED; this names the reason so it is not "cleaned up" later.

    Covered by the test above, which bans all three at once. Kept separate because the
    failure means something completely different: torch is a startup-cost boundary, this
    one is a correctness boundary, and a future edit that reaches for the fast engine
    inside `engine/` for a "harmless" speedup should fail with the right explanation.
    """
    result = _run("import ticket_to_ride.engine\nimport ticket_to_ride.engine.state\n")
    assert result.returncode == 0, (
        "the Python engine imported ttr_rust. It is the oracle the Rust core is checked "
        f"against and must stay independent of it:\n{result.stderr}"
    )


def test_guard_actually_fires() -> None:
    """Guard the guard: a test that can never fail proves nothing."""
    result = _run("import numpy\n")
    assert result.returncode != 0, "import guard did not fire on a banned module"
    assert "must not import" in result.stderr
