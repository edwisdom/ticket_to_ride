"""The board layouts are what `tools/gen_layout.py` would write today, and its geometry holds."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("gen_layout", ROOT / "tools" / "gen_layout.py")
assert _spec is not None and _spec.loader is not None
gen_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_layout)


def test_generated_layouts_are_not_stale() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_layout.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bend", [0.0, 0.2, -0.2])
def test_the_tracks_of_a_double_stay_parallel(bend: float) -> None:
    # Both tracks, bowed either way, keep their separation all along the route.
    arc = gen_layout.Arc((0.0, 0.0), (300.0, 40.0), bend)
    for s in (0.0, arc.length / 3, arc.length):
        left, right = arc.at(s, 8.5), arc.at(s, -8.5)
        assert math.dist(left[:2], right[:2]) == pytest.approx(17.0)
        assert left[2] == pytest.approx(right[2])


def test_an_arc_ends_at_its_cities() -> None:
    arc = gen_layout.Arc((10.0, 20.0), (210.0, 120.0), 0.15)
    assert arc.at(0.0, 0.0)[:2] == pytest.approx((10.0, 20.0))
    assert arc.at(arc.length, 0.0)[:2] == pytest.approx((210.0, 120.0))
    # Bowing left of the direction of travel lengthens the route past its chord.
    assert arc.length > math.dist((10.0, 20.0), (210.0, 120.0))
