"""The generated board files are what `tools/gen_board.py` would write today."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generated_board_files_are_not_stale() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "gen_board.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
