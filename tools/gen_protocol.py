#!/usr/bin/env python3
"""Protocol schema generator: the Pydantic models -> ticket_to_ride/ui/web/protocol.schema.json.

    uv run python tools/gen_protocol.py           # rewrite the schema
    uv run python tools/gen_protocol.py --check   # fail if it is stale (CI, pre-commit)

The models in `ticket_to_ride/ui/web/protocol.py` are the source of truth (PLAN.md §14.2).
The schema is checked in so the web client can generate its TypeScript types from it
without running Python, and so a protocol change shows up in review as a schema diff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ticket_to_ride.ui.web.protocol import schema_text

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "ticket_to_ride" / "ui" / "web" / "protocol.schema.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the schema is stale")
    args = parser.parse_args(argv)

    content = schema_text()
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
    if current == content:
        if args.check:
            print("protocol schema is up to date")
        return 0
    if args.check:
        print(
            f"stale generated file:\n  {OUT.relative_to(REPO_ROOT)}\n\nrun: make protocol",
            file=sys.stderr,
        )
        return 1
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
