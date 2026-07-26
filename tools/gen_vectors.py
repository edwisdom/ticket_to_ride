#!/usr/bin/env python3
"""Generate the frozen contract test vectors in tests/golden/contract_vectors.json.

    uv run python tools/gen_vectors.py --check    # fail if the implementation drifted
    uv run python tools/gen_vectors.py --write    # rewrite (a deliberate contract bump)

These vectors are the Phase 1 -> Phase 2 gate. The Rust engine will be validated against
this exact file, so a diff here is never "just regenerate it": it means the contract
changed and every recorded replay in existence is now unreplayable. `--write` therefore
also requires bumping `CONTRACT_VERSION` in `ticket_to_ride/engine/contract.py`.

See docs/CONTRACT.md for the prose specification these vectors pin down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ticket_to_ride.engine.contract import CONTRACT_VERSION  # noqa: E402
from ticket_to_ride.engine.hashing import hash64, hash128  # noqa: E402
from ticket_to_ride.engine.rng import Pcg32, derive, stream  # noqa: E402

OUT = REPO_ROOT / "tests" / "golden" / "contract_vectors.json"


def rng_vectors() -> dict[str, object]:
    """PRNG, bounded draws, shuffle, and stream derivation."""
    # The published pcg32-demo vector. Any correct PCG-XSH-RR 64/32 reproduces it, so it
    # validates a Rust port against the upstream reference rather than against us.
    reference = Pcg32.seeded(42, 54)

    seeded: dict[str, dict[str, object]] = {}
    for initstate, initseq in ((0, 0), (1, 2), (2**64 - 1, 2**64 - 1), (20260726, 0)):
        rng = Pcg32.seeded(initstate, initseq)
        seeded[f"{initstate},{initseq}"] = {
            "values": [rng.next_u32() for _ in range(4)],
            "final_state": rng.state,
            "final_inc": rng.inc,
        }

    below: dict[str, list[int]] = {}
    for bound in (1, 2, 6, 9, 30, 100, 110):
        rng = stream(20260726, "vectors", "below", bound)
        below[str(bound)] = [rng.below(bound) for _ in range(12)]

    # A shuffle vector per deck size we actually materialize (USA 110, mini 54, tickets).
    shuffles: dict[str, dict[str, object]] = {}
    for n in (14, 30, 54, 110):
        rng = stream(20260726, "vectors", "shuffle", n)
        items = list(range(n))
        rng.shuffle(items)
        shuffles[str(n)] = {
            "head": items[:12],
            "digest": hash128(bytes(items)),
            "final_state": rng.state,
        }

    # Keys are JSON so `()`, `("",)` and `(1,)` vs `("1",)` stay distinguishable -- a
    # "|".join key silently collapses exactly the cases the separator exists to separate.
    part_sets: tuple[tuple[str | int, ...], ...] = (
        (),
        ("env",),
        ("env", "deck"),
        ("agent", 3, "game", 7),
        ("",),
        (1,),
        ("1",),
    )
    derived = {json.dumps(list(parts)): list(derive(20260726, *parts)) for parts in part_sets}

    return {
        "reference_42_54": [reference.next_u32() for _ in range(6)],
        "seeded": seeded,
        "below": below,
        "shuffle": shuffles,
        "derive": derived,
    }


def hashing_vectors() -> dict[str, object]:
    cases = [b"", b"\x00", b"ticket to ride", bytes(range(256))]
    return {
        "hash64": {c.hex(): hash64(c) for c in cases},
        "hash128": {c.hex(): hash128(c) for c in cases},
    }


def build() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "rng": rng_vectors(),
        "hashing": hashing_vectors(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if vectors drifted")
    group.add_argument("--write", action="store_true", help="rewrite (a contract bump)")
    args = parser.parse_args(argv)

    content = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else None

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(content, encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO_ROOT)} at contract version {CONTRACT_VERSION}")
        return 0

    if current is None:
        print(f"missing {OUT.relative_to(REPO_ROOT)}; run --write", file=sys.stderr)
        return 1
    if current != content:
        print(
            "CONTRACT VIOLATION: the frozen test vectors no longer reproduce.\n"
            "This is not a regeneration -- every recorded replay depends on these bytes.\n"
            "If the change is intended, bump CONTRACT_VERSION and run --write.",
            file=sys.stderr,
        )
        return 1
    print(f"contract vectors reproduce at version {CONTRACT_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
