"""The 64-bit state digest. **Frozen** -- see docs/CONTRACT.md.

One primitive, blake2b, used at three digest widths across the project:

* 128-bit for `DATA_HASH` (board identity, stamped into every replay);
* 64-bit for `state_hash()` (the differential-testing key) and `position_hash()`
  (the MCTS transposition key);
* 128-bit for seed-stream derivation (`engine/rng.derive`).

**Deviation from PLAN.md §5.8, deliberate.** The plan called for FNV-1a-64 or xxh3.
FNV-1a is a per-byte loop, which in CPython costs ~40 us over a ~450-byte serialized state
-- about 25x the cost of everything else `state_hash()` does, and the differential harness
calls it after *every* step of 100k games. blake2b is in the standard library, runs at C
speed (~1 us for the same input), is specified precisely enough to reimplement from the RFC,
and has a mature Rust crate. xxh3 would also be fast but is a third-party dependency on both
sides for no additional benefit. Collision risk at 64 bits over the ~15M hashes a nightly
differential run produces is ~6e-6, which is the same for any of the three.
"""

from __future__ import annotations

import hashlib


def hash64(data: bytes) -> int:
    """blake2b-64 of `data`, read back as a little-endian u64.

    No key, no salt, no personalization -- a bare blake2b with `digest_size=8`, so a Rust
    implementation is `Blake2bVar::new(8)` with nothing else to get wrong.
    """
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def hash128(data: bytes) -> str:
    """blake2b-128 of `data` as 32 lowercase hex characters. Used for board and rule ids."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()
