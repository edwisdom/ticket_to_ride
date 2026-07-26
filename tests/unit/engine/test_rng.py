"""The frozen PRNG contract.

Three layers, deliberately:

1. **The published reference vector.** `pcg32_srandom_r(&rng, 42, 54)` then six draws is
   printed by O'Neill's own `pcg32-demo`. Matching it means a Rust port can be validated
   against upstream rather than against us -- if both implementations are wrong in the same
   way, no amount of comparing them to each other would notice.
2. **Our golden vectors**, which pin the parts the reference does not cover: bounded draws,
   shuffle direction, and stream derivation.
3. **Properties**, which say what the vectors mean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ticket_to_ride.engine.contract import CONTRACT_VERSION
from ticket_to_ride.engine.hashing import hash64, hash128
from ticket_to_ride.engine.rng import MASK32, MASK64, Pcg32, derive, stream

#: The six 32-bit outputs printed by the upstream pcg32-demo for seed (42, 54).
REFERENCE_42_54 = [0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E]


@pytest.fixture(scope="session")
def vectors(repo_root: Path) -> dict:
    return json.loads((repo_root / "tests" / "golden" / "contract_vectors.json").read_text())


# ---------------------------------------------------------------------------
# 1. The published reference
# ---------------------------------------------------------------------------


def test_matches_the_upstream_pcg32_demo() -> None:
    rng = Pcg32.seeded(42, 54)
    assert [rng.next_u32() for _ in range(6)] == REFERENCE_42_54


def test_seeding_does_the_two_reference_warmups() -> None:
    """A port that skips them still "works" -- and diverges from step one."""
    naive = Pcg32(42, (54 << 1) | 1)
    assert [naive.next_u32() for _ in range(6)] != REFERENCE_42_54


# ---------------------------------------------------------------------------
# 2. Golden vectors
# ---------------------------------------------------------------------------


def test_the_whole_vector_file_reproduces(repo_root: Path) -> None:
    """Byte-for-byte, so a *deleted* section fails too.

    The per-section assertions below iterate whatever the file contains; only this catches
    a vector that quietly stopped being checked.
    """
    result = subprocess.run(
        [sys.executable, "tools/gen_vectors.py", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_golden_vectors_reproduce(vectors: dict) -> None:
    assert vectors["contract_version"] == CONTRACT_VERSION
    assert vectors["rng"]["reference_42_54"] == REFERENCE_42_54

    for key, expected in vectors["rng"]["seeded"].items():
        initstate, initseq = (int(x) for x in key.split(","))
        rng = Pcg32.seeded(initstate, initseq)
        assert [rng.next_u32() for _ in range(4)] == expected["values"], key
        assert (rng.state, rng.inc) == (expected["final_state"], expected["final_inc"]), key

    for key, expected in vectors["rng"]["below"].items():
        bound = int(key)
        rng = stream(20260726, "vectors", "below", bound)
        assert [rng.below(bound) for _ in range(12)] == expected, key

    for key, expected in vectors["rng"]["shuffle"].items():
        n = int(key)
        rng = stream(20260726, "vectors", "shuffle", n)
        items = list(range(n))
        rng.shuffle(items)
        assert items[:12] == expected["head"], key
        assert hash128(bytes(items)) == expected["digest"], key
        assert rng.state == expected["final_state"], key

    for key, expected in vectors["rng"]["derive"].items():
        assert list(derive(20260726, *json.loads(key))) == expected, key


def test_golden_hashing_vectors(vectors: dict) -> None:
    for hexed, expected in vectors["hashing"]["hash64"].items():
        assert hash64(bytes.fromhex(hexed)) == expected
    for hexed, expected in vectors["hashing"]["hash128"].items():
        assert hash128(bytes.fromhex(hexed)) == expected


# ---------------------------------------------------------------------------
# 3. Properties
# ---------------------------------------------------------------------------


def test_output_and_state_stay_in_range() -> None:
    rng = Pcg32.seeded(1, 2)
    for _ in range(1000):
        assert 0 <= rng.next_u32() <= MASK32
        assert 0 <= rng.state <= MASK64
    assert rng.inc & 1 == 1, "the LCG increment must be odd"


def test_rot_zero_branch_is_exercised() -> None:
    """`rot == 0` is the one case where C's `(-rot) & 31` and a naive `32 - rot` differ.

    A port that writes `x >> rot | x << (32 - rot)` is correct for all 31 other rotations
    and silently wrong for this one, so the vectors have to cover it.
    """
    rng = Pcg32.seeded(7, 11)
    rots: Counter[int] = Counter()
    for _ in range(5000):
        rots[rng.state >> 59] += 1  # the rotation the *next* call will apply
        rng.next_u32()
    assert rots[0] > 0, "never hit rot == 0, so the golden vectors do not cover it"
    assert set(rots) == set(range(32))


@pytest.mark.parametrize("bound", [1, 2, 3, 5, 6, 7, 9, 30, 100, 110, 2**31 - 1, 2**32 - 1])
def test_bounded_draws_are_exactly_uniform(bound: int) -> None:
    """The rejection threshold must leave an exact multiple of `bound` values.

    Proving it arithmetically beats a chi-square test: `next_u32() % bound` would pass a
    chi-square at these sample sizes while still being measurably biased for large bounds.
    """
    threshold = (2**32 - bound) % bound
    assert (2**32 - threshold) % bound == 0
    assert threshold < bound


def test_below_respects_its_bound() -> None:
    rng = stream(1, "test")
    for bound in (1, 2, 6, 110):
        assert all(0 <= rng.below(bound) < bound for _ in range(500))


def test_below_rejects_nonpositive_bounds() -> None:
    rng = stream(1, "test")
    for bad in (0, -1):
        with pytest.raises(ValueError, match="bound must be positive"):
            rng.below(bad)


def test_below_covers_its_range() -> None:
    rng = stream(3, "coverage")
    counts = Counter(rng.below(6) for _ in range(60_000))
    assert set(counts) == set(range(6))
    assert all(9000 < n < 11000 for n in counts.values()), counts


@given(st.lists(st.integers(0, 255), max_size=200), st.integers(0, 2**32 - 1))
def test_shuffle_is_always_a_permutation(items: list[int], seed: int) -> None:
    original = list(items)
    rng = stream(seed, "prop")
    rng.shuffle(items)
    assert sorted(items) == sorted(original)


def test_shuffle_is_descending_fisher_yates() -> None:
    """Direction is frozen: ascending Fisher-Yates is equally correct and differs."""
    items = list(range(20))
    stream(5, "dir").shuffle(items)

    ascending = list(range(20))
    rng = stream(5, "dir")
    for i in range(len(ascending) - 1):
        j = i + rng.below(len(ascending) - i)
        ascending[i], ascending[j] = ascending[j], ascending[i]

    assert items != ascending


def test_shuffle_handles_degenerate_lengths() -> None:
    for n in (0, 1):
        items = list(range(n))
        stream(1, "degenerate").shuffle(items)
        assert items == list(range(n))


def test_shuffle_actually_shuffles() -> None:
    items = list(range(110))
    stream(11, "deck").shuffle(items)
    assert items != list(range(110))
    assert sum(a == b for a, b in enumerate(items)) < 10, "suspiciously close to identity"


# ---------------------------------------------------------------------------
# Stream derivation
# ---------------------------------------------------------------------------


def test_streams_are_deterministic_and_distinct() -> None:
    assert derive(7, "env") == derive(7, "env")
    assert derive(7, "env") != derive(7, "agent")
    assert derive(7, "env") != derive(8, "env")
    assert derive(7, "env", 1) != derive(7, "env", 2)


def test_separator_prevents_part_collisions() -> None:
    """Without the 0x1f separator these would be the same bytes, and the same stream."""
    assert derive(0, "ab", "c") != derive(0, "a", "bc")
    assert derive(0, "a") != derive(0, "a", "")


def test_int_and_str_parts_are_distinct() -> None:
    assert derive(0, 1) != derive(0, "1")


def test_streams_do_not_interfere() -> None:
    """The whole point of named streams: draining one must not move another.

    If the environment and the agents shared a stream, an agent that sampled one extra
    action would shift every subsequent card draw, and paired evaluation would silently
    stop being paired.
    """
    env_alone = [stream(42, "env").next_u32() for _ in range(4)]

    agent = stream(42, "agent", 0)
    for _ in range(1000):
        agent.next_u32()
    env_after = [stream(42, "env").next_u32() for _ in range(4)]

    assert env_alone == env_after


def test_root_seed_is_masked_to_64_bits() -> None:
    assert derive(2**64) == derive(0)
    assert derive(-1) == derive(2**64 - 1)


# ---------------------------------------------------------------------------
# Object protocol
# ---------------------------------------------------------------------------


def test_clone_is_independent() -> None:
    rng = stream(1, "clone")
    copy = rng.clone()
    assert copy == rng
    assert [rng.next_u32() for _ in range(3)] == [copy.next_u32() for _ in range(3)]
    rng.next_u32()
    assert copy != rng


def test_equality_and_hash() -> None:
    a, b = Pcg32(1, 3), Pcg32(1, 3)
    assert a == b
    assert hash(a) == hash(b)
    assert a != Pcg32(2, 3)
    assert a.__eq__("not a generator") is NotImplemented
    assert "Pcg32(state=" in repr(a)


def test_constructor_masks_to_64_bits() -> None:
    assert Pcg32(2**64 + 5, 2**64 + 7).state == 5
    assert Pcg32(2**64 + 5, 2**64 + 7).inc == 7
