"""PCG-XSH-RR 64/32 and the seed-stream derivation. **Frozen** -- see docs/CONTRACT.md.

Nothing in this file may change without bumping the contract version, because every
recorded replay, every golden test vector, and (from Phase 2) the Rust engine reproduce
these exact bit patterns. Two equally-correct PRNGs produce different, still perfectly
valid trajectories -- which is precisely what destroys a differential-testing oracle.

Why PCG rather than `random.Random` or SplitMix: it is ~30 lines, has a published
reference implementation that Rust and Python can both be checked against byte for byte,
and its state is two u64s that fit inside a POD game state. `random.Random` is a
Mersenne Twister with 2.5 KB of state and no realistic chance of a matching Rust port.

The engine never touches global `random`/`np.random`. That is the precondition that makes
mirrored paired evaluation possible at all: environment randomness must be a pure function
of the seed, independent of what the agents do.
"""

from __future__ import annotations

import hashlib
from typing import Final

MASK64: Final = 0xFFFF_FFFF_FFFF_FFFF
MASK32: Final = 0xFFFF_FFFF

#: The LCG multiplier from O'Neill's reference `pcg_basic.c`.
PCG_MULTIPLIER: Final = 6364136223846793005

#: 2**32, used to compute the rejection threshold in `below()`.
_TWO32: Final = 0x1_0000_0000


class Pcg32:
    """A PCG-XSH-RR 64/32 generator: 64-bit LCG state, 32-bit permuted output.

    Mutable and deliberately unshared -- each game owns its own. `state` and `inc` are the
    two u64s stored in `State`; `inc` is always odd (it is the LCG's additive constant and
    must be coprime to 2**64).
    """

    __slots__ = ("inc", "state")

    def __init__(self, state: int, inc: int) -> None:
        self.state = state & MASK64
        self.inc = inc & MASK64

    @classmethod
    def seeded(cls, initstate: int, initseq: int) -> Pcg32:
        """`pcg32_srandom_r`, transcribed exactly.

        The two warm-up draws and the `+= initstate` between them are part of the reference
        seeding routine, not decoration: skipping them makes low seeds correlate badly.
        """
        rng = cls(0, ((initseq << 1) | 1) & MASK64)
        rng.next_u32()
        rng.state = (rng.state + (initstate & MASK64)) & MASK64
        rng.next_u32()
        return rng

    def next_u32(self) -> int:
        """`pcg32_random_r`: advance the LCG, then permute the *previous* state.

        XSH (xorshift high bits into the low half) then RR (rotate right by the top 5
        bits). Outputting from the old state is what lets the multiply and the output
        permutation overlap in hardware; it also means the very first output depends on the
        seed rather than on one round of mixing.
        """
        old = self.state
        self.state = (old * PCG_MULTIPLIER + self.inc) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59
        # (-rot) & 31 in C; for rot == 0 both halves collapse to `xorshifted`.
        return ((xorshifted >> rot) | (xorshifted << (32 - rot))) & MASK32

    def below(self, bound: int) -> int:
        """Uniform on `[0, bound)` by rejection -- `pcg32_boundedrand_r`.

        The threshold discards the shortest prefix of the 32-bit range that makes the
        remainder an exact multiple of `bound`, so the result is *exactly* uniform. Plain
        `next_u32() % bound` is biased, and the bias is largest for the large bounds we
        actually use (shuffling a 110-card deck). Lemire's multiply-shift method is faster
        but yields different values, so it is not interchangeable here.
        """
        if bound <= 0:
            raise ValueError(f"bound must be positive, got {bound}")
        threshold = (_TWO32 - bound) % bound
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % bound

    def shuffle(self, items: list[int]) -> None:
        """Fisher-Yates in place, descending. Frozen direction.

        Ascending and descending Fisher-Yates are both correct and both uniform, and they
        produce *different permutations from the same stream*. The direction is therefore
        part of the contract, not an implementation detail.
        """
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def clone(self) -> Pcg32:
        return Pcg32(self.state, self.inc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pcg32):
            return NotImplemented
        return self.state == other.state and self.inc == other.inc

    def __hash__(self) -> int:
        return hash((self.state, self.inc))

    def __repr__(self) -> str:
        return f"Pcg32(state=0x{self.state:016x}, inc=0x{self.inc:016x})"


def derive(root: int, *parts: str | int) -> tuple[int, int]:
    """Derive a `(initstate, initseq)` seed pair for a named stream.

    One `seed` in the config; everything else derived, never drawn. `derive(root, *parts)`
    is blake2b-128 over the root and the parts, split into two u64s. Named streams
    (`"env"`, `"deck"`, `"agent"`, seat, game id, ...) keep the environment's randomness
    independent of the agents' -- if they shared a stream, an agent that sampled one extra
    action would shift every subsequent card draw and paired evaluation would silently stop
    being paired.

    Encoding, frozen: the root as 8 little-endian bytes, then for each part a `0x1f`
    separator followed by the part's UTF-8 bytes (str) or 8 little-endian bytes (int). The
    separator is what stops `("ab", "c")` and `("a", "bc")` from colliding.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update((root & MASK64).to_bytes(8, "little"))
    for part in parts:
        h.update(b"\x1f")
        h.update(part.encode() if isinstance(part, str) else (part & MASK64).to_bytes(8, "little"))
    digest = h.digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")


def stream(root: int, *parts: str | int) -> Pcg32:
    """A seeded generator for the named stream. See `derive`."""
    return Pcg32.seeded(*derive(root, *parts))
