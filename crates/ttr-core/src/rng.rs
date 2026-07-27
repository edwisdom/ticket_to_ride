//! PCG-XSH-RR 64/32 and the seed-stream derivation. **Frozen** -- see docs/CONTRACT.md §1.
//!
//! Transcribed from O'Neill's reference `pcg_basic.c`, and validated against *upstream's*
//! own `pcg32-demo` output as well as against our vectors. That distinction matters: a
//! test that only checks Rust against Python catches drift but not a shared
//! misunderstanding of PCG. `seed(42, 54)` reproducing `a15c02b7 7b47f409 ...` catches
//! both.
//!
//! Three things a port gets wrong quietly, each with an equally-correct alternative that
//! produces different -- still perfectly legal -- trajectories, which is exactly what
//! destroys a differential-testing oracle:
//!
//! * **The output permutes the *old* LCG state**, not the newly advanced one.
//! * **The rotation is `(-rot) mod 32`, not `32 - rot`.** They agree for all 31 non-zero
//!   rotations and differ at `rot == 0`, where `32 - rot` shifts a `u32` by 32 -- undefined
//!   behaviour in C and a panic in debug Rust. [`Pcg32::next_u32`] uses
//!   [`u32::rotate_right`], which is correct at zero by construction, and a test asserts
//!   the zero case is actually reached rather than merely handled.
//! * **Fisher-Yates runs descending**, and bounded draws **reject** rather than using
//!   Lemire's multiply-shift. Both alternatives are uniform, faster, and wrong here.

use crate::hashing;

/// The LCG multiplier from O'Neill's reference `pcg_basic.c`.
pub const PCG_MULTIPLIER: u64 = 6364136223846793005;

/// A PCG-XSH-RR 64/32 generator: 64-bit LCG state, 32-bit permuted output.
///
/// `Copy`, because `State` carries one by value and cloning a position must stay a
/// memcpy. `inc` is the LCG's additive constant and is always odd -- it must be coprime
/// to 2^64 for the generator to have full period.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pcg32 {
    pub state: u64,
    pub inc: u64,
}

impl Pcg32 {
    /// A generator restored from an existing `(state, inc)` pair, as carried in `State`
    /// and reproduced by `state_hash()`.
    pub const fn new(state: u64, inc: u64) -> Self {
        Self { state, inc }
    }

    /// `pcg32_srandom_r`, transcribed exactly.
    ///
    /// The two warm-up draws and the `+= initstate` between them are part of the reference
    /// seeding routine, not decoration: skipping them makes low seeds correlate badly.
    pub fn seeded(initstate: u64, initseq: u64) -> Self {
        let mut rng = Self {
            state: 0,
            // High bit shifted out, as in C. Rust discards it rather than trapping;
            // only a shift *amount* at or above the width would panic.
            inc: (initseq << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(initstate);
        rng.next_u32();
        rng
    }

    /// `pcg32_random_r`: advance the LCG, then permute the *previous* state.
    ///
    /// XSH (xorshift the high bits into the low half) then RR (rotate right by the top 5
    /// bits). Outputting from the old state lets the multiply and the output permutation
    /// overlap in hardware, and means the first output depends on the seed rather than on
    /// one round of mixing.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        // `rotate_right` is `(x >> rot) | (x << ((32 - rot) % 32))`, i.e. C's
        // `(-rot) & 31`. Writing `x << (32 - rot)` by hand is correct for 31 of the 32
        // rotations and a debug panic for the 32nd.
        xorshifted.rotate_right(rot)
    }

    /// Uniform on `[0, bound)` by rejection -- `pcg32_boundedrand_r`.
    ///
    /// The threshold discards the shortest prefix of the 32-bit range that leaves an exact
    /// multiple of `bound`, so the result is *exactly* uniform. Plain `next_u32() % bound`
    /// is biased, worst for the large bounds actually used here (shuffling 110 cards).
    /// Lemire's multiply-shift is faster and equally unbiased and yields **different
    /// values**, so it is not a permitted substitution.
    ///
    /// # Panics
    /// If `bound` is zero.
    #[inline]
    pub fn below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "bound must be positive");
        // `wrapping_neg()` on a u32 is exactly `(2^32 - bound) mod 2^32`, and for
        // `bound >= 1` that is `2^32 - bound` with no truncation.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }

    /// Fisher-Yates in place, **descending**. Frozen direction.
    ///
    /// Ascending Fisher-Yates is equally correct and equally uniform, and produces a
    /// different permutation from the same stream. The direction is contract, not style.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// One component of a stream name: text or an integer.
///
/// The two are encoded differently (UTF-8 bytes vs 8 little-endian bytes), so `1` and
/// `"1"` name different streams -- which the contract vectors pin down explicitly.
#[derive(Clone, Copy, Debug)]
pub enum Part<'a> {
    Str(&'a str),
    Int(u64),
}

impl<'a> From<&'a str> for Part<'a> {
    fn from(s: &'a str) -> Self {
        Part::Str(s)
    }
}

impl From<u64> for Part<'_> {
    fn from(i: u64) -> Self {
        Part::Int(i)
    }
}

/// Derive an `(initstate, initseq)` seed pair for a named stream. **Frozen.**
///
/// One `seed` in the config; every other stream is derived, never drawn. Named streams
/// keep the environment's randomness independent of the agents': if they shared a stream,
/// an agent that sampled one extra action would shift every subsequent card draw and
/// paired evaluation would silently stop being paired.
///
/// Encoding: the root as 8 little-endian bytes, then for each part a `0x1f` separator
/// followed by the part's UTF-8 bytes (text) or 8 little-endian bytes (integer). **The
/// separator is load-bearing** -- without it `("ab", "c")` and `("a", "bc")` derive the
/// same stream.
pub fn derive(root: u64, parts: &[Part]) -> (u64, u64) {
    let mut hasher = hashing::blake2b_128_state();
    hasher.update(&root.to_le_bytes());
    for part in parts {
        hasher.update(&[0x1f]);
        match part {
            Part::Str(s) => hasher.update(s.as_bytes()),
            Part::Int(i) => hasher.update(&i.to_le_bytes()),
        };
    }
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    (
        u64::from_le_bytes(bytes[0..8].try_into().expect("16-byte digest")),
        u64::from_le_bytes(bytes[8..16].try_into().expect("16-byte digest")),
    )
}

/// A seeded generator for the named stream. See [`derive`].
pub fn stream(root: u64, parts: &[Part]) -> Pcg32 {
    let (initstate, initseq) = derive(root, parts);
    Pcg32::seeded(initstate, initseq)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `pcg32-demo` output. Any correct PCG-XSH-RR 64/32 reproduces it, so
    /// this validates the port against **O'Neill** rather than against our own Python --
    /// which is the one check that catches both sides being wrong in the same way.
    #[test]
    fn reproduces_upstream_pcg32_demo() {
        let mut rng = Pcg32::seeded(42, 54);
        let got: Vec<u32> = (0..6).map(|_| rng.next_u32()).collect();
        assert_eq!(
            got,
            vec![
                0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e
            ]
        );
    }

    /// `rot == 0` is the one case where C's `(-rot) & 31` and a naive `32 - rot` differ.
    ///
    /// A port writing `x >> rot | x << (32 - rot)` is correct for all 31 other rotations
    /// and wrong for this one, so it has to be reached, not merely handled. Mirrors
    /// `test_rot_zero_branch_is_exercised` on the Python side.
    #[test]
    fn rot_zero_branch_is_exercised() {
        let mut rng = Pcg32::seeded(7, 11);
        let mut seen = [0u32; 32];
        for _ in 0..5000 {
            seen[(rng.state >> 59) as usize] += 1;
            rng.next_u32();
        }
        assert!(seen[0] > 0, "never hit rot == 0, so the branch is untested");
        assert!(seen.iter().all(|&c| c > 0), "some rotation never occurred");
    }

    /// The rejection threshold must leave an exact multiple of `bound` values. Proving it
    /// arithmetically beats a chi-square: `next_u32() % bound` would pass a chi-square at
    /// realistic sample sizes while still being measurably biased for large bounds.
    #[test]
    fn bounded_draws_are_exactly_uniform() {
        for bound in [1u32, 2, 3, 5, 6, 7, 9, 30, 100, 110, u32::MAX / 2, u32::MAX] {
            let threshold = bound.wrapping_neg() % bound;
            assert!(threshold < bound, "bound {bound}");
            let usable = (1u64 << 32) - u64::from(threshold);
            assert_eq!(usable % u64::from(bound), 0, "bound {bound}");
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = stream(20260726, &[Part::Str("vectors"), Part::Str("shuffle")]);
        let mut items: Vec<u8> = (0..110).collect();
        rng.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..110).collect::<Vec<u8>>());
    }

    /// The `0x1f` separator is what stops `("ab", "c")` and `("a", "bc")` colliding.
    #[test]
    fn the_separator_is_load_bearing() {
        let ab_c = derive(1, &[Part::Str("ab"), Part::Str("c")]);
        let a_bc = derive(1, &[Part::Str("a"), Part::Str("bc")]);
        assert_ne!(ab_c, a_bc);
    }

    /// An integer part and its decimal spelling are different streams.
    #[test]
    fn integer_and_text_parts_are_distinct() {
        assert_ne!(derive(1, &[Part::Int(1)]), derive(1, &[Part::Str("1")]));
    }
}
