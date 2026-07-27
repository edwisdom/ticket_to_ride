//! The project's one hash primitive. **Frozen** -- see docs/CONTRACT.md §3.
//!
//! blake2b at three widths, matching `ticket_to_ride.engine.hashing` byte for byte:
//!
//! * 64-bit for `state_hash()` (the differential-testing key) and `position_hash()` (the
//!   MCTS transposition key), read back as a little-endian `u64`;
//! * 128-bit for `DATA_HASH` (board identity, stamped into every replay) and for
//!   seed-stream derivation in [`crate::rng`].
//!
//! Bare in every case: no key, no salt, no personalisation. Python's
//! `hashlib.blake2b(data, digest_size=n)` and `blake2b_simd`'s
//! `Params::new().hash_length(n)` agree exactly, and the contract vectors prove it rather
//! than assuming it.
//!
//! The contract records why this is blake2b and not the FNV-1a-64 PLAN.md §5.8 called
//! for: FNV-1a is a per-byte loop costing ~40 µs in CPython over a ~450-byte state, and
//! the differential harness hashes after *every* step of 100k games.

use blake2b_simd::Params;

/// blake2b-64 of `data`, read back as a little-endian `u64`.
pub fn hash64(data: &[u8]) -> u64 {
    let digest = Params::new().hash_length(8).hash(data);
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(digest.as_bytes());
    u64::from_le_bytes(bytes)
}

/// blake2b-128 of `data` as raw bytes.
pub fn hash128(data: &[u8]) -> [u8; 16] {
    let digest = Params::new().hash_length(16).hash(data);
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(digest.as_bytes());
    bytes
}

/// blake2b-128 of `data` as 32 lowercase hex characters -- the form board and rule ids
/// take in `RawMap::data_hash` and in every replay header.
pub fn hash128_hex(data: &[u8]) -> String {
    Params::new()
        .hash_length(16)
        .hash(data)
        .to_hex()
        .to_string()
}

/// A streaming blake2b-128 state.
///
/// Exists so seed derivation, whose input is defined incrementally as separator-delimited
/// parts (see [`crate::rng::derive`]), still goes through this module. One primitive
/// configured in one place is the whole point; a second `Params::new()` elsewhere is how
/// a stray `hash_length` or personalisation eventually diverges from Python.
pub fn blake2b_128_state() -> blake2b_simd::State {
    Params::new().hash_length(16).to_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_and_bytes_agree() {
        let data = b"ticket to ride";
        let hex: String = hash128(data).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, hash128_hex(data));
    }

    #[test]
    fn hash64_is_the_low_eight_bytes_read_little_endian() {
        // Not the truncation of the 128-bit digest: blake2b mixes its output length into
        // the parameter block, so an 8-byte digest is a different hash, not a prefix. A
        // port that truncates hash128 would pass casual inspection and fail the vectors.
        let data = b"ticket to ride";
        assert_ne!(hash64(data).to_le_bytes(), hash128(data)[..8]);
    }
}
