//! The frozen PRNG contract. Ported from `tests/unit/engine/test_rng.py`.
//!
//! Three layers, deliberately:
//!
//! 1. **The published reference vector.** `pcg32_srandom_r(&rng, 42, 54)` then six draws is
//!    printed by O'Neill's own `pcg32-demo`; matching it validates the port against upstream
//!    rather than against itself (`rng::tests::reproduces_upstream_pcg32_demo`).
//! 2. **Our golden vectors**, which pin what the reference does not cover: bounded draws,
//!    shuffle direction, stream derivation (`tests/contract_vectors.rs`, plus the whole-file
//!    regeneration below).
//! 3. **Properties**, which say what the vectors mean -- most of this file.

use std::path::PathBuf;

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use ttr_core::hashing::{hash64, hash128_hex};
use ttr_core::rng::{Part, Pcg32, derive, stream};

/// The six 32-bit outputs printed by the upstream pcg32-demo for seed (42, 54).
const REFERENCE_42_54: [u32; 6] = [
    0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e,
];

/// The message of the panic `f` raises, or `None` if it returns normally.
fn panic_message<R>(f: impl FnOnce() -> R) -> Option<String> {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default(),
    )
}

// ---------------------------------------------------------------------------
// 1. The published reference
// ---------------------------------------------------------------------------

/// A port that skips the two warm-up draws still "works" -- and diverges from step one.
#[test]
fn seeding_does_the_two_reference_warmups() {
    let mut naive = Pcg32::new(42, (54 << 1) | 1);
    let got: Vec<u32> = (0..6).map(|_| naive.next_u32()).collect();
    assert_ne!(got, REFERENCE_42_54);
}

// ---------------------------------------------------------------------------
// 2. Golden vectors
// ---------------------------------------------------------------------------

/// Byte-for-byte, so a *deleted* section fails too.
///
/// The per-section checks in `tests/contract_vectors.rs` iterate whatever the file contains;
/// only this catches a vector that quietly stopped being checked. It rebuilds the whole file
/// from this crate exactly as `tools/gen_vectors.py` built it (`json.dumps(indent=2,
/// sort_keys=True)` plus a newline, which `to_string_pretty` over sorted maps reproduces).
#[test]
fn the_whole_vector_file_reproduces() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/contract_vectors.json")
        .canonicalize()
        .expect("tests/golden/contract_vectors.json is missing");
    let text = std::fs::read_to_string(&path).expect("reading contract vectors");
    let built = build_vectors();

    // Structural first, so a missing or extra vector shows up as a value diff.
    let on_disk: Value = serde_json::from_str(&text).expect("contract vectors are not JSON");
    assert_eq!(
        on_disk, built,
        "CONTRACT VIOLATION: the frozen test vectors no longer reproduce. Every recorded \
         replay depends on these values; a deliberate change bumps CONTRACT_VERSION."
    );
    let rendered = serde_json::to_string_pretty(&built).expect("vectors serialize") + "\n";
    assert!(
        text == rendered,
        "the vector file has the right values but not the generator's exact bytes"
    );
}

/// The vector file's content, built the way `tools/gen_vectors.py::build()` built it.
fn build_vectors() -> Value {
    const ROOT: u64 = 20260726;

    let mut reference = Pcg32::seeded(42, 54);
    let reference: Vec<u32> = (0..6).map(|_| reference.next_u32()).collect();

    let mut seeded = Map::new();
    for (initstate, initseq) in [(0, 0), (1, 2), (u64::MAX, u64::MAX), (ROOT, 0)] {
        let mut rng = Pcg32::seeded(initstate, initseq);
        let values: Vec<u32> = (0..4).map(|_| rng.next_u32()).collect();
        seeded.insert(
            format!("{initstate},{initseq}"),
            json!({"values": values, "final_state": rng.state, "final_inc": rng.inc}),
        );
    }

    let mut below = Map::new();
    for bound in [1u32, 2, 6, 9, 30, 100, 110] {
        let mut rng = stream(
            ROOT,
            &[
                Part::Str("vectors"),
                Part::Str("below"),
                Part::Int(u64::from(bound)),
            ],
        );
        let draws: Vec<u32> = (0..12).map(|_| rng.below(bound)).collect();
        below.insert(bound.to_string(), json!(draws));
    }

    // One shuffle per size the engine materializes: USA's 110 cards, mini's 54, tickets.
    let mut shuffles = Map::new();
    for n in [14u8, 30, 54, 110] {
        let mut rng = stream(
            ROOT,
            &[
                Part::Str("vectors"),
                Part::Str("shuffle"),
                Part::Int(u64::from(n)),
            ],
        );
        let mut items: Vec<u8> = (0..n).collect();
        rng.shuffle(&mut items);
        shuffles.insert(
            n.to_string(),
            json!({"head": items[..12], "digest": hash128_hex(&items), "final_state": rng.state}),
        );
    }

    // `()`, `("",)`, `(1,)` and `("1",)` are exactly the cases the separator and the
    // int/str split exist to tell apart.
    let part_sets: [&[Part]; 7] = [
        &[],
        &[Part::Str("env")],
        &[Part::Str("env"), Part::Str("deck")],
        &[
            Part::Str("agent"),
            Part::Int(3),
            Part::Str("game"),
            Part::Int(7),
        ],
        &[Part::Str("")],
        &[Part::Int(1)],
        &[Part::Str("1")],
    ];
    let mut derived = Map::new();
    for parts in part_sets {
        let (initstate, initseq) = derive(ROOT, parts);
        derived.insert(python_json_list(parts), json!([initstate, initseq]));
    }

    let every_byte: Vec<u8> = (0..=255).collect();
    let cases: [&[u8]; 4] = [b"", b"\x00", b"ticket to ride", &every_byte];
    let mut h64 = Map::new();
    let mut h128 = Map::new();
    for case in cases {
        let hex: String = case.iter().map(|b| format!("{b:02x}")).collect();
        h64.insert(hex.clone(), json!(hash64(case)));
        h128.insert(hex, json!(hash128_hex(case)));
    }

    json!({
        "contract_version": ttr_core::CONTRACT_VERSION,
        "rng": {
            "reference_42_54": reference,
            "seeded": seeded,
            "below": below,
            "shuffle": shuffles,
            "derive": derived,
        },
        "hashing": {"hash64": h64, "hash128": h128},
    })
}

/// A part list spelled the way Python's `json.dumps(list(parts))` spells it -- the file's keys.
fn python_json_list(parts: &[Part]) -> String {
    let items: Vec<String> = parts
        .iter()
        .map(|part| match part {
            Part::Str(s) => Value::from(*s).to_string(),
            Part::Int(i) => i.to_string(),
        })
        .collect();
    format!("[{}]", items.join(", "))
}

// ---------------------------------------------------------------------------
// 3. Properties
// ---------------------------------------------------------------------------

/// Python masked every step to 32/64 bits by hand. Here the types are the widths, and what is
/// left to check is that the arithmetic *wraps*: a plain `*` or `+` on the LCG state overflows
/// within a draw or two, which panics in the debug builds this suite runs in.
#[test]
fn output_and_state_stay_in_range() {
    let mut rng = Pcg32::seeded(1, 2);
    let inc = rng.inc;
    for _ in 0..1000 {
        rng.next_u32();
    }
    assert_eq!(rng.inc & 1, 1, "the LCG increment must be odd");
    assert_eq!(rng.inc, inc, "the increment is a constant of the stream");
}

#[test]
fn below_respects_its_bound() {
    let mut rng = stream(1, &[Part::Str("test")]);
    for bound in [1u32, 2, 6, 110] {
        for _ in 0..500 {
            assert!(rng.below(bound) < bound, "below({bound})");
        }
    }
}

/// `below(0)` has no answer. A negative bound is unrepresentable (`u32`), so zero is the whole
/// check, and it is a hard `assert!`: it must refuse in release builds too.
#[test]
fn below_rejects_nonpositive_bounds() {
    let mut rng = stream(1, &[Part::Str("test")]);
    let message = panic_message(move || rng.below(0)).expect("below(0) returned a value");
    assert!(message.contains("bound must be positive"), "{message}");
}

#[test]
fn below_covers_its_range() {
    let mut rng = stream(3, &[Part::Str("coverage")]);
    let mut counts = [0u32; 6];
    for _ in 0..60_000 {
        counts[rng.below(6) as usize] += 1;
    }
    assert!(counts.iter().all(|&n| 9000 < n && n < 11000), "{counts:?}");
}

proptest! {
    /// Any multiset, any seed: a shuffle may reorder cards but never invent or lose one.
    #[test]
    fn shuffle_is_always_a_permutation(
        items in prop::collection::vec(any::<u8>(), 0..=200),
        seed in any::<u32>(),
    ) {
        let mut shuffled = items.clone();
        stream(u64::from(seed), &[Part::Str("prop")]).shuffle(&mut shuffled);
        shuffled.sort_unstable();
        let mut original = items;
        original.sort_unstable();
        prop_assert_eq!(shuffled, original);
    }
}

/// Direction is frozen: ascending Fisher-Yates is equally correct and differs.
#[test]
fn shuffle_is_descending_fisher_yates() {
    let mut items: Vec<u8> = (0..20).collect();
    stream(5, &[Part::Str("dir")]).shuffle(&mut items);

    let mut ascending: Vec<u8> = (0..20).collect();
    let mut rng = stream(5, &[Part::Str("dir")]);
    let n = ascending.len();
    for i in 0..n - 1 {
        let j = i + rng.below((n - i) as u32) as usize;
        ascending.swap(i, j);
    }

    assert_ne!(items, ascending);
}

/// Also checks that nothing is drawn: a one-card reshuffle that consumed randomness would
/// shift every later reshuffle on the stream.
#[test]
fn shuffle_handles_degenerate_lengths() {
    for n in [0u8, 1] {
        let mut items: Vec<u8> = (0..n).collect();
        let mut rng = stream(1, &[Part::Str("degenerate")]);
        let before = rng;
        rng.shuffle(&mut items);
        assert_eq!(items, (0..n).collect::<Vec<u8>>());
        assert_eq!(rng, before, "shuffling {n} items drew from the stream");
    }
}

#[test]
fn shuffle_actually_shuffles() {
    let mut items: Vec<u8> = (0..110).collect();
    stream(11, &[Part::Str("deck")]).shuffle(&mut items);
    assert_ne!(items, (0..110).collect::<Vec<u8>>());
    let fixed = items
        .iter()
        .enumerate()
        .filter(|&(i, &card)| i == usize::from(card))
        .count();
    assert!(
        fixed < 10,
        "suspiciously close to identity: {fixed} fixed points"
    );
}

// ---------------------------------------------------------------------------
// Stream derivation
// ---------------------------------------------------------------------------

#[test]
fn streams_are_deterministic_and_distinct() {
    let env = |root| derive(root, &[Part::Str("env")]);
    assert_eq!(env(7), env(7));
    assert_ne!(env(7), derive(7, &[Part::Str("agent")]));
    assert_ne!(env(7), env(8));
    assert_ne!(
        derive(7, &[Part::Str("env"), Part::Int(1)]),
        derive(7, &[Part::Str("env"), Part::Int(2)])
    );
}

/// Without the 0x1f separator these would be the same bytes, and the same stream.
#[test]
fn separator_prevents_part_collisions() {
    assert_ne!(
        derive(0, &[Part::Str("ab"), Part::Str("c")]),
        derive(0, &[Part::Str("a"), Part::Str("bc")])
    );
    assert_ne!(
        derive(0, &[Part::Str("a")]),
        derive(0, &[Part::Str("a"), Part::Str("")])
    );
}

/// The whole point of named streams: draining one must not move another. If the environment
/// and the agents shared a stream, an agent that sampled one extra action would shift every
/// subsequent card draw, and paired evaluation would silently stop being paired.
#[test]
fn streams_do_not_interfere() {
    let env_draws = || {
        let mut env = stream(42, &[Part::Str("env")]);
        (0..4).map(|_| env.next_u32()).collect::<Vec<u32>>()
    };
    let env_alone = env_draws();

    let mut agent = stream(42, &[Part::Str("agent"), Part::Int(0)]);
    for _ in 0..1000 {
        agent.next_u32();
    }

    assert_eq!(env_alone, env_draws());
}

/// Python's roots were unbounded ints, masked to 64 bits so every caller agreed with the u64
/// engine. Here the root *is* a `u64`, so the mask is the type; this pins that the `as u64`
/// conversion (two's complement, truncating) is the same map Python's `& MASK64` was.
#[test]
fn root_seed_is_masked_to_64_bits() {
    assert_eq!(derive((1u128 << 64) as u64, &[]), derive(0, &[]));
    assert_eq!(derive(-1i64 as u64, &[]), derive(u64::MAX, &[]));
}

// ---------------------------------------------------------------------------
// Object protocol
// ---------------------------------------------------------------------------

/// `Pcg32` is `Copy`: a copied generator must be a separate stream, as a cloned `State`'s is.
#[test]
fn clone_is_independent() {
    let mut rng = stream(1, &[Part::Str("clone")]);
    let mut copy = rng;
    assert_eq!(copy, rng);
    let a: Vec<u32> = (0..3).map(|_| rng.next_u32()).collect();
    let b: Vec<u32> = (0..3).map(|_| copy.next_u32()).collect();
    assert_eq!(a, b);
    rng.next_u32();
    assert_ne!(copy, rng);
}

/// Equality covers both words: the same state with a different increment is a different
/// stream from the next draw on. (`__hash__` and `NotImplemented` are Python object protocol.)
#[test]
fn equality_and_hash() {
    let a = Pcg32::new(1, 3);
    assert_eq!(a, Pcg32::new(1, 3));
    assert_ne!(a, Pcg32::new(2, 3));
    assert_ne!(a, Pcg32::new(1, 5));
    let shown = format!("{a:?}");
    assert!(
        shown.contains("Pcg32") && shown.contains("state"),
        "{shown}"
    );
}

/// `Pcg32::new` takes `u64`s, so an out-of-range word is unrepresentable. The one place
/// construction can leave 64 bits is `seeded`'s `initseq << 1`, which must drop the top bit
/// exactly as C does and as Python's mask did -- and must wrap, not overflow, at the extremes.
#[test]
fn constructor_masks_to_64_bits() {
    let wide = Pcg32::new(((1u128 << 64) + 5) as u64, ((1u128 << 64) + 7) as u64);
    assert_eq!((wide.state, wide.inc), (5, 7));

    assert_eq!(Pcg32::seeded(42, 54 | (1 << 63)), Pcg32::seeded(42, 54));
    let extreme = Pcg32::seeded(u64::MAX, u64::MAX);
    assert_eq!(extreme.inc, u64::MAX, "(MAX << 1) | 1 keeps every low bit");
}
