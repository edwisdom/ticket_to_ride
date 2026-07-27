//! The frozen contract vectors, checked from Rust with no Python in the loop.
//!
//! This reads `tests/golden/contract_vectors.json` -- the *same file*
//! `tools/gen_vectors.py --check` validates the Python engine against -- rather than a
//! hand-copied constant. Copying the numbers into Rust source would make the two sides
//! drift the moment someone regenerated one and not the other, which is precisely the
//! failure the contract exists to prevent.
//!
//! Running under `cargo test` matters too: a PRNG regression is caught even when the PyO3
//! bindings are broken or unbuilt, so the failure points at the PRNG instead of at the
//! FFI layer.

use std::path::PathBuf;

use serde_json::Value;
use ttr_core::hashing::{hash64, hash128_hex};
use ttr_core::rng::{Part, Pcg32, derive, stream};

fn vectors() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/contract_vectors.json")
        .canonicalize()
        .expect("tests/golden/contract_vectors.json is missing; run `make vectors`");
    let text = std::fs::read_to_string(&path).expect("reading contract vectors");
    serde_json::from_str(&text).expect("contract vectors are not valid JSON")
}

fn u64_of(v: &Value) -> u64 {
    v.as_u64().expect("expected an unsigned integer")
}

#[test]
fn contract_version_matches() {
    assert_eq!(
        u64_of(&vectors()["contract_version"]),
        u64::from(ttr_core::CONTRACT_VERSION),
        "the vectors were generated under a different contract version"
    );
}

#[test]
fn reference_vector_from_upstream_pcg32_demo() {
    let v = vectors();
    let expected: Vec<u64> = v["rng"]["reference_42_54"]
        .as_array()
        .expect("reference_42_54")
        .iter()
        .map(u64_of)
        .collect();
    let mut rng = Pcg32::seeded(42, 54);
    let got: Vec<u64> = (0..expected.len())
        .map(|_| u64::from(rng.next_u32()))
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn seeding_reproduces_values_and_final_state() {
    let v = vectors();
    let cases = v["rng"]["seeded"].as_object().expect("seeded");
    assert!(!cases.is_empty());
    for (key, case) in cases {
        let (initstate, initseq) = key.split_once(',').expect("key is 'initstate,initseq'");
        let mut rng = Pcg32::seeded(
            initstate.parse().expect("initstate"),
            initseq.parse().expect("initseq"),
        );
        let expected: Vec<u64> = case["values"]
            .as_array()
            .expect("values")
            .iter()
            .map(u64_of)
            .collect();
        let got: Vec<u64> = (0..expected.len())
            .map(|_| u64::from(rng.next_u32()))
            .collect();
        assert_eq!(got, expected, "seeded({key}) values");
        // The final state is checked as well as the outputs: two generators can agree on
        // a prefix of outputs and still have diverged internally, which would only show
        // up much later in a game.
        assert_eq!(
            rng.state,
            u64_of(&case["final_state"]),
            "seeded({key}) state"
        );
        assert_eq!(rng.inc, u64_of(&case["final_inc"]), "seeded({key}) inc");
    }
}

#[test]
fn bounded_draws_reproduce() {
    let v = vectors();
    let cases = v["rng"]["below"].as_object().expect("below");
    assert!(!cases.is_empty());
    for (bound_text, expected) in cases {
        let bound: u32 = bound_text.parse().expect("bound");
        let mut rng = stream(
            20260726,
            &[
                Part::Str("vectors"),
                Part::Str("below"),
                Part::Int(u64::from(bound)),
            ],
        );
        let expected: Vec<u64> = expected
            .as_array()
            .expect("draws")
            .iter()
            .map(u64_of)
            .collect();
        let got: Vec<u64> = (0..expected.len())
            .map(|_| u64::from(rng.below(bound)))
            .collect();
        assert_eq!(got, expected, "below({bound})");
    }
}

#[test]
fn shuffles_reproduce() {
    let v = vectors();
    let cases = v["rng"]["shuffle"].as_object().expect("shuffle");
    assert!(!cases.is_empty());
    for (size_text, case) in cases {
        let n: usize = size_text.parse().expect("size");
        let mut rng = stream(
            20260726,
            &[
                Part::Str("vectors"),
                Part::Str("shuffle"),
                Part::Int(n as u64),
            ],
        );
        let mut items: Vec<u8> = (0..n as u8).collect();
        rng.shuffle(&mut items);

        let head: Vec<u64> = case["head"]
            .as_array()
            .expect("head")
            .iter()
            .map(u64_of)
            .collect();
        let got_head: Vec<u64> = items
            .iter()
            .take(head.len())
            .map(|&x| u64::from(x))
            .collect();
        assert_eq!(got_head, head, "shuffle({n}) head");
        // The head alone would not catch a direction error in the tail, so the whole
        // permutation is digested. This is the vector that fails if someone "tidies"
        // Fisher-Yates into its ascending form.
        assert_eq!(
            hash128_hex(&items),
            case["digest"].as_str().expect("digest"),
            "shuffle({n}) digest"
        );
        assert_eq!(
            rng.state,
            u64_of(&case["final_state"]),
            "shuffle({n}) state"
        );
    }
}

#[test]
fn stream_derivation_reproduces() {
    let v = vectors();
    let cases = v["rng"]["derive"].as_object().expect("derive");
    assert!(!cases.is_empty());
    for (key, expected) in cases {
        // Keys are JSON arrays so `[]`, `[""]`, `[1]` and `["1"]` stay distinguishable --
        // exactly the cases the `0x1f` separator and the int/str split exist to separate.
        let parsed: Vec<Value> = serde_json::from_str(key).expect("key is a JSON array");
        let owned: Vec<Part> = parsed
            .iter()
            .map(|p| match p {
                Value::String(s) => Part::Str(s.as_str()),
                Value::Number(n) => Part::Int(n.as_u64().expect("integer part")),
                other => panic!("unexpected part {other:?}"),
            })
            .collect();
        let expected = expected.as_array().expect("(initstate, initseq)");
        assert_eq!(
            derive(20260726, &owned),
            (u64_of(&expected[0]), u64_of(&expected[1])),
            "derive({key})"
        );
    }
}

#[test]
fn hashing_reproduces() {
    let v = vectors();
    for (hex, expected) in v["hashing"]["hash64"].as_object().expect("hash64") {
        let data = decode_hex(hex);
        assert_eq!(hash64(&data), u64_of(expected), "hash64({hex})");
    }
    for (hex, expected) in v["hashing"]["hash128"].as_object().expect("hash128") {
        let data = decode_hex(hex);
        assert_eq!(
            hash128_hex(&data),
            expected.as_str().expect("hex digest"),
            "hash128({hex})"
        );
    }
}

fn decode_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
        .collect()
}
