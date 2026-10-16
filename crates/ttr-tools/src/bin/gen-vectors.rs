//! Write or verify the frozen contract vectors in `tests/golden/contract_vectors.json`.
//!
//! ```text
//! cargo run -p ttr-tools --bin gen-vectors -- --check
//! cargo run -p ttr-tools --bin gen-vectors -- --write
//! ```
//!
//! The port of `tools/gen_vectors.py`. The vectors pin the PRNG, bounded draws, the shuffle,
//! seed-stream derivation and the hash primitive (docs/CONTRACT.md §1 and §3); every recorded
//! replay depends on those bytes, so a diff here is never "just regenerate it".
//!
//! `--check` rebuilds the vectors from `ttr_core::rng` and `ttr_core::hashing` and compares
//! them with the file **byte for byte**. The rendering reproduces Python's
//! `json.dumps(obj, indent=2, sort_keys=True) + "\n"` exactly -- serde_json's two-space
//! pretty printer over recursively sorted keys is the same layout -- so a deleted section or
//! a reformatted file fails, not only a changed value. The one place the two serializers
//! differ is escaping: Python's `ensure_ascii` writes non-ASCII and DEL as `\uXXXX` and
//! serde_json writes them raw. Nothing in the vectors is outside printable ASCII, and
//! [`render`] asserts that it stays so rather than trusting it.
//!
//! `--write` refuses unless `ttr_core::CONTRACT_VERSION` differs from the file's
//! `contract_version`. The vectors are frozen: rewriting them at the same version would be
//! accepting a contract change by regenerating the evidence. Bump `CONTRACT_VERSION` first,
//! and accept that every replay recorded under the old version is now unreplayable.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{Map, Value, json};
use ttr_core::CONTRACT_VERSION;
use ttr_core::hashing::{hash64, hash128_hex};
use ttr_core::rng::{Part, Pcg32, derive, stream};

/// The root every vector stream is derived from.
const ROOT: u64 = 20260726;

fn vectors_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/contract_vectors.json")
}

/// `path` relative to the repository root where it is under it, for messages.
fn shown(path: &Path) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    path.strip_prefix(&root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// PRNG, bounded draws, shuffle, and stream derivation.
fn rng_vectors() -> Value {
    // The published pcg32-demo vector. Any correct PCG-XSH-RR 64/32 reproduces it, so it
    // validates the port against the upstream reference rather than against ourselves.
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

    // A shuffle vector per deck size actually materialized: USA 110, mini 54, the tickets.
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
            json!({"head": &items[..12], "digest": hash128_hex(&items), "final_state": rng.state}),
        );
    }

    // Keys are JSON so `[]`, `[""]` and `[1]` vs `["1"]` stay distinguishable -- a "|"-joined
    // key silently collapses exactly the cases the separator exists to separate.
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
        derived.insert(json_key(parts), json!([initstate, initseq]));
    }

    json!({
        "reference_42_54": reference,
        "seeded": seeded,
        "below": below,
        "shuffle": shuffles,
        "derive": derived,
    })
}

/// `json.dumps(list(parts))`, with Python's default `", "` separator: `["agent", 3]`.
fn json_key(parts: &[Part]) -> String {
    let items: Vec<String> = parts
        .iter()
        .map(|part| match part {
            Part::Str(s) => Value::from(*s).to_string(),
            Part::Int(i) => i.to_string(),
        })
        .collect();
    format!("[{}]", items.join(", "))
}

fn hashing_vectors() -> Value {
    let cases: [Vec<u8>; 4] = [
        Vec::new(),
        vec![0],
        b"ticket to ride".to_vec(),
        (0..=255).collect(),
    ];
    let (mut h64, mut h128) = (Map::new(), Map::new());
    for case in &cases {
        let key: String = case.iter().map(|b| format!("{b:02x}")).collect();
        h64.insert(key.clone(), json!(hash64(case)));
        h128.insert(key, json!(hash128_hex(case)));
    }
    json!({"hash64": h64, "hash128": h128})
}

fn build() -> Value {
    json!({
        "contract_version": CONTRACT_VERSION,
        "rng": rng_vectors(),
        "hashing": hashing_vectors(),
    })
}

/// Python's `json.dumps(value, indent=2, sort_keys=True) + "\n"`, byte for byte.
///
/// Sorted explicitly rather than by relying on serde_json's default map, which keeps
/// insertion order instead if any crate in the build enables its `preserve_order` feature.
fn render(mut value: Value) -> String {
    value.sort_all_objects();
    let mut out = serde_json::to_string_pretty(&value).expect("a JSON value always serializes");
    out.push('\n');
    assert!(
        out.bytes()
            .all(|b| b == b'\n' || (b' '..=b'~').contains(&b)),
        "the vectors left printable ASCII, where serde_json and Python escape differently"
    );
    out
}

/// Where two renderings first part ways, for a failure that says more than "differs".
fn first_difference(stored: &str, built: &str) -> String {
    for (i, (a, b)) in stored.lines().zip(built.lines()).enumerate() {
        if a != b {
            return format!("line {}: the file has {a:?}, the engine gives {b:?}", i + 1);
        }
    }
    let (a, b) = (stored.lines().count(), built.lines().count());
    if a == b {
        "line endings or the final newline differ".to_string()
    } else {
        format!("the file has {a} lines, the engine gives {b}")
    }
}

fn check(path: &Path) -> Result<String, String> {
    let stored = std::fs::read_to_string(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("missing {}; run --write", shown(path)),
        _ => format!("{}: {e}", shown(path)),
    })?;
    let built = render(build());
    if stored != built {
        return Err(format!(
            "CONTRACT VIOLATION: the frozen test vectors no longer reproduce ({}).\n\
             This is not a regeneration -- every recorded replay depends on these bytes.\n\
             If the change is intended, bump CONTRACT_VERSION and run --write.",
            first_difference(&stored, &built)
        ));
    }
    Ok(format!(
        "contract vectors reproduce at version {CONTRACT_VERSION}"
    ))
}

fn write(path: &Path) -> Result<String, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let recorded = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["contract_version"].as_u64())
                .ok_or_else(|| {
                    format!(
                        "{}: no readable contract_version; refusing to overwrite it",
                        shown(path)
                    )
                })?;
            if recorded == u64::from(CONTRACT_VERSION) {
                return Err(format!(
                    "{}: written at contract version {recorded}, the current CONTRACT_VERSION; \
                     refusing to overwrite. The vectors are frozen: rewriting them at the same \
                     version would accept a contract change by regenerating the evidence. \
                     Bump CONTRACT_VERSION first.",
                    shown(path)
                ));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", shown(path))),
    }
    std::fs::write(path, render(build())).map_err(|e| format!("{}: {e}", shown(path)))?;
    Ok(format!(
        "wrote {} at contract version {CONTRACT_VERSION}",
        shown(path)
    ))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let (write_flag, check_flag) = (has("--write"), has("--check"));
    if write_flag == check_flag {
        eprintln!("usage: gen-vectors (--check | --write)");
        return ExitCode::from(2);
    }
    let path = vectors_path();
    let result = if write_flag {
        write(&path)
    } else {
        check(&path)
    };
    match result {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch copy of a file, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str, contents: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ttr-gen-vectors-{}-{name}.json",
                std::process::id()
            ));
            std::fs::write(&path, contents).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Byte for byte, so a *deleted* section fails too.
    ///
    /// `crates/ttr-core/tests/contract_vectors.rs` iterates whatever the file contains; only
    /// this catches a vector that quietly stopped being checked. The port of
    /// `test_rng.py::test_the_whole_vector_file_reproduces`.
    #[test]
    fn the_whole_vector_file_reproduces() {
        if let Err(e) = check(&vectors_path()) {
            panic!("{e}");
        }
    }

    /// The layout claim the byte-for-byte check rests on, against output from CPython's
    /// `json.dumps(obj, indent=2, sort_keys=True) + "\n"`: empty containers inline, keys
    /// sorted at every depth by code point, u64s exact, quotes and backslashes escaped.
    #[test]
    fn render_matches_python_json_dumps() {
        let value = json!({
            "b": [],
            "a": {},
            "c": {"z": [1, u64::MAX], "y": "q\"uo\\te"},
            "[\"x\", 1]": 0,
        });
        let python = r#"{
  "[\"x\", 1]": 0,
  "a": {},
  "b": [],
  "c": {
    "y": "q\"uo\\te",
    "z": [
      1,
      18446744073709551615
    ]
  }
}
"#;
        assert_eq!(render(value), python);
        assert_eq!(
            json_key(&[
                Part::Str("agent"),
                Part::Int(3),
                Part::Str("game"),
                Part::Int(7)
            ]),
            r#"["agent", 3, "game", 7]"#
        );
        assert_eq!(json_key(&[]), "[]");
    }

    /// The check is not vacuous: one changed digit, or one deleted section, fails it.
    #[test]
    fn a_drifted_or_deleted_vector_is_a_contract_violation() {
        let current = std::fs::read_to_string(vectors_path()).unwrap();
        // The first pcg32-demo output, a15c02b7.
        let changed = current.replacen("2707161783", "2707161784", 1);
        assert_ne!(changed, current);
        let drifted = Scratch::new("drifted", &changed);
        let err = check(&drifted.0).expect_err("a changed value passed");
        assert!(err.contains("CONTRACT VIOLATION"), "{err}");

        let mut value: Value = serde_json::from_str(&current).unwrap();
        value["rng"].as_object_mut().unwrap().remove("derive");
        let deleted = Scratch::new("deleted", &render(value));
        let err = check(&deleted.0).expect_err("a deleted section passed");
        assert!(err.contains("CONTRACT VIOLATION"), "{err}");
    }

    #[test]
    fn write_refuses_the_current_contract_version() {
        let current = std::fs::read_to_string(vectors_path()).unwrap();
        let scratch = Scratch::new("same", &current);
        let err = write(&scratch.0).expect_err("a rewrite at the same version was allowed");
        assert!(err.contains("refusing to overwrite"), "{err}");
        assert_eq!(std::fs::read_to_string(&scratch.0).unwrap(), current);
    }

    #[test]
    fn write_proceeds_once_the_contract_version_moved() {
        let other = u64::from(CONTRACT_VERSION) + 1;
        let scratch = Scratch::new("bumped", &format!("{{\"contract_version\": {other}}}"));
        write(&scratch.0).unwrap();
        assert_eq!(
            std::fs::read_to_string(&scratch.0).unwrap(),
            render(build())
        );
        check(&scratch.0).unwrap();
    }

    #[test]
    fn write_refuses_a_file_whose_version_cannot_be_read() {
        let scratch = Scratch::new("garbled", "not json");
        let err = write(&scratch.0).expect_err("overwrote an unreadable file");
        assert!(err.contains("refusing to overwrite"), "{err}");
    }
}
