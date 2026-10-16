//! Write or verify the golden replay corpus in `tests/golden/replays.bin`.
//!
//! ```text
//! cargo run -p ttr-tools --bin gen-replays -- --check
//! cargo run -p ttr-tools --bin gen-replays -- --write
//! ```
//!
//! The port of `tools/gen_replays.py`. The corpus is a checked-in set of finished games
//! across every map and seat count, each carrying the actions that produced it and the
//! state hash and scores it must produce again -- the strongest regression guard in the
//! project. A change to the rules, the draw procedure, the PRNG or the scoring shows up here
//! as a concrete failing game rather than as a subtly different win rate three weeks later.
//!
//! `--check` confirms every game is in place, then replays each record through
//! `ttr_core::replay::replay`, which checks the contract version, the board and the rules
//! and proves the final `state_hash()` and the final scores. It does not regenerate, as the
//! Python tool did: a replay proves the engine still plays the recorded games whoever chose
//! them, and the file was written by the Python engine, its greedy half by a throwaway
//! Python H1 that is being deleted.
//!
//! `--write` refuses unless `ttr_core::CONTRACT_VERSION` differs from the version the stored
//! records carry. A failing `--check` means the engine plays a different game than it did,
//! and the only correct responses are to fix the regression or, if the contract really
//! moved, to bump `CONTRACT_VERSION` and rewrite -- never to regenerate the evidence at the
//! same version. When allowed it rebuilds the Python layout: per (map, seat count), `SEEDS`
//! random games then `SEEDS` greedy ones.
//!
//! **The greedy half is Rust's H1** (`ttr_core::heuristic`, `Tier::H1`), seeded as Python's
//! was (agent seed `10 + seat`, re-seated per game), in place of the Python throwaway H1.
//! Rust's H1 is a transcription of that agent, and at this port the two agree on every one
//! of the stored games: the generator rebuilds `replays.bin` byte for byte, which a test
//! pins, so a rewrite would differ only where the engine itself moved.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ttr_core::CONTRACT_VERSION;
use ttr_core::config::RuleConfig;
use ttr_core::heuristic::{Heuristic, HeuristicParams, Tier};
use ttr_core::replay::{Replay, pack, record, replay, unpack};
use ttr_core::rng::{Part, stream};
use ttr_core::state::Game;
use ttr_tools::corpus::configs;

/// Seeds per configuration and drive. Small enough to stay a fast test, wide enough that
/// reshuffles, locomotive flushes and the end trigger all appear somewhere in the corpus.
const SEEDS: u64 = 6;

fn replays_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/replays.bin")
}

/// `path` relative to the repository root where it is under it, for messages.
fn shown(path: &Path) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    path.strip_prefix(&root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The `(map, seats, seed)` of every record, in file order: per configuration, `SEEDS`
/// random games then `SEEDS` greedy ones.
fn layout() -> Vec<(String, u8, u64)> {
    let mut out = Vec::new();
    for (map, n_players) in configs() {
        for _drive in ["random", "greedy"] {
            out.extend((0..SEEDS).map(|seed| (map.to_string(), n_players as u8, seed)));
        }
    }
    out
}

/// Random games and greedy games, across every map and seat count, in [`layout`] order.
fn build() -> Vec<Replay> {
    let mut records = Vec::new();
    for (map, n_players) in configs() {
        let cfg =
            RuleConfig::new(map, n_players).unwrap_or_else(|e| panic!("{map} {n_players}P: {e}"));
        let game = Game::new(cfg).unwrap_or_else(|e| panic!("{map} {n_players}P: {e}"));
        records.extend(random_games(&game));
        // A greedy game too: random play rarely reaches long routes or a deep endgame.
        records.extend(greedy_games(&game));
    }
    records
}

/// Uniform over the legal actions, from each seed's own `("policy",)` stream.
fn random_games(game: &Game) -> Vec<Replay> {
    (0..SEEDS)
        .map(|seed| {
            let mut state = game.new_initial_state(seed);
            let mut rng = stream(seed, &[Part::Str("policy")]);
            while !state.is_terminal() {
                let action = state
                    .sample_legal(&mut rng)
                    .expect("a live position has a legal action");
                state.step(action).expect("a sampled action is legal");
            }
            record(&state).expect("a finished game with history")
        })
        .collect()
}

/// H1 self-play: one agent per seat, built once per configuration and re-seated per game.
fn greedy_games(game: &Game) -> Vec<Replay> {
    let mut agents: Vec<Heuristic> = (0..usize::from(game.rules.n_players))
        .map(|seat| Heuristic::new(Tier::H1, HeuristicParams::default(), 10 + seat as u64))
        .collect();
    (0..SEEDS)
        .map(|seed| {
            for (seat, agent) in agents.iter_mut().enumerate() {
                agent.begin_game(seat, seed);
            }
            let mut state = game.new_initial_state(seed);
            while !state.is_terminal() {
                let seat = usize::from(state.current_player());
                let action = agents[seat].act(&mut state);
                state.step(action).expect("H1 plays a legal action");
            }
            record(&state).expect("a finished game with history")
        })
        .collect()
}

/// Replay every stored record. Returns how many reproduced.
///
/// The records must also sit in [`layout`] order, every one present: replaying alone would
/// pass a corpus with games missing, down to an empty one.
fn check(path: &Path) -> Result<usize, String> {
    let data = std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("missing {}; run --write", shown(path)),
        _ => format!("{}: {e}", shown(path)),
    })?;
    let records = unpack(&data).map_err(|e| format!("{}: {e}", shown(path)))?;

    let found: Vec<(String, u8, u64)> = records
        .iter()
        .map(|r| (r.map_name.clone(), r.n_players, r.seed))
        .collect();
    let expected = layout();
    if found != expected {
        let at = found
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(found.len().min(expected.len()));
        return Err(format!(
            "{}: {} records where the layout has {}; first difference at #{at}: found {:?}, \
             expected {:?}",
            shown(path),
            found.len(),
            expected.len(),
            found.get(at),
            expected.get(at)
        ));
    }

    let failures: Vec<String> = records
        .iter()
        .enumerate()
        .filter_map(|(i, rec)| {
            replay(rec, None).err().map(|e| {
                format!(
                    "  #{i} {} {}P seed {}: {e}",
                    rec.map_name, rec.n_players, rec.seed
                )
            })
        })
        .collect();
    if !failures.is_empty() {
        return Err(format!(
            "REGRESSION: the engine no longer plays {} of the {} recorded games.\n{}\n\
             Fix it, or -- if the contract really moved -- bump CONTRACT_VERSION and rerun \
             with --write.",
            failures.len(),
            records.len(),
            failures[..failures.len().min(5)].join("\n")
        ));
    }
    Ok(records.len())
}

fn write(path: &Path) -> Result<String, String> {
    match std::fs::read(path) {
        Ok(data) => {
            let stored = unpack(&data).map_err(|e| {
                format!(
                    "{}: {e}; refusing to overwrite a corpus whose version cannot be read",
                    shown(path)
                )
            })?;
            if stored
                .iter()
                .any(|r| r.contract_version == CONTRACT_VERSION)
            {
                return Err(format!(
                    "{}: recorded at the current CONTRACT_VERSION ({CONTRACT_VERSION}); refusing \
                     to overwrite. The corpus is frozen: regenerating it at the same version \
                     would accept a behaviour change by rewriting the evidence. Bump \
                     CONTRACT_VERSION if the contract really moved.",
                    shown(path)
                ));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", shown(path))),
    }
    let records = build();
    let blob = pack(&records).map_err(|e| e.to_string())?;
    std::fs::write(path, &blob).map_err(|e| format!("{}: {e}", shown(path)))?;
    Ok(format!(
        "wrote {} replays, {} bytes, at contract version {CONTRACT_VERSION}",
        records.len(),
        blob.len()
    ))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let (write_flag, check_flag) = (has("--write"), has("--check"));
    if write_flag == check_flag {
        eprintln!("usage: gen-replays (--check | --write)");
        return ExitCode::from(2);
    }
    let path = replays_path();
    let result = if write_flag {
        write(&path)
    } else {
        check(&path).map(|n| format!("{n} golden replays reproduce"))
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

    /// A scratch file, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str, contents: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("ttr-gen-replays-{}-{name}.bin", std::process::id()));
            std::fs::write(&path, contents).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn label(rec: &Replay) -> String {
        format!("{} {}P seed {}", rec.map_name, rec.n_players, rec.seed)
    }

    /// The port of `test_golden_replays.py::test_the_corpus_is_not_stale`, which ran the
    /// tool's `--check`: this runs the check `--check` runs. Python's check regenerated the
    /// file; that half of the promise is `a_rewrite_after_a_version_bump_reproduces_the_file`.
    #[test]
    fn the_corpus_is_not_stale() {
        let stored = unpack(&std::fs::read(replays_path()).unwrap()).unwrap();
        match check(&replays_path()) {
            Ok(n) => assert_eq!(n, stored.len()),
            Err(e) => panic!("{e}"),
        }
    }

    /// Regenerating must produce byte-identical games.
    ///
    /// After a simulated contract bump, `--write` must rebuild today's file exactly. The
    /// generator otherwise runs only at a real bump, when there is nothing left to compare it
    /// against. This also pins *how* the games were chosen -- `sample_legal`'s draw order and
    /// H1's play -- which replaying alone does not. If it fails while
    /// `the_corpus_is_not_stale` passes, the engine still plays every recorded game and only
    /// the choosing moved: not a contract change, and no reason to rewrite the corpus.
    #[test]
    fn a_rewrite_after_a_version_bump_reproduces_the_file() {
        let current = std::fs::read(replays_path()).unwrap();
        let stored = unpack(&current).unwrap();
        let bumped: Vec<Replay> = stored
            .iter()
            .map(|r| Replay {
                contract_version: CONTRACT_VERSION.wrapping_add(1),
                ..r.clone()
            })
            .collect();
        let scratch = Scratch::new("bumped", &pack(&bumped).unwrap());
        write(&scratch.0).unwrap();

        let fresh = std::fs::read(&scratch.0).unwrap();
        if fresh != current {
            let rebuilt = unpack(&fresh).unwrap();
            let first = stored
                .iter()
                .zip(&rebuilt)
                .position(|(a, b)| a != b)
                .map(|i| label(&stored[i]));
            panic!(
                "the generator no longer rebuilds tests/golden/replays.bin: {} records \
                 against {}, first difference at {first:?}",
                rebuilt.len(),
                stored.len()
            );
        }
    }

    /// Replaying alone passes vacuously on a corpus with games missing.
    #[test]
    fn a_corpus_with_a_game_missing_fails_the_check() {
        let mut stored = unpack(&std::fs::read(replays_path()).unwrap()).unwrap();
        stored.remove(stored.len() / 2);
        let scratch = Scratch::new("missing", &pack(&stored).unwrap());
        let err = check(&scratch.0).expect_err("a corpus with a game missing passed");
        assert!(err.contains("where the layout has"), "{err}");
        let empty = Scratch::new("empty", &pack(&[]).unwrap());
        assert!(check(&empty.0).is_err(), "an empty corpus passed");
    }

    #[test]
    fn write_refuses_a_corpus_at_the_current_contract_version() {
        let current = std::fs::read(replays_path()).unwrap();
        let scratch = Scratch::new("same", &current);
        let err = write(&scratch.0).expect_err("a rewrite at the same version was allowed");
        assert!(err.contains("refusing to overwrite"), "{err}");
        assert_eq!(std::fs::read(&scratch.0).unwrap(), current);
    }

    #[test]
    fn write_refuses_a_file_it_cannot_read() {
        let scratch = Scratch::new("garbled", b"not a corpus");
        let err = write(&scratch.0).expect_err("overwrote an unreadable corpus");
        assert!(err.contains("refusing to overwrite"), "{err}");
    }
}
