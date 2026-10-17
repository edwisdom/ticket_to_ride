//! Write or verify the golden corpus in `tests/golden/corpus/`.
//!
//! ```text
//! cargo run --release -p ttr-tools --bin gen-corpus -- --check [--play-only]
//! cargo run --release -p ttr-tools --bin gen-corpus -- --write
//! ```
//!
//! `--check` re-derives every game with the live crate and compares. `--play-only` leaves
//! out the observation digests and the recorded `OBS_VERSION`: what must still hold across
//! an `OBS_VERSION` bump, and so what to run before re-signing for one. `--write` refuses to
//! overwrite a file recorded under the current `CONTRACT_VERSION` and `OBS_VERSION`: the
//! corpus is a frozen fixture, and rewriting it at the same versions would be accepting a
//! behaviour change by regenerating the evidence. A deliberate re-sign at the same versions
//! needs `--allow-same-version`, and a worklog entry saying why.

use std::process::ExitCode;
use std::time::Instant;

use ttr_tools::corpus::{
    Corpus, Scope, check, configs, corpus_dir, corpus_path, current_header, generate,
};
use ttr_tools::engine::Live;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let (write, verify) = (has("--write"), has("--check"));
    if write == verify {
        eprintln!("usage: gen-corpus (--check [--play-only] | --write [--allow-same-version])");
        return ExitCode::from(2);
    }
    let play_only = has("--play-only");

    let mut failed = false;
    for (map, n_players) in configs() {
        let started = Instant::now();
        let path = corpus_path(map, n_players);
        if write {
            if let Ok(existing) = Corpus::load(map, n_players) {
                let now = current_header(map, n_players);
                let same = existing.header.contract_version == now.contract_version
                    && existing.header.obs_version == now.obs_version;
                if same && !has("--allow-same-version") {
                    eprintln!(
                        "{}: recorded at the current CONTRACT_VERSION and OBS_VERSION; \
                         refusing to overwrite (see --allow-same-version)",
                        path.display()
                    );
                    failed = true;
                    continue;
                }
            }
            let corpus = generate(map, n_players);
            let bytes = corpus.to_bytes();
            std::fs::create_dir_all(corpus_dir()).expect("creating tests/golden/corpus");
            std::fs::write(&path, &bytes).expect("writing the corpus");
            let steps: u64 = corpus.random.iter().map(|g| u64::from(g.steps)).sum();
            println!(
                "{map} {n_players}P: wrote {} random + {} heuristic games ({steps} random steps), \
                 {} bytes, {:.1}s",
                corpus.random.len(),
                corpus.heuristic.len(),
                bytes.len(),
                started.elapsed().as_secs_f64()
            );
        } else {
            let corpus = match Corpus::load(map, n_players) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{e}");
                    failed = true;
                    continue;
                }
            };
            let mut now = current_header(map, n_players);
            if play_only {
                now.obs_version = corpus.header.obs_version;
            }
            if corpus.header != now {
                eprintln!(
                    "{map} {n_players}P: header {:?} != current {:?}",
                    corpus.header, now
                );
                failed = true;
                continue;
            }
            let scope = if play_only {
                Scope::all().without_observations()
            } else {
                Scope::all()
            };
            match check(&Live::new(map, n_players), &corpus, &scope) {
                Ok(games) => println!(
                    "{map} {n_players}P: {games} games reproduce, {:.1}s",
                    started.elapsed().as_secs_f64()
                ),
                Err(e) => {
                    eprintln!("REGRESSION: {e}");
                    failed = true;
                }
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
