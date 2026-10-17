//! The golden corpus reproduces. A prefix of every section under plain `cargo test`; every
//! game behind `#[ignore]` (`cargo test --release -p ttr-tools -- --ignored`).

use ttr_tools::corpus::{
    Corpus, HEURISTIC_SEEDS, OBS_SEEDS, RANDOM_SEEDS, Scope, check, configs, current_header,
};

fn load(map: &str, n: usize) -> Corpus {
    Corpus::load(map, n).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn every_configuration_has_a_complete_corpus_under_the_current_versions() {
    for (map, n) in configs() {
        let corpus = load(map, n);
        assert_eq!(
            corpus.header,
            current_header(map, n),
            "{map} {n}P was recorded under a different contract, observation version, \
             board or rules"
        );
        assert_eq!(corpus.random.len(), RANDOM_SEEDS as usize);
        assert_eq!(corpus.random_obs.len(), OBS_SEEDS as usize);
        assert_eq!(corpus.heuristic.len(), HEURISTIC_SEEDS as usize);
    }
}

#[test]
fn a_prefix_of_every_section_reproduces() {
    for (map, n) in configs() {
        check(&load(map, n), &Scope::prefix(300, 6, 4)).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn a_tampered_corpus_is_caught() {
    // Guard the guard, once per section: flip one recorded value and the check must say
    // which game, in which section.
    let corpus = load("mini", 2);
    let scope = Scope::prefix(3, 3, 3);

    let mut bad = corpus.clone();
    bad.random[2].chain ^= 1;
    let err = check(&bad, &scope).expect_err("a flipped chain went unnoticed");
    assert!(
        err.contains("random seed 2") && err.contains("chain"),
        "{err}"
    );

    let mut bad = corpus.clone();
    bad.random_obs[1] ^= 1;
    let err = check(&bad, &scope).expect_err("a flipped digest went unnoticed");
    assert!(
        err.contains("random seed 1") && err.contains("observation"),
        "{err}"
    );

    let mut bad = corpus;
    bad.heuristic[0].actions.pop();
    let err = check(&bad, &scope).expect_err("a truncated game went unnoticed");
    assert!(err.contains("heuristic seed 0"), "{err}");
}

#[test]
#[ignore = "every game: run with --release -- --ignored"]
fn the_whole_corpus_reproduces() {
    for (map, n) in configs() {
        check(&load(map, n), &Scope::all()).unwrap_or_else(|e| panic!("{e}"));
    }
}
