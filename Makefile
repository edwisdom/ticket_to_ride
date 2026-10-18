.PHONY: setup fmt lint type test test-all cov bench bench-check board layout vectors tb clean clean-rust help \
        rust-fmt rust-lint rust-test rust-sweep rust-cov mutants

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:  ## Install everything (builds the Rust core) and wire up pre-commit
	uv sync --all-extras --dev
	uv run pre-commit install

# --- Rust -----------------------------------------------------------------
#
# The engine is crates/ttr-core. `uv sync` (and `uv run`) builds its Python extension,
# ttr_rust, as a workspace member in the release profile, rebuilding whenever a Rust source
# changes -- see the cache-keys note in crates/ttr-py/pyproject.toml.

rust-fmt:  ## Format the Rust sources
	cargo fmt --all

rust-lint:  ## Rust formatting + clippy, warnings are errors
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

rust-test:  ## Every Rust test's fast tier, the golden-corpus prefix included (~20 s)
	cargo test --workspace

rust-sweep:  ## The #[ignore]d sweeps: the whole corpus, the naive references at width (~3 min)
	cargo test --workspace --release -- --ignored

# Coverage needs llvm-tools, which a Homebrew Rust lacks; this uses rustup's toolchain.
rust-cov:  ## ttr-core line coverage (rustup toolchain + cargo-llvm-cov)
	RUSTUP_TOOLCHAIN=stable cargo llvm-cov -p ttr-core --summary-only

mutants:  ## Mutation-test rules, scoring, graph and the encoder (hours; see .cargo/mutants.toml)
	cargo mutants -j 4

fmt:  ## Format and autofix
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lints
	uv run ruff check .
	uv run ruff format --check .

type:  ## Type check (needs --all-extras so torch-importing modules resolve)
	uv run ty check

test:  ## Fast tests (unit + property), parallel
	uv run pytest -n auto

test-all:  ## Everything including slow integration tests
	uv run pytest -n auto -m ""

cov:  ## Fast tests with coverage
	uv run pytest --cov --cov-report=term-missing

bench:  ## Run benchmarks and save a new baseline
	uv run pytest -m bench --benchmark-autosave

# Needs a quiet machine. Running it alongside anything else -- the nightly sweeps are the
# obvious offender -- reports a 40-80% "regression" that is
# purely core contention. Measured, not hypothesised.
bench-check:  ## Fail if throughput regressed >20% vs the saved baseline (M2, idle machine)
	uv run pytest -m bench --benchmark-compare --benchmark-compare-fail=mean:20%

board:  ## Regenerate the board data (RULES.md -> usa.toml, board_gen.rs)
	uv run python tools/gen_board.py

layout:  ## Regenerate the board layouts the client draws (data/layout/*.toml -> *.json)
	uv run python tools/gen_layout.py

corpus:  ## Verify the golden corpus, every game
	cargo run -q --release -p ttr-tools --bin gen-corpus -- --check

tb:  ## Launch tensorboard over all runs
	uv run tensorboard --logdir runs/

clean:  ## Remove tool caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache .benchmarks
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-rust:  ## Remove the Rust build directory (a few GB once benches have run)
	cargo clean
