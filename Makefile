.PHONY: setup fmt lint type test test-all cov bench bench-check board vectors tb clean clean-rust help \
        rust rust-fmt rust-lint rust-test

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:  ## Install everything and wire up pre-commit
	uv sync --all-extras --dev
	uv run pre-commit install

# --- Rust -----------------------------------------------------------------
#
# Deliberately not part of `uv sync`: the pure-Python edit loop and the torch-free CI job
# stay Rust-free, and the ~8s release build only happens when asked for. Tests needing the
# extension skip when it is absent -- unless TTR_REQUIRE_RUST is set, which CI does.

rust:  ## Build the Rust core and install it into .venv (release, the profile self-play uses)
	uv run maturin develop --release -m crates/ttr-py/Cargo.toml

rust-fmt:  ## Format the Rust sources
	cargo fmt --all

rust-lint:  ## Rust formatting + clippy, warnings are errors
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

rust-test:  ## ttr-core's standalone tests -- no Python interpreter involved
	cargo test --workspace

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

# Needs a quiet machine. Running it alongside anything else -- the nightly differential
# sweep under `-n auto` is the obvious offender -- reports a 40-80% "regression" that is
# purely core contention. Measured, not hypothesised.
bench-check:  ## Fail if throughput regressed >20% vs the saved baseline (M2, idle machine)
	uv run pytest -m bench --benchmark-compare --benchmark-compare-fail=mean:20%

board:  ## Regenerate board data and the observation feature-spec table
	uv run python tools/gen_board.py
	uv run python tools/gen_obs_spec.py

vectors:  ## Verify the frozen contract vectors still reproduce (see docs/CONTRACT.md)
	uv run python tools/gen_vectors.py --check
	uv run python tools/gen_replays.py --check

tb:  ## Launch tensorboard over all runs
	uv run tensorboard --logdir runs/

clean:  ## Remove tool caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache .benchmarks
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-rust:  ## Remove the Rust build directory (a few GB once benches have run)
	cargo clean
