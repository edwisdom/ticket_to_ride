.PHONY: setup fmt lint type test test-all cov bench bench-check board vectors tb clean help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:  ## Install everything and wire up pre-commit
	uv sync --all-extras --dev
	uv run pre-commit install

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

bench-check:  ## Fail if throughput regressed >20% vs the saved baseline (M2 only)
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
