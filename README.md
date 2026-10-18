# Ticket to Ride — engine + self-play RL

A fast, exactly-correct Ticket to Ride (base USA map) game engine, and a platform for
training reinforcement learning agents against it via self-play.

**Goal:** iterate on algorithms until agents beat strong heuristic opponents, then humans.

## Status

| Phase | | |
| --- | --- | --- |
| 0 | Project scaffolding | done |
| 1 | Python reference engine | done |
| 2 | Rust core + differential harness | done |
| 3 | Heuristic agents + arena + Elo | done |
| 3.1 | Retire the Python engine; Rust becomes its own oracle | done |
| 3.2 | Rust performance (observation encoder) | done |
| 3.3 | Modal cloud runs | done |
| 3.4 | PPO spike on TTR-mini (tentative) | done |
| 4 | Web client: play the heuristics in a browser | engine bindings and server done (4.1, 4.2); client next |
| 5 | Determinization + ISMCTS | |
| 6 | PPO self-play + league | |
| 7 | ISMCTS + learned priors | |
| 8 | Scaling, public deployment, human gauntlet | |

The engine plays 2–5 players on the USA map and 2–4 on TTR-mini, and it is Rust
(`crates/ttr-core`). A Python reference engine came first and the two were byte-identical
over 700,000 games compared at every step; the Python engine is retired, and the Rust
engine is checked against:

- **a golden corpus** -- 71,400 games, hash-chained at every position, with observation
  digests -- first recorded and checked by the Python engine;
- **naive in-crate references** -- brute-force Steiner, exhaustive longest trail, a
  6300-way legality scan, a one-line-per-feature observation encoder;
- **mutation testing** over the rules, scoring, graph and observation code.

| | µs/step | full USA 2P games/s |
| --- | --- | --- |
| Rust, one thread | 0.14 | ~45,000 |
| Rust, 8 performance cores | 0.021 | ~303,000 |
| *(the retired Python engine)* | *6.6* | *~980* |

A training step also needs an observation (3,355 floats on USA) and a legal-action mask.
The encoder costs 8.9 µs per observation from scratch and 4.6 µs with its per-seat cache,
on one thread (126.9 before Phase 3.2), and `VecEnv` -- one call per batch, buffers
written in place -- runs observe + mask + step at **~980,000 env-steps/s** over 1024 USA 2P
games (`ttr bench --suite vecenv`).

On an M2 Max. 362 Rust tests (and 7 `#[ignore]`d sweeps) and 195 Python tests; ttr-core line
coverage 96%.

Five scripted agents, H0–H4, rated over 10,000 paired games in 12.6 s. **H3 is the permanent
Elo zero**, and its identity is a hash of what it *plays* over a frozen probe set rather than
of its constants — a params hash would leave a reordered tiebreak or a fixed bug free to move
the anchor and silently re-base every rating ever recorded.

| agent | usa 2P | mini 2P | usa 4P |
| --- | --- | --- | --- |
| h4 | −4 [−20, +14] | +21 [+8, +33] | −21 [−28, −14] |
| **h3** | **0** (anchor) | **0** | **0** |
| h2 | −100 [−119, −82] | −117 [−137, −99] | −120 [−129, −111] |
| h1 | −726 [−778, −684] | −545 [−582, −510] | −724 [−744, −706] |
| h0 | −1276 [−1348, −1217] | −816 [−858, −782] | −1252 [−1279, −1223] |

Bradley–Terry MLE, 95% intervals from a bootstrap that resamples **seed blocks** rather than
games — games inside a block share a deck, and treating them as independent inflates
significance by about √P. Both seatings of every block are played, never one mirrored.

## Quickstart

```bash
make setup             # uv sync --all-extras --dev (builds the Rust core), pre-commit hooks
make rust-test         # every Rust test's fast tier, a golden-corpus prefix included (~20 s)
make test              # the Python suite (CLI, eval, bindings), no torch
make lint type         # ruff + ty
make rust-sweep        # the #[ignore]d sweeps: the whole corpus, the naive references at width
make corpus            # every golden-corpus game still reproduces
uv run ttr map         # board data and the invariants it satisfies
uv run ttr bench       # engine throughput, one thread and all performance cores
uv run ttr bench --suite agents   # microseconds per decision, per heuristic
uv run ttr arena       # round robin over h0-h4, recorded to runs/results.db
uv run ttr leaderboard # refit ratings over everything recorded
uv run ttr serve       # play and watch the heuristics at http://127.0.0.1:8000
```

A Rust toolchain is required: `uv sync` builds the engine's extension, `ttr_rust`, as a
workspace member, and rebuilds it whenever a Rust source changes.

`ttr arena --sprt h4,h3` stops as soon as a sequential test decides, on block boundaries
only — half a block is one seating.

### Playing the heuristics

`ttr serve` is the game server ([docs/WEB_PROTOCOL.md](docs/WEB_PROTOCOL.md)): play any of
H0-H4 on either map, watch two of them play with every hand and each decision's reasoning on
screen, and review a finished game move by move with each bot decision's candidates broken
down by term. The browser never holds hidden information -- no seed, no deck, nothing of an
opponent's hand beyond what is public -- and every game is recorded in `runs/play.db`, from
which it can be rebuilt and replayed. Rated games put you on the same leaderboard as the
bots, on the H3 = 0 scale: play H3, or run `ttr arena` first so the other bots are anchored
to it. The client itself is Phase 4.3 (`web/`); until it is built the server shows a
placeholder page and serves the API.

`ticket_to_ride.engine` re-exports the extension, so Python code never names it:

```python
from ticket_to_ride.engine import Game, Rng

game = Game("usa", 2)
state = game.new_initial_state(42)
rng = Rng.stream(42, "policy")
while not state.is_terminal():
    legal = state.legal_actions()
    state.step(legal[rng.below(len(legal))])
print(state.final_scores())
```

Batched play for training is `VecEnv`, which writes observations and masks for every
environment into caller-owned buffers in one call.

### On Modal

```bash
uv sync --group cloud                       # the Modal client; `modal token new` once
uv run ttr cloud bench --suite engine       # any ttr command, in a container, results pulled
uv run ttr cloud arena --agents h1,h2,h3 --blocks 100 --check-local
uv run ttr cloud run --cpu 16 -- arena --agents h3,h4 --blocks 2000
uv run ttr cloud pull                       # fetch any jobs not yet pulled
```

`infra/modal_app.py` builds the engine into a Linux image and runs each command in its
own directory on a `ttr-runs` Volume; arenas come back as their own databases and are
merged into `runs/results.db`. `--check-local` replays the arena here and requires every
game to match -- Linux x86 and macOS ARM agree game for game (12,800 games checked).
Credentials stay in `~/.modal.toml`.

## Layout

```
ticket_to_ride/
  data/       board data (generated from RULES.md), for `ttr map` and the UI
  engine/     a thin re-export of the Rust engine
  agents/     agent specs and the registry (the heuristics are Rust); search and learned agents
  rl/         networks, PPO, AlphaZero-style training
  eval/       paired-seed arena, Bradley-Terry ratings, SPRT, results store
  ui/web/     the game server (`ttr serve`): protocol, views, games, play store, ratings
tools/        gen_board.py: RULES.md -> the engine's board data; gen_protocol.py: the web schema
crates/
  ttr-core/   the engine, the observation encoder, H0-H4 -- pure Rust, standalone `cargo test`
  ttr-py/     thin PyO3 shim (`ttr_rust`); contains no game logic
  ttr-tools/  never shipped: the golden corpus (format, generator, checker) and bench-obs
tests/golden/ the golden corpus
```

[RULES.md](RULES.md) is the human-readable source of truth for the rules, the map, and
the destination tickets. It is transcribed into checked-in constants by
`tools/gen_board.py`, never parsed at runtime.

### Generated files

Generated files are checked in, each with a `--check` mode wired into pre-commit and the
test suite. **Do not edit them by hand; regenerate them.**

| Output | Source | Regenerate |
| --- | --- | --- |
| `data/maps/usa.toml`, `crates/ttr-core/src/board_gen.rs` | RULES.md | `make board` |
| `ui/web/protocol.schema.json` | `ui/web/protocol.py` | `make protocol` |

The golden corpus under `tests/golden/corpus/` is a **frozen fixture**, not a regenerated
output: `gen-corpus` checks it and refuses to rewrite it except at a `CONTRACT_VERSION` or
`OBS_VERSION` bump -- a diff means the engine plays a different game, never "regenerate it".

## Docs

- **[docs/CONTRACT.md](docs/CONTRACT.md)** — the frozen contract: PRNG, draw procedure and
  `state_hash()`, plus the golden-corpus format. Changing anything in it is a versioned
  event that re-signs the corpus.
- **[docs/PLAN.md](docs/PLAN.md)** — the approved implementation plan. Design of record:
  engine architecture, action/observation spaces, algorithm ladder, evaluation methodology,
  and the phase breakdown with exit criteria.
- **[docs/WORKLOG.md](docs/WORKLOG.md)** — dated log of what was done and why, including
  deviations from the plan.
- **[docs/WEB_PROTOCOL.md](docs/WEB_PROTOCOL.md)** — the game server's REST and WebSocket
  protocol; the Pydantic models in `ticket_to_ride/ui/web/protocol.py` are its source.
- **[docs/GOTCHAS.md](docs/GOTCHAS.md)** — traps already hit (with the fix in place), and
  known traps in work not yet written. Read before touching build config or the engine.
- **[docs/reproducibility.md](docs/reproducibility.md)** — the three reproducibility levels,
  what each actually promises, and the seeding discipline behind them.

`docs/lab_notebook.md` arrives with Phase 6.
