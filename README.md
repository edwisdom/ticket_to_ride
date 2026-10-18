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
| 4 | Web client: play the heuristics in a browser | built (4.1-4.3); your games against H3-H5 next |
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

On an M2 Max. 366 Rust tests (and 7 `#[ignore]`d sweeps) and 195 Python tests; ttr-core line
coverage 96%.

Six scripted agents, H0–H5, rated over 15,000 paired games in a few seconds. **H3 is the
Elo anchor, fixed at 1500**, which puts random play near the bottom of a chess-like scale.
Its identity is a hash of what it *plays* over a frozen probe set rather than of its
constants — a params hash would leave a reordered tiebreak or a fixed bug free to move
the anchor and silently re-base every rating ever recorded. The anchor moves only on
purpose: it did once, when H3 learned to plan the network its cards can pay for and to
spend its last turn on points (+36 Elo over its predecessor at USA 2P; docs/WORKLOG.md).
Above it, each tier adds what stepping through the losses of the tier below turned up:
H4's off-plan claims pay for the trains and cards they take from its plan. H5 prices a
ticket draw against its own network, so it draws tickets late, when they are cheap; it also
races for the longest route against the longest trail on the board, spends its trains on
points once its tickets are done, and late in the game plans the network its hand can pay
for.

| agent | usa 2P | mini 2P | usa 4P | µs/decision (usa 2P) |
| --- | --- | --- | --- | --- |
| h5 | 1693 [1677, 1713] | 1548 [1531, 1566] | 1624 [1618, 1631] | 26.7 |
| h4 | 1598 [1584, 1612] | 1532 [1517, 1548] | 1564 [1559, 1569] | 25.5 |
| **h3** | **1500** (anchor) | **1500** | **1500** | 18.0 |
| h2 | 1329 [1311, 1347] | 1364 [1346, 1383] | 1352 [1344, 1358] | 11.4 |
| h1 | 691 [632, 744] | 925 [889, 955] | 694 [674, 711] | 0.2 |
| h0 | 141 [63, 209] | 655 [610, 696] | 180 [156, 202] | 0.2 |

Bradley–Terry MLE, 95% intervals from a bootstrap that resamples **seed blocks** rather than
games — games inside a block share a deck, and treating them as independent inflates
significance by about √P. Both seatings of every block are played, never one mirrored.
The cost column is `ttr bench --suite agents`: one thread, self-play, on an M2 Max.

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
uv run ttr arena       # round robin over every tier, recorded to runs/results.db
uv run ttr leaderboard # refit ratings over everything recorded
uv run ttr serve       # play and watch the heuristics at http://127.0.0.1:8000
make web               # build the browser client it serves (Node >= 22.12, once per change)
```

A Rust toolchain is required: `uv sync` builds the engine's extension, `ttr_rust`, as a
workspace member, and rebuilds it whenever a Rust source changes.

`ttr arena --sprt h4,h3` stops as soon as a sequential test decides, on block boundaries
only — half a block is one seating.

### Playing the heuristics

`ttr serve` is the game server ([docs/WEB_PROTOCOL.md](docs/WEB_PROTOCOL.md)): play any of
H0-H5 on either map, watch two of them play with every hand and each decision's reasoning on
screen, and review a finished game move by move with each bot decision's candidates broken
down by term. The browser never holds hidden information -- no seed, no deck, nothing of an
opponent's hand beyond what is public -- and every game is recorded in `runs/play.db`, from
which it can be rebuilt and replayed. Rated games put you on the same leaderboard as the
bots, on the same H3-anchored scale: play H3, or run `ttr arena` first so the other bots are anchored
to it.

### The web client

`web/` is the browser client: Svelte and TypeScript, built by Vite into
`ticket_to_ride/ui/web/static/` (gitignored), which `ttr serve` serves. Node is needed only
to build it -- 22.12 or later; `web/.nvmrc` names 24 (`nvm install 24`).

```bash
make web               # npm ci (when the lockfile changed) + the production build
uv run ttr serve       # then open http://127.0.0.1:8000
make web-dev           # or: the dev server on :5173, hot reload, proxying /api and /ws to ttr serve
cd web && npm run check && npm test   # svelte-check and vitest
```

The board is drawn in the physical game's style with original art: Natural Earth geography,
cities at their real latitude and longitude, routes as chains of car spaces in the card
colours with a symbol on each for colour-blind players, claimed routes as raised trains in
the owner's colour, and the score track around the frame. Its geometry --
every car space of every track -- comes from `tools/gen_layout.py`, so the client only draws.
The lobby starts games against any bot or between two; the game screen shows what the bots'
observation holds (the opponent's known cards, the discards, what you cannot account for,
your tickets' remaining cost); spectating adds each decision's reasoning; and the review
scrubs a finished game with everything revealed and every bot decision broken down by term
(`#/review/{id}?at=N` links a move). Development builds also serve a board gallery at
`#/dev`.

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
  data/       board data (generated from RULES.md), for `ttr map` and the UI; layout/, where
              the client draws each city and route (`make layout`)
  engine/     a thin re-export of the Rust engine
  agents/     agent specs and the registry (the heuristics are Rust); search and learned agents
  rl/         networks, PPO, AlphaZero-style training
  eval/       paired-seed arena, Bradley-Terry ratings, SPRT, results store
  ui/web/     the game server (`ttr serve`): protocol, views, games, play store, ratings
web/          the browser client: Svelte + TypeScript, built by Vite into ui/web/static/
tools/        gen_board.py: RULES.md -> the engine's board data; gen_protocol.py: the web
              schema; gen_layout.py: the board's layout and basemap for the client
crates/
  ttr-core/   the engine, the observation encoder, H0-H5 -- pure Rust, standalone `cargo test`
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
| `data/layout/{usa,mini}.json` | `data/layout/*.toml`, the maps, Natural Earth | `make layout` |
| `web/src/api/protocol.ts` | `ui/web/protocol.schema.json` | `cd web && npm run types` |

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
