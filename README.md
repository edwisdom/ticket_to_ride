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
| 3 | Heuristic agents + arena + Elo | next |
| 4 | Terminal client | |
| 5 | Determinization + ISMCTS | |
| 6 | PPO self-play + league | |
| 7 | ISMCTS + learned priors | |
| 8 | Scaling, web UI, human gauntlet | |

The engine plays 2–5 players on the USA map and 2–4 on TTR-mini. Two implementations,
byte-identical over 700,000 games (100k seeds x every map and seat count), compared at
every step:

| | µs/step | full USA 2P games/s |
| --- | --- | --- |
| Python (the reference, and the permanent oracle) | 6.6 | ~980 |
| Rust, one thread | 0.14 | ~45,000 |
| Rust, 8 performance cores | 0.021 | ~303,000 |

Measured back to back in one process on an M2 Max; the Python baseline drifts 10–15%
between sessions, so `ttr bench` only ever compares within a run. ~500 Python tests plus
70 Rust tests, engine coverage 98%.

## Quickstart

```bash
make setup             # uv sync --all-extras --dev, then install pre-commit hooks
make test              # fast unit + property tests, no torch
make lint type         # ruff + ty
make rust              # build the Rust core into .venv (not part of `uv sync`)
make rust-test         # ttr-core's standalone tests -- no Python interpreter involved
uv run ttr map         # board data and the invariants it satisfies
uv run ttr bench       # both engines, side by side
```

The Rust core is a separate distribution, `ttr_rust`, and `ticket_to_ride.engine` never
imports it. That is enforced, not just intended: the Python engine is the permanent
differential-testing oracle, and an oracle that imports the implementation it validates
would agree with its own bugs. Tests needing the extension skip when it is unbuilt —
except under `TTR_REQUIRE_RUST=1`, which CI sets, because a differential harness that
silently skips reports green having compared nothing.

```python
from ticket_to_ride.engine import Game, RuleConfig, final_scores
from ticket_to_ride.engine.rng import stream

game = Game(RuleConfig(n_players=2))
state = game.new_initial_state(seed=42)
rng = stream(42, "policy")
while not state.is_terminal():
    state.step(state.sample_legal(rng))
print(final_scores(state))
```

## Layout

```
ticket_to_ride/
  data/       board data (generated from RULES.md) and city coordinates
  engine/     the game itself -- imports without torch or numpy, no global state
  agents/     random, heuristics, search
  rl/         observation encoding, networks, PPO, AlphaZero-style training
  eval/       paired-seed arena, Bradley-Terry ratings, SPRT, results store
  ui/         terminal and web clients
tools/        generators: board data, contract vectors, observation spec, golden replays
crates/
  ttr-core/   the Rust engine -- pure Rust, no PyO3, standalone `cargo test`
  ttr-py/     thin PyO3 shim; contains no game logic
tests/golden/ frozen artifacts: contract vectors and an 84-game replay corpus
```

[RULES.md](RULES.md) is the human-readable source of truth for the rules, the map, and
the destination tickets. It is transcribed into checked-in constants by
`tools/gen_board.py`, never parsed at runtime.

### Generated files

Four things are generated and checked in, each with a `--check` mode wired into pre-commit
and the test suite. **Do not edit them by hand; run `make board`.**

| Output | Source |
| --- | --- |
| `data/maps/usa.toml`, `data/board_gen.py`, `crates/ttr-core/src/board_gen.rs` | RULES.md |
| `crates/ttr-core/src/obs_spec_gen.rs` | `rl/encode/spec.py` |
| `tests/golden/contract_vectors.json` | `engine/rng.py`, `engine/hashing.py` |
| `tests/golden/replays.bin` | the engine itself |

The board data is emitted into **both languages from one canonicalized spec**, which is what
makes the Python and Rust engines structurally incapable of disagreeing about the board.

## Docs

- **[docs/CONTRACT.md](docs/CONTRACT.md)** — the frozen contract: PRNG, draw procedure and
  `state_hash()`, pinned bit for bit with test vectors. The Phase 1 → Phase 2 gate. Changing
  anything in it is a versioned event that invalidates every recorded replay.
- **[docs/PLAN.md](docs/PLAN.md)** — the approved implementation plan. Design of record:
  engine architecture, action/observation spaces, algorithm ladder, evaluation methodology,
  and the phase breakdown with exit criteria.
- **[docs/WORKLOG.md](docs/WORKLOG.md)** — dated log of what was done and why, including
  deviations from the plan.
- **[docs/GOTCHAS.md](docs/GOTCHAS.md)** — traps already hit (with the fix in place), and
  known traps in work not yet written. Read before touching build config or the engine.
- **[docs/reproducibility.md](docs/reproducibility.md)** — the three reproducibility levels,
  what each actually promises, and the seeding discipline behind them.

`docs/lab_notebook.md` arrives with Phase 6.
