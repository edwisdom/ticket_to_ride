# Ticket to Ride — engine + self-play RL

A fast, exactly-correct Ticket to Ride (base USA map) game engine, and a platform for
training reinforcement learning agents against it via self-play.

**Goal:** iterate on algorithms until agents beat strong heuristic opponents, then humans.

## Status

| Phase | | |
| --- | --- | --- |
| 0 | Project scaffolding | done |
| 1 | Python reference engine | done |
| 2 | Rust core + differential harness | next |
| 3 | Heuristic agents + arena + Elo | |
| 4 | Terminal client | |
| 5 | Determinization + ISMCTS | |
| 6 | PPO self-play + league | |
| 7 | ISMCTS + learned priors | |
| 8 | Scaling, web UI, human gauntlet | |

The engine plays 2–5 players on the USA map and 2–4 on TTR-mini, at ~5.9 µs/step
(~1100 full USA 2P random games/s). 450 tests, engine coverage 98%.

## Quickstart

```bash
make setup             # uv sync --all-extras --dev, then install pre-commit hooks
make test              # fast unit + property tests, no torch
make lint type         # ruff + ty
uv run ttr map         # board data and the invariants it satisfies
uv run ttr bench       # engine throughput
```

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
crates/       Rust engine core (Phase 2)
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

`docs/reproducibility.md` and `docs/lab_notebook.md` arrive with Phases 5 and 6.
