# Ticket to Ride — engine + self-play RL

A fast, exactly-correct Ticket to Ride (base USA map) game engine, and a platform for
training reinforcement learning agents against it via self-play.

**Goal:** iterate on algorithms until agents beat strong heuristic opponents, then humans.

## Status

| Phase | | |
| --- | --- | --- |
| 0 | Project scaffolding | in progress |
| 1 | Python reference engine | |
| 2 | Rust core + differential harness | |
| 3 | Heuristic agents + arena + Elo | |
| 4 | Terminal client | |
| 5 | Determinization + ISMCTS | |
| 6 | PPO self-play + league | |
| 7 | ISMCTS + learned priors | |
| 8 | Scaling, web UI, human gauntlet | |

## Quickstart

```bash
make setup     # uv sync --all-extras --dev, then install pre-commit hooks
make test      # fast unit + property tests
make lint type # ruff + mypy
uv run ttr --help
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
tools/        board data generator
crates/       Rust engine core (Phase 2)
```

[RULES.md](RULES.md) is the human-readable source of truth for the rules, the map, and
the destination tickets. It is transcribed into checked-in constants by
`tools/gen_board.py`, never parsed at runtime.

## Docs

- **[docs/PLAN.md](docs/PLAN.md)** — the approved implementation plan. Design of record:
  engine architecture, action/observation spaces, algorithm ladder, evaluation methodology,
  and the phase breakdown with exit criteria.
- **[docs/WORKLOG.md](docs/WORKLOG.md)** — dated log of what was done and why, including
  deviations from the plan.
- **[docs/GOTCHAS.md](docs/GOTCHAS.md)** — traps already hit (with the fix in place), and
  known traps in work not yet written. Read before touching build config or the engine.

`docs/reproducibility.md` and `docs/lab_notebook.md` arrive with Phases 1 and 6.
