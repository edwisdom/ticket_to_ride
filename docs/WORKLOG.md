# Worklog

Reverse-chronological log of what was actually done, and why. Append an entry per work
session. Decisions that deviate from [PLAN.md](PLAN.md) belong here, with the reason.

---

## 2026-07-26 — Planning, Step 0, and Phase 0

### Planning

Researched and wrote the full implementation plan ([PLAN.md](PLAN.md)). Four decisions were
put to the user and confirmed:

- Pure-Python reference engine first, then a Rust core via PyO3/maturin, with the Python
  engine kept permanently as a differential-testing oracle.
- Engine supports 2–5 players; training and evaluation target 2P first.
- Algorithm ladder runs through search + learning (heuristics → PPO self-play → ISMCTS with
  learned policy/value priors), not model-free only.
- Rich terminal client now, browser SVG map later.

**Phase order was changed after the plan was first written.** The Rust core moved from
Phase 4 to **Phase 2, ahead of all agent work**, at the user's request — a fast engine makes
agent iteration cheap, and abandoning an unproductive approach costs minutes instead of
hours. A second benefit surfaced while reworking it: H3 doubles as the ISMCTS rollout
policy and must live inside Rust anyway (a rollout crossing the FFI boundary per step
defeats the batching design), so building agents *after* the port means writing the
heuristics once rather than twice. The cost — freezing the engine API before agents and
search have exercised it — is mitigated by throwaway H0/H1 agents in Phase 1, designing the
search hooks in from day one, and explicitly budgeting one API revision at the start of
Phase 5.

### Prior art

Reviewed the one published attempt: Yang et al., *"Reinforcement Learning Agents Playing
Ticket to Ride"*, IEEE Access 11:60737–60757, 2023 (doi 10.1109/ACCESS.2023.3287100). Six
ideas were folded into the plan (§1) — most usefully the **ticket reward redistribution**
(−points on keeping a ticket, +2× on completion) and the **"threatened edge" primitive**
(an unclaimed edge that would merge two disconnected components of an opponent's network,
which is a much sharper notion of blocking than "a route they might want").

Their results should not be treated as evidence of agent strength: their map is 8 cities /
4 colors / max route length 3 / no double routes, and their heuristic baselines are lifted
unmodified from a public repo whose ticket-drawing thresholds are hard-coded at 15 trains —
on their 10-train map those conditions can never fire, so every baseline plays its opening
tickets and never draws more. Their round-robin matrix is also mirrored rather than
measured, so first-player advantage is confounded into every cell. Two process lessons were
adopted as a result: heuristic constants must be config-driven and re-tuned per map (with a
test asserting each heuristic exercises every action type), and win-rate matrices must
always measure both seatings.

Their strongest finding *does* validate a choice already made: their self-play agent had the
lowest average score yet beat every heuristic-trained agent head-to-head, because their
reward was pure score. **Optimize win probability, not score.**

### Step 0 — RULES.md normalization

At the user's request, made RULES.md internally consistent:

- `pink` → `purple` (7 occurrences, all in the map table). The rules prose already called
  this color "Purple"; only the map table disagreed. `PURPLE` is now the canonical name.
- `Montréal` → `Montreal` (5 occurrences: 2 in prose examples, 3 in the Tickets table). The
  map table already spelled it without the accent.

Verified after the change: 36 cities / 78 pairs / 100 segments / 309 spaces / 30 tickets,
44 gray + exactly 7 segments of each of the 8 colors, 22 double routes, graph connected,
and — the point of the exercise — **every ticket endpoint now matches the map spelling
exactly, with zero normalization**. Before this, three of the thirty tickets would have been
silently dropped or raised `KeyError` in a naive parser.

### Phase 0 — scaffolding

Complete. `make lint type test` green, `ttr version` works, nothing installed that isn't
needed.

- `pyproject.toml`: `requires-python = ">=3.14,<3.15"`; base deps numpy/msgspec/typer/rich;
  `rl` and `web` optional extras; dev dependency group; `[tool.uv] environments`;
  pytorch-cpu index for Linux; ruff, ty, pytest, and coverage config.
- Package tree under `ticket_to_ride/` per PLAN.md §2, plus `tools/`, `tests/`, `configs/`,
  `docs/`, `benchmarks/`. `main.py` deleted.
- `cli/app.py` with the `ttr` entry point.
- `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.gitignore` additions.
- Tests: import-boundary guard and RULES.md structural invariants (10 tests).

**Verified rather than assumed** — the Python 3.14 risk flagged during planning is real but
resolved: `uv lock` produces `torch-2.13.0-cp314-cp314-macosx_14_0_arm64.whl`. The
`torch>=2.13` pin is load-bearing, not cosmetic; see [GOTCHAS.md](GOTCHAS.md).

### Phase 0 addendum — mypy replaced with ty

At the user's request, swapped mypy for Astral's `ty` (0.0.63). Roughly **35× faster**
(0.04 s vs ~1.5 s), which matters for the edit loop.

The swap is not one-for-one. ty is inference-based and has **no `disallow_untyped_defs`
equivalent**, so an unannotated function would type check clean — exactly what
`mypy --strict` existed to prevent. Ruff's `ANN` ruleset now covers that half, with `ANN401`
ignored (`Any` is occasionally correct for `**kwargs`). The division is: **ty answers "are
these types consistent?", `ANN` answers "are these functions annotated at all?"** Removing
either leaves a hole.

Both were verified to actually fire, not assumed: a probe assigning `int` to `str` produced
`error[invalid-assignment]`, and an unannotated `def f(x)` produced `ANN001` + `ANN202`. The
`[tool.ty]` config keys were confirmed genuinely recognized (a bogus key is rejected), since
a config that looks applied but is silently ignored is the failure mode here.

This forced one CI restructure: ty must see torch to resolve imports under `rl/`, so
`typecheck` (syncs `--all-extras`) is now a separate job from `lint` (ruff only, no torch).
`test-fast` remains completely torch-free.

### Next

**Phase 1 — Python reference engine.** The highest-leverage phase in the project: the last
point at which the rules contract, state layout, and observation feature-spec are free to
change, because Phase 2 ports them to Rust. Deliverables in PLAN.md §14; the gate into
Phase 2 is the frozen PRNG / draw-procedure / `state_hash` contract with test vectors
(PLAN.md §5.8).
