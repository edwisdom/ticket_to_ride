# Worklog

Reverse-chronological log of what was actually done, and why. Append an entry per work
session. Decisions that deviate from [PLAN.md](PLAN.md) belong here, with the reason.

---

## 2026-07-26 — Phase 1: the Python reference engine

Complete. `make lint type test` green, 450 tests (434 in the fast suite), engine
coverage 98%.

### The gate: docs/CONTRACT.md

Written first, because everything downstream depends on it and PLAN.md §5.8 makes it the
Phase 1 → Phase 2 gate. It pins the PRNG, the draw procedure and `state_hash()` bit for
bit, with machine-checked vectors in `tests/golden/contract_vectors.json`.

The PCG-XSH-RR 64/32 implementation reproduces **O'Neill's own `pcg32-demo` output** for
seed (42, 54). That matters more than matching our own vectors: a Rust port can be checked
against upstream, so both sides being wrong in the same way is caught. Two things a port
gets wrong quietly are called out and tested — the output permutes the *old* LCG state, and
the rotation is `(-rot) mod 32` rather than `32 - rot`, which differ only at `rot == 0`.

Fisher–Yates direction and the bounded-draw method are contract, not style: ascending
Fisher–Yates and Lemire's bounded generator are both equally correct and both produce
different values from the same stream.

**Three deviations from PLAN.md, all deliberate and all in the contract doc:**

- `state_hash()` is **blake2b-64, not FNV-1a-64**. FNV-1a is a per-byte Python loop costing
  ~40 µs over a ~450-byte state — about 25× everything else `state_hash()` does — and the
  differential harness calls it after *every* step of 100k games. blake2b is stdlib, ~1 µs,
  has a mature Rust crate, and is already the project's hash for `DATA_HASH` and seed
  derivation, so there is one primitive instead of two.
- The **locomotive is card type `n_colors`, not a hard-coded 8.** TTR-mini has six colours,
  so its locomotive is card type 6. Hard-coding 8 works on the USA map and silently
  corrupts mini.
- Initial ticket offers are dealt **one seat at a time** rather than all up front. Identical
  cards either way (the deck is long enough that a return never wraps into a later seat's
  deal, which a test enforces per map), and it keeps the pending offer a single field.

### Board data

One generator parses RULES.md and emits three checked-in artifacts: `usa.toml`,
`board_gen.py`, and `crates/ttr-core/src/board_gen.rs`. Emitting both languages from one
canonicalized spec is what makes the two engines structurally incapable of disagreeing about
the board — a data file parsed independently on each side would reintroduce exactly that
risk at the root of differential testing.

Index order is **derived, never taken from file order**: cities and colours sorted, segments
and tickets sorted by normalized endpoints. A re-transcription of the map in a different row
order therefore produces byte-identical output.

**TTR-mini ships as a data file, not a code path**: 14 cities / 24 pairs / 30 segments / 101
spaces / 14 tickets / 54 cards, 20 trains, max 4 seats. Against the prior art's mini map it
keeps 6 colours, max route length 5 and 6 double routes — dropping those removes colour
scarcity, the high-commitment end of the scoring curve, and blocking, i.e. most of what
makes TTR interesting. Ticket points are shortest-path costs in train cars, which a test
asserts rather than trusts.

### Engine

Every §5.2 rules edge case implemented next to a comment saying what it guards against, and
each with a named test. The ones worth repeating:

- The **3-locomotive flush has both locks**: it only fires when the available pool still
  holds three non-locomotives (an all-locomotive pool would cascade forever), and a hard
  cascade cap backs that up. `validate()` asserts the converse — 3+ locomotives face-up is
  only legitimate when the guard was the reason. That is the assertion most engines lack.
- **Double routes differ by seat count.** 2–3P marks the sibling `CLOSED`, a distinct
  sentinel from "owned" because blocked and enemy-held are very different to a blocking
  heuristic; 4–5P leaves it open to others and denies it only to the owner.
- The `hand[c] >= 1` guard on claim payments, without which a hand of pure locomotives makes
  all eight gray pay slots legal *and identical*.
- Reshuffle is **lazy at draw time**, so "deck empty, discards available" stays
  distinguishable from "both empty" — they have different legal actions.

Two things fell out of implementing it that the plan did not anticipate. The display is now
**refilled at end of turn** as well: without it, once every pool empties the display stays
empty forever, since only *taking* a face-up card would trigger a refill. And `step()`
records the action in history **after** validating, because a rejected action was leaving a
phantom entry that would make the replay of that game diverge.

### Throughput — the one exit criterion missed

**5.9 µs/step: ~1100 full USA 2P random games/s, ~2900 on TTR-mini.** PLAN.md §14 asks for
≥2000 playouts/s/core. Mini clears it; USA does not.

Profiling says there is nothing left to remove: claim legality is about a third of the time
and the rest is spread across a dozen small functions. Two rounds of optimization (a
discard fast-count, `list.count` in the flush check, an early-out refill, a sorted-prefix
pay-slot table) moved it from 965 to ~1100 and no further; the sorted-prefix version
measured identically and was reverted rather than kept as unearned complexity.

This is worth stating plainly rather than hiding: the plan's own §5.3 measured 4–8 µs for a
realistic claim mask, which at ~155 steps a game is 800–1600 games/s. The 2000 figure and
the 4–8 µs figure were never consistent with each other. **Closing that gap is what Phase 2
exists for**, and the number is now recorded by `ttr bench` so the ≥50× target has a real
baseline.

### Tests

450 tests. `tests/unit/engine/test_rules.py` walks the §5.2 checklist one named test per
item, with positions rigged directly by `tests/rig.py` (which keeps card conservation exact,
so `validate()` still applies to a rigged state). `tests/property` sweeps 25 seeds across
every (map, seat count) asserting `validate()` after every single step.

Three properties are worth naming:

- **An illegal action can never get through `step()`**, probed action-by-action, and a
  rejected action leaves the state byte-identical including its history.
- **Environment randomness does not depend on agent behaviour** — two games on one seed
  whose agents diverge still hold the same shuffle. That is the property paired evaluation
  rests on, and lazy count-based dealing would break it with no error anywhere.
- **Two guards on the guards**: a seed in the sweep really does reach a reshuffle, and one
  really does trigger a flush. An untested branch that never runs proves nothing.

The longest trail is checked against an **independent oracle** — every permutation of the
owned segments, longest valid prefix — over hand-built shapes and 300 subgraphs from real
games. It shares no code with the optimized version, which matters because that has four
layers and a bug in any of them would otherwise be invisible. The Steiner DP is checked
against the three-terminal closed form, exhaustively on mini.

`tests/golden/replays.bin` holds **84 finished games** across every map and seat count. A
rules, draw-order, PRNG or scoring regression shows up there as a concrete failing game with
a seed you can step through, rather than as a drifting win rate noticed three weeks later.

### Agents (throwaway) and the observation spec

The throwaway H0/H1 and flat-MC stub are **API exercise, not deliverables** (§8.1
mitigation 1): they put `clone()`, `clone_into()`, `legal_actions()` and `step()` under
search-like load while the interface is still free to change. Measured: H1 beats random ~95%
on USA and ~87% on mini, and flat MC with random rollouts beats random — which is the
de-risking signal from §13 that the rollout plumbing works.

Every H1 constant is a **fraction of the map's own train supply**, never an absolute, and a
test asserts H1 exercises every action type on both boards. That is the prior art's fatal
bug reproduced as a regression test.

The observation layout is a **declarative table** generated into both languages
(§8.2): 3355 dims on USA, 1169 on mini, with opponent slots pinned at four so one network
plays any seat count. The reference encoder is pure Python with no numpy — a test asserts it
imports with both torch and numpy banned — because it is the oracle Phase 2's Rust encoder
gets checked against.

Two more deviations recorded: the observation is 3355 dims rather than the ~2000–2800 the
plan estimated (the 100×9 required-colour block is the difference and §6.2 needs it as an
input for the colour-symmetry regularizer), and the claim-mask buckets number 45, not the 33
§5.3 guessed.

### Next

**Phase 2 — the Rust core.** The gate is passed: the PRNG, draw procedure and `state_hash`
are frozen with test vectors, the board data is generated into both languages, the
observation spec is a table in both languages, and the golden corpus gives the differential
harness something to fail against on day one. Compare at **every step**, not just the
terminal — terminal-only comparison tells you *that* you diverged, never *where*.

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

### Next at the time

Phase 1 — the Python reference engine. See the entry above for what actually happened.
