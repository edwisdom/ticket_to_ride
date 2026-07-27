# Worklog

Reverse-chronological log of what was actually done, and why. Append an entry per work
session. Decisions that deviate from [PLAN.md](PLAN.md) belong here, with the reason.

---

## 2026-07-27 — Phase 2: the Rust core

Complete. `make lint type test` green, `cargo test` green (70 Rust tests), and the Rust
engine is **byte-identical to the Python oracle** across every map and seat count.

### Three decisions taken before writing much code

**1. Layout.** `crates/ttr-core` (pure Rust, no PyO3, standalone `cargo test`) +
`crates/ttr-py` (thin PyO3 shim) under a workspace at the repo root, per §8. The one thing
§8 does not decide is how the extension reaches `.venv`, and it matters: the shim is **its
own distribution**, `ttr_rust`, not a submodule of `ticket_to_ride`. The Python engine is
the permanent differential-testing oracle, and an oracle that imports the implementation it
validates is not an oracle — a shared bug would agree with itself and the harness would
report green. `test_import_boundary.py` now bans `ttr_rust` from `engine/` alongside torch
and numpy. Consequence: `uv sync --dev` stays Rust-free, `make rust` builds, and
Rust-dependent tests skip when it is absent — *except* under `TTR_REQUIRE_RUST=1`, which CI
sets, because a differential harness that silently skips reports green having compared
nothing.

**2. `chance_mode="explicit"` is deferred to Phase 5, and §14's "both chance modes" is
amended.** Not because it is unwritten — because the frozen contract does not specify it and
partly contradicts it. CONTRACT.md §2.1 says the `deck_counts()` view "is never the source
of a draw", which is exactly what an explicit chance node makes it; and §3.1's serialization
has no field for a pending chance event. Implementing it means adding serialized state,
i.e. a `CONTRACT_VERSION` bump that invalidates all 84 golden replays and every vector —
during the phase whose entire job is to be checked against them. There is also no oracle:
Python raises `NotImplementedError`, so building it in Rust first would mean writing the
reference implementation in the language being checked. **When it lands it lands Python
first, then Rust, then the harness sweeps both.** `ChanceMode::Explicit` exists as a
declared variant that construction refuses with that reason in the error.

**3. Search hooks.** `clone_into` and `position_hash` ported directly.
`resample_from_infoset` **built for real**, because it is the one hook that constrains the
*state layout* — it has to close arithmetically against `certain`/`unknown`/`discard`/
`faceup`, and discovering it does not in Phase 5 is the expensive version (§8.1 mitigation
2). It closed. `collect_leaves`/`backup` **neither built nor stubbed**: they are search-tree
operations, not state operations, and writing them now would mean designing an MCTS before
one exists — the exact thing mitigation 3 budgets a revision for. A stub with a signature
nobody validated is worse than an absent one. What *was* built is the substrate they would
be expensive to retrofit onto: batched `VecEnv::step`, `observe` into caller buffers, and
arena reuse.

### Throughput — measured, not quoted

On an M2 Max, both engines back to back in one process, median of repeated runs:

| config | python | rust ×1 | rust ×8 |
| --- | --- | --- | --- |
| usa 2P | 6.62 µs/step, 979 g/s | 0.143 µs, 45.1k g/s (**46×**) | 0.021 µs, 303k g/s (**311×**) |
| usa 4P | 6.67 µs/step, 504 g/s | 0.156 µs, 21.7k g/s (**43×**) | 0.023 µs, 149k g/s (**295×**) |
| mini 2P | 5.24 µs/step, 2.7k g/s | 0.101 µs, 136k g/s (**52×**) | 0.016 µs, 863k g/s (**330×**) |

§14 asks for ≥50× with a batched target of 85–170k games/s. **Batched clears it by ~2×.
Single-threaded is 46× on USA 2P and 43× on 4P — under the 50× line**; only mini clears it
on one core. Reported rather than rounded up.

An earlier reading of 50.4× was an artifact: that run's Python baseline came in at 7.47
µs/step against a stable median of 6.6. The Python engine has been observed at 5.9, 6.5 and
7.5 µs/step for identical code across sessions, so **a ratio taken against a number recorded
on another day measures the weather**. `ttr bench` now takes a median of `--repeat` runs and
interleaves engines per configuration so both see the same thermal state.

One optimization tried and **reverted**. Profiling put the per-length pay-slot table at ~27%
of a claim scan (48 iterations regardless of the hand), so it was rewritten as bitmasks
costing `sum(min(reach[c], max_len))`. End to end: 0.145 vs 0.143 µs/step — no change.
Reverted on the same reasoning Phase 1 used for its sorted-prefix pay table: identical
measurement, unearned complexity. Recorded so nobody tries it a third time. Legality is
**81% of a step**, so any real win has to come from there.

### What the differential harness found, and what it taught

Byte-identical over **100k seeds x 7 (map, seat-count) configurations -- 700,000 games,
compared at every step, zero divergences.** That is wider than §14 asks for: the criterion
is 100k seeds x {2,3,4,5}P on one map, and this covers USA 2-5P *and* mini 2-4P. All 84
golden replays reproduce their recorded final hash *and* scores.

Compared at **every** step: `state_hash`,
`position_hash`, sorted `legal_actions`, `is_terminal`, `current_player`, and at the
terminal the longest trails, itemized breakdown, final scores, winners and `returns`.

Two findings worth more than the port itself:

- **CPython's `sum()` is Neumaier-compensated; Rust's `Iterator::sum` is not.** They
  disagree in the last ULP on `[0.0, 1.0, 2/3, 1/3]`, which propagated into every seat's
  `returns()`. Caught **only** because returns are compared exactly — a tolerance would have
  hidden it, and would equally hide a real difference in the formula. Fixed by transcribing
  Neumaier into `numeric.rs`; Python is the oracle and compensated summation is also strictly
  more accurate, so this is not a fidelity-for-parity trade. Two plausible culprits were
  ruled out first: FMA contraction on aarch64, and Rust parsing `if c {1.0} else {0.0} / n`
  as `if c {1.0} else {0.0/n}` (it does not, but the expression is now parenthesized).
- **A fast tier that only drives USA 2P is worthless.** Deleting the end-of-turn refill from
  Rust survives 60 seeds of USA 2P/3P/4P undetected, because reaching it needs deck *and*
  discard to run dry. It is caught at mini seed 0 and USA 5P seed 8. `FAST_CONFIGS` is
  shaped by that measurement, which is recorded next to it. **Pick the fast tier's shape by
  mutating the engine and seeing what catches, not by taking the biggest map.**

The harness reports *where*, not just *that*: a hash mismatch prints the first differing
serialized **field** and a one-line reproduction. Field-wise using each side's own
`deck_len`, because a reshuffle at a different moment changes the image length and a
byte-offset diff then blames the layout for what is a value difference several fields
earlier.

### Observation encoder

3355 dims on USA, 1169 on mini, driven entirely by the generated feature table. Verified
over 4463 observations — every seat, every map, every seat count, sampled every 7 steps
through full games plus the terminal — with **zero differing slots**, compared exactly on
f32. Sampling through the game is the point: on an opening position `remaining_cost`,
`fragility`, `on_my_steiner_tree`, `extends_my_chain` and `is_dead` are all identically
zero.

The **thermometer bucket edges are now generated into Rust** alongside the layout. Before
this the table guaranteed both encoders agreed on a thermometer's *width* while its step
positions were re-typed by hand on each side — a divergence that shows up as a wrong value
under a perfectly correct layout, which is the hardest kind to find.

One documented claim was withdrawn rather than left standing. The encoder header first
asserted that computing in f32 instead of narrowing from f64 "would disagree in the last
bits on values as ordinary as 1/45". It does not: over every divisor this encoder uses,
across the full numerator range, the two are bit-identical, and a mutation to direct-f32
division was not caught by 20 seeds. The real reason to compute in f64 is that it matches
Python's arithmetic *structurally*, so agreement does not depend on the ranges being benign.
That is now a Rust test instead of a sentence.

### Deviations from PLAN.md, all deliberate

- **`State` is `Clone`, not `Copy`** (§5.1 says `Copy`). It carries the action history
  exactly as Python's does. Everything `Copy` was for still holds: the POD arrays clone as a
  memcpy, `clone_into` reuses the destination's allocation, and search runs with
  `track_history` off.
- **rayon is pool-*sized* to the performance cores, not pinned to them** (§8.3 says pin).
  macOS exposes no thread-affinity API — no `sched_setaffinity`, only a QoS hint the
  scheduler may ignore. Sizing from `hw.perflevel0.logicalcpu` is what is achievable, and
  the difference is recorded in `vecenv.rs` rather than left as an unmet claim.
- **§14's "both chance modes" is amended to sampled only**, for the reasons above.
- **§14's ≥50× is met batched and missed single-threaded on the USA map**, as measured
  above.

### Also shipped

The extension carries a hand-written `.pyi` plus `py.typed`, because a compiled module is
untyped and every consumer from here to Phase 8 imports it. A test compares the stub against
the real module in **both** directions — stubbed-but-absent being the dangerous one, since
it type checks and fails at runtime — and was verified to catch a deliberately bogus entry.

Three PyO3 traps are in GOTCHAS: `extension-module` breaking a bare `cargo build` on macOS,
`panic = "abort"` being incompatible with unwinding into CPython (and not overridable
per-package), and a `Clone` pyclass's automatic `FromPyObject` silently copying an RNG
handle at the call boundary — which would leave the caller's stream unadvanced and make
every determinization identical.

### Two operational notes

`make bench-check` needs an idle machine. Run alongside the nightly sweep under `-n auto`,
pytest-benchmark reports a 40–80% "regression" that is pure core contention, with a
`PerformanceRegression` traceback that names your code. Measured, not hypothesised — the
same gate passes cleanly on a quiet box. That is also why CI asserts a *ratio* and leaves
the absolute figures to local runs.

The nightly sweep takes ~20 minutes wall-clock under `-n auto` on 12 cores. Redirecting it
through `nohup` can lose pytest's final summary line; the per-test progress characters are
the reliable signal, since a failure prints `F` inline as it happens.

### Next

**Phase 3 — H0–H4 in Rust with config-driven constants, plus `eval/`.** The engine they run
on is now ~46× faster single-threaded and ~300× batched, which is what makes a 10k-game
arena a few seconds rather than a few minutes. Watch the §14 Phase 3 traps: every heuristic
must exercise every action type on **both** maps (the published prior art's fatal bug), and
win-rate matrices must measure both seatings rather than mirroring one.

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
