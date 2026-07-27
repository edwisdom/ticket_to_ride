# Worklog

Reverse-chronological log of what was actually done, and why. Append an entry per work
session. Decisions that deviate from [PLAN.md](PLAN.md) belong here, with the reason.

---

## 2026-07-27 — Phase 3: the agent ladder and the evaluation harness

Complete. `make lint type test` green, `cargo test` green (97 Rust tests), 10k paired games
across five agents in 12.6 s on the USA map.

### Three decisions taken before writing much code

**1. The Elo anchor is pinned by *behaviour*, not by its constants.** §11 makes H3 the
permanent zero of the rating scale, which only works if H3 is the same player months apart.
Putting its constants in a struct is necessary and not sufficient: it pins the numbers and
leaves the code around them free. A reordered tiebreak, a changed
`MAX_STEINER_TERMINALS`, a fixed bug in path reconstruction — each moves H3's play with the
params hash unmoved, and every rating ever recorded silently re-bases.

So an agent's identity is a **behaviour hash**: its action sequence over ten frozen
`(map, seats, seed)` probes, played to the end, in self-play *and* against random — the
second because it lands the agent in ragged positions it would never reach on its own,
where the branches that never otherwise run live. `params_hash` survives as provenance.
The property this buys was exercised for real: every H4 change in this phase left H3's hash
at `5b906156d4b1204b`, because H3 cannot read those code paths. A params hash would have
demanded an anchor re-base for a change H3 never sees.

The corollary is the tuning workflow. A retuned H3 is a **new agent** (`h3@config.toml`),
rated *against* the anchor; it never becomes the anchor. That is the entire mechanism by
which ratings accumulate.

**One correction to §11 while implementing it: `git_sha` must not be part of agent
identity.** §11 suggests content-addressing on `(checkpoint_path, config_hash, git_sha)`.
A hundred commits that never touch the heuristics would leave a hundred "different" H3s,
each holding a slice of the games and none with a usable rating — shattering exactly the
accumulation the scheme exists for. It is recorded on the *match*.

**2. Building all four statistics, not three.** Paired blocks, BT-MLE and the block
bootstrap are load-bearing for the exit criteria. SPRT was the one worth arguing about,
since its headline consumer is Phase 5's promotion gate. It is built, because its stopping
rule is validatable by simulation *without* a promotion loop — which is a stronger test than
any Phase 5 usage would give it — and because `ttr arena --sprt` makes heuristic tuning the
interactive loop §8.1 promised. Over 3000 simulated experiments at nominal α = β = 0.05:
**0.045 under H0, 0.072 under H1**, median 405 blocks to decide at Δ = 0 and 115 at +60 Elo.
β above nominal is expected — Wald's bounds ignore overshoot and the variance is estimated —
and is recorded rather than rounded away. It stops on **block boundaries only**: half a
block is one seating, which is the mirrored fiction the whole design exists to avoid.

Deferred, with consumers named: the parquet per-turn store and msgpack replay sampling
(nothing reads them until Phase 6; the SQLite summary rows are the ACID part that matters),
the BR-probe (needs PPO), and the training-side diagnostics — entropy, `mask_violations`,
value calibration. The action/ticket/route diagnostics *are* collected, because they are the
evidence for the coverage criterion rather than a separate feature.

**3. Recompute ratings; never accumulate them.** Bradley-Terry is an exponential family
whose sufficient statistic is the pairwise win-count matrix, so however many games pile up
they reduce to an `A×A` table and the fit is `O(A²)` per Newton step. Measured on synthetic
rows, so this is the store and the fit rather than the engine:

| games | read | reduce + fit | bootstrap | total | db |
| --- | --- | --- | --- | --- | --- |
| 10,000 | 0.02 s | 0.04 s | 0.35 s | 0.39 s | 2 MB |
| 100,000 | 0.23 s | 0.38 s | 2.67 s | 3.05 s | 21 MB |
| 1,000,000 | 2.44 s | 4.03 s | 27.34 s | 31.37 s | 213 MB |

Linear, and **the fit is not what grows — the bootstrap is**, because it scales with blocks
rather than with agents. So no maintained `pair_stat` table: it would be a cache in front of
a grouped sum. `--resamples` is the knob if a million games ever needs one. Incremental Elo
would have been strictly worse — order-dependent, drifting against the batch fit, and buying
nothing, because the expensive object was already small.

### The ladder, measured

10k paired games per configuration, every rotation played, 1000-resample block bootstraps.
§11's predicted ladder is in the last column.

| agent | usa 2P | mini 2P | usa 4P | predicted |
| --- | --- | --- | --- | --- |
| **h4** | −4 [−20, +14] | **+21 [+8, +33]** | −21 [−28, −14] | +130 |
| **h3** | 0 (anchor) | 0 (anchor) | 0 (anchor) | 0 |
| **h2** | −100 [−119, −82] | −117 [−137, −99] | −120 [−129, −111] | −120 |
| **h1** | −726 [−778, −684] | −545 [−582, −510] | −724 [−744, −706] | −400 |
| **h0** | −1276 [−1348, −1217] | −816 [−858, −782] | −1252 [−1279, −1223] | −900 |

**The ordering holds everywhere; the spacings do not.** H2 lands on its prediction on all
three boards, which is a good sign the scale is calibrated. H1 and H0 are far weaker than
predicted and H4 far weaker — H4 is the one that matters, because §14's Phase 6 target is
"beats H4 ≥65%", and an H4 that is level with H3 makes that target easier than intended.
Reported rather than reconciled.

`cycle_fraction` is 0.00 on all three: the scripted ladder is transitive, as it should be.

### Throughput

| configuration | games | wall | engine |
| --- | --- | --- | --- |
| usa 2P | 10,000 | 12.6 s | 11.9 s |
| mini 2P | 10,000 | 0.5 s | 0.3 s |
| usa 4P | 10,000 | 23.8 s | 23.6 s |

§14 asks for "10k paired games across 5 agents in **seconds**". Met. Python never loops per
decision: one FFI call plays a whole lineup across the rayon pool and returns two columnar
tables that go straight into SQLite.

### What the exit criteria caught, and it was not the tests

Three bugs, and every one of them was the published prior art's failure wearing new clothes.
None produced an error; each looked like a strategy.

**The ticket-keep filter §7 specifies is blind to the train budget.** `marginal_steiner_cost
≤ points` accepts everything on TTR-mini, where ticket points *are* shortest-path costs by
construction so marginal always equals points. Measured: 3 opening tickets costing **22 train
cars against a 20-train supply**, in 59 of 60 games — after which the agent correctly refused
to draw another all game, because the plan it had just accepted was already unaffordable.
`draw_tickets` fired **zero times across 30 mini games** while the USA board drew freely.
That is the paper's symptom reached from the opposite direction: not a gate that can never
fire, but one that always does. Replaced by expected settlement against the train budget —
`points × (2·odds − 1)` — one model shared by keeping *and* drawing, since they are the same
question asked twice. It costs one parameter fewer than the rule it replaces.

**The ticket-draw model priced the mean of the deck, not the best of the three offered.**
With `draw_ticket_keep_min = 1` you keep the cheapest of the offer, so the expected marginal
cost is the expected *minimum* of `deal` draws, not the mean. Using `2/(deal+1)` — derived
from the board's own deal size, not fitted — is what finally made mini 2P draw tickets at
all. It is rare there (3 draws per 40 games) and that is correct: forcing eagerness measured
catastrophically, at −0.66 and −1.00 mean return.

**The threatened-edge primitive priced the sum of the two components instead of the
bottleneck.** Merging a 2-car stub into a 20-car trunk gains an opponent two cars of trail,
not twenty-two. Summing made denial worth ~12 points against a 15-point route, and **H4 lost
to H3 at 92.5%** while claiming 30% more routes and drawing a third as many tickets. §1's
four pricing rules were absent as well; they are in now.

### Blocking is a two-player idea, and the seat count is what proves it

The most interesting measurement of the phase. At 10k games H4 was **+21 Elo on mini 2P and
−50 [−58, −42] on USA 4P** — same code, same constants, opposite signs. Ablating at 4P and
5P localised it: threat alone −12.9 at 4P, hoarding alone −12.3 at 5P, and with both off
exactly **+0.0** (it plays as H3, which is its own consistency check).

Both mechanisms assume two players. Cars spent denying one opponent help every *other* rival
as much as they help me, so the share I capture is about `1/(P−1)`; and "waiting costs
nothing" fails for the same reason, since more rivals means the parallel route is likelier
gone when I come back. Dividing both by the rival count took USA 4P from −50 to −6 and 5P
from −19.5 to −15.6, with 2P unchanged because the divisor is 1 there.

It is still mildly negative above two seats, and that is where it is being left. The residual
is a real design gap rather than a tuning one: **the free-rider structure says to block the
leader, and H4 blocks whoever is most blockable.** A 3–5P opponent model needs to know who
is winning, which is a different primitive. §13 targets 2P first and §14's H4 gate is a 2P
gate, so this is aligned with the plan rather than a deviation from it — but it is a real
limitation and is not going to fix itself.

Two earlier readings were noise and are worth naming so nobody trusts the method that
produced them: at 80 seed blocks the standard error on a mean return is ~0.08, which is
±30 Elo, and every H4 ablation "difference" at that size was inside it. The seat-count
result only became visible at 10k games.

### H3's per-decision cost is a Phase 5 constraint, measured now rather than discovered later

`ttr bench --suite agents`, single-threaded because the number is read as a latency:

| map | seats | h0 | h1 | h2 | h3 | h4 |
| --- | --- | --- | --- | --- | --- | --- |
| usa | 2 | 1.07 | 1.00 | 95.48 | **76.34** | 79.60 |
| usa | 4 | 0.46 | 0.48 | 97.96 | **85.12** | 92.67 |
| mini | 2 | 0.26 | 0.30 | 3.21 | **3.46** | 3.67 |

microseconds per decision. **H3 doubles as the ISMCTS rollout policy** (§7, §8.3), and a
rollout from mid-game is ~100 decisions — so 1000 sims is **~7.6 s per move** on the USA
board. §7.2's S2 tier (SO-ISMCTS, 800–2000 iters, "strong with zero NNs") is not reachable
with this rollout policy. That is a Phase 5 design constraint discovered in Phase 3, which
is the cheap place to discover it. When it is addressed, the cheaper policy gets **its own
name and its own rating** — never a silent substitution for the agent the anchor is defined
by. Two optimizations already took H3 from 76 to 26 µs at the smaller plan sizes it had
before retuning: pricing a detour by re-routing the segment's own endpoints instead of
re-solving the plan, and keeping a plan across any claim that misses it (claims only remove
options, so a surviving plan is still optimal).

### The seat-flatness criterion is half right, and the half matters

§14 asks for "seat win rate flat within CI" and §11 calls it the canary that mirroring
works. With cyclic rotation each *agent* occupies each seat equally often, so agent ratings
carry no seat bias **by construction** — that is the harness property, and the arena asserts
it directly. The raw per-seat win rate is a different thing, and at 10k games it is **not**
flat:

| configuration | seat 0 | expected if flat |
| --- | --- | --- |
| usa 2P | 0.5070 [0.5003, 0.5137] | 0.5 |
| mini 2P | 0.5197 [0.5110, 0.5290] | 0.5 |
| usa 4P | 0.2640 [0.2532, 0.2737] | 0.25 |

The intervals exclude the flat value. That is **first-player advantage — a real property of
Ticket to Ride, not a harness bug** — and at 10k games it is finally resolvable: about +0.7
points of win rate on USA 2P and +2.0 on mini, where the shorter game gives the tempo less
time to wash out. So the harness reports it with an interval instead of asserting it away.
Conflating the two would have made a genuine finding look like a broken arena.

### Deviations from PLAN.md, all deliberate

- **The H3 ticket-keep filter is expected settlement, not `marginal_steiner_cost ≤ points`**
  (§7), for the measurement above. Strictly the same idea with the missing budget constraint
  put back, and one parameter fewer.
- **`ticket_keep_ratio` was removed** rather than kept as a knob the better model makes
  redundant. `plan_utilization_target` now governs keeping *and* drawing, because a board
  where those two disagree is one where the agent takes on commitments it then refuses to
  extend.
- **H4's threat and hoard terms are divided by the rival count** — §1's pricing rules do not
  mention the seat count, and the measurement says they must.
- **§14's "H3 > H2 > H1 > random by predicted margins" is met as an ordering and missed as
  spacings**, as tabulated above.
- **`keep_extra` and `claim_wild` are reported but not required** by the coverage test.
  Requiring them would assert a strategy rather than detect a dead branch: H2 minimizes added
  track, so keeping the minimum *is* its rule, and declining to pay a route entirely in
  locomotives is defensible play. `draw_tickets` **is** required, on every map — that is the
  one the prior art never reached.

### Also found

`cargo test --workspace` has been aborting on macOS since Phase 2 and nobody noticed. The
ttr-py cdylib's test harness links — thanks to the `-undefined dynamic_lookup` already in
`.cargo/config.toml` — and then dies at load with `symbol not found in flat namespace
'_PyBaseObject_Type'`, because `extension-module` deliberately does not link libpython. The
four ttr-core suites pass and print first, so the command reads as green until you scroll
past a page of passes to the `SIGABRT`. Nothing was lost — that crate converts types and
holds no logic by design — but `test = false` makes the failure impossible rather than
merely harmless.

And a smaller one: **SQLite's `INTEGER` is signed 64-bit**, so a `u64` `state_hash` above
2⁶³ raises `OverflowError` on insert. Stored as hex text; the two's-complement view would
work and would print as a negative number matching nothing the engine ever reports.

### Next

**Phase 4 — the terminal client.** The ladder it will be played against is measured and
anchored, so "you lose to H3 at least once" is a criterion with a known opponent behind it.
Phase 5 should read the per-decision table above before choosing a sim count.

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
