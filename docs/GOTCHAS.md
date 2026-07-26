# Gotchas

Traps in this project. Part 1 is things already hit, with the fix in place — **do not
"clean up" these without reading why they exist.** Part 2 is traps extracted from
[PLAN.md](PLAN.md) so they aren't rediscovered the hard way; the ones marked ✅ have been
hit and handled, and stay because each is a thing to re-check when the Rust port
reimplements it.

---

## Part 1 — already hit, fix in place

### RULES.md escapes its double-route pipes

Double routes are written `gray \|\| gray` — markdown-escaped pipes. Splitting a table row
naively on `|` silently yields **78 segments / 256 spaces instead of 100 / 309**, and the
parse looks successful. Un-escape `\|` before splitting and restore after:

```python
protected = line.replace(r"\|", "\x00")
cells = [c.strip().replace("\x00", "|") for c in protected.strip("|").split("|")]
```

Cost me one wrong answer before I noticed the counts disagreed with RULES.md's own header.
`tests/unit/engine/test_rules_md_invariants.py` asserts the counts precisely so this can't
regress silently.

### `torch>=2.13` is load-bearing, not a style preference

torch 2.12 **dropped the `macosx_11_0_arm64` wheel tag**; wheels are now
`macosx_14_0_arm64` only. uv's `aarch64-apple-darwin` platform alias implies a macOS 11
deployment target, so a loose constraint **silently resolves torch 2.11** instead of 2.13 —
no error, no warning, just an old torch. The `>=2.13` pin turns that downgrade into a hard
resolution failure. Verify with:

```bash
uv lock && grep -o 'torch-[0-9.]*-cp314[^"]*macos[^"]*\.whl' uv.lock
```

Expect `torch-2.13.0-cp314-cp314-macosx_14_0_arm64.whl`.

### torch on Linux drags in ~3 GB of CUDA

Plain `torch` on linux/cp314 resolves 18 CUDA packages. `[tool.uv.sources]` routes Linux
through the `pytorch-cpu` index. Removing that block makes every CI run download gigabytes
it will never use. macOS is unaffected (no CUDA wheels exist).

### `.gitignore` has no trailing comments

`runs/  # training runs` is a literal pattern named `runs/  # training runs`, not `runs/`
with a comment. Comments must be on their own line. Silent, and the directory stays tracked.

### `filterwarnings = ["error"]` + pytest-benchmark + xdist = INTERNALERROR

pytest-benchmark emits a warning when xdist is active ("Benchmarks are automatically
disabled…"). Under `filterwarnings = ["error"]` that becomes a **collection-time
`INTERNALERROR`**, so the entire suite fails to run with a stack trace that doesn't mention
your code. A scoped ignore is in `[tool.pytest.ini_options]`; keep the strict `error`
default and keep the scoped ignore.

### Typer collapses a single-command app

A `typer.Typer()` holding exactly one command drops the subcommand name, so `ttr version`
fails with "Got unexpected extra argument(s)". `ticket_to_ride/cli/app.py` defines an empty
`@app.callback()` (`_root`) purely to force multi-command mode. **Do not delete it as dead
code** — it breaks the moment the app is down to one command, which is easy to hit while
refactoring.

### `ruff format` rewrites Python inside markdown

ruff 0.16 formats fenced ```python blocks in `.md` files. That silently rewrites design
docs — it tried to split `a.step(act); b.step(act)` across two lines in a PLAN.md code
sketch, which failed `ruff format --check` and would have turned CI red. `docs/` is in
`extend-exclude` for this reason. If you add a docs directory elsewhere, exclude it too.

### ty has no `disallow_untyped_defs`

`ty` (0.0.63) is inference-based. An unannotated function type checks clean, so swapping
mypy → ty without compensation silently drops annotation enforcement. Ruff's `ANN` ruleset
covers it (`ANN001` args, `ANN201`/`ANN202` returns), with `ANN401` ignored. **Removing
`ANN` from `[tool.ruff.lint] select` loses the guarantee with no visible signal.**

Also: `ty` is 0.0.x and pre-1.0. Expect config-schema churn on upgrades. Verify config keys
are recognized rather than silently ignored — a bogus key should be *rejected*:

```bash
uv run ty check -c 'terminal.nonsense-key=true'   # must error
```

### ty needs torch installed to check `rl/`

Hence `typecheck` is a separate CI job that syncs `--all-extras`, while `lint` (ruff) and
`test-fast` stay torch-free. Merging them back would either slow the fast loop or produce
spurious `unresolved-import` diagnostics.

### `aim` is unusable on cp314

Resolving `aim` drags in `tensorboard==2.3.0` (2020), `pysqlite3`, `pytz==2020.1`. Ruled
out; tracking is TensorBoard with a JSONL dual-write. `wandb` and `mlflow` do resolve fine
if ever wanted.

### Free-threaded 3.14t is not viable yet

numpy/scipy/torch/msgspec ship `cp314t` wheels, but **polars ships abi3 wheels, which are
ABI-incompatible with free-threaded CPython**. Revisit later; standard 3.14 is the pin.

### The face-up display can empty permanently

Refilling only when a face-up card is *taken* looks right and has an absorbing state: once
every pool empties the display goes to five empty slots, and then nobody can take a face-up
card, so nothing ever triggers a refill again -- even after a claim puts five cards back in
the discard. The engine also refills at **end of turn**; see docs/CONTRACT.md §2.6. Costs
five comparisons a turn.

### The ticket ring must blank vacated slots

`tdeck` is a ring buffer, and the whole array is serialized into `state_hash()`. Leaving
stale ticket ids in consumed slots makes two semantically identical positions hash
differently, so a drawn slot is set to 255 on the way out. Same class of bug as padding
bytes in a struct hash.

### `state_hash()` must exclude the union-find

The per-player DSU is a *cache*, fully derivable from `seg_owner`. Including it would let a
pure-performance change to path compression move the hash and break differential testing
for no reason. The consumed deck prefix is the mirror-image call: `state_hash()` keeps it
(identical game means identical down to the unrealized future) and `position_hash()` zeroes
it (two states differing only in dealt-card *order* are the same position).

### `step()` must record history *after* it validates

Appending the action first leaves a phantom entry when the action is rejected, and the
replay of that game then diverges. Every `_step_*` path validates before it mutates, so the
history has to follow the same rule. Caught by
`test_a_rejected_action_leaves_the_state_untouched`.

### A colour only has twelve cards

Rigging a test deck as `[0] * 20` asks for twenty black cards when the box holds twelve.
`tests/rig.py` raises rather than silently truncating, and `rig.filler()` produces a legal
run. The same arithmetic bites when handing several seats the same colour: use
`rig.spread_hands()`.

### `ty` rule names are not guessable, and a wrong one fails the build

`# ty: ignore[possibly-unbound-attribute]` is not a rule -- it is `possibly-missing-attribute`;
`non-subscriptable` is `not-subscriptable`. An unknown rule name is a *warning*, and
`error-on-warning = true` turns it into a build failure. So does an ignore that stops being
necessary. Read the diagnostic; it suggests the right name.

### PLAN.md's estimates that turned out different

Three, all harmless but worth not rediscovering: the claim-mask buckets are **45**, not 33
(§5.3); the observation is **3355** dims on the USA map, not ~2000-2800 (§6.1), because the
100x9 required-colour block is genuinely needed as an input; and the engine reaches ~1100
random USA 2P games/s, not the 2000 in the §14 exit criterion. TTR-mini clears 2000. See
the throughput note in `tests/unit/test_cli.py`.

### Path halving does not flatten a chain in one pass

`dsu_find` uses path *halving* -- one pointer update per step, no second pass -- so a test
asserting `parent[x] == root` after a single find on a 20-long chain fails. Repeated finds
converge; that is the trade being made.

### A "first subset size that works" Steiner brute force is wrong

More edges can weigh less. An oracle that returns as soon as some edge subset of size *k*
connects the terminals will happily report a worse tree than the DP found. The check that
actually works for three terminals is the closed form: `min over v of d(v,a)+d(v,b)+d(v,c)`
in the shortest-path metric.

---

## Part 2 — known traps in the work ahead

These come from the design work in [PLAN.md](PLAN.md).

The first seven were Phase 1 traps and are now **handled** -- kept here because each one is
a thing to check when the Rust port reimplements it, and the note is the check. Everything
after "Never report a mirrored win-rate matrix" is still ahead.

### The 3-locomotive flush can hang forever  ✅ handled

When 3+ of the 5 face-up cards are locomotives, all five are discarded and replaced — and
the replacement can itself contain 3 locomotives, so it cascades. With 14 locomotives in the
deck, an all-locomotive available pool **loops forever**. Guard the loop on
`nonloco(deck) + nonloco(discard) >= 3` plus a hard iteration cap. This is the most common
hang bug in TTR implementations, and the assertion that catches it (never 3+ locomotives
face-up unless the guard blocks the flush) is the one most engines lack. See PLAN.md §5.2.

### The deck must be a pre-materialized permutation  ✅ handled

Sampling lazily from card *counts* + RNG is distributionally identical for play, and is
tempting because it makes clones smaller and reshuffles free. **It silently destroys paired
evaluation.** If agent A draws 3 cards and agent B draws 5, the two mirrored games realize
*different shuffles*, and the entire variance-reduction scheme evaporates with no error.
Materialize the permutation at game construction so environment randomness is a pure
function of the seed, independent of agent behavior. Keep a counts *view* for determinization
sampling. PLAN.md §5.1.

### Double-route rules differ by player count  ✅ handled

2–3P: once *either* track of a double pair is claimed by anyone, the sibling is closed to
**everyone**. 4–5P: the sibling stays open to other players, but one player may never own
both tracks. Getting this wrong is the single most common TTR implementation bug. Test both
directions — and watch the agent's double-route claim rate: if the engine wrongly locks both
halves, a trained agent will silently learn to avoid them and nothing will look broken.

### "Longest continuous path" is a longest *trail*, weighted in train cars  ✅ handled

Not a path, not a segment count. Each segment may be used at most once, cities may repeat,
loops are allowed, and the length is measured in **train cars**, not routes. **All tied
players score the +10 bonus.** It's NP-hard in general but bounded here (≤25 segments), and
the component-split + Eulerian-shortcut layers matter: without them the adversarial case is
~126 ms, which shows up as a p99.9 latency spike in self-play workers. PLAN.md §5.6.

### Potential-based shaping requires `Φ(terminal) ≡ 0`

If `Φ(s)` = "score if the game froze here" and the terminal potential isn't zeroed, the
shaping telescopes to `+final_score` and you have **silently switched from optimizing win
probability to optimizing `win + λ·score`** — a real objective change, not a free lunch.
Put this in the code comment; it is the single most commonly botched piece of reward
shaping. PLAN.md §6.3.

### Action space is 915, and the keep-mask starts at 1  ✅ handled

900 claim + 6 draw + 1 draw-tickets + **7** keep + 1 pass. The ticket-keep action is
`bitmask - 1` for `bitmask ∈ 1..7` — keeping *nothing* is never legal, so there are 7 keep
actions, not 8. Off-by-one here shifts every action index above it.

### Claim legality needs the `hand[c] >= 1` guard  ✅ handled

Without it, a hand of pure locomotives makes all 8 gray-color pay slots legal *and
identical*, so eight distinct action ids denote the same payment and the policy distribution
is poisoned. With it, action ids map bijectively to payments.

### Never report a mirrored win-rate matrix

If every off-diagonal pair sums to exactly 1.00, you reported one seating and its complement
rather than measuring both — and first-player advantage is confounded into every cell. This
is a real flaw in the published prior art. Measure both seatings.

### Aggregate to seed blocks before computing statistics

Games within a paired seed block are correlated. Aggregate to a per-block score first, then
treat **blocks** as the i.i.d. units; bootstrap CIs must resample blocks, never games.
Treating paired games as independent inflates significance by ~√P and manufactures
improvements that aren't there.

### Heuristic constants must be config-driven and re-tuned per map

The published prior art's baselines were invalidated by hard-coded thresholds (`draw a
ticket only if trains > 15`) carried onto a 10-train map, where the condition could never
fire — so every "well-designed heuristic" played its opening tickets and never drew more.
Ship a test asserting each heuristic actually exercises **every action type** on **every
map**, including drawing extra tickets.

### Freeze the PRNG, draw procedure, and `state_hash` *before* writing Rust  ✅ handled

Differential testing is impossible unless all three are pinned bit-for-bit with test
vectors. Two equally-correct sampling implementations produce different — still valid —
trajectories, which destroys the oracle. This is the gate between Phase 1 and Phase 2.
PLAN.md §5.8. **Done: [CONTRACT.md](CONTRACT.md) plus `tests/golden/contract_vectors.json`
and the 84-game `tests/golden/replays.bin` corpus.**

### Compare Python vs Rust at every step, not just at the terminal

Terminal-only comparison tells you *that* you diverged, never *where*. Assert
`state_hash()` equality and `legal_actions()` set equality after every single step.

### MPS will silently fall back to CPU

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` and **grep the logs for fallbacks** — one silent
CPU-fallback op in the trunk will 10× your step time invisibly. Also: no `.item()` or
`.cpu()` in the hot loop (each is a full GPU sync), `pin_memory` is meaningless on unified
memory, and validate bf16 by comparing one CPU and one MPS training iteration at the same
seed to 1e-3 before trusting it.

### `policy/mask_violations` must be exactly 0

Log it and assert on it. Apply the mask *before* the softmax, and **recompute it at update
time** from the stored state so old and new log-probs share support. A stale or omitted mask
in the PPO ratio is the classic silent RL bug — it poisons training with no visible symptom.
