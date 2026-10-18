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
`tests/unit/data/test_rules_md_invariants.py` asserts the counts precisely so this can't
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

Hence `make type` needs the `--all-extras` environment `make setup` builds, while the test
suite stays torch-free. Checking without torch produces spurious `unresolved-import`
diagnostics.

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
`properties::a_rejected_action_leaves_the_state_untouched` (crates/ttr-core/tests/).

### A colour only has twelve cards

Rigging a test deck as `[0; 20]` asks for twenty black cards when the box holds twelve.
The test rig (`crates/ttr-core/tests/common/mod.rs`) panics rather than silently truncating, and `filler()` produces a legal
run. The same arithmetic bites when handing several seats the same colour: use
`spread_hands()`.

### `ty` rule names are not guessable, and a wrong one fails the build

`# ty: ignore[possibly-unbound-attribute]` is not a rule -- it is `possibly-missing-attribute`;
`non-subscriptable` is `not-subscriptable`. An unknown rule name is a *warning*, and
`error-on-warning = true` turns it into a build failure. So does an ignore that stops being
necessary. Read the diagnostic; it suggests the right name.

### The locomotive-flush cascade cap is load-bearing, not belt and braces

It fires in real late-game 5P positions: once most of the deck is in players' hands the
available pool can be small and locomotive-heavy, and every reflush deals three more of
them. Measured on seed 15 of the 5P sweep, at turn 243. Found by *strengthening*
`validate()`'s flush assertion, which until then read
`nonloco_available < 3 or deck_has_cards` — and the second clause made it pass vacuously in
almost every state. `State.flush_capped` now records the bail-out explicitly so the
assertion can distinguish "the guard blocked it" from "the engine forgot to flush".

### PLAN.md's estimates that turned out different

Three, all harmless but worth not rediscovering: the claim-mask buckets are **45**, not 33
(§5.3); the observation is **3355** dims on the USA map, not ~2000-2800 (§6.1), because the
100x9 required-colour block is genuinely needed as an input; and the engine reaches ~1100
random USA 2P games/s, not the 2000 in the §14 exit criterion. TTR-mini clears 2000. See
the Phase 1 entry in WORKLOG.md (the Python engine and its throughput floor are retired).

### Path halving does not flatten a chain in one pass

`dsu_find` uses path *halving* -- one pointer update per step, no second pass -- so a test
asserting `parent[x] == root` after a single find on a 20-long chain fails. Repeated finds
converge; that is the trade being made.

### A "first subset size that works" Steiner brute force is wrong

More edges can weigh less. An oracle that returns as soon as some edge subset of size *k*
connects the terminals will happily report a worse tree than the DP found. The check that
actually works for three terminals is the closed form: `min over v of d(v,a)+d(v,b)+d(v,c)`
in the shortest-path metric.

### `sum()` in CPython is not naive summation

Since 3.12 the builtin `sum()` takes a **Neumaier compensated** fast path for floats.
Rust's `Iterator::sum` adds left to right with no compensation. They disagree in the last
ULP on inputs as ordinary as `[0.0, 1.0, 2/3, 1/3]` — Python gives exactly `2.0`, Rust
gives `1.9999999999999998` — and the gap propagates through any mean computed from them.
It surfaced as every seat's `returns()` differing in the 16th digit.

The giveaway probe:

```python
sum([1e16, 1.0, -1e16])   # 1.0 in Python; 0.0 with a naive fold
```

`crates/ttr-core/src/numeric.rs` transcribes Neumaier and `returns()` uses it. **It was
only caught because the harness compares returns exactly rather than within a tolerance** —
and a tolerance would equally have hidden a real difference in the formula, which is the
whole reason not to use one.

### `cargo fmt` rewrites generated Rust, and the file-wide opt-out does not compile

rustfmt explodes the generated board and obs-spec tables to one entry per line (+894 lines)
and sets up a permanent fight: `make board` writes the compact form, the fmt hook rewrites
it, and `gen_board.py --check` then fails on a file nobody edited. Same class as `ruff
format` rewriting Python inside `docs/*.md`.

The fix is a **per-item outer** `#[rustfmt::skip]`, emitted by the generators. The obvious
file-wide `#![rustfmt::skip]` is a *custom inner attribute*: rustfmt honours it, and rustc
rejects it as unstable ([rust-lang/rust#54726]) — so the crate silently formats fine and
then does not build.

[rust-lang/rust#54726]: https://github.com/rust-lang/rust/issues/54726

### A differential fast tier that only drives USA 2P is worthless

Deleting the end-of-turn refill from the Rust engine — a real bug, and the one Phase 1
found the hard way — survives **60 seeds of USA 2P, 3P and 4P undetected**. Reaching it
requires the deck *and* the discard to run dry, which those configurations rarely do in a
random game. It is caught at mini seed 0 (3P and 4P) and USA 5P seed 8.

So a fast tier covers every configuration -- today's, the golden-corpus prefix, runs the
small map and the full table. The general lesson: pick the fast
tier's shape by mutating the engine and seeing what catches, not by taking the biggest map.

### Comparing hashes tells you *that*; comparing fields tells you *where*

Learned by the per-step differential harness (retired with `oracle-v1`), and still true of
any diff of two states. Diff the serialized image **field-wise
using each side's own `deck_len`**, not byte-wise: a reshuffle at a different moment
changes `deck_len`, which changes the image length, and a byte-offset diff then reports
"the images are different lengths" — which reads like a layout bug and sends you to check
field widths when the real difference is one value several fields earlier.

### `make bench-check` needs an idle machine

Run alongside anything else — `make rust-sweep` is the obvious offender — pytest-benchmark reports a **40–80% "regression"** that is purely core
contention, complete with a `PerformanceRegression` traceback that names your code.
Measured, not hypothesised. The absolute figures are only meaningful on a quiet box, which
is also why the gate is local-only and the test suite asserts a *ratio* instead.

### `cargo test --workspace` was aborting on macOS and reading as green

The fourth member of the `extension-module` family below, and the nastiest, because it
**passed inspection for a whole phase**. `cargo test` builds a test executable from the
ttr-py cdylib. It links, because `.cargo/config.toml` supplies `-undefined dynamic_lookup`;
then it dies at load with `dyld: symbol not found in flat namespace '_PyBaseObject_Type'`,
because `extension-module` deliberately does not link libpython and nothing is hosting an
interpreter. The four ttr-core suites run first and all pass, so the output is a page of
`ok` followed by a `SIGABRT` you have to scroll to.

`[lib] test = false` in `crates/ttr-py/Cargo.toml`. That crate converts types and holds no
logic by design; its behaviour is covered from Python by `tests/unit/rust/`. **Do not remove
it to "add a quick unit test there"** — the test would be unrunnable, and its failure would
hide behind everything that passes.

### SQLite's `INTEGER` is signed, and a `u64` hash overflows it

`state_hash()` is a `u64`. Anything above 2⁶³ raises
`OverflowError: Python int too large to convert to SQLite INTEGER` on insert. Storing the
two's-complement view would work and would print as a negative number that matches nothing
the engine ever reports, so `game.final_hash` is 16 hex characters instead.

### A heuristic constant that transfers can still fail to transfer

The prior-art lesson is "express thresholds as fractions of a board quantity". Necessary,
and **not sufficient** — this project hit the same failure twice with perfectly
board-relative constants:

- The keep filter PLAN.md §7 specifies, `marginal_steiner_cost <= points`, has no fraction
  in it at all and looks board-independent. On TTR-mini, where ticket points *are*
  shortest-path costs by construction, marginal always equals points, so it accepts
  everything: 3 opening tickets costing 22 train cars against a 20-train supply, in 59 of
  60 games. `draw_tickets` then fired **zero times in 30 mini games** — correctly, because
  the plan was already unaffordable.
- Pricing a ticket draw off the *mean* of the deck rather than the best of the three
  actually offered. On a board where a ticket costs half your trains, the difference between
  the mean and the expected minimum of three decides whether the action ever fires.

The check that works is the one §14 asks for: **count the action types an agent actually
reaches, on every map**. `crates/ttr-core/tests/heuristics.rs` does it over 40 games per
(map, seat count). Both bugs surfaced as `draw_tickets == 0` on mini and nothing else.

### A blocking heuristic tuned at 2P gets worse as seats are added

H4 measured **+21 Elo on TTR-mini 2P and −50 on USA 4P** — the same code and the same
constants, opposite signs. Ablation localised it exactly: threat alone −12.9 at 4P, hoarding
alone −12.3 at 5P, both off exactly 0.0.

The cause is structural, not a mis-tune. Cars spent denying one opponent help every *other*
rival as much as they help you, so the share you capture is about `1/(P−1)`; and
"waiting costs nothing" fails for the same reason, because more rivals means the parallel
route is likelier gone when you come back. Both terms are divided by the rival count, which
took 4P from −50 to −6.

It is still mildly negative above two seats, and the remaining gap is a design gap: **the
free-rider structure says block the leader, and H4 blocks whoever is most blockable.** Any
3–5P opponent model needs to know who is winning.

### 80 seed blocks cannot resolve a 30-Elo difference

The standard error on a mean return over `n` paired blocks is about `1/sqrt(2n)`, so 80
blocks is ±0.08 — roughly ±30 Elo. Every H4 ablation "difference" measured at that size was
inside it, and the seat-count result above only became visible at 10k games. If a difference
matters, measure it with the arena and read the CI; do not eyeball a mean return from a
scratch harness.

### PyO3: `extension-module` breaks a plain `cargo build` on macOS

The feature deliberately does not link libpython. maturin passes
`-undefined dynamic_lookup` itself; a bare `cargo build`/`cargo test` over the workspace
does not, and the failure is a wall of undefined `_Py*` symbols that looks like a broken
toolchain. `.cargo/config.toml` supplies it. Linux needs nothing.

Two more from the same family. `panic = "abort"` must **not** be set in the release
profile: PyO3 turns a Rust panic into a Python exception by unwinding across the FFI
boundary, and Cargo does not allow `panic` in a per-package override, so it cannot be set
globally and exempted for the shim. And a `#[pyclass]` that derives `Clone` gets an
automatic `FromPyObject` — for an RNG handle that means the stream is **copied at the call
boundary**, the caller's generator never advances, and every determinization comes out
identical. `skip_from_py_object` is load-bearing.

### uv only rebuilds a maturin package when its cache keys say so

`ttr_rust` is a uv workspace member built by maturin. uv decides whether a local package is
stale from its `cache-keys`, which default to Python metadata (`pyproject.toml`,
`setup.py`, `setup.cfg`) -- so **a Rust edit leaves the old extension installed, with no
warning**. Verified: an edit to ttr-core changed nothing until
`crates/ttr-py/pyproject.toml` listed the Rust sources, including ttr-core's through `../`.
The keys compare modification times, so even a `touch` rebuilds; cargo's incremental cache
keeps that cheap. Add a new crate the extension depends on, add its sources to the list.

### The test profile is opt-level 1 on purpose

`[profile.test] opt-level = 1` in the workspace `Cargo.toml`. Unoptimized, `cargo test -p
ttr-core` took ~145 s, nearly all of it `heuristics.rs` playing H3 games; at opt-level 1 it
is ~20 s, and mutation testing runs the whole suite once per mutant. Debug assertions and
overflow checks stay on. Do not "restore the default" without measuring.

### Coverage needs rustup's toolchain, not Homebrew's

`cargo llvm-cov` needs the `llvm-tools` component, which a Homebrew Rust does not have. On
this machine Homebrew's `cargo` is first on `PATH`; `make rust-cov` runs
`RUSTUP_TOOLCHAIN=stable cargo llvm-cov`, which uses rustup's stable toolchain (1.87 here,
inside the workspace's 1.85 floor) with `llvm-tools-preview` installed.

### A stub constant nobody checks

`ttr_rust.pyi` declared `CONTRACT_VERSION: int` from Phase 2 on, and the module never
defined it: it type checked, and failed at import the first time anything used it (the
thin `ticket_to_ride.engine` re-export, in Phase 3.1). The stub test compared functions and
classes in both directions and never looked at module constants. It does now.

### A ported test can be weaker than it looks

Porting the Python suite turned up two of its tests that could not fail for the reason
they were written: the flush-guard test passed with the guard deleted (re-flushing an
all-locomotive pool deals five locomotives straight back, so the display "didn't change"),
and the sweep's flush detector compared face-up slots with `FREE` (255, a *segment*
sentinel) instead of `EMPTY_SLOT` (-1), so its condition was always true. Both were found by
breaking the engine on purpose and watching which test failed. The Rust ports check the
deck cursor and discard, and use the right sentinel. **A test is only evidence once
something has been seen to make it fail.**

### `longest_trail` numbered its mask bits across the whole network

Found by the exhaustive reference in `naive_graph`, not by any game: each edge's bit in the
`u32` DFS mask was its index among *all* the seat's segments, so a seat owning more than 32
overflowed the shift even when every component fit -- a debug panic, and a short trail in
release. Unreachable in play (a seat can buy at most 27 USA segments), and fixed: bits are
numbered per component. The general lesson is the one the naive references exist for: the
fast path's guards and limits are where a brute force finds things random play never does.

### A benchmark that times its own checksum

`bench-obs` summed each observation's 3,355 floats into an `f64` checksum inside the timed
loop. Floating-point addition cannot be reassociated, so that sum is a serial chain of
dependent adds -- about 3 µs per observation, invisible at 125 µs and a third of the figure
at 9. It now times only the encoder calls. **When the thing measured gets fast, re-check
what else the timer covers.**

### Modal's `cpu=` is a request, not a limit

`@app.function(cpu=8.0)` reserves eight cores and caps nothing: the container may burst,
and it *shows* the whole host slice -- 24 CPUs, no cgroup quota visible under gVisor -- so
`available_parallelism` said 24 and the engine ran 22 threads on an 8-core reservation.
With `cpu=(8.0, 8.0)` (request and limit) gVisor shows the container exactly 8 CPUs and
the engine follows. `ttr cloud` always sets both, and the job sets `TTR_THREADS` to the
visible count, because the engine's desktop default (all CPUs but two) leaves a quarter of
a dedicated 8-core container idle.

Related, in the image: `uv sync` is exact, so syncing the project into the interpreter
Modal runs the function on would uninstall Modal's own client. The project gets its own
venv.

### PyO3 returns `Vec<u8>` as `bytes`

`VecEnv.terminal_flags()` and `current_players()` come back as `bytes`, and
`np.array(b"\x01\x00", dtype=bool)` is a single scalar `True`, not two flags. The PPO
spike's first runs treated every environment as finished on every step. Use
`np.frombuffer(flags, dtype=np.uint8)`. The stub said `list[int]`; it says `bytes` now.

Six `State` accessors had the same lie in the stub until Phase 4.1 -- `hand_of`,
`certain_of`, `tickets_of`, `deck_counts` and the `trains` and `seg_owner` properties all
return `bytes` -- and it bites again at the web layer: `json.dumps` refuses `bytes`, and a
Pydantic `list[int]` field accepts them only by coercion. The stub says `bytes` now; the
accessors added for the web server (`offer`, `discard`, `ranks`, `board_data`'s arrays)
widen to `u16` so they arrive as lists.

### `explain` *is* `act`: call one or the other, never both

`Heuristic.explain` runs the same decision as `act` with a recording sink, so it advances
the agent exactly as `act` does -- including H0's random stream. Calling `explain` for the
reasoning and then `act` for the move draws twice, and from then on the game diverges
from any replay of it. The server calls `explain` *instead of* `act` when it wants the
reasoning (spectating, review) and `act` otherwise; a rebuild calls `act`, which reproduces
an `explain`ed game because the two are one code path
(`crates/ttr-core/tests/explain.rs::explaining_h0_draws_exactly_what_acting_would`).

### H1's priority scale must stay above every value it ranks

Phase 4.1 wrote H1's rule cascade as one priority per action, `rule * 100 + value`, so its
explanation is what it compared. That is the cascade only while every value -- a route's
points, cards of one colour in hand, an offer's total cars (69 at most on USA) -- stays
below the scale; a board whose three-ticket offer could need 100 cars would let a value
outrank a rule. A `debug_assert` guards it, and H1's behaviour hash would move.

### A seat view can be resample-tested only where the sampler resamples

The masking test compares a seat's view of a position with its view of
`resample_from_infoset(seat)` of it. The sampler redeals hidden cards and the deck, but
keeps every seat's tickets and any open offer as they are (`ttr_core::resample`), so an
opponent's tickets or offer leaking into a view would pass that test. Those are asserted
directly (`tests/unit/web/test_masking.py`); a new hidden field needs the same.

### A one-block bootstrap is a point, not an interval

Resampling a single seed block draws the same games every time, so a competitor with one
block -- a person after one rated game, each human game being its own block -- gets
`lo == elo == hi`, which reads as certainty. The web ratings report no interval below two
blocks (`MIN_INTERVAL_BLOCKS`).

### Starlette's test client and its HTTP library

Starlette 1.3 deprecates `httpx` for `starlette.testclient` in favour of `httpx2`, and
warns at import. Under `filterwarnings = ["error"]` that warning fails every test that
imports `fastapi.testclient`. The dev group has `httpx2`.

### `asyncio.TaskGroup` inside a Starlette WebSocket handler

Starlette ends a connection by cancelling its handler through an anyio cancel scope, and
`asyncio.TaskGroup`'s own cancellation bookkeeping turns that into a `CancelledError` that
escapes the handler -- in the test client it surfaced as `concurrent.futures.CancelledError`
on leaving `websocket_connect`. The game socket uses an anyio task group.

### Two test files may not share a basename

`tests/` has no `__init__.py`, so pytest imports each test file by its basename, and
`tests/unit/web/test_store.py` beside `tests/unit/eval/test_store.py` is an "import file
mismatch" that stops collection. The web one is `test_play_store.py`.

### `Categorical.sample()` on MPS can pick a masked action

With masked logits at -1e9, torch's categorical sampling on MPS still returned a masked
index -- an illegal move the engine refused. Sample by Gumbel-max (`argmax(logits -
log(-log(U)))`), which cannot select a -1e9 logit on any device, and keep counting
`mask_violations`.

### The client needs Node 22.12, and TypeScript 6 for svelte-check

Vite 8, its Svelte plugin and Vitest 5 require Node `^20.19 || >=22.12` (they lean on
`require(esm)`); an older Node -- nvm's 22.3 on the dev machine -- fails at startup.
`web/.nvmrc` names 24 and CI reads it. And svelte-check 4.7 supports TypeScript `^5 || ^6`
only, while npm's `latest` is TypeScript 7 (the Go port): `typescript` is pinned `~6.0`.

### `.gitignore`'s `lib/` would swallow Svelte's `src/lib/`

The Python template's `lib/` pattern matches at any depth, so a SvelteKit-style
`web/src/lib/` would be silently untracked. The client has no `lib/` directory; keep it
that way, or add a negation.

### A CSS transform replaces an SVG element's `transform` attribute

A Svelte transition animating `transform` on `<g transform="translate(...) rotate(...)">`
moves the element to the origin while it runs: the CSS property wins over the attribute.
The trains' drop animates an outer `<g>` that has no transform attribute of its own.

### Alegreya's old-style figures draw 0 as a small o

The body font's default figures are old-style, and its zero is the height of a lowercase
letter: "0 of 45 train pieces" read as "o of 45". Counts a player acts on use
`font-variant-numeric: lining-nums` (the seat panels' stats, your own counts), and the log
says "no train pieces" rather than "0".

### A `.choices` row can shrink to nothing in a flex row

`.choices` hides its overflow, for its rounded corners, and an element that hides its
overflow has no automatic minimum size in flexbox: in the game's status bar a long prompt
squeezed the speed control until it was clipped. Beside text that can wrap, give it `flex:
none`.

### `math.copysign(x, y)` keeps only the sign of `y`

`gen_layout.py` put a double route's tracks at `radius + copysign(offset, h)`, which is the
same radius for `+offset` and `-offset`: both tracks of every bowed double drew on one line,
and the crowding check reported them overlapping each other. It is `offset * copysign(1,
h)`; `test_the_tracks_of_a_double_stay_parallel` holds their separation.

---

## Part 2 — known traps in the work ahead

These come from the design work in [PLAN.md](PLAN.md).

The first seven were Phase 1 traps and are now **handled** -- kept here because each one is
a thing to check when the Rust port reimplements it, and the note is the check. **Phase 2
did that reimplementation**, and every one of them held: the Rust engine was
byte-identical to the Python reference across every map and seat count, which is what let
Phase 3.1 retire the Python engine. They stay because Phase 5's search and Phase 6's
vectorized env will touch the same rules again.

**Phase 3 handled the three evaluation traps** -- mirrored matrices, block aggregation, and
config-driven heuristic constants. The last of those was hit twice on the way, in a form
the note as written did not cover; see "A heuristic constant that transfers can still fail
to transfer" in Part 1.

Everything from "Potential-based shaping" onward is still ahead.

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

### The Elo anchor is pinned by behaviour, and moving it re-bases every stored rating

H3 is the permanent anchor of the rating scale (PLAN.md §11), fixed at Elo 1500
(`eval.stats.ANCHOR_ELO`). Its identity is the **behaviour
hash** in `crates/ttr-core/src/heuristic/probe.rs` -- its action sequence over ten frozen
probes -- not its constants, because a params hash pins the numbers and leaves a reordered
tiebreak or a fixed bug free to move its play.

`crates/ttr-core/tests/heuristics.rs::the_h3_anchor_has_not_moved` holds it. **If that test
fails, do not refresh the literal to go green.** Either register the new H3 as a separate
agent (`h3@config.toml`) rated against the old anchor, which is the supported path, or move
the anchor deliberately and record the re-basing in the worklog -- every stored rating
predating it is then on a different scale and the leaderboard cannot tell.

The converse is a feature and was exercised in Phase 3: every H4 change left H3's hash
untouched, because H3 cannot read those code paths. A params hash would have demanded an
anchor re-base for a change H3 never sees.

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

### Never report a mirrored win-rate matrix  ✅ handled

If every off-diagonal pair sums to exactly 1.00, you reported one seating and its complement
rather than measuring both — and first-player advantage is confounded into every cell. This
is a real flaw in the published prior art. Measure both seatings.

**Done: every rotation of every seed block is played** (`crates/ttr-core/src/arena.rs`), and
a test asserts each agent occupies each seat the same number of times — which is the
property that makes the resulting symmetry honest rather than assumed.

One follow-on that was not obvious in advance. §14 also asks for a **flat seat win rate**
and calls it the canary that mirroring works. That is half right: rotation removes seat bias
from *agent* ratings by construction, while the raw per-seat rate stays a property of the
game. Measured at 10k games it is **not** flat — seat 0 wins 0.5070 [0.5003, 0.5137] on USA
2P and 0.5197 [0.5110, 0.5290] on mini — and that is genuine first-player advantage, not a
broken harness. Report it with an interval; asserting it flat turns a finding into a failing
test.

### Aggregate to seed blocks before computing statistics  ✅ handled

Games within a paired seed block are correlated. Aggregate to a per-block score first, then
treat **blocks** as the i.i.d. units; bootstrap CIs must resample blocks, never games.
Treating paired games as independent inflates significance by ~√P and manufactures
improvements that aren't there.

**Done: `ticket_to_ride/eval/stats.py` takes a block index everywhere and never accepts a
flat list of games**, and a test measures the difference rather than trusting it -- a
game-level bootstrap over perfectly-correlated blocks produces intervals dramatically
narrower than the block-level one.

Two follow-ons. **SPRT must stop on block boundaries too**: half a block is one seating, so
an early stop mid-block reports exactly the mirrored result blocks exist to prevent. And
blocks are keyed by their **seed**, not by the match that recorded them -- block seeds are
`seed_root + i`, so two runs at the same root replay the same decks, and counting them as
distinct blocks would quietly restore the independence assumption.

### Heuristic constants must be config-driven and re-tuned per map  ✅ handled

The published prior art's baselines were invalidated by hard-coded thresholds (`draw a
ticket only if trains > 15`) carried onto a 10-train map, where the condition could never
fire — so every "well-designed heuristic" played its opening tickets and never drew more.
Ship a test asserting each heuristic actually exercises **every action type** on **every
map**, including drawing extra tickets.

**Done: every constant lives in `HeuristicParams`, and the coverage test is
`crates/ttr-core/tests/heuristics.rs`.** It earned its place immediately — it caught two
separate bugs, both of which had perfectly board-relative constants and still failed to
transfer. See "A heuristic constant that transfers can still fail to transfer" in Part 1:
the fractions are necessary, and the *test* is what actually catches it.

### Freeze the PRNG, draw procedure, and `state_hash` *before* writing Rust  ✅ handled

Differential testing is impossible unless all three are pinned bit-for-bit with test
vectors. Two equally-correct sampling implementations produce different — still valid —
trajectories, which destroys the oracle. This is the gate between Phase 1 and Phase 2.
PLAN.md §5.8. **Done: [CONTRACT.md](CONTRACT.md); the golden corpus pins it now.**

### Compare against the oracle at every step, not just at the terminal  ✅ handled

Terminal-only comparison tells you *that* you diverged, never *where*. Assert
`state_hash()` equality and `legal_actions()` set equality after every single step.
**Done while two implementations existed:** Phase 2's Python-versus-Rust harness, then
the live crate against `oracle-v1` -- retired 2026-10-17, once the golden corpus (a hash
chain over every position) and the naive references carried the check. Two follow-ons are
in Part 1: field-wise diffing, and choosing the fast tier's shape by mutation testing.

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
