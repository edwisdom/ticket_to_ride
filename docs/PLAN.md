<!--
Approved implementation plan, saved verbatim from the planning session on 2026-07-26.
This is the design of record. If you deviate from it, note the deviation and the reason
in docs/WORKLOG.md rather than silently editing this file.

Revision 1, 2026-10-15, at the user's direction: the Python engine is retired and the Rust
core becomes its own oracle. New phases 3.1-3.4 are inserted between Phases 3 and 4 (§14,
§14.1). Superseded text is marked in place rather than deleted. Reasoning in docs/WORKLOG.md.

Revision 2, 2026-10-17, at the user's direction: Phase 4 becomes a web client and the terminal
client is dropped; the web UI moves up from Phase 8 (§14.2). Superseded text is marked in place.
-->

# Ticket to Ride — Game Engine + Self-Play RL

## Context

`/Users/ejain/Projects/ticket_to_ride` is a bare `uv init` scaffold (empty package, zero deps) plus
[RULES.md](RULES.md), which contains the complete rules, the full 100-segment USA edge list, and all
30 destination tickets.

The goal is a research platform: a fast, exactly-correct Ticket to Ride engine, then iteration on RL
algorithms via self-play until agents beat strong heuristic opponents and eventually human players.

**Decisions confirmed with the user:**
- Pure-Python reference engine first; Rust core (PyO3/maturin) once rules are frozen, with the Python
  engine kept permanently as a differential-testing oracle. *(Superseded by Revision 1: the Python
  engine is retired in Phase 3.1, and a frozen Rust build plus a golden corpus take over as the
  oracle. See §14.1.)*
- **The Rust core lands in Phase 2, before any agent work**, so every later phase iterates on a fast
  engine (see §8.1 for what this buys and what it costs).
- Engine supports 2–5 players (incl. 2–3P double-route blocking); training/eval targets 2P first.
- Ladder runs through search + learning: heuristics → PPO self-play → Information-Set MCTS with learned
  policy/value priors.
- Rich terminal client now; browser SVG map later. *(Superseded by Revision 2: a web client now, and
  no terminal client. See §14.2.)*

**Hardware:** MacBook Pro M2 Max, 12 cores (8P/4E), 32 GB unified memory, **no CUDA** — PyTorch MPS only.
Everything below is sized for one machine. This is a weeks-of-overnight-runs project, not a cluster project.

### Step 0 — data normalization the user asked for

Plan mode blocks file edits, so this is the **first commit** of Phase 0:

1. **`pink` → `purple` everywhere in RULES.md** (28 occurrences across the map table). The rules prose
   already calls this color "Purple"; only the map table says `pink`. `PURPLE` becomes the canonical enum.
2. **`Montréal` → `Montreal`** in the Tickets table (3 tickets: Montréal–Atlanta 9, Montréal–New Orleans 13,
   Vancouver–Montréal 20). The map table already spells it `Montreal`.

After this, RULES.md is internally consistent and the ingest parser needs no alias table — but keep a
defensive alias map anyway (`"pink"→PURPLE`, `"Montréal"→Montreal`, `"Sault Ste. Marie"→"Sault St. Marie"`)
so external data sources import cleanly. **An unknown city name must be a hard error**, never silent
normalization — that's how you end up with 37 cities.

### Data validation already done

I parsed RULES.md and verified it is fully self-consistent. These become the board-data assertions:

| Check | Result |
| --- | --- |
| Cities / unique pairs / segments / spaces | 36 / 78 / 100 / 309 |
| Segment colors | 44 gray + **exactly 7 of each of the 8 colors** |
| Segment lengths | 1:9, 2:36, 3:20, 4:16, 5:10, 6:9 |
| Double-route pairs | 22 → 78 segments / 256 spaces playable in 2–3P |
| Total route points if all 100 claimed | 508 |
| Tickets | 30, values 4–22, sum 349, all endpoints resolve |
| City-index degree table vs edge list | agrees; Σdeg = 156 = 2×78 |
| Graph connectivity | connected; max degree 7 (Denver, Helena, Pittsburgh) |
| Train deck | 8×12 + 14 loco = 110 |
| **Max segments one player can own** | **25** (45 trains, greedy over shortest pairs) |

**Parser gotcha:** double routes are written with markdown-escaped pipes (`gray \|\| gray`). Un-escape `\|`
before splitting cells or you silently get 78 segments / 256 spaces instead of 100 / 309. I hit this.

---

## 1. Prior art — what to take from the one published attempt

Yang et al., *"Reinforcement Learning Agents Playing Ticket to Ride — A Complex Imperfect Information Board
Game With Delayed Rewards"*, IEEE Access 11:60737–60757, 2023 ([doi](https://doi.org/10.1109/ACCESS.2023.3287100)),
UNSW Canberra + DSTG. PPO via Stable-Baselines3, 64×64 MLP, 1M timesteps, 8 agents differing only in
training opponent. No code released.

The user's skepticism is justified, and more so than expected. Their map is 8 cities / 16 edges / **4 colors**
/ max edge length **3** / **zero double routes** / 10 trains — their own Appendix survey of 13 official
versions shows every one uses 6–8 colors and max length 4–8, so the scale-down removes color scarcity, the
high-value end of the scoring curve, and blocking. Worse: their heuristic baselines are lifted unmodified
from `github.com/willdzeng/ticket_to_ride`, whose ticket-drawing thresholds are hard-coded at 15 trains — on
a 10-train map **those conditions can never fire, so every baseline plays its opening tickets and never draws
more**. That's the exact strategic dimension the paper claims to test. Their round-robin matrix is also
mirrored rather than measured (every off-diagonal pair sums to exactly 1.00), so first-player advantage is
confounded into every cell, and there are no CIs and one training seed.

**Genuinely worth stealing:**

1. **Ticket reward redistribution** — charge `−points` the moment a ticket is *kept*, pay `+2×points` on
   completion. Net is identical to the real game (+p completed, −p failed) but converts a terminal signal
   into two mid-episode ones and makes drawing tickets a deliberate, locally-costly act. Best idea in the
   paper. Ablate against potential-based shaping.
2. **The "threatened edge" primitive** — not "a route they might want," but *an unclaimed edge that would
   merge two currently-disconnected components of the opponent's network*, weighted by the train-length of
   the components it would join. Pure public information, no search, no hidden-state inference. This is a
   sharper notion of blocking than I had, and it goes straight into H4.
3. **Threat pricing** — drop double routes from the blockable set (you can't close a pair in one action);
   penalize by `(n_threats − 1) × k` because many simultaneous threats mean none is decisive; scale by game
   progress (blocking matters late); super-linear in how much network the block severs.
4. **Convex path cost** — square the number of *still-needed* cards for colored edges, linear for gray.
   Encodes "accumulating 3 of one color is much worse than 1 each of 3 colors" for free.
5. **Gray-route cannibalization penalty** — when paying a gray route with colored cards, subtract
   `cards_needed[c]` per card, so you don't burn the color you need elsewhere.
6. **Plan-urgency term** — `path.score − remaining_edge_score`, which automatically escalates the value of
   the last edges of a plan as it nears completion.
7. **Their design-principle survey** (mean degree 4 in every official version; #tickets ≈ #cities; ticket
   points = shortest-path segment count) is the best reference for designing our own TTR-mini fixture.

**Their strongest empirical result, which validates a choice we'd already made:** the self-play agent had the
*lowest* average score (12.8) yet beat every heuristic-trained agent head-to-head (0.59–0.70). Their reward
was pure score, so heuristic-trained agents learned to maximize points while the self-play agent implicitly
learned to win. **Optimize win probability, not score.**

**Ignore:** their win rates as evidence of strength (their best heuristic beats *random* only 91% of the
time — a competent player beats random ~100%), the claim that PPO handles imperfect information (there is no
belief model, no memory, no inference of any kind), and 64×64 as an adequate architecture.

**Two process lessons this hands us for free:** (a) heuristic constants must be config-driven and re-tuned
per variant, with a test asserting each heuristic actually exercises every action type on every map;
(b) never report a mirrored win-rate matrix — measure both seatings.

---

## 2. Repo layout

```
ticket_to_ride/
  data/       maps/usa.toml · board_gen.py (generated) · board.py (derived tables) · coords.py (lat/lon)
  engine/     board · cards · actions · state · rules · scoring · graph · rng · replay   # NO torch, NO numpy
  agents/     base · registry · human · heuristic/{h1..h4} · search/{ismcts,determinize,particles}
  rl/         encode/{observation,action_space,versions} · nets · vecenv · buffers · ppo · alphazero · league
  eval/       pairing · runner · arena · ratings · sprt · metrics · store
  ui/         render/{projection,palette} · terminal/ · web/
  config/     schema · load · seeding
  tracking/   writer · backends/{jsonl,tensorboard} · manifest
  cli/        app + cmd_*
tools/        gen_board.py          # parses RULES.md -> board_gen.py AND board_gen.rs
crates/       ttr-core (pure Rust) · ttr-py (PyO3 shim)          # Phase 4
configs/ · docs/ · tests/{unit,property,integration,bench,golden} · runs/ · artifacts/
```

Four boundaries that carry weight:

1. **`engine/` imports without torch or numpy and holds zero global mutable state.** Ten self-play workers
   each importing torch costs ~1.5 s and ~300 MB RSS; a torch-free engine starts in ~50 ms. It also lets CI
   run the whole unit+property suite with no torch and no CUDA download. Enforce mechanically: torch is an
   optional extra, ruff `flake8-tidy-imports` bans module-level `torch`/`numpy` under `engine/**`, and a test
   imports the engine in a subprocess with a torch import hook that raises.
2. **`rl/encode/` is the only module that knows both game objects and tensor shapes.** It carries
   `OBS_VERSION` and `ACTION_SPACE_VERSION`, baked into every checkpoint; loading a mismatched checkpoint is
   a hard error. Over a months-long league this is the difference between "old checkpoints fail loudly" and
   "old checkpoints silently play garbage and corrupt your Elo table."
3. **`eval/` does not import `rl/`.** Neural agents come through `agents/registry.py`, which lazy-imports
   torch only for specs like `ppo:runs/x/best.pt`. The arena is then a general tournament tool.
4. **`ui/render/projection.py` is shared by terminal and web**, so both derive from one lat/lon table.
   *(Revision 2: there is one client, the web one, and the projection moves into the layout
   generator. §14.2.)*

Delete `main.py` once `ttr` exists.

## 3. Dependencies & Python version

**Python 3.14 verified safe.** torch ships cp314 macOS-arm64 wheels from 2.9.1 through 2.13.0; numpy 2.5.1
likewise. Three real traps found:

- **torch 2.12+ dropped the `macosx_11_0_arm64` tag** (wheels are now `macosx_14_0_arm64`). uv's
  `aarch64-apple-darwin` platform alias implies a macOS 11 target and **silently resolves torch 2.11**. Pin
  `torch>=2.13` so a downgrade becomes a hard resolution error, and constrain `[tool.uv] environments`.
- **torch on Linux pulls 18 CUDA packages (~3 GB)** — catastrophic for CI. Route Linux through the
  `pytorch-cpu` index via `[tool.uv.sources]`.
- **`aim` is unusable on cp314** (drags in tensorboard 2.3.0 from 2020). Ruled out.

Set `requires-python = ">=3.14,<3.15"` — torch 2.13's own metadata gates deps on `python_version < "3.15"`,
and `uv python pin 3.15` is one keystroke away.

```bash
uv add "numpy>=2.5" "msgspec>=0.21" "typer>=0.27" "rich>=15"
uv add --optional rl "torch>=2.13" "tensorboard>=2.21"
uv add --optional web "fastapi" "uvicorn"
uv add --dev pytest pytest-xdist pytest-cov pytest-benchmark pytest-timeout hypothesis syrupy ruff ty pre-commit polars
uv add --dev maturin        # Phase 4
```

`msgspec` earns its place three times over: typed TOML config decode, fast binary replay serialization, and
frozen Structs light enough to sit in the engine hot path — replacing pydantic + orjson + a config library.

**Type checking is `ty`, not mypy** (user's call, and ~35× faster in practice — 0.04 s vs ~1.5 s, which
matters for a tight edit loop). One real gap to compensate for: ty is inference-based and has no
`disallow_untyped_defs` equivalent, so an unannotated function would type check clean — exactly what
`mypy --strict` existed to prevent. Ruff's **`ANN`** ruleset covers it (`ANN001` missing arg annotation,
`ANN201`/`ANN202` missing return type), with `ANN401` ignored since `Any` is occasionally correct. The split
is: **ty answers "are these types consistent?", ANN answers "are these functions annotated at all?"** Both
are needed; either alone leaves a hole. `ty check` runs with `error-on-warning = true`, or warnings exit 0
and the gate goes green while problems accumulate.

**Deliberately rejected:** networkx (our two graph needs — union-find and longest-trail — are ~15 and ~25
lines and run millions of times; networkx has no longest-trail function at all), hydra (TOML + one `extends`
key + `--set a.b.c=v` in ~120 lines we own), scipy (40 MB for a Wilson interval and a convex MLE),
gymnasium as the primary API (poor fit for turn-based multi-agent with masks — keep it as a thin adapter),
trueskill (unmaintained since 2018), and **pytest-randomly** (it reseeds global `random` per test, which
fights the seeding-discipline tests and makes determinism failures look flaky).

## 4. Board data

One generator, `tools/gen_board.py`, parses RULES.md and emits **both** `data/board_gen.py` and
`crates/ttr-core/src/board_gen.rs`, both checked in. A test asserts regeneration is byte-identical. This
makes the Python and Rust engines structurally incapable of disagreeing on board data — a JSON file loaded
by two independent parsers would reintroduce exactly that risk. Checked-in generated Python also means zero
import-time parse cost across 12 worker processes, and reviewable git diffs on the actual constants.

Precompute at import: adjacency, segment↔pair index, `SIBLING[]` for double routes, the 33 non-empty
`(length, color)` buckets used by the claim mask, the 36×36 all-pairs distance table, and `DATA_HASH`
(blake2b-128 over canonicalized data). `DATA_HASH` goes into every replay so a board edit makes old replays
fail loudly instead of silently replaying a different game.

Validation asserts every row of the table in Context above, plus: double-route siblings are mutual and equal
in length and endpoints, every ticket endpoint is a known city, every ticket pair is reachable, and the
derived degree table matches the independent City Index table in RULES.md.

## 5. Engine design

### 5.1 State representation

Flat POD arrays — ~545 bytes in Rust, a memcpy to clone (~15–30 ns); measured 0.53 µs in pure CPython.

| Field | Type | Notes |
| --- | --- | --- |
| `seg_owner` | `u8[100]` | 255 = free |
| `hand` | `u8[5][9]` | counts, index 8 = locomotive |
| `trains`, `score` | `u8[5]`, `i16[5]` | |
| `tickets` | `u32[5]` | 30-bit mask |
| `deck` | `u8[110]` + cursor | **pre-materialized permutation** (see below) |
| `discard` | `u8[9]` | counts — order is never observable |
| `faceup` | `i8[5]` | −1 = empty slot |
| `tdeck` | `u8[30]` ring + head/len | **ordered** — returned tickets go to the bottom (observable) |
| `dsu` | `u8[5][36]` | per-player union-find over cities |
| `certain`, `unknown` | `u8[5][9]`, `u8[5]` | public knowledge of each hidden hand (see 5.5) |
| `phase`, `cur`, `draws_left`, `final_left`, `pass_streak`, `turn` | small ints | |
| `rng` | `u64[2]` | PCG-XSH-RR 64/32 |

**Deck as a pre-materialized permutation, with a derived counts view.** The two design reviews disagreed
here and the permutation wins, for one reason: **paired evaluation**. If the deck is sampled lazily from
counts + RNG, then agent A drawing 3 cards and agent B drawing 5 causes the two mirrored games to realize
*different shuffles*, and the entire variance-reduction scheme silently evaporates. That is the single most
common bug in game-eval harnesses. Materializing the permutation at construction makes environment
randomness a pure function of the seed, independent of agent behavior. The cost is 110 bytes on a 545-byte
state — nothing. Keep a `deck_counts()` view for determinization sampling and for exact chance
probabilities in `EXPLICIT` mode. Reshuffles re-permute the discard from a seed-derived stream.

**`State: Copy`, clone-based, no undo log.** At this size a clone beats maintaining an undo journal for a
claim (which touches `seg_owner`, `hand`, `discard`, `trains`, `score`, and the DSU). It also lets the DSU
use **path compression** freely; an undo log would force rollback-DSU with no compression. MCTS at 800
sims/move stores ≤ 800 × 545 B = 436 KB. Provide `clone_into(&mut dst)` for arena reuse so search never
allocates.

### 5.2 Rules edge cases — the correctness checklist

Every item gets a named test. Grouped; ~40 total.

**Setup** — deal 4 cards before flipping 5 face-up · run the 3-loco flush check after setup and after every
refill · initial keep ≥2 of 3 (exactly 4 legal subsets) · returned tickets go to the bottom in a *defined*
order (seat, then offer index) so replays are exact · initial ticket choice is physically simultaneous;
sequentializing in seat order is information-equivalent (keeps are secret) — the one documented deviation.

**Drawing** — taking a face-up **locomotive as the first card ends the draw** · a face-up locomotive may
**never** be taken as the second card · a locomotive drawn **blind** does not end the turn · refill the taken
slot **before** the second draw · the refill may itself be a locomotive (falls out of the rule above) ·
**3+ locomotives face-up → discard all five, deal five new, then re-check (cascade)** · deck exhausted
mid-refill → reshuffle discards, continue · deck **and** discard empty → the display legitimately holds <5,
empty slots untakeable · everything empty → drawing is entirely illegal · if no second card is obtainable
the turn ends after one card (never construct a zero-legal-action node) · reshuffle **lazily at draw time**,
not eagerly on `deck==0`, or you can't distinguish the last two cases.

> ⚠️ **The 3-loco flush needs a termination guard.** With 14 locomotives, an all-locomotive available pool
> loops forever. Condition the flush on `nonloco(deck) + nonloco(discard) >= 3`, plus a hard iteration cap
> with a deterministic bail-out. This is the most common hang bug in TTR implementations, and the assertion
> that catches it is the one most engines lack.

**Claiming** — exactly `len` cards, one color + locomotives, no partial claims or overpaying · gray takes any
single color, or all locomotives · requires `trains >= len` · segment unclaimed · **2–3P: once either track
of a double pair is claimed by anyone, the sibling is closed to everyone** · **4–5P: the sibling stays open
to others but never to the same player** · cards paid go to the discard and are **public** · scoring
{1:1, 2:2, 3:4, 4:7, 5:10, 6:15} · one route per turn · no adjacency requirement.

**Tickets** — draw `min(3, remaining)`, keep ≥1 · empty ticket deck → the action is illegal · exactly 1 left
→ auto-resolve the forced keep (don't burn a network eval) · returns go to the bottom · counts are public,
contents secret · no hand limit.

**End & scoring** — trigger fires at *end of turn* when `trains <= 2`, once only, after **every** turn type
(a player at 2 trains who spends the turn drawing still triggers) · exactly `n_players` more turns follow ·
0 trains is legal · tickets settle ±value on own-network connectivity · longest path is a **trail** measured
in **train cars**, loops allowed, each segment used once, **all tied players score +10** · tiebreak
points → most completed tickets → longest-path card, and **a true draw is still possible**.

**Degenerate** — a player can have no legal action (all pools empty, no affordable open route) → explicit
`PASS` · **if all players pass consecutively the state is frozen — terminate and score**, or the engine
hangs · a global turn cap (1000) as a belt-and-braces assert; the analytical 4P bound is ~335 turns.

### 5.3 Action space — 915, flat and maskable

| Range | Size | Meaning |
| --- | --- | --- |
| `0..899` | 900 | `CLAIM`: `segment*9 + pay`; pay 0–7 = color, 8 = pay entirely with locomotives |
| `900..905` | 6 | `DRAW`: face-up slot 0–4, or 5 = blind |
| `906` | 1 | `DRAW_TICKETS` |
| `907..913` | 7 | `KEEP`: `bitmask − 1` over the offer, mask ∈ 1..7 |
| `914` | 1 | `PASS` |

**Locomotive payment is canonical, and there's a clean dominance argument for it.** For a route of length
`L` in color `c`, let `k = min(hand[c], L)`. Paying `k` colored + `L−k` locomotives weakly dominates paying
`j < k` colored: the two resulting hands differ only by trading `k−j` cards of color `c` for `k−j`
locomotives, and a locomotive substitutes for `c` in every legal claim but not conversely. The usual
objection — "hoarding locomotives is strategically costly" — **doesn't apply to base TTR, which has no hand
limit**; the only residual cost is information leakage, which is second-order. So the naive 100×9×7 = 6300
space collapses to 900 with no meaningful loss. Keep `WILD_POLICY = canonical | explicit` to ablate later.

**The `hand[c] >= 1` guard matters.** Without it, a hand of pure locomotives makes all 8 gray color slots
legal *and identical*, which poisons the policy distribution. With it, distinct action ids map to distinct
payments, exactly. Legality for pay-slot `c`: `hand[c] >= 1 and hand[c] + hand[LOCO] >= L`; for slot 8:
`hand[LOCO] >= L`.

**The two card draws are two sequential decisions, not one 36-way joint action.** The second choice
genuinely depends on the refill flipped after the first, on a flush possibly firing between them, and on
locomotive-first terminating the action. Same 6 ids, different mask, weights shared.

**Claim masking is bucketed, not a 900-way scan.** Precompute `reach[c] = hand[c] + hand[LOCO]`, then iterate
the 33 non-empty `(length, color)` buckets. Measured **14.4 µs** in CPython for a contrived 421-legal-action
worst case; 4–8 µs realistic.

Consider an **optional reduced 115-action mode** for early milestones (100 claim actions with a rule-based
payment resolver, + 15). It cuts the MCTS branching factor ~10× at near-zero strength cost. Unlock the full
915 as a measured ablation once an agent is strong.

### 5.4 Phases

`INITIAL_TICKETS → MAIN → {DRAW_SECOND | TICKET_KEEP} → end_turn → …`, plus `CHANCE` (explicit mode only)
and `TERMINAL`. `end_turn` handles the ≤2-train trigger, the `final_left` countdown, and the all-pass freeze.

**Chance is `SAMPLED` by default** — the engine resolves flips and reshuffles internally, so PPO/MCTS never
see chance nodes. An `EXPLICIT` mode exists for CFR-style solvers: because the unseen deck reduces to
counts, `chance_outcomes()` is exact and tiny (≤9 outcomes for a card, ≤30 for a ticket). **Max chance
branching anywhere in the game is 30** — which is what makes an OpenSpiel-shaped adapter feasible at all.
But for MCTS the primitive that actually matters is not chance nodes, it's `resample_from_infoset`.

### 5.5 Information-set bookkeeping — get this right or ISMCTS is garbage

Far less is hidden in TTR than people assume: face-up takes are public, and **cards spent on a claim are
publicly discarded**. Maintain the invariant `hand[p] == certain[p] + unknown[p] blind-drawn cards`:

| Event | Update |
| --- | --- |
| initial deal of 4 | `unknown[p] += 4` |
| take face-up card of color `c` | `certain[p][c] += 1` |
| blind draw | `unknown[p] += 1` |
| pay one card of color `c` | `certain[p][c] -= 1` if positive, else `unknown[p] -= 1` |

Then determinization is: unseen pool = 110 − (my hand) − (face-up) − (discard) − (each opponent's `certain`);
deal `unknown[q]` cards to each opponent by multivariate hypergeometric; the remainder *is* the deck.
Two documented approximations (state them in a docstring so nobody "fixes" them): uniform resampling ignores
correlation with reshuffle boundaries, and the bottom-return ordering of the ticket deck isn't reconstructed.

**Ship `assert_consistent(sample, public_state)`** and run it on every sample in debug builds — determinization
bugs are silent and devastating. It checks per-type conservation to `{12×8, 14}`, each opponent's hand size,
`opp_hand ⊇ certain`, ticket disjointness and counts, tickets *you saw and returned* being in the deck (real
information most implementations forget), and exact reproduction of all public state.

### 5.6 Incremental computation

- **Ticket connectivity:** per-player DSU over 36 cities (180 bytes), path-halving, union on claim. Depth ≤25
  worst case, typically ≤3.
- **Remaining ticket cost:** Dijkstra where my segments cost 0, free segments cost their length, opponent
  segments cost ∞. Yields "trains still needed"; ∞ means dead. Cached per (player, board-version).
- **Steiner tree over held tickets' terminal cities** (Dreyfus–Wagner, ≤8 terminals on 36 nodes, <1 ms). The
  per-ticket Dijkstras double-count shared trunk lines; the Steiner cost is the truth, and it's what H2+ plan
  against.
- **Longest trail** (game end only): NP-hard in general, but bounded here — **≤25 segments**, ≤26 vertices,
  max degree 7. Four layers, in order: (1) split into connected components via DSU; (2) **Eulerian shortcut** —
  a component with 0 or 2 odd-degree vertices admits a trail using *every* edge, so the answer is `Σw` with
  zero search, and this fires on a large fraction of real subgraphs; (3) memoized DFS on
  `(vertex, used_edge_bitmask)` with the mask in a `u32`; (4) early exit when `best == reachable_weight`.
  Measured over 300 plausible player subgraphs: **501 DFS nodes / 173 µs** average in pure Python. A
  hill-climbed adversarial subgraph (22 edges, dense Nashville–Pittsburgh–Raleigh cluster) hits 126 ms naive
  → 23 ms memoized; layers 1–2 kill that tail. Runs once per player at terminal. Oracle: brute-force all
  permutations on ≤10-edge components.

### 5.7 API and replay

```python
class Game:   new_initial_state(seed) · num_distinct_actions (915) · action_to_string
class State:  legal_actions() · legal_action_mask(out) · step(a) · clone() · clone_into(dst)
              is_terminal() · current_player() · observation(player, out) · returns() · final_scores()
              score_breakdown() · resample_from_infoset(observer, rng) · chance_outcomes()
              state_hash() · position_hash() · validate() · history()
```

`position_hash()` (excludes RNG) is the MCTS transposition key; `state_hash()` (includes it) is the
differential-testing key. Adapters for Gymnasium / OpenSpiel-shaped APIs live *outside* the engine package.

**Replay = `(schema, DATA_HASH, rules_hash, n_players, seed, u16 actions, final_hash, final_scores)`** —
~500 B/game binary, ~150 MB per million games with zstd. `replay()` asserts `DATA_HASH` matches and that the
final state hash reproduces.

### 5.8 Freeze the contract before any Rust exists

**A Phase 1 deliverable.** Differential testing is impossible unless three things are pinned bit-for-bit in a
shared doc with test vectors, **before** the Rust port starts — and since the port is now Phase 2, this is
the gate between them:

1. **The PRNG** — PCG-XSH-RR 64/32, specified exactly.
2. **The draw procedure** — the exact linear-scan / permutation-cursor semantics. Any other equally-valid
   sampling method produces different (still correct) trajectories, which destroys the oracle.
3. **`state_hash()`** — FNV-1a-64 or xxh3 over a canonical byte serialization with fixed field order and no
   padding.

## 6. Observations & network

### 6.1 Features (~2,000–2,800 dims, perspective-normalized)

Encoded from the acting player's POV, opponents ordered by **distance in turn order** (next-to-act = slot 0).
This is seat-*relative*, not fully permutation-invariant — correct, because "who moves before me" is decisive
in TTR. It gives a free N× data multiplier and lets one net play any seat.

| Block | Contents |
| --- | --- |
| Board, 100×~18 | owner one-hot {free, me, opp0–3}, `twin_locked`, `i_can_afford_now`, `cards_short` one-hot, `on_my_steiner_tree`, `extends_my_longest_chain` |
| Own hand | counts + per-color thermometer over {0,1,2,3,4,5,6+} |
| Own tickets, 30×~12 | one row per ticket in the **fixed 30-ticket deck** (so the net can memorize that LA–NY 21 and Denver–El Paso 4 are different objectives): held, connected, **`remaining_cost`** + thermometer, `is_dead`, points, **`fragility`** = worst-case cost increase if one enemy claim lands on the best path |
| Steiner globals | `steiner_remaining_cost`, `steiner_cost − trains_left` |
| Face-up | 5 slots × one-hot 10 (slot order matters — actions index it) |
| Piles | deck/discard/ticket-deck sizes, **discard composition (9)**, **`unseen[c]` (9)** |
| Per opponent ×4 | trains, score, hand size, ticket count, **`certain[c]` (9)**, `max_possible[c]` (9), n_blind_draws, longest chain, segments claimed |
| Phase/clock | phase one-hot, `draws_left`, final-turn countdown, turn index, score rank, gap to leader |

**`remaining_cost` is the single most important engineered feature in the project.** Static segment
properties (length, color, endpoints) go into a learned per-segment embedding rather than the vector.

### 6.2 Architecture — entity-encoder MLP with DeepSets pooling, **not** a GNN

```
segment tower  [18 dyn ‖ 64-d segment-ID emb ‖ 9-d color one-hot] → 128 → 128   (shared ×100)
               → keep per-segment vectors (pointer head) + mean‖max pool
ticket tower   [12 feats ‖ 32-d ticket-ID emb] → 64                (shared ×30) → masked mean‖max
opponent tower [~93 feats] → 128 → 128                             (shared ×4)
               → DeepSets sum‖max (order-invariant) + seat-relative concat (order-sensitive)
global tower   [hand ‖ trains ‖ faceup ‖ piles ‖ phase] → 256
trunk          concat → 1024 → 3× residual(1024, LayerNorm, GELU)
heads          policy (flat 915, masked) | V_win | V_margin | aux heads
```

**Why not a GNN:** the map is *fixed*, so node-permutation equivariance isn't a useful prior — it actively
fights the memorization you want (Denver is a degree-7 hub; the Chicago–St. Louis corridor is contested).
And the property a GNN would supposedly buy — connectivity reasoning — is what message passing does *badly*:
hop-diameter ~10 would need ~10 layers and over-squash through the hubs. **Compute connectivity exactly in
the environment** (Dijkstra/DSU, microseconds) and hand the net the answer. Strictly better than hoping six
layers rediscover Dijkstra. A route-token transformer is the Phase-6 upgrade, not the start.

**Color-permutation symmetry is the cheapest sample-efficiency win available.** The 8 non-locomotive colors
can be simultaneously permuted across hand, face-up, discard, deck, and per-segment required color to give an
isomorphic game — an 8! = 40,320× augmentation. Exploit it as a consistency regularizer: per batch sample a
permutation σ and add `λ·KL(π(s) ‖ π(σ·s)) + (V(s) − V(σ·s))²`, λ≈0.1. This requires segment color to be an
*input feature* rather than baked into the ID embedding — hence the 9-d color one-hot above.

**Masking correctness, since it silently ruins PPO:** apply `logits.masked_fill(~mask, -1e9)` *before* the
softmax, and **recompute the mask at update time** from the stored state so old and new log-probs share
support. Log `policy/mask_violations` and assert it is exactly 0.

### 6.3 Reward

**Terminal only, optimizing win probability** — validated by the paper's own result that its score-optimizing
agents lost head-to-head to its win-optimizing self-play agent.

- **2P:** `z ∈ {+1, 0, −1}` with full rulebook tiebreaks.
- **3–5P:** `0.75 · win_only + 0.25 · rank_norm`. Both terms are constant-sum, so self-play stays
  well-behaved and MCTS value backup stays valid; the rank term keeps a losing agent's gradient informative.
- **γ = 1.0.** Every turn is equally valuable in a finite game; γ=0.99 over 60 turns discounts the terminal by
  0.55, systematically undervaluing endgame ticket completion — exactly what decides games. If early PPO is
  unstable, start at 0.997 and anneal.

**Auxiliary heads, not shaped reward** — highest value-per-line in the project, and zero influence on the
policy objective: `V_margin` (`tanh(margin/40)`, MSE, w 0.3 — a dense low-variance target that pulls a useful
representation out of a 60-turn sparse game far faster than `V_win` alone), per-ticket `P(complete)` (0.3),
`P(longest path)` (0.1), **opponent ticket belief, 30 logits/opponent** (0.3), opponent hand composition
(0.1), turns remaining (0.05). Ground truth for all of these is free in self-play.

**Two shaping schemes to ablate, not stack:**

- **(A) The paper's ticket redistribution** — `−points` on keeping a ticket, `+2×points` on completing it.
  Preserves the true episode return exactly and is trivially implementable.
- **(B) Potential-based** — `F = γΦ(s′) − Φ(s)`, `Φ = 0.3·tanh(frozen_margin/25)`, γ=1. **The episodic case
  requires `Φ(terminal) ≡ 0`.** If you set Φ = "score if the game froze here" and don't zero the terminal,
  the shaping telescopes to `+final_score` and you have silently switched to optimizing `win + λ·score`.
  Say this in the code comment; people get it wrong constantly.

**Never use raw route points as the reward** — it directly incentivizes long-route greed over ticket
completion and is the #1 cause of the "agent never draws tickets" failure.

## 7. Agent ladder

| Tier | Agent | Notes |
| --- | --- | --- |
| H0 | Random legal | anchor |
| H1 | Greedy points: highest-point affordable route, else draw the face-up card most represented in hand | |
| H2 | Ticket-path: exact Steiner tree over held tickets, claim the next needed segment preferring the most contested, draw toward the cheapest missing color | |
| **H3** | H2 + draw-value model (expected reduction in turns-to-complete) + endgame mode (max points/train under 8 trains) + longest-path tiebreaks + ticket-keep EV filter (`marginal_steiner_cost ≤ points`) + the paper's convex path cost and gray-cannibalization penalty | **Elo anchor = 0, permanently** |
| **H4** | H3 + opponent modeling: the **threatened-edge** primitive (an unclaimed edge merging two components of the opponent's network, weighted by the network length it would join), the threat-pricing rules from §1, early grabs on contested double-route halves in 2–3P, hoard-and-strike | **the bar that matters** |
| S1 | Flat Monte Carlo: K determinized rollouts per legal action with H3 as rollout policy | plumbing sanity check |
| S2 | **SO-ISMCTS** with H3 rollouts, 800–2000 iters/move | strong with zero NNs |
| R1 | **PPO self-play** + prioritized league | target: beat H4 ≥65% |
| R2 | **ISMCTS + learned prior/value + belief-driven determinization** | the destination |

A learned agent that doesn't beat **H4** at ≥65% paired win rate is not done. And per §1, every heuristic's
constants live in config, with a test asserting each heuristic actually exercises every action type
(including drawing extra tickets) on every map — that's the exact bug that invalidated the paper's baselines.

**H1–H4 are implemented in Rust** (§8.3). H3 doubles as the ISMCTS rollout policy, and a rollout that crosses
the FFI boundary per step defeats the batching design — so writing them in Rust from the start means writing
them once. Python-side heuristics remain possible via the `Agent` protocol for quick experiments.

### 7.1 PPO

Masked categorical, GAE λ=0.95, γ=1.0, clip 0.2, entropy 0.02→0.003 annealed, 2 epochs (>3 punishes
non-stationarity), lr 3e-4→5e-5 cosine, value targets already in [−1,1] so no PopArt. n_envs and batch size
scale with whatever the Rust env delivers.

**League — prioritized fictitious self-play.** Per game sample the opponent: **50% current policy, 35% from
the last-30 checkpoints + hall-of-fame with PFSP weighting `∝ (1 − winrate_vs_them)²`, 15% scripted
(H1–H4 + ISMCTS)**. Keeping the scripted agents in *permanently* is the cheapest anti-overfit insurance
available and directly guards against never learning to block. In 3–5P, **never fill all seats with the
identical current policy** — mix pool members across seats.

### 7.2 ISMCTS

Per decision: K determinizations (K≈30) drawn from the belief-weighted sampler, PUCT guided by the network
prior, leaf value from `V_win` instead of a rollout, visit counts aggregated. c_puct 1.5, Dirichlet(0.3, 0.25)
at root **after** masking, temperature 1.0 for 12 plies then 0.15, playout-cap randomization (full 400 sims on
25% of moves for policy targets, 100 on the rest → ~2.5× more games).

**Batched MCTS is make-or-break.** Naive one-eval-per-leaf on MPS is ~1–2 ms per call (dispatch-dominated),
so 400 sims × 120 moves = 48k evals → **60 s/game. Unacceptable.** Run 256–1024 games concurrently, collect
8 leaves per tree per round with virtual loss 3, and batch 2,048–8,192 leaves into one forward:

```
(obs_batch, leaf_ids) = rust_mcts.collect_leaves(n_per_tree=8)   # 1 FFI call
(policy, value)       = net(obs_batch)                            # 1 MPS forward
rust_mcts.backup(leaf_ids, policy, value)                         # 1 FFI call
```

Tree, PUCT, virtual loss, and determinization all live in Rust. This is the Lc0/KataGo pattern.

**Bootstrap:** pretrain the policy head supervised on ~200k S2 (ISMCTS-2000) positions before starting the
self-play loop. A few hours of S2 compute removes the cold-start entirely.

**Known limitations to measure, not ignore.** *Strategy fusion* is mild-to-moderate here — your own hand is
known to you, and SO-ISMCTS eliminates fusion at your own nodes; the residual case is blocking races.
*Non-locality* is weak for hands (card counting is nearly exact) but **strong and real for tickets** — a
player who claimed Denver–Salt Lake almost certainly holds a western ticket, and uniform determinization
throws that away. That's what the learned belief head fixes.

### 7.3 Belief modeling

Two components, both needed:

- **Particle filter over ticket assignments** (M≈200 particles, each a consistent assignment of tickets to
  players and deck). After each opponent claim, reweight `w *= exp(β · 1[claimed segment lies on the min-cost
  Steiner tree of that particle's ticket set])`, β≈1.5; systematic resampling when ESS < M/2. ~50 lines, and
  it doubles as H4's opponent model.
- **Learned belief head** (30 logits/opponent, BCE against free self-play ground truth). Two payoffs: it
  forces the trunk to encode opponent intent even if you never read it, and at search time its marginals
  **reweight the particle filter's proposal** subject to disjointness and count constraints. That's the
  tractable "ReBeL-lite" — it captures the non-locality pure determinization discards at a tiny fraction of
  the cost of real re-solving.

Also add a learned opponent-hand head even though the analytic tracker is near-exact — **if the network beats
your tracker, your tracker has a bug.**

### 7.4 What is *not* worth attempting — be explicit, it saves a month

- **ReBeL / DeepStack re-solving: not tractable.** These need a public belief state and a CFR-solvable
  subgame. A TTR private state is `(multiset of ~0–25 cards from 9 types) × (subset of 30 tickets)` ≈ 10¹⁴
  per player, and the game never resets into natural subgames. Poker works because a private hand is 2 of 52.
  Ten-plus orders of magnitude out.
- **Deep CFR / MCCFR on the full game: not tractable.** Only justified use is on **TTR-mini**, to get one
  honest absolute exploitability number and calibrate how far self-play sits from Nash.
- **MuZero:** card draws break deterministic dynamics, and we already have a perfect fast simulator.
- **R-NaD** is the one stretch worth attempting *after* R2 works, 2P only: model-free, no belief state, no
  search, handles imperfect info natively, and it's a ~200-line modification to an actor-critic loop. Honest
  caveat: DeepNash used a TPU cluster and ~5.5×10¹⁰ steps; 10⁹ steps here is 1–3 weeks of continuous running,
  and my expectation is it lands *comparable* to R2 at equal wall-clock but with meaningfully lower
  exploitability — which is what you want against unpredictable humans.

## 8. Rust port — Phase 2, before agents

`crates/ttr-core` (pure Rust, standalone-tested) + `crates/ttr-py` (thin PyO3 shim), built with maturin into
the existing `.venv`.

### 8.1 Why this ordering, and what it costs

**What it buys.** Every later phase runs on a fast engine: a 10k-game arena drops from ~2 minutes to a few
seconds, so heuristic tuning becomes an interactive loop rather than a batch job, and abandoning an approach
that isn't working costs minutes instead of hours. That was the user's reasoning and it's sound. There's a
second benefit that makes the ordering strictly better rather than merely earlier: **H3 is the ISMCTS rollout
policy, and batched search requires it inside Rust** (a rollout that crosses the FFI boundary per step defeats
the entire batching design). Building agents *after* the port means writing H1–H4 once, in Rust, instead of
writing them in Python and porting them in Phase 5.

**What it costs.** The engine API gets frozen before real consumers — agents, search, RL — have exercised it,
so there's a genuine risk of a second port pass. Three mitigations, all cheap:

1. **Phase 1 ships throwaway H0/H1 agents and a flat-Monte-Carlo stub against the *Python* engine.** Not
   deliverables — API exercise. They put `clone()`, `legal_actions()`, and `step()` under search-like load and
   surface interface gaps while changes are still free.
2. **Build the Rust API with the search hooks already in it.** We know precisely what ISMCTS needs from §5.7
   and §7.2 — `clone_into`, `resample_from_infoset`, `position_hash`, and `collect_leaves`/`backup` — so they
   go in from day one rather than being discovered in Phase 5.
3. **Budget one planned API revision** at the start of Phase 5, when search is the first heavyweight consumer.
   Planning for it beats pretending it won't happen.

### 8.2 Observation encoding — declarative, so churn stays cheap

Observation encoding must live in Rust (Python-side encoding at ~50–200 µs/step would cap PPO around 10k
steps/s, throwing away the whole point of the port), but it is also **the most churn-prone code in the
project**. Resolve this by making it declarative rather than hand-written: a single feature-spec table —
block name, source, width, offset — generated into both languages by the same generator that emits the board
data, versioned by `OBS_VERSION`. Adding a feature is then a table edit plus one accessor, not a rewrite in
two languages, and the Python reference encoder stays as the differential-testing oracle for the Rust one.

*(Superseded by Revision 1: once Python has no encoder, the generator indirection buys nothing. The
feature-spec table moves into Rust as the single source of truth, and the network reads block
offsets from `ttr_rust.obs_spec()` at runtime. See §14.1, Phase 3.1.)*

### 8.3 Heuristics in Rust

H1–H4 live in `ttr-core` with **config-driven constants** (already required by the §1 lesson about the paper's
broken baselines), exposed to Python for tuning and inspection. The Python `Agent` protocol still accepts
pure-Python agents, so quick experimental heuristics remain easy — but anything that needs to run inside a
rollout is Rust.

**The FFI boundary is where the performance is, not the Rust itself.** Per-step FFI is call-bound at ~3–5 µs
and gets you 10–17k games/s across 12 cores. A batched `VecEnv::step(&[u16])` / `observe(&mut [f32])` writing
zero-copy into a torch-shared buffer drops to ~0.3–0.6 µs and gets **85–170k games/s**. Python must never
loop over envs. Pin rayon to the 8 P-cores — E-core stragglers hurt batch-synchronized self-play more than
their throughput helps.

**Why this isn't premature.** A game is ~230 decision nodes; at 400 sims/move that's ~92k NN evals/game, or
0.12–0.5 s/game of pure inference. At 0.25 µs/step the env is 5–15% of wall clock; at 30 µs/step (Python) it
is 85–95%. That's the difference between a 6-hour and a 3-day training run.

**Explicitly not worth doing: a numpy-vectorized-over-games env.** TTR's control flow is deeply branch-heavy
(5 phases, locomotive gating, flush cascades, lazy reshuffle, double-route predicates); a SIMD-over-games
version would be dominated by `np.where` masking of divergent branches, likely *slower* than plain Python,
and an order of magnitude harder to verify. Use numpy only where the work is genuinely rectangular:
observation encoding and masks for a batch of already-stepped states.

**Differential harness** — compare at **every** step, not just terminal, or you learn only *that* you
diverged, never *where*:

```python
while not a.is_terminal():
    assert a.state_hash() == b.state_hash(), f"diverged at turn {a.turn}"
    assert a.legal_actions() == b.legal_actions()
    act = la[policy_rng.below(len(la))]
    a.step(act); b.step(act)
assert a.final_scores() == b.final_scores()
```

100k seeds × {2,3,4,5} players nightly, 1k pre-commit, both chance modes, fuzzed `RuleConfig` variants.

## 9. Training infrastructure

**One machine, in-process Rust env. Do not build a distributed actor-learner** — IPC buys nothing on one box.
PPO runs fully synchronous in one process. AlphaZero splits into self-play (Rust batched ISMCTS + MPS
inference, reloading weights from a memory-mapped file every ~60 s) and trainer (shared-memory ring buffer),
throttled to give self-play ~70% of the GPU.

**Store game records, not observations.** A 2,760-float obs is 11 KB; 2M positions = 22 GB. A game record is
`seed + action list` ≈ 300 bytes; 2M positions ≈ 5 MB, regenerated in Rust at ~8 µs/position — free relative
to the NN. The larger payoff: **you can change the feature set without discarding data.** Over a project
where the encoding will be revised several times, that's worth more than the disk.

**MPS practicalities that cost days if unknown:** no `.item()`/`.cpu()` in the hot loop (each is a full GPU
sync — accumulate on-device); set `PYTORCH_ENABLE_MPS_FALLBACK=1` and *grep the logs for fallbacks*, since
one silent CPU-fallback op in the trunk will 10× your step time invisibly; `pin_memory` is meaningless on
unified memory; validate bf16 by running one iteration on CPU and MPS with the same seed and comparing losses
to 1e-3 before trusting it. **In eval workers, run inference on CPU with `torch.set_num_threads(1)`** — ten
MPS contexts thrash unified memory, and for small nets dispatch overhead makes MPS slower than CPU anyway.

**Throughput targets:** Rust step <3 µs, step+obs <8 µs (→<1 µs incremental), ~1.2M steps/s across 10 threads;
net forward ~165k obs/s at 9M params / batch 4096, ~350k obs/s for a trimmed 4M variant; PPO end-to-end
60–150k env steps/s; MCTS self-play ~15k games/h at 400 sims, ~45k at 128.

*(Measured 2026-10-15, M2 Max, USA 2P, random actions: step is 0.04 µs/env batched, well inside
target, but **observe costs ~120–190 µs of CPU per observation** (exact Steiner, `remaining_cost`,
`fragility`). Batched at 1024 envs that is 23.8 µs/env-step of wall time on 8 cores, so the loop caps
at **~35k env-steps/s**, about 30× short of target and below what the network can consume. FFI
overhead is ~0.04 µs per call and is not the problem. Phase 3.2 exists to close this gap.)*

**Budgets (2P):** PPO beats H3 at 30–60M steps (4–8 h); beats H4 at 150–400M steps (1–3 days); R2 clearly
above H4 and ISMCTS-2000 at 300k–1M self-play games (10–35 h); "beats strong humans" at 2–5M self-play games
(1–3 weeks continuous).

## 10. Reproducibility

One `seed` in the config; everything else **derived**, never drawn:
`derive(root, *parts) = blake2b(root ‖ parts)`, with named streams `env/game_id`, `agent/seat/game_id`,
`torch`, `init`, `worker/wid`, `buffer_shuffle/epoch`.

Four non-negotiable rules: the engine never touches global `random`/`np.random` (this is the precondition
that makes mirrored evaluation possible at all); **environment randomness is fully materialized at game
construction** (§5.1); agent stochasticity is a separate per-seat stream; worker streams come from
`SeedSequence(root).spawn(n)` so results don't depend on scheduling order.

Run dir `runs/{ts}-{name}-{git_sha}[-dirty]/` holds `manifest.json` (git sha, diff hash, `uv.lock` sha256,
python/torch versions, device, argv, resolved config + hash, `OBS_VERSION`/`ACTION_SPACE_VERSION`,
determinism level), `config.resolved.toml`, `workdir.patch`, `metrics.jsonl`, `tb/`, `checkpoints/`, `eval/`,
`replays/`.

Checkpoints **self-describe their architecture and encoder contract** so `ttr arena` can load any historical
checkpoint months later: `{format_version, step, arch{class,kwargs}, obs_version, action_space_version,
model_state, optim_state, rng, config_hash, git_sha, run_id, elo}`. Refuse to load on version mismatch.
`ttr gc` keeps last-N + every-K + `best.pt` + **every checkpoint ever promoted into the league pool**
(permanent — they're your opponents), stripping optimizer state from older ones.

**State the three reproducibility levels honestly** in `docs/reproducibility.md`: **L0 Replay** (bitwise,
always, tested); **L1 Rerun** (bitwise for engine/heuristics/MCTS given same seed + lock + sha, enforced by a
CI test that runs a seed twice and diffs action logs); **L2 Statistical** (NN training on MPS is not bitwise
reproducible across torch versions — same distribution, same curve within noise). Record which level each run
achieved. Pretending L2 is L1 will cost you a day chasing ghosts.

## 11. Evaluation

**Paired/mirrored seeding, mandatory.** A **seed block** is both the unit of work and the unit of statistical
analysis: `block_seed` fixes the deck permutation, ticket permutation, and initial deals; rotation `r` seats
agent `i` at `(i+r) % P`. P=2 → 2 games/block; P=4 → **cyclic rotations only, 4 games/block, not all 24
permutations** (most of the variance reduction at 1/6 the cost).

**Statistics rule that's easy to get wrong:** games within a block are correlated. Aggregate to a per-block
score first, then treat **blocks** as the i.i.d. units; bootstrap CIs resample *blocks*, never games.
Treating paired games as independent inflates significance by ~√P and will make you believe in improvements
that aren't there.

**Sample sizes:** ~400 paired games (800 total) to resolve 55% vs 50%; ~150–200 paired for an obvious 65%;
200 blocks per training-curve eval point with the band drawn on the plot. **Use SPRT for promotion decisions**
(H0: Δelo=0, H1: Δelo=+20, α=β=0.05) — it typically stops in 300–3000 games and, crucially, stops *early* on
the many changes that don't help.

**Ratings: Bradley–Terry MLE batch-refit over the full game table, not incremental Elo.** Decompose each
P-player game into `C(P,2)` pairwise comparisons by rank; fit by convex MLE (~30 lines of numpy Newton);
**anchor H3 = 0 permanently** so the scale is comparable for the project's lifetime; weak `N(anchor, 400²)`
prior so a 6-game checkpoint doesn't land at ±∞; CIs by bootstrap over blocks. **Also store the raw win-rate
matrix** — Elo compresses a matrix to a vector and hides rock-paper-scissors, which is exactly what you need
to see in a league. From it compute **`league/cycle_fraction`** (fraction of triples with A>B>C>A): the one
number that says whether the league is progressing or just spinning.

**Expected ladder:** random ≈ −900 · H1 ≈ −400 · H2 ≈ −120 · **H3 = 0** · H4 ≈ +130 · ISMCTS-2000 ≈ +350 ·
R2 target **+700 to +900**.

**Exploitability probe.** True exploitability is uncomputable here, so: freeze agent A, train a fresh PPO
agent B *only* against A (stationary opponent → clean single-agent problem), fixed 30M-step budget. BR >70%
means A is badly exploitable; BR stalling near 55% means A is reasonably robust. **Track this at every
milestone** — it's the best "actually good, or just good against itself?" diagnostic, and each BR agent is a
free new league member.

**Results store: SQLite, WAL mode**, summary rows only (`agent` / `match` / `game` / `seat` / `rating`), with
bulk per-turn data in parquet and sampled replays in msgpack. ACID matters — a crashed 100k-game arena
mustn't corrupt prior results. Agents are content-addressed by `(checkpoint_path, config_hash, git_sha)` so
ratings accumulate across months. All writes happen in the parent; workers return plain records.

**Diagnostics — what the agent actually learned.** Outcome: win rate vs each baseline, Elo + CI, score
distribution, **`margin_mean`** (continuous, moves thousands of steps before win rate does), **seat win rate**
(must be flat — the canary that mirroring works). Tickets: initial kept (2 vs 3 is a real choice), completion
rate, points earned/lost, redraw turn histogram, mean value kept. Routes: length histogram,
**points-per-train** (watching this rise is watching it learn the scoring table), gray fraction,
**blocking claims** (claims reducing an opponent's inferred remaining cost by ≥3 — this distinguishes
"learned solitaire optimization" from "learned to play against someone"), double-route claim rate.
Longest-path bonus rate. Cards: hand-size curve, face-up vs blind ratio, **cards and trains wasted at end**
(strong play ends with 0–2 trains). Actions: fraction by type, drawing vs claiming. Policy health: entropy
**split by game phase** (catches collapse long before aggregate entropy does), top-1 prob, legal-action count,
`mask_violations`, value calibration bins.

**Tracking: TensorBoard, with mandatory JSONL dual-write.** The architectural point is the JSONL, not the
viewer — `metrics.jsonl` is greppable, polars-readable, diffable, survives a crashed run, and isn't hostage
to an event-file format. TB is a *rendering* of it; a W&B backend later is 30 lines and can backfill every
historical run.

### Failure modes to watch for

| Failure | Detection | Fix |
| --- | --- | --- |
| **Never draws extra tickets** — most likely and most damaging; the −points penalty looks risky to a value function that hasn't learned to complete them | `rate_of_extra_ticket_draws`; `mean_tickets_held_at_end` (strong 2P ≈ 3.5–4.5, not 2.0) | `P(complete)` aux head; entropy bonus on the action-type head; ε% of games force a ticket draw at a random turn |
| Ticket gluttony (the inverse) | `ticket_points_lost` (strong: <3) | usually self-corrects with the `P(complete)` head |
| Card hoarding | hand-size curve, `cards_wasted_at_end` (strong: 3–6) | **don't over-correct** — moderate hoarding is genuinely strong (hides info, enables a burst); only hand>25 / 15+ unused is a bug |
| **Ignores blocking** — 2P self-play has a stable "neither side blocks" equilibrium | blocking counter; **the BR probe finds and crushes a non-blocker instantly** | keep H4 permanently in the pool at ≥10% |
| **Self-play pool overfit** | canonical alarm: **win rate vs *scripted* baselines falls while win rate vs the *pool* rises** — log both; also entropy <0.3 nats, or <20 distinct opening claims per 1000 games | PFSP weighting, permanent scripted anchors, KL-to-past |
| Longest-path bonus ignored (10 pts, often decisive) | bonus rate (strong 2P >55%) | `P(longest_path)` aux head + `extends_my_longest_chain` |
| **Double-route bugs in 2–3P** — the most common bug in TTR implementations | dedicated test *and* a runtime assertion; also check the agent's double-route claim rate isn't 0 (if the engine wrongly locks both halves, the agent silently avoids them) | test both directions |
| Value head learns turn count, not position | compare `V` against a turn-index-only baseline predictor, bucketed by phase | more aux heads; check for a dead trunk |
| Seat leakage in N>2 | rotate seats in eval; performance must be identical | assert no absolute seat index leaks into the encoding |

## 12. Human interface

**A human is just an `Agent`** — `agents/human.py` implements the same protocol, blocking on a caller-supplied
prompt function. "Human vs bot" is then literally an arena match: no special-casing in the engine, no separate
game loop, and the human path is exercised by every arena test. *(Revision 2: the web server drives turns
instead. A human seat is where the game service waits for a client message, and bots still go through one
agent interface. §14.2.)*

**Terminal MVP** (`ttr play --seat 0 --opponents h3`): routes grouped by region and colored by owner, doubles
marked, your hand as colored counts, the face-up row, your tickets with live *connected?* and *remaining cost
in trains*, scores/trains/deck, and **a numbered `legal_actions()` menu**. That last item is load-bearing —
it's exactly the action space the agents see, so it's the human UI and the best debugging tool in one. When a
bot does something baffling, play the same position and read the menu. *(Superseded by Revision 2: no terminal
client. The menu's virtue survives as the server sending decoded legal actions, so the client never re-derives
the rules.)*

**Coordinates:** RULES.md has none, so hand-author **real lat/lon** for the 36 cities (~20 minutes, public
factual data) plus cosmetic `nudge_x/y` for the Boston/NY/Washington cluster, and project once for every
renderer. Do *not* transcribe positions off the physical board — that artwork is copyrighted. Growth path
`--map=list | ascii | braille` (braille gives 2×4 subpixels per cell → a 220×160 canvas, smooth lines).
*(Revision 2: the growth path goes with the terminal client; the lat/lon table and the nudges stay.)*

**Web** (`ttr serve`): FastAPI + **one static HTML file, no npm, no bundler** — vanilla JS + inline SVG from
the same `projection.py`. `/api/state?seat=0` returning a seat-limited view is also the perfect test that
information-set masking is correct. Growth: human play → replay scrubber → **agent introspection overlay
showing policy probabilities and predicted final score as you scrub**. That overlay is the single highest-value
debugging tool in the project — it turns "the agent is bad" into "the agent thinks claiming Denver–Omaha is
worth 0.02 on turn 12, which is insane." *(Revision 2: Svelte + TypeScript built by Vite into static files,
not one hand-written HTML file, at the user's choice. Human play and the replay scrubber land in Phase 4,
with the heuristics' per-term reasoning in the scrubber; the policy/value overlay for learned agents stays in
Phase 8. §14.2.)*

## 13. Scope reduction & TTR-mini

**2P first**, then 4P, then 3P/5P — one net throughout (DeepSets opponent encoder + player-count feature
handle variable N; four nets would be wasteful and transfer poorly). 2P isn't throwaway work: the
double-route lockout makes it a genuinely distinct, more tactical game.

**Build TTR-mini in week 1 as a data file, not a code path** — same engine, different board. It pays for
itself three ways: brute-force engine verification, 10-minute PPO runs instead of 8-hour ones for algorithm
iteration, and **the only setting where Deep CFR can give a real exploitability number**.

Design it using the paper's own survey (mean degree 4; #tickets ≈ #cities; ticket points = shortest-path
segment count) — but **keep 6 colors, keep max route length ≥5, and keep some double routes**. The paper
dropped to 4 colors and max length 3 with zero doubles, which removes color scarcity, the high-commitment
end of the scoring curve, and blocking — i.e. most of what makes TTR interesting. Target ~14 cities /
~27 edges / 6 colors / 20 trains.

**Train-count curriculum** (20 → 30 → 45) halves the horizon early and makes tickets relatively riskier,
which teaches ticket evaluation faster — but it changes the optimal policy, so ≥40% of the budget must be at
45 trains. One ablation, not a default.

**Fixed ticket deals: never for training** (destroys generalization), **always for evaluation** (§11).

**De-risking ablations, in strict priority order:** (1) engine matches a slow reference on 10k random games
including longest-trail and all tiebreaks; (2) H3 > H2 > H1 > random by the predicted margins — if not,
everything downstream is built on sand; (3) flat Monte Carlo with H3 rollouts beats H3 (if policy improvement
doesn't happen, the rollout/determinization code is broken); (4) PPO beats random within 2M steps (if not,
the observation or the mask is broken); (5) value calibration by turn index; (6) **remove per-ticket
`remaining_cost` and see if it matters — the single most diagnostic ablation in the project**; (7)
determinization validator passes on 1M samples.

## 14. Phases & exit criteria

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| **0** (½ d) | RULES.md normalization (§Step 0); pyproject with extras/dev group/uv indices; layout; ruff+ANN/ty/pytest/pre-commit/Makefile/CI; delete `main.py` | `make setup lint type test` green; CI green; `uv sync --dev` completes with no torch and no CUDA |
| **1** (2–3 d) | `usa.toml` + generator → `board_gen.{py,rs}`; full Python engine; replay; **frozen PRNG/draw/hash contract (§5.8)**; feature-spec table (§8.2); TTR-mini; throwaway H0/H1 + flat-MC stub as API exercise (§8.1); `ttr map`, `ttr bench` | all ~40 edge-case tests + 14 property tests green; engine coverage ≥95%; imports with torch blocked; ≥2000 random playouts/s/core; a seed replays bitwise; all board invariants asserted; **the contract doc has test vectors** |
| **2** (4–6 d) | **Rust core** + maturin + batched `VecEnv` + Rust obs encoder + search hooks (`clone_into`, `resample_from_infoset`, `position_hash`, `collect_leaves`/`backup`) + differential harness | byte-identical to Python on 100k seeds × {2,3,4,5}P, compared **at every step**, both chance modes; obs encoder matches the Python oracle exactly; ≥50× throughput (target ~85–170k games/s batched); `pytest -m bench` baseline recorded |
| **3** (2–3 d) | H0–H4 **in Rust** with config-driven constants; `eval/` (pairing, parallel runner, SQLite, BT-MLE + bootstrap CIs, SPRT); `ttr arena`, `ttr leaderboard` | 10k paired games across 5 agents in **seconds**; leaderboard stable across two runs of the same seed root; **seat win rate flat within CI**; ratings accumulate across invocations; H3>H2>H1>random by predicted margins; **every heuristic exercises every action type on both maps** (the paper's bug) |
| **3.1** (3–4 d) | *Revision 1.* Retire the Python engine; Rust becomes its own oracle: golden corpus from the last Python-verified build, frozen `oracle-v1` crate for per-step differential tests, Python tests ported to Rust, naive reference implementations, mutation testing; obs spec moves into Rust; `uv sync` builds `ttr_rust` (§14.1) | golden corpus reproduced exactly; per-step differential vs `oracle-v1` green; every Python rules/property test has a Rust counterpart; `ttr-core` coverage ≥95%; no surviving mutants in rules/scoring; Python engine, encoder and differential harness deleted; docs updated |
| **3.2** (3–5 d) | *Revision 1.* Rust performance, free of Python's structural constraints: profile, then cache/incrementalize Steiner and `remaining_cost`, fixed-size bitsets instead of maps, f32, no hot-path allocation (§14.1) | observe ≤15 µs CPU/obs single-threaded (from ~150); ≥250k env-steps/s batched observe+mask+step at 1024 envs (from 35k); state hashes identical to `oracle-v1` at every step; H3 behaviour hash unchanged (`5b906156d4b1204b`); every cache asserted against recomputation |
| **3.3** (1–2 d) | *Revision 1.* Modal: image that builds `ttr_rust`, a runs Volume, `ttr cloud …` wrapper, `device` setting, spend cap (§14.1) | `ttr bench` and a 10k-game arena run on Modal (CPU and GPU containers); arena outcomes identical to local for the same seed root; results land locally; budget cap set; cost per experiment recorded |
| **3.4** (3–5 d) | *Revision 1, tentative.* PPO spike on TTR-mini 2P, plus the `VecEnv` RL additions it forces: returns before reset, seat→policy routing with scripted opponents stepping in Rust, aux targets, game-record export (§14.1) | beats random ≥95% within 2M steps (§13 ablation 4); beats H1 on mini; `mask_violations == 0`; measured steps/s on MPS and on a Modal GPU replace the §9 estimates |
| **4** (5–8 d) | *Revision 2: the web client (§14.2).* Engine bindings for one decision at a time with the heuristics' reasoning; a FastAPI game server and play store; a Svelte client with a board in the physical game's style, spectating and post-game review; human ratings. *(Was: `HumanAgent`, projection + lat/lon, terminal client, `ttr play`, `ttr replay`.)* | you play full 2P games against H3 and H4 on both maps without confusion; every game rebuilds from the store; the masking test passes; your rating sits on the bot leaderboard with an interval; **you lose to H3 at least once** |
| **5** (3–4 d) | Planned API revision pass (§8.1); determinization sampler + validator + particle filter; flat MC; SO-ISMCTS | ISMCTS-1000 beats H4 with SPRT-confirmed Δelo >+100; **strength monotone in sim count (200<500<1000<2000)** — the canonical proof the search is correct; validator passes 1M samples |
| **6** (4–6 d) | `rl/encode` (versioned); nets; vec env; PPO; league; tracking; run dirs/manifests/checkpoints *(Revision 1 adds: history features as `OBS_VERSION` 2 and the history-probe experiment, §14.1; **now version 3** — 2 went to the plan-walk fix before Phase 3.4, see WORKLOG 2026-10-17)* | run dir complete and `--resume` continues the same trajectory; **`mask_violations == 0` throughout**; beats H3 by ~40M steps and H4 ≥65%; checkpoints refuse to load under a bumped `OBS_VERSION`; `points_per_train` visibly rises; history probe reported either way |
| **7** (1–2 w) | Batched ISMCTS + learned prior/value + belief head; AZ trainer; PFSP league | R2 beats the PPO agent (SPRT); Elo rises monotonically over ≥5 promotion generations; `cycle_fraction` tracked; BR-probe exploitability trending down; a 24 h unattended run completes |
| **8** | Transformer net + color-symmetry regularizer; introspection overlay for learned agents *(Revision 2: the web UI itself moved to Phase 4)*; public deployment and human gauntlet; optional R-NaD / Deep-CFR-on-mini | **beats you consistently**; your Elo lands on the same leaderboard as the bots |

Phase 1 is now the highest-leverage phase in the project: it's the last point at which the rules contract,
the state layout, and the feature-spec table are free to change. Spend the extra half-day there.

### 14.1 Revision 1 (2026-10-15): retire Python, make Rust fast, get to the cloud

**Why.** The Python engine's job was to catch Rust bugs by disagreeing. That value is mostly banked:
rules were verified identical over 700k games, base TTR rules never change, and every further
Python-side line is paid twice. It also has a cost that didn't show until measurement. Rust code was
being written to match Python *structurally* (f64 arithmetic, the same algorithm shapes), and the
encoder that resulted is the bottleneck (§9 note). The user chose to retire it outright rather than
freeze it.

**Ordering.** The oracle replacement comes *before* deletion and before the performance rewrite: the
rewrite is exactly the change that most needs a check, and Python's last useful act is to sign off on
a golden corpus. Phases 3.1–3.3 are firm. Phase 3.4 and what follows are tentative and get re-planned
once 3.3 lands. Phase 5 (ISMCTS) can run alongside 3.4 or 6, since it needs no network.

Numbering: the new phases are 3.1–3.4 rather than a renumbering, so every existing "Phase 5" reference
in code and docs stays correct.

#### Phase 3.1 — Rust becomes the oracle, then Python goes

In order:

1. **Python's last word, a golden corpus.** On the last commit where Python and Rust agree, generate:
   - for 10k seeds × every (map, seat count) × both chance modes: final scores, plus a per-game
     **hash chain** over the `state_hash` after every step (one u64 per game, about 1.5 MB total);
     *(done 2026-10-16, sampled mode only -- explicit mode was never specified, as Phase 2
     recorded -- and extended with 200 H3 games per configuration; see WORKLOG)*
   - observation digests at sampled steps, for `OBS_VERSION` 1.

   The existing 84-game `replays.bin` and `contract_vectors.json` stay as frozen fixtures, and their
   regeneration moves into Rust behind `CONTRACT_VERSION`.
2. **A frozen Rust oracle.** Tag that commit `oracle-v1`. Add `ttr-oracle = { package = "ttr-core",
   git = <this repo>, tag = "oracle-v1" }` as a **dev-dependency only**. *(Done 2026-10-16 with
   `rev = "7923e28…"` in place of `tag`: a SHA cannot be moved, and it resolves whether or not
   the tag is pushed. The tag exists locally. The dev-dependency lives in `crates/ttr-tools`.)* The differential harness is
   rewritten as new-vs-frozen Rust and still compares **at every step**: state hash, legal actions,
   observation, final scores. That keeps the property that a divergence names its step. It costs
   nothing to maintain, since nobody edits a tag. If a git dependency proves awkward in CI, vendor
   the tagged source read-only under `crates/ttr-oracle/` with a pre-commit hook that rejects edits.
   *(Retired 2026-10-17, after Phase 3.4: it had served its purpose through the 3.2 rewrite, it
   already skipped the features `OBS_VERSION` 2 changed and would diverge further with every
   observation change, and its full sweep was most of the verification time. The golden corpus and
   the naive references carry the check; see WORKLOG.)*
   Moving the tag is a deliberate, versioned event, the same as a contract bump.
3. **Port the tests.** The ~40 rules edge cases and the property tests (Hypothesis → `proptest`) are
   ported to Rust, tracked one-to-one in a table in WORKLOG so none silently drops. The RULES.md
   structural invariants stay with the board generator.
4. **Naive references inside the crate**, behind `#[cfg(test)]`: brute-force Steiner on mini, brute-force
   longest trail, a naive 6300-way legal-mask scan, from-scratch recomputation of anything incremental.
   This is differential testing that survives Python: fast path against obviously-correct slow path.
5. **Mutation testing** (`cargo-mutants`) over rules, scoring, graph and obs to measure whether the
   suite catches real bugs, which matters once there is only one implementation. Surviving mutants
   are triaged; none may survive in rules or scoring.
6. **Then delete.** Delete `ticket_to_ride/engine/` (state, actions, graph, scoring, replay, rng,
   hashing), `rl/encode/observation.py`, the throwaway agents, `tests/differential.py`, the
   Python-engine unit/property tests, the import-boundary test, `tools/gen_vectors.py` and
   `tools/gen_replays.py` (both become Rust bins).
7. **What remains in Python:**
   - `ticket_to_ride.engine` becomes a thin re-export of `ttr_rust`, so agent, CLI and UI import
     paths don't move.
   - `data/` stays: board data and city coordinates for the generator, `ttr map`, and the UI.
     It is data, not rules.
   - `tools/gen_board.py` stays, emitting Rust from RULES.md.
   - The obs feature-spec table moves into Rust (§8.2 note), so `spec.py` and `gen_obs_spec.py` go.
8. **Packaging.** `ttr_rust` becomes a required dependency built by `uv sync`, as a workspace member
   with the maturin backend in release profile. CI's test job gains a Rust toolchain. Torch-free CI
   stays. `TTR_REQUIRE_RUST` is retired because Rust is no longer optional.
9. **Docs.** README, GOTCHAS, CONTRACT and reproducibility.md drop "permanent oracle" claims and
   describe the new validation stack.

#### Phase 3.2 — Performance, free of Python's constraints

**Profile first** (`samply` or `cargo flamegraph`) and record the breakdown in WORKLOG before
changing anything. Candidates, roughly by expected payoff:

- **Cache the Steiner tree and `remaining_cost` per seat.** They change only when a segment is
  claimed or that seat's tickets change. Card draws, about half of all steps, are pure cache hits.
- **`fragility` only for segments on the current best tree.** Re-pricing off-tree segments cannot
  change the answer.
- **Bitsets and fixed arrays.** Segments as `u128`, cities as `u64`, fixed `[u16; 36]` distance
  arrays instead of `HashMap<u8, Vec<u16>>`.
- **No allocation on the hot path.** Per-thread scratch buffers, the pattern `sample_legal_into`
  already uses.
- **Incremental connectivity.** Union-find updated on claim, not rebuilt per observation.
- **f32 where f64 only existed to match Python.**

**Behaviour preservation is the contract of this phase:**

- State hashes stay identical to `oracle-v1` at every step.
- Observations: discrete fields bit-exact, continuous fields within 1e-6.
- H3's behaviour hash stays `5b906156d4b1204b`. The heuristics share the graph code, and a moved
  hash would re-base every rating.
- Any change that can't meet these is a named `OBS_VERSION` bump done separately, never folded into
  a speedup.
- Every cache gets a debug-build assertion against from-scratch recomputation, exercised over 10k
  games.

`ttr bench --suite vecenv` reports observe, mask and step separately, with the usual 20% regression
gate.

#### Phase 3.3 — Modal

- **User-only steps:** create the Modal account, authenticate (`modal token new`), set a workspace
  spend limit. The $30/month Starter credit covers early use. Apply for Modal's research credits once
  there are results to show.
- **`cloud` dependency group** with `modal`; `infra/modal_app.py` defines:
  - an image (Debian slim + rustup + uv) that builds the `ttr_rust` release wheel in a layer keyed on
    `Cargo.lock` and the crate sources;
  - a `ttr-runs` Volume;
  - functions for `bench`, `arena`, and a generic `run(argv)` with selectable CPU count and GPU.
- **`ttr cloud bench|arena|run …` and `ttr cloud pull`.** Each cloud job writes **its own** results
  file, merged locally. SQLite on a shared volume with concurrent writers is a corruption risk, not a
  shortcut.
- **A `device` setting** (`auto|cuda|mps|cpu`) in config, so training code is portable from its first
  line.
- **Check `performance_threads()` inside a container.** Off macOS it uses `available_parallelism`,
  which must respect the container's CPU quota rather than the host's core count.
- **Exit check, cross-platform determinism.** An arena run on Linux x86 must reproduce the local
  macOS ARM outcomes game for game.

#### Phase 3.4 — PPO spike (tentative)

Pulled forward from Phase 6 to retire the largest unknowns early: does PPO learn on this observation,
and what does the full loop really sustain on MPS versus a cloud GPU?

- **`VecEnv` additions (these are kept):**
  - final returns readable before `auto_reset`;
  - per-env seat→policy assignment, with scripted opponents stepping inside Rust until the learner's
    turn, so Python never loops over envs;
  - aux-target export (ticket completion, opponents' true tickets, final margin);
  - game-record export (seed + actions, §9).
- **The PPO itself may be throwaway:** mini 2P, small MLP, masked categorical, per-seat GAE.

Phase 6 is then re-planned from what it measures.

#### Additions to later phases

- **Phase 4** is trimmed to `ttr play` (about 1 day). Its exit criterion, that you lose to H3, is a
  real check that the heuristics are strong. `ttr replay` moves to Phase 8's web UI. *(Superseded by
  Revision 2: Phase 4 is the web client, §14.2. The exit criterion survives.)*
- **Phase 6** gains:
  - Cheap history features as `OBS_VERSION` 2 *(now 3: version 2 fixed the plan walk and two quirks before Phase 3.4 — WORKLOG 2026-10-17)*: claim turn per segment, tickets I returned, face-up
    colours each opponent passed on, when each opponent last drew tickets.
  - **The history probe:** a supervised classifier predicting opponents' tickets from H4 self-play
    games, given (a) the observation versus (b) the observation plus action history. If (b) isn't
    clearly better, a recurrent or history-token network is not worth its estimated 1.5–3× training
    cost and its awkward fit with ISMCTS. If it is, that's the Phase 8 architecture question settled
    with data.

### 14.2 Revision 2 (2026-10-17): Phase 4 becomes the web client

**Why.** A terminal is a poor way to play a game whose state is a map, and Phase 4's job is real: the
user, an average-to-above-average player, wants to measure the heuristics against themselves and catch
degenerate decisions before H3/H4 become the RL initialization and benchmark. Later the same client
opens to the public, to rate the bots against people and to collect human games as supervision data. So
the web UI moves up from Phase 8, built from the start as something that can be deployed, and the
terminal client is dropped: no `ttr play`, no `--map=ascii`.

**Decisions confirmed with the user:**

- **Svelte + TypeScript**, built by Vite into static files that FastAPI serves. Node is a build-time
  tool only. This replaces §12's "one static HTML file, no npm": a polished, animated client with a
  lobby, a replay scrubber and, later, accounts outgrows hand-written DOM code.
- **Scope:** USA and mini, 2 players, you against any heuristic; and **spectating** bot-vs-bot games
  with every hand and ticket visible and each bot's reasoning on screen. Not now: 3–5 players, phone
  layouts. The protocol stays N-player so neither needs a redesign.
- **Assistance is bot parity, the same in every rated game.** The UI shows the public information the
  bots' observation holds: each opponent's known cards (`certain`), the discard composition, unseen
  counts per colour, and each of your tickets' progress and remaining train cost. Ratings then measure
  strategy rather than memory. The level is recorded with each game (`assist = "parity-1"`), so a later
  change splits the data cleanly instead of silently.
- **Review:** after a game, a scrubber over every decision with all hidden information revealed and, at
  each bot decision, its top candidates with their scores broken down by term. Not now: flagging moves,
  same-deal comparisons, bot reasoning during rated play.
- **Look: the physical board's.** See *Visual design* for what that can and cannot mean.

**Architecture.**

```
Svelte client ──REST──────────────────▶ FastAPI ──▶ GameService ──▶ ttr_rust: State, Heuristic, board_data
      ▲  └────── WebSocket /ws/games/{id} ──▲             │
      └─────────────────────────────────────┘             └──▶ PlayStore: runs/play.db (SQLite WAL, portable SQL)
```

- **Server-authoritative.** The browser never receives hidden information: no seed, no deck order,
  nothing of an opponent's hand beyond `certain`, no opponent tickets before the end. Legal actions
  arrive decoded from the server (claim segment *s* paying 3 blue + 1 locomotive; draw slot 2; keep
  tickets {4, 17}) -- the old terminal menu's virtue -- so the client implements no rules and there is
  still one implementation of them.
- **The store is the source of truth; memory is a cache.** Each decision is appended as it happens, and
  a game is rebuilt from `(seed, actions)` in microseconds. A rebuild also calls each bot at each of its
  past decisions and **asserts it reproduces the recorded action**. That warms its plan cache, advances
  H0's stream, and makes every rebuild an L1 check: a bot whose code changed mid-game is reported, never
  papered over. A restart, or one day a second server, loses nothing.
- **Bots run server-side** through `agents/registry.py`: spec strings `h0`–`h4`, `h3@file.toml` for
  parameter overrides, and later `ppo:path` (torch imported lazily, §2 boundary 3). Agents are identified
  as in the arena store, `(family, params_hash, behaviour_hash, checkpoint)`, so one bot is one identity
  everywhere. Moves stream to the client as they are made (search agents will take seconds per move),
  and the client paces the animation at an adjustable speed.
- **The human's seat is random in rated games.** First-player advantage is real (seat 0 wins 50.7% on
  USA 2P and 52.0% on mini; GOTCHAS), and unpaired games cannot rotate it away.

**Protocol (a sketch; 4.2 pins it).** Pydantic models are the source of truth. They emit a JSON Schema,
checked in with a `--check` mode like the board data, from which the client generates its TypeScript
types.

- `GET /api/boards/{map}` -- the engine's board data plus the layout (below). The client reads this,
  never the map TOML: mini's TOML is not in engine order (the generator canonicalizes it).
- `GET /api/agents` -- opponents, with ratings on the H3 = 0 scale.
- `POST /api/games` -- `{map, opponent, seat}` to play; `{map, agents: [a, b]}` to spectate.
- `WS /ws/games/{id}` -- server to client: `snapshot` (view, legal actions, log), `event` (one action,
  decoded and public, with the view after it), `over` (the final view with everything revealed, and the
  score breakdown); client to server: `act {ply, action}`, **idempotent on `ply`**, so a double click or
  a reconnect cannot apply a move twice.
- `GET /api/games/{id}/review` -- finished games only: the full history, the hidden information, and
  each bot decision's explanation, **recomputed by replay** rather than stored. It is deterministic; for
  learned agents the same slot will carry policy probabilities and the value estimate (Phase 8's
  overlay).

A **seat view** is the public state plus the viewer's private state: the owner of each segment and which
double halves are closed; face-up cards, deck size, discard counts, ticket-deck size; my hand, my tickets
with their progress, and my offer during a keep; per opponent, trains, visible score, hand size, known
cards, ticket count and longest trail; phase, current player, draws left, final-turn countdown; a
plain-English log. Spectator and review views are omniscient.

**Information-set masking is tested, not assumed.** §12's "perfect test", made exact: on random
positions, a seat's view of a state equals its view of `resample_from_infoset(seat)` of that state, which
differs in everything the seat cannot see. A second test searches every pre-terminal payload for the
seed.

**Engine additions** (`ttr-core`, `ttr-py`). Behaviour-preserving: no change to the rules, `state_hash`,
the corpus or `OBS_VERSION`, and every tier's behaviour hash unchanged. Heuristic code is touched but the
rules and encoder are not, so the fast tier plus the behaviour-hash tests are the check; the full sweeps
do not apply.

- A **`Heuristic` pyclass**: `Heuristic(tier, params=None, seed=0)`, `begin_game(seat, game_id)`,
  `act(state)`, and `explain(state, top_k)`, which returns each candidate's total and its per-term
  breakdown (route points, train cost, plan value, contest, double lockout, longest chain, gray
  cannibalization, locomotive thrift, endgame urgency, H4's threat terms; the keep phase's subsets with
  their value; H1's rule that fired). **`explain` and `act` share one code path** -- an optional trace
  sink threaded through the scorer, not a second scorer -- and a test asserts that explain's argmax is
  act's choice over 10k positions, on both maps, for every tier.
- `State.offer()`, `State.discard()`, `State.ticket_progress(seat)`; `Game.decode_action(a)`, so neither
  Python nor TypeScript re-derives the action layout; and `board_data(map)`, the engine's own arrays
  (cities, segments with siblings, tickets, colours), so the client draws exactly the board the engine
  plays.

**Visual design.** The user wants it to look like the board game, so the client follows the physical
board, with photos of it as a style reference: a vintage map of North America (parchment land, blue sea
and Great Lakes, faint borders); routes as chains of rounded car spaces in the eight card colours, gray
for any colour, doubles as parallel chains; claimed routes covered by trains in the owner's colour;
cities as ringed dots with name plaques; the score track around the edge; destination tickets as cream
cards with a small map; and a train card per colour, with locomotive art. Every colour also carries a
symbol, for colour-blind players.

**What "looks like" cannot mean: the published artwork is copyrighted.** It is never downloaded into the
repo, embedded, traced or committed (§12 already ruled out transcribing positions off the board), and
the public site needs a name and logo of its own. Everything is drawn: geography from Natural Earth
(public domain), cities at their real lat/lon (§12) through the same projection, cosmetic nudges where
clusters crowd (New York–Boston–Washington, Vancouver–Seattle–Portland), and per-route curvature where a
straight chain would cross a city or squeeze its cars. The layout source is
`data/layout/{usa,mini}.toml` (mini's cities are a subset of USA's), and `tools/gen_layout.py`
projects it, with the basemap, into checked-in output with a `--check` mode. It is UI data, outside
`DATA_HASH`, so the engine, the corpus and the contract are untouched. Not under `data/maps/`:
`gen_board.py` reads every TOML there as a board and asserts the set of maps.

**The play store** is `runs/play.db`, gitignored like `results.db`. It is a separate database because it
is the web app's operational store, the one that becomes Postgres:

| table | holds |
| --- | --- |
| `player` | humans: id and display name now; auth provider and external id later. Bots are rows too, pointing at their agent identity |
| `game` | map, mode (play or spectate), rated, assist level, status, seed (served only once finished), contract and obs versions, data and rules hashes, git sha, client build, timestamps |
| `seat` | game, seat, player, final score and its breakdown, rank |
| `action` | game, ply, seat, action, server time, think time |

Replays and supervision data are these rows replayed; nothing the engine can recompute is stored.

**Ratings.** Finished rated human games join the arena's games in the Bradley–Terry fit, on the same
H3 = 0 scale, each game its own block (§11), a human identified by player id. A profile page shows your
rating with its interval, your record per opponent, and score distributions, and the leaderboard lists
people and bots together -- Phase 8's exit line, "your Elo lands on the same leaderboard as the bots",
available from Phase 4.

**Layout.**

```
web/                        Svelte + TypeScript (Vite); builds into ticket_to_ride/ui/web/static/ (gitignored)
ticket_to_ride/ui/web/      FastAPI app: protocol (pydantic), views, games, store; `ttr serve`
ticket_to_ride/agents/registry.py
ticket_to_ride/data/layout/  {usa,mini}.toml, and gen_layout's output
tools/gen_layout.py
```

`ui/terminal/` and `ui/render/` are deleted.

#### Sub-phases

| | Deliverable | Exit criterion |
| --- | --- | --- |
| **4.1** (1–2 d) | Engine additions: the `Heuristic` pyclass with `explain`, the `State` and `Game` accessors, `board_data` | explain's argmax = act over 10k positions, both maps, H1–H4; every behaviour hash unchanged; stub test green; no rules, hash or corpus change |
| **4.2** (2–3 d) | Server: registry, game service, play store, REST + WebSocket protocol, views, schema generation, `ttr serve`; human games in the leaderboard | full games through the API on both maps; a restart mid-game rebuilds and continues, every bot reproducing its moves; duplicate and illegal `act` handled; masking test over ≥1k positions × seats × maps; no seed before the end; human games rated on the H3 = 0 scale |
| **4.3** (3–5 d) | Client: layout, basemap and board renderer, shown to the user and signed off before the screens; then lobby, game screen, spectate, review, profile; a CI job | a full game on both maps against the real server; review shows every bot decision's breakdown; spectating runs live; `svelte-check`, `vitest` and the build green in CI |

4.1 → 4.2 is a strict order. 4.3's board work needs neither (segment order matters only once the board
is wired to game state), so it runs alongside them, and its screens integrate once 4.2 is merged.

**Phase 4 exit:** you play full 2P games against H3 and H4 on both maps without confusion; every game
rebuilds from the store; your rating sits on the leaderboard with an interval; and **you lose to H3 at
least once**.

**Planned for the public deployment, not built now:** one container (a Node stage builds the client;
Python and `ttr_rust` serve it); the database URL from the environment (Postgres); accounts with guest
play; a data-use notice, since games become training data; rate limits; one live rated game per player;
server-timed moves; our own name and art. Modal can host the ASGI app only with an external database,
since its containers are ephemeral; a small VM with SQLite and Litestream is the alternative. Decide when
deploying.

## 15. Main risks

| Risk | Mitigation |
| --- | --- |
| **Rust API frozen before agents/search exercise it → second port pass** (the main cost of the new ordering) | Phase 1 throwaway H0/H1 + flat-MC stub against the Python engine; search hooks designed in from day one; one API revision explicitly budgeted at the start of Phase 5 (§8.1) |
| Rust port stalls and blocks everything downstream *(retired by Revision 1: the port landed)* | The Python engine stays fully functional — agents and arena can be built against it if the port slips; only throughput suffers, nothing is blocked |
| Observation encoding churn means rewriting Rust repeatedly | Declarative feature-spec table generated into both languages (§8.2); adding a feature is a table edit. *(Revision 1: one language now, so the churn cost halves.)* |
| *(Rev. 1)* With Python gone, Rust validates itself and correlated bugs go unseen | Golden corpus signed off by Python before deletion; frozen `oracle-v1` per-step differential; in-crate naive references; mutation testing (§14.1) |
| *(Rev. 1)* The performance rewrite silently changes behaviour | Per-step differential against `oracle-v1`; H3 behaviour hash pinned; every cache asserted against recomputation; semantic changes only via a named `OBS_VERSION` bump |
| *(Rev. 1)* Cloud spend runs away | Modal workspace budget cap set by the user; per-function timeouts; start inside the free monthly credit |
| *(Rev. 1)* Linux x86 and macOS ARM disagree, so cloud results don't match local | Phase 3.3 exit requires game-for-game identical arena outcomes across platforms |
| H4 isn't actually strong, making "beats heuristics" hollow | Phase 4 exists partly so H3/H4 play *you*. H4 must beat random ~100%, not the paper's 91%. |
| Self-play collapse / cycling | PFSP + permanent scripted anchors + BR probe + `cycle_fraction`, all from day one of Phase 6. |
| Silent illegal-action leak poisons months of training | `mask_violations` logged and asserted; mask recomputed at update time. |
| Determinization sampler subtly wrong | `assert_consistent` on every sample in debug builds; 1M-sample test. |
| MPS silently falls back to CPU for one op | `PYTORCH_ENABLE_MPS_FALLBACK=1` + grep logs; CPU-vs-MPS loss comparison at 1e-3. |
| Luck variance masks real gains | Seed blocks, bootstrap over blocks not games, SPRT for promotions. |
| *(Rev. 2)* Hidden information leaks to the browser, making human games worthless as ratings or data | Server-authoritative views; the `resample_from_infoset` masking test; no seed in any pre-terminal payload (§14.2) |
| *(Rev. 2)* A bot's explanation drifts from what it actually plays | One scorer with an optional trace sink, never a second scorer; explain's argmax asserted equal to act over 10k positions |
| *(Rev. 2)* The board's look copies the published game's art or branding | Original art from public-domain geography; photos are a style reference only, never committed; own name and logo before any public deployment |

## 16. Verification

```bash
uv run pytest -n auto                    # unit + property, <20 s, no torch
uv run pytest -m slow                    # 100k random games, golden replays, determinism
uv run ttr map --format ascii            # board data + coordinates sanity check
uv run ttr bench --suite engine          # games/s; absolute floor in CI, 20% regression gate locally
uv run ttr arena -c configs/arena/panel_v1.toml --blocks 500   # Elo table + win-rate matrix + CIs
uv run ttr serve                         # play the heuristics in a browser (Revision 2: was `ttr play`)
                                         # finished games open in the review scrubber (was `ttr replay --web`)
```

CI (GitHub Actions on `git@github.com:edwisdom/ticket_to_ride.git`): `lint` (~30 s) and `test-fast`
(~1–2 min, **no torch installed**) on every push; `test-rl` via the pytorch-cpu index only on PRs touching
`rl/**`; nightly full arena + PPO smoke asserting no crash, `mask_violations == 0`, and loss decreased.
Include the determinism check in `test-fast` — it's ~2 seconds and it guards the property everything else
depends on.
