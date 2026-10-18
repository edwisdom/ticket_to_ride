# Worklog

Reverse-chronological log of what was actually done, and why. Append an entry per work
session. Decisions that deviate from [PLAN.md](PLAN.md) belong here, with the reason.

---

## 2026-10-18 — H3 reads 1500, not 0

**Departs from PLAN.md §11** ("anchor H3 = 0"). H3 is still the anchor, pinned by its
behaviour hash; what changed is the number it reads. A competent heuristic sitting at 0,
with everything below it negative, reads strangely to anyone who knows chess ratings,
so the scale is now offset to look like one.

**Why 1500.** It is the Glicko/Lichess starting rating, the conventional "unknown,
average club player". With H3 there, the measured ladder lands where chess intuition puts
it: random play at 140 (USA 2P) to 637 (Mini 2P), near USCF's floor of 100; H1 around
700-900, a rank beginner; H2 around 1350. §11's predicted ISMCTS (+350) and R2 target
(+700 to +900) become roughly 1850 and 2200-2400, which reads as "master". 1200 would
put random play below zero on USA (−160); 2000 would call an agent a person beat 123-4 an
expert.

**What it is not.** The numbers feel like chess numbers but are not chess strength. TTR
has far more luck than chess, so its spreads are compressed (Mini's most of all), and a
rating means something only within a configuration's pool. The 400-points-is-10:1 scale
was already chess's and is unchanged.

**How.** One constant, `eval.stats.ANCHOR_ELO = 1500`, added after the fit's existing
shift. The `N(0, 400²)` prior acts on the unshifted parameters, so it pulls exactly as
before; every difference, interval width and SPRT bound is unchanged. Absolute ratings
lose their plus sign (CLI and web); differences keep theirs. The leaderboard protocol
carries `anchor_elo` so the client never hardcodes the number.

**Stored ratings.** `results.db` schema 1 → 2. Opening a v1 store shifts every
`rating` row's `elo`/`lo`/`hi` by +1500 in one transaction (`store.MIGRATIONS`). That is
exact, since v1 rows were anchored at 0, and it runs once. A build still on schema 1 then
refuses the migrated store, which is the version guard working as intended. Changing
`ANCHOR_ELO` again is a schema change too, with a migration that shifts by the difference.

---

## 2026-10-18 — H3 moves: the plan follows the cards

**The anchor moved, on purpose.** PLAN.md §11 meant H3 to be the Elo zero for the
project's lifetime, with tuning done as new agents rated against it. H3 was not strong
enough to be that bar, so H3 itself changed instead. Every rating recorded before this
entry is on the old scale.

### Why

A person beat H3 4-123 on USA 2P (seed 1120284448765548802). From mid-game, four of its
five tickets hinged on joining two halves of its network, and two ways of doing it cost
the same 11 cars. The plan's tie went by city order to Duluth-Helena (6 orange) and
El Paso-Oklahoma City (5 yellow). That needed nine more orange, yellow or locomotive
cards, while the 33-card draw pile held eight; the rest sat in the discard pile, which never
came back because the pile never ran dry. The other network (Oklahoma City-Santa Fe,
Santa Fe-El Paso, Denver-Santa Fe, Denver-Helena) was payable from its hand plus two blues
that sat face up for the whole second half. H3 drew blind on 8 of its last 10 turns and
ended holding 12 cards and 12 trains. Not a fluke: in 1,000 H3 self-play games, 44% of
seats ended missing a ticket while holding 10+ cards.

### What changed (H3 and H4; H2 untouched)

- **The plan prices cards** (`plan_card_weight`, 0.01 cars per draw). The edge set is
  chosen with each free segment costing its cars plus the draws until this seat could pay
  for it: face-up cards first at one draw each, then blind at the ratio of unseen cards to
  unseen cards of that colour or locomotives, locomotives in hand counting toward
  everything, a gray route at its cheapest colour. At this weight a car outweighs dozens
  of draws, so in practice cards break ties between equally short networks. `steiner_cost`
  stays the exact car count, so the train budget and every ticket decision are unchanged.
  The edge set follows the cards (`Plan::follow_cards`); it is recomputed only when the
  prices or the board moved.
- **The last turn is the last turn.** Once the final round starts, a draw scores 0,
  tickets are ruled out (`final_turn`), and a claim scores its route points plus twice
  the value of each held ticket it completes, plus the longest-chain tiebreak. The old H3
  valued that turn like any other and drew whenever a blind card out-scored its best route.

### What was measured and dropped

Head-to-head, new vs old H3 in one build: with both changes off, H2-H4 reproduced their
recorded behaviour hashes exactly, so "off" *was* the old anchor.

- **A heavier card weight loses.** At 0.3-0.5 it costs 8-70 Elo: the plan strings short,
  cheap-to-pay routes together (route points 61 to 54 per game, since route points are
  convex in length) and stops contesting the highways its opponent needs. That
  opponent's ticket points rose from 26 to 33.
- **Guessing when the game will end loses.** I tried estimating turns left from the
  fastest opponent's trains at an assumed pace, calling a plan unfinishable when it
  needed more turns than that, then cashing in (cards devalued, trains free). Against old
  H3 at USA 2P, 4,000 games each, each piece was neutral or negative: −1 to −4 for the
  trigger and the turn test, −5 more for devaluing cards, −5 more for free trains. Final-turn
  scoring alone was +18. The asymmetry explains it: abandoning a finishable ticket costs
  twice its points, while failing to spend leftover cards costs a few route points, and any
  estimate of turns left errs on the costly side somewhere.
- **Card-aware planning is a two-player gain.** Alone it is +26 at USA 2P but −12/−21 at
  USA 4P/5P. At more seats H3 gains about a point while each opponent gains 1.5, from the
  face-up row it refreshes for them and the highways it no longer takes. 0.01 is the
  compromise: positive against old H3 everywhere, largest where the web game is played.

New H3 vs old H3, fresh seeds (seed root 5,000,000), every rotation played; the last
column is the share of the new H3's seats that ended missing a ticket while holding 10+
cards (old H3 self-play: 45% at USA 2P):

| configuration | games | Elo vs old H3 [95% CI] | stalled seats |
| --- | --- | --- | --- |
| usa 2P | 10,000 | **+36** [+30, +43] | 21% |
| usa 3P | 9,000 | +20 [+13, +27] | 25% |
| usa 4P | 10,000 | +12 [+6, +19] | 17% |
| usa 5P | 10,000 | +7 [−1, +15] | 17% |
| mini 2P | 10,000 | +12 [+6, +18] | 8% |
| mini 3P | 9,000 | +11 [+5, +17] | 7% |
| mini 4P | 10,000 | +20 [+13, +26] | 5% |

The reviewed game, replayed from the stall with the new H3 against the person's recorded
claims: H3 builds the other network, completes all five tickets and loses 119-123, not
4-123.

### Consequences

- **Every heuristic is a new identity.** H3 `5b906156` → `1e84ef3c`, H4 `1fa8740e` →
  `c8846dd9`, H0-H2 play unchanged. A new field means `PARAMS_VERSION` 1 → 2, which moves
  every agent's `params_hash`, so stored arena results and stored games against any bot
  no longer match the bots this build plays. `ttr serve` refuses to rebuild such a game
  (`replay_mismatch`); re-run the arena rather than mixing scales.
- **Cost.** H3 goes from 10.5 to 17.5 µs per decision at USA 2P (11.8 → 19.3 at 4P, 1.4 →
  2.1 on mini), single-threaded: the price of re-choosing the network as the cards move.
  That is still under the 26 µs once accepted, but the Phase 5 rollout budget is now
  proportionally tighter.
- **Ratings refreshed** (README): H2 is now −156 at USA 2P (was −100), −132 on mini 2P,
  −142 at USA 4P; H4 is level with H3 at 2P and −20 at USA 4P.
- **Tests:** the reviewed position as a regression test (`h3_plans_the_network_its_cards_can_pay_for`);
  `on_its_last_turn_h3_claims_whenever_it_can` over every map and seat count, which the old
  H3 fails on its first game; card-price unit tests; the glossary gained
  `tickets_finished` and `final_turn`.

---

## 2026-10-18 — Phase 4.3 follow-up: train pieces are not cards

Playing H3 on USA (game `48deeca390e6ddff`), the user claimed Helena-Seattle with their last
6 pieces while holding 6 cards, read the cards as "6 trains", and took the final round that
followed -- and a last turn with no claim -- for bugs. A review found the engine and the
final-round rule right: the client let a supply of plastic trains and a hand of cards with
railcars on them share one word. Now:

- **Pieces are labelled as pieces** wherever the supply shows: "6 of 45 train pieces left"
  beside "6 cards in hand" over your hand, "Train pieces" in each seat panel (red at the
  trigger or below, with a tooltip saying they are not cards), "pieces to go" on tickets, and
  the glossary's `trains`.
- **The claim picker says what a claim takes**: "Takes 2 of your 6 train pieces, leaving 4."
  When it leaves 2 or fewer before the final round: "This starts the final round. Uses your
  last 6 train pieces: everyone, you included, gets one more turn, and you will have none
  left to claim with."
- **The prompt names only the moves the server allows**, and when train pieces rule out
  every open route it says why: "No route can be claimed: you have no train pieces left."
- **The log**: "Edwin has no train pieces left: everyone gets one more turn." -- "no", not
  "0", which the body font draws as a small o (see GOTCHAS).
- **The threshold is the engine's**: `board_data` and `Board` carry `end_trigger_trains` (2),
  so the client warns by the engine's constant, not one of its own; a test plays games and
  checks that the final round starts exactly there.
- The status bar no longer clips the speed control when the prompt is long.

Checked in the browser on a copy of the play store with that game cut back to just before
the claim: the warning, the last turn's prompt, the log line and the review read as above,
and the game replays to the same 123-4.

---

## 2026-10-18 — Phase 4.3: the web client

Complete, and merged with 4.1-4.2. `make web` builds a Svelte + TypeScript client that
`ttr serve` serves: a board in the physical game's style with original art, a lobby, play
against any bot, spectating with each decision's reasoning, a review scrubber, a profile and
a leaderboard. The user signed off on the board's look before any screen was built. What is
left of Phase 4 is the user's own games against H3 and H4.

### The board

- **Layout data:** `data/layout/{usa,mini}.toml` -- each city's real latitude and longitude
  (mini's taken from usa.toml), cosmetic nudges, plaque placement, per-route bends and the
  decorations -- and `tools/gen_layout.py` (`make layout`, `--check` in pre-commit and the
  tests), which writes everything the client draws: the frame and score track, city
  positions, **every car space of every track**, and a basemap. The client only draws.
- **Geography:** Natural Earth 5.1.2 (public domain) -- land, lakes, rivers, borders and
  mountain ranges, the last drawn as peak glyphs -- downloaded into a cache outside the repo,
  pinned by SHA-256, clipped, projected and simplified. One Lambert conformal conic per map
  (USA: centre 39 N 96 W, parallels 33 and 45; mini: 41 N 110 W, 35 and 47). The basemap is
  keyed by its inputs, so `--check` is offline and editing a nudge never needs the network.
- **Car spaces flex in length** (34-56 board units): at real geography USA's routes have
  128-363 km per car, so uniform spaces either do not fit the tight routes or leave the long
  ones as dashes. Where a city's routes leave it close together, each starts far enough out
  to clear its neighbours (from the angle between them), rather than overlapping at the hub.
- **The generator checks the drawing** and fails on crowding: car spaces on each other or on
  a city, plaques on a car, a city, each other or off the map, tracks without room. Both
  maps are clean. The bends, nudges and plaque positions were found by a throwaway search
  over that same check (fewest problems, then the smallest offsets and bends), finished by
  hand and reviewed by eye; the search is not kept, the TOML is the data.
- **The look** (original throughout; photos of the published board were a style reference
  only, never saved): an oxblood frame with the 100-square score track, parchment land with
  water-lining along the coasts, routes as rounded car spaces in the card colours with a
  symbol per colour for colour-blind players, doubles as parallel chains, closed halves
  ghosted, cities as ringed dots with plaques. A claimed route is a raised train in the
  owner's colour -- glossy, ribbed, coupled and shadowed -- because a flat drawing has no 3D
  plastic to tell it from unclaimed spaces of that colour. Train cards (a railcar per
  colour, the locomotive on all eight) and tickets (a small map of the board) are original
  art too. Fonts are OFL (Alegreya, Alegreya SC, Alegreya Sans, Rye), bundled.

### The client

- `web/src/api/protocol.ts` is generated from the server's schema (`npm run types`); CI
  regenerates it and fails on a difference. Boards come only from `/api/boards/{map}`, joined
  to the layout by city pair, colour and length, the server's siblings checked -- tested on
  both maps' real payloads, mini's engine order included.
- **`GameSession`** shows a game at the viewer's pace: moves arrive as fast as the bots make
  them and are queued, so the legal actions offered are always for the position on screen.
  A move unanswered for 5 s reconnects for a fresh snapshot (`act` is idempotent on its
  ply). Pacing, acting once per ply, reconnection and the end are unit-tested with fake
  timers.
- **Screens:** the lobby (a name, a game against any bot -- rated with a random seat, or
  unrated with a chosen one -- two bots to watch, games to resume and review); the game
  (the board fitted to the window, the opponent's public seat, the face-up cards and piles,
  your hand and tickets with their remaining cost, the bot-parity panel; claim by clicking
  a route and choosing a payment; draws, ticket draws and keeps offered only as the server's
  legal actions; bot moves animated at a chosen speed; a plain-English log; the score
  breakdown at the end); spectating (both hands and tickets, each decision's reasoning, play,
  pause, step); the review (a scrubber with everything revealed, stepping by move or by bot
  decision, the candidates' terms drawn as bars on one scale, the glossary on hover,
  `?at=N` in the address); the profile and the leaderboard.
- 4,700 lines of Svelte and TypeScript, 35 vitest tests; the production bundle is 135 kB of
  JavaScript (46 kB gzipped) and 39 kB of CSS, plus fonts.

### Integration

Built against a throwaway export of the 4.2 branch (`git archive` into scratch space, its
own environment and play store) until the user said the server was ready; then merged. After
the merge the types regenerated byte for byte and the board payloads matched the fixtures.

| 4.3 exit criterion | result |
| --- | --- |
| a full game on both maps against the real server | USA vs H3 (141 plies) and mini vs H4 (62 plies), rated, through the client's own session |
| review shows every bot decision's breakdown | 68 of 68 decisions (USA), 31 of 31 (mini); one an endgame tie at 0, now labelled a tie |
| spectating runs live | H2 vs H4 on mini, H3 vs H4 on USA, paced, paused and stepped |
| `svelte-check`, `vitest` and the build green in CI | green locally; the CI job is in place but has not run (nothing pushed) |
| the user's sign-off on the look | given 2026-10-17, before the screens |

The full games were played by a script driving the page's session -- the path every button
takes -- not by hand; playing for real is the user's half of the Phase 4 exit.

### Open

- City plaques are about 9 px on a laptop-sized board: legible, small. Enlarging them means
  re-placing them.
- A few long northern routes (Duluth-Helena, Helena-Winnipeg) read as dashed: real geography
  gives them more room than the physical board's distortion would.
- The dev machine's nvm default is Node 22.3, too old to build the client: `nvm install 24`.

### Next

The user plays H3 and H4 on both maps: the Phase 4 exit (lose to H3 at least once), and the
first look at where the heuristics are weak.

---

## 2026-10-17 — Phases 4.1 and 4.2: the engine bindings and the game server

Complete. The heuristics explain their decisions through the scorer that makes them, and
`ttr serve` hosts play, spectating and post-game review over a play store every game
rebuilds from. The Svelte client (4.3) is a parallel session; its integration waits on
this branch being merged.

### 4.1 -- engine additions

- **`explain` is a trace sink, not a second scorer.** `heuristic/trace.rs` defines a
  `Sink` the scorer reports to: each candidate, its additive terms (in points), the facts
  behind them, a rule where one decided, and the total -- the very number the decision
  compared. `act` passes `Silent`, whose empty methods compile away; `explain` passes a
  `Recorder`. Both run one `decide`, so `explain` returns `act`'s action and leaves the
  plan cache and H0's stream exactly as `act` would.
- **H1's cascade is written as a priority** (`rule * 100 + value`, highest wins, ties to the
  lowest id), so its explanation is what it compared rather than a story about it. Same
  rules, same order; its behaviour hash did not move.
- **Bindings:** `Heuristic`, `Explanation`, `Candidate`, `DecodedAction`;
  `Game.decode_action` (from a new `ActionSpace::decode`, which `to_string` now uses);
  `State.offer`, `discard`, `ticket_progress` (the encoder's own per-ticket remaining cost,
  `obs::ticket_costs`), and `board_data(map)`. Two more than the plan listed, so Python
  re-derives no rule: `State.claim_payment` (the canonical payment, for "3 blue + 1
  locomotive") and `State.ranks` (`scoring::competition_ranks`, which the arena now uses
  too). The stub types six existing accessors as the `bytes` they return (GOTCHAS).

| exit criterion | result |
| --- | --- |
| explain's argmax = act, 10k positions, both maps, H1-H4 | 10,000 per (map, tier), 80,000 in all, an `act` agent and an `explain` agent in lockstep: same action at every one, the explanation's first candidate is it, candidates are exactly the legal actions in decision order, terms sum to the total (`tests/explain.rs`, 3.4 s). Reversing the tiebreak or dropping one term fails it |
| every behaviour hash unchanged | H0 `5c62d98f`, H1 `2cd42911`, H2 `f7411bec`, H3 `5b906156` (the anchor), H4 `1fa8740e` -- recorded before the change and now pinned (`no_tier_plays_differently_by_accident`) |
| stub test green | yes, with the four new classes in it |
| no rules, hash or corpus change | `state_hash`, the rules and `OBS_VERSION` untouched; the golden-corpus prefix reproduces |

### 4.2 -- the server

- **`agents/registry.py`** is now the one place a spec becomes a player: `h0`-`h4`,
  `h3@file.toml`, and `ppo:path` parsed with a seam behind it (building one says Phase 6;
  nothing imports torch). `eval/arena.py`'s copy of the parser is gone.
- **`ui/web/`:** `protocol.py` (Pydantic, the source of truth; `protocol.schema.json` checked
  in with `make protocol` / `--check` in pre-commit and the suite), `views.py` (seat,
  spectator and review views; decoded actions; the plain-English log), `store.py`
  (`runs/play.db`, SQLite WAL, versioned, SQL kept to what Postgres also runs),
  `games.py` (the `GameService`), `ratings.py`, `glossary.py`, `app.py` (REST, the game
  socket, the built client or a placeholder). docs/WEB_PROTOCOL.md describes it.
- **The store first, memory second.** A move is checked on a copy, appended to the store,
  and only then made. A game not in memory is rebuilt from seed and moves, each bot asked
  for each of its past decisions and required to give the recorded answer; a bot whose
  identity or answer changed is `replay_mismatch`. A new game is loaded through the same
  rebuild, so there is one way a live game comes to exist.
- **Ratings:** `eval/leaderboard.py`'s fit is split into `tabulate` + `rate`, and
  `fit_with` adds external games -- rated human games, each its own block, a person by
  player id, a bot by its identity -- to a configuration's arena games on the H3 = 0 scale.
  `fit` is bit-identical to before (ratings, intervals and seat scores compared on a
  720-game store against the previous code).
- `ttr serve` (uvicorn; `websockets` joins the `web` extra), `make serve`, CI's fast job
  installs the `web` extra. `ui/terminal/` and `ui/render/` are deleted.

| exit criterion | result |
| --- | --- |
| full games through the API on both maps | USA vs H3 and H4, mini vs H3, with a scripted person; each replays from its revealed seed to the same scores. Also against a real uvicorn server over WebSocket |
| a restart mid-game rebuilds and continues | two restarts mid-game end exactly as the uninterrupted game (actions, scores, final view); a game the store left on a bot's move carries on; a tampered bot move is reported as `replay_mismatch` |
| duplicate and illegal `act` handled | duplicate -> `ack` with `duplicate: true`; `stale_ply`, `future_ply`, `illegal_action` (the engine's reason), `bad_message`, `not_your_turn`, `spectating`, `game_over` |
| masking test, >= 1k positions x seats x maps | 1,000 positions per map from H3-vs-H3 and H3-vs-H0 games, both seats: view and decoded legal moves identical on a resampled infoset; most resamples change the omniscient view, so the comparison is not vacuous. An opponent's hand or the deck's composition in a seat view fails it. Tickets and offers, which the sampler does not resample, are asserted directly, and the log for an opponent's blind draw or keep names no card or ticket |
| no seed before the end | every REST response and socket message of a game is searched for the seed in decimal and hex; it appears first in `over` |
| human games rated on the H3 = 0 scale | rated games put the person on the leaderboard and profile with an interval; unrated ones do not count; with an arena store, H2 is one competitor across arena and human games |

### Decided here, open to objection

- **Protocol departures from the §14.2 sketch** (docs/WEB_PROTOCOL.md lists them): `POST
  /api/games` names its `mode` and needs a `player_id` (`POST /api/players` first); rated
  games refuse a chosen seat; the socket adds `ack` and `error`; seeds are decimal strings;
  a spectated game starts on its first spectator; the board's `layout` is the client's
  file passed through, not re-keyed onto engine segment ids. Extra endpoints: meta,
  glossary, players and profiles, leaderboard, game lists.
- **Opponents are an allowlist** (`ttr serve --agents`, default h0-h4): a spec can name a
  parameter file, and a public server must not read files on request.
- **No interval below two seed blocks.** One block resampled is the same game each time,
  so a one-game rating would show a zero-width interval (GOTCHAS).
- **Not built:** accounts and seat tokens -- anyone holding a play game's id can open its
  socket and play the person's seat. Harmless on a local server, and the first thing the
  public deployment needs.
- `obs_version` is stored with each game but not required on rebuild: the heuristics do
  not read observations. Contract, rules and data hashes are required.

### Next

Merge, then 4.3 integrates its screens against this protocol (a hand-over note is in the
session group's shared notes). Phase 4's exit is the user's: full games against H3 and H4
on both maps, a rating with an interval, and a loss to H3.

---

## 2026-10-17 — Plan revision 2: Phase 4 becomes the web client

At the user's direction, Phase 4 is a web client instead of a terminal one: a terminal is a poor
way to play a game whose state is a map. The user wants to measure the heuristics against
themselves (an average-to-above-average player) and catch degenerate decisions before H3/H4 become
the RL initialization and benchmark, and later to open the client to the public to rate the bots
against people and collect human games as supervision data. PLAN.md §14.2 has the design; the
superseded Phase 4, §12 and Phase 8 text is marked in place.

Decided with the user:

- **Svelte + TypeScript, built by Vite**, served as static files by FastAPI; Node at build time
  only. Replaces §12's "one static HTML file, no npm".
- **Scope:** USA and mini, 2 players, plus spectating bot-vs-bot games with everything visible.
  Not now: 3-5 players, phone layouts.
- **Assistance is bot parity** in every rated game: the UI shows the public information the bots'
  observation holds (known opponent cards, discard and unseen counts, ticket progress and
  remaining cost), so a rating measures strategy rather than memory.
- **Review:** a post-game scrubber with all hidden information revealed and each bot decision's
  candidates broken down by term. Not now: move flagging, same-deal comparisons, live reasoning.
- **The look follows the physical board.** The user asked for it to look like the board game
  itself. It will, in style -- vintage map, coloured car spaces, score track around the edge --
  with original art: the published artwork is copyrighted, so photos of it are a style reference
  only, never committed, embedded or traced, and the public site needs its own name and logo.

Decided here, open to objection: server-authoritative play with no hidden information in the
browser, tested by comparing a seat's view of a state with its view of a resampled infoset; the
play store (`runs/play.db`) as the source of truth, every game rebuilt from seed and actions with
each bot re-asserting its recorded moves; one scorer with an optional trace sink for the
heuristics' explanations, never a second scorer; human games rated in the same Bradley-Terry fit
as the arena's, H3 = 0; a random human seat in rated games, because first-player advantage is real.

Checked while planning: `usa.toml` lists segments in engine order and `mini.toml` does not (the
generator canonicalizes mini), so the client reads the engine's `board_data`, never the TOML.

### Next

Phase 4.1 (engine additions) and 4.2 (server) in one session, 4.3 (client) alongside it in another,
starting with the board's look for the user's sign-off.

---

## 2026-10-17 — Cleanup: `oracle-v1` retired, the Python-migration scaffolding removed

For faster iteration, as asked: less code, one implementation of everything, shorter
checks. Net −4,170 lines (4,780 deleted, 610 added) and two binary fixtures; the full
`#[ignore]` sweep set runs in under 2 minutes (113 s with the build), from 7-10.

### Removed

- **The `oracle-v1` differential** -- the frozen-build git dependency, `tests/oracle.rs`,
  the `Engine` trait and macro that abstracted over two builds, `diff.rs`. It served its
  purpose through the 3.2 rewrite; it already skipped the features `OBS_VERSION` 2 changed
  and would diverge further with every observation change; and its sweep, mostly the old
  build's slow encoder, was most of the verification time.
- **The replay module** (`replay.rs`, a byte-compatible port of Python's replay format),
  its 84 frozen replays, their tests and generator. Nothing used it; the corpus already
  replays 14,000 recorded games from seed + actions, and `VecEnv.finished_games()` is the
  game record when one is needed.
- **The contract vectors** (written by Python as the cross-language contract), their test
  and their byte-for-byte generator. The golden corpus depends on every value they pinned;
  the one independent reference, upstream PCG32's demo vector, was already in
  `tests/rng.rs`.
- **The Python board** (`ticket_to_ride/data/board.py`, `board_gen.py`, `rawmap.py`), a
  second copy of the board data with its own distance tables, its tests, and the Python
  half of `tools/gen_board.py`. `ttr map` reads `ttr_rust.board_summary()` now;
  configuration lists come from `ticket_to_ride.engine.configurations()`. `msgspec` is no
  longer a dependency.
- Python-history comments in module headers, and every reference to the above in CI,
  pre-commit, the Makefile and the docs.

### Simplified

- **The corpus tooling** (`crates/ttr-tools/src/corpus.rs`, 913 → ~560 lines) uses
  `ttr_core` directly, and keeps **one observation digest per game** (format 2) instead of
  discrete and continuous halves, a split that existed only for the cross-implementation
  1e-6 tolerance. Re-signed only after checking every header, chain, score, winner and
  recorded action against the format-1 files: identical in all seven.
- The PPO spike's rollout is its own class (lint, and readability).

### Kept

The golden corpus, the naive in-crate references (`naive_graph`, `naive_legal`,
`naive_obs`), mutation testing, the encoder cache and its check -- the self-checks that do
not depend on a second build.

---

## 2026-10-17 — Phase 3.4: the RL additions to `VecEnv`, and a PPO spike that learns

Complete. A throwaway PPO on TTR-mini 2P beats uniform random 99.2% and H1 99.6% after 2M
learner steps (1,000 evaluation games each, both seatings), with `mask_violations == 0`,
in **21.5 s on MPS** (93.8k learner steps/s; CPU: 25.8 s, 78.2k). It had already passed
both bars at 300k steps.

### Kept: the `VecEnv` additions

- **`assign(env, seats)`**: each seat `"learner"` or `"h0"`..`"h4"`. Scripted seats move
  inside Rust (in parallel across envs) after every `step` and `auto_reset`, so every env
  is finished or waiting on the learner -- Python never loops over opponents. Heuristics
  are re-seeded from each new game's seed; an assignment outlives resets.
- **`returns(out)`**: finished games' returns (`scoring::returns`), zero otherwise; read
  before `auto_reset`. **`final_scores()`** likewise.
- **`ticket_status(out)`**: every seat's tickets as not held / open / complete -- the
  auxiliary targets (my completions at the end; opponents' true tickets at any time; the
  margin comes from `final_scores`).
- **`finished_games()`**: `(env, seed, actions)` per finished game, enough to replay it
  (§9), with `track_history=True`.

### Throwaway: `ticket_to_ride/rl/ppo_spike.py`

MLP 1169 → 256 → 256, masked categorical policy and value head, PPO (2 epochs, 4
minibatches, clip 0.2, entropy 0.01, lr 3e-4), GAE λ 0.95 with γ 1 and the terminal return
as the only reward. 256 envs × 64-step rollouts; learner in seat `i % 2`, opponents H0 and
H1 by env. Evaluation is greedy.

| | run | learner steps/s | vs H0 | vs H1 | mask violations |
| --- | --- | --- | --- | --- | --- |
| MPS | 2M steps, 21.5 s | 93.8k | 0.992 | 0.996 | 0 |
| CPU | 2M steps, 25.8 s | 78.2k | 0.998 | 0.988 | 0 |

At this network size MPS is only 1.2x the CPU: the loop is bound by per-step overheads
(small batches through Python and torch), not by the environment, which runs at ~1M
env-steps/s. Phase 6's bigger nets and batches are where MPS should earn its keep.

### Two bugs the spike found

- **`terminal_flags()` returns `bytes`**, not a list: PyO3 hands `Vec<u8>` back as
  `bytes`, and `np.array(b"...", dtype=bool)` is one scalar `True` -- every env looked
  finished on every step, so the first runs "trained" on nothing and evaluated to exactly
  0.5. The stub claimed `list[int]`; it says `bytes` now (also `current_players`). (GOTCHAS.)
- **`Categorical.sample()` on MPS returned a masked action** -- an illegal move the engine
  refused (`IllegalAction`). Sampling is Gumbel-max over the masked logits now, which cannot
  pick a -1e9 logit on any device; `mask_violations` is counted and 0.

### Exit criteria

| criterion | result |
| --- | --- |
| beats random ≥95% within 2M steps | 99.2% at 2M (MPS); both bars passed by 300k |
| beats H1 on mini | 99.6% |
| `mask_violations == 0` | 0 over every run |
| steps/s measured | MPS 93.8k, CPU 78.2k learner steps/s; Modal GPU not measured (it waits on the spend-limit question, 3.3) |

What it does not show: anything about H3/H4, or about USA. H1 is a weak bar (−545 Elo on
mini), and mini is small; Phase 6 is where strength gets measured.

### Next

Phase 6 is re-planned from these numbers (PLAN.md §14.1): the environment is not the
constraint, PPO learns on this observation quickly at mini scale, and MPS needs larger
batches than the spike's to beat the CPU by much.

---

## 2026-10-17 — Phase 3.4 re-plan note (before starting)

PLAN.md asks for this first. What 3.1–3.3 changed for 3.4, and whether it changes 3.4:

- **The environment is no longer the cost.** ~1M env-steps/s (USA 2P, 1024 envs), 1.9M on
  mini; a small MLP's forward and backward on MPS will bound the spike's steps/s, so what it
  measures is the training loop, which is what Phase 6 needs to know.
- **Scripted opponents are nearly free**: H1 0.17 µs and H3 1.4 µs per decision on mini, so
  stepping them inside Rust until the learner's turn costs nothing next to a network call.
- **The features are fixed** (`OBS_VERSION` 2), so the spike trains on plan features that
  mean what they say.
- **Modal works for CPU; the GPU measurement waits** on the spend-limit question, so steps/s
  is measured on MPS (and CPU) here, and the Modal GPU figure is left for the user to unlock.

None of this changes what 3.4 is, so it proceeds as planned: the four `VecEnv` additions
(kept), then a throwaway PPO on TTR-mini 2P, run locally. Exit: beats H0 (uniform random)
≥95% within 2M steps, beats H1 on mini, `mask_violations == 0`, steps/s measured. 3.4
touches neither the engine nor the encoder, so `oracle-v1` sweeps do not apply; its tests
are its own.

---

## 2026-10-17 — `OBS_VERSION` 2: the plan walk and two quirks, fixed

The named bump agreed at the end of 3.1: after the bit-exact 3.2, before 3.4, so the PPO
spike trains on features that mean what their docs say. The layout is unchanged (same
slots, same layout digest); three features' values changed.

- **The plan walk** (`on_my_steiner_tree`, `fragility`). Version 1 recovered a ticket's
  route greedily from the far end with my own track at weight 0; at a city whose neighbour
  across my track had the same distance it could step back across it, ping-pong until its
  guard ran out, and return a truncated route -- 43-44% of open-ticket plans in H3 games
  on the USA map, 23% on mini (found in 3.1, `tests/obs_plan.rs`). Version 2 searches
  breadth-first back from the far end over *tight* segments (`dist[near] + cost ==
  dist[far]`): a cheapest route with the fewest segments, ties to search order. It cannot
  loop, and it reads the same cost table the distances came from (version 1 re-derived
  weights from owners and ignored the double-route rule).
- **`extends_my_chain`**. Version 1 read `dsu_root[c] != c`, which is true only for the
  cities of my components other than each component's root (its smallest city), so a
  segment at a root city never counted. Version 2: a city any of my segments reaches. That
  is a function of the board, so it moved into the cached `Derived`, and the encoder no
  longer reads the union-find at all -- **`encode` takes `&State` now**.
- **Own-hand counts**. Version 1 divided every card type by the colour count (12),
  locomotives (14 printed) included; version 2 divides each by its own printed count.
  *Not in the original agreement*: it is the third version-1 quirk `naive_obs` had been
  reproducing, folded in because the bump was happening anyway. Recorded here so it can be
  objected to.

### How it was checked

- **`naive_obs` covers every feature now.** The chain and hand features from scratch, as
  before. For the plan and fragility it takes the encoder's routes
  (`obs::ticket_route`) and checks everything promised of them: a path between the
  ticket's cities, costing exactly the Bellman-Ford distance, with the fewest segments of
  any cheapest route (Bellman-Ford over (cost, segments) pairs); the plan is the union of
  their free segments; fragility is recomputed by Bellman-Ford with each free segment
  removed. Version 1 could not be checked this way -- its routes were not cheapest routes.
  Its sweep (60 seeds, every configuration, random and H3, every third step, every seat)
  passes.
- **`tests/obs_plan.rs`**: the test that stated the right behaviour, ignored since 3.1,
  passes; `plans_survive_a_real_game` asserts the converse it could not before (an open
  ticket always has a plan).
- **`oracle-v1`**: live and frozen builds compared at every step as before, every slot
  except `CHANGED_SINCE_ORACLE` (ttr-tools). A guard test fails if a skipped feature never
  differs from the oracle (the list would be stale) or an unskipped slot ever does: all
  four changes are observable, and nothing else moved. Full sweep green (380 s).
- **The corpus**, re-signed in two steps, now the documented procedure (CONTRACT.md §4.1):
  `gen-corpus --check --play-only` (new) first -- all 71,400 games' play, scores and hash
  chains reproduce under version 2 -- then `--write`. About 11,150 bytes changed per
  file: the version field and the 700 games' observation digests (700 x 16 bytes, less
  the bytes that coincide), nothing else. `oracle-v1`'s corpus check covers play only now.
- **Mutation testing** over the changed functions: 144 mutants, 126 caught, 11 timeouts,
  4 unviable, 3 missed -- the pinned `segment_dynamic` equivalent, and two pieces of
  redundant code (a bound check a blocked segment could never pass, `|` on disjoint bits),
  simplified away; their re-run, 50 mutants, missed none. A version-1 pin became a real gap
  (`obs_version() -> 1` survives once the version is 2) and is now a test.

### Cost

Single-threaded µs per observation, version 1 (end of 3.2) → version 2, cold / cached:
usa 2P random 8.9 → 9.5 / 4.6 → 5.1; usa 4P random 9.5 → 10.0 / 5.9 → 6.4; usa 2P H3
6.5 → 6.5 / 3.3 → 3.3; usa 4P H3 7.9 → 7.8 / 4.8 → 5.8; mini 2P H3 1.5 → 2.1 / 0.8 →
1.2. Most of the increase is honest work version 1 skipped: a truncated route has fewer
free segments for fragility to re-price. Still well inside 3.2's exit criteria.

### Next

Phase 3.4, starting with its re-plan note.

---

## 2026-10-17 — Phase 3.3: Modal, proven with smoke tests

Complete, within the brief's scope: cheap smoke tests only, experiments stay local. The
image builds the engine; `ttr bench` and three arenas ran on Modal; **12,800 arena games
on Linux x86 reproduced the macOS ARM runs game for game**; results land in the local
store. **Everything cost $0.0053** by Modal's own billing report. The optional GPU check
was not run (it needs the spend-limit question first; below).

### What landed

- **`infra/modal_app.py`.** Debian slim with Modal's Python 3.14 (its 2025.06 image
  builder supports 3.14, so uv-managed Python was not needed), Rust 1.97.1 and uv 0.11.17,
  both pinned. The project goes in as two layers -- the Cargo workspace and crates, then
  `uv sync --no-install-project` (which builds `ttr_rust` in the release profile, 19.5 s on
  Modal's builder); then the Python package and `uv sync`, which finds the extension
  current -- so a Python edit does not rebuild Rust. A whole image builds in about a
  minute. The project lives in its own venv: `uv sync` is exact, and syncing into the
  interpreter Modal runs the function on would uninstall Modal's client.
- **One job function**, `run(argv, job)`: `ttr <argv>` in `/runs/<job>/` on the
  `ttr-runs` Volume, saving stdout, stderr and a `job.json` of where it ran (CPUs visible,
  CPU quota, the engine's thread count, platform). PLAN.md listed functions for `bench`,
  `arena` and a generic `run`; the first two are `ttr cloud` subcommands over the third,
  one implementation.
- **`ttr cloud bench|arena|run|pull`** (`ticket_to_ride/cli/cmd_cloud.py`), running the
  app ephemerally: nothing is deployed and nothing is left running. Each command pulls its
  own job when it finishes; `pull` fetches the rest. **Every job writes its own files**;
  an arena's `results.db` is merged into the local store by `Store.merge`, which re-keys
  configurations and agents by their natural keys (an agent by its content address), so
  cloud games count towards the same players as local ones. Markers make pulls resumable
  and merges happen once. `modal` is imported only inside the commands -- `ttr --help`
  stays instant, and nothing else needs the `cloud` group.
- **`--check-local`**, the exit check made a command: replay the arena locally into a
  scratch database (no game counted twice) and require `Store.outcomes()` -- every seat of
  every game, final state hash, scores and ranks, free of ids and timings -- to be equal.
- **`device`** (`ticket_to_ride/config/device.py`): `auto|cuda|mps|cpu`; `auto` takes the
  best accelerator present, and naming an absent one is an error, not a fallback. `auto`
  resolves to `mps` here. 3.4's training config carries the setting.
- **`TTR_THREADS`**: a positive integer overrides `performance_threads()`. Cloud jobs set
  it to the CPUs their container shows.

### The container check PLAN.md asked for

`performance_threads()` off macOS is `available_parallelism() - 2`. On Modal:

| function config | CPUs the container shows | cgroup quota visible | engine threads |
| --- | --- | --- | --- |
| `cpu=8.0` (request only) | 24 | none | 22 |
| `cpu=(8.0, 8.0)` (request and limit) | **8** | none | 6, then 8 with `TTR_THREADS` |

So `available_parallelism` does follow a container's CPU limit -- gVisor presents the limit
as the CPU count rather than as a cgroup quota -- but Modal's bare `cpu=` is a reservation
with no limit, under which the container shows the host slice. `ttr cloud` therefore
always sets request = limit (cost and thread count no longer depend on the neighbours),
and the job sets `TTR_THREADS` to the visible count, since the desktop default's two
spare CPUs are a quarter of a dedicated 8-core container. (GOTCHAS.)

### Smoke tests

| job | what | container time |
| --- | --- | --- |
| first bench | image build + `ttr bench --suite engine --games 2000` (uncapped, 22 threads) | 0.8 s |
| capped bench | the same at `cpu=(8, 8)` | 1.0 s |
| arena 1 | h0-h4 round robin, USA 2P, 100 blocks: **2,000 games identical** to local | 1.4 s |
| arena 2 | h0-h4, USA 2P, 500 blocks: **10,000 games identical** (PLAN.md's 10k) | 3.3 s |
| arena 3 | h0-h4, USA 4P, 40 blocks: **800 games identical** | 0.9 s |
| final bench | `--games 4000 --repeat 3`, 8 threads | 2.4 s |

Throughput in a capped 8-CPU container (x86, gVisor): USA 2P one thread 34k games/s
(0.19 µs/step; the M2 Max does ~45k), eight threads 169k games/s; mini 2P 102k and 464k.
The merged store refits to the same ladder as the README's (h4 −5, h2 −96, h1 −699, h0
−1181 over the 12,800 games).

**Cost.** Modal's billing report (`modal.Workspace().billing.report(...)`, hourly):

| app runs | cost |
| --- | --- |
| the first bench and the first arena, each of which built image layers | $0.0020, $0.0019 |
| the other four (a bench or an arena each, 1-3 s of container time) | $0.0002-0.0005 each |
| **all six** | **$0.0053** |

So a smoke test is a fraction of a cent, and an image rebuild a fifth of one; the $30
monthly Starter credit would cover this whole set of six about 5,700 times.

### Not done, and why

- **The GPU throughput check** (optional in the brief). Before any GPU run the brief asks
  whether a workspace spend limit is set in the Modal dashboard; that is the user's setting
  and I have not asked yet. `ttr cloud run --gpu T4 -- …` is wired; the image has no CUDA
  torch yet (3.4's PPO runs locally, on MPS).
- **The spend limit** itself is a user-only step (PLAN.md §14.1); not verified from here.

### Decisions

- **Ephemeral apps, not a deployment**: one command, one app run, nothing idle to bill.
- **Request = limit** for every job (above).
- **Merge by natural keys, not by copying the file**: two stores are never the same
  database, and `record_match` already gives a match-per-transaction write.

### Next

The `OBS_VERSION` 2 bump (the plan walk and `extends_my_chain`), then Phase 3.4, which
starts with its re-plan note.

---

## 2026-10-16 — Phase 3.2: the observation encoder, fourteen times cheaper and bit for bit

Complete. One observation costs **8.9 µs cold and 4.6 µs with the encoder cache** (USA 2P,
random-play positions, one thread), from 126.9; the training loop's environment side
(observe + mask + step + reset) runs **981k env-steps/s at 1024 environments**, from 52k.
Exit criteria were ≤15 µs and ≥250k. Nothing moved: every observation is bit-identical to
`oracle-v1`'s at every step -- continuous fields included, so the 1e-6 allowance was never
used -- state hashes and every heuristic's choices agree, and H3's behaviour hash is still
`5b906156d4b1204b`. 410 Rust tests (13 behind `#[ignore]`); ttr-core line coverage 96.4%,
the naive references aside, as in 3.1.

### Measured first

The brief's loop -- `observe_current -> legal_masks -> random masked pick in numpy -> step
-> auto_reset` for 2 s -- became `ttr bench --suite vecenv` (616af7d) before anything
changed, with phases timed separately and a `make bench-check` 20% gate. Its first reading
matched the brief's diagnosis exactly: at 1024 USA 2P environments observe was 19.1 µs of
a 19.3 µs env-step, mask 0.13, step 0.04.

`bench-obs` (ttr-tools) records positions from whole games -- uniformly random, or H3
self-play -- and encodes each for the seat to act, single-threaded. samply on it, with the
encoder's internals un-inlined for attribution:

| | share of an observation |
| --- | --- |
| `steiner_cost_exact` | 75% -- of which the DP's per-mask multi-source Dijkstra 53% |
| `fragility` (a Dijkstra per free route segment) | 10% |
| `longest_trail` | 6% |
| allocation (`HashMap<u8, Vec<u16>>` per call, `Vec`s) | ~10%, spread across the above |

Why the DP dominated: random play holds a lot of tickets. Of USA 2P random positions, 42%
have nine or more Steiner roots (the MST fallback), and 12%, 10% and 11% have eight, seven
and six -- the subset DP at its widest. H3 positions mostly have two to six.

### What changed, in the order the profile asked for it

All in `graph.rs` and `obs.rs`; µs/obs on USA 2P random positions (the first four steps in
the original harness, which also timed a checksum -- see below):

| step | µs/obs |
| --- | --- |
| baseline | 125.0 |
| one all-pairs table per observation; the DP's relaxation a min-plus pass over it | 47.2 |
| a fixed-size view: no `HashMap`, no `Vec`, the plan a `u128` | 30.5 |
| Dial's bucket-queue Dijkstra; `longest_trail` over arrays and bitmasks | 16.4 |
| the Steiner DP in `u16`, each split visited once, no relaxation of the last row | 14.3 |
| fragility by A*, guided by the table | 11.5 |

- **`all_pairs`**: Floyd–Warshall over the seat's crossing costs, a fixed-width `u16`
  min-plus that compiles to SIMD (rows padded to `DIST_STRIDE` = 40). Ticket costs, the
  DP's terminal rows, contraction, the MST bound, the planned routes and fragility's guide
  all read it. The DP relaxation `min over u of row[u] + d(u, v)` is equivalent to the
  multi-source Dijkstra it replaces because every DP cell stays at or below the sentinel
  and the table holds exact distances (the argument is in `steiner_cost_with`'s doc).
- **Dijkstra** is a bucket queue over a ring of `max_len + 1` city bitmasks -- crossing
  costs are integers 0..=6 -- with zero-cost own track re-draining the current bucket.
- **Fragility** re-prices the ticket's route with each free segment priced out, as before,
  but by A* toward the far end with the unpriced table as the heuristic: distances over a
  table that one closure only raises are a consistent lower bound, so the first time the
  target settles its distance is exact. The ring is `2 * max_len + 1` wide, since a step
  raises `g + h` by up to twice a segment's cost.
- **`longest_trail`**: adjacency in arrays, components by bitmask flood fill, per-component
  edge bits, and the DFS memo a thread-local map with a Fibonacci hash instead of SipHash
  (SipHash alone had been a tenth of an observation).
- **The Steiner DP** in `u16` (every cell is at most the sentinel, so a sum of two fits),
  enumerating each split once (the half holding the lowest terminal), and skipping the
  last row's relaxation, which cannot lower its minimum. That last one is most of the win
  for the 2-4-root plans H3 positions have.

Every one of these computes an exact, unique value -- a shortest distance, the minimum
Steiner cost, an MST weight, a longest trail -- so none depends on search order.

### The encoder cache

Nearly all of an observation is a function of the board and the observing seat's tickets
alone: the table, the Steiner DP, the plan, fragility, the opponents' trails. That work is
now `Derived`, and `EncoderCache` keeps it per seat; `VecEnv` holds one cache per
environment. **The key is the inputs themselves** -- the board's identity, seat count,
double-route rule, every segment's owner (a 128-byte compare) and the seat's ticket mask --
not `board_version`. A counter-keyed cache is correct only while every path that changes
its inputs remembers to bump the counter; a content key cannot go stale, so resets,
replaced states and reused caches need no invalidation at all, and a hit is equal to
recomputing by construction.

Hit rates, by measurement: 55% (USA 2P random), 45% (USA 4P random), 62% (USA 2P H3),
48% (USA 4P H3), 62-64% (mini). Card draws are about half of all steps, as the plan said.
Debug builds recompute on every hit and assert equality; `tests/encoder_cache.rs` compares
cached and recomputed buffers bit for bit over 10k games -- every configuration, one game
in five H3 self-play, every seat encoded every third position, one cache reused across 25
consecutive games of different configurations -- in 51 s under the test profile.

### Results

`bench-obs`, single-threaded µs per observation; all three columns agree on the checksum:

| positions | before | cold | cached (hit rate) |
| --- | --- | --- | --- |
| usa 2P random | 126.9 | **8.9** | **4.6** (55%) |
| usa 4P random | 149.2 | 9.5 | 5.9 (45%) |
| mini 2P random | 37.3 | 3.4 | 1.6 (62%) |
| usa 2P H3 | 93.8 | 6.5 | 3.3 (62%) |
| usa 4P H3 | 128.0 | 7.9 | 4.8 (48%) |
| mini 2P H3 | 7.0 | 1.5 | 0.8 (64%) |

`ttr bench --suite vecenv`, µs of wall time per env-step, both builds measured back to back:

| map | envs | observe | mask | step | env-steps/s before | after |
| --- | --- | --- | --- | --- | --- | --- |
| usa 2P | 1 | 135.4 → 5.3 | 0.5 | 0.2 | 7,300 | 168,000 |
| usa 2P | 64 | 26.5 → 2.7 | 0.8 | 0.1 | 36,000 | 277,000 |
| usa 2P | 1024 | 19.1 → **0.85** | 0.12 | 0.04 | 52,000 | **981,000** |
| mini 2P | 1024 | 6.6 → 0.37 | 0.09 | 0.04 | 147,000 | 1,910,000 |

The loop including the numpy pick is now 264k steps/s at 1024 USA environments, and the
pick (2.8 µs) is most of it -- the environment is no longer the expensive side of a step.

After, a cold observation splits: Steiner DP 30%, fragility 26% (A* 18, the route walk 8),
`all_pairs` 15%, the planned routes 7%, the segment block 7%, trails 6%.

### By-product: every heuristic is cheaper

The heuristics share `graph.rs`. `ttr bench --suite agents`, µs per decision, single-
threaded, the pre-3.2 build and this one back to back:

| map | seats | h0 | h1 | h2 | h3 | h4 |
| --- | --- | --- | --- | --- | --- | --- |
| usa | 2 | 1.03 → 0.40 | 1.02 → 0.35 | 104.3 → 11.1 | **83.7 → 10.6** | 86.9 → 14.5 |
| usa | 4 | 0.58 → 0.22 | 0.62 → 0.19 | 107.1 → 12.2 | 93.1 → 12.0 | 101.5 → 17.1 |
| mini | 2 | 0.28 → 0.15 | 0.34 → 0.17 | 3.54 → 1.21 | 3.83 → 1.37 | 4.02 → 1.66 |

This moves the Phase 3 finding that **H3 is too slow to be the ISMCTS rollout policy**:
1000 simulations of ~100 decisions were ~7.6 s per move, and are now ~1.1 s. §7.2's S2 tier
(800-2000 iterations) is back within reach of H3 rollouts. Phase 5 should re-measure
rather than inherit either number.

### How "nothing moved" was checked

- **`oracle-v1`, full sweep**: 70k random games and 3,500 heuristic games, every step
  compared -- state, position hash, legal actions, the observation (every seat every 7
  steps), terminal settlement, every tier's choice. Zero divergences, 356 s (714 s before:
  the live side's encoder is most of a step). `diff::CONTINUOUS_TOLERANCE` stays 0.0.
- **The golden corpus** reproduces in full on both builds (26 s, from 44).
- **H3's anchor**, `the_h3_anchor_has_not_moved`, unchanged.
- **Naive references**: `naive_graph` now also checks `all_pairs` row by row and
  `guided_distance` between every pair of cities with each seventh crossable segment priced
  out, against Floyd–Warshall without that segment, on random and real-game layouts; the
  sweep passes (71 s).
- **The cache**: as above.
- **Mutation testing**, below.

### Mutation testing

`cargo mutants` over `graph.rs` and `obs.rs`, the two files this phase rewrote, against
ttr-core's own tests: 554 mutants in 34 min -- 473 caught, 33 timeouts, 12 unviable,
**36 missed**. Triaged:

- **One real gap.** No ttr-core test noticed fragility re-pricing *my own* track on the
  route (only the oracle differential, in ttr-tools, would have). An enemy cannot claim my
  track; `fragility_never_prices_out_my_own_track` (`tests/obs_plan.rs`) now searches for
  positions where counting it would move the feature.
- **21 pointed at dead code, deleted rather than excluded**: both searches' `settled`
  masks (8 mutants) -- the stale-distance check already rejects everything they did, since
  only a strict improvement re-queues a city; the hand-rolled memo hasher (8, all
  speed-only), replaced by `rustc-hash`'s `FxHashMap`; fragility's early return and
  capacity guard (4), which its only caller and the walk's own bound make unreachable; and
  an unread bound on the seat-order loop (2). With `settled` gone, the two `<` → `<=`
  mutants in the searches loop forever on zero-cost own track instead of surviving.
- **11 equivalent, pinned** in `.cargo/mutants.toml` with the reason for each: speed-only
  shortcuts (the Eulerian one, `dsu_union`'s unreachable tie), `|` versus `^` on bits
  known disjoint, ties in `mst_bound`, overlapping DP splits that can only add upper
  bounds, a priced-out edge that can never improve a distance, and `obs_version` while it
  is 1. The 3.1 pins were all stale (their lines moved) and came back as missed, as
  intended.

A re-run over every edited function: 129 mutants, 104 caught, 17 timeouts, 3 unviable,
5 missed -- exactly the pinned equivalents among them.

### Decisions

- **Content keys, not `board_version`**, for the cache (above). The key compare is ~150
  bytes per lookup, nothing next to what a hit saves.
- **`f64` arithmetic stays.** PLAN.md listed "f32 where f64 only existed to match Python";
  the profile shows narrowing is not where the time goes, and the corpus and `oracle-v1`
  were signed under f64-then-narrow. Changing it would spend the continuous tolerance for
  no measured gain.
- **No incremental all-pairs.** A claim by me only lowers costs (an O(n²) update), but an
  opponent's raises them, which needs a recompute; the cache already skips every position
  where nothing changed, and Floyd–Warshall is 15% of a cold observation.
- **The rayon pool stays global.** A pool sized to the 8 performance cores measured the
  same as the global 12 threads at 1024 environments (0.94 vs 0.86 µs observe) and slightly
  better at 64 (2.26 vs 2.72): within noise for what training needs. PLAN §8.3's caveat (macOS
  cannot pin threads) is unchanged.
- **The 15 µs criterion is read on the cold figure**, which is the conservative one: the
  cached figure depends on the policy's draw rate.
- **`bench-obs` timed its own checksum.** Summing an observation's 3,355 floats in `f64` is
  a serial chain of additions -- about 3 µs, a third of what it was timing once the encoder
  got fast. It now times only the encoder calls; the before/after table above was measured
  with the corrected harness on both builds. (GOTCHAS.)
- Removed, being unused: `distance_between` (A* replaced it) and Dijkstra's early-exit
  target. One search per purpose.
- **`rustc-hash` is a new dependency of ttr-core**, for the trail memo. A ten-line hasher
  of our own was the alternative, and every mutant of it survived -- hash quality is not
  observable, only speed.

### Deviations from PLAN.md

- "Cache the Steiner tree and `remaining_cost` per seat": done, and wider -- every piece of
  board-derived encoder work is cached, keyed by content rather than invalidated on claims.
- "f32 where f64 only existed to match Python": not done, deliberately (above).
- "Incremental connectivity, union-find updated on claim": already the case since Phase 2;
  the encoder reads roots, it never rebuilt them.

### Next

Phase 3.3 (Modal), which does not touch the encoder; then the `OBS_VERSION` 2 bump agreed
at the end of 3.1 -- the plan walk and `extends_my_chain` -- as its own commit, so that 3.4
trains on the fixed features.

---

## 2026-10-16 — Phase 3.1: the Rust engine becomes its own oracle

Complete. The Python engine, encoder and differential harness are deleted (about 8,200
lines of Python and tests); the Rust engine is checked against a golden corpus the Python
engine signed, a frozen build of itself, naive in-crate references and mutation testing.
406 Rust tests (12 behind `#[ignore]`), 129 Python tests, `cargo test --workspace` in 26 s.
ttr-core line coverage 96.4% (from 90.3%), and no surviving mutant in the rules or scoring.

### The order, kept

1. **The freeze point was checked first.** The full Python/Rust differential -- 100k seeds
   x every (map, seat count) compared at every step, the 84 golden replays, the encoder
   bitwise -- went green on 7923e28: 904 tests, 8m17s on 12 cores.
2. **The golden corpus, Python's last word** (`tests/golden/corpus/`, one file per
   configuration, 1.8 MB). Per configuration: 10k random-drive games with final scores,
   winners and a blake2b-64 hash chain over `state_hash`, `position_hash` and the sorted
   legal actions at every position plus the terminal settlement; observation digests for
   500 of them; and 200 H3 self-play games with their actions. 71,400 games, 13.5M random
   steps, 240k H3 steps, 504k observation vectors. Generated by the frozen build and
   **recomputed by the Python engine and encoder**: all 71,400 agree (210 s on 12 workers).
   The H3 games were the first time the encoder had been checked against Python on
   positions a competent player reaches.
3. **`oracle-v1` = 7923e28**, tagged locally, a git dev-dependency of `crates/ttr-tools`
   pinned by `rev`. The new-versus-frozen differential compares state hash, position hash,
   legal actions, the acting seat's observation at every step (every seat every 7 steps),
   the terminal settlement, and -- in heuristic drives -- every tier's choice at every
   decision. Full sweep: 70k random games and 3,500 heuristic games, zero divergences;
   the whole corpus reproduces on both builds (44 s).
4. **The tests ported**, one to one (table below), by six agents working in parallel
   worktrees on disjoint files over a shared rig (`crates/ttr-core/tests/common/`). None
   failed when ported.
5. **Naive references inside the crate**, `#[cfg(test)]`: `naive_graph` (Floyd–Warshall,
   brute-force Steiner on mini, exhaustive longest trail), `naive_legal` (a 6300-candidate
   claim scan from the rules, every action id stepped, every cached field recomputed:
   14k positions by default, 288k in the sweep), and `naive_obs` (each feature one line from
   the state; written during mutation triage, below).
6. **Mutation testing**, then the deletion, then packaging and docs.

### Mutation testing

`cargo mutants` over `state.rs`, `scoring.rs`, `graph.rs` and `obs.rs`, against
**ttr-core's own tests** -- deliberately not the oracle differential or the corpus, which
catch nearly anything by comparison and stop applying the day `OBS_VERSION` moves.

| pass | mutants | caught | timeout | unviable | missed |
| --- | --- | --- | --- | --- | --- |
| 1 | 949 (70 min) | 796 | 25 | 17 | 111: state 9, scoring 15, graph 12, obs 75 |
| 2, after round-one tests | 136 | 91 | 24 | -- | 21 |
| 3, after round-two tests | 45 | 3 | 24 | -- | 18, then 16 |
| final, the code as committed | 929 (57 min) | 887 | 26 | 16 | **0** |

Timeouts are mutants that make a step do nothing, so a game never ends: caught. The 16
left are equivalent -- unreachable branches, speed-only changes, ties -- and are recorded
in `.cargo/mutants.toml` with the reason for each, pinned to file:line:col so they come
back for re-triage when the code moves. The final run, on the code as committed (after the
deletion and the spec move), misses nothing. **Nothing survives in `state.rs` or
`scoring.rs`, or anywhere else.**

What the survivors said, and what closed them:

- **3-5P returns had no test of their values.** The ports checked constant sum and that the
  winner gets most, which a win share multiplied instead of divided keeps. `returns.rs`
  pins them to hand-computed values.
- **`validate()` could always return `Ok`.** Every test called it on good positions.
  `validate.rs` breaks each invariant on purpose.
- **Nothing stated what `position_hash` is.** `hashing.rs` rebuilds the CONTRACT.md §3
  image field by field, sharing no code with `serialize`.
- **Most of the encoder was checked only by comparison with a frozen copy** -- replacing
  `own_hand` with nothing survived. `naive_obs` compares every feature bit for bit with a
  one-line recomputation, for every seat through random and H3 games on every
  configuration; `obs_plan.rs` pins the plan features, fragility and the terminal set.
- Two rule edges: the flush guard with locomotives in the *discard*, and the forced
  opening keep (unreachable with real boards, so rigged).

### What the new checks found

- **An `OBS_VERSION` 1 encoder defect, large.** The walk that recovers a planned route gives
  my own track weight 0, so it can step back and forth across one of my segments until its
  guard expires and return a truncated route: **43-44% of open-ticket plans in H3 games on
  the USA map, 23% on mini.** `on_my_steiner_tree` and `fragility` both read it. The
  retired Python encoder had it too, so it is part of v1 as signed, and fixing it is a
  named `OBS_VERSION` bump -- not something 3.1 or 3.2 may change. The test stating the
  right behaviour is `#[ignore]`d with the defect as its reason.
- **A second, smaller one:** `extends_my_chain` reads "touches my network" as "is not its
  component's union-find root", so a free segment at the smallest-index city of my network
  reads as not extending it. Also inherited; reproduced and labelled in `naive_obs`.
- **`longest_trail` numbered its mask bits across the whole network**, so a seat owning
  more than 32 segments overflowed the shift even when every component fit. Found by the
  exhaustive reference; unreachable in play (at most 27 USA segments); fixed.
- **Two Python tests could not fail for the reason they were written:** the flush-guard
  test passed with the guard deleted, and the sweep's flush detector compared face-up
  slots with `FREE` instead of `EMPTY_SLOT`. The Rust ports check what was meant.
- **`ttr_rust.CONTRACT_VERSION` was declared in the stub and never defined**, since Phase 2.
  It type checked and failed at import the first time anything used it.

### Decisions

- **`rev`, not `tag`.** The plan says `tag = "oracle-v1"`. A SHA cannot be moved by a
  force-push, and it resolves from the public repository whether or not the tag has been
  pushed -- so CI and Modal need nothing from the user. The tag exists locally as the
  human-readable name.
- **`crates/ttr-tools`**, a crate that is never shipped, holds the corpus format and drives,
  the oracle differential and the generator bins (`gen-corpus`, `gen-vectors`,
  `gen-replays`), so the engine crate stays an engine and only the tooling crate
  dev-depends on the oracle.
- **The corpus goes beyond the plan in two ways:** H3 games, because random play leaves the
  features 3.2 rewrites nearly inert; and observation digests in two halves, flags and
  ratios, so 3.2's "flags bit-exact, ratios within 1e-6" contract has no gap.
- **The observation spec is hand-maintained Rust** (`obs_spec.rs`): the declarative table,
  symbolic widths, a kind and a doc string per field. Its layout is pinned to the v1
  digest, so the move is proven layout-preserving, and Python reads it through
  `ttr_rust.obs_spec()`.
- **`ttr_rust` is a uv workspace member built by `uv sync`.** The trap, verified: uv
  rebuilds a local package only when its cache keys change, and those default to Python
  metadata -- a Rust edit left the old extension installed. The keys now list every Rust
  input.
- **The test profile builds at opt-level 1**: `cargo test -p ttr-core` went from ~145 s to
  ~20 s, which is what made mutation testing a matter of hours.
- **`data/board.py` stays**, as the plan says, with its derived tables -- now a second
  derivation of what `board.rs` computes, used only by `ttr map`. Flagged in its docstring
  as a candidate for deletion once the UI decides what it reads.

### Deviations from PLAN.md

- **"Both chance modes"** is sampled only: explicit mode was never specified by the
  contract (CONTRACT.md §6), the same amendment Phase 2 recorded.
- **The oracle is pinned by `rev`**, and the `oracle-v1` tag is local; pushing it is the
  user's call.
- The corpus extensions and `ttr-tools` above.
- **Coverage is measured with rustup's 1.87 toolchain** (Homebrew's Rust lacks
  `llvm-tools`), within the workspace's 1.85 floor.

### Every Python test, and where it went

The rule for "ported": the Rust test has the Python name minus `test_`, in the file named,
and asserts the same thing (often more). Exceptions are listed per file. 284 Python test
functions in the files the deletion touches: **225 ported, 22 covered by an existing Rust
test, 13 stay in Python, 24 dropped** -- and every drop is a Python-only API or code that
was deleted with the engine.

| Python file | Rust home | ported | covered | stays | dropped |
| --- | --- | --- | --- | --- | --- |
| `unit/engine/test_rules.py` (67) | `tests/rules.rs` | 64 | 3 | | |
| `property/test_engine_properties.py` (27) | `tests/properties.rs` | 27 | | | |
| `unit/engine/test_graph.py` (21) | `tests/graph.rs` | 21 | | | |
| `unit/engine/test_api.py` (25) | `tests/api.rs` | 19 | 5 | | 1 |
| `unit/engine/test_rng.py` (23) | `tests/rng.rs` | 16 | 7 | | |
| `unit/engine/test_board.py` (22) | `tests/board.rs` | 18 | 2 | 2 | |
| `unit/engine/test_rules_md_invariants.py` (8) | stays with `tools/gen_board.py` | | | 8 | |
| `unit/engine/test_import_boundary.py` (3) | -- | | | | 3 |
| `unit/engine/test_replay.py` (17) | `tests/replay.rs` | 17 | | | |
| `integration/test_golden_replays.py` (6) | `tests/golden_replays.rs`, `gen-replays` | 6 | | | |
| `unit/rl/test_observation.py` (33) | `tests/observation.rs` | 29 | | | 4 |
| `integration/test_differential.py` (4) | `ttr-tools/tests/oracle.rs`, `golden_replays.rs` | 4 | | | |
| `integration/test_obs_differential.py` (4) | `ttr-tools/tests/oracle.rs`, `golden_corpus.rs` | 4 | | | |
| `unit/agents/test_throwaway.py` (17) | -- | | 5 | | 12 |
| `unit/rust/test_extension.py` (4) | | | | 1 | 3 |
| `integration/test_bench.py` (3) | | | | 2 | 1 |

Exceptions, file by file:

- **test_rules.py.** Covered: `a_player_with_no_track_never_ties_for_the_bonus`
  (`scoring::tests::a_seat_with_no_track_never_ties_for_the_bonus`), `mini_plays_a_full_game`
  (`random_games::random_games_finish_and_conserve_everything` +
  `actions::tests::mini_is_two_hundred_and_twenty_five`), `mini_locomotive_is_not_card_type_eight`
  (`board::tests::the_locomotive_is_not_hard_coded_to_eight`). The two flush tests reach the
  check through a ticket turn rather than a private `_flush_check()`.
- **test_api.py.** Covered: `rules_hash_moves_with_the_rules_but_not_with_bookkeeping`,
  `doubles_lock_by_seat_count` (`config::tests`), `action_space_layout_matches_the_documented_ranges`,
  `claim_ids_are_bijective_with_segment_and_pay`, `mini_action_space_is_smaller_and_still_consistent`
  (`actions::tests`). Dropped: `action_space_is_cached_per_board` -- a Python memo cache; the
  Rust `ActionSpace` is a `Copy` value.
- **test_rng.py.** Covered: the upstream pcg32-demo vector, rot-zero, bounded uniformity and
  int-vs-str parts (`rng::tests`), the golden PRNG and hashing vectors
  (`tests/contract_vectors.rs`), and `the_whole_vector_file_reproduces` (`gen-vectors`, byte
  for byte).
- **test_board.py.** The file itself is kept (moved to `tests/unit/data/`): it tests
  `ticket_to_ride/data/board.py`, which stays. 18 of its tests also gained Rust
  counterparts for the engine's own tables; `generated_files_are_not_stale` and
  `rust_and_python_agree_on_every_constant` are about the generator and have none;
  `action_space_is_915` and `mini_ticket_points_are_shortest_path_costs` were already
  covered in Rust.
- **test_import_boundary.py.** All three dropped: the boundary they guarded (the Python
  engine importing torch, numpy or `ttr_rust`) disappears with the engine.
- **test_golden_replays.py.** `the_corpus_is_not_stale` became `gen-replays`'s
  `the_corpus_is_not_stale` + `a_rewrite_after_a_version_bump_reproduces_the_file`; the two
  slow sweeps are `#[ignore]`d.
- **test_observation.py.** Dropped: `generated_rust_spec_is_not_stale` (the spec is Rust
  source now; its layout is pinned by digest instead), `unknown_field_names_are_rejected`
  (a Python by-name lookup), `the_encoder_imports_without_numpy_or_torch` (Python-only),
  `spec_describe_is_printable` (a Python printing helper).
- **test_differential.py / test_obs_differential.py.** `engines_agree_at_every_step` →
  `oracle::the_live_engine_matches_oracle_v1_at_every_step`; `the_harness_can_actually_fail`
  and `the_comparison_can_actually_fail` → `oracle::the_comparison_can_actually_fail`;
  `golden_replays_reproduce_in_rust` → `golden_replays::every_golden_replay_still_reproduces`;
  `engines_agree_over_the_full_sweep` → `oracle::full_sweep_random`;
  `rust_encoder_matches_the_python_oracle` and `the_two_encoders_agree_on_size` → the
  observation comparison inside every oracle game, plus the corpus's Python-signed digests;
  `obs_version_is_recorded_alongside_the_layout` → `oracle::the_oracle_is_the_build_it_claims_to_be`
  and the corpus header check.
- **test_throwaway.py.** The throwaway agents are deleted. Covered by `tests/heuristics.rs`:
  legal actions only, reproducible from a seed, reseated per game, beats random on both
  maps (the 2P ladder), every action type on every map
  (`every_heuristic_exercises_every_action_type_on_both_maps`). Dropped: the Python greedy
  agent's internals, flat Monte Carlo, the registry and match helpers, the multiplayer
  beats-random check (the arena measures the Rust 4P ladder), and the fraction-not-absolute
  threshold check (Rust's constants are fractions in `HeuristicParams`, and the coverage
  test is what catches one that fails to transfer).
- **test_extension.py.** Stays: `extension_imports`. Dropped: `contract_versions_agree` (one
  implementation; the corpus header pins `CONTRACT_VERSION`) and the two `TTR_REQUIRE_RUST`
  tests (retired with optional Rust).
- **test_bench.py.** Dropped: `rust_is_far_faster_than_the_oracle` -- there is no oracle to be
  faster than.

### Next

**Phase 3.2 -- performance**, held to exact behaviour preservation against `oracle-v1` and
the corpus. The plan-walk defect above is for the user to schedule as a named
`OBS_VERSION` 2; it matters to 3.4, whose PPO reads these features.

---

## 2026-10-15 — Plan revision 1: retire the Python engine

A planning session, no code changed. PLAN.md gains Phases 3.1–3.4 (§14, §14.1) ahead of
Phase 4. Superseded text is marked in place.

### What prompted it

A measurement. Batched `VecEnv` at 1024 envs, USA 2P, random actions, M2 Max:

| per env-step | µs |
| --- | --- |
| observe | 23.8 (wall, 8 cores; ~120–190 µs of CPU per observation) |
| legal mask | 0.19 |
| step | 0.04 |
| an empty FFI call | 0.04 |

The loop caps at ~35k env-steps/s, about 30× under §9's target, and the cost is all in the
encoder (exact Steiner, `remaining_cost`, `fragility`), not the FFI boundary. The Python
encoder runs at 3,166 µs/obs, so Rust is ~26× faster single-threaded. The port worked; the
encoder just isn't fast enough for Phase 6 or 7.

### Decisions

**The Python engine is retired, not frozen.** The user's call. The case for it: the oracle's
value was in catching Rust bugs by disagreement, and that value is mostly banked (700k
games byte-identical, rules that never change). Both implementations came from one author
and one plan, so their bugs are correlated, and every further feature is paid for twice.
Matching Python *structurally* (f64, the same algorithm shapes) is also part of why the
encoder is slow. The alternative considered was freezing it at zero maintenance cost. It was
rejected in favour of one implementation and a validation stack built for it.

**Oracle before deletion, deletion before speed.** Python signs off on a golden corpus,
the verified Rust build is frozen as `oracle-v1` for per-step differential tests, and only
then is Python deleted and the encoder rewritten. A performance rewrite is exactly the
change that most needs a check, so its check has to exist first.

**Modal for cloud experiments.** The workload needs many CPU cores (the encoder) plus a
modest GPU. Modal's per-second billing, CPU-count choice and $30/month free credit fit
that. Colab and Kaggle (2–4 vCPUs, session limits) and TPU programs do not. Academic grant
programs (Nebius, NSF ACCESS) require an institutional affiliation.

**Phase 3.4 and onward are tentative.** A PPO spike on TTR-mini is pulled forward from Phase 6
to retire the biggest unknowns early. Phase 4 is trimmed. Phase 6 gains history features and
a history-probe experiment. All of it is re-planned once Phase 3.3 lands.

### Next

Phase 3.1 — the golden corpus, then `oracle-v1`.

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
