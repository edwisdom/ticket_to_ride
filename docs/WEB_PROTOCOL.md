# Web protocol

The game server's API, as built in Phase 4.2 (PLAN.md §14.2). **The Pydantic models in
`ticket_to_ride/ui/web/protocol.py` are the source of truth**; their JSON Schema is checked
in at `ticket_to_ride/ui/web/protocol.schema.json` (`make protocol` regenerates it, `--check`
in pre-commit and the test suite keeps it current), and the client generates its TypeScript
types from that file. This page is the map: the endpoints, the message flow, and where the
protocol departs from the §14.2 sketch.

Run it with `uv run ttr serve` (http://127.0.0.1:8000). FastAPI also serves an interactive
description of the REST half at `/docs`.

## Conventions

- **Indices are the engine's.** Seats, cities, segments, tickets and card types are
  positions in `GET /api/boards/{map}`. Card types are the board's colours in order, then the
  locomotive (`board.locomotive == len(board.colors)`). mini's TOML lists segments in a
  different order from the engine's, so the board endpoint is the only source of indices.
- **64-bit values are strings.** A seed is a decimal string; a JavaScript number holds 53
  bits.
- **Hidden information is `null`, never present.** A seat view holds only what that seat
  knows (§14.2, "Server-authoritative"); spectator, review and end-of-game views are
  omniscient (`viewer: null`).
- **Every server-sent field is always present** (the schema marks them required); request
  fields with defaults may be left out.
- **REST errors** are `{"detail": {"code": ..., "message": ...}}` (`ApiError`); a malformed
  body is FastAPI's usual 422.

## REST

| method | path | body / query | returns |
| --- | --- | --- | --- |
| GET | `/api/meta` | | `Meta`: contract and obs versions, assist level, maps, client build |
| GET | `/api/boards/{map}` | | `Board`: the engine's arrays, plus `layout` |
| GET | `/api/glossary` | | `Glossary`: what every explanation term, fact, context and rule means |
| GET | `/api/agents` | | `Agent[]`: the opponents this server seats, with ratings |
| POST | `/api/players` | `NewPlayer {name}` | `Player` (201) |
| GET | `/api/players/{id}` | | `Profile`: ratings with intervals, record per opponent, finished games |
| GET | `/api/leaderboard` | `?map=usa` | `Leaderboard`: people and bots in one fit |
| POST | `/api/games` | `NewGame` | `GameSummary` (201) |
| GET | `/api/games` | `?player_id=` | `GameSummary[]`, newest first |
| GET | `/api/games/{id}` | | `GameSummary` |
| GET | `/api/games/{id}/review` | | `Review`; **409 until the game is over** |

`Board.layout` is the client's own drawing, `ticket_to_ride/data/layout/{map}.json` from
`tools/gen_layout.py`, passed through verbatim (`null` until generated). The server does not
interpret it; in particular it does not re-key its segments, which follow the map TOML's
order, onto engine segment ids.

### Starting a game

```json
{"mode": "play", "map": "usa", "opponent": "h3", "player_id": "…", "rated": true}
{"mode": "play", "map": "mini", "opponent": "h4", "player_id": "…", "rated": false, "seat": 1}
{"mode": "spectate", "map": "mini", "agents": ["h2", "h4"]}
```

- People are players: `POST /api/players` once, keep the id (no accounts yet).
- `opponent` and `agents` must be specs `GET /api/agents` lists (`ttr serve --agents`
  sets them). A spec can name a parameter file, so arbitrary specs are refused.
- **A rated game seats you at random** (seat 0 moves first, and first-player advantage is
  real); asking for a `seat` in one is a 400. Unrated games take one.
- A play game's bot moves at once if it is to move; a spectated game starts when its first
  spectator connects.

## The game socket: `WS /ws/games/{id}`

Server to client, each a `ServerMessage` discriminated by `type`:

| `type` | when | holds |
| --- | --- | --- |
| `snapshot` | on connect | `game` (`GameSummary`), `view`, `legal`, `log` (every move so far) |
| `event` | each move after the snapshot | `entry` (the move: ply, seat, decoded action, plain-English text), `view` after it, `legal`, `explanation` (spectating only), `think_ms` |
| `over` | once, after the last event (or right after the snapshot of a finished game) | `game`, an omniscient `view`, `scores` (the breakdown and ranks), **`seed`** |
| `ack` | reply to an `act` | `ply`, `action`, `duplicate` |
| `error` | reply to a bad `act`, or instead of a snapshot | `code`, `message`, `ply` |

Client to server, a `ClientMessage`: `{"type": "act", "ply": n, "action": id}`.

- **`legal` is non-empty exactly when the viewer is to move**, and lists every legal action
  decoded: a claim with its segment and payment (`color`, `colored`, `locomotives`), a draw
  with its slot (null for the deck) and the face-up card, a keep with the tickets it keeps.
  The client implements no rules: it sends back an action's `id`.
- **`ply` is the number of moves played before this one.** `act` is idempotent on it: the
  same action again for a ply you already played is acknowledged as a `duplicate` and
  changes nothing, so a double click or a reconnect cannot move twice.
- Each `act` gets one `ack` or `error`; the moves it causes (yours, then the bot's) arrive as
  `event`s, which may come before or after the `ack`.
- Error codes: `bad_message`, `not_found`, `spectating` (nobody plays a spectated game),
  `game_over`, `not_your_turn`, `stale_ply` (that ply was played, differently),
  `future_ply`, `illegal_action` (with the engine's reason), `replay_mismatch` (the store
  and a bot disagree; see below), `unavailable`.
- Bots move as fast as they think and every move is streamed as it is made; **the client
  paces the animation.** A spectator connecting late gets a snapshot at the current ply and
  every move after it; the review has the whole game.

## Views

A `View` is a position for one viewer: segment owners (`owners`, and `closed` for the
other track of a claimed double at 2-3 players), face-up cards, deck size, discard counts,
ticket-deck size, phase, current player, draws left, the final-round countdown
(`final_turns`), and per seat its train pieces left (`trains`: the plastic trains, not
cards), banked route points, hand size, publicly known cards (`known`), ticket count and
longest trail. The viewer's own `hand` and `tickets` (each with `connected` and `remaining`
train pieces) are filled in, and `offer` during the viewer's keep.

**Bot parity** (`assist: "parity-1"`, recorded with every game): a seat view also carries
what the bots' observation holds -- opponents' known cards, the discard composition,
`unseen` (per card type, what the viewer cannot account for), and ticket progress. A change
to what a view shows is a new assist level.

## Explanations

At each bot decision (live when spectating, and in every review) an `Explanation` lists the
best candidates, each with its `total` -- exactly the number the bot compared -- its
additive `terms` in points (they sum to the total), the `facts` behind them, and a `rule`
where one decided (H1's cascade, a forced move, H0's uniform draw). For H1-H4 the first
candidate is the move played. `GET /api/glossary` says what every name means.

They come from the engine's one scorer through a trace sink (Phase 4.1), never a second
scorer, and a review recomputes them by replaying the game.

## The store, rebuilds and review

Every move is written to `runs/play.db` before the game in memory advances, and a game not
in memory is rebuilt from its seed and moves -- asking each bot for each of its past
decisions and requiring the recorded answer. A bot whose code changed mid-game fails that
(`replay_mismatch`) rather than continuing as a different player. A finished game's
`review` replays it with everything revealed: an omniscient view after every move and each
bot decision explained.

## Ratings

Rated finished games join the arena's games (`--results-db`, default `runs/results.db`) in
one Bradley-Terry fit per map with H3 fixed at 1500 (`Leaderboard.anchor_elo`): each human game its own seed block, a
person by player id, a bot by its identity -- so the H3 you played is the H3 the arena
measured. A rating needs a path to H3: play H3, or run `ttr arena` so the other bots are
anchored. Intervals are 95% block-bootstrap; with fewer than two blocks there is no interval
(`lo`/`hi` null) rather than a zero-width one.

## Departures from the §14.2 sketch

- `POST /api/games` names its `mode` (`"play"` or `"spectate"`) explicitly, and a play game
  needs a `player_id`; `rated` (default true) and `seat` (unrated only) are new.
- The socket adds `ack` and `error`. An `event` is `{entry, view, legal, explanation,
  think_ms}`; `over` also carries the final `game` summary.
- New endpoints: `/api/meta`, `/api/glossary`, `/api/players` (+ profile),
  `/api/leaderboard`, `/api/games` (list) and `/api/games/{id}`.
- The board's `layout` is passed through as generated, not merged with engine indices.
- A spectated game starts on its first spectator, not on creation.

Not built, for the public deployment (§14.2): accounts and seat tokens -- today anyone who
knows a play game's id can open its socket and play its seat -- rate limits, and one live
rated game per player.
