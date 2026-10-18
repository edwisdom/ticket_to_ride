"""The web protocol: every message between the game server and the client.

**These models are the source of truth** (PLAN.md §14.2). `tools/gen_protocol.py` writes
their JSON Schema to `protocol.schema.json` beside this file -- checked in, and verified
like the board data (`make protocol`, `--check`) -- and the client generates its
TypeScript types from it. docs/WEB_PROTOCOL.md describes the endpoints and the message
flow; the field docs here are the reference.

Conventions:

* Seats, cities, segments, tickets and card types are the engine's own indices, as
  `GET /api/boards/{map}` lists them. Card types are the board's colours in order, then
  the locomotive.
* 64-bit values -- a seed -- travel as decimal strings: a JavaScript number holds 53 bits.
* **Nothing hidden from the viewer is ever in a payload.** Hidden fields are `null`, and
  the masking tests (tests/unit/web/) hold that.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.json_schema import models_json_schema

#: The assistance a seat view gives, recorded with every game: the public information
#: the bots' observation holds (PLAN.md §14.2). A change to what a view shows is a new
#: level, so a rating can be split by it rather than silently mixing two.
ASSIST = "parity-1"

type MapName = Literal["usa", "mini"]
type Phase = Literal["initial_tickets", "main", "draw_second", "ticket_keep", "terminal"]
type Mode = Literal["play", "spectate"]
type Status = Literal["active", "finished"]


class Model(BaseModel):
    """Sent by the server. Every field is always present, so the schema requires them."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_serialization_defaults_required=True
    )


class Request(BaseModel):
    """Sent by the client. Fields with defaults may be left out."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_mode_override="validation")


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------


class Segment(Model):
    """One track. A double route is two segments, each the other's `sibling`."""

    id: int
    a: int
    b: int
    length: int
    color: int | None = Field(description="The colour required; null for gray (any one colour).")
    sibling: int | None


class Ticket(Model):
    id: int
    a: int
    b: int
    points: int


class Board(Model):
    """A board as the engine plays it, in the engine's order -- the client's only source of
    board data, since mini's TOML is not in engine order."""

    name: MapName
    data_hash: str
    cities: list[str]
    segments: list[Segment]
    tickets: list[Ticket]
    colors: list[str] = Field(description="Colour names; card type i < len(colors) is colors[i].")
    locomotive: int = Field(description="The locomotive's card type: len(colors).")
    card_counts: list[int] = Field(description="Cards printed per card type.")
    trains_per_player: int = Field(description="Train pieces each seat starts with.")
    end_trigger_trains: int = Field(
        description="A seat ending its turn with this many train pieces or fewer starts the"
        " final round: every seat, itself included, gets one more turn."
    )
    route_points: list[int] = Field(description="route_points[n] scores a route of length n.")
    longest_bonus: int
    players: list[int] = Field(description="[fewest, most] seats the map supports.")
    initial_ticket_deal: int
    initial_ticket_keep_min: int
    draw_ticket_deal: int
    draw_ticket_keep_min: int
    faceup_slots: int
    action_space: int
    layout: dict[str, Any] | None = Field(
        description="The client's drawing of this board (ticket_to_ride/data/layout/{map}.json),"
        " passed through verbatim; null until it is generated."
    )


# ---------------------------------------------------------------------------
# Players, agents, ratings
# ---------------------------------------------------------------------------


class Player(Model):
    """A person or a bot. A bot's id is derived from its identity -- what it plays -- so
    one bot is one player in every store."""

    id: str
    kind: Literal["human", "bot"]
    name: str
    spec: str | None = Field(description="A bot's agent spec, e.g. 'h3'; null for a person.")


class NewPlayer(Request):
    name: str = Field(min_length=1, max_length=40, pattern=r"\S")


class Rating(Model):
    """Bradley-Terry Elo anchored on H3 (`Leaderboard.anchor_elo`), with a 95% interval."""

    elo: float
    lo: float | None = Field(description="null when there is too little data for an interval.")
    hi: float | None
    games: int = Field(description="Pairwise comparisons the rating rests on.")
    score: float | None = Field(description="Share of those comparisons won.")


class MapRating(Model):
    map: MapName
    rating: Rating


class Agent(Model):
    """An opponent the server will seat."""

    spec: str
    name: str
    family: str
    description: str
    anchor: bool = Field(description="True for H3 itself, the fixed point of the rating scale.")
    player: Player
    behaviour_hash: str
    ratings: list[MapRating]


class Standing(Model):
    player: Player
    rating: Rating


class Leaderboard(Model):
    """People and bots in one Bradley-Terry fit over arena and rated human games."""

    map: MapName
    n_players: int
    anchor: str
    anchor_elo: float = Field(description="The anchor's rating, by definition.")
    games: int
    blocks: int
    standings: list[Standing]
    note: str | None = Field(description="Why the board is empty, when it is.")


class OpponentRecord(Model):
    map: MapName
    opponent: Player
    games: int
    wins: int
    losses: int
    draws: int
    mean_score: float
    mean_opponent_score: float


class GameResult(Model):
    game_id: str
    map: MapName
    rated: bool
    opponent: Player
    seat: int
    score: int
    opponent_score: int
    result: Literal["win", "loss", "draw"]
    finished: str


class Profile(Model):
    player: Player
    ratings: list[MapRating]
    record: list[OpponentRecord] = Field(description="Finished games against each opponent.")
    games: list[GameResult] = Field(description="Finished games, newest first.")


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------


class NewPlayGame(Request):
    """Play a bot. A rated game seats you at random: seat 0 moves first, and first-player
    advantage is real."""

    mode: Literal["play"]
    map: MapName
    opponent: str = Field(description="An agent spec from GET /api/agents.")
    player_id: str
    rated: bool = True
    seat: int | None = Field(default=None, description="Unrated games only.")


class NewSpectateGame(Request):
    """Watch two bots, with every hand visible and each decision's reasoning."""

    mode: Literal["spectate"]
    map: MapName
    agents: list[str] = Field(min_length=2, max_length=2)


class NewGame(RootModel[Annotated[NewPlayGame | NewSpectateGame, Field(discriminator="mode")]]):
    pass


class SeatInfo(Model):
    seat: int
    player: Player
    score: int | None = Field(description="Final score, once the game is over.")
    rank: int | None = Field(description="1 is first; ties share a rank.")


class GameSummary(Model):
    id: str
    map: MapName
    mode: Mode
    rated: bool
    assist: str
    status: Status
    n_players: int
    seats: list[SeatInfo]
    human_seat: int | None = Field(description="The person's seat in a play game.")
    ply: int = Field(description="Actions played so far.")
    created: str
    finished: str | None


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class TicketProgress(Model):
    ticket: int
    connected: bool
    remaining: int | None = Field(
        description="Train pieces still needed over your track and open track; null once no "
        "route is left."
    )


class SeatView(Model):
    """One seat, as the viewer may see it."""

    seat: int
    trains: int = Field(description="Train pieces left to place: the supply, not cards.")
    score: int = Field(description="Route points banked so far; tickets settle at the end.")
    hand_size: int
    known: list[int] = Field(
        description="Cards publicly known to be in hand, per card type: face-up takes not yet "
        "spent."
    )
    ticket_count: int
    longest_trail: int
    hand: list[int] | None = Field(
        description="Per card type; the viewer's own, or all omniscient."
    )
    tickets: list[TicketProgress] | None = Field(description="Hidden like the hand.")


class View(Model):
    """A position as one viewer may see it. A seat view holds only what that seat knows;
    `viewer` null is omniscient (spectating, review, the end)."""

    viewer: int | None
    ply: int
    turn: int
    phase: Phase
    current: int
    draws_left: int
    final_turns: int | None = Field(
        description="Once the end is triggered: turns left, the current one included."
    )
    owners: list[int | None] = Field(description="Per segment: the owning seat, or null.")
    closed: list[int] = Field(
        description="Segments closed because the other track of their double was claimed "
        "(2-3 players)."
    )
    faceup: list[int | None] = Field(description="Card type per face-up slot; null when empty.")
    deck: int = Field(description="Cards left in the draw pile.")
    discard: list[int]
    ticket_deck: int
    seats: list[SeatView]
    offer: list[int] | None = Field(
        description="Tickets on offer during a keep, to the keeper (or omniscient)."
    )
    unseen: list[int] | None = Field(
        description="Seat views: per card type, cards the viewer cannot account for -- the "
        "deck plus opponents' blind draws."
    )
    deck_cards: list[int] | None = Field(
        description="Omniscient views: the draw pile's composition (never its order)."
    )


# ---------------------------------------------------------------------------
# Actions, decoded by the server so the client implements no rules
# ---------------------------------------------------------------------------


class ClaimAction(Model):
    kind: Literal["claim"]
    id: int
    segment: int
    color: int = Field(description="The card type paid with; the locomotive for all-locomotive.")
    colored: int = Field(description="Cards of `color` spent.")
    locomotives: int
    label: str


class DrawAction(Model):
    kind: Literal["draw"]
    id: int
    slot: int | None = Field(description="The face-up slot; null for the top of the deck.")
    card: int | None = Field(
        description="The card taken: always for face-up, for the deck only to its drawer."
    )
    label: str


class DrawTicketsAction(Model):
    kind: Literal["draw_tickets"]
    id: int
    dealt: int | None = Field(description="Tickets dealt (in a log entry).")
    tickets: list[int] | None = Field(description="Which, to the drawer only.")
    label: str


class KeepAction(Model):
    kind: Literal["keep"]
    id: int
    offer_indices: list[int]
    count: int
    tickets: list[int] | None = Field(description="Tickets kept, to the keeper only.")
    returned: list[int] | None = Field(description="Tickets returned, to the keeper only.")
    label: str


class PassAction(Model):
    kind: Literal["pass"]
    id: int
    label: str


type Action = Annotated[
    ClaimAction | DrawAction | DrawTicketsAction | KeepAction | PassAction,
    Field(discriminator="kind"),
]


class LogEntry(Model):
    ply: int
    seat: int
    action: Action
    text: str


# ---------------------------------------------------------------------------
# A bot's reasoning
# ---------------------------------------------------------------------------


class Term(Model):
    name: str
    value: float


class Candidate(Model):
    action: Action
    total: float | None = Field(description="What the decision compared; null when ruled out.")
    rule: str | None
    terms: list[Term] = Field(description="Additive, in points; they sum to `total`.")
    facts: list[Term] = Field(description="Inputs behind the terms, each in its own unit.")


class Explanation(Model):
    """One bot decision: its best candidates and why. For H1-H4 the first is the action
    played; H0's candidates are the uniform distribution it drew from."""

    agent: str
    tier: str
    chosen: int
    rule: str | None
    context: list[Term]
    candidates: list[Candidate]


class Glossary(Model):
    """What every name in an explanation means."""

    terms: dict[str, str]
    facts: dict[str, str]
    context: dict[str, str]
    rules: dict[str, str]


# ---------------------------------------------------------------------------
# The game socket
# ---------------------------------------------------------------------------


class Score(Model):
    seat: int
    routes: int
    tickets_made: int
    tickets_missed: int
    ticket_points: int
    longest_bonus: int
    longest_trail: int
    total: int
    rank: int


class Snapshot(Model):
    """On connect: where the game stands. `legal` is non-empty exactly when the viewer is
    to move."""

    type: Literal["snapshot"]
    game: GameSummary
    view: View
    legal: list[Action]
    log: list[LogEntry]


class Event(Model):
    """One action, as the viewer may see it, and the position after it."""

    type: Literal["event"]
    entry: LogEntry
    view: View
    legal: list[Action]
    explanation: Explanation | None = Field(description="Spectating only.")
    think_ms: int


class Over(Model):
    """The end: everything revealed, the score breakdown and the seed."""

    type: Literal["over"]
    game: GameSummary
    view: View
    scores: list[Score]
    seed: str


class Ack(Model):
    """The reply to an `act`. A duplicate -- the same action for a ply already played by
    you -- changes nothing and says so."""

    type: Literal["ack"]
    ply: int
    action: int
    duplicate: bool


type ErrorCode = Literal[
    "bad_message",
    "not_found",
    "spectating",
    "game_over",
    "not_your_turn",
    "stale_ply",
    "future_ply",
    "illegal_action",
    "replay_mismatch",
    "unavailable",
]


class ErrorMessage(Model):
    type: Literal["error"]
    code: ErrorCode
    message: str
    ply: int | None


class ServerMessage(
    RootModel[
        Annotated[
            Snapshot | Event | Over | Ack | ErrorMessage,
            Field(discriminator="type"),
        ]
    ]
):
    pass


class Act(Request):
    """Play `action` at `ply` -- the count of actions played before it, so a double click
    or a reconnect can never apply a move twice."""

    type: Literal["act"]
    ply: int = Field(ge=0)
    action: int = Field(ge=0)


class ClientMessage(RootModel[Annotated[Act, Field(discriminator="type")]]):
    pass


# ---------------------------------------------------------------------------
# Review, REST errors, meta
# ---------------------------------------------------------------------------


class ReviewPly(Model):
    entry: LogEntry
    view: View
    explanation: Explanation | None
    think_ms: int


class Review(Model):
    """A finished game with everything revealed, every decision in order, and each bot
    decision's explanation -- recomputed by replay, not stored."""

    game: GameSummary
    seed: str
    initial: View
    plies: list[ReviewPly]
    scores: list[Score]


class ApiError(Model):
    """The body of every REST error, under `detail`."""

    code: str
    message: str


class Meta(Model):
    contract_version: int
    obs_version: int
    assist: str
    maps: list[MapName]
    client_build: str | None


#: Every top-level type, in the order the schema lists them.
TOP_LEVEL: tuple[type[BaseModel], ...] = (
    Board,
    Agent,
    Player,
    NewPlayer,
    Profile,
    Leaderboard,
    NewGame,
    GameSummary,
    Review,
    Glossary,
    Meta,
    ApiError,
    ServerMessage,
    ClientMessage,
)


def json_schema() -> dict[str, Any]:
    """The checked-in schema: every model under `$defs`, and a root object naming the
    top-level ones so a generator that follows references from the root emits them all."""
    _, schema = models_json_schema(
        [(model, "serialization") for model in TOP_LEVEL],
        ref_template="#/$defs/{model}",
        title="TicketToRideWebProtocol",
        description="Generated from ticket_to_ride/ui/web/protocol.py by tools/gen_protocol.py.",
    )
    for definition in schema["$defs"].values():
        # Property titles are generated noise ("Kind", "Seat"), and some generators turn
        # each into a named type of its own.
        for prop in definition.get("properties", {}).values():
            prop.pop("title", None)
    schema["type"] = "object"
    schema["properties"] = {
        model.__name__: {"$ref": f"#/$defs/{model.__name__}"} for model in TOP_LEVEL
    }
    return schema


def schema_text() -> str:
    return json.dumps(json_schema(), indent=2, sort_keys=False) + "\n"
