"""The observation feature spec: **one declarative table**, not hand-written encoders.

Observation encoding has to live in Rust eventually -- Python-side encoding at 50-200 us a
step would cap PPO around 10k steps/s and throw away the point of the port -- and it is also
the most churn-prone code in the project. Both are true, so it is declared once here and
generated into both languages. Adding a feature is a table edit plus one accessor, not a
rewrite in two languages.

`OBS_VERSION` is baked into every checkpoint. Loading a checkpoint whose version does not
match is a hard error, which over a months-long league is the difference between "old
checkpoints fail loudly" and "old checkpoints silently play garbage and corrupt your Elo
table".

Widths are symbolic where they depend on the board (`K` = card types, `C1` = segment colours
including gray), so the same table describes the 36-city USA map and 14-city TTR-mini. Slot
counts that the *network* must see as fixed -- opponent slots above all -- are pinned to the
maximum seat count so one net plays any table size.

Design notes worth keeping next to the table:

* **Perspective-normalized.** Everything is from the acting seat's point of view, with
  opponents ordered by distance in turn order (next to act = slot 0). This is seat-*relative*
  rather than permutation-invariant, which is correct: "who moves before me" decides games.
  It also gives a free N-times data multiplier.
* **Connectivity is computed, not learned.** `remaining_cost` is the single most important
  engineered feature in the project. Message passing does connectivity badly -- a hop
  diameter around 10 would need ten layers and over-squash through the hubs -- so the
  environment computes it exactly in microseconds and hands the network the answer.
* **Segment colour is an input feature, not part of the ID embedding.** The colour-symmetry
  regularizer permutes the eight non-locomotive colours across hand, display, discard and
  required colour to produce an isomorphic game; that only works if the required colour is
  something the permutation can reach.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from ticket_to_ride.data.board import Board
from ticket_to_ride.engine.config import MAX_PLAYERS

#: Bumped whenever the table below changes shape or meaning. Baked into every checkpoint.
OBS_VERSION: Final = 1

#: Opponent slots are fixed at the maximum table size so one network plays any seat count;
#: unused slots are zeroed and carry a `present` flag.
OPPONENT_SLOTS: Final = MAX_PLAYERS - 1

#: Thermometer bucket edges. A thermometer beats a raw count for a network because "at
#: least three" is a linearly separable question and "exactly three" is not.
HAND_BUCKETS: Final = (1, 2, 3, 4, 5, 6, 7)
COST_BUCKETS: Final = (1, 3, 5, 8, 12, 17)
TRAIN_BUCKETS: Final = (1, 3, 6, 10, 15, 25, 35)

#: Fewer than two ticket endpoints is nothing to route between.
MIN_TERMINALS: Final = 2


class Field(NamedTuple):
    """One named run of floats inside a block."""

    name: str
    #: An integer, or a symbolic size resolved against the board (see `resolve_width`).
    width: int | str
    doc: str


class Block(NamedTuple):
    """A group of fields, optionally repeated once per entity."""

    name: str
    #: "" for a single instance, else "segments" / "tickets" / "opponents" / "faceup".
    repeat: str
    fields: tuple[Field, ...]


#: Symbolic widths, resolved per board.
#:   K  = card types (colours + locomotive)      USA 9, mini 7
#:   K1 = card types plus "empty slot"           USA 10, mini 8
#:   C1 = segment colours plus gray              USA 9, mini 7
#:   OWNER = free + me + every opponent slot
def resolve_width(width: int | str, board: Board) -> int:
    if isinstance(width, int):
        return width
    return {
        "K": board.n_card_types,
        "K1": board.n_card_types + 1,
        "C1": board.n_colors + 1,
        "OWNER": 2 + OPPONENT_SLOTS,
    }[width]


FEATURE_SPEC: Final = (
    Block(
        "segment_static",
        "segments",
        (
            Field(
                "required_color",
                "C1",
                "One-hot over the colours plus gray. An input feature rather than part of "
                "the segment-ID embedding, so the colour-permutation regularizer can reach it.",
            ),
        ),
    ),
    Block(
        "segment_dynamic",
        "segments",
        (
            Field("owner", "OWNER", "One-hot: free, me, then opponents in turn order."),
            Field("closed", 1, "Blocked for everyone: a 2-3P double whose twin is claimed."),
            Field("twin_locked", 1, "I own the other track of this double route."),
            Field("can_afford_now", 1, "A legal claim for me this turn."),
            Field("cards_short", len(HAND_BUCKETS), "Thermometer over cards still needed."),
            Field("on_my_steiner_tree", 1, "Lies on the cheapest tree spanning my tickets."),
            Field("extends_my_chain", 1, "Touches a city my network already reaches."),
        ),
    ),
    Block(
        "own_hand",
        "",
        (
            Field("counts", "K", "Normalized card counts, locomotive last."),
            Field("thermometer", "K", "Per colour: at least one."),
            Field("thermometer2", "K", "Per colour: at least two."),
            Field("thermometer3", "K", "Per colour: at least three."),
            Field("thermometer4", "K", "Per colour: at least four."),
            Field("thermometer5", "K", "Per colour: at least five."),
            Field("thermometer6", "K", "Per colour: at least six."),
        ),
    ),
    Block(
        "tickets",
        "tickets",
        (
            Field("held", 1, "In my hand."),
            Field("connected", 1, "Already completed by my network."),
            Field(
                "remaining_cost",
                1,
                "Train cars still needed, normalized. The single most important "
                "engineered feature in the project.",
            ),
            Field("cost_thermometer", len(COST_BUCKETS), "Buckets over remaining cost."),
            Field("is_dead", 1, "An opponent has cut every route; the points are lost."),
            Field("points", 1, "Ticket value, normalized."),
            Field(
                "fragility",
                1,
                "Worst-case cost increase if one enemy claim lands on my best path. "
                "Separates a safe connection from one hanging on a single contested edge.",
            ),
        ),
    ),
    Block(
        "steiner",
        "",
        (
            Field("remaining_cost", 1, "Cheapest tree spanning my held tickets, normalized."),
            Field("cost_minus_trains", 1, "Slack: negative means I cannot finish."),
            Field("exact", 1, "0 when the terminal cap forced the MST upper bound."),
        ),
    ),
    Block(
        "faceup",
        "faceup",
        (
            Field(
                "card", "K1", "One-hot per slot, last position = empty. Slot order is action order."
            ),
        ),
    ),
    Block(
        "piles",
        "",
        (
            Field("deck_size", 1, "Undrawn cards, normalized."),
            Field("discard_size", 1, "Discarded cards, normalized."),
            Field("ticket_deck_size", 1, "Tickets left, normalized."),
            Field("discard_composition", "K", "What is in the discard, normalized."),
            Field("unseen", "K", "What I cannot account for: deck plus opponents' blind draws."),
        ),
    ),
    Block(
        "opponents",
        "opponents",
        (
            Field("present", 1, "0 for an unused slot at a smaller table."),
            Field("trains", 1, "Trains left, normalized."),
            Field("trains_thermometer", len(TRAIN_BUCKETS), "Buckets over trains left."),
            Field("score", 1, "Banked route points, normalized."),
            Field("hand_size", 1, "Cards held, normalized."),
            Field("ticket_count", 1, "Tickets held, normalized."),
            Field("blind_draws", 1, "Cards drawn blind and not yet spent, normalized."),
            Field("segments_claimed", 1, "Routes claimed, normalized."),
            Field("longest_chain", 1, "Longest trail so far, normalized."),
            Field("certain", "K", "Cards this seat is publicly known to hold."),
            Field("max_possible", "K", "Upper bound on this seat's holding of each colour."),
        ),
    ),
    Block(
        "clock",
        "",
        (
            Field("phase", 5, "One-hot over the five phases."),
            Field("seats", OPPONENT_SLOTS + 1, "One-hot over table size, 2..5."),
            Field("draws_left", 1, "1 when a second card draw is owed."),
            Field("final_triggered", 1, "The end has been triggered."),
            Field("final_countdown", 1, "Turns left once triggered, normalized."),
            Field("turn", 1, "Turn index, normalized."),
            Field("my_trains", 1, "My trains left, normalized."),
            Field("my_trains_thermometer", len(TRAIN_BUCKETS), "Buckets over my trains."),
            Field("my_score", 1, "My banked route points, normalized."),
            Field("my_ticket_count", 1, "Tickets I hold, normalized."),
            Field("score_rank", 1, "My rank on banked points, 1.0 = leading."),
            Field("gap_to_leader", 1, "Points behind the leader, normalized."),
        ),
    ),
)


class FieldLayout(NamedTuple):
    name: str
    offset: int
    width: int
    doc: str


class BlockLayout(NamedTuple):
    name: str
    repeat: str
    #: How many entities this block is repeated over. 1 for a singleton block.
    count: int
    #: Floats per entity.
    stride: int
    #: Absolute offset of entity 0.
    offset: int
    fields: tuple[FieldLayout, ...]

    @property
    def size(self) -> int:
        return self.count * self.stride

    def at(self, entity: int, field: str) -> int:
        """Absolute offset of `field` for entity `entity`."""
        for f in self.fields:
            if f.name == field:
                return self.offset + entity * self.stride + f.offset
        raise KeyError(f"{self.name} has no field {field!r}")


class ObsSpec:
    """The feature table resolved against one board: concrete widths and offsets."""

    __slots__ = ("blocks", "board", "by_name", "size")

    def __init__(self, board: Board) -> None:
        self.board = board
        counts = {
            "": 1,
            "segments": board.n_segments,
            "tickets": board.n_tickets,
            "opponents": OPPONENT_SLOTS,
            "faceup": 5,
        }

        blocks: list[BlockLayout] = []
        offset = 0
        for block in FEATURE_SPEC:
            fields: list[FieldLayout] = []
            stride = 0
            for field in block.fields:
                width = resolve_width(field.width, board)
                fields.append(FieldLayout(field.name, stride, width, field.doc))
                stride += width
            count = counts[block.repeat]
            blocks.append(
                BlockLayout(block.name, block.repeat, count, stride, offset, tuple(fields))
            )
            offset += count * stride

        self.blocks = tuple(blocks)
        self.by_name = {b.name: b for b in blocks}
        self.size = offset

    def block(self, name: str) -> BlockLayout:
        return self.by_name[name]

    def describe(self) -> list[tuple[str, int, int, int]]:
        """`(block, offset, stride, size)` rows -- what `ttr map`-style tooling prints."""
        return [(b.name, b.offset, b.stride, b.size) for b in self.blocks]

    def __repr__(self) -> str:
        return f"<ObsSpec {self.board.name} v{OBS_VERSION} size={self.size}>"


_CACHE: dict[str, ObsSpec] = {}


def obs_spec(board: Board) -> ObsSpec:
    """The (cached) resolved spec for a board."""
    spec = _CACHE.get(board.name)
    if spec is None:
        spec = _CACHE[board.name] = ObsSpec(board)
    return spec
