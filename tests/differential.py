"""The differential harness: Python oracle vs Rust core, compared at **every** step.

Terminal-only comparison tells you *that* you diverged, never *where* -- so this asserts
`state_hash()` equality and `legal_actions()` set equality after every single step, and
when they disagree it reports the step, the turn, the phase, the action that caused it,
and **the name of the first serialized field that differs**.

That last part is the difference between a two-minute fix and an afternoon. A hash
mismatch says "these states differ"; "byte 214 is inside `certain[1]`" says which line of
which function to look at. The layout below is CONTRACT.md §3.1 read back as field spans.

Not importable by `ticket_to_ride.engine` -- this lives in `tests/`, and the engine's
import boundary test bans `ttr_rust` from it, because an oracle that imports the
implementation it validates is not an oracle.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple, Protocol

from ticket_to_ride.engine.config import PHASE_NAMES, RuleConfig
from ticket_to_ride.engine.replay import Replay
from ticket_to_ride.engine.rng import Pcg32, stream
from ticket_to_ride.engine.scoring import (
    Breakdown,
    final_scores,
    longest_trails,
    returns,
    score_breakdown,
    winners,
)
from ticket_to_ride.engine.state import Game, State

if TYPE_CHECKING:
    from ticket_to_ride.data.board import Board


class RustState(Protocol):
    """The slice of `ttr_rust.State` this harness drives.

    A protocol rather than `Any`: `ttr_rust` is an optional dependency that may not be
    built, so it cannot be imported for typing, but the comparison should still fail type
    checking if it reaches for a method the shim does not expose.
    """

    def step(self, action: int) -> None: ...
    def legal_actions(self) -> list[int]: ...
    def state_hash(self) -> int: ...
    def position_hash(self) -> int: ...
    def serialize(self, canonical: bool = False) -> bytes: ...
    def is_terminal(self) -> bool: ...
    def current_player(self) -> int: ...
    def history(self) -> list[int]: ...
    def final_scores(self) -> list[int]: ...
    def score_breakdown(self) -> list[tuple[int, ...]]: ...
    def returns(self) -> list[float]: ...
    def winners(self) -> list[int]: ...
    def longest_trails(self) -> list[int]: ...
    def observation(self, player: int) -> list[float]: ...

    @property
    def observation_size(self) -> int: ...


class RustGame(Protocol):
    def new_initial_state(self, seed: int) -> RustState: ...


class Divergence(AssertionError):  # noqa: N818 - the domain term, as in "diverged at turn 59"
    """The two engines disagreed. Carries enough context to reproduce in one line."""


class Span(NamedTuple):
    """One field's byte range in the canonical serialization."""

    name: str
    start: int
    length: int

    def contains(self, offset: int) -> bool:
        return self.start <= offset < self.start + self.length


def layout(board: Board, n_players: int, deck_len: int) -> list[Span]:
    """The field spans of CONTRACT.md §3.1, in order.

    `deck_len` is a field of the image rather than a constant: a reshuffle installs a
    shorter deck, so the spans after it shift. Read it out of the bytes with
    `deck_len_of()` rather than assuming the board's deck size.
    """
    n, k = n_players, board.n_card_types
    fields = [
        ("contract_version", 1),
        ("n_players", 1),
        ("phase", 1),
        ("current_player", 1),
        ("draws_left", 1),
        ("final_left", 1),
        ("pass_streak", 1),
        ("turn", 2),
        ("seg_owner", board.n_segments),
        ("hand", n * k),
        ("trains", n),
        ("score", 2 * n),
        ("tickets", 4 * n),
        ("deck_cursor", 2),
        ("deck_len", 2),
        ("deck", deck_len),
        ("discard", k),
        ("faceup", 5),
        ("tdeck_head", 1),
        ("tdeck_len", 1),
        ("tdeck", board.n_tickets),
        ("certain", n * k),
        ("unknown", n),
        ("offer_len", 1),
        ("offer", 3),
        ("rng.state", 8),
        ("rng.inc", 8),
    ]
    spans: list[Span] = []
    offset = 0
    for name, length in fields:
        spans.append(Span(name, offset, length))
        offset += length
    return spans


def deck_len_of(board: Board, n_players: int, image: bytes) -> int:
    """Read `deck_len` out of a serialized image, so the later spans can be placed."""
    n, k = n_players, board.n_card_types
    offset = 9 + board.n_segments + n * k + n + 2 * n + 4 * n + 2
    return int.from_bytes(image[offset : offset + 2], "little")


def locate(board: Board, n_players: int, mine: bytes, theirs: bytes) -> str:
    """Name the first serialized **field** that differs.

    Field-wise rather than byte-wise, and each image is measured with its *own* `deck_len`.
    That matters: a reshuffle happening at a different moment changes `deck_len`, which
    changes the image length, and a naive byte-offset diff would report "the images are
    different lengths" -- which reads like a layout bug and sends you to check field widths
    when the actual difference is one value several fields earlier.
    """
    my_spans = layout(board, n_players, deck_len_of(board, n_players, mine))
    their_spans = layout(board, n_players, deck_len_of(board, n_players, theirs))

    for a_span, b_span in zip(my_spans, their_spans, strict=True):
        a = mine[a_span.start : a_span.start + a_span.length]
        b = theirs[b_span.start : b_span.start + b_span.length]
        if a == b:
            continue
        if len(a) != len(b):
            return (
                f"field {a_span.name!r} has different lengths: python {len(a)}, rust "
                f"{len(b)} (implied by an earlier length field)"
            )
        index = next(i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y)
        return (
            f"first differing field is {a_span.name!r}[{index}] "
            f"(byte {a_span.start + index}): python {a[index]}, rust {b[index]}"
        )

    if len(mine) != len(theirs):
        return (
            f"every known field matches but the images differ in length: python "
            f"{len(mine)}, rust {len(theirs)} -- a serializer is writing trailing bytes"
        )
    return "images are byte-identical (so the mismatch is not in the serialization)"


class Mismatch(NamedTuple):
    """What the engines disagreed about at one step."""

    map_name: str
    n_players: int
    seed: int
    step: int
    turn: int
    phase: str
    action: int | None
    detail: str

    def __str__(self) -> str:
        where = "at setup" if self.action is None else f"after step {self.step}"
        acted = "" if self.action is None else f" (action {self.action})"
        return (
            f"{self.map_name} {self.n_players}P seed {self.seed} diverged {where}{acted}, "
            f"turn {self.turn}, phase {self.phase}: {self.detail}\n"
            f"  reproduce: compare_game({self.map_name!r}, {self.n_players}, {self.seed})"
        )


def compare_position(
    py: State,
    rs: RustState,
    context: tuple[str, int, int, int, int | None],
) -> None:
    """Assert everything two engines must agree on at one position.

    `context` is `(map, seats, seed, step index, action just applied)` -- carried only so
    the failure message can name the step and print a one-line reproduction.
    """
    map_name, n_players, seed, index, action = context
    board = py.game.board

    def fail(detail: str) -> None:
        raise Divergence(
            str(
                Mismatch(
                    map_name=map_name,
                    n_players=n_players,
                    seed=seed,
                    step=index,
                    turn=py.turn,
                    phase=PHASE_NAMES[py.phase],
                    action=action,
                    detail=detail,
                )
            )
        )

    if py.is_terminal() != rs.is_terminal():
        fail(f"is_terminal: python {py.is_terminal()}, rust {rs.is_terminal()}")

    py_hash, rs_hash = py.state_hash(), rs.state_hash()
    if py_hash != rs_hash:
        fail(
            f"state_hash {py_hash:016x} != {rs_hash:016x}. "
            + locate(board, n_players, py._serialize(canonical=False), rs.serialize(False))
        )

    # position_hash is checked too: it is the MCTS transposition key, it zeroes different
    # bytes, and a bug in the zeroing would be invisible to state_hash alone.
    py_pos, rs_pos = py.position_hash(), rs.position_hash()
    if py_pos != rs_pos:
        fail(
            f"position_hash {py_pos:016x} != {rs_pos:016x}. "
            + locate(board, n_players, py._serialize(canonical=True), rs.serialize(True))
        )

    py_legal, rs_legal = py.legal_actions(), rs.legal_actions()
    if py_legal != rs_legal:
        only_py = sorted(set(py_legal) - set(rs_legal))
        only_rs = sorted(set(rs_legal) - set(py_legal))
        named = [py.game.action_to_string(a) for a in (only_py + only_rs)[:5]]
        fail(
            f"legal_actions differ: {len(py_legal)} python vs {len(rs_legal)} rust; "
            f"python-only {only_py[:8]}, rust-only {only_rs[:8]} ({named})"
        )

    if not py.is_terminal() and py.current_player() != rs.current_player():
        fail(f"current_player: python {py.current_player()}, rust {rs.current_player()}")


def compare_game(
    map_name: str,
    n_players: int,
    seed: int,
    *,
    rust: ModuleType | None = None,
    max_steps: int = 20_000,
) -> int:
    """Play one game in both engines with identical actions. Returns the step count.

    The action at each step is drawn from the **sorted** legal list with an independent
    policy stream, so the drive is a pure function of the seed and the two engines are
    never allowed to disagree about which action was taken -- only about what it did.
    """
    game: RustGame = _rust(rust).Game(map_name, n_players)
    py = Game(RuleConfig(map_name=map_name, n_players=n_players)).new_initial_state(seed)
    rs = game.new_initial_state(seed)
    policy: Pcg32 = stream(seed, "differential", "policy")

    compare_position(py, rs, (map_name, n_players, seed, 0, None))

    index = 0
    while not py.is_terminal():
        legal = py.legal_actions()
        action = legal[policy.below(len(legal))]
        py.step(action)
        rs.step(action)
        index += 1
        compare_position(py, rs, (map_name, n_players, seed, index, action))
        if index >= max_steps:
            raise Divergence(f"{map_name} {n_players}P seed {seed} did not terminate")

    if py.history_actions() != rs.history():
        raise Divergence(f"{map_name} {n_players}P seed {seed}: recorded histories differ")
    compare_terminal(py, rs, (map_name, n_players, seed))
    return index


def compare_terminal(py: State, rs: RustState, context: tuple[str, int, int]) -> None:
    """Everything that can only be known once the game is over.

    Not covered by `state_hash`: the terminal hash proves the two engines reached the same
    *position*, and says nothing about whether they score it the same way. Ticket
    settlement, the longest trail and the tiebreak chain are all computed from the position
    rather than stored in it, so they need their own comparison.
    """
    map_name, n_players, seed = context

    def fail(detail: str) -> None:
        raise Divergence(
            f"{map_name} {n_players}P seed {seed} diverged at the terminal: {detail}\n"
            f"  reproduce: compare_game({map_name!r}, {n_players}, {seed})"
        )

    py_trails = longest_trails(py)
    rs_trails = rs.longest_trails()
    if py_trails != rs_trails:
        # Named first because it is the hardest of the three to get right, and a mismatch
        # here explains both the scores and the tiebreak that follow.
        fail(f"longest trails: python {py_trails}, rust {rs_trails}")

    py_break = [tuple(b) for b in score_breakdown(py)]
    rs_break = [tuple(b) for b in rs.score_breakdown()]
    if py_break != rs_break:
        fields = Breakdown._fields
        for seat, (a, b) in enumerate(zip(py_break, rs_break, strict=True)):
            for name, x, y in zip(fields, a, b, strict=True):
                if x != y:
                    fail(f"seat {seat} breakdown field {name!r}: python {x}, rust {y}")
        fail(f"breakdowns differ: python {py_break}, rust {rs_break}")

    py_scores, rs_scores = final_scores(py), rs.final_scores()
    if py_scores != rs_scores:
        fail(f"final scores: python {py_scores}, rust {rs_scores}")

    py_win, rs_win = winners(py), rs.winners()
    if py_win != rs_win:
        fail(f"winners: python {py_win}, rust {rs_win}")

    # Exact, not approximate. Both sides evaluate the same IEEE-754 double operations in
    # the same order, so any drift is a real difference in the formula rather than
    # floating-point noise -- and a tolerance here would hide exactly that.
    py_returns, rs_returns = returns(py), rs.returns()
    if py_returns != rs_returns:
        fail(f"returns: python {py_returns}, rust {rs_returns}")


def _rust(module: ModuleType | None) -> ModuleType:
    """The `ttr_rust` module, imported lazily.

    Deferred because `ttr_rust` is built by `make rust` rather than `uv sync`, so a
    top-level import would make this module unimportable whenever the extension is absent
    -- including in the torch-free CI job that has no Rust toolchain at all.
    """
    if module is not None:
        return module
    import ttr_rust  # noqa: PLC0415 - see the docstring

    return ttr_rust


def compare_replay(record: Replay, *, rust: ModuleType | None = None) -> int:
    """Re-run a recorded golden game through the Rust engine, comparing at every step.

    Stronger than `compare_game` in one specific way: the action sequence comes from a file
    written by the *Python* engine months of edits ago, so it exercises positions a fresh
    random drive may never reach, and the recorded final hash is an independent third
    party rather than whatever the two engines agree on today.
    """
    cfg = RuleConfig(map_name=record.map_name, n_players=record.n_players)
    game: RustGame = _rust(rust).Game(record.map_name, record.n_players)
    py = Game(cfg).new_initial_state(record.seed)
    rs = game.new_initial_state(record.seed)
    context = (record.map_name, record.n_players, record.seed)

    compare_position(py, rs, (*context, 0, None))
    for index, action in enumerate(record.actions, start=1):
        py.step(action)
        rs.step(action)
        compare_position(py, rs, (*context, index, action))

    if rs.state_hash() != record.final_hash:
        raise Divergence(
            f"{record.map_name} {record.n_players}P seed {record.seed}: rust final hash "
            f"{rs.state_hash():016x} != recorded {record.final_hash:016x}"
        )
    if not rs.is_terminal():
        raise Divergence(f"seed {record.seed}: the replay ended before the rust game did")
    compare_terminal(py, rs, context)
    if list(record.final_scores) != rs.final_scores():
        raise Divergence(
            f"{record.map_name} {record.n_players}P seed {record.seed}: rust scores "
            f"{rs.final_scores()} != recorded {list(record.final_scores)}"
        )
    return len(record.actions)
