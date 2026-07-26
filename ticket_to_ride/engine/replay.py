"""Game replays: ~360 bytes per game, and bitwise or it fails loudly.

A replay is `(contract version, map, seats, seed, DATA_HASH, rules_hash, actions,
final state hash, final scores)`. Everything except the actions is there to make a stale
replay *fail* rather than quietly replay a different game -- a board edit moves
`DATA_HASH`, a rule change moves `rules_hash`, and a change to the PRNG or the draw
procedure moves `CONTRACT_VERSION`.

This is reproducibility level **L0**: bitwise, always, and tested. See PLAN.md §10 for the
three levels and why conflating L1 and L2 costs a day chasing ghosts.
"""

from __future__ import annotations

import struct
from typing import Final

import msgspec

from ticket_to_ride.engine.config import RuleConfig
from ticket_to_ride.engine.contract import CONTRACT_VERSION
from ticket_to_ride.engine.scoring import final_scores
from ticket_to_ride.engine.state import Game, State

#: File magic. Present so a truncated or unrelated file is rejected immediately.
MAGIC: Final = b"TTR"

#: Actions are u16 on the wire, which every map's space fits inside with room to spare.
MAX_ACTION: Final = 0xFFFF

#: The map name is length-prefixed with a single byte.
MAX_NAME: Final = 0xFF

_HEADER = struct.Struct("<3sBBB")
_BODY = struct.Struct("<HQ16s16sQ")


class ReplayError(Exception):
    """A replay that cannot be trusted: wrong board, wrong rules, or a diverged outcome."""


class Replay(msgspec.Struct, frozen=True):
    """One recorded game."""

    map_name: str
    n_players: int
    seed: int
    actions: tuple[int, ...]
    #: Board identity. A map edit invalidates every replay taken on the old board.
    data_hash: str
    #: Rule identity, covering everything in `RuleConfig` that affects play.
    rules_hash: str
    #: `state_hash()` of the terminal position. The thing a replay actually proves.
    final_hash: int
    final_scores: tuple[int, ...]
    contract_version: int = CONTRACT_VERSION

    @property
    def n_actions(self) -> int:
        return len(self.actions)


def record(state: State) -> Replay:
    """Capture a finished game. Requires `track_history`, which is on by default."""
    if not state.is_terminal():
        raise ReplayError("only a finished game can be recorded")
    if not state.game.cfg.track_history:
        raise ReplayError("this game was played with track_history off; nothing to record")
    game = state.game
    if game.space.n > MAX_ACTION:
        raise ReplayError(f"action space {game.space.n} does not fit in a u16")
    return Replay(
        map_name=game.board.name,
        n_players=game.n_players,
        seed=state.seed,
        actions=tuple(state.history),
        data_hash=game.board.data_hash,
        rules_hash=game.cfg.rules_hash,
        final_hash=state.state_hash(),
        final_scores=tuple(final_scores(state)),
    )


def encode(rec: Replay) -> bytes:
    """Pack to the wire format. Fixed field order, little-endian, no padding."""
    name = rec.map_name.encode()
    if len(name) > MAX_NAME:
        raise ReplayError(f"map name {rec.map_name!r} is too long")
    return b"".join(
        (
            _HEADER.pack(MAGIC, rec.contract_version, rec.n_players, len(name)),
            name,
            _BODY.pack(
                len(rec.actions),
                rec.seed,
                bytes.fromhex(rec.data_hash),
                bytes.fromhex(rec.rules_hash),
                rec.final_hash,
            ),
            struct.pack(f"<{rec.n_players}h", *rec.final_scores),
            struct.pack(f"<{len(rec.actions)}H", *rec.actions),
        )
    )


def decode(data: bytes) -> Replay:
    """Unpack the wire format, rejecting anything that is not one of ours."""
    if len(data) < _HEADER.size:
        raise ReplayError("truncated replay")
    magic, contract_version, n_players, name_len = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ReplayError(f"not a replay: magic {magic!r}")

    offset = _HEADER.size
    map_name = data[offset : offset + name_len].decode()
    offset += name_len

    n_actions, seed, data_hash, rules_hash, final_hash = _BODY.unpack_from(data, offset)
    offset += _BODY.size

    scores = struct.unpack_from(f"<{n_players}h", data, offset)
    offset += 2 * n_players
    actions = struct.unpack_from(f"<{n_actions}H", data, offset)
    offset += 2 * n_actions
    if offset != len(data):
        raise ReplayError(f"trailing bytes: {len(data) - offset}")

    return Replay(
        map_name=map_name,
        n_players=n_players,
        seed=seed,
        actions=actions,
        data_hash=data_hash.hex(),
        rules_hash=rules_hash.hex(),
        final_hash=final_hash,
        final_scores=scores,
        contract_version=contract_version,
    )


def pack(records: list[Replay]) -> bytes:
    """Concatenate replays with a u32 length prefix each, for a corpus in one file."""
    out = [len(records).to_bytes(4, "little")]
    for rec in records:
        blob = encode(rec)
        out.append(len(blob).to_bytes(4, "little"))
        out.append(blob)
    return b"".join(out)


def unpack(data: bytes) -> list[Replay]:
    count = int.from_bytes(data[:4], "little")
    offset = 4
    records: list[Replay] = []
    for _ in range(count):
        size = int.from_bytes(data[offset : offset + 4], "little")
        offset += 4
        records.append(decode(data[offset : offset + size]))
        offset += size
    if offset != len(data):
        raise ReplayError(f"trailing bytes in pack: {len(data) - offset}")
    return records


def replay(rec: Replay, cfg: RuleConfig | None = None) -> State:
    """Re-run a recorded game and prove it reproduced.

    Every check here exists because its failure mode is otherwise silent. A replay taken
    on a different board, under different rules, or under a different PRNG will happily
    produce *a* game -- just not the one that was recorded.
    """
    if rec.contract_version != CONTRACT_VERSION:
        raise ReplayError(
            f"replay was recorded under contract version {rec.contract_version}, "
            f"this engine is version {CONTRACT_VERSION}; see docs/CONTRACT.md"
        )

    if cfg is None:
        cfg = RuleConfig(map_name=rec.map_name, n_players=rec.n_players)
    game = Game(cfg)

    if game.board.data_hash != rec.data_hash:
        raise ReplayError(
            f"board {rec.map_name!r} has changed since this replay was recorded "
            f"({rec.data_hash} != {game.board.data_hash})"
        )
    if game.cfg.rules_hash != rec.rules_hash:
        raise ReplayError(
            f"rules have changed since this replay was recorded "
            f"({rec.rules_hash} != {game.cfg.rules_hash})"
        )

    state = game.new_initial_state(rec.seed)
    for index, action in enumerate(rec.actions):
        if state.is_terminal():
            raise ReplayError(f"replay is longer than the game: {index} of {rec.n_actions}")
        state.step(action)

    if not state.is_terminal():
        raise ReplayError("replay ended before the game did")
    if state.state_hash() != rec.final_hash:
        raise ReplayError(
            f"replay diverged: final hash {state.state_hash():016x} != {rec.final_hash:016x}"
        )
    if tuple(final_scores(state)) != rec.final_scores:
        raise ReplayError(f"replay diverged on scores: {final_scores(state)} != {rec.final_scores}")
    return state
