#!/usr/bin/env python3
"""Generate the golden replay corpus in tests/golden/replays.bin.

    uv run python tools/gen_replays.py --check    # fail if the engine stopped reproducing
    uv run python tools/gen_replays.py --write    # rewrite (an intentional rules change)

This is the strongest regression guard in the project: a checked-in set of finished games
across every map and seat count, each carrying the actions that produced it and the state
hash and scores it must produce again. Any change to the rules, the draw procedure, the
PRNG or the scoring shows up here as a concrete failing game rather than as a subtly
different win rate three weeks later.

`--check` is not "regenerate if it drifted". A diff here means the engine plays a different
game than it did, and the only correct responses are to fix the regression or to accept it
deliberately, bump `CONTRACT_VERSION` if the contract moved, and rewrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ticket_to_ride.agents.registry import make_agent  # noqa: E402
from ticket_to_ride.data.board import BOARDS  # noqa: E402
from ticket_to_ride.engine.config import RuleConfig  # noqa: E402
from ticket_to_ride.engine.replay import Replay, pack, record, unpack  # noqa: E402
from ticket_to_ride.engine.rng import stream  # noqa: E402
from ticket_to_ride.engine.state import Game  # noqa: E402

OUT = REPO_ROOT / "tests" / "golden" / "replays.bin"

#: Seeds per configuration. Small enough to stay a fast test, wide enough that reshuffles,
#: locomotive flushes and the end trigger all appear somewhere in the corpus.
SEEDS = 6


def build() -> list[Replay]:
    """Random games and greedy games, across every map and seat count."""
    records: list[Replay] = []
    for name, board in BOARDS.items():
        for n_players in range(board.raw.min_players, board.raw.max_players + 1):
            game = Game(RuleConfig(map_name=name, n_players=n_players))
            for seed in range(SEEDS):
                state = game.new_initial_state(seed)
                rng = stream(seed, "policy")
                while not state.is_terminal():
                    state.step(state.sample_legal(rng))
                records.append(record(state))

            # A greedy game too: random play rarely reaches long routes or a deep endgame.
            agents = [make_agent("h1", 10 + seat) for seat in range(n_players)]
            for seed in range(SEEDS):
                for seat, agent in enumerate(agents):
                    agent.begin_game(seat, seed)
                state = game.new_initial_state(seed)
                while not state.is_terminal():
                    state.step(agents[state.current_player()].act(state))
                records.append(record(state))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    records = build()
    blob = pack(records)

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(blob)
        print(f"wrote {len(records)} replays, {len(blob):,} bytes")
        return 0

    if not OUT.exists():
        print(f"missing {OUT.relative_to(REPO_ROOT)}; run --write", file=sys.stderr)
        return 1
    stored = unpack(OUT.read_bytes())
    if stored != records:
        differing = [
            (a.map_name, a.n_players, a.seed)
            for a, b in zip(stored, records, strict=False)
            if a != b
        ]
        print(
            "REGRESSION: the engine no longer plays the recorded games.\n"
            f"first differing: {differing[:5]}\n"
            "Fix it, or accept it deliberately and rerun with --write.",
            file=sys.stderr,
        )
        return 1
    print(f"{len(records)} golden replays reproduce")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
