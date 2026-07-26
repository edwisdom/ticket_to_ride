"""Union-find, reachability costs, the Steiner tree, and the longest trail.

The longest trail gets an independent oracle: a brute force over *permutations* of the
owned segments, which shares no code with the optimized version. That matters because the
optimization has four layers -- component split, Eulerian shortcut, memoized DFS, early
exit -- and a bug in any of them would otherwise be invisible.
"""

from __future__ import annotations

import itertools

import pytest

from ticket_to_ride.data.board import MINI, UNREACHABLE, USA, Board
from ticket_to_ride.engine import CLOSED, FREE, Game, RuleConfig
from ticket_to_ride.engine.graph import (
    MAX_TRAIL_EDGES,
    dsu_connected,
    dsu_find,
    dsu_union,
    edge_cost,
    longest_trail,
    owned_segments,
    remaining_costs_from,
    steiner_cost,
    steiner_cost_exact,
)
from ticket_to_ride.engine.rng import stream


def segment_named(board: Board, a: str, b: str, *, which: int = 0) -> int:
    ia, ib = sorted((board.city_index[a], board.city_index[b]))
    found = [s for s in range(board.n_segments) if (board.seg_a[s], board.seg_b[s]) == (ia, ib)]
    return found[which]


def own(board: Board, segments: list[int], player: int = 0) -> bytearray:
    owner = bytearray([FREE]) * board.n_segments
    for s in segments:
        owner[s] = player
    return owner


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------


def test_union_find_basics() -> None:
    parent = bytearray(range(10))
    assert dsu_find(parent, 7) == 7
    assert not dsu_connected(parent, 1, 2)
    assert dsu_union(parent, 1, 2)
    assert dsu_connected(parent, 1, 2)
    assert not dsu_union(parent, 2, 1), "a redundant union reports no change"
    dsu_union(parent, 2, 3)
    assert dsu_connected(parent, 1, 3), "connectivity is transitive"
    assert not dsu_connected(parent, 1, 4)


def test_union_find_is_deterministic_regardless_of_merge_order() -> None:
    """Union by smaller index, so the forest is a function of the claims, not their order."""
    a, b = bytearray(range(8)), bytearray(range(8))
    for x, y in [(1, 2), (3, 4), (2, 3)]:
        dsu_union(a, x, y)
    for x, y in [(3, 4), (2, 3), (1, 2)]:
        dsu_union(b, x, y)
    for i in range(8):
        assert dsu_find(a, i) == dsu_find(b, i)


def test_path_halving_shortens_chains() -> None:
    """Halving, not full compression: one pointer update per step and no second pass.

    A single find does not flatten the whole chain -- it halves it -- so repeated finds
    converge instead of paying a two-pass walk every time.
    """

    def depth(parent: bytearray, x: int) -> int:
        steps = 0
        while parent[x] != x:
            x = parent[x]
            steps += 1
        return steps

    parent = bytearray(range(20))
    for i in range(19, 0, -1):
        parent[i] = i - 1  # a degenerate chain
    assert depth(parent, 19) == 19
    assert dsu_find(parent, 19) == 0
    assert depth(parent, 19) < 19, "the chain must get shorter"
    for _ in range(6):
        dsu_find(parent, 19)
    assert parent[19] == 0, "repeated finds converge to a flat forest"


# ---------------------------------------------------------------------------
# Edge costs and remaining cost
# ---------------------------------------------------------------------------


def test_edge_cost_prices_track_by_who_owns_it() -> None:
    segment = segment_named(USA, "Atlanta", "Miami")  # length 5
    owner = bytearray([FREE]) * USA.n_segments
    assert edge_cost(USA, owner, 0, segment, doubles_locked=True) == 5, (
        "free track costs its length"
    )
    owner[segment] = 0
    assert edge_cost(USA, owner, 0, segment, doubles_locked=True) == 0, "my own track is free"
    owner[segment] = 1
    assert edge_cost(USA, owner, 0, segment, doubles_locked=True) >= UNREACHABLE
    owner[segment] = CLOSED
    assert edge_cost(USA, owner, 0, segment, doubles_locked=True) >= UNREACHABLE


def test_edge_cost_blocks_a_double_i_already_half_own_in_four_or_five_player_games() -> None:
    segment = segment_named(USA, "Boston", "Montreal")
    sibling = USA.sibling[segment]
    owner = bytearray([FREE]) * USA.n_segments
    owner[segment] = 0
    assert edge_cost(USA, owner, 0, sibling, doubles_locked=False) >= UNREACHABLE
    assert edge_cost(USA, owner, 1, sibling, doubles_locked=False) == USA.seg_len[sibling]


def test_remaining_cost_on_an_empty_board_is_the_static_distance() -> None:
    owner = bytearray([FREE]) * USA.n_segments
    dist = remaining_costs_from(USA, owner, 0, 0, doubles_locked=True)
    assert dist == list(USA.dist[0])


def test_owning_track_makes_it_free() -> None:
    src = USA.city_index["Atlanta"]
    dst = USA.city_index["Miami"]
    empty = bytearray([FREE]) * USA.n_segments
    before = remaining_costs_from(USA, empty, 0, src, doubles_locked=True)[dst]
    owner = own(USA, [segment_named(USA, "Atlanta", "Miami")])
    after = remaining_costs_from(USA, owner, 0, src, doubles_locked=True)[dst]
    assert before == 5
    assert after == 0


def test_an_opponent_can_cut_a_connection_dead() -> None:
    """`UNREACHABLE` is the `is_dead` signal the ticket features report."""
    owner = bytearray([FREE]) * USA.n_segments
    vancouver = USA.city_index["Vancouver"]
    for nb, segment in USA.adjacency[vancouver]:
        del nb
        owner[segment] = 1
    dist = remaining_costs_from(USA, owner, 0, vancouver, doubles_locked=True)
    assert dist[USA.city_index["Denver"]] >= UNREACHABLE


# ---------------------------------------------------------------------------
# Steiner tree
# ---------------------------------------------------------------------------


def test_steiner_of_fewer_than_two_terminals_is_zero() -> None:
    owner = bytearray([FREE]) * USA.n_segments
    assert steiner_cost(USA, owner, 0, [], doubles_locked=True) == 0
    assert steiner_cost(USA, owner, 0, [3], doubles_locked=True) == 0


def test_steiner_of_two_terminals_is_the_shortest_path() -> None:
    owner = bytearray([FREE]) * USA.n_segments
    for a, b in [("Denver", "El Paso"), ("Seattle", "New York"), ("Boston", "Miami")]:
        ia, ib = USA.city_index[a], USA.city_index[b]
        assert steiner_cost(USA, owner, 0, [ia, ib], doubles_locked=True) == USA.dist[ia][ib]


def test_steiner_never_exceeds_the_sum_of_separate_paths() -> None:
    """The whole point: per-ticket Dijkstras double-count the shared trunk line."""
    owner = bytearray([FREE]) * USA.n_segments
    hub = USA.city_index["Chicago"]
    spokes = [USA.city_index[c] for c in ("Seattle", "Miami", "Los Angeles")]
    separate = sum(USA.dist[hub][s] for s in spokes)
    together = steiner_cost(USA, owner, 0, [hub, *spokes], doubles_locked=True)
    assert together <= separate
    assert together < separate, "these three share the trunk out of Chicago"


def test_steiner_is_zero_once_the_terminals_are_connected() -> None:
    board = USA
    owner = own(board, [segment_named(board, "Nashville", "Atlanta")])
    a = board.city_index["Nashville"]
    b = board.city_index["Atlanta"]
    assert steiner_cost(board, owner, 0, [a, b], doubles_locked=True) == 0


def test_steiner_falls_back_to_an_upper_bound_past_the_terminal_cap() -> None:
    owner = bytearray([FREE]) * USA.n_segments
    terminals = list(range(10))
    cost, exact = steiner_cost_exact(USA, owner, 0, terminals, doubles_locked=True)
    assert not exact, "ten terminals must take the documented MST fallback"
    assert cost > 0
    tree, _ = steiner_cost_exact(USA, owner, 0, terminals[:4], doubles_locked=True)
    assert tree > 0


def test_steiner_reports_unreachable_when_a_terminal_is_walled_off() -> None:
    owner = bytearray([FREE]) * USA.n_segments
    vancouver = USA.city_index["Vancouver"]
    for _, segment in USA.adjacency[vancouver]:
        owner[segment] = 1
    cost = steiner_cost(USA, owner, 0, [vancouver, USA.city_index["Miami"]], doubles_locked=True)
    assert cost >= UNREACHABLE


def test_steiner_of_three_terminals_matches_the_closed_form() -> None:
    """A three-terminal Steiner tree has at most one branch vertex, which gives an oracle.

    The optimum is `min over v of d(v,a) + d(v,b) + d(v,c)` in the shortest-path metric:
    any three-terminal tree decomposes into three paths meeting at one vertex, and where
    two of those paths overlap, the branch vertex simply slides along the overlap. That is
    a genuinely independent check on the subset DP -- it shares no code with it. Run over
    every triple on TTR-mini and a sample on the USA map.
    """
    for board, triples in (
        (MINI, itertools.combinations(range(MINI.n_cities), 3)),
        (USA, itertools.combinations(range(0, USA.n_cities, 5), 3)),
    ):
        owner = bytearray([FREE]) * board.n_segments
        for a, b, c in triples:
            expected = min(
                board.dist[v][a] + board.dist[v][b] + board.dist[v][c]
                for v in range(board.n_cities)
            )
            got = steiner_cost(board, owner, 0, [a, b, c], doubles_locked=True)
            assert got == expected, f"{board.name} {(a, b, c)}: {got} != {expected}"


# ---------------------------------------------------------------------------
# Longest trail -- against an independent oracle
# ---------------------------------------------------------------------------


def _is_trail(board: Board, order: tuple[int, ...]) -> bool:
    """Does this exact sequence of segments form a walk that reuses no segment?"""
    for start in (board.seg_a[order[0]], board.seg_b[order[0]]):
        here = board.seg_b[order[0]] if start == board.seg_a[order[0]] else board.seg_a[order[0]]
        ok = True
        for segment in order[1:]:
            a, b = board.seg_a[segment], board.seg_b[segment]
            if here == a:
                here = b
            elif here == b:
                here = a
            else:
                ok = False
                break
        if ok:
            return True
    return False


def brute_force_longest_trail(board: Board, segments: list[int]) -> int:
    """The oracle: every ordering, longest valid prefix. Shares no code with the engine."""
    best = 0
    for order in itertools.permutations(segments):
        for size in range(len(order), 0, -1):
            prefix = order[:size]
            weight = sum(board.seg_len[s] for s in prefix)
            if weight <= best:
                break
            if _is_trail(board, prefix):
                best = weight
                break
    return best


@pytest.mark.parametrize(
    "pairs",
    [
        [("Nashville", "Atlanta")],
        [("Nashville", "Atlanta"), ("Atlanta", "Miami")],
        # A triangle: an Eulerian circuit, so the answer is every edge.
        [("Nashville", "Atlanta"), ("Atlanta", "Raleigh"), ("Raleigh", "Nashville")],
        # Two disconnected pieces: the answer is the bigger one, not the sum.
        [("Nashville", "Atlanta"), ("Vancouver", "Calgary")],
        # A star at Denver: degree 4, so no Eulerian trail and the DFS has to run.
        [
            ("Denver", "Omaha"),
            ("Denver", "Santa Fe"),
            ("Denver", "Helena"),
            ("Denver", "Phoenix"),
        ],
        # A hub with a tail, mixing the layers.
        [
            ("Denver", "Omaha"),
            ("Denver", "Santa Fe"),
            ("Denver", "Helena"),
            ("Denver", "Phoenix"),
            ("Santa Fe", "El Paso"),
            ("Omaha", "Chicago"),
        ],
    ],
)
def test_longest_trail_matches_brute_force(pairs: list[tuple[str, str]]) -> None:
    segments = [segment_named(USA, a, b) for a, b in pairs]
    owner = own(USA, segments)
    assert longest_trail(USA, owner, 0) == brute_force_longest_trail(USA, segments)


def test_longest_trail_matches_brute_force_on_real_game_subgraphs() -> None:
    """Positions a random game actually reaches, capped where the oracle stays affordable."""
    game = Game(RuleConfig(n_players=2))
    checked = 0
    for seed in range(40):
        state = game.new_initial_state(seed)
        rng = stream(seed, "policy")
        while not state.is_terminal():
            state.step(state.sample_legal(rng))
            for p in range(2):
                segments = owned_segments(state.seg_owner, p)
                if not 1 <= len(segments) <= 7:
                    continue
                assert longest_trail(game.board, state.seg_owner, p) == brute_force_longest_trail(
                    game.board, segments
                )
                checked += 1
                if checked > 300:
                    return
    assert checked > 50, f"only {checked} subgraphs were checked"


def test_longest_trail_of_nothing_is_zero() -> None:
    assert longest_trail(USA, bytearray([FREE]) * USA.n_segments, 0) == 0


def test_the_eulerian_shortcut_fires_and_is_right() -> None:
    """A path (2 odd vertices) and a circuit (0) both admit a trail over every edge."""
    path = [
        segment_named(USA, "Vancouver", "Calgary"),
        segment_named(USA, "Calgary", "Helena"),
        segment_named(USA, "Helena", "Denver"),
    ]
    owner = own(USA, path)
    assert longest_trail(USA, owner, 0) == sum(USA.seg_len[s] for s in path)


def test_longest_trail_ignores_other_players_and_closed_track() -> None:
    owner = bytearray([FREE]) * USA.n_segments
    mine = segment_named(USA, "Atlanta", "Miami")
    owner[mine] = 0
    owner[segment_named(USA, "Nashville", "Atlanta")] = 1
    owner[segment_named(USA, "Charleston", "Atlanta")] = CLOSED
    assert longest_trail(USA, owner, 0) == USA.seg_len[mine]


def test_a_component_wider_than_the_bitmask_is_rejected_loudly() -> None:
    """Better a clear error than a silently truncated mask. The train supply prevents it."""
    board = USA
    owner = bytearray([FREE]) * board.n_segments
    # A long chain plus a hub, forced past the 32-edge limit and made non-Eulerian.
    hub = board.city_index["Denver"]
    chain = [s for s in range(board.n_segments) if board.seg_len[s] <= 2]
    for s in chain[: MAX_TRAIL_EDGES + 8]:
        owner[s] = 0
    del hub
    try:
        longest_trail(board, owner, 0)
    except ValueError as exc:
        assert "mask limit" in str(exc)
