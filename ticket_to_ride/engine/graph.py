"""Graph work the engine needs: union-find, reachability costs, and the longest trail.

Deliberately not networkx. The two things we actually need are ~15 and ~40 lines, run
millions of times, and networkx has no longest-trail function at all.

Everything here is a pure function of its arguments. No state, no caching -- callers that
want caching (per player, per board version) own it, because they know when it is stale.
"""

from __future__ import annotations

import heapq
from typing import Final

from ticket_to_ride.data.board import NO_SIBLING, UNREACHABLE, Board
from ticket_to_ride.engine.config import FREE

#: The DFS in `longest_trail` keeps used edges in one machine word.
MAX_TRAIL_EDGES: Final = 32

#: Fewer than two terminals is nothing to connect.
MIN_TERMINALS: Final = 2

#: Above this, exact Steiner is too slow to be worth it and we fall back (see
#: `steiner_cost`). Held-ticket terminal sets are almost always far smaller.
MAX_STEINER_TERMINALS: Final = 8


# ---------------------------------------------------------------------------
# Union-find, one per player over the cities
# ---------------------------------------------------------------------------


def dsu_find(parent: bytearray, x: int) -> int:
    """Path *halving*: one pointer update per step, no second pass, no recursion.

    A clone-based state can compress freely; an undo-log design could not, which is one of
    the reasons `State` is `Copy` (PLAN.md §5.1).
    """
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def dsu_union(parent: bytearray, a: int, b: int) -> bool:
    """Union by smaller index, so the forest is a deterministic function of the claims."""
    ra, rb = dsu_find(parent, a), dsu_find(parent, b)
    if ra == rb:
        return False
    if ra > rb:
        ra, rb = rb, ra
    parent[rb] = ra
    return True


def dsu_connected(parent: bytearray, a: int, b: int) -> bool:
    return dsu_find(parent, a) == dsu_find(parent, b)


# ---------------------------------------------------------------------------
# Remaining cost -- "how many more train cars do I need?"
# ---------------------------------------------------------------------------


def edge_cost(
    board: Board,
    seg_owner: bytearray,
    player: int,
    segment: int,
    doubles_locked: bool,
) -> int:
    """Cost in train cars for `player` to traverse `segment`, or `UNREACHABLE`.

    Already mine: free. Unclaimed and still claimable by me: its length. Anyone else's,
    closed, or the sibling of a track I already own in a 4-5P game: impassable.
    """
    owner = seg_owner[segment]
    if owner == player:
        return 0
    if owner != FREE:
        return UNREACHABLE
    if not doubles_locked:
        # 4-5P: the sibling stays open to *others*, never to me.
        sibling = board.sibling[segment]
        if sibling != NO_SIBLING and seg_owner[sibling] == player:
            return UNREACHABLE
    return board.seg_len[segment]


def remaining_costs_from(
    board: Board,
    seg_owner: bytearray,
    player: int,
    source: int,
    doubles_locked: bool,
) -> list[int]:
    """Dijkstra from `source` with my track free, free track priced, enemy track blocked.

    The result is "train cars still needed" to reach each city. `UNREACHABLE` means the
    connection is dead -- an opponent has cut every route -- which is exactly the signal a
    ticket-valuation heuristic needs and the `is_dead` observation feature reports.
    """
    dist = [UNREACHABLE] * board.n_cities
    dist[source] = 0
    queue: list[tuple[int, int]] = [(0, source)]
    adjacency = board.adjacency
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue
        for nb, segment in adjacency[u]:
            w = edge_cost(board, seg_owner, player, segment, doubles_locked)
            if w >= UNREACHABLE:
                continue
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(queue, (nd, nb))
    return dist


def steiner_cost(
    board: Board,
    seg_owner: bytearray,
    player: int,
    terminals: list[int],
    doubles_locked: bool,
) -> int:
    """Cheapest set of new track connecting every terminal. Dreyfus-Wagner.

    Why this and not the sum of per-ticket shortest paths: those double-count shared trunk
    lines, so a player holding Seattle-New York and Portland-New York looks like it needs
    two transcontinentals when it needs one and a spur. The Steiner cost is the truth, and
    it is what the H2+ heuristics plan against.

    Terminals already connected to each other (cost 0 between them) are contracted first,
    which is what usually keeps the subset DP small: mid-game most held tickets share a
    component. Above `MAX_STEINER_TERMINALS` distinct components the DP is abandoned and a
    minimum-spanning-tree approximation over the terminal metric is returned instead --
    documented, upper-bounding, and never silent (`exact=False`).
    """
    return _steiner(board, seg_owner, player, terminals, doubles_locked)[0]


def steiner_cost_exact(
    board: Board,
    seg_owner: bytearray,
    player: int,
    terminals: list[int],
    doubles_locked: bool,
) -> tuple[int, bool]:
    """`steiner_cost`, plus whether the result is exact or the MST upper bound."""
    return _steiner(board, seg_owner, player, terminals, doubles_locked)


def _steiner(
    board: Board,
    seg_owner: bytearray,
    player: int,
    terminals: list[int],
    doubles_locked: bool,
) -> tuple[int, bool]:
    if len(terminals) < MIN_TERMINALS:
        return 0, True

    # Contract terminals that already cost nothing to reach from one another.
    distances = {
        t: remaining_costs_from(board, seg_owner, player, t, doubles_locked) for t in set(terminals)
    }
    roots: list[int] = []
    for t in sorted(distances):
        if not any(distances[t][r] == 0 for r in roots):
            roots.append(t)
    if len(roots) < MIN_TERMINALS:
        return 0, True
    if any(distances[roots[0]][r] >= UNREACHABLE for r in roots):
        return UNREACHABLE, True

    if len(roots) > MAX_STEINER_TERMINALS:
        return _mst_bound(roots, distances), False

    n = board.n_cities
    full = (1 << len(roots)) - 1
    # dp[mask][v] = cheapest tree spanning the terminals in `mask` plus the vertex v.
    dp = [[UNREACHABLE] * n for _ in range(full + 1)]
    for i, t in enumerate(roots):
        row = dp[1 << i]
        source = distances[t]
        for v in range(n):
            row[v] = source[v]

    for mask in range(1, full + 1):
        if mask & (mask - 1) == 0:
            continue  # single terminal: already initialized from its Dijkstra
        row = dp[mask]
        sub = (mask - 1) & mask
        while sub:
            other = dp[mask ^ sub]
            left = dp[sub]
            for v in range(n):
                combined = left[v] + other[v]
                row[v] = min(row[v], combined)
            sub = (sub - 1) & mask
        _relax(board, seg_owner, player, row, doubles_locked)

    return min(dp[full]), True


def _relax(
    board: Board,
    seg_owner: bytearray,
    player: int,
    dist: list[int],
    doubles_locked: bool,
) -> None:
    """Multi-source Dijkstra in place: allow the tree's root to slide along cheap track."""
    queue = [(d, v) for v, d in enumerate(dist) if d < UNREACHABLE]
    heapq.heapify(queue)
    adjacency = board.adjacency
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue
        for nb, segment in adjacency[u]:
            w = edge_cost(board, seg_owner, player, segment, doubles_locked)
            if w >= UNREACHABLE:
                continue
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(queue, (nd, nb))


def _mst_bound(roots: list[int], distances: dict[int, list[int]]) -> int:
    """Prim over the terminal metric. Always >= the Steiner cost, never < it."""
    inside = {roots[0]}
    total = 0
    while len(inside) < len(roots):
        best, best_t = UNREACHABLE, -1
        for t in roots:
            if t in inside:
                continue
            for u in inside:
                if distances[u][t] < best:
                    best, best_t = distances[u][t], t
        if best_t < 0:
            return UNREACHABLE
        inside.add(best_t)
        total += best
    return total


# ---------------------------------------------------------------------------
# Longest continuous path -- a longest *trail*, weighted in train cars
# ---------------------------------------------------------------------------


def longest_trail(board: Board, seg_owner: bytearray, player: int) -> int:
    """The longest continuous path bonus, measured in train cars.

    It is a **trail**, not a path and not a segment count: each segment may be used at most
    once, cities may repeat, and loops are allowed. Four layers, in order, because the
    naive version has a 126 ms tail that shows up as a p99.9 latency spike in self-play:

    1. **Split into connected components.** The answer is the max over components; a
       6-edge and a 4-edge component are two tiny searches instead of one 10-edge one.
    2. **Eulerian shortcut.** A component with 0 or 2 odd-degree vertices admits a trail
       using *every* edge, so the answer is the total weight with no search at all. This
       fires on a large fraction of real player subgraphs.
    3. **Memoized DFS** on `(vertex, used_edge_bitmask)`, the mask in one machine word.
    4. **Early exit** as soon as the component's total weight is reached.
    """
    owned = owned_segments(seg_owner, player)
    if not owned:
        return 0

    adjacency: dict[int, list[tuple[int, int, int]]] = {}
    for local, segment in enumerate(owned):
        a, b, w = board.seg_a[segment], board.seg_b[segment], board.seg_len[segment]
        adjacency.setdefault(a, []).append((b, local, w))
        adjacency.setdefault(b, []).append((a, local, w))

    best = 0
    for component in _components(adjacency):
        best = max(best, _component_longest_trail(adjacency, component))
    return best


def _components(adjacency: dict[int, list[tuple[int, int, int]]]) -> list[list[int]]:
    seen: set[int] = set()
    out: list[list[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            component.append(v)
            for nb, _, _ in adjacency[v]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(component)
    return out


def _component_longest_trail(
    adjacency: dict[int, list[tuple[int, int, int]]],
    component: list[int],
) -> int:
    edges = {local: w for v in component for _, local, w in adjacency[v]}
    total = sum(edges.values())

    odd = sum(1 for v in component if len(adjacency[v]) % 2)
    if odd in (0, 2):
        return total  # layer 2: an Eulerian trail uses every edge

    if len(edges) > MAX_TRAIL_EDGES:
        raise ValueError(
            f"longest-trail component has {len(edges)} edges, above the {MAX_TRAIL_EDGES}-bit "
            "mask limit; the train supply should make this unreachable"
        )

    memo: dict[tuple[int, int], int] = {}

    def best_from(v: int, used: int) -> int:
        key = (v, used)
        cached = memo.get(key)
        if cached is not None:
            return cached
        best = 0
        for nb, local, w in adjacency[v]:
            bit = 1 << local
            if used & bit:
                continue
            candidate = w + best_from(nb, used | bit)
            best = max(best, candidate)
        memo[key] = best
        return best

    overall = 0
    for start in component:
        overall = max(overall, best_from(start, 0))
        if overall == total:
            break  # layer 4: cannot do better than every edge
    return overall


def owned_segments(seg_owner: bytearray, player: int) -> list[int]:
    """Segment ids `player` has claimed. `CLOSED` siblings are nobody's, and excluded."""
    return [s for s, owner in enumerate(seg_owner) if owner == player]
