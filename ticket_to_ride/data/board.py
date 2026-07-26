"""Derived board tables, built once at import from the generated constants.

Everything here is a pure function of `board_gen.py`, so it is deliberately *not*
generated: regenerating derived tables into two languages would double the surface area
for no gain, and these are all cheap to recompute (~1 ms for the USA map).

The tables exist because the engine's hot loops must not do graph work:

* `sibling` turns the double-route rules into an array lookup.
* `buckets` turns claim masking into ~33 affordability tests plus a short scan of only the
  affordable buckets, instead of a 900-way action scan (PLAN.md §5.3).
* `dist` is the all-pairs shortest path in train cars, the base of every ticket heuristic.

No torch, no numpy -- this module is imported by `engine/`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from ticket_to_ride.data.board_gen import MAPS as _RAW_MAPS
from ticket_to_ride.data.rawmap import GRAY, RawMap

#: Sentinel for "this segment has no twin". Same width as a segment id.
NO_SIBLING: Final = 255

#: Unreachable, in the all-pairs distance table. Large enough to add without overflow.
UNREACHABLE: Final = 10_000

#: A double route is exactly two parallel tracks; the sibling table assumes it.
TRACKS_PER_DOUBLE: Final = 2


class Board:
    """One map's immutable derived tables.

    Instances are built at import and shared by every `State`; nothing here is ever
    mutated, which is what lets `State` stay a flat POD clone.
    """

    __slots__ = (
        "adjacency",
        "buckets",
        "cities",
        "city_index",
        "color_names",
        "data_hash",
        "deck_composition",
        "deck_composition_counts",
        "deck_size",
        "dist",
        "locomotive",
        "max_len",
        "n_card_types",
        "n_cities",
        "n_colors",
        "n_pairs",
        "n_segments",
        "n_tickets",
        "name",
        "pair_of_segment",
        "raw",
        "route_points",
        "seg_a",
        "seg_b",
        "seg_bucket",
        "seg_color",
        "seg_len",
        "sibling",
        "ticket_a",
        "ticket_b",
        "ticket_points",
        "total_spaces",
    )

    def __init__(self, raw: RawMap) -> None:
        self.raw = raw
        self.name = raw.name
        self.cities = raw.cities
        self.color_names = raw.color_names
        self.route_points = raw.route_points
        self.data_hash = raw.data_hash

        self.n_cities = len(raw.cities)
        self.n_segments = len(raw.segments)
        self.n_tickets = len(raw.tickets)
        self.n_colors = raw.n_colors
        self.n_card_types = raw.n_card_types
        self.locomotive = raw.locomotive
        self.city_index = MappingProxyType({c: i for i, c in enumerate(raw.cities)})

        self.seg_a = tuple(s[0] for s in raw.segments)
        self.seg_b = tuple(s[1] for s in raw.segments)
        self.seg_len = tuple(s[2] for s in raw.segments)
        self.seg_color = tuple(s[3] for s in raw.segments)
        self.max_len = max(self.seg_len)
        self.total_spaces = sum(self.seg_len)

        self.ticket_a = tuple(t[0] for t in raw.tickets)
        self.ticket_b = tuple(t[1] for t in raw.tickets)
        self.ticket_points = tuple(t[2] for t in raw.tickets)

        self.deck_composition = tuple(
            [c for c in range(self.n_colors) for _ in range(raw.cards_per_color)]
            + [self.locomotive] * raw.locomotives
        )
        self.deck_size = len(self.deck_composition)
        self.deck_composition_counts = tuple(
            [raw.cards_per_color] * self.n_colors + [raw.locomotives]
        )

        self._build_pairs()
        self._build_buckets()
        self.dist = _all_pairs_distance(self.n_cities, self.adjacency, self.seg_len)

    # -- construction ------------------------------------------------------

    def _build_pairs(self) -> None:
        """Sibling table, city pair ids, and adjacency."""
        by_pair: dict[tuple[int, int], list[int]] = {}
        for s in range(self.n_segments):
            by_pair.setdefault((self.seg_a[s], self.seg_b[s]), []).append(s)

        sibling = [NO_SIBLING] * self.n_segments
        pair_of_segment = [0] * self.n_segments
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.n_cities)]

        for pair_id, ((a, b), segs) in enumerate(sorted(by_pair.items())):
            if len(segs) == TRACKS_PER_DOUBLE:
                sibling[segs[0]], sibling[segs[1]] = segs[1], segs[0]
            for s in segs:
                pair_of_segment[s] = pair_id
                adjacency[a].append((b, s))
                adjacency[b].append((a, s))

        self.n_pairs = len(by_pair)
        self.sibling = tuple(sibling)
        self.pair_of_segment = tuple(pair_of_segment)
        self.adjacency = tuple(tuple(row) for row in adjacency)

    def _build_buckets(self) -> None:
        """Group segments by `(length, color)`.

        Affordability of a claim depends only on the bucket, so the legal-action scan tests
        each bucket once and only then walks its segments. The USA map has 45 non-empty
        buckets against 100 segments (PLAN.md §5.3 guessed 33 during planning; 45 is the
        measured count), and a typical hand can afford well under half of them.
        """
        grouped: dict[tuple[int, int], list[int]] = {}
        for s in range(self.n_segments):
            grouped.setdefault((self.seg_len[s], self.seg_color[s]), []).append(s)

        buckets = sorted(grouped.items())
        self.buckets = tuple((length, color, tuple(segs)) for (length, color), segs in buckets)

        seg_bucket = [0] * self.n_segments
        for i, (_, _, segs) in enumerate(self.buckets):
            for s in segs:
                seg_bucket[s] = i
        self.seg_bucket = tuple(seg_bucket)

    # -- queries -----------------------------------------------------------

    def segment_name(self, s: int) -> str:
        color = "gray" if self.seg_color[s] == GRAY else self.color_names[self.seg_color[s]]
        return f"{self.cities[self.seg_a[s]]}-{self.cities[self.seg_b[s]]}:{self.seg_len[s]}{color}"

    def ticket_name(self, t: int) -> str:
        return (
            f"{self.cities[self.ticket_a[t]]}-{self.cities[self.ticket_b[t]]}"
            f"({self.ticket_points[t]})"
        )

    def color_name(self, c: int) -> str:
        if c == GRAY:
            return "gray"
        return "loco" if c == self.locomotive else self.color_names[c]

    def __repr__(self) -> str:
        return (
            f"<Board {self.name}: {self.n_cities} cities, {self.n_pairs} pairs, "
            f"{self.n_segments} segments, {self.total_spaces} spaces, "
            f"{self.n_tickets} tickets, {self.deck_size} cards>"
        )


def _all_pairs_distance(
    n_cities: int,
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
    seg_len: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Shortest path between every city pair, measured in train cars.

    Dijkstra from each source over the *cheapest* track of each pair. On 36 nodes this is
    a handful of milliseconds once at import; the alternative (Floyd-Warshall) is simpler
    but 36x more work and no clearer.
    """
    best_edge: list[dict[int, int]] = [{} for _ in range(n_cities)]
    for city, row in enumerate(adjacency):
        for nb, seg in row:
            w = seg_len[seg]
            if w < best_edge[city].get(nb, UNREACHABLE):
                best_edge[city][nb] = w

    out: list[tuple[int, ...]] = []
    for src in range(n_cities):
        dist = [UNREACHABLE] * n_cities
        dist[src] = 0
        done = [False] * n_cities
        for _ in range(n_cities):
            u, best = -1, UNREACHABLE
            for v in range(n_cities):
                if not done[v] and dist[v] < best:
                    u, best = v, dist[v]
            if u < 0:
                break
            done[u] = True
            for nb, w in best_edge[u].items():
                dist[nb] = min(dist[nb], best + w)
        out.append(tuple(dist))
    return tuple(out)


BOARDS: Final = MappingProxyType({name: Board(raw) for name, raw in _RAW_MAPS.items()})
USA: Final = BOARDS["usa"]
MINI: Final = BOARDS["mini"]


def get_board(name: str = "usa") -> Board:
    """Look up a board by name, with a message that lists the alternatives."""
    try:
        return BOARDS[name]
    except KeyError:
        raise KeyError(f"unknown map {name!r}; known maps: {sorted(BOARDS)}") from None
