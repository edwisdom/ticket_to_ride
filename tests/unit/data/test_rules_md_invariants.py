"""Structural invariants of RULES.md.

RULES.md is the human-readable source of truth for the board. In Phase 1 the generator
will transcribe it into checked-in constants and this file will additionally assert the
generated data still matches. For now it guards the Step 0 normalization (pink -> purple,
Montreal de-accented) and the counts the whole engine design is sized against.

The ticket-endpoint check is the one that matters most: before normalization, three of the
thirty tickets spelled the city "Montreal" with an accent while the map table did not, so a
naive parser silently dropped 10% of the ticket deck or raised KeyError.
"""

from __future__ import annotations

from collections import Counter

CITIES = 36
PAIRS = 78
SEGMENTS = 100
SPACES = 309
TICKETS = 30
DOUBLE_PAIRS = 22
ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
COLORS = frozenset({"black", "blue", "green", "orange", "purple", "red", "white", "yellow"})


def _table(lines: list[str], header: str) -> list[list[str]]:
    """Rows of the markdown table introduced by `header`, as stripped cell lists.

    Double routes are written with markdown-escaped pipes (`gray \\|\\| gray`). Splitting
    naively on "|" silently yields 78 segments / 256 spaces instead of 100 / 309, so the
    escapes are protected before the split and restored after.
    """
    start = lines.index(header) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        protected = line.replace(r"\|", "\x00")
        rows.append([c.strip().replace("\x00", "|") for c in protected.strip("|").split("|")])
    return rows


def _segments(lines: list[str]) -> list[tuple[str, str, int, str]]:
    out: list[tuple[str, str, int, str]] = []
    for a, b, length, colors in _table(lines, "| From | To | Length | Color |"):
        out.extend((a, b, int(length), c.strip()) for c in colors.split("||"))
    return out


def _tickets(lines: list[str]) -> list[tuple[str, str, int]]:
    rows = _table(lines, "| From | To | Points |")
    return [(a, b, int(pts)) for a, b, pts in rows]


def test_no_unnormalized_spellings(rules_md: str) -> None:
    assert "pink" not in rules_md.lower()
    assert "Montréal" not in rules_md


def test_counts(rules_md: str) -> None:
    lines = rules_md.splitlines()
    segments = _segments(lines)
    cities = {s[0] for s in segments} | {s[1] for s in segments}
    pairs = {frozenset((s[0], s[1])) for s in segments}

    assert len(cities) == CITIES
    assert len(pairs) == PAIRS
    assert len(segments) == SEGMENTS
    assert sum(s[2] for s in segments) == SPACES
    assert len(_tickets(lines)) == TICKETS


def test_color_balance(rules_md: str) -> None:
    """44 gray plus exactly 7 segments of each of the 8 colors -- a strong structural check."""
    hist = Counter(s[3] for s in _segments(rules_md.splitlines()))
    assert hist.pop("gray") == 44
    assert set(hist) == COLORS
    assert set(hist.values()) == {7}


def test_double_routes(rules_md: str) -> None:
    per_pair = Counter(frozenset((s[0], s[1])) for s in _segments(rules_md.splitlines()))
    doubles = {pair: n for pair, n in per_pair.items() if n > 1}
    assert len(doubles) == DOUBLE_PAIRS
    assert set(doubles.values()) == {2}, "no pair should have more than two tracks"
    # 2-3P blocks one track of each double route.
    assert SEGMENTS - len(doubles) == PAIRS


def test_lengths_are_scorable(rules_md: str) -> None:
    lengths = Counter(s[2] for s in _segments(rules_md.splitlines()))
    assert set(lengths) <= set(ROUTE_POINTS)
    assert dict(lengths) == {1: 9, 2: 36, 3: 20, 4: 16, 5: 10, 6: 9}


def test_every_ticket_endpoint_is_a_known_city(rules_md: str) -> None:
    lines = rules_md.splitlines()
    segments = _segments(lines)
    cities = {s[0] for s in segments} | {s[1] for s in segments}
    for a, b, _ in _tickets(lines):
        assert a in cities, f"ticket endpoint {a!r} is not a city on the map"
        assert b in cities, f"ticket endpoint {b!r} is not a city on the map"
        assert a != b


def test_map_is_connected(rules_md: str) -> None:
    adjacency: dict[str, set[str]] = {}
    for a, b, _, _ in _segments(rules_md.splitlines()):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    start = next(iter(adjacency))
    seen, stack = {start}, [start]
    while stack:
        for neighbour in adjacency[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    assert len(seen) == CITIES


def test_city_index_table_agrees_with_edge_list(rules_md: str) -> None:
    """The City Index is an independent redundancy check on the edge list."""
    lines = rules_md.splitlines()
    derived: dict[str, set[str]] = {}
    for a, b, _, _ in _segments(lines):
        derived.setdefault(a, set()).add(b)
        derived.setdefault(b, set()).add(a)

    for city, degree, connects in _table(lines, "| City | Degree | Connects to |"):
        listed = {c.strip() for c in connects.split(",")}
        assert listed == derived[city], f"City Index disagrees with the edge list for {city}"
        assert int(degree) == len(derived[city])

    assert sum(len(v) for v in derived.values()) == 2 * PAIRS
