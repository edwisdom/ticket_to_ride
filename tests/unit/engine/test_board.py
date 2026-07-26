"""Board data invariants -- the generated constants and everything derived from them.

`test_rules_md_invariants.py` checks the markdown; this checks that the generator
transcribed it faithfully and that the derived tables agree with the raw data. The two
overlap on purpose: the whole point of the City Index cross-check is redundancy.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ticket_to_ride.data.board import BOARDS, MINI, NO_SIBLING, UNREACHABLE, USA, Board, get_board
from ticket_to_ride.data.rawmap import GRAY

ALL_BOARDS = pytest.mark.parametrize("board", list(BOARDS.values()), ids=list(BOARDS))


# ---------------------------------------------------------------------------
# The generator itself
# ---------------------------------------------------------------------------


def test_generated_files_are_not_stale(repo_root: Path) -> None:
    """`board_gen.{py,rs}` and usa.toml must reproduce byte-identically.

    This is the check that makes "the Python and Rust engines cannot disagree about the
    board" true rather than aspirational -- both files come out of one canonicalized spec,
    and drift in either is a hard failure here.
    """
    result = subprocess.run(
        [sys.executable, "tools/gen_board.py", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rust_and_python_agree_on_every_constant(repo_root: Path) -> None:
    """Parse the generated Rust back out and diff it against the Python tuples.

    Cheap, and it catches an emitter bug that would otherwise only surface in Phase 2 as a
    mysterious differential-test failure.
    """
    rust = (repo_root / "crates" / "ttr-core" / "src" / "board_gen.rs").read_text()
    for board in BOARDS.values():
        up = board.name.upper()
        assert f'    name: "{board.name}",' in rust
        assert f'    data_hash: "{board.data_hash}",' in rust
        assert f"pub const {up}_SEGMENTS: [(u8, u8, u8, u8); {board.n_segments}]" in rust
        assert f"pub const {up}_TICKETS: [(u8, u8, u8); {board.n_tickets}]" in rust
        assert f"pub const {up}_CITIES: [&str; {board.n_cities}]" in rust
        assert f"    trains_per_player: {board.raw.trains_per_player}," in rust
        block = rust.split(f"pub const {up}_SEGMENTS")[1].split("];")[0]
        for a, b, n, c in board.raw.segments:
            assert f"({a}, {b}, {n}, {c})," in block, f"{board.name} segment missing from Rust"


# ---------------------------------------------------------------------------
# Structure, on every map
# ---------------------------------------------------------------------------


@ALL_BOARDS
def test_indices_are_canonically_ordered(board: Board) -> None:
    """Sorted, deduplicated, and endpoint-normalized, so a re-transcription is stable."""
    assert list(board.cities) == sorted(set(board.cities))
    assert list(board.color_names) == sorted(set(board.color_names))
    assert list(board.raw.segments) == sorted(board.raw.segments)
    assert list(board.raw.tickets) == sorted(board.raw.tickets)
    for s in range(board.n_segments):
        assert board.seg_a[s] < board.seg_b[s]


@ALL_BOARDS
def test_siblings_are_mutual_and_agree(board: Board) -> None:
    """Getting double routes wrong is the most common TTR implementation bug."""
    for s, sib in enumerate(board.sibling):
        if sib == NO_SIBLING:
            continue
        assert board.sibling[sib] == s, "sibling relation is not mutual"
        assert sib != s
        assert (board.seg_a[s], board.seg_b[s]) == (board.seg_a[sib], board.seg_b[sib])
        assert board.seg_len[s] == board.seg_len[sib]
        assert board.pair_of_segment[s] == board.pair_of_segment[sib]

    doubles = sum(1 for x in board.sibling if x != NO_SIBLING) // 2
    assert board.n_segments - doubles == board.n_pairs


@ALL_BOARDS
def test_buckets_partition_the_segments(board: Board) -> None:
    seen: set[int] = set()
    for i, (length, color, segs) in enumerate(board.buckets):
        assert segs, "empty bucket"
        for s in segs:
            assert (board.seg_len[s], board.seg_color[s]) == (length, color)
            assert board.seg_bucket[s] == i
            assert s not in seen
            seen.add(s)
    assert seen == set(range(board.n_segments))


@ALL_BOARDS
def test_adjacency_matches_the_segment_table(board: Board) -> None:
    for city, row in enumerate(board.adjacency):
        for nb, seg in row:
            assert {city, nb} == {board.seg_a[seg], board.seg_b[seg]}
    assert sum(len(r) for r in board.adjacency) == 2 * board.n_segments


@ALL_BOARDS
def test_distance_table_is_a_metric_and_finite(board: Board) -> None:
    n = board.n_cities
    for i in range(n):
        assert board.dist[i][i] == 0
        for j in range(n):
            assert board.dist[i][j] == board.dist[j][i], "distance is not symmetric"
            assert board.dist[i][j] < UNREACHABLE, "map is not connected"
            for k in range(n):
                assert board.dist[i][j] <= board.dist[i][k] + board.dist[k][j]


@ALL_BOARDS
def test_deck_composition(board: Board) -> None:
    raw = board.raw
    assert len(board.deck_composition) == raw.cards_per_color * raw.n_colors + raw.locomotives
    for c in range(raw.n_colors):
        assert board.deck_composition.count(c) == raw.cards_per_color
    assert board.deck_composition.count(board.locomotive) == raw.locomotives
    assert list(board.deck_composition) == sorted(board.deck_composition), "canonical order"
    # A route must be payable from a single color run, or it is unclaimable by design.
    assert raw.cards_per_color >= board.max_len


@ALL_BOARDS
def test_tickets_are_reachable_and_distinct(board: Board) -> None:
    assert len(set(zip(board.ticket_a, board.ticket_b, strict=True))) == board.n_tickets
    for t in range(board.n_tickets):
        a, b = board.ticket_a[t], board.ticket_b[t]
        assert a != b
        assert board.dist[a][b] < UNREACHABLE
        assert board.ticket_points[t] > 0


@ALL_BOARDS
def test_enough_tickets_for_a_full_table(board: Board) -> None:
    """Setup deals `initial_deal` tickets per player; running the deck dry there is a bug."""
    needed = board.raw.max_players * board.raw.initial_ticket_deal
    assert board.n_tickets >= needed, f"{board.name}: {board.n_tickets} tickets < {needed}"


@ALL_BOARDS
def test_enough_track_for_a_full_table(board: Board) -> None:
    """Every seat must be able to spend its trains without the board running out."""
    assert board.total_spaces >= board.raw.max_players * board.raw.trains_per_player * 0.9


# ---------------------------------------------------------------------------
# USA specifics -- the numbers the whole engine is sized against
# ---------------------------------------------------------------------------


def test_usa_headline_counts() -> None:
    assert (USA.n_cities, USA.n_pairs, USA.n_segments, USA.total_spaces) == (36, 78, 100, 309)
    assert (USA.n_tickets, USA.deck_size, USA.n_colors) == (30, 110, 8)
    assert USA.raw.trains_per_player == 45
    assert sum(1 for s in USA.sibling if s != NO_SIBLING) // 2 == 22


def test_usa_colour_balance() -> None:
    """44 gray plus exactly 7 segments of each of the 8 colors."""
    assert USA.seg_color.count(GRAY) == 44
    for c in range(USA.n_colors):
        assert USA.seg_color.count(c) == 7, USA.color_names[c]


def test_usa_length_histogram() -> None:
    hist = {n: USA.seg_len.count(n) for n in range(1, USA.max_len + 1)}
    assert hist == {1: 9, 2: 36, 3: 20, 4: 16, 5: 10, 6: 9}


def test_usa_degree_hubs() -> None:
    """Denver, Helena and Pittsburgh are the degree-7 hubs the feature spec assumes."""
    degree = {USA.cities[c]: len({nb for nb, _ in row}) for c, row in enumerate(USA.adjacency)}
    assert {c for c, d in degree.items() if d == 7} == {"Denver", "Helena", "Pittsburgh"}
    assert max(degree.values()) == 7


def test_action_space_is_915() -> None:
    """100 claim x 9 pay + 6 draw + 1 draw-tickets + 7 keep + 1 pass."""
    assert USA.n_segments * (USA.n_colors + 1) + 15 == 915


# ---------------------------------------------------------------------------
# TTR-mini specifics
# ---------------------------------------------------------------------------


def test_mini_keeps_what_makes_ttr_interesting() -> None:
    """The prior art's mini map dropped to 4 colors / max length 3 / no doubles."""
    assert MINI.n_colors == 6
    assert MINI.max_len >= 5
    assert sum(1 for s in MINI.sibling if s != NO_SIBLING) // 2 == 6
    assert 12 <= MINI.n_cities <= 16
    assert MINI.n_tickets == MINI.n_cities, "#tickets should track #cities"
    assert 2 * MINI.n_pairs / MINI.n_cities > 3.0, "mean degree too low"


def test_mini_ticket_points_are_shortest_path_costs() -> None:
    """The documented design rule for mini ticket values, asserted rather than trusted."""
    for t in range(MINI.n_tickets):
        expected = MINI.dist[MINI.ticket_a[t]][MINI.ticket_b[t]]
        assert MINI.ticket_points[t] == expected, MINI.ticket_name(t)


def test_mini_colour_balance() -> None:
    for c in range(MINI.n_colors):
        assert MINI.seg_color.count(c) == 3, MINI.color_names[c]
    assert MINI.seg_color.count(GRAY) == 12


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def test_get_board_rejects_unknown_maps() -> None:
    assert get_board("usa") is USA
    with pytest.raises(KeyError, match="unknown map"):
        get_board("europe")


def test_data_hashes_are_distinct_and_well_formed() -> None:
    hashes = {b.name: b.data_hash for b in BOARDS.values()}
    assert len(set(hashes.values())) == len(hashes)
    for h in hashes.values():
        assert len(h) == 32
        int(h, 16)


def test_names_render() -> None:
    assert USA.segment_name(0).count("-") >= 1
    assert USA.color_name(GRAY) == "gray"
    assert USA.color_name(USA.locomotive) == "loco"
    assert USA.color_name(0) == "black"
    assert "(" in USA.ticket_name(0)
    assert "usa" in repr(USA)
