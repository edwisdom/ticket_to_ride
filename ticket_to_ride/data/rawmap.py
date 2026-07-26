"""The frozen shape of generated board data.

`board_gen.py` is machine-written and holds nothing but literals; this module holds the
type those literals are poured into. Keeping the two apart means the generated file has a
stable, reviewable diff and no logic, and that `ty`/`ruff` can skip it without skipping
anything that matters.

Index conventions, frozen (see docs/CONTRACT.md):

* **Cities** are indexed by their position in `RawMap.cities`, which is sorted ascending by
  Python string order. A re-transcription of the map in a different row order therefore
  produces identical indices.
* **Card types** are `0..n_colors-1` for the colors in `RawMap.color_names` (also sorted),
  and `n_colors` for the locomotive. `LOCOMOTIVE` is *not* a constant 8 -- TTR-mini has six
  colors, so its locomotive is card type 6.
* **Segment colors** use the same indices, plus `GRAY` (255) for "any single color".
"""

from __future__ import annotations

from typing import NamedTuple

# Sentinel for a route that accepts a set of any one color. 255 rather than -1 so the
# whole board table stays u8 on both sides of the FFI boundary.
GRAY = 255


class RawMap(NamedTuple):
    """One board's data, exactly as generated. Derived tables live in `board.py`."""

    name: str
    cities: tuple[str, ...]
    #: (city_a, city_b, length, color) with city_a < city_b, sorted; color may be GRAY.
    segments: tuple[tuple[int, int, int, int], ...]
    #: (city_a, city_b, points) with city_a < city_b, sorted.
    tickets: tuple[tuple[int, int, int], ...]
    color_names: tuple[str, ...]
    trains_per_player: int
    cards_per_color: int
    locomotives: int
    #: route_points[n] is the score for claiming a route of length n; index 0 is unused.
    route_points: tuple[int, ...]
    longest_bonus: int
    initial_ticket_deal: int
    initial_ticket_keep_min: int
    draw_ticket_deal: int
    draw_ticket_keep_min: int
    min_players: int
    max_players: int
    #: blake2b-128 over the canonical encoding in tools/gen_board.py, as 32 hex chars.
    data_hash: str

    @property
    def n_colors(self) -> int:
        return len(self.color_names)

    @property
    def locomotive(self) -> int:
        """Card-type index of the locomotive: one past the last color."""
        return len(self.color_names)

    @property
    def n_card_types(self) -> int:
        return len(self.color_names) + 1
