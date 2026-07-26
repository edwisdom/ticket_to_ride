"""The flat, maskable action space. 915 on the USA map, 225 on TTR-mini.

| Range | Size | Meaning |
| --- | --- | --- |
| `0 .. S*K-1` | `S*K` | `CLAIM`: `segment*K + pay`; pay `0..n_colors-1` = that color, `n_colors` = pay entirely with locomotives |
| next 6 | 6 | `DRAW`: face-up slot 0-4, or 5 = blind |
| next 1 | 1 | `DRAW_TICKETS` |
| next 7 | 7 | `KEEP`: `bitmask - 1`, mask in 1..7 |
| next 1 | 1 | `PASS` |

`S` is the segment count and `K` the card-type count (`n_colors + 1`). USA: 100*9 + 15 =
915.

**The keep mask starts at 1, so there are 7 keep actions, not 8.** Keeping *nothing* is
never legal. An off-by-one there shifts every action index above it, silently.

**Locomotive payment is canonical**, which is what collapses the naive 100x9x7 = 6300
space to 900. For a route of length `L` in color `c`, let `k = min(hand[c], L)`; paying `k`
colored plus `L-k` locomotives weakly dominates paying fewer colored cards, because the
two resulting hands differ only by trading colored cards for locomotives, and a locomotive
substitutes for `c` in every legal claim but not conversely. The usual objection --
hoarding locomotives is strategically costly -- does not apply to base TTR, which has no
hand limit.

**The `hand[c] >= 1` guard is load-bearing.** Without it a hand of pure locomotives makes
all 8 gray-color pay slots legal *and identical*, so eight distinct action ids denote the
same payment and the policy distribution is poisoned. With it, action ids map bijectively
to payments.
"""

from __future__ import annotations

from typing import Final

from ticket_to_ride.data.board import Board
from ticket_to_ride.data.rawmap import GRAY

#: Bumped when the layout above changes. Baked into every checkpoint.
ACTION_SPACE_VERSION: Final = 1

#: 5 face-up slots plus the blind draw.
N_DRAW_ACTIONS: Final = 6
BLIND_SLOT: Final = 5

#: Keep masks 1..7 over a 3-ticket offer. Never 8: the empty keep is illegal.
N_KEEP_ACTIONS: Final = 7

#: 6 draw + 1 draw-tickets + 7 keep + 1 pass.
N_NON_CLAIM_ACTIONS: Final = N_DRAW_ACTIONS + 1 + N_KEEP_ACTIONS + 1


class ActionSpace:
    """Action id arithmetic for one board. Immutable; built once and shared."""

    __slots__ = ("board", "claim_end", "draw_base", "draw_tickets", "k", "n", "pass_action")

    def __init__(self, board: Board) -> None:
        self.board = board
        self.k = board.n_card_types
        self.claim_end = board.n_segments * self.k
        self.draw_base = self.claim_end
        self.draw_tickets = self.claim_end + N_DRAW_ACTIONS
        self.n = self.claim_end + N_NON_CLAIM_ACTIONS
        self.pass_action = self.n - 1

    # -- encode ------------------------------------------------------------

    def claim(self, segment: int, pay: int) -> int:
        return segment * self.k + pay

    def draw(self, slot: int) -> int:
        return self.draw_base + slot

    def keep(self, mask: int) -> int:
        if not 1 <= mask <= N_KEEP_ACTIONS:
            raise ValueError(f"keep mask must be in 1..7, got {mask}")
        return self.draw_tickets + mask

    # -- decode ------------------------------------------------------------

    @property
    def keep_base(self) -> int:
        """`keep(mask) == keep_base + mask`, i.e. the id of the (illegal) empty keep."""
        return self.draw_tickets

    def is_claim(self, action: int) -> bool:
        return action < self.claim_end

    def decode_claim(self, action: int) -> tuple[int, int]:
        return divmod(action, self.k)

    def decode_draw(self, action: int) -> int:
        return action - self.draw_base

    def decode_keep(self, action: int) -> int:
        return action - self.draw_tickets

    # -- naming ------------------------------------------------------------

    def to_string(self, action: int) -> str:
        """Human-readable, for the terminal client, replays and test failure messages."""
        if not 0 <= action < self.n:
            raise ValueError(f"action {action} outside 0..{self.n - 1}")
        board = self.board
        if action < self.claim_end:
            segment, pay = self.decode_claim(action)
            required = board.seg_color[segment]
            required_name = "gray" if required == GRAY else board.color_names[required]
            paid = "loco" if pay == board.locomotive else board.color_names[pay]
            return (
                f"CLAIM {board.cities[board.seg_a[segment]]}-"
                f"{board.cities[board.seg_b[segment]]}"
                f"[{board.seg_len[segment]}{required_name}] pay {paid}"
            )
        if action < self.draw_tickets:
            slot = self.decode_draw(action)
            return "DRAW blind" if slot == BLIND_SLOT else f"DRAW faceup[{slot}]"
        if action == self.draw_tickets:
            return "DRAW_TICKETS"
        if action < self.pass_action:
            mask = self.decode_keep(action)
            kept = [i for i in range(3) if mask & (1 << i)]
            return f"KEEP offer{kept}"
        return "PASS"


_CACHE: dict[str, ActionSpace] = {}


def action_space(board: Board) -> ActionSpace:
    """The (cached) action space for a board. One instance per map, shared by every state."""
    space = _CACHE.get(board.name)
    if space is None:
        space = _CACHE[board.name] = ActionSpace(board)
    return space
