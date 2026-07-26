"""The reference observation encoder -- the oracle the Rust encoder is checked against.

Pure Python and no numpy on purpose: this has to be importable and runnable inside the
Phase 2 differential harness, which runs with no torch installed. It is not fast and does
not need to be. The fast one is Rust; this one is *right*, and it is what makes "the Rust
encoder matches the Python oracle exactly" a checkable statement.

Layout comes entirely from `spec.ObsSpec`, so a feature is added by editing the table and
writing one accessor here.
"""

from __future__ import annotations

from array import array

from ticket_to_ride.data.board import NO_SIBLING, UNREACHABLE
from ticket_to_ride.data.rawmap import GRAY
from ticket_to_ride.engine.config import (
    CLOSED,
    FREE,
    NOT_TRIGGERED,
    PHASE_DRAW_SECOND,
)
from ticket_to_ride.engine.graph import (
    dsu_find,
    longest_trail,
    remaining_costs_from,
    steiner_cost_exact,
)
from ticket_to_ride.engine.state import EMPTY_SLOT, State
from ticket_to_ride.rl.encode.spec import (
    COST_BUCKETS,
    HAND_BUCKETS,
    MIN_TERMINALS,
    OPPONENT_SLOTS,
    TRAIN_BUCKETS,
    ObsSpec,
    obs_spec,
)


def observation_size(state: State) -> int:
    return obs_spec(state.game.board).size


def encode(state: State, player: int, out: array | None = None) -> array:
    """Encode `state` from `player`'s point of view into a float array.

    Opponents are ordered by **distance in turn order**, so slot 0 is whoever acts next.
    Seat-relative rather than permutation-invariant, which is the right prior: in Ticket to
    Ride, who moves before you is decisive.
    """
    spec = obs_spec(state.game.board)
    if out is None:
        out = array("f", bytes(4 * spec.size))
    elif len(out) != spec.size:
        raise ValueError(f"buffer holds {len(out)} floats, spec needs {spec.size}")
    else:
        for i in range(spec.size):
            out[i] = 0.0

    view = _View(state, player, spec)
    _segment_static(view, out)
    _segment_dynamic(view, out)
    _own_hand(view, out)
    _tickets(view, out)
    _steiner(view, out)
    _faceup(view, out)
    _piles(view, out)
    _opponents(view, out)
    _clock(view, out)
    return out


class _View:
    """Everything the blocks need, computed once: seat order, costs, the Steiner tree."""

    __slots__ = (
        "board",
        "connected",
        "dead",
        "dsu_root",
        "me",
        "order",
        "reach_cost",
        "spec",
        "state",
        "steiner",
        "steiner_edges",
        "steiner_exact",
        "ticket_cost",
        "trains_supply",
    )

    def __init__(self, state: State, player: int, spec: ObsSpec) -> None:
        self.state = state
        self.spec = spec
        self.board = state.game.board
        self.me = player
        n = state.game.n_players
        # Opponents by distance in turn order: the seat that acts next is slot 0.
        self.order = [(player + 1 + i) % n for i in range(n - 1)]
        self.trains_supply = self.board.raw.trains_per_player

        locked = state.game.doubles_locked
        parent = state.dsu[player]
        self.dsu_root = [dsu_find(parent, c) for c in range(self.board.n_cities)]

        held = state.tickets_of(player)
        terminals: list[int] = []
        self.ticket_cost: dict[int, int] = {}
        self.connected: dict[int, bool] = {}
        self.dead: dict[int, bool] = {}
        cache: dict[int, list[int]] = {}

        for ticket in held:
            a, b = self.board.ticket_a[ticket], self.board.ticket_b[ticket]
            if a not in cache:
                cache[a] = remaining_costs_from(self.board, state.seg_owner, player, a, locked)
            cost = cache[a][b]
            self.ticket_cost[ticket] = cost
            self.connected[ticket] = cost == 0
            self.dead[ticket] = cost >= UNREACHABLE
            if cost > 0 and cost < UNREACHABLE:
                terminals.extend((a, b))

        self.reach_cost = cache
        self.steiner, self.steiner_exact = steiner_cost_exact(
            self.board, state.seg_owner, player, terminals, locked
        )
        self.steiner_edges = _steiner_edges(self, terminals)


def _steiner_edges(view: _View, terminals: list[int]) -> set[int]:
    """Free segments on *a* cheapest route between consecutive terminals.

    An approximation of "on my Steiner tree", and deliberately so: the exact edge set of a
    Dreyfus-Wagner solution needs back-pointers through the subset DP, which is a lot of
    machinery for a boolean hint. What the network needs is "this route is on my plan",
    and a union of cheapest paths gives that.
    """
    if len(terminals) < MIN_TERMINALS:
        return set()
    state, board = view.state, view.board
    locked = state.game.doubles_locked
    edges: set[int] = set()
    for i in range(0, len(terminals) - 1, 2):
        source, target = terminals[i], terminals[i + 1]
        dist = view.reach_cost.get(source)
        if dist is None:
            dist = remaining_costs_from(board, state.seg_owner, view.me, source, locked)
            view.reach_cost[source] = dist
        if dist[target] >= UNREACHABLE:
            continue
        here = target
        guard = 0
        while here != source and guard <= board.n_cities:
            guard += 1
            for nb, segment in board.adjacency[here]:
                owner = state.seg_owner[segment]
                weight = 0 if owner == view.me else board.seg_len[segment]
                if owner not in (FREE, view.me):
                    continue
                if dist[nb] + weight == dist[here]:
                    if owner == FREE:
                        edges.add(segment)
                    here = nb
                    break
            else:  # pragma: no cover - a shortest path always has a predecessor
                break
    return edges


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _thermometer(out: array, base: int, value: int, buckets: tuple[int, ...]) -> None:
    for i, edge in enumerate(buckets):
        out[base + i] = 1.0 if value >= edge else 0.0


def _segment_static(view: _View, out: array) -> None:
    block = view.spec.block("segment_static")
    board = view.board
    for segment in range(board.n_segments):
        color = board.seg_color[segment]
        index = board.n_colors if color == GRAY else color
        out[block.at(segment, "required_color") + index] = 1.0


def _segment_dynamic(view: _View, out: array) -> None:
    block = view.spec.block("segment_dynamic")
    state, board = view.state, view.board
    seat_slot = {seat: i for i, seat in enumerate(view.order)}
    hand_base = view.me * board.n_card_types
    wilds = state.hand[hand_base + board.locomotive]
    reach = [
        (state.hand[hand_base + c] + wilds if state.hand[hand_base + c] else 0)
        for c in range(board.n_colors)
    ]
    my_trains = state.trains[view.me]

    for segment in range(board.n_segments):
        owner = state.seg_owner[segment]
        length = board.seg_len[segment]
        color = board.seg_color[segment]

        owner_slot = (
            0 if owner in (FREE, CLOSED) else (1 if owner == view.me else 2 + seat_slot[owner])
        )
        out[block.at(segment, "owner") + owner_slot] = 1.0
        out[block.at(segment, "closed")] = 1.0 if owner == CLOSED else 0.0

        twin = board.sibling[segment]
        out[block.at(segment, "twin_locked")] = (
            1.0 if twin != NO_SIBLING and state.seg_owner[twin] == view.me else 0.0
        )

        best = max(reach) if color == GRAY else reach[color]
        best = max(best, wilds)
        short = max(0, length - best)
        out[block.at(segment, "can_afford_now")] = (
            1.0 if owner == FREE and short == 0 and my_trains >= length else 0.0
        )
        _thermometer(out, block.at(segment, "cards_short"), short, HAND_BUCKETS)

        out[block.at(segment, "on_my_steiner_tree")] = 1.0 if segment in view.steiner_edges else 0.0
        touches = (
            view.dsu_root[board.seg_a[segment]] != board.seg_a[segment]
            or view.dsu_root[board.seg_b[segment]] != board.seg_b[segment]
        )
        out[block.at(segment, "extends_my_chain")] = 1.0 if owner == FREE and touches else 0.0


def _own_hand(view: _View, out: array) -> None:
    block = view.spec.block("own_hand")
    board, state = view.board, view.state
    base = view.me * board.n_card_types
    for c in range(board.n_card_types):
        count = state.hand[base + c]
        out[block.at(0, "counts") + c] = count / board.raw.cards_per_color
        for level in range(1, 7):
            name = "thermometer" if level == 1 else f"thermometer{level}"
            out[block.at(0, name) + c] = 1.0 if count >= level else 0.0


def _tickets(view: _View, out: array) -> None:
    block = view.spec.block("tickets")
    board, state = view.board, view.state
    max_points = max(board.ticket_points)
    for ticket in range(board.n_tickets):
        out[block.at(ticket, "points")] = board.ticket_points[ticket] / max_points
        if not state.tickets[view.me] >> ticket & 1:
            continue
        out[block.at(ticket, "held")] = 1.0
        out[block.at(ticket, "connected")] = 1.0 if view.connected[ticket] else 0.0
        out[block.at(ticket, "is_dead")] = 1.0 if view.dead[ticket] else 0.0
        if view.dead[ticket]:
            continue
        cost = view.ticket_cost[ticket]
        out[block.at(ticket, "remaining_cost")] = cost / view.trains_supply
        _thermometer(out, block.at(ticket, "cost_thermometer"), cost, COST_BUCKETS)
        out[block.at(ticket, "fragility")] = _fragility(view, ticket) / view.trains_supply


def _fragility(view: _View, ticket: int) -> float:
    """Worst extra cost if one enemy claim lands on my cheapest route for this ticket.

    Measured by re-pricing the route with each of its free segments taken in turn. That is
    a handful of Dijkstras per ticket, which is fine for an oracle and is exactly the kind
    of thing the Rust encoder will do in microseconds.
    """
    state, board = view.state, view.board
    locked = state.game.doubles_locked
    a, b = board.ticket_a[ticket], board.ticket_b[ticket]
    baseline = view.ticket_cost[ticket]
    if baseline == 0 or baseline >= UNREACHABLE:
        return 0.0

    on_path = [s for s in _steiner_edges_for(view, a, b) if state.seg_owner[s] == FREE]
    worst = 0
    for segment in on_path:
        state.seg_owner[segment] = CLOSED
        cost = remaining_costs_from(board, state.seg_owner, view.me, a, locked)[b]
        state.seg_owner[segment] = FREE
        if cost >= UNREACHABLE:
            return float(view.trains_supply)
        worst = max(worst, cost - baseline)
    return float(worst)


def _steiner_edges_for(view: _View, source: int, target: int) -> list[int]:
    """One cheapest route from `source` to `target`, as segment ids."""
    state, board = view.state, view.board
    dist = view.reach_cost.get(source)
    if dist is None:
        dist = remaining_costs_from(
            board, state.seg_owner, view.me, source, state.game.doubles_locked
        )
        view.reach_cost[source] = dist
    if dist[target] >= UNREACHABLE:
        return []
    path: list[int] = []
    here = target
    guard = 0
    while here != source and guard <= board.n_cities:
        guard += 1
        for nb, segment in board.adjacency[here]:
            owner = state.seg_owner[segment]
            if owner not in (FREE, view.me):
                continue
            weight = 0 if owner == view.me else board.seg_len[segment]
            if dist[nb] + weight == dist[here]:
                path.append(segment)
                here = nb
                break
        else:  # pragma: no cover - a shortest path always has a predecessor
            break
    return path


def _steiner(view: _View, out: array) -> None:
    block = view.spec.block("steiner")
    cost = min(view.steiner, view.trains_supply * 2)
    out[block.at(0, "remaining_cost")] = cost / view.trains_supply
    slack = view.state.trains[view.me] - cost
    out[block.at(0, "cost_minus_trains")] = slack / view.trains_supply
    out[block.at(0, "exact")] = 1.0 if view.steiner_exact else 0.0


def _faceup(view: _View, out: array) -> None:
    block = view.spec.block("faceup")
    empty = view.board.n_card_types
    for slot, card in enumerate(view.state.faceup):
        index = empty if card == EMPTY_SLOT else card
        out[block.at(slot, "card") + index] = 1.0


def _piles(view: _View, out: array) -> None:
    block = view.spec.block("piles")
    state, board = view.state, view.board
    size = board.deck_size
    out[block.at(0, "deck_size")] = (len(state.deck) - state.deck_pos) / size
    out[block.at(0, "discard_size")] = state.discard_total / size
    out[block.at(0, "ticket_deck_size")] = state.tdeck_len / board.n_tickets
    unseen = state.unseen_counts(view.me)
    for c in range(board.n_card_types):
        out[block.at(0, "discard_composition") + c] = state.discard[c] / board.cards_per_type(c)
        out[block.at(0, "unseen") + c] = unseen[c] / board.cards_per_type(c)


def _opponents(view: _View, out: array) -> None:
    block = view.spec.block("opponents")
    state, board = view.state, view.board
    k = board.n_card_types
    for slot in range(OPPONENT_SLOTS):
        if slot >= len(view.order):
            continue
        seat = view.order[slot]
        out[block.at(slot, "present")] = 1.0
        trains = state.trains[seat]
        out[block.at(slot, "trains")] = trains / view.trains_supply
        _thermometer(out, block.at(slot, "trains_thermometer"), trains, TRAIN_BUCKETS)
        out[block.at(slot, "score")] = state.score[seat] / 100.0
        out[block.at(slot, "hand_size")] = state.hand_size(seat) / board.deck_size
        out[block.at(slot, "ticket_count")] = int(state.tickets[seat]).bit_count() / board.n_tickets
        out[block.at(slot, "blind_draws")] = state.unknown[seat] / board.deck_size
        claimed = sum(1 for owner in state.seg_owner if owner == seat)
        out[block.at(slot, "segments_claimed")] = claimed / board.n_segments
        out[block.at(slot, "longest_chain")] = (
            longest_trail(board, state.seg_owner, seat) / view.trains_supply
        )
        unseen = state.unseen_counts(view.me)
        for c in range(k):
            certain = state.certain[seat * k + c]
            out[block.at(slot, "certain") + c] = certain / board.cards_per_type(c)
            # Everything I cannot see could in principle be theirs, on top of what I know.
            possible = certain + min(unseen[c], state.unknown[seat])
            out[block.at(slot, "max_possible") + c] = possible / board.cards_per_type(c)


def _clock(view: _View, out: array) -> None:
    block = view.spec.block("clock")
    state, board = view.state, view.board
    n = state.game.n_players
    out[block.at(0, "phase") + state.phase] = 1.0
    out[block.at(0, "seats") + (n - 2)] = 1.0
    out[block.at(0, "draws_left")] = 1.0 if state.phase == PHASE_DRAW_SECOND else 0.0

    triggered = state.final_left != NOT_TRIGGERED
    out[block.at(0, "final_triggered")] = 1.0 if triggered else 0.0
    out[block.at(0, "final_countdown")] = (state.final_left / n) if triggered else 0.0
    out[block.at(0, "turn")] = min(state.turn, 200) / 200.0

    trains = state.trains[view.me]
    out[block.at(0, "my_trains")] = trains / view.trains_supply
    _thermometer(out, block.at(0, "my_trains_thermometer"), trains, TRAIN_BUCKETS)
    out[block.at(0, "my_score")] = state.score[view.me] / 100.0
    out[block.at(0, "my_ticket_count")] = int(state.tickets[view.me]).bit_count() / board.n_tickets

    leader = max(state.score)
    behind = sum(1 for s in state.score if s > state.score[view.me])
    out[block.at(0, "score_rank")] = 1.0 - behind / max(1, n - 1)
    out[block.at(0, "gap_to_leader")] = (leader - state.score[view.me]) / 100.0
