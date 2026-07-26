"""The engine: `Game` (immutable setup) and `State` (a flat, cloneable position).

`State` is deliberately plain arrays and small ints, with no undo log. At this size a clone
beats maintaining a journal for a claim -- which touches `seg_owner`, `hand`, `discard`,
`trains`, `score` and the union-find -- and it is what lets the DSU use path compression
freely. Search at 800 simulations per move stores well under a megabyte of states.

Rules edge cases live next to the code that implements them. The checklist they come from
is PLAN.md §5.2; the frozen ordering is docs/CONTRACT.md §2.
"""

from __future__ import annotations

from typing import Final

from ticket_to_ride.data.board import NO_SIBLING
from ticket_to_ride.data.rawmap import GRAY
from ticket_to_ride.engine.actions import BLIND_SLOT, ActionSpace, action_space
from ticket_to_ride.engine.config import (
    CLOSED,
    END_TRIGGER_TRAINS,
    FACEUP_SLOTS,
    FLUSH_LOCOS,
    FREE,
    NOT_TRIGGERED,
    PHASE_DRAW_SECOND,
    PHASE_INITIAL_TICKETS,
    PHASE_MAIN,
    PHASE_NAMES,
    PHASE_TERMINAL,
    PHASE_TICKET_KEEP,
    RuleConfig,
)
from ticket_to_ride.engine.contract import CONTRACT_VERSION
from ticket_to_ride.engine.graph import dsu_connected, dsu_union
from ticket_to_ride.engine.hashing import hash64
from ticket_to_ride.engine.rng import Pcg32, stream

#: Vacated slot in the ticket ring, and "no ticket" in a pending offer. Keeping vacated
#: slots at a fixed value is what makes the ring's byte image canonical, and therefore
#: hashable: stale leftovers would let two identical positions hash differently.
NO_TICKET: Final = 255

#: Empty face-up slot. Serialized as a signed byte, so 255 on the wire.
EMPTY_SLOT: Final = -1

#: `tickets` is a per-seat bitmask in a u32.
MAX_TICKETS: Final = 32


class IllegalAction(Exception):  # noqa: N818 - the domain term, not "IllegalActionError"
    """An action that is not legal in the current state.

    Raised rather than silently tolerated: a policy that leaks an illegal action past its
    mask is the classic silent RL bug, and it poisons training with no visible symptom.
    Every `step()` path re-checks the specific preconditions of the action it is given,
    which costs a handful of comparisons -- far cheaper than a full `legal_actions()` scan
    and enough to make the leak impossible.
    """


class Game:
    """Immutable per-configuration setup: the board, the action space, the rule constants."""

    __slots__ = ("board", "cfg", "doubles_locked", "n_players", "space")

    def __init__(self, cfg: RuleConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else RuleConfig()
        self.board = self.cfg.board
        self.space = action_space(self.board)
        self.n_players = self.cfg.n_players
        self.doubles_locked = self.cfg.doubles_locked_for_everyone
        if self.board.n_tickets > MAX_TICKETS:
            raise ValueError(
                f"{self.board.name} has {self.board.n_tickets} tickets; the per-seat "
                f"bitmask holds {MAX_TICKETS}"
            )

    @property
    def num_distinct_actions(self) -> int:
        return self.space.n

    def new_initial_state(self, seed: int) -> State:
        return State(self, seed)

    def action_to_string(self, action: int) -> str:
        return self.space.to_string(action)

    def __repr__(self) -> str:
        return f"<Game {self.board.name} {self.n_players}P actions={self.space.n}>"


class State:
    """One position. Cloneable, hashable, and steppable."""

    __slots__ = (
        "board_version",
        "certain",
        "cur",
        "deck",
        "deck_pos",
        "discard",
        "discard_total",
        "draws_left",
        "dsu",
        "faceup",
        "final_left",
        "game",
        "hand",
        "history",
        "offer",
        "offer_len",
        "pass_streak",
        "phase",
        "rng",
        "score",
        "seed",
        "seg_owner",
        "tdeck",
        "tdeck_head",
        "tdeck_len",
        "tickets",
        "trains",
        "turn",
        "unknown",
    )

    def __init__(self, game: Game, seed: int) -> None:
        self.game = game
        self.seed = seed
        board, n = game.board, game.n_players
        k = board.n_card_types

        self.seg_owner = bytearray([FREE]) * board.n_segments
        self.hand = bytearray(n * k)
        self.trains = bytearray([board.raw.trains_per_player]) * n
        self.score = [0] * n
        self.tickets = [0] * n
        self.discard = bytearray(k)
        # Kept alongside `discard` purely for speed: `any(discard)` and `sum(discard)` sit
        # in the innermost legality loop and a 9-element scan there is measurable.
        self.discard_total = 0
        self.faceup = [EMPTY_SLOT] * FACEUP_SLOTS
        self.certain = bytearray(n * k)
        self.unknown = bytearray(n)
        self.dsu = [bytearray(range(board.n_cities)) for _ in range(n)]
        self.offer = [NO_TICKET, NO_TICKET, NO_TICKET]
        self.offer_len = 0
        self.draws_left = 0
        self.final_left = NOT_TRIGGERED
        self.pass_streak = 0
        self.turn = 0
        self.board_version = 0
        self.history: list[int] = []

        self._setup(seed)

    # -----------------------------------------------------------------
    # Setup -- docs/CONTRACT.md §2.2
    # -----------------------------------------------------------------

    def _setup(self, seed: int) -> None:
        board, n = self.game.board, self.game.n_players

        deck = list(board.deck_composition)
        stream(seed, "env", "deck").shuffle(deck)
        self.deck = bytes(deck)
        self.deck_pos = 0

        tickets = list(range(board.n_tickets))
        stream(seed, "env", "tickets").shuffle(tickets)
        self.tdeck = tickets
        self.tdeck_head = 0
        self.tdeck_len = board.n_tickets

        # Advanced by reshuffles and nothing else, so a game that never exhausts the deck
        # never touches it.
        self.rng = stream(seed, "env", "reshuffle")

        # Hands first, in per-seat blocks, *then* the display. Flipping first would deal
        # every seat different cards.
        k = board.n_card_types
        for p in range(n):
            for _ in range(self.game.cfg.initial_hand):
                card = self._draw_card()
                if card is None:  # pragma: no cover - the deck is validated large enough
                    raise ValueError("deck exhausted during the initial deal")
                self.hand[p * k + card] += 1
                self.unknown[p] += 1

        self._refill()
        self._flush_check()

        self.phase = PHASE_INITIAL_TICKETS
        self.cur = 0
        self._begin_initial_offer()

    def _begin_initial_offer(self) -> None:
        """Deal and, where forced, auto-resolve every seat's opening ticket choice.

        Physically the offers are dealt and chosen simultaneously. Dealing in seat order
        gives each seat the same cards it would have got, and because keeps are secret and
        no seat observes another's choice, sequentializing the *choice* is
        information-equivalent. This is the one documented rules deviation.
        """
        raw = self.game.board.raw
        while self.cur < self.game.n_players:
            if self._deal_offer(raw.initial_ticket_deal, raw.initial_ticket_keep_min):
                return  # this seat has a real decision to make
            self.cur += 1
        self.cur = 0
        self.phase = PHASE_MAIN
        self.turn = 0

    # -----------------------------------------------------------------
    # Card plumbing -- docs/CONTRACT.md §2.3-2.6
    # -----------------------------------------------------------------

    def _draw_card(self) -> int | None:
        """The top card, reshuffling **lazily** if the deck is spent.

        Reshuffling eagerly the moment the cursor reaches the end would make "deck empty,
        discards available" and "deck and discards both empty" indistinguishable -- and
        those two states have different legal actions.
        """
        if self.deck_pos >= len(self.deck) and not self._reshuffle():
            return None
        card = self.deck[self.deck_pos]
        self.deck_pos += 1
        return card

    def _reshuffle(self) -> bool:
        """Rebuild the deck from the discard. Returns False if there is nothing to reshuffle."""
        if not self.discard_total:
            return False
        # Canonical order before shuffling: discard *order* is never observable, so the
        # multiset is the only real information and the permutation must come from the
        # stream alone.
        discard = self.discard
        cards: list[int] = []
        for card_type, count in enumerate(discard):
            cards.extend([card_type] * count)
            discard[card_type] = 0
        self.discard_total = 0
        self.rng.shuffle(cards)
        self.deck = bytes(cards)
        self.deck_pos = 0
        return True

    def _refill(self) -> None:
        """Top the display back up to five, ascending slot order."""
        faceup = self.faceup
        if EMPTY_SLOT not in faceup:
            return
        for i in range(FACEUP_SLOTS):
            if faceup[i] == EMPTY_SLOT:
                card = self._draw_card()
                if card is None:
                    return  # every pool is empty; the display legitimately holds < 5
                faceup[i] = card

    def _cards_available(self) -> bool:
        return self.deck_pos < len(self.deck) or self.discard_total > 0

    def _flush_check(self) -> None:
        """Three or more face-up locomotives: discard all five and deal five new ones.

        The replacement five can themselves contain three locomotives, so this cascades.
        Two locks on the most common hang bug in TTR implementations:

        1. The **guard** -- with 14 locomotives in a 110-card deck an all-locomotive pool
           would loop forever, so a flush only happens when the available pool still holds
           three non-locomotives.
        2. The **cascade cap**, a deterministic bail-out that simply stops flushing.

        After a bail-out the display may legitimately show 3+ locomotives; `validate()`
        checks that this only ever happens when the guard was the reason.
        """
        loco = self.game.board.locomotive
        faceup = self.faceup
        for _ in range(self.game.cfg.flush_cascade_cap):
            if faceup.count(loco) < FLUSH_LOCOS:
                return
            if self._nonloco_available() < FLUSH_LOCOS:
                return
            for i in range(FACEUP_SLOTS):
                card = faceup[i]
                if card != EMPTY_SLOT:
                    self.discard[card] += 1
                    self.discard_total += 1
                    faceup[i] = EMPTY_SLOT
            self._refill()

    def _nonloco_available(self) -> int:
        """Non-locomotives left in the deck plus the discard, measured before a flush."""
        loco = self.game.board.locomotive
        deck, pos = self.deck, self.deck_pos
        in_deck = (len(deck) - pos) - deck.count(loco, pos)
        in_discard = self.discard_total - self.discard[loco]
        return in_deck + in_discard

    # -----------------------------------------------------------------
    # Ticket plumbing
    # -----------------------------------------------------------------

    def _take_ticket(self) -> int:
        ticket = self.tdeck[self.tdeck_head]
        self.tdeck[self.tdeck_head] = NO_TICKET
        self.tdeck_head = (self.tdeck_head + 1) % len(self.tdeck)
        self.tdeck_len -= 1
        return ticket

    def _return_ticket(self, ticket: int) -> None:
        """To the *bottom*, which is observable and therefore part of the state."""
        self.tdeck[(self.tdeck_head + self.tdeck_len) % len(self.tdeck)] = ticket
        self.tdeck_len += 1

    def _deal_offer(self, deal: int, keep_min: int) -> bool:
        """Deal an offer. Returns True if the seat actually has a decision to make.

        With exactly one legal keep mask -- one ticket left in the deck, or a forced
        `keep >= n` -- the choice is resolved here rather than burning a network evaluation
        on a decision with a single option.
        """
        take = min(deal, self.tdeck_len)
        for i in range(take):
            self.offer[i] = self._take_ticket()
        for i in range(take, 3):
            self.offer[i] = NO_TICKET
        self.offer_len = take

        masks = self._legal_keep_masks(keep_min)
        if len(masks) == 1:
            self._apply_keep(masks[0])
            return False
        return True

    def _legal_keep_masks(self, keep_min: int | None = None) -> list[int]:
        """Non-empty subsets of the offer that keep at least the minimum.

        The initial 3-of-3-keep-2 offer has exactly four: 011, 101, 110, 111.
        """
        if keep_min is None:
            raw = self.game.board.raw
            keep_min = (
                raw.initial_ticket_keep_min
                if self.phase == PHASE_INITIAL_TICKETS
                else raw.draw_ticket_keep_min
            )
        n = self.offer_len
        need = min(keep_min, n)
        return [m for m in range(1, 1 << n) if m.bit_count() >= need]

    def _apply_keep(self, mask: int) -> None:
        cur = self.cur
        for i in range(self.offer_len):
            ticket = self.offer[i]
            if mask & (1 << i):
                self.tickets[cur] |= 1 << ticket
            else:
                self._return_ticket(ticket)
        self.offer_len = 0
        self.offer[0] = self.offer[1] = self.offer[2] = NO_TICKET

    # -----------------------------------------------------------------
    # Turn structure
    # -----------------------------------------------------------------

    def _end_turn(self, *, passed: bool) -> None:
        n = self.game.n_players
        self.turn += 1
        self.draws_left = 0
        self.pass_streak = self.pass_streak + 1 if passed else 0

        # Every seat passing in a row means the position is frozen. Without this the engine
        # simply hangs.
        if self.pass_streak >= n:
            self.phase = PHASE_TERMINAL
            return

        if self.final_left == NOT_TRIGGERED:
            # Fires at *end of turn*, after every turn type: a seat sitting on two trains
            # that spends its turn drawing still triggers the ending.
            if self.trains[self.cur] <= END_TRIGGER_TRAINS:
                self.final_left = n
        else:
            self.final_left -= 1
            if self.final_left == 0:
                self.phase = PHASE_TERMINAL
                return

        if self.turn >= self.game.cfg.turn_cap:  # pragma: no cover - belt and braces
            self.phase = PHASE_TERMINAL
            return

        # A claim moves cards into the discard, so a display that ran short earlier can be
        # topped up now. Without this the display would stay short forever once every pool
        # emptied, since only *taking* a face-up card would otherwise trigger a refill.
        self._refill()
        self._flush_check()

        self.cur = (self.cur + 1) % n
        self.phase = PHASE_MAIN

    # -----------------------------------------------------------------
    # Stepping
    # -----------------------------------------------------------------

    def step(self, action: int) -> None:
        space = self.game.space
        if not 0 <= action < space.n:
            raise IllegalAction(f"action {action} outside 0..{space.n - 1}")

        phase = self.phase
        if phase == PHASE_MAIN:
            self._step_main(action, space)
        elif phase == PHASE_DRAW_SECOND:
            self._step_draw_second(action, space)
        elif phase in (PHASE_INITIAL_TICKETS, PHASE_TICKET_KEEP):
            self._step_keep(action, space)
        else:
            raise IllegalAction("the game is over")

        # Recorded only once the action has actually been applied. Every `_step_*` path
        # validates before it mutates, so a rejected action leaves the state untouched --
        # and the history has to stay untouched with it, or a replay of a game that
        # survived an illegal action would diverge.
        if self.game.cfg.track_history:
            self.history.append(action)

    def _step_main(self, action: int, space: ActionSpace) -> None:
        if action < space.claim_end:
            segment, pay = divmod(action, space.k)
            self._claim(segment, pay)
        elif action < space.draw_tickets:
            self._draw_first(action - space.draw_base)
        elif action == space.draw_tickets:
            self._draw_tickets()
        elif action == space.pass_action:
            self._pass()
        else:
            raise IllegalAction("KEEP is only legal while a ticket offer is open")

    def _step_draw_second(self, action: int, space: ActionSpace) -> None:
        if not space.draw_base <= action < space.draw_tickets:
            raise IllegalAction("only a card draw is legal as the second draw")
        slot = action - space.draw_base
        board = self.game.board
        if slot == BLIND_SLOT:
            card = self._draw_card()
            if card is None:
                raise IllegalAction("no cards left to draw")
            self.hand[self.cur * board.n_card_types + card] += 1
            self.unknown[self.cur] += 1
        else:
            card = self.faceup[slot]
            if card == EMPTY_SLOT:
                raise IllegalAction(f"face-up slot {slot} is empty")
            # A face-up locomotive is worth two cards, so it may never be the second.
            if card == board.locomotive:
                raise IllegalAction("a face-up locomotive cannot be taken as the second card")
            self._take_faceup(slot, card)
        self._end_turn(passed=False)

    def _step_keep(self, action: int, space: ActionSpace) -> None:
        if not space.keep_base < action < space.pass_action:
            raise IllegalAction("only a ticket keep is legal here")
        mask = action - space.keep_base
        if mask not in self._legal_keep_masks():
            raise IllegalAction(f"keep mask {mask:03b} is not legal for this offer")

        initial = self.phase == PHASE_INITIAL_TICKETS
        self._apply_keep(mask)
        if initial:
            self.cur += 1
            self._begin_initial_offer()
        else:
            self._end_turn(passed=False)

    # -- the three real turn actions ---------------------------------------

    def _claim(self, segment: int, pay: int) -> None:
        board = self.game.board
        cur = self.cur
        owner = self.seg_owner[segment]
        if owner != FREE:
            raise IllegalAction(f"segment {segment} is {'closed' if owner == CLOSED else 'taken'}")

        length = board.seg_len[segment]
        if self.trains[cur] < length:
            raise IllegalAction(f"{self.trains[cur]} trains left, route needs {length}")

        sibling = board.sibling[segment]
        # 4-5P: the sibling stays open to *others*, never to me. In 2-3P the sibling was
        # marked CLOSED when the first track was claimed, so `owner != FREE` covered it.
        if (
            not self.game.doubles_locked
            and sibling != NO_SIBLING
            and self.seg_owner[sibling] == cur
        ):
            raise IllegalAction("one player may not own both tracks of a double route")

        k = board.n_card_types
        base = cur * k
        loco = board.locomotive
        if pay == loco:
            if self.hand[base + loco] < length:
                raise IllegalAction("not enough locomotives")
            colored, wilds = 0, length
        else:
            required = board.seg_color[segment]
            if required not in (GRAY, pay):
                raise IllegalAction(
                    f"route needs {board.color_name(required)}, not {board.color_name(pay)}"
                )
            have = self.hand[base + pay]
            # The `have >= 1` guard is what keeps action ids bijective with payments: a
            # hand of pure locomotives would otherwise make all eight gray pay slots legal
            # *and identical*.
            if have < 1 or have + self.hand[base + loco] < length:
                raise IllegalAction(f"cannot pay {length} with {board.color_name(pay)}")
            colored = have if have < length else length
            wilds = length - colored

        self._spend(pay, colored)
        self._spend(loco, wilds)

        self.seg_owner[segment] = cur
        if self.game.doubles_locked and sibling != NO_SIBLING:
            self.seg_owner[sibling] = CLOSED
        self.trains[cur] -= length
        self.score[cur] += board.route_points[length]
        dsu_union(self.dsu[cur], board.seg_a[segment], board.seg_b[segment])
        self.board_version += 1
        self._end_turn(passed=False)

    def _spend(self, card_type: int, count: int) -> None:
        """Pay `count` cards and keep the public knowledge of my hand consistent.

        Cards spent on a claim are publicly discarded, so they cancel what opponents know
        for certain before they cancel what they only know the count of.
        """
        if count == 0:
            return
        k = self.game.board.n_card_types
        index = self.cur * k + card_type
        self.hand[index] -= count
        self.discard[card_type] += count
        self.discard_total += count
        known = self.certain[index]
        if known >= count:
            self.certain[index] = known - count
        else:
            self.certain[index] = 0
            self.unknown[self.cur] -= count - known

    def _draw_first(self, slot: int) -> None:
        board = self.game.board
        if slot == BLIND_SLOT:
            card = self._draw_card()
            if card is None:
                raise IllegalAction("no cards left to draw")
            self.hand[self.cur * board.n_card_types + card] += 1
            self.unknown[self.cur] += 1
        else:
            card = self.faceup[slot]
            if card == EMPTY_SLOT:
                raise IllegalAction(f"face-up slot {slot} is empty")
            self._take_faceup(slot, card)
            # A face-up locomotive counts as the whole turn. A locomotive drawn *blind*
            # does not -- it is an ordinary card and the turn continues.
            if card == board.locomotive:
                self._end_turn(passed=False)
                return

        if self._can_draw_second():
            self.phase = PHASE_DRAW_SECOND
            self.draws_left = 1
        else:
            # Never construct a node with zero legal actions.
            self._end_turn(passed=False)

    def _take_faceup(self, slot: int, card: int) -> None:
        board = self.game.board
        index = self.cur * board.n_card_types + card
        self.hand[index] += 1
        self.certain[index] += 1
        self.faceup[slot] = EMPTY_SLOT
        # Refill *before* the second draw, and the replacement may itself be a locomotive.
        self._refill()
        self._flush_check()

    def _can_draw_second(self) -> bool:
        loco = self.game.board.locomotive
        if any(c not in (EMPTY_SLOT, loco) for c in self.faceup):
            return True
        return self._cards_available()

    def _draw_tickets(self) -> None:
        if self.tdeck_len == 0:
            raise IllegalAction("the ticket deck is empty")
        raw = self.game.board.raw
        if self._deal_offer(raw.draw_ticket_deal, raw.draw_ticket_keep_min):
            self.phase = PHASE_TICKET_KEEP
        else:
            self._end_turn(passed=False)

    def _pass(self) -> None:
        probe: list[int] = []
        self._legal_main_into(probe)
        if probe:
            raise IllegalAction("PASS is only legal with no other option")
        self._end_turn(passed=True)

    # -----------------------------------------------------------------
    # Legality
    # -----------------------------------------------------------------

    def legal_actions(self) -> list[int]:
        """Sorted ascending, so two engines' lists compare directly."""
        out: list[int] = []
        self._legal_into(out)
        out.sort()
        return out

    def sample_legal(self, rng: Pcg32) -> int:
        """A uniformly random legal action, skipping the sort `legal_actions()` pays for.

        Rollouts do not care about order, and at ~150 steps a game the sort is a real
        fraction of a random playout.
        """
        out: list[int] = []
        self._legal_into(out)
        if not out:
            raise IllegalAction("no legal actions (the state is terminal)")
        return out[rng.below(len(out))]

    def legal_action_mask(self, out: bytearray | None = None) -> bytearray:
        """A 0/1 mask over the whole action space, reusing `out` when given."""
        n = self.game.space.n
        if out is None:
            out = bytearray(n)
        else:
            for i in range(n):
                out[i] = 0
        for action in self.legal_actions():
            out[action] = 1
        return out

    def _legal_into(self, out: list[int]) -> None:
        phase = self.phase
        if phase == PHASE_MAIN:
            self._legal_main_into(out)
            if not out:
                out.append(self.game.space.pass_action)
        elif phase == PHASE_DRAW_SECOND:
            self._legal_draws_into(out, first=False)
        elif phase in (PHASE_INITIAL_TICKETS, PHASE_TICKET_KEEP):
            base = self.game.space.keep_base
            out.extend(base + m for m in self._legal_keep_masks())

    def _legal_main_into(self, out: list[int]) -> None:
        """Everything legal in MAIN *except* PASS, which is only legal if this is empty."""
        self._legal_claims_into(out)
        self._legal_draws_into(out, first=True)
        if self.tdeck_len:
            out.append(self.game.space.draw_tickets)

    def _legal_draws_into(self, out: list[int], *, first: bool) -> None:
        base = self.game.space.draw_base
        # A face-up locomotive is worth two cards, so it is only takeable as the first.
        blocked = EMPTY_SLOT if first else self.game.board.locomotive
        out.extend(
            base + i for i, card in enumerate(self.faceup) if card not in (EMPTY_SLOT, blocked)
        )
        if self._cards_available():
            out.append(base + BLIND_SLOT)

    def _pay_slots_by_length(
        self, wilds: int, reach: list[int]
    ) -> tuple[list[list[int]], list[bool]]:
        """Which pay slots each route length admits, resolved once per legality scan.

        Returns `(slots_for_gray_routes, can_pay_all_locomotives)`, both indexed by route
        length with index 0 unused. A colored route's own slot is a single `reach` compare,
        so it does not need a table. (A sorted-prefix version of this is asymptotically
        better and measured identically at eight colors, so the plain one stays.)
        """
        loco = self.game.board.locomotive
        gray_slots: list[list[int]] = [[]]
        colored_ok: list[bool] = [False]
        for length in range(1, self.game.board.max_len + 1):
            slots = [c for c, r in enumerate(reach) if r >= length]
            wild_ok = wilds >= length
            if wild_ok:
                slots.append(loco)
            gray_slots.append(slots)
            colored_ok.append(wild_ok)
        return gray_slots, colored_ok

    def _legal_claims_into(self, out: list[int]) -> None:  # noqa: PLR0912
        """Bucketed, not a 900-way scan.

        Affordability depends only on `(length, color)`, so each of the board's buckets is
        tested once and only affordable ones have their segments walked. Buckets are sorted
        by length, so the loop can stop outright once routes are longer than the trains
        left.

        The branch count is over ruff's limit deliberately. This is the hottest loop in the
        engine -- roughly a third of a random playout -- and every candidate extraction
        costs a Python call per bucket, about 45 per step, which measures worse than the
        function it tidies.
        """
        board = self.game.board
        cur = self.cur
        trains = self.trains[cur]
        if trains == 0:
            return

        k = board.n_card_types
        base = cur * k
        hand = self.hand
        loco = board.locomotive
        wilds = hand[base + loco]

        # The longest route each color can pay for, 0 where we hold none of it. Folding the
        # `hand[c] >= 1` guard in here is what keeps pay slots bijective with payments.
        reach = [(hand[base + c] + wilds if hand[base + c] else 0) for c in range(board.n_colors)]

        gray_slots, colored_ok = self._pay_slots_by_length(wilds, reach)

        seg_owner = self.seg_owner
        sibling = board.sibling
        doubles_locked = self.game.doubles_locked
        append = out.append
        extend = out.extend

        for length, color, segments in board.buckets:
            if length > trains:
                break
            if color == GRAY:
                slots = gray_slots[length]
            elif reach[color] >= length:
                slots = [color, loco] if colored_ok[length] else [color]
            elif colored_ok[length]:
                slots = [loco]
            else:
                continue
            if not slots:
                continue
            single = slots[0] if len(slots) == 1 else -1
            for segment in segments:
                if seg_owner[segment] != FREE:
                    continue
                if not doubles_locked:
                    twin = sibling[segment]
                    if twin != NO_SIBLING and seg_owner[twin] == cur:
                        continue
                offset = segment * k
                # The single-slot case is the overwhelming majority; skipping the
                # temporary list there is worth the branch.
                if single >= 0:
                    append(offset + single)
                else:
                    extend([offset + p for p in slots])

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.phase == PHASE_TERMINAL

    def current_player(self) -> int:
        return self.cur

    def hand_of(self, player: int) -> memoryview:
        k = self.game.board.n_card_types
        return memoryview(self.hand)[player * k : (player + 1) * k]

    def certain_of(self, player: int) -> memoryview:
        k = self.game.board.n_card_types
        return memoryview(self.certain)[player * k : (player + 1) * k]

    def hand_size(self, player: int) -> int:
        k = self.game.board.n_card_types
        return sum(self.hand[player * k : (player + 1) * k])

    def tickets_of(self, player: int) -> list[int]:
        mask = self.tickets[player]
        return [t for t in range(self.game.board.n_tickets) if mask >> t & 1]

    def ticket_complete(self, player: int, ticket: int) -> bool:
        board = self.game.board
        return dsu_connected(self.dsu[player], board.ticket_a[ticket], board.ticket_b[ticket])

    def deck_counts(self) -> list[int]:
        """Composition of the undrawn deck.

        A *view* for determinization sampling and exact chance probabilities -- never the
        source of a draw. Drawing from counts instead of the materialized permutation is
        what silently destroys paired evaluation.
        """
        counts = [0] * self.game.board.n_card_types
        for card in self.deck[self.deck_pos :]:
            counts[card] += 1
        return counts

    def unseen_counts(self, observer: int) -> list[int]:
        """Cards `observer` cannot account for: the deck plus every opponent's blind draws."""
        board = self.game.board
        k = board.n_card_types
        counts = list(board.deck_composition_counts)
        for c in range(k):
            counts[c] -= self.hand[observer * k + c]
            counts[c] -= self.discard[c]
        for card in self.faceup:
            if card != EMPTY_SLOT:
                counts[card] -= 1
        for p in range(self.game.n_players):
            if p == observer:
                continue
            for c in range(k):
                counts[c] -= self.certain[p * k + c]
        return counts

    def tickets_remaining(self) -> int:
        return self.tdeck_len

    def history_actions(self) -> list[int]:
        """The action sequence, if `track_history` is on. Empty otherwise."""
        return list(self.history)

    # -----------------------------------------------------------------
    # Serialization and hashing -- docs/CONTRACT.md §3
    # -----------------------------------------------------------------

    def _serialize(self, *, canonical: bool) -> bytes:
        """The canonical byte image. `canonical=True` is the `position_hash` variant."""
        n = self.game.n_players
        if canonical:
            deck = bytes(self.deck_pos) + self.deck[self.deck_pos :]
            rng_state = rng_inc = 0
        else:
            deck = self.deck
            rng_state, rng_inc = self.rng.state, self.rng.inc

        return b"".join(
            (
                bytes(
                    (
                        CONTRACT_VERSION,
                        n,
                        self.phase,
                        self.cur,
                        self.draws_left,
                        self.final_left,
                        self.pass_streak,
                    )
                ),
                self.turn.to_bytes(2, "little"),
                bytes(self.seg_owner),
                bytes(self.hand),
                bytes(self.trains),
                b"".join(s.to_bytes(2, "little", signed=True) for s in self.score),
                b"".join(t.to_bytes(4, "little") for t in self.tickets),
                self.deck_pos.to_bytes(2, "little"),
                len(self.deck).to_bytes(2, "little"),
                deck,
                bytes(self.discard),
                bytes(c & 0xFF for c in self.faceup),
                bytes((self.tdeck_head, self.tdeck_len)),
                bytes(self.tdeck),
                bytes(self.certain),
                bytes(self.unknown),
                bytes((self.offer_len, *self.offer)),
                rng_state.to_bytes(8, "little"),
                rng_inc.to_bytes(8, "little"),
            )
        )

    def state_hash(self) -> int:
        """The differential-testing key: everything, RNG and undrawn deck included."""
        return hash64(self._serialize(canonical=False))

    def position_hash(self) -> int:
        """The MCTS transposition key: RNG and the already-dealt deck prefix zeroed.

        Cards that have been drawn are in hands, on the table or in the discard, so two
        states differing only in the order those cards came out are the same position.
        """
        return hash64(self._serialize(canonical=True))

    # -----------------------------------------------------------------
    # Self-check
    # -----------------------------------------------------------------

    def validate(self) -> None:
        """Assert every conservation law. Cheap enough to run after every step in tests.

        These are the invariants that would otherwise fail silently: a card that stops
        existing, a train that was never spent, information-set bookkeeping that drifts out
        of step with the hand it describes.
        """
        board, n = self.game.board, self.game.n_players
        k = board.n_card_types

        in_play = self.deck_pos < len(self.deck)
        held = sum(self.hand)
        on_table = sum(1 for c in self.faceup if c != EMPTY_SLOT)
        in_deck = len(self.deck) - self.deck_pos
        assert self.discard_total == sum(self.discard), "the discard fast-count drifted"
        assert held + on_table + self.discard_total + in_deck == board.deck_size, (
            f"cards leaked: {held} held + {on_table} face-up + {self.discard_total} discarded "
            f"+ {in_deck} in deck != {board.deck_size}"
        )

        for p in range(n):
            hand = sum(self.hand[p * k : (p + 1) * k])
            known = sum(self.certain[p * k : (p + 1) * k])
            assert known + self.unknown[p] == hand, (
                f"seat {p}: certain {known} + unknown {self.unknown[p]} != hand {hand}"
            )
            for c in range(k):
                assert self.certain[p * k + c] <= self.hand[p * k + c], (
                    f"seat {p} is publicly known to hold more {board.color_name(c)} than it has"
                )

            spent = sum(board.seg_len[s] for s in range(board.n_segments) if self.seg_owner[s] == p)
            assert self.trains[p] + spent == board.raw.trains_per_player, f"seat {p} trains"

        # Ticket conservation across hands, the deck and any open offer.
        in_hands = sum(int(self.tickets[p]).bit_count() for p in range(n))
        assert in_hands + self.tdeck_len + self.offer_len == board.n_tickets, "tickets leaked"
        ring = [t for t in self.tdeck if t != NO_TICKET]
        assert len(ring) == self.tdeck_len, "ticket ring length disagrees with tdeck_len"

        # The flush assertion most engines lack: 3+ locomotives face-up is only legitimate
        # when the guard (or the cascade cap) stopped the flush.
        locos = sum(1 for c in self.faceup if c == board.locomotive)
        if locos >= FLUSH_LOCOS:
            assert self._nonloco_available() < FLUSH_LOCOS or in_play, (
                f"{locos} locomotives face-up with {self._nonloco_available()} non-locomotives "
                "available: the flush should have fired"
            )

        assert 0 <= self.cur < n
        assert self.phase == PHASE_TERMINAL or self.legal_actions(), (
            "a non-terminal state with no legal actions"
        )

    # -----------------------------------------------------------------
    # Cloning
    # -----------------------------------------------------------------

    def clone(self) -> State:
        """A fully independent copy. `deck` is immutable `bytes` and is shared."""
        other = State.__new__(State)
        other.game = self.game
        other.seed = self.seed
        other.seg_owner = bytearray(self.seg_owner)
        other.hand = bytearray(self.hand)
        other.trains = bytearray(self.trains)
        other.score = self.score[:]
        other.tickets = self.tickets[:]
        other.deck = self.deck
        other.deck_pos = self.deck_pos
        other.discard = bytearray(self.discard)
        other.discard_total = self.discard_total
        other.faceup = self.faceup[:]
        other.tdeck = self.tdeck[:]
        other.tdeck_head = self.tdeck_head
        other.tdeck_len = self.tdeck_len
        other.dsu = [bytearray(d) for d in self.dsu]
        other.certain = bytearray(self.certain)
        other.unknown = bytearray(self.unknown)
        other.offer = self.offer[:]
        other.offer_len = self.offer_len
        other.phase = self.phase
        other.cur = self.cur
        other.draws_left = self.draws_left
        other.final_left = self.final_left
        other.pass_streak = self.pass_streak
        other.turn = self.turn
        other.board_version = self.board_version
        other.rng = Pcg32(self.rng.state, self.rng.inc)
        other.history = self.history[:] if self.game.cfg.track_history else []
        return other

    def clone_into(self, dst: State) -> None:
        """Overwrite `dst` in place, so search never allocates inside its arena."""
        dst.game = self.game
        dst.seed = self.seed
        dst.seg_owner[:] = self.seg_owner
        dst.hand[:] = self.hand
        dst.trains[:] = self.trains
        dst.score[:] = self.score
        dst.tickets[:] = self.tickets
        dst.deck = self.deck
        dst.deck_pos = self.deck_pos
        dst.discard[:] = self.discard
        dst.discard_total = self.discard_total
        dst.faceup[:] = self.faceup
        dst.tdeck[:] = self.tdeck
        dst.tdeck_head = self.tdeck_head
        dst.tdeck_len = self.tdeck_len
        for mine, theirs in zip(self.dsu, dst.dsu, strict=True):
            theirs[:] = mine
        dst.certain[:] = self.certain
        dst.unknown[:] = self.unknown
        dst.offer[:] = self.offer
        dst.offer_len = self.offer_len
        dst.phase = self.phase
        dst.cur = self.cur
        dst.draws_left = self.draws_left
        dst.final_left = self.final_left
        dst.pass_streak = self.pass_streak
        dst.turn = self.turn
        dst.board_version = self.board_version
        dst.rng.state, dst.rng.inc = self.rng.state, self.rng.inc
        dst.history[:] = self.history if self.game.cfg.track_history else []

    def __repr__(self) -> str:
        return (
            f"<State {self.game.board.name} {self.game.n_players}P turn={self.turn} "
            f"phase={PHASE_NAMES[self.phase]} cur={self.cur} "
            f"trains={list(self.trains)} score={self.score}>"
        )
