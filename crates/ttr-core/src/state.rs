//! The engine: [`Game`] (immutable setup) and [`State`] (a flat, cloneable position).
//!
//! A transcription of `ticket_to_ride/engine/state.py` against docs/CONTRACT.md §2-§3,
//! not a redesign. Rules edge cases sit next to the code implementing them, and each is
//! the same edge case, in the same order, as the Python engine -- because the differential
//! harness compares them step by step, and a difference that is "obviously equivalent" is
//! exactly the kind that is not.
//!
//! **Deviation from PLAN.md §5.1, small and deliberate:** the plan says `State: Copy`.
//! `State` here is `Clone`, not `Copy`, because it carries a `Vec<u16>` action history
//! just as the Python one does. Everything `Copy` was meant to buy is still true: the POD
//! arrays clone as a memcpy, [`State::clone_into`] reuses the destination's allocation so
//! a search arena never allocates, and search runs with `track_history` off, where the
//! vector is empty and costs nothing.

use crate::actions::{ActionSpace, BLIND_SLOT, N_KEEP_ACTIONS};
use crate::board::{
    Board, FACEUP_SLOTS, GRAY, MAX_CARD_TYPES, MAX_CITIES, MAX_DECK, MAX_PLAYERS, MAX_ROUTE_LEN,
    MAX_SEGMENTS, MAX_TICKETS, NO_SIBLING,
};
use crate::config::{
    CLOSED, ConfigError, EMPTY_SLOT, END_TRIGGER_TRAINS, FLUSH_LOCOS, FREE, NO_TICKET,
    NOT_TRIGGERED, PHASE_DRAW_SECOND, PHASE_INITIAL_TICKETS, PHASE_MAIN, PHASE_NAMES,
    PHASE_TERMINAL, PHASE_TICKET_KEEP, RuleConfig,
};
use crate::graph::{dsu_connected, dsu_find, dsu_union};
use crate::hashing::hash64;
use crate::rng::{Part, Pcg32, stream};

/// An action that is not legal in the current state.
///
/// Reported rather than silently tolerated: a policy that leaks an illegal action past its
/// mask is the classic silent RL bug, and it poisons training with no visible symptom.
/// Every step path re-checks the specific preconditions of the action it is given, which
/// costs a handful of comparisons -- far cheaper than a full legality scan, and enough to
/// make the leak impossible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IllegalAction(pub String);

impl std::fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for IllegalAction {}

macro_rules! illegal {
    ($($arg:tt)*) => { Err(IllegalAction(format!($($arg)*))) };
}

/// The play-affecting scalars of a [`RuleConfig`], flattened into the state so the hot
/// paths never chase a pointer to the config or re-derive `doubles_locked`.
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub n_players: u8,
    pub doubles_locked: bool,
    pub initial_hand: u8,
    pub turn_cap: u16,
    pub flush_cascade_cap: u8,
    pub track_history: bool,
}

/// Immutable per-configuration setup: the board, the action space, the rule constants.
#[derive(Clone, Debug)]
pub struct Game {
    pub cfg: RuleConfig,
    pub board: &'static Board,
    pub space: ActionSpace,
    pub rules: Rules,
}

impl Game {
    pub fn new(cfg: RuleConfig) -> Result<Self, ConfigError> {
        cfg.validate()?;
        let board = cfg.board();
        Ok(Self {
            space: ActionSpace::new(board),
            rules: Rules {
                n_players: cfg.n_players as u8,
                doubles_locked: cfg.doubles_locked_for_everyone(),
                initial_hand: cfg.initial_hand,
                turn_cap: cfg.turn_cap,
                flush_cascade_cap: cfg.flush_cascade_cap,
                track_history: cfg.track_history,
            },
            board,
            cfg,
        })
    }

    pub fn num_distinct_actions(&self) -> u16 {
        self.space.n
    }

    pub fn new_initial_state(&self, seed: u64) -> State {
        State::new(self, seed)
    }

    pub fn action_to_string(&self, action: u16) -> String {
        self.space.to_string(self.board, action)
    }
}

/// One position. Cloneable, hashable, and steppable.
///
/// Arrays are sized to the largest generated board rather than the board in play (see
/// [`crate::board`]'s compile-time maxima), so a clone is a fixed-size memcpy. Only the
/// prefixes the board actually uses are ever read, and only those reach the hash.
#[derive(Clone)]
pub struct State {
    pub board: &'static Board,
    pub space: ActionSpace,
    pub rules: Rules,
    pub seed: u64,

    pub seg_owner: [u8; MAX_SEGMENTS],
    /// Seat-major card counts: `hand[p * n_card_types + c]`.
    pub hand: [u8; MAX_PLAYERS * MAX_CARD_TYPES],
    pub trains: [u8; MAX_PLAYERS],
    pub score: [i16; MAX_PLAYERS],
    /// One 30-bit ticket mask per seat.
    pub tickets: [u32; MAX_PLAYERS],

    /// The materialized permutation. The consumed prefix is **retained, not erased** --
    /// `state_hash()` includes it, because for differential testing "the same game" means
    /// identical down to the unrealized future.
    pub deck: [u8; MAX_DECK],
    /// Live length: a reshuffle installs a shorter deck, and the serialization records it.
    pub deck_len: u16,
    pub deck_pos: u16,

    pub discard: [u8; MAX_CARD_TYPES],
    /// Kept alongside `discard` purely for speed: this sits in the innermost legality
    /// loop, where a 9-element scan is measurable. Derived, so never serialized.
    pub discard_total: u16,
    pub faceup: [i8; FACEUP_SLOTS],
    /// Whether the last flush stopped on its cascade cap. Derived bookkeeping for
    /// [`State::validate`], deliberately **not** serialized: two states differing only in
    /// it have identical futures.
    pub flush_capped: bool,

    pub certain: [u8; MAX_PLAYERS * MAX_CARD_TYPES],
    pub unknown: [u8; MAX_PLAYERS],
    /// Per-player union-find over the cities, `dsu[p * MAX_CITIES + city]`. A **cache**,
    /// fully derivable from `seg_owner`, and excluded from every hash (CONTRACT.md §3.1).
    pub dsu: [u8; MAX_PLAYERS * MAX_CITIES],

    /// Ring buffer with head and length. Ticket order *is* observable in one direction:
    /// returned tickets go to the bottom and can come back around.
    pub tdeck: [u8; MAX_TICKETS],
    pub tdeck_head: u8,
    pub tdeck_len: u8,

    pub offer: [u8; 3],
    pub offer_len: u8,

    pub phase: u8,
    pub cur: u8,
    pub draws_left: u8,
    pub final_left: u8,
    pub pass_streak: u8,
    pub turn: u16,
    /// Bumped on every claim, for callers caching per-(player, board-version) graph work.
    pub board_version: u32,

    /// Seeded from `("env", "reshuffle")` and advanced by **nothing else**, so a game that
    /// never exhausts the deck never touches it.
    pub rng: Pcg32,

    pub history: Vec<u16>,
}

impl State {
    // -----------------------------------------------------------------
    // Setup -- docs/CONTRACT.md §2.2
    // -----------------------------------------------------------------

    fn new(game: &Game, seed: u64) -> Self {
        let board = game.board;
        let n = game.rules.n_players as usize;

        let mut dsu = [0u8; MAX_PLAYERS * MAX_CITIES];
        for p in 0..n {
            for c in 0..board.n_cities {
                dsu[p * MAX_CITIES + c] = c as u8;
            }
        }
        let mut trains = [0u8; MAX_PLAYERS];
        trains[..n].fill(board.raw.trains_per_player);

        let mut state = Self {
            board,
            space: game.space,
            rules: game.rules,
            seed,
            seg_owner: [FREE; MAX_SEGMENTS],
            hand: [0; MAX_PLAYERS * MAX_CARD_TYPES],
            trains,
            score: [0; MAX_PLAYERS],
            tickets: [0; MAX_PLAYERS],
            deck: [0; MAX_DECK],
            deck_len: 0,
            deck_pos: 0,
            discard: [0; MAX_CARD_TYPES],
            discard_total: 0,
            faceup: [EMPTY_SLOT; FACEUP_SLOTS],
            flush_capped: false,
            certain: [0; MAX_PLAYERS * MAX_CARD_TYPES],
            unknown: [0; MAX_PLAYERS],
            dsu,
            tdeck: [NO_TICKET; MAX_TICKETS],
            tdeck_head: 0,
            tdeck_len: 0,
            offer: [NO_TICKET; 3],
            offer_len: 0,
            phase: PHASE_INITIAL_TICKETS,
            cur: 0,
            draws_left: 0,
            final_left: NOT_TRIGGERED,
            pass_streak: 0,
            turn: 0,
            board_version: 0,
            rng: Pcg32::new(0, 1),
            history: Vec::new(),
        };
        state.setup(seed);
        state
    }

    fn setup(&mut self, seed: u64) {
        let board = self.board;
        let n = self.rules.n_players as usize;

        let mut deck = board.deck_composition.clone();
        stream(seed, &[Part::Str("env"), Part::Str("deck")]).shuffle(&mut deck);
        self.deck[..deck.len()].copy_from_slice(&deck);
        self.deck_len = deck.len() as u16;
        self.deck_pos = 0;

        let mut tickets: Vec<u8> = (0..board.n_tickets as u8).collect();
        stream(seed, &[Part::Str("env"), Part::Str("tickets")]).shuffle(&mut tickets);
        self.tdeck[..tickets.len()].copy_from_slice(&tickets);
        self.tdeck_head = 0;
        self.tdeck_len = board.n_tickets as u8;

        self.rng = stream(seed, &[Part::Str("env"), Part::Str("reshuffle")]);

        // Hands first, in per-seat blocks, *then* the display. Flipping first would deal
        // every seat different cards.
        let k = board.n_card_types;
        for p in 0..n {
            for _ in 0..self.rules.initial_hand {
                let card = self
                    .draw_card()
                    .expect("deck exhausted during the initial deal");
                self.hand[p * k + card as usize] += 1;
                self.unknown[p] += 1;
            }
        }

        self.refill();
        self.flush_check();

        self.phase = PHASE_INITIAL_TICKETS;
        self.cur = 0;
        self.begin_initial_offer();
    }

    /// Deal and, where forced, auto-resolve every seat's opening ticket choice.
    ///
    /// Physically the offers are dealt and chosen simultaneously. Dealing in seat order
    /// gives each seat the same cards it would have got, and because keeps are secret and
    /// no seat observes another's choice, sequentializing the *choice* is
    /// information-equivalent. This is the one documented rules deviation (CONTRACT §2.2).
    fn begin_initial_offer(&mut self) {
        let raw = self.board.raw;
        while (self.cur as usize) < self.rules.n_players as usize {
            if self.deal_offer(raw.initial_ticket_deal, raw.initial_ticket_keep_min) {
                return; // this seat has a real decision to make
            }
            self.cur += 1;
        }
        self.cur = 0;
        self.phase = PHASE_MAIN;
        self.turn = 0;
    }

    // -----------------------------------------------------------------
    // Card plumbing -- docs/CONTRACT.md §2.3-2.6
    // -----------------------------------------------------------------

    /// The top card, reshuffling **lazily** if the deck is spent.
    ///
    /// Reshuffling eagerly the moment the cursor reaches the end would make "deck empty,
    /// discards available" and "deck and discards both empty" indistinguishable -- and
    /// those two states have different legal actions.
    fn draw_card(&mut self) -> Option<u8> {
        if self.deck_pos >= self.deck_len && !self.reshuffle() {
            return None;
        }
        let card = self.deck[self.deck_pos as usize];
        self.deck_pos += 1;
        Some(card)
    }

    /// Rebuild the deck from the discard. Returns false if there is nothing to reshuffle.
    fn reshuffle(&mut self) -> bool {
        if self.discard_total == 0 {
            return false;
        }
        // Canonical order before shuffling: discard *order* is never observable, so the
        // multiset is the only real information and the permutation must come from the
        // stream alone.
        let mut cards: Vec<u8> = Vec::with_capacity(self.discard_total as usize);
        for card_type in 0..self.board.n_card_types {
            let count = self.discard[card_type];
            cards.extend(std::iter::repeat_n(card_type as u8, count as usize));
            self.discard[card_type] = 0;
        }
        self.discard_total = 0;
        self.rng.shuffle(&mut cards);
        self.deck[..cards.len()].copy_from_slice(&cards);
        self.deck_len = cards.len() as u16;
        self.deck_pos = 0;
        true
    }

    /// Top the display back up to five, ascending slot order.
    fn refill(&mut self) {
        if !self.faceup.contains(&EMPTY_SLOT) {
            return;
        }
        for i in 0..FACEUP_SLOTS {
            if self.faceup[i] == EMPTY_SLOT {
                match self.draw_card() {
                    Some(card) => self.faceup[i] = card as i8,
                    // Every pool is empty; the display legitimately holds fewer than five.
                    None => return,
                }
            }
        }
    }

    fn cards_available(&self) -> bool {
        self.deck_pos < self.deck_len || self.discard_total > 0
    }

    /// Three or more face-up locomotives: discard all five and deal five new ones.
    ///
    /// The replacement five can themselves contain three locomotives, so this cascades.
    /// Two locks on the most common hang bug in TTR implementations:
    ///
    /// 1. The **guard** -- with 14 locomotives in a 110-card deck an all-locomotive pool
    ///    would loop forever, so a flush only happens when the available pool still holds
    ///    three non-locomotives.
    /// 2. The **cascade cap**, a deterministic bail-out that simply stops flushing.
    ///
    /// After a bail-out the display may legitimately show 3+ locomotives, and
    /// `flush_capped` records that it happened so [`State::validate`] can tell that case
    /// apart from a flush the engine simply forgot to run.
    ///
    /// **The cap is not merely belt and braces.** It fires in real late-game 5P positions:
    /// once most of the deck is in players' hands the available pool can be small and
    /// locomotive-heavy, and every reflush deals three more. Measured on seed 15 of the 5P
    /// sweep, at turn 243.
    fn flush_check(&mut self) {
        let loco = self.board.locomotive as i8;
        self.flush_capped = false;
        for _ in 0..self.rules.flush_cascade_cap {
            if self.faceup.iter().filter(|&&c| c == loco).count() < FLUSH_LOCOS {
                return;
            }
            if self.nonloco_available() < FLUSH_LOCOS {
                return;
            }
            for i in 0..FACEUP_SLOTS {
                let card = self.faceup[i];
                if card != EMPTY_SLOT {
                    self.discard[card as usize] += 1;
                    self.discard_total += 1;
                    self.faceup[i] = EMPTY_SLOT;
                }
            }
            self.refill();
        }
        // Fell out of the loop rather than returning: the cascade hit its cap.
        self.flush_capped = self.faceup.iter().filter(|&&c| c == loco).count() >= FLUSH_LOCOS;
    }

    /// Non-locomotives left in the deck plus the discard, measured before a flush.
    fn nonloco_available(&self) -> usize {
        let loco = self.board.locomotive;
        let remaining = &self.deck[self.deck_pos as usize..self.deck_len as usize];
        let in_deck = remaining.len() - remaining.iter().filter(|&&c| c == loco).count();
        let in_discard = self.discard_total as usize - self.discard[loco as usize] as usize;
        in_deck + in_discard
    }

    // -----------------------------------------------------------------
    // Ticket plumbing -- docs/CONTRACT.md §2.7
    // -----------------------------------------------------------------

    fn take_ticket(&mut self) -> u8 {
        // The ring's capacity is the board's ticket count, **not** the array length: the
        // array is sized for the largest map, and using its length as the modulus would
        // silently corrupt every wrap on TTR-mini.
        let capacity = self.board.n_tickets as u8;
        let ticket = self.tdeck[self.tdeck_head as usize];
        // Vacated slots are blanked because the whole array is serialized: a stale id left
        // behind would let two semantically identical positions hash differently.
        self.tdeck[self.tdeck_head as usize] = NO_TICKET;
        self.tdeck_head = (self.tdeck_head + 1) % capacity;
        self.tdeck_len -= 1;
        ticket
    }

    /// To the *bottom*, which is observable and therefore part of the state.
    fn return_ticket(&mut self, ticket: u8) {
        let capacity = self.board.n_tickets as u8;
        let slot = (self.tdeck_head as usize + self.tdeck_len as usize) % capacity as usize;
        self.tdeck[slot] = ticket;
        self.tdeck_len += 1;
    }

    /// Deal an offer. Returns true if the seat actually has a decision to make.
    ///
    /// With exactly one legal keep mask -- one ticket left in the deck, or a forced
    /// `keep >= n` -- the choice is resolved here rather than burning a network evaluation
    /// on a decision with a single option.
    fn deal_offer(&mut self, deal: u8, keep_min: u8) -> bool {
        let take = deal.min(self.tdeck_len);
        for i in 0..take as usize {
            self.offer[i] = self.take_ticket();
        }
        for i in take as usize..3 {
            self.offer[i] = NO_TICKET;
        }
        self.offer_len = take;

        let (masks, count) = self.legal_keep_masks(Some(keep_min));
        if count == 1 {
            self.apply_keep(masks[0]);
            return false;
        }
        true
    }

    /// Non-empty subsets of the offer that keep at least the minimum.
    ///
    /// The initial 3-of-3-keep-2 offer has exactly four: 011, 101, 110, 111. Returned as a
    /// fixed array rather than a `Vec` because this sits on the legality path.
    fn legal_keep_masks(&self, keep_min: Option<u8>) -> ([u16; N_KEEP_ACTIONS as usize], usize) {
        let raw = self.board.raw;
        let keep_min = keep_min.unwrap_or(if self.phase == PHASE_INITIAL_TICKETS {
            raw.initial_ticket_keep_min
        } else {
            raw.draw_ticket_keep_min
        });
        let n = self.offer_len;
        let need = u32::from(keep_min.min(n));
        let mut out = [0u16; N_KEEP_ACTIONS as usize];
        let mut count = 0;
        for m in 1u16..(1 << n) {
            if m.count_ones() >= need {
                out[count] = m;
                count += 1;
            }
        }
        (out, count)
    }

    fn apply_keep(&mut self, mask: u16) {
        let cur = self.cur as usize;
        for i in 0..self.offer_len as usize {
            let ticket = self.offer[i];
            if mask & (1 << i) != 0 {
                self.tickets[cur] |= 1 << ticket;
            } else {
                self.return_ticket(ticket);
            }
        }
        self.offer_len = 0;
        self.offer = [NO_TICKET; 3];
    }

    // -----------------------------------------------------------------
    // Turn structure
    // -----------------------------------------------------------------

    fn end_turn(&mut self, passed: bool) {
        let n = self.rules.n_players;
        self.turn += 1;
        self.draws_left = 0;
        self.pass_streak = if passed { self.pass_streak + 1 } else { 0 };

        // Every seat passing in a row means the position is frozen. Without this the
        // engine simply hangs.
        if self.pass_streak >= n {
            self.phase = PHASE_TERMINAL;
            return;
        }

        if self.final_left == NOT_TRIGGERED {
            // Fires at *end of turn*, after every turn type: a seat sitting on two trains
            // that spends its turn drawing still triggers the ending.
            if self.trains[self.cur as usize] <= END_TRIGGER_TRAINS {
                self.final_left = n;
            }
        } else {
            self.final_left -= 1;
            if self.final_left == 0 {
                self.phase = PHASE_TERMINAL;
                return;
            }
        }

        if self.turn >= self.rules.turn_cap {
            self.phase = PHASE_TERMINAL;
            return;
        }

        // A claim moves cards into the discard, so a display that ran short earlier can be
        // topped up now. Without this the display would stay short forever once every pool
        // emptied, since only *taking* a face-up card would otherwise trigger a refill.
        self.refill();
        self.flush_check();

        self.cur = (self.cur + 1) % n;
        self.phase = PHASE_MAIN;
    }

    // -----------------------------------------------------------------
    // Stepping
    // -----------------------------------------------------------------

    pub fn step(&mut self, action: u16) -> Result<(), IllegalAction> {
        if action >= self.space.n {
            return illegal!("action {action} outside 0..{}", self.space.n - 1);
        }

        match self.phase {
            PHASE_MAIN => self.step_main(action)?,
            PHASE_DRAW_SECOND => self.step_draw_second(action)?,
            PHASE_INITIAL_TICKETS | PHASE_TICKET_KEEP => self.step_keep(action)?,
            _ => return illegal!("the game is over"),
        }

        // Recorded only once the action has actually been applied. Every step path
        // validates before it mutates, so a rejected action leaves the state untouched --
        // and the history has to stay untouched with it, or a replay of a game that
        // survived an illegal action would diverge.
        if self.rules.track_history {
            self.history.push(action);
        }
        Ok(())
    }

    fn step_main(&mut self, action: u16) -> Result<(), IllegalAction> {
        let space = self.space;
        if action < space.claim_end {
            let (segment, pay) = space.decode_claim(action);
            self.claim(segment, pay)
        } else if action < space.draw_tickets {
            self.draw_first(action - space.draw_base)
        } else if action == space.draw_tickets {
            self.draw_tickets()
        } else if action == space.pass_action {
            self.do_pass()
        } else {
            illegal!("KEEP is only legal while a ticket offer is open")
        }
    }

    fn step_draw_second(&mut self, action: u16) -> Result<(), IllegalAction> {
        let space = self.space;
        if !(space.draw_base..space.draw_tickets).contains(&action) {
            return illegal!("only a card draw is legal as the second draw");
        }
        let slot = action - space.draw_base;
        let k = self.board.n_card_types;
        if slot == BLIND_SLOT {
            let Some(card) = self.draw_card() else {
                return illegal!("no cards left to draw");
            };
            self.hand[self.cur as usize * k + card as usize] += 1;
            self.unknown[self.cur as usize] += 1;
        } else {
            let card = self.faceup[slot as usize];
            if card == EMPTY_SLOT {
                return illegal!("face-up slot {slot} is empty");
            }
            // A face-up locomotive is worth two cards, so it may never be the second.
            if card == self.board.locomotive as i8 {
                return illegal!("a face-up locomotive cannot be taken as the second card");
            }
            self.take_faceup(slot as usize, card as u8);
        }
        self.end_turn(false);
        Ok(())
    }

    fn step_keep(&mut self, action: u16) -> Result<(), IllegalAction> {
        let space = self.space;
        if action <= space.keep_base() || action >= space.pass_action {
            return illegal!("only a ticket keep is legal here");
        }
        let mask = action - space.keep_base();
        let (masks, count) = self.legal_keep_masks(None);
        if !masks[..count].contains(&mask) {
            return illegal!("keep mask {mask:03b} is not legal for this offer");
        }

        let initial = self.phase == PHASE_INITIAL_TICKETS;
        self.apply_keep(mask);
        if initial {
            self.cur += 1;
            self.begin_initial_offer();
        } else {
            self.end_turn(false);
        }
        Ok(())
    }

    // -- the three real turn actions ---------------------------------------

    fn claim(&mut self, segment: u16, pay: u16) -> Result<(), IllegalAction> {
        let board = self.board;
        let cur = self.cur;
        let seg = segment as usize;
        let owner = self.seg_owner[seg];
        if owner != FREE {
            let what = if owner == CLOSED { "closed" } else { "taken" };
            return illegal!("segment {segment} is {what}");
        }

        let length = board.seg_len[seg];
        if self.trains[cur as usize] < length {
            return illegal!(
                "{} trains left, route needs {length}",
                self.trains[cur as usize]
            );
        }

        let sibling = board.sibling[seg];
        // 4-5P: the sibling stays open to *others*, never to me. In 2-3P the sibling was
        // marked CLOSED when the first track was claimed, so `owner != FREE` covered it.
        if !self.rules.doubles_locked
            && sibling != NO_SIBLING
            && self.seg_owner[sibling as usize] == cur
        {
            return illegal!("one player may not own both tracks of a double route");
        }

        let pay = pay as u8;
        let (colored, wilds) = self.payment_for(seg, length, pay)?;

        self.spend(pay, colored);
        self.spend(board.locomotive, wilds);

        self.seg_owner[seg] = cur;
        if self.rules.doubles_locked && sibling != NO_SIBLING {
            self.seg_owner[sibling as usize] = CLOSED;
        }
        self.trains[cur as usize] -= length;
        self.score[cur as usize] += i16::from(board.route_points[length as usize]);
        let (a, b) = (board.seg_a[seg], board.seg_b[seg]);
        dsu_union(self.dsu_mut(cur as usize), a, b);
        self.board_version += 1;
        self.end_turn(false);
        Ok(())
    }

    /// How the current seat would pay `pay` for a route of `length` on `seg`: `(coloured
    /// cards, locomotives)`.
    ///
    /// Split out of [`State::claim`] so the heuristics can price a payment without
    /// restating the rule. A second copy of this arithmetic in the agents would drift from
    /// the engine's, and the symptom would be an agent that scores a claim it cannot
    /// actually make -- silently, since it would simply never pick that action again.
    /// The check order and every message are `claim`'s, unchanged.
    pub fn payment_for(&self, seg: usize, length: u8, pay: u8) -> Result<(u8, u8), IllegalAction> {
        let board = self.board;
        let base = self.cur as usize * board.n_card_types;
        let loco = board.locomotive;
        if pay == loco {
            if self.hand[base + loco as usize] < length {
                return illegal!("not enough locomotives");
            }
            return Ok((0, length));
        }
        let required = board.seg_color[seg];
        if required != GRAY && required != pay {
            return illegal!(
                "route needs {}, not {}",
                board.color_name(required),
                board.color_name(pay)
            );
        }
        let have = self.hand[base + pay as usize];
        // The `have >= 1` guard is what keeps action ids bijective with payments: a hand of
        // pure locomotives would otherwise make all eight gray pay slots legal *and
        // identical*, which poisons the policy distribution.
        if have < 1 || have + self.hand[base + loco as usize] < length {
            return illegal!("cannot pay {length} with {}", board.color_name(pay));
        }
        let colored = have.min(length);
        Ok((colored, length - colored))
    }

    /// Pay `count` cards and keep the public knowledge of this hand consistent.
    ///
    /// Cards spent on a claim are publicly discarded, so they cancel what opponents know
    /// for certain before they cancel what they only know the count of.
    fn spend(&mut self, card_type: u8, count: u8) {
        if count == 0 {
            return;
        }
        let index = self.cur as usize * self.board.n_card_types + card_type as usize;
        self.hand[index] -= count;
        self.discard[card_type as usize] += count;
        self.discard_total += u16::from(count);
        let known = self.certain[index];
        if known >= count {
            self.certain[index] = known - count;
        } else {
            self.certain[index] = 0;
            self.unknown[self.cur as usize] -= count - known;
        }
    }

    fn draw_first(&mut self, slot: u16) -> Result<(), IllegalAction> {
        let k = self.board.n_card_types;
        if slot == BLIND_SLOT {
            let Some(card) = self.draw_card() else {
                return illegal!("no cards left to draw");
            };
            self.hand[self.cur as usize * k + card as usize] += 1;
            self.unknown[self.cur as usize] += 1;
        } else {
            let card = self.faceup[slot as usize];
            if card == EMPTY_SLOT {
                return illegal!("face-up slot {slot} is empty");
            }
            self.take_faceup(slot as usize, card as u8);
            // A face-up locomotive counts as the whole turn. A locomotive drawn *blind*
            // does not -- it is an ordinary card and the turn continues.
            if card == self.board.locomotive as i8 {
                self.end_turn(false);
                return Ok(());
            }
        }

        if self.can_draw_second() {
            self.phase = PHASE_DRAW_SECOND;
            self.draws_left = 1;
        } else {
            // Never construct a node with zero legal actions.
            self.end_turn(false);
        }
        Ok(())
    }

    fn take_faceup(&mut self, slot: usize, card: u8) {
        let index = self.cur as usize * self.board.n_card_types + card as usize;
        self.hand[index] += 1;
        self.certain[index] += 1;
        self.faceup[slot] = EMPTY_SLOT;
        // Refill *before* the second draw, and the replacement may itself be a locomotive.
        self.refill();
        self.flush_check();
    }

    fn can_draw_second(&self) -> bool {
        let loco = self.board.locomotive as i8;
        if self.faceup.iter().any(|&c| c != EMPTY_SLOT && c != loco) {
            return true;
        }
        self.cards_available()
    }

    fn draw_tickets(&mut self) -> Result<(), IllegalAction> {
        if self.tdeck_len == 0 {
            return illegal!("the ticket deck is empty");
        }
        let raw = self.board.raw;
        if self.deal_offer(raw.draw_ticket_deal, raw.draw_ticket_keep_min) {
            self.phase = PHASE_TICKET_KEEP;
        } else {
            self.end_turn(false);
        }
        Ok(())
    }

    fn do_pass(&mut self) -> Result<(), IllegalAction> {
        let mut probe = Vec::new();
        self.legal_main_into(&mut probe);
        if !probe.is_empty() {
            return illegal!("PASS is only legal with no other option");
        }
        self.end_turn(true);
        Ok(())
    }

    // -----------------------------------------------------------------
    // Legality
    // -----------------------------------------------------------------

    /// Sorted ascending, so two engines' lists compare directly.
    pub fn legal_actions(&self) -> Vec<u16> {
        let mut out = Vec::new();
        self.legal_into(&mut out);
        out.sort_unstable();
        out
    }

    /// A uniformly random legal action, skipping the sort [`State::legal_actions`] pays
    /// for. Rollouts do not care about order, and at ~150 steps a game the sort is a real
    /// fraction of a random playout.
    ///
    /// `scratch` is reused so a rollout never allocates.
    pub fn sample_legal_into(&self, rng: &mut Pcg32, scratch: &mut Vec<u16>) -> Option<u16> {
        scratch.clear();
        self.legal_into(scratch);
        if scratch.is_empty() {
            return None;
        }
        Some(scratch[rng.below(scratch.len() as u32) as usize])
    }

    pub fn sample_legal(&self, rng: &mut Pcg32) -> Option<u16> {
        let mut scratch = Vec::new();
        self.sample_legal_into(rng, &mut scratch)
    }

    /// A 0/1 mask over the whole action space, written into `out`.
    pub fn legal_action_mask(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.space.n as usize,
            "mask buffer is the wrong size"
        );
        out.fill(0);
        let mut actions = Vec::new();
        self.legal_into(&mut actions);
        for a in actions {
            out[a as usize] = 1;
        }
    }

    /// Every legal action, **unsorted**, in the same emission order as Python's.
    pub fn legal_into(&self, out: &mut Vec<u16>) {
        match self.phase {
            PHASE_MAIN => {
                self.legal_main_into(out);
                if out.is_empty() {
                    out.push(self.space.pass_action);
                }
            }
            PHASE_DRAW_SECOND => self.legal_draws_into(out, false),
            PHASE_INITIAL_TICKETS | PHASE_TICKET_KEEP => {
                let base = self.space.keep_base();
                let (masks, count) = self.legal_keep_masks(None);
                out.extend(masks[..count].iter().map(|m| base + m));
            }
            _ => {}
        }
    }

    /// Everything legal in MAIN *except* PASS, which is only legal if this is empty.
    fn legal_main_into(&self, out: &mut Vec<u16>) {
        self.legal_claims_into(out);
        self.legal_draws_into(out, true);
        if self.tdeck_len > 0 {
            out.push(self.space.draw_tickets);
        }
    }

    fn legal_draws_into(&self, out: &mut Vec<u16>, first: bool) {
        let base = self.space.draw_base;
        // A face-up locomotive is worth two cards, so it is only takeable as the first.
        let blocked = if first {
            EMPTY_SLOT
        } else {
            self.board.locomotive as i8
        };
        for (i, &card) in self.faceup.iter().enumerate() {
            if card != EMPTY_SLOT && card != blocked {
                out.push(base + i as u16);
            }
        }
        if self.cards_available() {
            out.push(base + BLIND_SLOT);
        }
    }

    /// Bucketed, not a 900-way scan.
    ///
    /// Affordability depends only on `(length, colour)`, so each of the board's buckets is
    /// tested once and only affordable ones have their segments walked. Buckets are sorted
    /// by length, so the loop stops outright once routes are longer than the trains left.
    fn legal_claims_into(&self, out: &mut Vec<u16>) {
        let board = self.board;
        let cur = self.cur;
        let trains = self.trains[cur as usize];
        if trains == 0 {
            return;
        }

        let k = board.n_card_types;
        let base = cur as usize * k;
        let loco = board.locomotive;
        let wilds = self.hand[base + loco as usize];

        // The longest route each colour can pay for, 0 where we hold none of it. Folding
        // the `hand[c] >= 1` guard in here is what keeps pay slots bijective with payments.
        let mut reach = [0u8; MAX_CARD_TYPES];
        for (r, &held) in reach
            .iter_mut()
            .zip(&self.hand[base..base + board.n_colors])
        {
            *r = if held > 0 { held + wilds } else { 0 };
        }

        // Which pay slots each route length admits, resolved once per legality scan. A
        // coloured route's own slot is a single `reach` compare, so it needs no table.
        let mut gray_slots = [[0u16; MAX_CARD_TYPES]; MAX_ROUTE_LEN + 1];
        let mut gray_n = [0usize; MAX_ROUTE_LEN + 1];
        let mut colored_ok = [false; MAX_ROUTE_LEN + 1];
        for length in 1..=board.max_len as usize {
            let mut n = 0;
            for (c, &r) in reach.iter().take(board.n_colors).enumerate() {
                if r as usize >= length {
                    gray_slots[length][n] = c as u16;
                    n += 1;
                }
            }
            let wild_ok = wilds as usize >= length;
            if wild_ok {
                gray_slots[length][n] = u16::from(loco);
                n += 1;
            }
            gray_n[length] = n;
            colored_ok[length] = wild_ok;
        }

        let k = k as u16;
        let mut pair = [0u16; 2];
        for bucket in &board.buckets {
            if bucket.length > trains {
                break;
            }
            let length = bucket.length as usize;
            let slots: &[u16] = if bucket.color == GRAY {
                &gray_slots[length][..gray_n[length]]
            } else if reach[bucket.color as usize] as usize >= length {
                pair[0] = u16::from(bucket.color);
                if colored_ok[length] {
                    pair[1] = u16::from(loco);
                    &pair[..2]
                } else {
                    &pair[..1]
                }
            } else if colored_ok[length] {
                pair[0] = u16::from(loco);
                &pair[..1]
            } else {
                continue;
            };
            if slots.is_empty() {
                continue;
            }
            for &segment in &bucket.segments {
                if self.seg_owner[segment as usize] != FREE {
                    continue;
                }
                if !self.rules.doubles_locked {
                    let twin = board.sibling[segment as usize];
                    if twin != NO_SIBLING && self.seg_owner[twin as usize] == cur {
                        continue;
                    }
                }
                let offset = segment * k;
                out.extend(slots.iter().map(|p| offset + p));
            }
        }
    }

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    pub fn is_terminal(&self) -> bool {
        self.phase == PHASE_TERMINAL
    }

    pub fn current_player(&self) -> u8 {
        self.cur
    }

    pub fn n_players(&self) -> usize {
        self.rules.n_players as usize
    }

    pub fn hand_of(&self, player: usize) -> &[u8] {
        let k = self.board.n_card_types;
        &self.hand[player * k..(player + 1) * k]
    }

    pub fn certain_of(&self, player: usize) -> &[u8] {
        let k = self.board.n_card_types;
        &self.certain[player * k..(player + 1) * k]
    }

    pub fn hand_size(&self, player: usize) -> u32 {
        self.hand_of(player).iter().map(|&c| u32::from(c)).sum()
    }

    pub fn tickets_of(&self, player: usize) -> Vec<u8> {
        let mask = self.tickets[player];
        (0..self.board.n_tickets as u8)
            .filter(|t| mask >> t & 1 != 0)
            .collect()
    }

    fn dsu_mut(&mut self, player: usize) -> &mut [u8] {
        let n = self.board.n_cities;
        &mut self.dsu[player * MAX_CITIES..player * MAX_CITIES + n]
    }

    pub fn ticket_complete(&mut self, player: usize, ticket: usize) -> bool {
        let (a, b) = (self.board.ticket_a[ticket], self.board.ticket_b[ticket]);
        dsu_connected(self.dsu_mut(player), a, b)
    }

    /// The root of `city`'s component in `player`'s network. Path-halving mutates the
    /// forest, which is why this takes `&mut self` -- the DSU is a cache and is excluded
    /// from every hash, so the mutation is invisible to any observer that matters.
    pub fn dsu_root(&mut self, player: usize, city: u8) -> u8 {
        dsu_find(self.dsu_mut(player), city)
    }

    /// Composition of the undrawn deck.
    ///
    /// A *view* for determinization sampling and exact chance probabilities -- **never**
    /// the source of a draw. Drawing from counts instead of the materialized permutation
    /// is what silently destroys paired evaluation.
    pub fn deck_counts(&self) -> Vec<u8> {
        let mut counts = vec![0u8; self.board.n_card_types];
        for &card in &self.deck[self.deck_pos as usize..self.deck_len as usize] {
            counts[card as usize] += 1;
        }
        counts
    }

    /// Cards `observer` cannot account for: the deck plus every opponent's blind draws.
    pub fn unseen_counts(&self, observer: usize) -> Vec<i16> {
        let board = self.board;
        let k = board.n_card_types;
        let mut counts: Vec<i16> = board
            .deck_composition_counts
            .iter()
            .map(|&c| i16::from(c))
            .collect();
        let mine = &self.hand[observer * k..(observer + 1) * k];
        for ((count, &held), &discarded) in counts.iter_mut().zip(mine).zip(&self.discard[..k]) {
            *count -= i16::from(held) + i16::from(discarded);
        }
        for &card in &self.faceup {
            if card != EMPTY_SLOT {
                counts[card as usize] -= 1;
            }
        }
        for p in 0..self.n_players() {
            if p == observer {
                continue;
            }
            let theirs = &self.certain[p * k..(p + 1) * k];
            for (count, &known) in counts.iter_mut().zip(theirs) {
                *count -= i16::from(known);
            }
        }
        counts
    }

    pub fn tickets_remaining(&self) -> u8 {
        self.tdeck_len
    }

    // -----------------------------------------------------------------
    // Serialization and hashing -- docs/CONTRACT.md §3
    // -----------------------------------------------------------------

    /// The canonical byte image. `canonical = true` is the `position_hash` variant.
    ///
    /// Field order, widths and endianness are frozen. The per-player DSU is excluded (a
    /// cache, derivable from `seg_owner`); so are `discard_total`, `flush_capped`,
    /// `board_version` and the history, all of which are derived or transient.
    pub fn serialize(&self, canonical: bool) -> Vec<u8> {
        let board = self.board;
        let n = self.n_players();
        let k = board.n_card_types;
        let deck_len = self.deck_len as usize;

        let mut out = Vec::with_capacity(600);
        out.push(crate::CONTRACT_VERSION);
        out.push(n as u8);
        out.push(self.phase);
        out.push(self.cur);
        out.push(self.draws_left);
        out.push(self.final_left);
        out.push(self.pass_streak);
        out.extend_from_slice(&self.turn.to_le_bytes());
        out.extend_from_slice(&self.seg_owner[..board.n_segments]);
        out.extend_from_slice(&self.hand[..n * k]);
        out.extend_from_slice(&self.trains[..n]);
        for s in &self.score[..n] {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for t in &self.tickets[..n] {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&self.deck_pos.to_le_bytes());
        out.extend_from_slice(&self.deck_len.to_le_bytes());
        if canonical {
            // Cards already drawn are in hands, on the table or in the discard, so two
            // states differing only in *which order* those came out are the same position
            // -- exactly the transposition MCTS wants to merge.
            let cursor = (self.deck_pos as usize).min(deck_len);
            out.resize(out.len() + cursor, 0);
            out.extend_from_slice(&self.deck[cursor..deck_len]);
        } else {
            out.extend_from_slice(&self.deck[..deck_len]);
        }
        out.extend_from_slice(&self.discard[..k]);
        out.extend(self.faceup.iter().map(|&c| c as u8));
        out.push(self.tdeck_head);
        out.push(self.tdeck_len);
        out.extend_from_slice(&self.tdeck[..board.n_tickets]);
        out.extend_from_slice(&self.certain[..n * k]);
        out.extend_from_slice(&self.unknown[..n]);
        out.push(self.offer_len);
        out.extend_from_slice(&self.offer);
        let (rng_state, rng_inc) = if canonical {
            (0, 0)
        } else {
            (self.rng.state, self.rng.inc)
        };
        out.extend_from_slice(&rng_state.to_le_bytes());
        out.extend_from_slice(&rng_inc.to_le_bytes());
        out
    }

    /// The differential-testing key: everything, RNG and undrawn deck included.
    pub fn state_hash(&self) -> u64 {
        hash64(&self.serialize(false))
    }

    /// The MCTS transposition key: RNG and the already-dealt deck prefix zeroed.
    pub fn position_hash(&self) -> u64 {
        hash64(&self.serialize(true))
    }

    // -----------------------------------------------------------------
    // Self-check
    // -----------------------------------------------------------------

    /// Assert every conservation law. Cheap enough to run after every step in tests.
    ///
    /// These are the invariants that would otherwise fail silently: a card that stops
    /// existing, a train that was never spent, information-set bookkeeping that drifts out
    /// of step with the hand it describes.
    pub fn validate(&self) -> Result<(), String> {
        let board = self.board;
        let n = self.n_players();
        let k = board.n_card_types;

        let held: u32 = self.hand[..n * k].iter().map(|&c| u32::from(c)).sum();
        let on_table = self.faceup.iter().filter(|&&c| c != EMPTY_SLOT).count() as u32;
        let in_deck = u32::from(self.deck_len - self.deck_pos);
        let discard_sum: u32 = self.discard[..k].iter().map(|&c| u32::from(c)).sum();
        if u32::from(self.discard_total) != discard_sum {
            return Err("the discard fast-count drifted".into());
        }
        if held + on_table + discard_sum + in_deck != board.deck_size as u32 {
            return Err(format!(
                "cards leaked: {held} held + {on_table} face-up + {discard_sum} discarded \
                 + {in_deck} in deck != {}",
                board.deck_size
            ));
        }

        for p in 0..n {
            let hand: u32 = self.hand_of(p).iter().map(|&c| u32::from(c)).sum();
            let known: u32 = self.certain_of(p).iter().map(|&c| u32::from(c)).sum();
            if known + u32::from(self.unknown[p]) != hand {
                return Err(format!(
                    "seat {p}: certain {known} + unknown {} != hand {hand}",
                    self.unknown[p]
                ));
            }
            for c in 0..k {
                if self.certain[p * k + c] > self.hand[p * k + c] {
                    return Err(format!(
                        "seat {p} is publicly known to hold more {} than it has",
                        board.color_name(c as u8)
                    ));
                }
            }
            let spent: u32 = (0..board.n_segments)
                .filter(|&s| self.seg_owner[s] == p as u8)
                .map(|s| u32::from(board.seg_len[s]))
                .sum();
            if u32::from(self.trains[p]) + spent != u32::from(board.raw.trains_per_player) {
                return Err(format!("seat {p} trains"));
            }
        }

        // Ticket conservation across hands, the deck and any open offer.
        let in_hands: u32 = self.tickets[..n].iter().map(|t| t.count_ones()).sum();
        if in_hands + u32::from(self.tdeck_len) + u32::from(self.offer_len)
            != board.n_tickets as u32
        {
            return Err("tickets leaked".into());
        }
        let ring = self.tdeck[..board.n_tickets]
            .iter()
            .filter(|&&t| t != NO_TICKET)
            .count();
        if ring != self.tdeck_len as usize {
            return Err("ticket ring length disagrees with tdeck_len".into());
        }

        // The flush assertion most engines lack: 3+ locomotives face-up is only legitimate
        // when the guard blocked the flush, or when the cascade hit its cap -- and the cap
        // case is recorded rather than assumed, so this cannot pass vacuously.
        let loco = board.locomotive as i8;
        let locos = self.faceup.iter().filter(|&&c| c == loco).count();
        if locos >= FLUSH_LOCOS {
            let available = self.nonloco_available();
            if available >= FLUSH_LOCOS && !self.flush_capped {
                return Err(format!(
                    "{locos} locomotives face-up with {available} non-locomotives available \
                     and no cascade bail-out: the flush should have fired"
                ));
            }
        }

        if self.cur as usize >= n {
            return Err(format!("current player {} outside 0..{n}", self.cur));
        }
        if self.phase != PHASE_TERMINAL && self.legal_actions().is_empty() {
            return Err("a non-terminal state with no legal actions".into());
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Cloning
    // -----------------------------------------------------------------

    /// Overwrite `dst` in place, so a search arena never allocates.
    ///
    /// This is `Clone::clone_from`, which for the POD arrays is a memcpy and for the
    /// history vector reuses the existing allocation instead of freeing and reallocating.
    /// Named after the API in PLAN.md §5.7 so search code reads the way the plan does.
    pub fn clone_into(&self, dst: &mut State) {
        dst.clone_from(self);
    }

    pub fn phase_name(&self) -> &'static str {
        PHASE_NAMES[self.phase as usize]
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.n_players();
        write!(
            f,
            "<State {} {}P turn={} phase={} cur={} trains={:?} score={:?}>",
            self.board.name,
            n,
            self.turn,
            self.phase_name(),
            self.cur,
            &self.trains[..n],
            &self.score[..n],
        )
    }
}
