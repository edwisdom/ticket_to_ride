//! Deliberately naive legality and bookkeeping references, for differential testing inside
//! the crate.
//!
//! PLAN.md §14.1 step 4. With the Python engine retired there is no second implementation
//! left to disagree with this one, so these are the second implementation: slow, obvious,
//! and derived from the rules rather than from the code they check.
//!
//! * [`naive_legal_actions`] decides legality from RULES.md, PLAN.md §5.2-§5.3 and
//!   CONTRACT.md §2 alone. Claims are a brute-force scan of every `(segment, pay colour,
//!   locomotives used)` candidate -- 100 x 9 x 7 = 6300 on the USA map -- where the engine
//!   tests ~45 affordability buckets. The scan reads the discard pile, the ticket ring and
//!   the offer slots themselves, never the `discard_total`, `tdeck_len` and `offer_len`
//!   counters, and takes action ids from the layout as PLAN.md §5.3 documents it rather
//!   than from [`crate::actions::ActionSpace`].
//! * [`check_derived`] recomputes every cached or incrementally maintained field of a
//!   [`State`] from its primary fields: the union-find against a flood fill of the seat's
//!   own track, the discard count, per-type card conservation, trains and banked route
//!   points against owned track, `board_version` against the claims on the board, the
//!   ticket ring's shape and every ticket's single location, the turn bookkeeping, and the
//!   views (`deck_counts`, `unseen_counts`, ...) built on them.
//! * [`Bookkeeper`] replays a game's public record to recompute what depends on history
//!   rather than on the position: `certain`/`unknown` from the public card events, and
//!   `turn`, `pass_streak`, `final_left` and the end of the game from the completed turns.
//!
//! The sweeps run every step of random games on every map and seat count, under a uniform
//! policy and under a hoarding one that drives games into card famine; rigged positions
//! cover the rules' edges; a proptest covers the space of hands, ownership and pools.
//!
//! Index loops are kept throughout: they read the way the rules are written, which is the
//! point of a reference.
#![allow(clippy::needless_range_loop)]

use crate::board::{Board, FACEUP_SLOTS, GRAY, MAX_CITIES, MAX_ROUTE_LEN, NO_SIBLING};
use crate::config::{
    CLOSED, EMPTY_SLOT, FREE, NO_TICKET, NOT_TRIGGERED, PHASE_DRAW_SECOND, PHASE_INITIAL_TICKETS,
    PHASE_MAIN, PHASE_NAMES, PHASE_TERMINAL, PHASE_TICKET_KEEP,
};
use crate::state::State;

/// RULES.md: "In 2 or 3 player games, only one of the Double-Routes can be used."
const DOUBLES_LOCKED_UP_TO: usize = 3;

/// RULES.md: the end comes when a stock "gets down to only 0, 1 or 2 trains".
const LAST_TRAINS: u8 = 2;

/// RULES.md: "three of the five face-up cards are Locomotives" flushes the display.
const FLUSH_AT: usize = 3;

/// The action layout as PLAN.md §5.3 documents it, restated here rather than read from
/// `ActionSpace`, so an arithmetic slip there shows up as a disagreement instead of being
/// shared.
#[derive(Clone, Copy, Debug)]
struct Layout {
    k: usize,
    draw_base: u16,
    draw_tickets: u16,
    pass: u16,
    n: u16,
}

impl Layout {
    fn of(board: &Board) -> Self {
        let k = board.n_card_types;
        // Claims first, `segment * K + pay`; then 6 draws (five face-up slots and the
        // blind draw), DRAW_TICKETS, 7 keeps (masks 1..=7) and PASS.
        let draw_base = (board.n_segments * k) as u16;
        let draw_tickets = draw_base + 6;
        let pass = draw_tickets + 7 + 1;
        Self {
            k,
            draw_base,
            draw_tickets,
            pass,
            n: pass + 1,
        }
    }

    fn claim(&self, segment: usize, pay: usize) -> u16 {
        (segment * self.k + pay) as u16
    }

    fn blind(&self) -> u16 {
        self.draw_base + 5
    }

    fn keep(&self, mask: u16) -> u16 {
        self.draw_tickets + mask
    }
}

// ---------------------------------------------------------------------------
// Legality, from the rules
// ---------------------------------------------------------------------------

/// A claim the naive scan found legal: its action id and the cards it spends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NaiveClaim {
    pub action: u16,
    pub segment: usize,
    /// The pay slot: a colour, or the locomotive for a pure-locomotive payment.
    pub pay: usize,
    /// Cards of the pay colour spent; zero for a pure-locomotive payment.
    pub colored: u8,
    /// Locomotives spent.
    pub wilds: u8,
}

/// Every legal claim, by brute force over every candidate payment.
pub(crate) fn naive_claims(state: &State) -> Vec<NaiveClaim> {
    let board = state.board;
    let layout = Layout::of(board);
    let mut out = Vec::new();
    // A claim is a whole turn: never the second half of a draw, never during an offer.
    if state.phase != PHASE_MAIN {
        return out;
    }
    for segment in 0..board.n_segments {
        for pay in 0..board.n_card_types {
            for wilds in 0..=MAX_ROUTE_LEN {
                if let Some(colored) = payment_if_legal(state, segment, pay, wilds) {
                    out.push(NaiveClaim {
                        action: layout.claim(segment, pay),
                        segment,
                        pay,
                        colored,
                        wilds: wilds as u8,
                    });
                }
            }
        }
    }
    out
}

/// Decide one candidate -- `length - wilds` cards of `pay` plus `wilds` locomotives for
/// `segment` -- from the rules alone. `Some(coloured cards)` if the seat to move may claim
/// the segment with exactly this payment *and* it is the payment the action id names.
fn payment_if_legal(state: &State, segment: usize, pay: usize, wilds: usize) -> Option<u8> {
    let board = state.board;
    let n = state.n_players();
    let cur = usize::from(state.cur);
    let k = board.n_card_types;
    let loco = usize::from(board.locomotive);
    let hand = &state.hand[cur * k..(cur + 1) * k];
    let length = usize::from(board.seg_len[segment]);

    // "A set of cards equal to the number of spaces": never more locomotives than that.
    if wilds > length {
        return None;
    }
    // An open route: nobody has claimed it...
    if state.seg_owner[segment] != FREE {
        return None;
    }
    // ...and the double-route rule allows it: at 2-3 seats a claimed twin closes it to
    // everyone; above that only to the twin's owner.
    let twin = board.sibling[segment];
    if twin != NO_SIBLING {
        let other = usize::from(state.seg_owner[usize::from(twin)]);
        if n <= DOUBLES_LOCKED_UP_TO && other < n {
            return None;
        }
        if n > DOUBLES_LOCKED_UP_TO && other == cur {
            return None;
        }
    }
    // A train for every space.
    if usize::from(state.trains[cur]) < length {
        return None;
    }
    if pay == loco {
        // Paying purely in locomotives is a single payment, listed once: all `length`.
        return (wilds == length && usize::from(hand[loco]) >= length).then_some(0);
    }
    // One colour: the route's own, or any single colour on a gray route.
    let required = board.seg_color[segment];
    if required != GRAY && usize::from(required) != pay {
        return None;
    }
    let colored = length - wilds;
    // At least one card of the pay colour. With none it is the pure-locomotive payment,
    // which has its own id -- PLAN.md §5.3's `hand[c] >= 1` guard.
    if colored == 0 {
        return None;
    }
    // The cards are actually in the hand.
    if usize::from(hand[pay]) < colored || usize::from(hand[loco]) < wilds {
        return None;
    }
    // Canonical: as many coloured cards as the hand allows, so each id names one payment.
    (colored == usize::from(hand[pay]).min(length)).then_some(colored as u8)
}

/// Whether a blind draw can produce a card: one in the deck, or discards to reshuffle.
fn blind_draw_possible(state: &State) -> bool {
    let k = state.board.n_card_types;
    let in_deck = usize::from(state.deck_len).saturating_sub(usize::from(state.deck_pos));
    let discarded: usize = state.discard[..k].iter().map(|&c| usize::from(c)).sum();
    in_deck + discarded > 0
}

/// Tickets still in the ring, counted slot by slot.
fn tickets_in_deck(state: &State) -> usize {
    state.tdeck[..state.board.n_tickets]
        .iter()
        .filter(|&&t| t != NO_TICKET)
        .count()
}

/// The card draws open as the first card of a turn, or as the second.
fn naive_draws(state: &State, second: bool) -> Vec<u16> {
    let layout = Layout::of(state.board);
    let loco = state.board.locomotive as i8;
    let mut out = Vec::new();
    for slot in 0..FACEUP_SLOTS {
        let card = state.faceup[slot];
        // An empty slot has nothing to take.
        if card == EMPTY_SLOT {
            continue;
        }
        // "If, after having drawn one card the replacement card is a Locomotive, the
        // player cannot take it": a face-up locomotive is the whole draw or nothing.
        if second && card == loco {
            continue;
        }
        out.push(layout.draw_base + slot as u16);
    }
    if blind_draw_possible(state) {
        out.push(layout.blind());
    }
    out
}

/// Keep masks over the open offer: never empty, never a slot holding nothing, and at least
/// the minimum ("at least two" of the opening three, "at least one" later), capped by the
/// offer's size.
fn naive_keeps(state: &State) -> Vec<u16> {
    let raw = state.board.raw;
    let layout = Layout::of(state.board);
    let keep_min = if state.phase == PHASE_INITIAL_TICKETS {
        raw.initial_ticket_keep_min
    } else {
        raw.draw_ticket_keep_min
    };
    let offered = (0..3)
        .filter(|&i| state.offer[i] != NO_TICKET)
        .fold(0u16, |mask, i| mask | 1 << i);
    let need = u32::from(keep_min).min(offered.count_ones());
    (1..=7u16)
        .filter(|&mask| mask & !offered == 0 && mask.count_ones() >= need)
        .map(|mask| layout.keep(mask))
        .collect()
}

/// Every legal action, sorted, decided from the rules.
///
/// Panics if two candidates land on one action id: that would mean ids are not bijective
/// with payments, which is what the canonical rule exists to guarantee.
pub(crate) fn naive_legal_actions(state: &State) -> Vec<u16> {
    let layout = Layout::of(state.board);
    let mut out: Vec<u16> = match state.phase {
        PHASE_MAIN => {
            let mut actions: Vec<u16> = naive_claims(state).iter().map(|c| c.action).collect();
            actions.extend(naive_draws(state, false));
            if tickets_in_deck(state) > 0 {
                actions.push(layout.draw_tickets);
            }
            // PASS exists for exactly the position where nothing else is possible.
            if actions.is_empty() {
                actions.push(layout.pass);
            }
            actions
        }
        PHASE_DRAW_SECOND => naive_draws(state, true),
        PHASE_INITIAL_TICKETS | PHASE_TICKET_KEEP => naive_keeps(state),
        _ => Vec::new(),
    };
    out.sort_unstable();
    let candidates = out.len();
    out.dedup();
    assert_eq!(
        out.len(),
        candidates,
        "{state:?}: two legal payments share an action id"
    );
    out
}

fn describe_difference(state: &State, naive: &[u16], engine: &[u16]) -> String {
    let name = |a: &u16| state.space.to_string(state.board, *a);
    let only_rules: Vec<String> = naive
        .iter()
        .filter(|a| !engine.contains(a))
        .map(name)
        .collect();
    let only_engine: Vec<String> = engine
        .iter()
        .filter(|a| !naive.contains(a))
        .map(name)
        .collect();
    format!(
        "{state:?} in {}: legal by the rules only {only_rules:?}; by the engine only \
         {only_engine:?}",
        PHASE_NAMES[usize::from(state.phase).min(PHASE_NAMES.len() - 1)]
    )
}

/// `legal_actions()` and `legal_action_mask()` against the naive reference.
pub(crate) fn check_legality(state: &State) -> Result<(), String> {
    let naive = naive_legal_actions(state);
    let engine = state.legal_actions();
    if naive != engine {
        return Err(describe_difference(state, &naive, &engine));
    }
    let size = usize::from(Layout::of(state.board).n);
    // A clean buffer and a dirty one: the mask must clear what it does not set.
    for fill in [0u8, 1] {
        let mut mask = vec![fill; size];
        state.legal_action_mask(&mut mask);
        let set: Vec<u16> = (0..size)
            .filter(|&a| mask[a] != 0)
            .map(|a| a as u16)
            .collect();
        if set != naive || mask.iter().any(|&bit| bit > 1) {
            return Err(format!(
                "legal_action_mask over a buffer of {fill}s: {}",
                describe_difference(state, &naive, &set)
            ));
        }
    }
    Ok(())
}

/// The strongest form: `step()` accepts exactly the actions the rules allow, and each
/// accepted action does what the rules say it does.
pub(crate) fn check_step_acceptance(state: &State) -> Result<(), String> {
    let naive = naive_legal_actions(state);
    let claims = naive_claims(state);
    for action in 0..Layout::of(state.board).n {
        let mut probe = state.clone();
        let accepted = probe.step(action).is_ok();
        if accepted != naive.binary_search(&action).is_ok() {
            return Err(format!(
                "{state:?}: step() {} {}, which the rules {}",
                if accepted { "accepted" } else { "refused" },
                state.space.to_string(state.board, action),
                if accepted { "refuse" } else { "allow" },
            ));
        }
        if accepted {
            check_effect(state, &claims, action, &probe)?;
        }
    }
    Ok(())
}

/// What a legal step did to the hands and the board, against what its id names.
///
/// A claim spends exactly the canonical payment its id denotes -- the bijection the naive
/// scan asserts is only worth anything if the engine then spends that payment -- and takes
/// exactly its segment (and, at 2-3 seats, closes the twin). A draw adds one card, the one
/// showing in that slot or the top of the deck; nothing else moves a hand.
pub(crate) fn check_effect(
    before: &State,
    claims: &[NaiveClaim],
    action: u16,
    after: &State,
) -> Result<(), String> {
    let board = before.board;
    let layout = Layout::of(board);
    let (n, k) = (before.n_players(), board.n_card_types);
    let cur = usize::from(before.cur);
    let loco = usize::from(board.locomotive);
    let name = before.space.to_string(board, action);

    for p in (0..n).filter(|&p| p != cur) {
        if before.hand[p * k..(p + 1) * k] != after.hand[p * k..(p + 1) * k] {
            return Err(format!("{before:?}: {name} moved seat {p}'s hand"));
        }
    }
    let gained: Vec<i32> = (0..k)
        .map(|c| i32::from(after.hand[cur * k + c]) - i32::from(before.hand[cur * k + c]))
        .collect();

    if action < layout.draw_base {
        let Some(claim) = claims.iter().find(|c| c.action == action) else {
            return Err(format!(
                "{before:?}: {name} went through but the rules refuse it"
            ));
        };
        let mut spent = vec![0i32; k];
        if claim.pay != loco {
            spent[claim.pay] = -i32::from(claim.colored);
        }
        spent[loco] -= i32::from(claim.wilds);
        if gained != spent {
            return Err(format!(
                "{before:?}: {name} changed the hand by {gained:?}; the id names {spent:?}"
            ));
        }
        let segment = claim.segment;
        let length = board.seg_len[segment];
        if after.seg_owner[segment] != before.cur {
            return Err(format!("{before:?}: {name} did not take the segment"));
        }
        let twin = board.sibling[segment];
        for s in 0..board.n_segments {
            let expect = if s == segment {
                before.cur
            } else if twin != NO_SIBLING && s == usize::from(twin) && n <= DOUBLES_LOCKED_UP_TO {
                CLOSED
            } else {
                before.seg_owner[s]
            };
            if after.seg_owner[s] != expect {
                return Err(format!(
                    "{before:?}: {name} left {} with owner {}, expected {expect}",
                    board.segment_name(s),
                    after.seg_owner[s]
                ));
            }
        }
        let trains = i32::from(before.trains[cur]) - i32::from(after.trains[cur]);
        let points = i32::from(after.score[cur]) - i32::from(before.score[cur]);
        let worth = i32::from(board.raw.route_points[usize::from(length)]);
        if trains != i32::from(length) || points != worth {
            return Err(format!(
                "{before:?}: {name} cost {trains} trains and scored {points}; a route of \
                 {length} costs {length} and scores {worth}"
            ));
        }
    } else if action < layout.blind() {
        let card = before.faceup[usize::from(action - layout.draw_base)];
        let mut one = vec![0i32; k];
        one[card as usize] = 1;
        if gained != one {
            return Err(format!(
                "{before:?}: {name} changed the hand by {gained:?}, not by the card showing"
            ));
        }
    } else if action == layout.blind() {
        if gained.iter().any(|&g| g < 0) || gained.iter().sum::<i32>() != 1 {
            return Err(format!("{before:?}: {name} changed the hand by {gained:?}"));
        }
        if before.deck_pos < before.deck_len {
            let top = usize::from(before.deck[usize::from(before.deck_pos)]);
            if gained[top] != 1 {
                return Err(format!("{before:?}: {name} did not take the top card"));
            }
        }
    } else if gained.iter().any(|&g| g != 0) {
        return Err(format!("{before:?}: {name} changed the hand by {gained:?}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cached and incremental fields, recomputed from the position
// ---------------------------------------------------------------------------

/// Every cached or incrementally maintained field, recomputed from the primary ones.
pub(crate) fn check_derived(state: &State) -> Result<(), String> {
    check_cards(state)?;
    check_information_sets(state)?;
    check_track(state)?;
    check_connectivity(state)?;
    check_tickets(state)?;
    check_turn_structure(state)?;
    check_display(state)?;
    check_views(state)?;
    Ok(())
}

/// The deck cursor, the discard fast-count, and conservation **per card type** --
/// `validate()` counts cards only in total, which a card changing colour on its way through
/// the engine would survive.
fn check_cards(state: &State) -> Result<(), String> {
    let board = state.board;
    let raw = board.raw;
    let (n, k) = (state.n_players(), board.n_card_types);
    let loco = usize::from(board.locomotive);
    let (pos, len) = (usize::from(state.deck_pos), usize::from(state.deck_len));
    if pos > len || len > board.deck_size {
        return Err(format!("{state:?}: deck cursor {pos} of {len}"));
    }
    let discarded: usize = state.discard[..k].iter().map(|&c| usize::from(c)).sum();
    if usize::from(state.discard_total) != discarded {
        return Err(format!(
            "{state:?}: discard_total {} but the pile holds {discarded}",
            state.discard_total
        ));
    }
    for &card in &state.faceup {
        if card != EMPTY_SLOT && !(0..k as i8).contains(&card) {
            return Err(format!("{state:?}: face-up card {card}"));
        }
    }
    for c in 0..k {
        let held: usize = (0..n).map(|p| usize::from(state.hand[p * k + c])).sum();
        let shown = state.faceup.iter().filter(|&&f| f == c as i8).count();
        let in_deck = state.deck[pos..len]
            .iter()
            .filter(|&&d| usize::from(d) == c)
            .count();
        let total = held + shown + usize::from(state.discard[c]) + in_deck;
        let printed = if c == loco {
            raw.locomotives
        } else {
            raw.cards_per_color
        };
        if total != usize::from(printed) {
            return Err(format!(
                "{state:?}: {} {} in play, {printed} printed ({held} held, {shown} face-up, \
                 {} discarded, {in_deck} in the deck)",
                total,
                board.color_name(c as u8),
                state.discard[c]
            ));
        }
    }
    Ok(())
}

/// Public knowledge of a hand never exceeds the hand, and accounts for all of it.
fn check_information_sets(state: &State) -> Result<(), String> {
    let k = state.board.n_card_types;
    for p in 0..state.n_players() {
        let hand = &state.hand[p * k..(p + 1) * k];
        let certain = &state.certain[p * k..(p + 1) * k];
        let size: usize = hand.iter().map(|&c| usize::from(c)).sum();
        let known: usize = certain.iter().map(|&c| usize::from(c)).sum();
        if known + usize::from(state.unknown[p]) != size || (0..k).any(|c| certain[c] > hand[c]) {
            return Err(format!(
                "{state:?}: seat {p} holds {hand:?}, publicly known {certain:?} plus {} unseen",
                state.unknown[p]
            ));
        }
    }
    Ok(())
}

/// Trains, banked route points and `board_version` against the track on the board, and
/// the double-route sentinels against the claims that set them.
fn check_track(state: &State) -> Result<(), String> {
    let board = state.board;
    let raw = board.raw;
    let n = state.n_players();
    let owners = &state.seg_owner[..board.n_segments];
    for (s, &owner) in owners.iter().enumerate() {
        if owner != FREE && owner != CLOSED && usize::from(owner) >= n {
            return Err(format!(
                "{state:?}: {} has owner {owner}",
                board.segment_name(s)
            ));
        }
    }
    for p in 0..n {
        let mine: Vec<usize> = (0..board.n_segments)
            .filter(|&s| usize::from(owners[s]) == p)
            .collect();
        let spent: usize = mine.iter().map(|&s| usize::from(board.seg_len[s])).sum();
        let points: i32 = mine
            .iter()
            .map(|&s| i32::from(raw.route_points[usize::from(board.seg_len[s])]))
            .sum();
        if usize::from(state.trains[p]) + spent != usize::from(raw.trains_per_player) {
            return Err(format!(
                "{state:?}: seat {p} has {} trains with {spent} on the board",
                state.trains[p]
            ));
        }
        if i32::from(state.score[p]) != points {
            return Err(format!(
                "{state:?}: seat {p} banked {} route points; its track is worth {points}",
                state.score[p]
            ));
        }
    }
    let claimed = owners.iter().filter(|&&o| usize::from(o) < n).count();
    if state.board_version as usize != claimed {
        return Err(format!(
            "{state:?}: board_version {} with {claimed} claims on the board",
            state.board_version
        ));
    }
    for s in 0..board.n_segments {
        let twin = board.sibling[s];
        let twin_owner = (twin != NO_SIBLING).then(|| owners[usize::from(twin)]);
        let twin_claimed = twin_owner.is_some_and(|o| usize::from(o) < n);
        let broken = if n <= DOUBLES_LOCKED_UP_TO {
            // Closed to everyone exactly when the twin is someone's.
            (owners[s] == CLOSED) != twin_claimed
        } else {
            owners[s] == CLOSED || (usize::from(owners[s]) < n && twin_owner == Some(owners[s]))
        };
        if broken {
            return Err(format!(
                "{state:?}: {} is {} with its twin {:?}",
                board.segment_name(s),
                owners[s],
                twin_owner
            ));
        }
    }
    Ok(())
}

/// The per-seat union-find against a flood fill of that seat's own track.
///
/// Every city pair, not just ticket endpoints, and through the public queries as well as
/// the raw forest. Parent pointers are walked read-only and with a bound, so a corrupted
/// forest fails here instead of hanging in `dsu_find`.
fn check_connectivity(state: &State) -> Result<(), String> {
    let board = state.board;
    let cities = board.n_cities;
    let city = |c: usize| board.raw.cities[c];
    // `dsu_root` and `ticket_complete` path-halve the forest, so they query a copy.
    let mut probe = state.clone();
    for p in 0..state.n_players() {
        let mut component = vec![usize::MAX; cities];
        for start in 0..cities {
            if component[start] != usize::MAX {
                continue;
            }
            component[start] = start;
            let mut stack = vec![start];
            while let Some(u) = stack.pop() {
                for s in 0..board.n_segments {
                    if usize::from(state.seg_owner[s]) != p {
                        continue;
                    }
                    let (a, b) = (usize::from(board.seg_a[s]), usize::from(board.seg_b[s]));
                    let v = if a == u {
                        b
                    } else if b == u {
                        a
                    } else {
                        continue;
                    };
                    if component[v] == usize::MAX {
                        component[v] = start;
                        stack.push(v);
                    }
                }
            }
        }

        let parent = &state.dsu[p * MAX_CITIES..p * MAX_CITIES + cities];
        let mut root = vec![0usize; cities];
        for c in 0..cities {
            let mut x = c;
            let mut steps = 0;
            loop {
                let up = usize::from(parent[x]);
                if up == x {
                    break;
                }
                if up >= cities || steps == cities {
                    return Err(format!(
                        "{state:?}: seat {p}'s union-find has a cycle or a stray pointer at {}",
                        city(c)
                    ));
                }
                x = up;
                steps += 1;
            }
            root[c] = x;
        }

        for a in 0..cities {
            for b in a + 1..cities {
                let by_track = component[a] == component[b];
                if by_track != (root[a] == root[b]) {
                    return Err(format!(
                        "{state:?}: seat {p}: {} and {} are {} by its track, but not by its \
                         union-find",
                        city(a),
                        city(b),
                        if by_track { "joined" } else { "apart" }
                    ));
                }
            }
            if usize::from(probe.dsu_root(p, a as u8)) != root[a] {
                return Err(format!("{state:?}: seat {p}: dsu_root({}) moved", city(a)));
            }
        }
        for t in 0..board.n_tickets {
            let (a, b) = (
                usize::from(board.ticket_a[t]),
                usize::from(board.ticket_b[t]),
            );
            if probe.ticket_complete(p, t) != (component[a] == component[b]) {
                return Err(format!(
                    "{state:?}: seat {p}: ticket_complete({}) disagrees with its track",
                    board.ticket_name(t)
                ));
            }
        }
    }
    Ok(())
}

/// The ticket ring is one contiguous run from its head with every other slot blank, the
/// offer is a prefix of its slots and open exactly while a seat is choosing, and every
/// ticket is in exactly one place. `validate()` counts tickets; this finds each one.
fn check_tickets(state: &State) -> Result<(), String> {
    let board = state.board;
    let n = state.n_players();
    let capacity = board.n_tickets;
    let (head, len) = (usize::from(state.tdeck_head), usize::from(state.tdeck_len));
    if head >= capacity || len > capacity {
        return Err(format!("{state:?}: ticket ring head {head}, length {len}"));
    }
    for i in 0..capacity {
        let slot = (head + i) % capacity;
        if (state.tdeck[slot] != NO_TICKET) != (i < len) {
            return Err(format!(
                "{state:?}: ticket ring slot {slot} is {} with head {head} and length {len}",
                state.tdeck[slot]
            ));
        }
    }

    let offer_len = usize::from(state.offer_len);
    if offer_len > 3 || (0..3).any(|i| (state.offer[i] != NO_TICKET) != (i < offer_len)) {
        return Err(format!(
            "{state:?}: offer {:?} of length {offer_len}",
            state.offer
        ));
    }
    let choosing = matches!(state.phase, PHASE_INITIAL_TICKETS | PHASE_TICKET_KEEP);
    if choosing != (offer_len > 0) {
        return Err(format!("{state:?}: an offer of {offer_len} in this phase"));
    }

    let mut places: Vec<usize> = Vec::new();
    for p in 0..n {
        places.extend((0..32).filter(|&t| state.tickets[p] >> t & 1 == 1));
    }
    places.extend(
        state.tdeck[..capacity]
            .iter()
            .filter(|&&t| t != NO_TICKET)
            .map(|&t| usize::from(t)),
    );
    places.extend(state.offer[..offer_len].iter().map(|&t| usize::from(t)));
    let mut seen = vec![0usize; capacity];
    for &t in &places {
        if t >= capacity {
            return Err(format!("{state:?}: ticket {t} does not exist"));
        }
        seen[t] += 1;
    }
    if let Some(t) = (0..capacity).find(|&t| seen[t] != 1) {
        return Err(format!(
            "{state:?}: ticket {} is in {} places",
            board.ticket_name(t),
            seen[t]
        ));
    }
    Ok(())
}

/// Turn bookkeeping that is a pure function of the position.
fn check_turn_structure(state: &State) -> Result<(), String> {
    let n = state.n_players();
    let (cur, turn) = (usize::from(state.cur), usize::from(state.turn));
    if cur >= n {
        return Err(format!("{state:?}: seat {cur} to move"));
    }
    if state.draws_left != u8::from(state.phase == PHASE_DRAW_SECOND) {
        return Err(format!("{state:?}: draws_left {}", state.draws_left));
    }
    // Seats move in order from seat 0 on turn 0; the turn that ends the game does not
    // pass the move on.
    let seat_ok = match state.phase {
        PHASE_INITIAL_TICKETS => turn == 0,
        PHASE_MAIN | PHASE_DRAW_SECOND | PHASE_TICKET_KEEP => cur == turn % n,
        PHASE_TERMINAL => turn > 0 && cur == (turn - 1) % n,
        _ => false,
    };
    if !seat_ok {
        return Err(format!("{state:?}: seat {cur} on turn {turn}"));
    }
    // Every seat passing in a row freezes the game.
    if state.phase != PHASE_TERMINAL && usize::from(state.pass_streak) >= n {
        return Err(format!(
            "{state:?}: pass streak {} still live",
            state.pass_streak
        ));
    }
    // Trains only fall on a seat's own claim, which ends its turn, and the trigger is
    // checked at the end of every turn: so the end is triggered exactly when some seat is
    // down to its last two trains.
    let low = state.trains[..n].iter().any(|&t| t <= LAST_TRAINS);
    let triggered = state.final_left != NOT_TRIGGERED;
    if low != triggered {
        return Err(format!(
            "{state:?}: final_left {} with trains {:?}",
            state.final_left,
            &state.trains[..n]
        ));
    }
    if triggered
        && (usize::from(state.final_left) > n
            || (state.final_left == 0 && state.phase != PHASE_TERMINAL))
    {
        return Err(format!("{state:?}: final_left {}", state.final_left));
    }
    Ok(())
}

/// The flush flag, the flush rule itself, and the end-of-turn refill.
fn check_display(state: &State) -> Result<(), String> {
    let board = state.board;
    let k = board.n_card_types;
    let loco = board.locomotive;
    let locos = state.faceup.iter().filter(|&&c| c == loco as i8).count();
    if state.flush_capped && locos < FLUSH_AT {
        return Err(format!(
            "{state:?}: flush_capped with only {locos} locomotives showing"
        ));
    }
    // Three locomotives may stay face-up only when the guard blocked the flush (fewer than
    // three non-locomotives left to deal) or the cascade stopped on its cap.
    if locos >= FLUSH_AT && !state.flush_capped {
        let (pos, len) = (usize::from(state.deck_pos), usize::from(state.deck_len));
        let in_deck = state.deck[pos..len].iter().filter(|&&d| d != loco).count();
        let in_discard: usize = (0..k)
            .filter(|&c| c != usize::from(loco))
            .map(|c| usize::from(state.discard[c]))
            .sum();
        if in_deck + in_discard >= FLUSH_AT {
            return Err(format!(
                "{state:?}: {locos} locomotives showing with {} non-locomotives available",
                in_deck + in_discard
            ));
        }
    }
    // A short display in a live game means every pool is dry: otherwise the next refill,
    // at the latest the one at the end of the turn, would have filled it (GOTCHAS: "The
    // face-up display can empty permanently"). Only the game's last turn skips it.
    if state.phase != PHASE_TERMINAL
        && state.faceup.contains(&EMPTY_SLOT)
        && blind_draw_possible(state)
    {
        return Err(format!(
            "{state:?}: display {:?} is short with cards left to deal",
            state.faceup
        ));
    }
    Ok(())
}

/// The derived views, against the fields they summarize.
fn check_views(state: &State) -> Result<(), String> {
    let board = state.board;
    let (n, k) = (state.n_players(), board.n_card_types);
    let mut deck = vec![0u8; k];
    for &card in &state.deck[usize::from(state.deck_pos)..usize::from(state.deck_len)] {
        deck[usize::from(card)] += 1;
    }
    if state.deck_counts() != deck {
        return Err(format!("{state:?}: deck_counts {:?}", state.deck_counts()));
    }
    for o in 0..n {
        // Unseen by `o`: the deck plus whatever of each opponent's hand is not public.
        let unseen: Vec<i16> = (0..k)
            .map(|c| {
                let hidden: i16 = (0..n)
                    .filter(|&q| q != o)
                    .map(|q| i16::from(state.hand[q * k + c]) - i16::from(state.certain[q * k + c]))
                    .sum();
                i16::from(deck[c]) + hidden
            })
            .collect();
        if state.unseen_counts(o) != unseen {
            return Err(format!(
                "{state:?}: unseen_counts({o}) {:?}, recomputed {unseen:?}",
                state.unseen_counts(o)
            ));
        }
        let size: u32 = state.hand[o * k..(o + 1) * k]
            .iter()
            .map(|&c| u32::from(c))
            .sum();
        let held: Vec<u8> = (0..board.n_tickets as u8)
            .filter(|&t| state.tickets[o] >> t & 1 == 1)
            .collect();
        if state.hand_size(o) != size || state.tickets_of(o) != held {
            return Err(format!("{state:?}: seat {o}'s hand_size or tickets_of"));
        }
    }
    if usize::from(state.tickets_remaining()) != tickets_in_deck(state) {
        return Err(format!(
            "{state:?}: tickets_remaining {}",
            state.tickets_remaining()
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// History-dependent fields, recomputed from the public record
// ---------------------------------------------------------------------------

/// Replays a game's public record and recomputes the fields `State` maintains as it goes.
///
/// `certain`/`unknown` follow PLAN.md §5.5's table, driven only by what the table says is
/// public: face-up takes, the arrival of blind draws, and the cards a claim discards (read
/// off the hand, so the payment rule is not assumed). The turn counter, the pass streak and
/// the end-of-game countdown follow from which steps completed a turn; the countdown is
/// recomputed in closed form from the turn that triggered it, so "exactly one more turn
/// each" is checked rather than restated.
pub(crate) struct Bookkeeper {
    certain: Vec<u8>,
    unknown: Vec<u8>,
    turns: u16,
    pass_streak: u8,
    /// The turn count right after the turn that triggered the end, once one has.
    triggered_at: Option<u16>,
    final_left: u8,
    claims: u32,
    /// Every action that went through, in order: what `history` must hold.
    actions: Vec<u16>,
}

impl Bookkeeper {
    /// From a freshly dealt game: nothing seen yet, every opening card dealt blind.
    pub(crate) fn fresh(state: &State) -> Result<Self, String> {
        let (n, k) = (state.n_players(), state.board.n_card_types);
        let books = Self {
            certain: vec![0; n * k],
            unknown: vec![state.rules.initial_hand; n],
            turns: 0,
            pass_streak: 0,
            triggered_at: None,
            final_left: NOT_TRIGGERED,
            claims: 0,
            actions: Vec::new(),
        };
        books.compare(state)?;
        Ok(books)
    }

    /// Adopt a rigged position as it stands, for branches random play rarely reaches.
    pub(crate) fn adopt(state: &State) -> Self {
        let (n, k) = (state.n_players(), state.board.n_card_types);
        let triggered_at = (state.final_left != NOT_TRIGGERED)
            .then(|| state.turn - (n as u16 - u16::from(state.final_left)));
        Self {
            certain: state.certain[..n * k].to_vec(),
            unknown: state.unknown[..n].to_vec(),
            turns: state.turn,
            pass_streak: state.pass_streak,
            triggered_at,
            final_left: state.final_left,
            claims: state.board_version,
            actions: state.history.clone(),
        }
    }

    /// Account for one step, then compare with what the engine now holds.
    pub(crate) fn observe(
        &mut self,
        before: &State,
        action: u16,
        after: &State,
    ) -> Result<(), String> {
        let board = before.board;
        let layout = Layout::of(board);
        let (n, k) = (before.n_players(), board.n_card_types);
        let cur = usize::from(before.cur);
        self.actions.push(action);

        if action < layout.draw_base {
            // A claim's cards are discarded face-up: what left the hand is public.
            for c in 0..k {
                let (was, now) = (before.hand[cur * k + c], after.hand[cur * k + c]);
                let Some(spent) = was.checked_sub(now) else {
                    return Err(format!("{before:?}: a claim added a card to the hand"));
                };
                for _ in 0..spent {
                    if self.certain[cur * k + c] > 0 {
                        self.certain[cur * k + c] -= 1;
                    } else if self.unknown[cur] > 0 {
                        self.unknown[cur] -= 1;
                    } else {
                        return Err(format!("{before:?}: seat {cur} spent a card it never had"));
                    }
                }
            }
            self.claims += 1;
        } else if action < layout.blind() {
            // Taking a face-up card shows everyone which card it was.
            let card = before.faceup[usize::from(action - layout.draw_base)];
            let Ok(card) = usize::try_from(card) else {
                return Err(format!("{before:?}: took an empty slot"));
            };
            self.certain[cur * k + card] += 1;
        } else if action == layout.blind() {
            // A blind draw shows only that a card arrived.
            self.unknown[cur] += 1;
        }

        // A turn is over when play is back in MAIN or the game has ended. The opening
        // ticket choices come before turn 0 and complete no turn.
        let turn_over = before.phase != PHASE_INITIAL_TICKETS
            && matches!(after.phase, PHASE_MAIN | PHASE_TERMINAL);
        if turn_over {
            self.turns += 1;
            self.pass_streak = if action == layout.pass {
                self.pass_streak + 1
            } else {
                0
            };
            // Every seat passing in a row freezes the position: it ends there.
            let mut over = usize::from(self.pass_streak) >= n;
            if !over {
                // "Gets down to only 0, 1 or 2 trains left at the end of his turn, each
                // player, including that player, gets one final turn."
                if self.triggered_at.is_none() && after.trains[cur] <= LAST_TRAINS {
                    self.triggered_at = Some(self.turns);
                }
                if let Some(at) = self.triggered_at {
                    let since = usize::from(self.turns - at);
                    self.final_left = (n - since) as u8;
                    over = since == n;
                }
                over |= self.turns >= before.rules.turn_cap;
            }
            if after.is_terminal() != over {
                return Err(format!(
                    "{after:?}: after turn {} the game should {}be over",
                    self.turns,
                    if over { "" } else { "not " }
                ));
            }
        } else if after.is_terminal() {
            return Err(format!("{after:?}: the game ended mid-turn"));
        }

        let bumped = after.board_version.wrapping_sub(before.board_version);
        if bumped != u32::from(action < layout.draw_base) {
            return Err(format!(
                "{after:?}: board_version moved by {bumped} on {}",
                before.space.to_string(board, action)
            ));
        }
        self.compare(after)
    }

    fn compare(&self, state: &State) -> Result<(), String> {
        let (n, k) = (state.n_players(), state.board.n_card_types);
        if state.certain[..n * k] != self.certain[..] || state.unknown[..n] != self.unknown[..] {
            return Err(format!(
                "{state:?}: certain {:?} / unknown {:?}; the public record says {:?} / {:?}",
                &state.certain[..n * k],
                &state.unknown[..n],
                self.certain,
                self.unknown
            ));
        }
        let counters = (
            state.turn,
            state.pass_streak,
            state.final_left,
            state.board_version,
        );
        let recomputed = (self.turns, self.pass_streak, self.final_left, self.claims);
        if counters != recomputed {
            return Err(format!(
                "{state:?}: (turn, pass_streak, final_left, board_version) = {counters:?}; \
                 the public record says {recomputed:?}"
            ));
        }
        if state.rules.track_history && state.history != self.actions {
            return Err(format!(
                "{state:?}: history has {} entries ending {:?} after {} steps ending {:?}",
                state.history.len(),
                state.history.last(),
                self.actions.len(),
                self.actions.last()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::*;
    use crate::board::{MAX_CARD_TYPES, MAX_SEGMENTS, MAX_TICKETS, all_boards, get_board};
    use crate::config::RuleConfig;
    use crate::rng::{Part, Pcg32, stream};
    use crate::state::Game;

    fn configs() -> Vec<(&'static str, usize)> {
        all_boards()
            .iter()
            .flat_map(|b| {
                (usize::from(b.raw.min_players)..=usize::from(b.raw.max_players))
                    .map(move |n| (b.name, n))
            })
            .collect()
    }

    fn game(map: &str, n: usize) -> Game {
        Game::new(RuleConfig::new(map, n).expect("a supported configuration"))
            .expect("a supported configuration")
    }

    // -----------------------------------------------------------------------
    // Sweeps
    // -----------------------------------------------------------------------

    #[derive(Clone, Copy, Debug)]
    enum Policy {
        /// Every legal action equally likely.
        Uniform,
        /// Claims one time in eight while anything else is possible, so hands swell, the
        /// deck and the discard run dry, the ticket deck empties and seats run out of
        /// moves -- the positions uniform play almost never reaches.
        Hoarder,
    }

    impl Policy {
        fn name(self) -> &'static str {
            match self {
                Policy::Uniform => "uniform",
                Policy::Hoarder => "hoarder",
            }
        }

        fn pick(self, state: &State, rng: &mut Pcg32) -> u16 {
            let legal = state.legal_actions();
            let any = |rng: &mut Pcg32, from: &[u16]| from[rng.below(from.len() as u32) as usize];
            match self {
                Policy::Uniform => any(rng, &legal),
                Policy::Hoarder => {
                    let (claims, rest): (Vec<u16>, Vec<u16>) = legal
                        .iter()
                        .copied()
                        .partition(|&a| state.space.is_claim(a));
                    if rest.is_empty() || (!claims.is_empty() && rng.below(8) == 0) {
                        any(rng, &claims)
                    } else {
                        any(rng, &rest)
                    }
                }
            }
        }
    }

    /// Which checks a sweep runs at every position.
    #[derive(Clone, Copy)]
    struct Checks {
        legality: bool,
        bookkeeping: bool,
        /// Run the full `step()` acceptance scan at every `accept_every`-th position; 0
        /// never.
        accept_every: usize,
    }

    /// What a sweep actually visited, so a test can insist the hard positions came up.
    #[derive(Debug, Default)]
    struct Reach {
        positions: usize,
        /// MAIN positions with neither a deck nor a discard to draw from.
        famine: usize,
        /// MAIN positions with an empty face-up slot.
        short_display: usize,
        /// MAIN positions with no ticket left to draw.
        no_tickets: usize,
        /// Second-draw positions with a locomotive showing.
        loco_on_second: usize,
        /// Claims taking one track of a double route.
        doubles: usize,
        passes: usize,
        /// Games frozen by every seat passing in a row.
        freezes: usize,
    }

    impl Reach {
        fn note(&mut self, state: &State) {
            self.positions += 1;
            let loco = state.board.locomotive as i8;
            if state.phase == PHASE_MAIN {
                self.famine += usize::from(!blind_draw_possible(state));
                self.short_display += usize::from(state.faceup.contains(&EMPTY_SLOT));
                self.no_tickets += usize::from(tickets_in_deck(state) == 0);
            }
            if state.phase == PHASE_DRAW_SECOND && state.faceup.contains(&loco) {
                self.loco_on_second += 1;
            }
        }
    }

    fn play(map: &str, n: usize, seed: u64, policy: Policy, checks: Checks, reach: &mut Reach) {
        let mut state = game(map, n).new_initial_state(seed);
        let mut rng = stream(seed, &[Part::Str("naive"), Part::Str(policy.name())]);
        let at = format!("{map} {n}P {} seed {seed}", policy.name());
        let mut books = checks
            .bookkeeping
            .then(|| Bookkeeper::fresh(&state).unwrap_or_else(|e| panic!("{at}: {e}")));
        let mut position = 0;
        loop {
            reach.note(&state);
            if checks.legality {
                check_legality(&state).unwrap_or_else(|e| panic!("{at}: {e}"));
            }
            if checks.bookkeeping {
                check_derived(&state).unwrap_or_else(|e| panic!("{at}: {e}"));
            }
            if checks.accept_every > 0 && position % checks.accept_every == 0 {
                check_step_acceptance(&state).unwrap_or_else(|e| panic!("{at}: {e}"));
            }
            position += 1;
            if state.is_terminal() {
                break;
            }
            let action = policy.pick(&state, &mut rng);
            let before = state.clone();
            state
                .step(action)
                .unwrap_or_else(|e| panic!("{at}: a legal action was refused: {e}"));
            if action == state.space.pass_action {
                reach.passes += 1;
            }
            if state.space.is_claim(action) {
                let segment = usize::from(state.space.decode_claim(action).0);
                reach.doubles += usize::from(state.board.sibling[segment] != NO_SIBLING);
            }
            if let Some(books) = books.as_mut() {
                let claims = naive_claims(&before);
                let checked = check_effect(&before, &claims, action, &state)
                    .and_then(|()| books.observe(&before, action, &state));
                checked.unwrap_or_else(|e| panic!("{at}: {e}"));
            }
        }
        if usize::from(state.pass_streak) >= n {
            reach.freezes += 1;
        }
    }

    fn sweep(uniform: u64, hoarder: u64, checks: Checks) -> Reach {
        let mut reach = Reach::default();
        for (map, n) in configs() {
            for seed in 0..uniform {
                play(map, n, seed, Policy::Uniform, checks, &mut reach);
            }
            for seed in 0..hoarder {
                play(map, n, seed, Policy::Hoarder, checks, &mut reach);
            }
        }
        reach
    }

    /// A guard on the guard: the comparison is only as good as the positions it saw.
    fn assert_reached_the_edges(reach: &Reach) {
        assert!(
            reach.famine > 0,
            "no position with the deck and discard both dry: {reach:?}"
        );
        assert!(
            reach.short_display > 0,
            "no position with a short display: {reach:?}"
        );
        assert!(
            reach.no_tickets > 0,
            "no position with the ticket deck empty: {reach:?}"
        );
        assert!(
            reach.loco_on_second > 0,
            "no second draw with a locomotive showing: {reach:?}"
        );
        assert!(
            reach.doubles > 0,
            "no double route was ever claimed: {reach:?}"
        );
        assert!(reach.passes > 0, "nobody ever had to pass: {reach:?}");
    }

    /// `legal_actions()` and the mask, against the 6300-way scan, at every step of every
    /// configuration under both policies.
    #[test]
    fn legal_actions_match_the_naive_scan_at_every_step() {
        let checks = Checks {
            legality: true,
            bookkeeping: false,
            accept_every: 0,
        };
        let reach = sweep(6, 4, checks);
        assert_reached_the_edges(&reach);
    }

    /// Every cache and counter in `State`, recomputed from scratch at every step: from the
    /// position by `check_derived`, from the public record by `Bookkeeper`.
    #[test]
    fn incremental_state_matches_recomputation_at_every_step() {
        let checks = Checks {
            legality: false,
            bookkeeping: true,
            accept_every: 0,
        };
        let reach = sweep(5, 3, checks);
        assert_reached_the_edges(&reach);
    }

    /// `step()` itself, not just the list: every one of the action ids is tried at sampled
    /// positions, and each accepted one is checked for doing what its id says.
    #[test]
    fn step_accepts_exactly_what_the_rules_allow() {
        let checks = Checks {
            legality: false,
            bookkeeping: false,
            accept_every: 4,
        };
        sweep(2, 2, checks);
    }

    #[test]
    #[ignore = "a long sweep of every naive reference; run with --release -- --ignored"]
    fn naive_references_hold_over_a_long_sweep() {
        let checks = Checks {
            legality: true,
            bookkeeping: true,
            accept_every: 1,
        };
        let reach = sweep(100, 100, checks);
        assert_reached_the_edges(&reach);
        assert!(
            reach.freezes > 0,
            "no game froze on an all-pass round: {reach:?}"
        );
        eprintln!("{reach:?}");
    }

    // -----------------------------------------------------------------------
    // Rigged positions on the rules' edges
    // -----------------------------------------------------------------------

    /// The first MAIN position of seed 0, every opening keep the lowest legal one.
    fn fresh(map: &str, n: usize) -> State {
        let mut state = game(map, n).new_initial_state(0);
        while state.phase != PHASE_MAIN {
            let keep = state.legal_actions()[0];
            state.step(keep).expect("the lowest keep is legal");
        }
        state
    }

    fn segment_named(board: &Board, a: &str, b: &str, which: usize) -> usize {
        let city = |name: &str| board.raw.cities.iter().position(|&c| c == name).unwrap() as u8;
        let (a, b) = (city(a).min(city(b)), city(a).max(city(b)));
        (0..board.n_segments)
            .filter(|&s| (board.seg_a[s], board.seg_b[s]) == (a, b))
            .nth(which)
            .expect("such a segment")
    }

    /// Replace a seat's hand, keeping the information set consistent with it (all of it
    /// unseen) so a claim can still be stepped.
    fn set_hand(state: &mut State, seat: usize, cards: &[(usize, u8)]) {
        let k = state.board.n_card_types;
        state.hand[seat * k..(seat + 1) * k].fill(0);
        state.certain[seat * k..(seat + 1) * k].fill(0);
        for &(card, count) in cards {
            state.hand[seat * k + card] = count;
        }
        state.unknown[seat] = cards.iter().map(|&(_, count)| count).sum();
    }

    /// Nothing left to draw blind: the deck spent and the discard empty.
    fn dry_pools(state: &mut State) {
        state.deck_pos = state.deck_len;
        state.discard = [0; MAX_CARD_TYPES];
        state.discard_total = 0;
    }

    fn no_tickets(state: &mut State) {
        state.tdeck = [NO_TICKET; MAX_TICKETS];
        state.tdeck_len = 0;
    }

    /// The naive and engine sets, asserted equal; returns the shared set.
    fn agree(state: &State) -> Vec<u16> {
        check_legality(state).unwrap_or_else(|e| panic!("{e}"));
        naive_legal_actions(state)
    }

    fn claims_in(legal: &[u16], layout: Layout) -> Vec<u16> {
        legal
            .iter()
            .copied()
            .filter(|&a| a < layout.draw_base)
            .collect()
    }

    fn draw_slots(legal: &[u16], layout: Layout) -> Vec<u16> {
        legal
            .iter()
            .filter(|&&a| (layout.draw_base..layout.draw_tickets).contains(&a))
            .map(|&a| a - layout.draw_base)
            .collect()
    }

    /// Without the `hand[c] >= 1` guard, a hand of pure locomotives makes every gray pay
    /// slot legal *and identical*, which poisons the policy distribution. With it, the
    /// locomotive slot is the only one, for every route the hand can cover.
    #[test]
    fn a_pure_locomotive_hand_pays_only_through_the_locomotive_slot() {
        for (map, n) in [("usa", 2), ("usa", 4), ("mini", 3)] {
            for locos in 0..=6u8 {
                let mut state = fresh(map, n);
                let board = state.board;
                let layout = Layout::of(board);
                let loco = usize::from(board.locomotive);
                let cur = usize::from(state.cur);
                set_hand(&mut state, cur, &[(loco, locos)]);
                let expected: Vec<u16> = (0..board.n_segments)
                    .filter(|&s| board.seg_len[s] <= locos)
                    .map(|s| layout.claim(s, loco))
                    .collect();
                assert_eq!(
                    claims_in(&agree(&state), layout),
                    expected,
                    "{map} {n}P holding {locos} locomotives"
                );
            }
        }
    }

    /// Exactly the route's length in cards, one colour plus locomotives: one card short is
    /// refused and exact change accepted, for every route on both maps and every colour it
    /// takes, through the colour's slot and the locomotive slot alike.
    #[test]
    fn one_card_short_is_refused_and_exact_change_is_accepted() {
        for map in ["usa", "mini"] {
            let base = fresh(map, 2);
            let board = base.board;
            let layout = Layout::of(board);
            let loco = usize::from(board.locomotive);
            let cur = usize::from(base.cur);
            for length in 1..=board.max_len {
                // (colour cards, locomotives, colour slot legal, locomotive slot legal)
                let mut cases = vec![
                    (length - 1, 0, false, false),
                    (length - 1, 1, length > 1, length == 1),
                    (length, 0, true, false),
                    (length + 1, length, true, true),
                    (0, length, false, true),
                    (0, length - 1, false, false),
                ];
                if length >= 2 {
                    cases.push((1, length - 2, false, false));
                }
                for c in 0..board.n_colors {
                    let routes: Vec<usize> = (0..board.n_segments)
                        .filter(|&s| board.seg_len[s] == length)
                        .filter(|&s| {
                            board.seg_color[s] == GRAY || usize::from(board.seg_color[s]) == c
                        })
                        .collect();
                    for &(colored, wilds, color_ok, loco_ok) in &cases {
                        let mut state = base.clone();
                        set_hand(&mut state, cur, &[(c, colored), (loco, wilds)]);
                        let legal = agree(&state);
                        for &s in &routes {
                            let what = format!(
                                "{map}: {} holding {colored} {} and {wilds} locomotives",
                                board.segment_name(s),
                                board.color_name(c as u8)
                            );
                            assert_eq!(legal.contains(&layout.claim(s, c)), color_ok, "{what}");
                            assert_eq!(legal.contains(&layout.claim(s, loco)), loco_ok, "{what}");
                        }
                    }
                }
            }
        }
    }

    /// "Requires trains >= len": the supply caps which routes are claimable, and a seat
    /// with no trains claims nothing -- yet can still draw, so it never has to pass.
    #[test]
    fn the_train_supply_caps_the_route_and_zero_trains_claim_nothing() {
        for (map, n) in [("usa", 2), ("mini", 2), ("usa", 5)] {
            let base = fresh(map, n);
            let board = base.board;
            let layout = Layout::of(board);
            let cur = usize::from(base.cur);
            let rich: Vec<(usize, u8)> = (0..board.n_card_types).map(|c| (c, 6)).collect();
            for trains in 0..=board.max_len + 1 {
                let mut state = base.clone();
                set_hand(&mut state, cur, &rich);
                state.trains[cur] = trains;
                let legal = agree(&state);
                let segments: BTreeSet<usize> = claims_in(&legal, layout)
                    .iter()
                    .map(|&a| usize::from(a) / layout.k)
                    .collect();
                let expected: BTreeSet<usize> = (0..board.n_segments)
                    .filter(|&s| board.seg_len[s] <= trains)
                    .collect();
                assert_eq!(segments, expected, "{map} {n}P with {trains} trains");
                assert!(legal.contains(&layout.blind()) && legal.contains(&layout.draw_tickets));
                assert!(!legal.contains(&layout.pass));
            }
        }
    }

    /// 2-3P: once either track of a double is claimed by anyone, the twin is closed to
    /// everyone. 4-5P: it stays open to the others but never to the owner. Getting this
    /// backwards is the most common TTR engine bug, and invisible in play: an agent simply
    /// learns to avoid doubles. Played through the engine, so the sentinels are its own.
    #[test]
    fn double_routes_close_to_everyone_at_two_and_three_seats_and_to_the_owner_above() {
        for (map, a, b) in [
            ("usa", "Portland", "Seattle"),
            ("mini", "Portland", "Seattle"),
        ] {
            let max = usize::from(get_board(map).unwrap().raw.max_players);
            for n in 2..=max {
                let mut state = fresh(map, n);
                let board = state.board;
                let layout = Layout::of(board);
                let (first, twin) = (segment_named(board, a, b, 0), segment_named(board, a, b, 1));
                // One colour per seat, two cards each: enough for a one-car gray route twice.
                for p in 0..n {
                    set_hand(&mut state, p, &[(p, 2)]);
                }
                state
                    .step(layout.claim(first, 0))
                    .expect("seat 0 can claim the first track");
                let legal = agree(&state);
                let open = legal.contains(&layout.claim(twin, 1));
                if n <= DOUBLES_LOCKED_UP_TO {
                    assert!(!open, "{map} {n}P: the twin is still open to seat 1");
                    assert_eq!(state.seg_owner[twin], CLOSED);
                } else {
                    assert!(
                        open,
                        "{map} {n}P: the twin closed to a seat that owns neither"
                    );
                    // Round the table back to the owner, everyone else drawing blind.
                    while state.cur != 0 {
                        state.step(layout.blind()).expect("a blind draw");
                        if state.phase == PHASE_DRAW_SECOND {
                            state.step(layout.blind()).expect("a second blind draw");
                        }
                    }
                    let legal = agree(&state);
                    assert!(
                        !claims_in(&legal, layout)
                            .iter()
                            .any(|&c| usize::from(c) / layout.k == twin),
                        "{map} {n}P: seat 0 may claim the twin of its own track"
                    );
                }
            }
        }
    }

    /// Empty face-up slots are untakeable, the blind draw needs a card in the deck or a
    /// discard to reshuffle (lazily, so the two cases stay distinguishable), and a face-up
    /// locomotive is only ever the first card.
    #[test]
    fn empty_slots_and_dry_pools_are_never_drawable() {
        for (map, n) in [("usa", 2), ("mini", 4)] {
            let mut state = fresh(map, n);
            let layout = Layout::of(state.board);
            let loco = state.board.locomotive as i8;
            state.faceup = [0, EMPTY_SLOT, 1, EMPTY_SLOT, loco];
            dry_pools(&mut state);
            assert_eq!(draw_slots(&agree(&state), layout), [0, 2, 4], "{map}: dry");
            state.discard[3] = 2;
            state.discard_total = 2;
            assert_eq!(
                draw_slots(&agree(&state), layout),
                [0, 2, 4, 5],
                "{map}: discards"
            );
            state.discard[3] = 0;
            state.discard_total = 0;
            state.deck_pos = state.deck_len - 1;
            assert_eq!(
                draw_slots(&agree(&state), layout),
                [0, 2, 4, 5],
                "{map}: one card"
            );

            // The second card is a draw or nothing: no claim, no tickets, no pass.
            state.phase = PHASE_DRAW_SECOND;
            state.draws_left = 1;
            let legal = agree(&state);
            assert_eq!(
                draw_slots(&legal, layout).len(),
                legal.len(),
                "{map}: {legal:?}"
            );
            assert_eq!(draw_slots(&legal, layout), [0, 2, 5], "{map}: second draw");
            dry_pools(&mut state);
            assert_eq!(
                draw_slots(&agree(&state), layout),
                [0, 2],
                "{map}: second, dry"
            );
        }
    }

    /// An empty ticket deck makes DRAW_TICKETS illegal; a single ticket left, wherever the
    /// ring's head has wandered to, makes it legal again.
    #[test]
    fn an_empty_ticket_deck_cannot_be_drawn_from() {
        for (map, n) in [("usa", 3), ("mini", 2)] {
            let mut state = fresh(map, n);
            let layout = Layout::of(state.board);
            let capacity = state.board.n_tickets;
            assert!(agree(&state).contains(&layout.draw_tickets));
            no_tickets(&mut state);
            assert!(
                !agree(&state).contains(&layout.draw_tickets),
                "{map}: empty"
            );
            let slot = capacity - 1;
            state.tdeck[slot] = 0;
            state.tdeck_head = slot as u8;
            state.tdeck_len = 1;
            assert!(
                agree(&state).contains(&layout.draw_tickets),
                "{map}: one left"
            );
        }
    }

    /// PASS is the explicit move for a seat with nothing else to do, and only then.
    #[test]
    fn pass_is_legal_exactly_when_nothing_else_is() {
        for (map, n) in [("usa", 2), ("usa", 5), ("mini", 3)] {
            let mut state = fresh(map, n);
            let board = state.board;
            let layout = Layout::of(board);
            let cur = usize::from(state.cur);
            state.faceup = [EMPTY_SLOT; FACEUP_SLOTS];
            dry_pools(&mut state);
            no_tickets(&mut state);
            set_hand(&mut state, cur, &[]);
            assert_eq!(agree(&state), [layout.pass], "{map} {n}P: nothing to do");

            // One card buys a one-car route, and then there is something to do.
            set_hand(&mut state, cur, &[(0, 1)]);
            let legal = agree(&state);
            assert!(
                !legal.is_empty() && legal.iter().all(|&a| a < layout.draw_base),
                "{map}"
            );

            // A full hand with no trains is as stuck as an empty one.
            let rich: Vec<(usize, u8)> = (0..board.n_card_types).map(|c| (c, 6)).collect();
            set_hand(&mut state, cur, &rich);
            state.trains[cur] = 0;
            assert_eq!(agree(&state), [layout.pass], "{map} {n}P: no trains");

            // A ticket to draw, or a card showing, is something to do.
            state.tdeck[0] = 0;
            state.tdeck_head = 0;
            state.tdeck_len = 1;
            assert_eq!(agree(&state), [layout.draw_tickets], "{map}: a ticket");
            no_tickets(&mut state);
            state.faceup[2] = board.locomotive as i8;
            assert_eq!(
                agree(&state),
                [layout.draw_base + 2],
                "{map}: a locomotive showing"
            );
        }
    }

    /// Keep masks: never empty, never a slot the offer does not fill, and at least the
    /// minimum -- two of the opening three, one later -- capped by the offer's size.
    #[test]
    fn keep_masks_honour_the_minimum_for_every_offer_size() {
        for map in ["usa", "mini"] {
            let mut state = fresh(map, 2);
            let layout = Layout::of(state.board);
            let cases: [(u8, usize, &[u16]); 6] = [
                (PHASE_TICKET_KEEP, 3, &[1, 2, 3, 4, 5, 6, 7]),
                (PHASE_TICKET_KEEP, 2, &[1, 2, 3]),
                (PHASE_TICKET_KEEP, 1, &[1]),
                (PHASE_INITIAL_TICKETS, 3, &[3, 5, 6, 7]),
                (PHASE_INITIAL_TICKETS, 2, &[3]),
                (PHASE_INITIAL_TICKETS, 1, &[1]),
            ];
            for (phase, size, masks) in cases {
                state.phase = phase;
                state.offer = [NO_TICKET; 3];
                for i in 0..size {
                    state.offer[i] = i as u8;
                }
                state.offer_len = size as u8;
                let kept: Vec<u16> = agree(&state)
                    .iter()
                    .map(|&a| a - layout.draw_tickets)
                    .collect();
                assert_eq!(
                    kept,
                    masks,
                    "{map}: {} with {size} offered",
                    PHASE_NAMES[usize::from(phase)]
                );
            }
        }
    }

    /// Every seat passing in a row freezes the game: it ends on the last pass, and the
    /// streak is counted from the public record.
    #[test]
    fn an_all_pass_round_freezes_the_game() {
        for (map, n) in [("usa", 2), ("usa", 5), ("mini", 3)] {
            let mut state = fresh(map, n);
            let pass = Layout::of(state.board).pass;
            state.faceup = [EMPTY_SLOT; FACEUP_SLOTS];
            dry_pools(&mut state);
            no_tickets(&mut state);
            for p in 0..n {
                set_hand(&mut state, p, &[]);
            }
            let mut books = Bookkeeper::adopt(&state);
            for _ in 0..n {
                assert!(!state.is_terminal(), "{map} {n}P froze early");
                assert_eq!(agree(&state), [pass]);
                let before = state.clone();
                state.step(pass).expect("PASS is legal");
                books
                    .observe(&before, pass, &state)
                    .unwrap_or_else(|e| panic!("{map} {n}P: {e}"));
            }
            assert!(
                state.is_terminal(),
                "{map} {n}P: {n} passes did not freeze the game"
            );
        }
    }

    /// The end fires at the end of the turn that leaves a seat on two trains or fewer --
    /// whether that turn was a claim or a draw -- and every seat, that one included, then
    /// gets exactly one more turn. Counted in closed form by the bookkeeper.
    #[test]
    fn the_end_triggers_on_two_trains_and_grants_one_more_turn_each() {
        for (map, n) in [("usa", 2), ("usa", 4), ("mini", 3)] {
            for by_claim in [true, false] {
                let mut state = fresh(map, n);
                let board = state.board;
                let layout = Layout::of(board);
                let short = (0..board.n_segments)
                    .find(|&s| board.seg_len[s] == 1 && board.seg_color[s] == GRAY)
                    .expect("a one-car gray route");
                set_hand(&mut state, 0, &[(0, 1)]);
                state.trains[0] = if by_claim { 3 } else { 2 };
                let mut books = Bookkeeper::adopt(&state);
                let mut step = |state: &mut State, action: u16| {
                    let before = state.clone();
                    state.step(action).expect("legal");
                    books
                        .observe(&before, action, state)
                        .unwrap_or_else(|e| panic!("{map} {n}P: {e}"));
                };
                if by_claim {
                    step(&mut state, layout.claim(short, 0));
                } else {
                    step(&mut state, layout.blind());
                    step(&mut state, layout.blind());
                }
                assert_eq!(
                    usize::from(state.final_left),
                    n,
                    "{map} {n}P: not triggered"
                );
                let mut turns = 0;
                while !state.is_terminal() {
                    step(&mut state, layout.blind());
                    if state.phase == PHASE_DRAW_SECOND {
                        step(&mut state, layout.blind());
                    }
                    turns += 1;
                }
                assert_eq!(turns, n, "{map} {n}P: final turns after the trigger");
            }
        }
    }

    /// The turn cap is belt and braces -- random games end far inside it -- so the branch
    /// that enforces it only ever runs on a rigged position: the turn that reaches it ends
    /// the game, and the bookkeeper expects exactly that.
    #[test]
    fn the_turn_cap_ends_the_game_on_the_turn_that_reaches_it() {
        for (map, n) in [("usa", 3), ("mini", 2)] {
            let mut state = fresh(map, n);
            let blind = Layout::of(state.board).blind();
            let cap = state.rules.turn_cap;
            state.turn = cap - 2;
            state.cur = ((cap - 2) % n as u16) as u8;
            let mut books = Bookkeeper::adopt(&state);
            for _ in 0..2 {
                assert!(!state.is_terminal(), "{map} {n}P ended before the cap");
                for _ in 0..2 {
                    let before = state.clone();
                    state.step(blind).expect("a blind draw");
                    books
                        .observe(&before, blind, &state)
                        .unwrap_or_else(|e| panic!("{map} {n}P: {e}"));
                }
            }
            assert!(
                state.is_terminal(),
                "{map} {n}P: turn {} did not end the game",
                state.turn
            );
            assert_eq!(state.turn, cap);
        }
    }

    // -----------------------------------------------------------------------
    // Random positions
    // -----------------------------------------------------------------------

    /// A position to rig onto a fresh game: each part shrinks on its own when it fails.
    ///
    /// The switches make the rare combinations common. Drawn independently per slot and per
    /// card type, an all-empty display or a dry discard would almost never come up, and
    /// PASS needs both at once.
    #[derive(Clone, Debug)]
    struct Rig {
        usa: bool,
        seats: usize,
        cur: usize,
        hand: Vec<u8>,
        trains: u8,
        /// Per segment: 0..=4 leaves it free, 5.. gives it to a seat.
        owners: Vec<u8>,
        /// Per slot: -1 is an empty slot.
        faceup: Vec<i8>,
        empty_display: bool,
        deck_left: u16,
        discard: Vec<u8>,
        dry_discard: bool,
        tickets_left: bool,
        second_draw: bool,
    }

    fn rig() -> impl Strategy<Value = Rig> {
        let seats = (any::<bool>(), 0usize..4, 0usize..5);
        let hand = (
            vec(0u8..=7, MAX_CARD_TYPES),
            prop_oneof![1 => Just(0u8), 3 => 0u8..=46],
        );
        let board = vec(0u8..=9, MAX_SEGMENTS);
        let display = (
            vec(prop_oneof![Just(-1i8), 0i8..=8], FACEUP_SLOTS),
            any::<bool>(),
        );
        let pools = (
            prop_oneof![Just(0u16), 1u16..=2],
            vec(0u8..=2, MAX_CARD_TYPES),
            any::<bool>(),
        );
        let turn = (any::<bool>(), any::<bool>());
        (seats, hand, board, display, pools, turn).prop_map(
            |(
                (usa, seats, cur),
                (hand, trains),
                owners,
                (faceup, empty_display),
                (deck_left, discard, dry_discard),
                (tickets_left, second_draw),
            )| Rig {
                usa,
                seats,
                cur,
                hand,
                trains,
                owners,
                faceup,
                empty_display,
                deck_left,
                discard,
                dry_discard,
                tickets_left,
                second_draw,
            },
        )
    }

    /// Build the rig on a fresh game, normalising double routes the way claims would have
    /// left them, and keeping the hand's information set and the discard count consistent
    /// (their own consistency is what the sweeps check).
    fn build(r: &Rig) -> State {
        let map = if r.usa { "usa" } else { "mini" };
        let board = get_board(map).unwrap();
        let (low, high) = (
            usize::from(board.raw.min_players),
            usize::from(board.raw.max_players),
        );
        let n = low + r.seats % (high - low + 1);
        let mut state = fresh(map, n);
        let k = board.n_card_types;
        let cur = r.cur % n;
        state.cur = cur as u8;
        let cards: Vec<(usize, u8)> = (0..k).map(|c| (c, r.hand[c])).collect();
        set_hand(&mut state, cur, &cards);
        state.trains[cur] = r.trains.min(board.raw.trains_per_player);
        for s in 0..board.n_segments {
            let code = r.owners[s];
            state.seg_owner[s] = if code < 5 {
                FREE
            } else {
                (usize::from(code - 5) % n) as u8
            };
        }
        for s in 0..board.n_segments {
            let twin = board.sibling[s];
            if twin == NO_SIBLING || usize::from(twin) < s {
                continue;
            }
            let t = usize::from(twin);
            let (mine, theirs) = (state.seg_owner[s], state.seg_owner[t]);
            if n <= DOUBLES_LOCKED_UP_TO {
                if mine != FREE {
                    state.seg_owner[t] = CLOSED;
                } else if theirs != FREE {
                    state.seg_owner[s] = CLOSED;
                }
            } else if mine != FREE && mine == theirs {
                state.seg_owner[t] = FREE;
            }
        }
        for slot in 0..FACEUP_SLOTS {
            let card = r.faceup[slot];
            state.faceup[slot] = if card < 0 || r.empty_display {
                EMPTY_SLOT
            } else {
                (card as usize % k) as i8
            };
        }
        state.deck_pos = state.deck_len.saturating_sub(r.deck_left);
        state.discard = [0; MAX_CARD_TYPES];
        if !r.dry_discard {
            state.discard[..k].copy_from_slice(&r.discard[..k]);
        }
        state.discard_total = state.discard[..k].iter().map(|&c| u16::from(c)).sum();
        if !r.tickets_left {
            no_tickets(&mut state);
        }
        if r.second_draw {
            state.phase = PHASE_DRAW_SECOND;
            state.draws_left = 1;
        }
        state
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        /// Hands from empty to seven of a kind, trains from none to a full supply, any
        /// ownership, short displays and dry pools: the engine and the rules agree on the
        /// legal set, and `step()` accepts exactly that set.
        #[test]
        fn random_positions_agree_with_the_naive_scan(r in rig()) {
            let state = build(&r);
            check_legality(&state).map_err(TestCaseError::fail)?;
            check_step_acceptance(&state).map_err(TestCaseError::fail)?;
        }
    }
}
