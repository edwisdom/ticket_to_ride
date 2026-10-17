//! The observation encoder.
//!
//! Layout comes entirely from [`crate::obs_spec`], the declarative feature table, so the
//! encoder decides only the *value* written into a slot, never where it goes (PLAN.md
//! §8.2).
//!
//! **Arithmetic: computed in `f64`, narrowed on store.** The buffer is `f32`; every
//! intermediate is an integer or an `f64`, and each feature is narrowed once. This began as
//! a match for the retired Python encoder, which computed in doubles and narrowed on store,
//! and it is now what the golden corpus and `oracle-v1` were signed under. Dividing
//! directly in `f32` happens to agree over every divisor used today --
//! `double_rounding_does_not_bite_over_the_ranges_used` checks that rather than assuming it
//! -- but it agrees by luck of the ranges, and a new feature with a different divisor could
//! double-round differently in one slot out of 3355.
//!
//! One thing that reads oddly and is correct: this takes `&mut State`, because the
//! union-find roots are read through path halving. The DSU reaches no hash.
//!
//! **Cost.** Nearly all of an observation is graph work that depends only on the board and
//! the observing seat's tickets ([`Derived`]): the all-pairs table, the Steiner DP,
//! fragility's searches, the opponents' longest trails. [`encode_cached`] keeps it per seat
//! in an [`EncoderCache`] and skips it whenever a step changed neither -- card draws, about
//! half of all steps. [`encode`] recomputes everything; the two write identical buffers.

use crate::board::{
    CityId, GRAY, MAX_CARD_TYPES, MAX_CITIES, MAX_PLAYERS, MAX_SEGMENTS, MAX_TICKETS, NO_SIBLING,
    SegmentId, UNREACHABLE,
};
use crate::config::{CLOSED, EMPTY_SLOT, FREE, NOT_TRIGGERED, PHASE_DRAW_SECOND};
use crate::graph::{
    Distances, MIN_TERMINALS, all_pairs, crossing_costs, guided_distance, longest_trail,
    steiner_cost_with,
};
pub use crate::obs_spec::spec_for;
use crate::obs_spec::{
    COST_BUCKETS, HAND_BUCKETS, OBS_VERSION, OPPONENT_SLOTS, ObsBlock, ObsSpec, TRAIN_BUCKETS,
};
use crate::state::State;

/// Score and gap-to-leader are normalized by this; the scale is a rough game maximum, not
/// a bound, so the feature may exceed 1.
const SCORE_SCALE: f64 = 100.0;

/// Turn index is clipped here before normalizing. The analytical 4P bound is ~335 turns,
/// but a typical game is well under 200 and the clip keeps the feature informative.
const TURN_CLIP: f64 = 200.0;

/// The number of `own_hand` thermometer levels: "at least one" through "at least six".
const HAND_THERMOMETER_LEVELS: usize = 6;

pub fn observation_size(state: &State) -> usize {
    spec_for(state.board.name).size
}

pub fn obs_version() -> u32 {
    OBS_VERSION
}

// ---------------------------------------------------------------------------
// Resolved layout
// ---------------------------------------------------------------------------

/// One block's geometry plus its field offsets, resolved by name **once** per board.
///
/// The spec is a table of names; looking a field up by name on every write would cost more
/// than the encoding. This resolves them at first use and caches per board.
#[derive(Clone, Copy)]
struct Blk {
    stride: usize,
    offset: usize,
}

impl Blk {
    #[inline]
    fn at(&self, entity: usize, field: usize) -> usize {
        self.offset + entity * self.stride + field
    }
}

fn block_of<'a>(spec: &'a ObsSpec, name: &str) -> &'a ObsBlock {
    spec.block(name)
        .unwrap_or_else(|| panic!("the spec has no block {name:?}"))
}

fn field_of(block: &ObsBlock, name: &str) -> usize {
    block
        .field(name)
        .unwrap_or_else(|| panic!("block {:?} has no field {name:?}", block.name))
        .offset
}

macro_rules! resolve {
    ($spec:expr, $block:literal, { $($field:ident : $fname:literal),* $(,)? }) => {{
        let b = block_of($spec, $block);
        (
            Blk { stride: b.stride, offset: b.offset },
            $( field_of(b, $fname), )*
        )
    }};
}

struct Layout {
    size: usize,
    seg_static: Blk,
    required_color: usize,
    seg_dyn: Blk,
    owner: usize,
    closed: usize,
    twin_locked: usize,
    can_afford_now: usize,
    cards_short: usize,
    on_my_steiner_tree: usize,
    extends_my_chain: usize,
    own_hand: Blk,
    counts: usize,
    hand_therm: [usize; HAND_THERMOMETER_LEVELS],
    tickets: Blk,
    held: usize,
    connected: usize,
    ticket_remaining_cost: usize,
    cost_thermometer: usize,
    is_dead: usize,
    points: usize,
    fragility: usize,
    steiner: Blk,
    steiner_remaining_cost: usize,
    cost_minus_trains: usize,
    exact: usize,
    faceup: Blk,
    card: usize,
    piles: Blk,
    deck_size: usize,
    discard_size: usize,
    ticket_deck_size: usize,
    discard_composition: usize,
    unseen: usize,
    opponents: Blk,
    present: usize,
    opp_trains: usize,
    trains_thermometer: usize,
    opp_score: usize,
    hand_size: usize,
    ticket_count: usize,
    blind_draws: usize,
    segments_claimed: usize,
    longest_chain: usize,
    certain: usize,
    max_possible: usize,
    clock: Blk,
    phase: usize,
    seats: usize,
    draws_left: usize,
    final_triggered: usize,
    final_countdown: usize,
    turn: usize,
    my_trains: usize,
    my_trains_thermometer: usize,
    my_score: usize,
    my_ticket_count: usize,
    score_rank: usize,
    gap_to_leader: usize,
}

impl Layout {
    fn build(spec: &ObsSpec) -> Self {
        let (seg_static, required_color) =
            resolve!(spec, "segment_static", { required_color: "required_color" });
        let (
            seg_dyn,
            owner,
            closed,
            twin_locked,
            can_afford_now,
            cards_short,
            on_my_steiner_tree,
            extends_my_chain,
        ) = resolve!(spec, "segment_dynamic", {
            owner: "owner",
            closed: "closed",
            twin_locked: "twin_locked",
            can_afford_now: "can_afford_now",
            cards_short: "cards_short",
            on_my_steiner_tree: "on_my_steiner_tree",
            extends_my_chain: "extends_my_chain",
        });
        let hand_block = block_of(spec, "own_hand");
        let own_hand = Blk {
            stride: hand_block.stride,
            offset: hand_block.offset,
        };
        let counts = field_of(hand_block, "counts");
        let mut hand_therm = [0usize; HAND_THERMOMETER_LEVELS];
        for (level, slot) in hand_therm.iter_mut().enumerate() {
            // The spec names level 1 "thermometer" and the rest "thermometerN".
            *slot = if level == 0 {
                field_of(hand_block, "thermometer")
            } else {
                field_of(hand_block, &format!("thermometer{}", level + 1))
            };
        }
        let (
            tickets,
            held,
            connected,
            ticket_remaining_cost,
            cost_thermometer,
            is_dead,
            points,
            fragility,
        ) = resolve!(spec, "tickets", {
            held: "held",
            connected: "connected",
            ticket_remaining_cost: "remaining_cost",
            cost_thermometer: "cost_thermometer",
            is_dead: "is_dead",
            points: "points",
            fragility: "fragility",
        });
        let (steiner, steiner_remaining_cost, cost_minus_trains, exact) = resolve!(spec, "steiner", {
            steiner_remaining_cost: "remaining_cost",
            cost_minus_trains: "cost_minus_trains",
            exact: "exact",
        });
        let (faceup, card) = resolve!(spec, "faceup", { card: "card" });
        let (piles, deck_size, discard_size, ticket_deck_size, discard_composition, unseen) = resolve!(spec, "piles", {
            deck_size: "deck_size",
            discard_size: "discard_size",
            ticket_deck_size: "ticket_deck_size",
            discard_composition: "discard_composition",
            unseen: "unseen",
        });
        let (
            opponents,
            present,
            opp_trains,
            trains_thermometer,
            opp_score,
            hand_size,
            ticket_count,
            blind_draws,
            segments_claimed,
            longest_chain,
            certain,
            max_possible,
        ) = resolve!(spec, "opponents", {
            present: "present",
            opp_trains: "trains",
            trains_thermometer: "trains_thermometer",
            opp_score: "score",
            hand_size: "hand_size",
            ticket_count: "ticket_count",
            blind_draws: "blind_draws",
            segments_claimed: "segments_claimed",
            longest_chain: "longest_chain",
            certain: "certain",
            max_possible: "max_possible",
        });
        let (
            clock,
            phase,
            seats,
            draws_left,
            final_triggered,
            final_countdown,
            turn,
            my_trains,
            my_trains_thermometer,
            my_score,
            my_ticket_count,
            score_rank,
            gap_to_leader,
        ) = resolve!(spec, "clock", {
            phase: "phase",
            seats: "seats",
            draws_left: "draws_left",
            final_triggered: "final_triggered",
            final_countdown: "final_countdown",
            turn: "turn",
            my_trains: "my_trains",
            my_trains_thermometer: "my_trains_thermometer",
            my_score: "my_score",
            my_ticket_count: "my_ticket_count",
            score_rank: "score_rank",
            gap_to_leader: "gap_to_leader",
        });

        Self {
            size: spec.size,
            seg_static,
            required_color,
            seg_dyn,
            owner,
            closed,
            twin_locked,
            can_afford_now,
            cards_short,
            on_my_steiner_tree,
            extends_my_chain,
            own_hand,
            counts,
            hand_therm,
            tickets,
            held,
            connected,
            ticket_remaining_cost,
            cost_thermometer,
            is_dead,
            points,
            fragility,
            steiner,
            steiner_remaining_cost,
            cost_minus_trains,
            exact,
            faceup,
            card,
            piles,
            deck_size,
            discard_size,
            ticket_deck_size,
            discard_composition,
            unseen,
            opponents,
            present,
            opp_trains,
            trains_thermometer,
            opp_score,
            hand_size,
            ticket_count,
            blind_draws,
            segments_claimed,
            longest_chain,
            certain,
            max_possible,
            clock,
            phase,
            seats,
            draws_left,
            final_triggered,
            final_countdown,
            turn,
            my_trains,
            my_trains_thermometer,
            my_score,
            my_ticket_count,
            score_rank,
            gap_to_leader,
        }
    }
}

fn layout_for(map: &str) -> &'static Layout {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<(&'static str, Layout)>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        crate::board::all_boards()
            .iter()
            .map(|board| (board.name, Layout::build(spec_for(board.name))))
            .collect()
    });
    &cache
        .iter()
        .find(|(name, _)| *name == map)
        .expect("every board has a resolved layout")
        .1
}

// ---------------------------------------------------------------------------
// The graph work, and the cache that skips it
// ---------------------------------------------------------------------------

/// What the encoder derives from the board and one seat's tickets: the graph work, and all
/// but a few percent of an observation's cost.
///
/// A pure function of [`Key`] -- the board, the seat count, the double-route rule, every
/// segment's owner and the seat's tickets -- and of nothing a card draw changes, which lets
/// [`EncoderCache`] reuse it. Fixed-size throughout, so computing one allocates nothing.
/// My crossing costs and the all-pairs table over them are built once, and every distance
/// read -- ticket costs, the Steiner DP's terminals, the planned routes, fragility's
/// guide -- is a row of that table; neither is kept, since nothing downstream needs them.
#[derive(Clone, Debug, PartialEq)]
struct Derived {
    /// Held ticket ids, ascending.
    held: [u8; MAX_TICKETS],
    n_held: usize,
    /// Remaining cost per ticket id; meaningful for held tickets only.
    ticket_cost: [u16; MAX_TICKETS],
    /// [`fragility`] per ticket id; meaningful for held tickets that are not dead.
    fragility: [f64; MAX_TICKETS],
    steiner: u32,
    steiner_exact: bool,
    /// Free segments on the planned routes, one bit per segment.
    plan: u128,
    /// Longest trail per seat; meaningful for opponents only.
    trails: [u16; MAX_PLAYERS],
}

const _: () = assert!(MAX_SEGMENTS <= 128, "the plan bitset is a u128");

impl Derived {
    fn compute(state: &State, player: usize) -> Self {
        let board = state.board;
        let seg_owner = &state.seg_owner[..board.n_segments];
        let me = player as u8;
        let cost = crossing_costs(board, seg_owner, me, state.rules.doubles_locked);
        let dist = all_pairs(board, &cost);
        let trains_supply = f64::from(board.raw.trains_per_player);

        let mut held = [0u8; MAX_TICKETS];
        let mut n_held = 0;
        let mut ticket_cost = [UNREACHABLE; MAX_TICKETS];
        let mut fragility_of = [0.0; MAX_TICKETS];
        let mut terminals = [0 as CityId; 2 * MAX_TICKETS];
        let mut n_terminals = 0;
        for (ticket, slot) in ticket_cost.iter_mut().enumerate().take(board.n_tickets) {
            if state.tickets[player] >> ticket & 1 == 0 {
                continue;
            }
            held[n_held] = ticket as u8;
            n_held += 1;
            let (a, b) = (board.ticket_a[ticket], board.ticket_b[ticket]);
            let c = dist[usize::from(a)][usize::from(b)];
            *slot = c;
            if c > 0 && c < UNREACHABLE {
                terminals[n_terminals] = a;
                terminals[n_terminals + 1] = b;
                n_terminals += 2;
                fragility_of[ticket] = fragility(state, me, &cost, &dist, ticket, trains_supply);
            }
        }
        let terminals = &terminals[..n_terminals];
        let (steiner, steiner_exact) = steiner_cost_with(board, &dist, terminals);
        let plan = planned_segments(state, player, &dist, terminals);

        let mut trails = [0u16; MAX_PLAYERS];
        for (seat, trail) in trails.iter_mut().enumerate().take(state.n_players()) {
            if seat != player {
                *trail = longest_trail(board, seg_owner, seat as u8);
            }
        }
        Self {
            held,
            n_held,
            ticket_cost,
            fragility: fragility_of,
            steiner,
            steiner_exact,
            plan,
            trails,
        }
    }

    fn held(&self) -> &[u8] {
        &self.held[..self.n_held]
    }
}

/// Everything [`Derived`] is a function of, compared whole: the inputs themselves rather
/// than a version counter, so nothing ever has to invalidate a cache entry.
#[derive(Clone, Debug, PartialEq)]
struct Key {
    /// The board's address: boards are `'static`, one per map.
    board: usize,
    n_players: usize,
    doubles_locked: bool,
    seg_owner: [u8; MAX_SEGMENTS],
    tickets: u32,
}

impl Key {
    fn of(state: &State, player: usize) -> Self {
        Self {
            board: std::ptr::from_ref(state.board) as usize,
            n_players: state.n_players(),
            doubles_locked: state.rules.doubles_locked,
            seg_owner: state.seg_owner,
            tickets: state.tickets[player],
        }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    key: Key,
    derived: Derived,
}

/// The encoder's graph work, remembered per seat -- one of these per environment.
///
/// Each seat keeps the [`Derived`] of the last position it was encoded in, keyed by the
/// inputs it came from. A reset, a replaced state or another game simply misses; a hit is
/// equal to recomputing by construction, and debug builds recompute on every hit and assert
/// it (PLAN.md §14.1, Phase 3.2). `tests/encoder_cache.rs` drives that over ten thousand
/// games.
#[derive(Clone, Debug)]
pub struct EncoderCache {
    seats: [Option<Entry>; MAX_PLAYERS],
    hits: u64,
    misses: u64,
}

impl Default for EncoderCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderCache {
    pub const fn new() -> Self {
        Self {
            seats: [const { None }; MAX_PLAYERS],
            hits: 0,
            misses: 0,
        }
    }

    /// Lookups answered from the cache.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Lookups that recomputed.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    fn lookup(&mut self, state: &State, player: usize) -> &Derived {
        let key = Key::of(state, player);
        let slot = &mut self.seats[player];
        if matches!(slot, Some(entry) if entry.key == key) {
            self.hits += 1;
            debug_assert_eq!(
                slot.as_ref().map(|entry| &entry.derived),
                Some(&Derived::compute(state, player)),
                "the encoder cache disagrees with recomputation"
            );
        } else {
            self.misses += 1;
            *slot = Some(Entry {
                derived: Derived::compute(state, player),
                key,
            });
        }
        &slot.as_ref().expect("filled above").derived
    }
}

/// What the blocks read: the board-derived work, and the per-position rest -- seat order,
/// union-find roots, unseen counts -- which is cheap and changes with every card.
struct View<'a> {
    me: usize,
    /// Opponents by distance in turn order: slot 0 acts next. `n_players - 1` entries.
    order: [usize; MAX_PLAYERS - 1],
    n_opponents: usize,
    trains_supply: f64,
    dsu_root: [u8; MAX_CITIES],
    unseen: [i16; MAX_CARD_TYPES],
    derived: &'a Derived,
}

impl<'a> View<'a> {
    fn build(state: &mut State, player: usize, derived: &'a Derived) -> Self {
        let board = state.board;
        let n = state.n_players();
        let mut order = [0usize; MAX_PLAYERS - 1];
        for (i, slot) in order.iter_mut().enumerate().take(n - 1) {
            *slot = (player + 1 + i) % n;
        }
        let mut dsu_root = [0u8; MAX_CITIES];
        for (city, root) in dsu_root.iter_mut().enumerate().take(board.n_cities) {
            *root = state.dsu_root(player, city as u8);
        }
        let mut unseen = [0i16; MAX_CARD_TYPES];
        for (slot, count) in unseen.iter_mut().zip(state.unseen_counts(player)) {
            *slot = count;
        }
        Self {
            me: player,
            order,
            n_opponents: n - 1,
            trains_supply: f64::from(board.raw.trains_per_player),
            dsu_root,
            unseen,
            derived,
        }
    }

    fn order(&self) -> &[usize] {
        &self.order[..self.n_opponents]
    }
}

/// Walk back from `target` to `source` along one cheapest route, calling `step` with each
/// segment crossed. `dist` is the table row from `source`.
///
/// Reproduces `OBS_VERSION` 1 exactly, defect included: my own track counts as weight 0,
/// so at a city whose neighbour across my track has the same distance the walk can step
/// back across it, and it ends when the guard runs out with the route truncated. See
/// `tests/obs_plan.rs`; fixing it is a named `OBS_VERSION` bump.
fn walk_route(
    state: &State,
    me: u8,
    dist: &[u16],
    source: CityId,
    target: CityId,
    mut step: impl FnMut(SegmentId, u8),
) {
    let board = state.board;
    if dist[usize::from(target)] >= UNREACHABLE {
        return;
    }
    let mut here = target;
    let mut guard = 0;
    while here != source && guard <= board.n_cities {
        guard += 1;
        let mut stepped = false;
        for &(nb, segment) in &board.adjacency[usize::from(here)] {
            let owner = state.seg_owner[usize::from(segment)];
            if owner != FREE && owner != me {
                continue;
            }
            let weight = if owner == me {
                0
            } else {
                u16::from(board.seg_len[usize::from(segment)])
            };
            if dist[usize::from(nb)] + weight == dist[usize::from(here)] {
                step(segment, owner);
                here = nb;
                stepped = true;
                break;
            }
        }
        if !stepped {
            break;
        }
    }
}

/// Free segments on *a* cheapest route between consecutive terminals.
///
/// An approximation of "on my Steiner tree", and deliberately so: the exact edge set of a
/// Dreyfus-Wagner solution needs back-pointers through the subset DP, which is a lot of
/// machinery for a boolean hint. What the network needs is "this route is on my plan", and
/// a union of cheapest paths gives that.
fn planned_segments(state: &State, me: usize, dist: &Distances, terminals: &[CityId]) -> u128 {
    if terminals.len() < MIN_TERMINALS {
        return 0;
    }
    let mut plan = 0u128;
    for pair in terminals.chunks_exact(2) {
        let (source, target) = (pair[0], pair[1]);
        walk_route(
            state,
            me as u8,
            &dist[usize::from(source)],
            source,
            target,
            |segment, owner| {
                if owner == FREE {
                    plan |= 1 << segment;
                }
            },
        );
    }
    plan
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encode `state` from `player`'s point of view into `out`, computing everything.
///
/// Opponents are ordered by **distance in turn order**, so slot 0 is whoever acts next.
/// Seat-relative rather than permutation-invariant, which is the right prior: in Ticket to
/// Ride, who moves before you is decisive.
///
/// # Panics
/// If `out` is not exactly the spec's size for this board.
pub fn encode(state: &mut State, player: usize, out: &mut [f32]) {
    let derived = Derived::compute(state, player);
    encode_with(state, player, &derived, out);
}

/// [`encode`], reusing the graph work `cache` holds for `player` when the board and their
/// tickets are as they were; the buffer written is identical either way.
///
/// # Panics
/// If `out` is not exactly the spec's size for this board.
pub fn encode_cached(state: &mut State, player: usize, out: &mut [f32], cache: &mut EncoderCache) {
    let derived = cache.lookup(state, player);
    encode_with(state, player, derived, out);
}

fn encode_with(state: &mut State, player: usize, derived: &Derived, out: &mut [f32]) {
    let layout = layout_for(state.board.name);
    assert_eq!(
        out.len(),
        layout.size,
        "buffer holds {} floats, spec needs {}",
        out.len(),
        layout.size
    );
    out.fill(0.0);

    let view = View::build(state, player, derived);
    segment_static(&view, state, layout, out);
    segment_dynamic(&view, state, layout, out);
    own_hand(&view, state, layout, out);
    tickets(&view, state, layout, out);
    steiner(&view, state, layout, out);
    faceup(&view, state, layout, out);
    piles(&view, state, layout, out);
    opponents(&view, state, layout, out);
    clock(&view, state, layout, out);
}

fn thermometer<T: Copy + Into<u32>>(out: &mut [f32], base: usize, value: u32, buckets: &[T]) {
    for (i, &edge) in buckets.iter().enumerate() {
        out[base + i] = if value >= edge.into() { 1.0 } else { 0.0 };
    }
}

fn segment_static(_view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    for segment in 0..board.n_segments {
        let color = board.seg_color[segment];
        let index = if color == GRAY {
            board.n_colors
        } else {
            color as usize
        };
        out[l.seg_static.at(segment, l.required_color) + index] = 1.0;
    }
}

fn segment_dynamic(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let mut seat_slot = [0usize; MAX_PLAYERS];
    for (i, &seat) in view.order().iter().enumerate() {
        seat_slot[seat] = i;
    }
    let hand_base = view.me * board.n_card_types;
    let wilds = state.hand[hand_base + board.locomotive as usize];
    let mut reach = [0u8; crate::board::MAX_CARD_TYPES];
    for (r, &held) in reach
        .iter_mut()
        .zip(&state.hand[hand_base..hand_base + board.n_colors])
    {
        *r = if held > 0 { held + wilds } else { 0 };
    }
    let best_any = reach[..board.n_colors].iter().copied().max().unwrap_or(0);
    let my_trains = state.trains[view.me];

    for segment in 0..board.n_segments {
        let owner = state.seg_owner[segment];
        let length = board.seg_len[segment];
        let color = board.seg_color[segment];

        let owner_slot = if owner == FREE || owner == CLOSED {
            0
        } else if owner as usize == view.me {
            1
        } else {
            2 + seat_slot[owner as usize]
        };
        out[l.seg_dyn.at(segment, l.owner) + owner_slot] = 1.0;
        out[l.seg_dyn.at(segment, l.closed)] = if owner == CLOSED { 1.0 } else { 0.0 };

        let twin = board.sibling[segment];
        out[l.seg_dyn.at(segment, l.twin_locked)] =
            if twin != NO_SIBLING && state.seg_owner[twin as usize] as usize == view.me {
                1.0
            } else {
                0.0
            };

        let best = if color == GRAY {
            best_any
        } else {
            reach[color as usize]
        }
        .max(wilds);
        let short = u32::from(length.saturating_sub(best));
        out[l.seg_dyn.at(segment, l.can_afford_now)] =
            if owner == FREE && short == 0 && my_trains >= length {
                1.0
            } else {
                0.0
            };
        thermometer(
            out,
            l.seg_dyn.at(segment, l.cards_short),
            short,
            &HAND_BUCKETS,
        );

        out[l.seg_dyn.at(segment, l.on_my_steiner_tree)] = if view.derived.plan >> segment & 1 == 1
        {
            1.0
        } else {
            0.0
        };
        let (a, b) = (board.seg_a[segment], board.seg_b[segment]);
        let touches = view.dsu_root[a as usize] != a || view.dsu_root[b as usize] != b;
        out[l.seg_dyn.at(segment, l.extends_my_chain)] =
            if owner == FREE && touches { 1.0 } else { 0.0 };
    }
}

fn own_hand(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let base = view.me * board.n_card_types;
    let per_color = f64::from(board.raw.cards_per_color);
    for c in 0..board.n_card_types {
        let count = state.hand[base + c];
        out[l.own_hand.at(0, l.counts) + c] = (f64::from(count) / per_color) as f32;
        for (level, &field) in l.hand_therm.iter().enumerate() {
            // `hand_therm[0]` is "at least one", so the threshold is the index plus one.
            out[l.own_hand.at(0, field) + c] = if u32::from(count) > level as u32 {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn tickets(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let max_points = f64::from(*board.ticket_points.iter().max().expect("tickets exist"));
    for ticket in 0..board.n_tickets {
        out[l.tickets.at(ticket, l.points)] =
            (f64::from(board.ticket_points[ticket]) / max_points) as f32;
    }
    for &t in view.derived.held() {
        let ticket = usize::from(t);
        let cost = view.derived.ticket_cost[ticket];
        let dead = cost >= UNREACHABLE;
        out[l.tickets.at(ticket, l.held)] = 1.0;
        out[l.tickets.at(ticket, l.connected)] = if cost == 0 { 1.0 } else { 0.0 };
        out[l.tickets.at(ticket, l.is_dead)] = if dead { 1.0 } else { 0.0 };
        if dead {
            continue;
        }
        out[l.tickets.at(ticket, l.ticket_remaining_cost)] =
            (f64::from(cost) / view.trains_supply) as f32;
        thermometer(
            out,
            l.tickets.at(ticket, l.cost_thermometer),
            u32::from(cost),
            &COST_BUCKETS,
        );
        out[l.tickets.at(ticket, l.fragility)] =
            (view.derived.fragility[ticket] / view.trains_supply) as f32;
    }
}

/// Worst extra cost if one enemy claim lands on my cheapest route for this ticket.
///
/// Measured by re-pricing the route with each of its free segments taken in turn: a
/// search per segment over my cost table with that one segment priced out, stopped as soon
/// as the far end is settled. That separates a safe connection from one hanging on a
/// single contested edge. The search is A* toward the far end, guided by the unblocked
/// distances already in the view, so a route with a cheap way around explores little.
fn fragility(
    state: &State,
    me: u8,
    cost: &[u16; MAX_SEGMENTS],
    dist: &Distances,
    ticket: usize,
    trains_supply: f64,
) -> f64 {
    let board = state.board;
    let (a, b) = (board.ticket_a[ticket], board.ticket_b[ticket]);
    let baseline = dist[usize::from(a)][usize::from(b)];
    if baseline == 0 || baseline >= UNREACHABLE {
        return 0.0;
    }

    let mut on_path = [0 as SegmentId; MAX_CITIES + 1];
    let mut n_on_path = 0;
    walk_route(state, me, &dist[usize::from(a)], a, b, |segment, owner| {
        if owner == FREE && n_on_path < on_path.len() {
            on_path[n_on_path] = segment;
            n_on_path += 1;
        }
    });

    let mut cost = *cost;
    let mut worst = 0u16;
    for &segment in &on_path[..n_on_path] {
        let s = usize::from(segment);
        let kept = cost[s];
        cost[s] = UNREACHABLE;
        let detour = guided_distance(board, &cost, &dist[usize::from(b)], a, b);
        cost[s] = kept;
        if detour >= UNREACHABLE {
            return trains_supply;
        }
        worst = worst.max(detour - baseline);
    }
    f64::from(worst)
}

fn steiner(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let cost = f64::from(view.derived.steiner.min((view.trains_supply as u32) * 2));
    out[l.steiner.at(0, l.steiner_remaining_cost)] = (cost / view.trains_supply) as f32;
    let slack = f64::from(state.trains[view.me]) - cost;
    out[l.steiner.at(0, l.cost_minus_trains)] = (slack / view.trains_supply) as f32;
    out[l.steiner.at(0, l.exact)] = if view.derived.steiner_exact { 1.0 } else { 0.0 };
}

fn faceup(_view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let empty = state.board.n_card_types;
    for (slot, &card) in state.faceup.iter().enumerate() {
        let index = if card == EMPTY_SLOT {
            empty
        } else {
            card as usize
        };
        out[l.faceup.at(slot, l.card) + index] = 1.0;
    }
}

fn piles(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let size = f64::from(board.deck_size as u32);
    out[l.piles.at(0, l.deck_size)] = (f64::from(state.deck_len - state.deck_pos) / size) as f32;
    out[l.piles.at(0, l.discard_size)] = (f64::from(state.discard_total) / size) as f32;
    out[l.piles.at(0, l.ticket_deck_size)] =
        (f64::from(state.tdeck_len) / f64::from(board.n_tickets as u32)) as f32;
    for c in 0..board.n_card_types {
        let printed = f64::from(board.cards_per_type(c as u8));
        out[l.piles.at(0, l.discard_composition) + c] =
            (f64::from(state.discard[c]) / printed) as f32;
        out[l.piles.at(0, l.unseen) + c] = (f64::from(view.unseen[c]) / printed) as f32;
    }
}

fn opponents(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let k = board.n_card_types;
    let n_segments = board.n_segments;
    for slot in 0..OPPONENT_SLOTS {
        let Some(&seat) = view.order().get(slot) else {
            continue;
        };
        out[l.opponents.at(slot, l.present)] = 1.0;
        let trains = state.trains[seat];
        out[l.opponents.at(slot, l.opp_trains)] = (f64::from(trains) / view.trains_supply) as f32;
        thermometer(
            out,
            l.opponents.at(slot, l.trains_thermometer),
            u32::from(trains),
            &TRAIN_BUCKETS,
        );
        out[l.opponents.at(slot, l.opp_score)] =
            (f64::from(state.score[seat]) / SCORE_SCALE) as f32;
        out[l.opponents.at(slot, l.hand_size)] =
            (f64::from(state.hand_size(seat)) / f64::from(board.deck_size as u32)) as f32;
        out[l.opponents.at(slot, l.ticket_count)] = (f64::from(state.tickets[seat].count_ones())
            / f64::from(board.n_tickets as u32))
            as f32;
        out[l.opponents.at(slot, l.blind_draws)] =
            (f64::from(state.unknown[seat]) / f64::from(board.deck_size as u32)) as f32;
        let claimed = state.seg_owner[..n_segments]
            .iter()
            .filter(|&&o| o as usize == seat)
            .count();
        out[l.opponents.at(slot, l.segments_claimed)] = (claimed as f64 / n_segments as f64) as f32;
        let trail = view.derived.trails[seat];
        out[l.opponents.at(slot, l.longest_chain)] = (f64::from(trail) / view.trains_supply) as f32;
        for c in 0..k {
            let printed = f64::from(board.cards_per_type(c as u8));
            let certain = state.certain[seat * k + c];
            out[l.opponents.at(slot, l.certain) + c] = (f64::from(certain) / printed) as f32;
            // Everything I cannot see could in principle be theirs, on top of what I know.
            let possible =
                i32::from(certain) + i32::from(view.unseen[c]).min(i32::from(state.unknown[seat]));
            out[l.opponents.at(slot, l.max_possible) + c] = (f64::from(possible) / printed) as f32;
        }
    }
}

fn clock(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let n = state.n_players();
    out[l.clock.at(0, l.phase) + state.phase as usize] = 1.0;
    out[l.clock.at(0, l.seats) + (n - 2)] = 1.0;
    out[l.clock.at(0, l.draws_left)] = if state.phase == PHASE_DRAW_SECOND {
        1.0
    } else {
        0.0
    };

    let triggered = state.final_left != NOT_TRIGGERED;
    out[l.clock.at(0, l.final_triggered)] = if triggered { 1.0 } else { 0.0 };
    out[l.clock.at(0, l.final_countdown)] = if triggered {
        (f64::from(state.final_left) / n as f64) as f32
    } else {
        0.0
    };
    out[l.clock.at(0, l.turn)] = (f64::from(state.turn).min(TURN_CLIP) / TURN_CLIP) as f32;

    let trains = state.trains[view.me];
    out[l.clock.at(0, l.my_trains)] = (f64::from(trains) / view.trains_supply) as f32;
    thermometer(
        out,
        l.clock.at(0, l.my_trains_thermometer),
        u32::from(trains),
        &TRAIN_BUCKETS,
    );
    out[l.clock.at(0, l.my_score)] = (f64::from(state.score[view.me]) / SCORE_SCALE) as f32;
    out[l.clock.at(0, l.my_ticket_count)] =
        (f64::from(state.tickets[view.me].count_ones()) / f64::from(board.n_tickets as u32)) as f32;

    let scores = &state.score[..n];
    let leader = *scores.iter().max().expect("at least one seat");
    let mine = state.score[view.me];
    let behind = scores.iter().filter(|&&s| s > mine).count();
    out[l.clock.at(0, l.score_rank)] = (1.0 - behind as f64 / (n - 1).max(1) as f64) as f32;
    out[l.clock.at(0, l.gap_to_leader)] = (f64::from(leader - mine) / SCORE_SCALE) as f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::rng::{Part, stream};
    use crate::state::Game;

    #[test]
    fn every_board_has_a_spec_and_the_sizes_are_the_recorded_ones() {
        // 3355 on USA rather than the ~2000-2800 PLAN.md §6.1 estimated: the 100x9
        // required-colour block is the difference, and §6.2 needs it as an input for the
        // colour-symmetry regularizer.
        assert_eq!(spec_for("usa").size, 3355);
        assert_eq!(spec_for("mini").size, 1169);
    }

    #[test]
    fn the_layout_resolves_for_every_board() {
        // The name lookups happen once, at first use, and a typo in a field name would
        // otherwise panic the first time some rare feature was written.
        for board in crate::board::all_boards() {
            assert_eq!(layout_for(board.name).size, spec_for(board.name).size);
        }
    }

    #[test]
    fn double_rounding_does_not_bite_over_the_ranges_used() {
        // Checks the claim in this module's header: the cheaper direct-f32 form would
        // *also* agree today, so the reason to keep f64 is the guarantee rather than a
        // difference anyone can currently observe. If a new feature introduces a divisor
        // where these disagree, this test says so instead of one wrong slot out of 3355.
        let divisors: [f64; 8] = [12.0, 45.0, 20.0, 110.0, 54.0, 100.0, 30.0, 200.0];
        for &d in &divisors {
            for n in -400i32..=400 {
                let via_f64 = (f64::from(n) / d) as f32;
                let via_f32 = n as f32 / d as f32;
                assert_eq!(via_f64.to_bits(), via_f32.to_bits(), "{n}/{d}");
            }
        }
    }

    #[test]
    fn encoding_is_deterministic_and_leaves_the_position_alone() {
        // `encode` takes &mut State -- path halving and the fragility probe both mutate.
        // Both must restore what they touched, or the observation would silently change
        // the game it was observing. Mid-game, with tickets held, so the fragility probe's
        // close-and-restore of seg_owner actually runs; at setup nobody holds a ticket.
        let game = Game::new(RuleConfig::new("usa", 3).unwrap()).unwrap();
        let mut state = game.new_initial_state(5);
        let mut rng = stream(5, &[Part::Str("obs"), Part::Str("policy")]);
        for _ in 0..60 {
            let action = state.sample_legal(&mut rng).expect("not over yet");
            state.step(action).unwrap();
        }
        assert!(
            (0..3).all(|p| state.tickets[p] != 0),
            "every seat holds tickets"
        );
        let before = state.state_hash();
        let owners = state.seg_owner;
        let size = observation_size(&state);

        for player in 0..3 {
            let mut first = vec![0.0f32; size];
            encode(&mut state, player, &mut first);
            assert_eq!(state.state_hash(), before, "encoding moved the state");
            assert_eq!(
                state.seg_owner, owners,
                "the fragility probe left a segment closed"
            );

            let mut second = vec![0.0f32; size];
            encode(&mut state, player, &mut second);
            assert_eq!(first, second, "encoding is not deterministic");
        }
    }
}
