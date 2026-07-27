//! The observation encoder. The fast twin of `ticket_to_ride/rl/encode/observation.py`.
//!
//! Layout comes entirely from [`crate::obs_spec_gen`], generated from the same declarative
//! table as the Python oracle, so the two cannot disagree about *where* a feature goes --
//! only about the value written into a slot, which is exactly what the differential
//! harness checks (PLAN.md §8.2).
//!
//! **Arithmetic follows Python's, deliberately.** The output buffer is `f32` because the
//! oracle writes into an `array("f")`, but every intermediate is computed in `f64` and
//! narrowed on store, because Python computes in doubles and narrows on store. That makes
//! agreement a property of the construction rather than of the inputs.
//!
//! Dividing directly in `f32` would in fact agree over every divisor this encoder uses --
//! `double_rounding_does_not_bite_over_the_ranges_used` tests exactly that, so the claim
//! is checked rather than assumed. The point is that it agrees *by luck of the ranges*: a
//! new feature with a different divisor could double-round differently, and would do so
//! silently in one slot out of 3355. Matching Python's arithmetic costs nothing and
//! removes the question.
//!
//! One thing that reads oddly and is correct: this takes `&mut State`. Two of the features
//! need it -- the union-find roots are read through path halving, and `fragility` re-prices
//! a route with each of its free segments temporarily closed. Both restore what they
//! touched, and neither the DSU nor a restored `seg_owner` reaches any hash.

use std::collections::HashMap;

use crate::board::{CityId, GRAY, NO_SIBLING, SegmentId, UNREACHABLE};
use crate::config::{CLOSED, EMPTY_SLOT, FREE, NOT_TRIGGERED, PHASE_DRAW_SECOND};
use crate::graph::{MIN_TERMINALS, longest_trail, remaining_costs_from, steiner_cost_exact};
use crate::obs_spec_gen::{
    COST_BUCKETS, HAND_BUCKETS, OBS_SPECS, OBS_VERSION, OPPONENT_SLOTS, ObsBlock, ObsSpec,
    TRAIN_BUCKETS,
};
use crate::state::State;

/// Score and gap-to-leader are normalized by this; the scale is a rough game maximum, not
/// a bound, so the feature may exceed 1. Matches the oracle.
const SCORE_SCALE: f64 = 100.0;

/// Turn index is clipped here before normalizing. The analytical 4P bound is ~335 turns,
/// but a typical game is well under 200 and the clip keeps the feature informative.
const TURN_CLIP: f64 = 200.0;

/// The number of `own_hand` thermometer levels: "at least one" through "at least six".
const HAND_THERMOMETER_LEVELS: usize = 6;

pub fn spec_for(map: &str) -> &'static ObsSpec {
    OBS_SPECS
        .iter()
        .copied()
        .find(|s| s.map == map)
        .expect("every board has a generated observation spec")
}

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
/// The generated spec is a table of `&'static str` names; looking a field up by name on
/// every write would cost more than the encoding. This resolves them at first use and
/// caches per board.
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

fn block_of(spec: &'static ObsSpec, name: &str) -> &'static ObsBlock {
    spec.blocks
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("the generated spec has no block {name:?}"))
}

fn field_of(block: &'static ObsBlock, name: &str) -> usize {
    block
        .fields
        .iter()
        .find(|f| f.name == name)
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
    fn build(spec: &'static ObsSpec) -> Self {
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
            // The oracle names level 1 "thermometer" and the rest "thermometerN".
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
        OBS_SPECS
            .iter()
            .map(|spec| (spec.map, Layout::build(spec)))
            .collect()
    });
    &cache
        .iter()
        .find(|(name, _)| *name == map)
        .expect("every board has a resolved layout")
        .1
}

// ---------------------------------------------------------------------------
// Per-encode derived view
// ---------------------------------------------------------------------------

/// Everything the blocks need, computed once: seat order, costs, the Steiner tree.
struct View {
    me: usize,
    order: Vec<usize>,
    trains_supply: f64,
    locked: bool,
    dsu_root: Vec<u8>,
    held: Vec<u8>,
    ticket_cost: HashMap<u8, u16>,
    connected: HashMap<u8, bool>,
    dead: HashMap<u8, bool>,
    reach_cost: HashMap<CityId, Vec<u16>>,
    steiner: u32,
    steiner_exact: bool,
    steiner_edges: Vec<SegmentId>,
    unseen: Vec<i16>,
}

impl View {
    fn build(state: &mut State, player: usize) -> Self {
        let board = state.board;
        let n = state.n_players();
        let locked = state.rules.doubles_locked;

        // Opponents by distance in turn order: the seat that acts next is slot 0.
        let order: Vec<usize> = (0..n - 1).map(|i| (player + 1 + i) % n).collect();
        let dsu_root: Vec<u8> = (0..board.n_cities as u8)
            .map(|c| state.dsu_root(player, c))
            .collect();

        let held = state.tickets_of(player);
        let seg_owner = &state.seg_owner[..board.n_segments];
        let mut terminals: Vec<CityId> = Vec::new();
        let mut ticket_cost = HashMap::new();
        let mut connected = HashMap::new();
        let mut dead = HashMap::new();
        let mut reach_cost: HashMap<CityId, Vec<u16>> = HashMap::new();

        for &ticket in &held {
            let (a, b) = (
                board.ticket_a[ticket as usize],
                board.ticket_b[ticket as usize],
            );
            reach_cost
                .entry(a)
                .or_insert_with(|| remaining_costs_from(board, seg_owner, player as u8, a, locked));
            let cost = reach_cost[&a][b as usize];
            ticket_cost.insert(ticket, cost);
            connected.insert(ticket, cost == 0);
            dead.insert(ticket, cost >= UNREACHABLE);
            if cost > 0 && cost < UNREACHABLE {
                terminals.push(a);
                terminals.push(b);
            }
        }

        let (steiner, steiner_exact) =
            steiner_cost_exact(board, seg_owner, player as u8, &terminals, locked);

        let mut view = Self {
            me: player,
            order,
            trains_supply: f64::from(board.raw.trains_per_player),
            locked,
            dsu_root,
            held,
            ticket_cost,
            connected,
            dead,
            reach_cost,
            steiner,
            steiner_exact,
            steiner_edges: Vec::new(),
            unseen: state.unseen_counts(player),
        };
        view.steiner_edges = steiner_edges(&mut view, state, &terminals);
        view
    }

    /// The cached Dijkstra from `source`, computing it if this is the first ask.
    fn costs_from(&mut self, state: &State, source: CityId) -> &Vec<u16> {
        let board = state.board;
        let locked = self.locked;
        let me = self.me as u8;
        let seg_owner = &state.seg_owner[..board.n_segments];
        self.reach_cost
            .entry(source)
            .or_insert_with(|| remaining_costs_from(board, seg_owner, me, source, locked))
    }
}

/// Free segments on *a* cheapest route between consecutive terminals.
///
/// An approximation of "on my Steiner tree", and deliberately so: the exact edge set of a
/// Dreyfus-Wagner solution needs back-pointers through the subset DP, which is a lot of
/// machinery for a boolean hint. What the network needs is "this route is on my plan", and
/// a union of cheapest paths gives that.
fn steiner_edges(view: &mut View, state: &State, terminals: &[CityId]) -> Vec<SegmentId> {
    if terminals.len() < MIN_TERMINALS {
        return Vec::new();
    }
    let board = state.board;
    let mut edges: Vec<SegmentId> = Vec::new();
    for pair in terminals.chunks_exact(2) {
        let (source, target) = (pair[0], pair[1]);
        let dist = view.costs_from(state, source).clone();
        if dist[target as usize] >= UNREACHABLE {
            continue;
        }
        let mut here = target;
        let mut guard = 0;
        while here != source && guard <= board.n_cities {
            guard += 1;
            let mut stepped = false;
            for &(nb, segment) in &board.adjacency[here as usize] {
                let owner = state.seg_owner[segment as usize];
                if owner != FREE && owner != view.me as u8 {
                    continue;
                }
                let weight = if owner == view.me as u8 {
                    0
                } else {
                    u16::from(board.seg_len[segment as usize])
                };
                if dist[nb as usize] + weight == dist[here as usize] {
                    if owner == FREE && !edges.contains(&segment) {
                        edges.push(segment);
                    }
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
    edges
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encode `state` from `player`'s point of view into `out`.
///
/// Opponents are ordered by **distance in turn order**, so slot 0 is whoever acts next.
/// Seat-relative rather than permutation-invariant, which is the right prior: in Ticket to
/// Ride, who moves before you is decisive.
///
/// # Panics
/// If `out` is not exactly the spec's size for this board.
pub fn encode(state: &mut State, player: usize, out: &mut [f32]) {
    let layout = layout_for(state.board.name);
    assert_eq!(
        out.len(),
        layout.size,
        "buffer holds {} floats, spec needs {}",
        out.len(),
        layout.size
    );
    out.fill(0.0);

    let mut view = View::build(state, player);
    segment_static(&view, state, layout, out);
    segment_dynamic(&view, state, layout, out);
    own_hand(&view, state, layout, out);
    tickets(&mut view, state, layout, out);
    steiner(&view, state, layout, out);
    faceup(&view, state, layout, out);
    piles(&view, state, layout, out);
    opponents(&view, state, layout, out);
    clock(&view, state, layout, out);
}

#[inline]
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
    let mut seat_slot = [0usize; crate::board::MAX_PLAYERS];
    for (i, &seat) in view.order.iter().enumerate() {
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

        out[l.seg_dyn.at(segment, l.on_my_steiner_tree)] =
            if view.steiner_edges.contains(&(segment as SegmentId)) {
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

fn tickets(view: &mut View, state: &mut State, l: &Layout, out: &mut [f32]) {
    let board = state.board;
    let max_points = f64::from(*board.ticket_points.iter().max().expect("tickets exist"));
    let held = view.held.clone();
    for ticket in 0..board.n_tickets {
        out[l.tickets.at(ticket, l.points)] =
            (f64::from(board.ticket_points[ticket]) / max_points) as f32;
        if !held.contains(&(ticket as u8)) {
            continue;
        }
        let t = ticket as u8;
        out[l.tickets.at(ticket, l.held)] = 1.0;
        out[l.tickets.at(ticket, l.connected)] = if view.connected[&t] { 1.0 } else { 0.0 };
        out[l.tickets.at(ticket, l.is_dead)] = if view.dead[&t] { 1.0 } else { 0.0 };
        if view.dead[&t] {
            continue;
        }
        let cost = view.ticket_cost[&t];
        out[l.tickets.at(ticket, l.ticket_remaining_cost)] =
            (f64::from(cost) / view.trains_supply) as f32;
        thermometer(
            out,
            l.tickets.at(ticket, l.cost_thermometer),
            u32::from(cost),
            &COST_BUCKETS,
        );
        out[l.tickets.at(ticket, l.fragility)] =
            (fragility(view, state, t) / view.trains_supply) as f32;
    }
}

/// Worst extra cost if one enemy claim lands on my cheapest route for this ticket.
///
/// Measured by re-pricing the route with each of its free segments taken in turn. That is
/// a handful of Dijkstras per ticket, which separates a safe connection from one hanging
/// on a single contested edge.
fn fragility(view: &mut View, state: &mut State, ticket: u8) -> f64 {
    let board = state.board;
    let (a, b) = (
        board.ticket_a[ticket as usize],
        board.ticket_b[ticket as usize],
    );
    let baseline = view.ticket_cost[&ticket];
    if baseline == 0 || baseline >= UNREACHABLE {
        return 0.0;
    }

    let route = cheapest_route(view, state, a, b);
    let on_path: Vec<SegmentId> = route
        .into_iter()
        .filter(|&s| state.seg_owner[s as usize] == FREE)
        .collect();

    let mut worst = 0u16;
    for segment in on_path {
        state.seg_owner[segment as usize] = CLOSED;
        let cost = remaining_costs_from(
            board,
            &state.seg_owner[..board.n_segments],
            view.me as u8,
            a,
            view.locked,
        )[b as usize];
        state.seg_owner[segment as usize] = FREE;
        if cost >= UNREACHABLE {
            return view.trains_supply;
        }
        worst = worst.max(cost - baseline);
    }
    f64::from(worst)
}

/// One cheapest route from `source` to `target`, as segment ids.
fn cheapest_route(
    view: &mut View,
    state: &State,
    source: CityId,
    target: CityId,
) -> Vec<SegmentId> {
    let board = state.board;
    let dist = view.costs_from(state, source).clone();
    if dist[target as usize] >= UNREACHABLE {
        return Vec::new();
    }
    let mut path = Vec::new();
    let mut here = target;
    let mut guard = 0;
    while here != source && guard <= board.n_cities {
        guard += 1;
        let mut stepped = false;
        for &(nb, segment) in &board.adjacency[here as usize] {
            let owner = state.seg_owner[segment as usize];
            if owner != FREE && owner != view.me as u8 {
                continue;
            }
            let weight = if owner == view.me as u8 {
                0
            } else {
                u16::from(board.seg_len[segment as usize])
            };
            if dist[nb as usize] + weight == dist[here as usize] {
                path.push(segment);
                here = nb;
                stepped = true;
                break;
            }
        }
        if !stepped {
            break;
        }
    }
    path
}

fn steiner(view: &View, state: &State, l: &Layout, out: &mut [f32]) {
    let cost = f64::from(view.steiner.min((view.trains_supply as u32) * 2));
    out[l.steiner.at(0, l.steiner_remaining_cost)] = (cost / view.trains_supply) as f32;
    let slack = f64::from(state.trains[view.me]) - cost;
    out[l.steiner.at(0, l.cost_minus_trains)] = (slack / view.trains_supply) as f32;
    out[l.steiner.at(0, l.exact)] = if view.steiner_exact { 1.0 } else { 0.0 };
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
        let Some(&seat) = view.order.get(slot) else {
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
        let trail = longest_trail(board, &state.seg_owner[..n_segments], seat as u8);
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
        for spec in OBS_SPECS {
            let layout = layout_for(spec.map);
            assert_eq!(layout.size, spec.size);
        }
    }

    #[test]
    fn double_rounding_does_not_bite_over_the_ranges_used() {
        // Checks the claim in this module's header. Computing in f64 and narrowing is what
        // makes agreement with Python structural; this asserts that the cheaper direct-f32
        // form would *also* agree today, so the reason to keep f64 is the guarantee rather
        // than a difference anyone can currently observe. If a new feature introduces a
        // divisor where these disagree, this test says so instead of the harness finding
        // one wrong slot out of 3355.
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
        // the game it was observing.
        let game = Game::new(RuleConfig::new("usa", 3).unwrap()).unwrap();
        let mut state = game.new_initial_state(5);
        let before = state.state_hash();
        let size = observation_size(&state);

        let mut first = vec![0.0f32; size];
        encode(&mut state, 0, &mut first);
        assert_eq!(state.state_hash(), before, "encoding moved the state");

        let mut second = vec![0.0f32; size];
        encode(&mut state, 0, &mut second);
        assert_eq!(first, second, "encoding is not deterministic");
        assert_eq!(state.state_hash(), before);
    }

    #[test]
    fn the_buffer_is_reused_rather_than_appended_to() {
        // `encode` zeroes first. Without that, a reused buffer would carry stale one-hots
        // from the previous position -- which looks like a plausible observation and is
        // wrong in exactly the features that changed.
        let game = Game::new(RuleConfig::new("mini", 2).unwrap()).unwrap();
        let mut state = game.new_initial_state(1);
        let size = observation_size(&state);
        let mut buffer = vec![7.0f32; size];
        encode(&mut state, 0, &mut buffer);
        let mut fresh = vec![0.0f32; size];
        encode(&mut state, 0, &mut fresh);
        assert_eq!(buffer, fresh);
    }
}
