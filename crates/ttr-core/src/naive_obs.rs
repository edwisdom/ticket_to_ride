//! A deliberately naive observation encoder, for differential testing inside the crate.
//!
//! PLAN.md §14.1 step 4, extended to the encoder after mutation testing showed most of
//! `obs.rs` was checked only by comparison with a frozen copy (`oracle-v1`, the golden
//! corpus): replacing `own_hand` with nothing, or a normalizer's `/` with `*`, survived
//! every explicit test. Those comparisons stop applying the day `OBS_VERSION` moves; this
//! does not.
//!
//! Every feature is recomputed here straight from the state and the spec's description of
//! it -- one line per feature, the union-find replaced by a flood fill, remaining costs by
//! Bellman-Ford -- and compared with [`crate::obs::encode`] bit for bit, at every few steps
//! of random and H3 games on every map and seat count, for every seat.
//!
//! Two features are left out: `on_my_steiner_tree` and `fragility` both read the encoder's
//! route-recovery walk, which has a known `OBS_VERSION` 1 defect (it can ping-pong across
//! zero-weight own track); `tests/obs_plan.rs` states what holds of them. The Steiner cost
//! comes from [`crate::graph::steiner_cost_exact`] and the longest trail from
//! [`crate::graph::longest_trail`], both checked against brute force in `naive_graph`.
//!
//! **Known `OBS_VERSION` 1 quirks, reproduced on purpose** (each marked `v1 quirk` below):
//! `extends_my_chain` misses segments at the root city of my network, and the hand counts
//! are normalized by the colour card count even for the locomotive. Fixing either is a
//! named `OBS_VERSION` bump, and this reference changes with it.
#![allow(clippy::needless_range_loop)]

use crate::board::{GRAY, NO_SIBLING, UNREACHABLE};
use crate::config::{CLOSED, EMPTY_SLOT, FREE, NO_TICKET, NOT_TRIGGERED, PHASE_DRAW_SECOND};
use crate::graph::{longest_trail, steiner_cost_exact};
use crate::obs::spec_for;
use crate::obs_spec::{COST_BUCKETS, HAND_BUCKETS, OPPONENT_SLOTS, TRAIN_BUCKETS};
use crate::state::State;

/// Offsets by name, from the spec (whose layout is pinned separately).
struct Lookup {
    map: &'static str,
}

impl Lookup {
    fn at(&self, block: &str, entity: usize, field: &str) -> usize {
        let spec = spec_for(self.map);
        let b = spec
            .blocks
            .iter()
            .find(|b| b.name == block)
            .unwrap_or_else(|| panic!("no block {block}"));
        let f = b
            .fields
            .iter()
            .find(|f| f.name == field)
            .unwrap_or_else(|| panic!("no field {block}.{field}"));
        b.offset + entity * b.stride + f.offset
    }

    fn width(&self, block: &str, field: &str) -> usize {
        let spec = spec_for(self.map);
        let b = spec.blocks.iter().find(|b| b.name == block).unwrap();
        b.fields.iter().find(|f| f.name == field).unwrap().width
    }
}

fn ratio(numerator: f64, denominator: f64) -> f32 {
    (numerator / denominator) as f32
}

fn flag(on: bool) -> f32 {
    if on { 1.0 } else { 0.0 }
}

/// Cost to me of crossing `segment`: free if mine, its length if free and claimable by me,
/// impassable otherwise (anyone else's, closed, or the twin of my own track at 4-5 seats).
fn crossing_cost(state: &State, me: u8, segment: usize) -> u32 {
    let board = state.board;
    let owner = state.seg_owner[segment];
    if owner == me {
        return 0;
    }
    if owner != FREE {
        return u32::from(UNREACHABLE);
    }
    let twin = board.sibling[segment];
    if !state.rules.doubles_locked && twin != NO_SIBLING && state.seg_owner[usize::from(twin)] == me
    {
        return u32::from(UNREACHABLE);
    }
    u32::from(board.seg_len[segment])
}

/// Bellman-Ford from `source` over [`crossing_cost`], capped at `UNREACHABLE`.
fn costs_from(state: &State, me: u8, source: u8) -> Vec<u32> {
    let board = state.board;
    let mut dist = vec![u32::from(UNREACHABLE); board.n_cities];
    dist[usize::from(source)] = 0;
    for _ in 0..board.n_cities {
        let mut changed = false;
        for s in 0..board.n_segments {
            let w = crossing_cost(state, me, s);
            if w >= u32::from(UNREACHABLE) {
                continue;
            }
            let (a, b) = (usize::from(board.seg_a[s]), usize::from(board.seg_b[s]));
            for (x, y) in [(a, b), (b, a)] {
                if dist[x] + w < dist[y] {
                    dist[y] = dist[x] + w;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    dist
}

/// The cities of my network's components, by flood fill over my own track. Returns, per
/// city, the smallest city index in its component, or `None` if I own no track there.
fn my_components(state: &State, me: u8) -> Vec<Option<u8>> {
    let board = state.board;
    let mut label: Vec<Option<u8>> = vec![None; board.n_cities];
    for start in 0..board.n_cities {
        if label[start].is_some() {
            continue;
        }
        let mut members = vec![start];
        let mut frontier = vec![start];
        while let Some(city) = frontier.pop() {
            for s in 0..board.n_segments {
                if state.seg_owner[s] != me {
                    continue;
                }
                let (a, b) = (usize::from(board.seg_a[s]), usize::from(board.seg_b[s]));
                let other = if a == city {
                    b
                } else if b == city {
                    a
                } else {
                    continue;
                };
                if !members.contains(&other) {
                    members.push(other);
                    frontier.push(other);
                }
            }
        }
        if members.len() > 1 {
            let smallest = *members.iter().min().unwrap() as u8;
            for &m in &members {
                label[m] = Some(smallest);
            }
        }
    }
    label
}

/// What `observer` cannot account for: the deck plus everyone else's blind draws.
fn unseen(state: &State, observer: usize) -> Vec<i32> {
    let board = state.board;
    let k = board.n_card_types;
    (0..k)
        .map(|c| {
            let mut n = i32::from(board.deck_composition_counts[c]);
            n -= i32::from(state.hand[observer * k + c]);
            n -= i32::from(state.discard[c]);
            n -= state.faceup.iter().filter(|&&f| f == c as i8).count() as i32;
            for p in 0..state.n_players() {
                if p != observer {
                    n -= i32::from(state.certain[p * k + c]);
                }
            }
            n
        })
        .collect()
}

/// The observation, recomputed. Slots of the two left-out features stay zero.
pub(crate) fn naive_encode(state: &State, player: usize) -> Vec<f32> {
    let board = state.board;
    let at = Lookup { map: board.name };
    let mut out = vec![0.0f32; spec_for(board.name).size];
    let n = state.n_players();
    let me = player as u8;
    let k = board.n_card_types;
    let supply = f64::from(board.raw.trains_per_player);
    let deck = board.deck_size as f64;
    let printed = |c: usize| f64::from(board.deck_composition_counts[c]);
    // Opponents by distance in turn order: whoever acts next is slot 0.
    let order: Vec<usize> = (1..n).map(|i| (player + i) % n).collect();

    // -- segments -----------------------------------------------------------------------
    let hand = &state.hand[player * k..(player + 1) * k];
    let wilds = hand[usize::from(board.locomotive)];
    let components = my_components(state, me);
    for s in 0..board.n_segments {
        let color = board.seg_color[s];
        let column = if color == GRAY {
            board.n_colors
        } else {
            usize::from(color)
        };
        out[at.at("segment_static", s, "required_color") + column] = 1.0;

        let owner = state.seg_owner[s];
        let owner_slot = if owner == FREE || owner == CLOSED {
            0
        } else if owner == me {
            1
        } else {
            2 + order.iter().position(|&o| o == usize::from(owner)).unwrap()
        };
        out[at.at("segment_dynamic", s, "owner") + owner_slot] = 1.0;
        out[at.at("segment_dynamic", s, "closed")] = flag(owner == CLOSED);
        let twin = board.sibling[s];
        out[at.at("segment_dynamic", s, "twin_locked")] =
            flag(twin != NO_SIBLING && state.seg_owner[usize::from(twin)] == me);

        // The most of this route I could pay for now: my best colour plus every wild, or
        // the wilds alone; a colour I hold none of contributes nothing.
        let length = board.seg_len[s];
        let reach = |c: usize| if hand[c] > 0 { hand[c] + wilds } else { 0 };
        let best = if color == GRAY {
            (0..board.n_colors).map(reach).max().unwrap_or(0)
        } else {
            reach(usize::from(color))
        }
        .max(wilds);
        let short = length.saturating_sub(best);
        out[at.at("segment_dynamic", s, "can_afford_now")] =
            flag(owner == FREE && short == 0 && state.trains[player] >= length);
        for (i, &edge) in HAND_BUCKETS.iter().enumerate() {
            out[at.at("segment_dynamic", s, "cards_short") + i] = flag(short >= edge);
        }
        // v1 quirk: "touches my network" is read as "is not its component's root", so a
        // segment at the smallest-index city of a component reads as not extending it.
        let (a, b) = (usize::from(board.seg_a[s]), usize::from(board.seg_b[s]));
        let off_root = |city: usize| components[city].is_some_and(|root| usize::from(root) != city);
        out[at.at("segment_dynamic", s, "extends_my_chain")] =
            flag(owner == FREE && (off_root(a) || off_root(b)));
    }

    // -- my hand ------------------------------------------------------------------------
    for c in 0..k {
        // v1 quirk: normalized by the colour count even for the locomotive.
        out[at.at("own_hand", 0, "counts") + c] =
            ratio(f64::from(hand[c]), f64::from(board.raw.cards_per_color));
        for level in 1..=6u8 {
            let name = if level == 1 {
                "thermometer".to_string()
            } else {
                format!("thermometer{level}")
            };
            out[at.at("own_hand", 0, &name) + c] = flag(hand[c] >= level);
        }
    }

    // -- tickets --------------------------------------------------------------------------
    let max_points = f64::from(*board.ticket_points.iter().max().unwrap());
    let mut terminals: Vec<u8> = Vec::new();
    for t in 0..board.n_tickets {
        out[at.at("tickets", t, "points")] = ratio(f64::from(board.ticket_points[t]), max_points);
        if state.tickets[player] >> t & 1 == 0 {
            continue;
        }
        let (a, b) = (board.ticket_a[t], board.ticket_b[t]);
        let cost = costs_from(state, me, a)[usize::from(b)];
        out[at.at("tickets", t, "held")] = 1.0;
        out[at.at("tickets", t, "connected")] = flag(cost == 0);
        let dead = cost >= u32::from(UNREACHABLE);
        out[at.at("tickets", t, "is_dead")] = flag(dead);
        if dead {
            continue;
        }
        out[at.at("tickets", t, "remaining_cost")] = ratio(f64::from(cost), supply);
        for (i, &edge) in COST_BUCKETS.iter().enumerate() {
            out[at.at("tickets", t, "cost_thermometer") + i] = flag(cost >= u32::from(edge));
        }
        if cost > 0 {
            terminals.push(a);
            terminals.push(b);
        }
    }

    // -- the tree over every open ticket --------------------------------------------------
    let (tree, exact) = steiner_cost_exact(
        board,
        &state.seg_owner[..board.n_segments],
        me,
        &terminals,
        state.rules.doubles_locked,
    );
    let capped = f64::from(tree.min(2 * u32::from(board.raw.trains_per_player)));
    out[at.at("steiner", 0, "remaining_cost")] = ratio(capped, supply);
    out[at.at("steiner", 0, "cost_minus_trains")] =
        ratio(f64::from(state.trains[player]) - capped, supply);
    out[at.at("steiner", 0, "exact")] = flag(exact);

    // -- the table ------------------------------------------------------------------------
    for (slot, &card) in state.faceup.iter().enumerate() {
        let column = if card == EMPTY_SLOT { k } else { card as usize };
        out[at.at("faceup", slot, "card") + column] = 1.0;
    }
    let discarded: u32 = state.discard[..k].iter().map(|&c| u32::from(c)).sum();
    let tickets_left = state.tdeck[..board.n_tickets]
        .iter()
        .filter(|&&t| t != NO_TICKET)
        .count();
    let hidden = unseen(state, player);
    out[at.at("piles", 0, "deck_size")] = ratio(f64::from(state.deck_len - state.deck_pos), deck);
    out[at.at("piles", 0, "discard_size")] = ratio(f64::from(discarded), deck);
    out[at.at("piles", 0, "ticket_deck_size")] = ratio(tickets_left as f64, board.n_tickets as f64);
    for c in 0..k {
        out[at.at("piles", 0, "discard_composition") + c] =
            ratio(f64::from(state.discard[c]), printed(c));
        out[at.at("piles", 0, "unseen") + c] = ratio(f64::from(hidden[c]), printed(c));
    }

    // -- opponents ------------------------------------------------------------------------
    assert!(order.len() <= OPPONENT_SLOTS);
    for (slot, &seat) in order.iter().enumerate() {
        out[at.at("opponents", slot, "present")] = 1.0;
        let trains = state.trains[seat];
        out[at.at("opponents", slot, "trains")] = ratio(f64::from(trains), supply);
        for (i, &edge) in TRAIN_BUCKETS.iter().enumerate() {
            out[at.at("opponents", slot, "trains_thermometer") + i] = flag(trains >= edge);
        }
        out[at.at("opponents", slot, "score")] = ratio(f64::from(state.score[seat]), 100.0);
        let held: u32 = state.hand[seat * k..(seat + 1) * k]
            .iter()
            .map(|&c| u32::from(c))
            .sum();
        out[at.at("opponents", slot, "hand_size")] = ratio(f64::from(held), deck);
        out[at.at("opponents", slot, "ticket_count")] = ratio(
            f64::from(state.tickets[seat].count_ones()),
            board.n_tickets as f64,
        );
        out[at.at("opponents", slot, "blind_draws")] = ratio(f64::from(state.unknown[seat]), deck);
        let claimed = (0..board.n_segments)
            .filter(|&s| state.seg_owner[s] == seat as u8)
            .count();
        out[at.at("opponents", slot, "segments_claimed")] =
            ratio(claimed as f64, board.n_segments as f64);
        let trail = longest_trail(board, &state.seg_owner[..board.n_segments], seat as u8);
        out[at.at("opponents", slot, "longest_chain")] = ratio(f64::from(trail), supply);
        for c in 0..k {
            let certain = i32::from(state.certain[seat * k + c]);
            out[at.at("opponents", slot, "certain") + c] = ratio(f64::from(certain), printed(c));
            let possible = certain + hidden[c].min(i32::from(state.unknown[seat]));
            out[at.at("opponents", slot, "max_possible") + c] =
                ratio(f64::from(possible), printed(c));
        }
    }

    // -- the clock -------------------------------------------------------------------------
    out[at.at("clock", 0, "phase") + usize::from(state.phase)] = 1.0;
    assert!(n - 2 < at.width("clock", "seats"));
    out[at.at("clock", 0, "seats") + (n - 2)] = 1.0;
    out[at.at("clock", 0, "draws_left")] = flag(state.phase == PHASE_DRAW_SECOND);
    let triggered = state.final_left != NOT_TRIGGERED;
    out[at.at("clock", 0, "final_triggered")] = flag(triggered);
    if triggered {
        out[at.at("clock", 0, "final_countdown")] = ratio(f64::from(state.final_left), n as f64);
    }
    out[at.at("clock", 0, "turn")] = ratio(f64::from(state.turn.min(200)), 200.0);
    let mine = state.trains[player];
    out[at.at("clock", 0, "my_trains")] = ratio(f64::from(mine), supply);
    for (i, &edge) in TRAIN_BUCKETS.iter().enumerate() {
        out[at.at("clock", 0, "my_trains_thermometer") + i] = flag(mine >= edge);
    }
    out[at.at("clock", 0, "my_score")] = ratio(f64::from(state.score[player]), 100.0);
    out[at.at("clock", 0, "my_ticket_count")] = ratio(
        f64::from(state.tickets[player].count_ones()),
        board.n_tickets as f64,
    );
    let scores = &state.score[..n];
    let ahead = scores.iter().filter(|&&s| s > state.score[player]).count();
    out[at.at("clock", 0, "score_rank")] = (1.0 - ahead as f64 / (n - 1).max(1) as f64) as f32;
    let leader = *scores.iter().max().unwrap();
    out[at.at("clock", 0, "gap_to_leader")] = ratio(f64::from(leader - state.score[player]), 100.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::heuristic::{Heuristic, HeuristicParams, Tier};
    use crate::obs::encode;
    use crate::rng::{Part, stream};
    use crate::state::Game;

    fn configs() -> Vec<(&'static str, usize)> {
        crate::board::all_boards()
            .iter()
            .flat_map(|b| (b.raw.min_players..=b.raw.max_players).map(|n| (b.name, usize::from(n))))
            .collect()
    }

    /// Slots of the two features this reference leaves out.
    fn left_out(map: &'static str) -> Vec<bool> {
        let at = Lookup { map };
        let spec = spec_for(map);
        let mut skip = vec![false; spec.size];
        let board = crate::board::get_board(map).unwrap();
        for s in 0..board.n_segments {
            skip[at.at("segment_dynamic", s, "on_my_steiner_tree")] = true;
        }
        for t in 0..board.n_tickets {
            skip[at.at("tickets", t, "fragility")] = true;
        }
        skip
    }

    /// `block[entity].field[i]` for a slot, for failure messages.
    fn name_of(map: &str, slot: usize) -> String {
        for b in spec_for(map).blocks.iter() {
            if slot >= b.offset && slot < b.offset + b.count * b.stride {
                let (entity, inside) = ((slot - b.offset) / b.stride, (slot - b.offset) % b.stride);
                for f in b.fields.iter() {
                    if inside >= f.offset && inside < f.offset + f.width {
                        return format!("{}[{entity}].{}[{}]", b.name, f.name, inside - f.offset);
                    }
                }
            }
        }
        "?".into()
    }

    /// Compare every seat's observation at this position; returns how many were compared.
    fn check(state: &mut State, skip: &[bool], context: &str) -> usize {
        let size = skip.len();
        let mut fast = vec![0.0f32; size];
        for player in 0..state.n_players() {
            let naive = naive_encode(state, player);
            encode(state, player, &mut fast);
            for slot in 0..size {
                if skip[slot] {
                    continue;
                }
                assert_eq!(
                    fast[slot].to_bits(),
                    naive[slot].to_bits(),
                    "{context} seat {player}: slot {slot} ({}) encoder {} naive {}",
                    name_of(state.board.name, slot),
                    fast[slot],
                    naive[slot]
                );
            }
        }
        state.n_players()
    }

    fn sweep(seeds: std::ops::Range<u64>, every: usize) -> usize {
        let mut compared = 0;
        for (map, n) in configs() {
            let game = Game::new(RuleConfig::new(map, n).unwrap()).unwrap();
            let skip = left_out(map);
            for seed in seeds.clone() {
                // Random play reaches the corners; H3 reaches positions a player would.
                for heuristic in [false, true] {
                    let mut state = game.new_initial_state(seed);
                    let mut rng = stream(seed, &[Part::Str("naive-obs")]);
                    let mut agents: Vec<Heuristic> = (0..n)
                        .map(|s| {
                            let mut h = Heuristic::new(Tier::H3, HeuristicParams::default(), 0);
                            h.begin_game(s, seed);
                            h
                        })
                        .collect();
                    let mut step = 0;
                    loop {
                        if step % every == 0 || state.is_terminal() {
                            let context =
                                format!("{map} {n}P seed {seed} h3={heuristic} step {step}");
                            compared += check(&mut state, &skip, &context);
                        }
                        if state.is_terminal() {
                            break;
                        }
                        let action = if heuristic {
                            let seat = usize::from(state.current_player());
                            agents[seat].act(&mut state)
                        } else {
                            state.sample_legal(&mut rng).unwrap()
                        };
                        state.step(action).unwrap();
                        step += 1;
                    }
                }
            }
        }
        compared
    }

    #[test]
    fn the_encoder_matches_a_naive_reference_everywhere() {
        let compared = sweep(0..2, 11);
        assert!(compared > 1_000, "only {compared} observations compared");
    }

    #[test]
    #[ignore = "sweep; run with --release -- --ignored"]
    fn the_encoder_matches_a_naive_reference_sweep() {
        let compared = sweep(0..60, 3);
        assert!(compared > 100_000, "only {compared} observations compared");
    }

    #[test]
    fn the_reference_can_disagree() {
        // Guard the guard: a reference that encodes a different seat must not match.
        let game = Game::new(RuleConfig::new("usa", 3).unwrap()).unwrap();
        let mut state = game.new_initial_state(4);
        let mut rng = stream(4, &[Part::Str("naive-obs")]);
        for _ in 0..40 {
            let a = state.sample_legal(&mut rng).unwrap();
            state.step(a).unwrap();
        }
        let mut fast = vec![0.0f32; spec_for("usa").size];
        encode(&mut state, 0, &mut fast);
        assert_ne!(naive_encode(&state, 1), fast);
    }
}
