//! The observation feature spec and the encoder.
//!
//! Ported one-to-one from `tests/unit/rl/test_observation.py` (PLAN.md §14.1). What matters
//! here is that the layout is exactly what the spec says, and that every feature actually
//! moves when the thing it describes changes.
//!
//! The Rust spec is plain data, so features are read by name through two small extension
//! traits standing in for the Python spec's `block(name)` and `at(entity, field)`.

mod common;

use std::collections::HashSet;

use common::{
    Rig, all_configs, cards, city, color, force, game, give, policy, segment_named, to_main,
};
use ttr_core::board::{Board, FACEUP_SLOTS, MAX_PLAYERS, NO_SIBLING, all_boards};
use ttr_core::config::{CLOSED, PHASE_MAIN};
use ttr_core::obs::{encode, obs_version, observation_size, spec_for};
use ttr_core::obs_spec::{OPPONENT_SLOTS, ObsBlock, ObsSpec};
use ttr_core::state::State;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The Python spec's `block(name)`, panicking on a typo rather than returning `None`.
trait SpecExt {
    fn named(&self, name: &str) -> &ObsBlock;
}

impl SpecExt for ObsSpec {
    fn named(&self, name: &str) -> &ObsBlock {
        self.block(name)
            .unwrap_or_else(|| panic!("the {} spec has no block {name:?}", self.map))
    }
}

/// The Python spec's `at(entity, field)`: the absolute offset of `field` for `entity`.
trait BlockExt {
    fn at(&self, entity: usize, field: &str) -> usize;
}

impl BlockExt for ObsBlock {
    fn at(&self, entity: usize, field: &str) -> usize {
        let found = self
            .fields
            .iter()
            .find(|f| f.name == field)
            .unwrap_or_else(|| panic!("{} has no field {field:?}", self.name));
        assert!(
            entity < self.count,
            "{} repeats over {} entities; {entity} is out of range",
            self.name,
            self.count
        );
        self.offset + entity * self.stride + found.offset
    }
}

/// `state` encoded from `player`'s seat into a fresh buffer.
fn observe(state: &mut State, player: usize) -> Vec<f32> {
    let mut out = vec![0.0; observation_size(state)];
    encode(state, player, &mut out);
    out
}

/// The first slot where two observations differ bitwise, if any.
fn first_difference(a: &[f32], b: &[f32]) -> Option<usize> {
    assert_eq!(a.len(), b.len(), "observations of different sizes");
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

/// USA at `n_players`, seed 1, with the opening ticket choices played out.
fn in_main(n_players: usize) -> State {
    let mut state = game("usa", n_players).new_initial_state(1);
    to_main(&mut state);
    state
}

/// A mid-game position: USA 3P, seed 1, sixty uniformly random legal actions in.
fn played() -> State {
    let mut state = game("usa", 3).new_initial_state(1);
    let mut rng = policy(1);
    for _ in 0..60 {
        if state.is_terminal() {
            break;
        }
        let action = state.sample_legal(&mut rng).expect("a non-terminal state");
        state.step(action).expect("a sampled action is legal");
    }
    state
}

/// Seat 0 to act, holding exactly `ticket`; seat 1 holds none.
fn hold_only(state: &mut State, ticket: usize) {
    force(
        state,
        Rig {
            tickets: Some(vec![1 << ticket, 0]),
            cur: Some(0),
            ..Rig::default()
        },
    );
}

/// The ticket with the shortest route on the open board; the first of any tie.
fn shortest_ticket(board: &Board) -> usize {
    (0..board.n_tickets)
        .min_by_key(|&t| board.dist[usize::from(board.ticket_a[t])][usize::from(board.ticket_b[t])])
        .expect("the board has tickets")
}

/// One hop from `here` along a shortest open-board route to `target`:
/// `(next city, segment)`, the first such hop in adjacency order.
fn step_towards(board: &Board, here: u8, target: u8) -> (u8, usize) {
    let &(next, segment) = board.adjacency[usize::from(here)]
        .iter()
        .min_by_key(|&&(nb, s)| {
            u16::from(board.seg_len[usize::from(s)])
                + board.dist[usize::from(nb)][usize::from(target)]
        })
        .expect("every city has track");
    (next, usize::from(segment))
}

// ---------------------------------------------------------------------------
// The spec table
// ---------------------------------------------------------------------------

/// No gaps, no overlaps: every float belongs to exactly one field of one block.
#[test]
fn blocks_tile_the_vector_exactly() {
    for board in all_boards() {
        let spec = spec_for(board.name);
        let mut cursor = 0;
        for block in &spec.blocks {
            assert_eq!(
                block.offset, cursor,
                "{}: {} starts at {}, not {cursor}",
                spec.map, block.name, block.offset
            );
            let widths: usize = block.fields.iter().map(|f| f.width).sum();
            assert_eq!(widths, block.stride, "{}: {}", spec.map, block.name);
            let mut field_cursor = 0;
            for field in &block.fields {
                assert_eq!(
                    field.offset, field_cursor,
                    "{}: {}.{}",
                    spec.map, block.name, field.name
                );
                field_cursor += field.width;
            }
            cursor += block.count * block.stride;
        }
        assert_eq!(cursor, spec.size, "{}", spec.map);
    }
}

#[test]
fn field_offsets_resolve_for_every_entity() {
    for board in all_boards() {
        let spec = spec_for(board.name);
        for block in &spec.blocks {
            for entity in 0..block.count {
                for field in &block.fields {
                    let offset = block.at(entity, field.name);
                    let place = format!("{}: {}[{entity}].{}", spec.map, block.name, field.name);
                    assert!(offset < spec.size, "{place} starts past the end");
                    assert!(
                        offset + field.width <= spec.size,
                        "{place} runs past the end"
                    );
                }
            }
        }
    }
}

/// Python built and cached one resolved spec per board, and so does Rust (resolved once at
/// first use); what carries over is that the lookup is per board, the spec names its board,
/// and asking twice hands back the same resolved table.
#[test]
fn spec_is_cached_per_board() {
    for board in all_boards() {
        assert_eq!(spec_for(board.name).map, board.name);
    }
    assert_eq!(spec_for("usa").map, "usa");
    assert_eq!(spec_for("mini").map, "mini");
    assert!(std::ptr::eq(spec_for("usa"), spec_for("usa")));
}

/// Block names are unique within a spec, field names within a block, and every field has a
/// doc string -- the doc is the only description of what a feature means.
#[test]
fn block_names_are_unique_and_documented() {
    for spec in all_boards().iter().map(|b| spec_for(b.name)) {
        let names: HashSet<&str> = spec.blocks.iter().map(|b| b.name).collect();
        assert_eq!(names.len(), spec.blocks.len(), "{}", spec.map);
        for block in &spec.blocks {
            let fields: HashSet<&str> = block.fields.iter().map(|f| f.name).collect();
            assert_eq!(
                fields.len(),
                block.fields.len(),
                "{}: {}",
                spec.map,
                block.name
            );
            for field in &block.fields {
                assert!(
                    !field.doc.trim().is_empty(),
                    "{}: {}.{} is undocumented",
                    spec.map,
                    block.name,
                    field.name
                );
            }
        }
    }
}

/// A 2P and a 5P observation are the same width; unused slots carry `present` = 0.
#[test]
fn opponent_slots_are_fixed_so_one_network_plays_any_table() {
    assert_eq!(OPPONENT_SLOTS, 4);
    // Python derived the slot count from the largest table; the Rust constant is written
    // out, so check it still agrees.
    assert_eq!(OPPONENT_SLOTS, MAX_PLAYERS - 1);
    for spec in all_boards().iter().map(|b| spec_for(b.name)) {
        assert_eq!(
            spec.named("opponents").count,
            OPPONENT_SLOTS,
            "{}",
            spec.map
        );
    }
    for (map, n) in all_configs() {
        let state = game(map, n).new_initial_state(1);
        assert_eq!(observation_size(&state), spec_for(map).size, "{map} {n}P");
    }
}

#[test]
fn the_spec_is_smaller_on_a_smaller_board() {
    assert!(spec_for("mini").size < spec_for("usa").size);
    assert_eq!(spec_for("usa").size, 3355);
    assert!(obs_version() >= 1);
}

// ---------------------------------------------------------------------------
// The encoder
// ---------------------------------------------------------------------------

#[test]
fn encoding_fills_a_buffer_of_the_right_size() {
    for board in all_boards() {
        let mut state = game(board.name, 2).new_initial_state(1);
        to_main(&mut state);
        assert_eq!(observation_size(&state), spec_for(board.name).size);
        let view = observe(&mut state, 0);
        assert!(
            view.iter().all(|v| (-5.0..=5.0).contains(v)),
            "{}: features should stay in a sane range",
            board.name
        );
    }
}

#[test]
fn a_reused_buffer_is_cleared() {
    let mut state = played();
    let mut buffer = vec![7.0; observation_size(&state)];
    encode(&mut state, 0, &mut buffer);
    let fresh = observe(&mut state, 0);
    assert_eq!(
        first_difference(&buffer, &fresh),
        None,
        "stale values leaked through the reused buffer"
    );
}

/// Python raised `ValueError`; the Rust encoder asserts.
#[test]
#[should_panic(expected = "buffer holds")]
fn a_wrong_sized_buffer_is_rejected() {
    let mut state = played();
    encode(&mut state, 0, &mut [0.0; 3]);
}

#[test]
fn encoding_is_deterministic() {
    let mut state = played();
    let first = observe(&mut state, 0);
    let second = observe(&mut state, 0);
    assert_eq!(first_difference(&first, &second), None);
}

#[test]
fn every_seat_sees_a_different_observation() {
    let mut state = played();
    let views: Vec<Vec<f32>> = (0..3).map(|p| observe(&mut state, p)).collect();
    for (a, first) in views.iter().enumerate() {
        for (b, second) in views.iter().enumerate().skip(a + 1) {
            assert!(
                first != second,
                "seats {a} and {b} see the same thing: the observation must be \
                 perspective-normalized"
            );
        }
    }
}

/// Slot 0 is whoever acts next. Seat-relative, because who moves before you decides games.
#[test]
fn opponents_are_ordered_by_distance_in_turn_order() {
    let mut state = in_main(4);
    force(
        &mut state,
        Rig {
            trains: Some(vec![45, 40, 30, 20]),
            ..Rig::default()
        },
    );
    let block = spec_for(state.board.name).named("opponents");
    let supply = f64::from(state.board.raw.trains_per_player);

    let mut trains_seen = |observer: usize| -> Vec<i64> {
        let view = observe(&mut state, observer);
        (0..3)
            .map(|i| (f64::from(view[block.at(i, "trains")]) * supply).round() as i64)
            .collect()
    };
    assert_eq!(trains_seen(0), [40, 30, 20]);
    assert_eq!(trains_seen(2), [20, 45, 40]);
}

#[test]
fn unused_opponent_slots_are_absent_and_zero() {
    let mut state = in_main(2);
    let block = spec_for(state.board.name).named("opponents");
    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(0, "present")], 1.0);
    for slot in 1..OPPONENT_SLOTS {
        assert_eq!(view[block.at(slot, "present")], 0.0, "slot {slot}");
        assert_eq!(view[block.at(slot, "trains")], 0.0, "slot {slot}");
    }
}

// ---------------------------------------------------------------------------
// Individual features move when the thing they describe moves
// ---------------------------------------------------------------------------

#[test]
fn segment_owner_one_hot_tracks_claims() {
    let mut state = in_main(3);
    let board = state.board;
    let block = spec_for(board.name).named("segment_dynamic");
    let segment = segment_named(board, "Atlanta", "Miami", 0);
    let owner = block.at(segment, "owner");

    assert_eq!(observe(&mut state, 0)[owner], 1.0, "free");

    let mut mine = state.clone();
    give(&mut mine, 0, segment);
    assert_eq!(observe(&mut mine, 0)[owner + 1], 1.0, "mine");

    let mut theirs = state.clone();
    theirs.seg_owner[segment] = 1;
    assert_eq!(
        observe(&mut theirs, 0)[owner + 2],
        1.0,
        "the seat that acts next"
    );
    assert_eq!(observe(&mut theirs, 2)[owner + 3], 1.0, "two seats away");
}

/// A blocked double and an enemy's track are very different to a blocking heuristic.
#[test]
fn closed_is_distinct_from_owned() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("segment_dynamic");
    let segment = segment_named(board, "Boston", "Montreal", 0);
    state.seg_owner[segment] = CLOSED;
    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(segment, "closed")], 1.0);
    assert_eq!(
        view[block.at(segment, "owner")],
        1.0,
        "closed is nobody's, not an opponent's"
    );
}

#[test]
fn twin_locked_marks_the_other_half_of_my_double() {
    let mut state = in_main(4);
    let board = state.board;
    let block = spec_for(board.name).named("segment_dynamic");
    let segment = segment_named(board, "Boston", "Montreal", 0);
    let twin = board.sibling[segment];
    assert_ne!(twin, NO_SIBLING, "Boston-Montreal is a double route");
    let twin = usize::from(twin);

    give(&mut state, 0, segment);
    assert_eq!(observe(&mut state, 0)[block.at(twin, "twin_locked")], 1.0);
    assert_eq!(observe(&mut state, 1)[block.at(twin, "twin_locked")], 0.0);
}

#[test]
fn can_afford_now_and_cards_short_track_the_hand() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("segment_dynamic");
    let segment = segment_named(board, "Atlanta", "Miami", 0);
    assert_eq!(
        (board.seg_len[segment], board.seg_color[segment]),
        (5, color(board, "blue")),
        "the rig assumes Atlanta-Miami is a blue 5"
    );
    let empty = vec![0; board.n_card_types];

    force(
        &mut state,
        Rig {
            hands: Some(vec![cards(board, &[("blue", 5)]), empty.clone()]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(segment, "can_afford_now")], 1.0);
    assert_eq!(view[block.at(segment, "cards_short")], 0.0);

    force(
        &mut state,
        Rig {
            hands: Some(vec![cards(board, &[("blue", 2)]), empty]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let view = observe(&mut state, 0);
    let short = block.at(segment, "cards_short");
    assert_eq!(view[block.at(segment, "can_afford_now")], 0.0);
    assert_eq!(view[short], 1.0, "at least one card short");
    assert_eq!(view[short + 2], 1.0, "three short");
    assert_eq!(view[short + 3], 0.0);
}

/// The colour-symmetry regularizer permutes colours; it has to be able to reach this.
#[test]
fn required_colour_is_a_feature_not_an_embedding() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("segment_static");
    let view = observe(&mut state, 0);
    let blue = usize::from(color(board, "blue"));
    let colored = segment_named(board, "Atlanta", "Miami", 0);
    let gray = segment_named(board, "Nashville", "Atlanta", 0);
    assert_eq!(view[block.at(colored, "required_color") + blue], 1.0);
    assert_eq!(
        view[block.at(gray, "required_color") + board.n_colors],
        1.0,
        "gray goes last"
    );
}

/// The most important engineered feature in the project.
#[test]
fn remaining_cost_falls_as_the_route_gets_built() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("tickets");
    let ticket = shortest_ticket(board);
    hold_only(&mut state, ticket);

    let before = observe(&mut state, 0)[block.at(ticket, "remaining_cost")];
    assert!(before > 0.0);
    assert_eq!(observe(&mut state, 0)[block.at(ticket, "held")], 1.0);

    let (here, target) = (board.ticket_a[ticket], board.ticket_b[ticket]);
    let (_, segment) = step_towards(board, here, target);
    give(&mut state, 0, segment);
    assert!(observe(&mut state, 0)[block.at(ticket, "remaining_cost")] < before);
}

#[test]
fn a_completed_ticket_reads_connected_at_zero_cost() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("tickets");
    let ticket = shortest_ticket(board);
    hold_only(&mut state, ticket);

    let (mut here, target) = (board.ticket_a[ticket], board.ticket_b[ticket]);
    while here != target {
        let (next, segment) = step_towards(board, here, target);
        give(&mut state, 0, segment);
        here = next;
    }

    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(ticket, "connected")], 1.0);
    assert_eq!(view[block.at(ticket, "remaining_cost")], 0.0);
    assert_eq!(view[block.at(ticket, "is_dead")], 0.0);
}

#[test]
fn a_cut_ticket_reads_dead() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("tickets");
    let boston = city(board, "Boston");
    let ticket = (0..board.n_tickets)
        .find(|&t| board.ticket_a[t] == boston)
        .expect("a ticket from Boston");
    hold_only(&mut state, ticket);
    for &(_, segment) in &board.adjacency[usize::from(boston)] {
        state.seg_owner[usize::from(segment)] = 1;
    }

    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(ticket, "is_dead")], 1.0);
    assert_eq!(
        view[block.at(ticket, "remaining_cost")],
        0.0,
        "a dead ticket carries no cost"
    );
}

#[test]
fn fragility_rises_when_a_route_hangs_on_one_edge() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("tickets");
    let (denver, el_paso) = (city(board, "Denver"), city(board, "El Paso"));
    let ticket = (0..board.n_tickets)
        .find(|&t| {
            let ends = (board.ticket_a[t], board.ticket_b[t]);
            ends == (denver, el_paso) || ends == (el_paso, denver)
        })
        .expect("a Denver-El Paso ticket");
    hold_only(&mut state, ticket);
    let open_board = observe(&mut state, 0)[block.at(ticket, "fragility")];

    // Cut every alternative out of Denver except the Santa Fe corridor.
    let santa_fe = city(board, "Santa Fe");
    for &(nb, segment) in &board.adjacency[usize::from(denver)] {
        if nb != santa_fe {
            state.seg_owner[usize::from(segment)] = 1;
        }
    }
    let squeezed = observe(&mut state, 0)[block.at(ticket, "fragility")];
    assert!(
        squeezed > open_board,
        "losing the last corridor should read as fragile"
    );
}

#[test]
fn faceup_one_hot_marks_empty_slots() {
    let mut state = in_main(2);
    let board = state.board;
    let block = spec_for(board.name).named("faceup");
    force(
        &mut state,
        Rig {
            hands: Some(vec![
                board.deck_composition_counts.clone(),
                vec![0; board.n_card_types],
            ]),
            faceup: Some(vec![]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    let view = observe(&mut state, 0);
    for slot in 0..FACEUP_SLOTS {
        assert_eq!(
            view[block.at(slot, "card") + board.n_card_types],
            1.0,
            "slot {slot}"
        );
    }
}

#[test]
fn unseen_counts_reach_the_observation() {
    let mut state = played();
    let board = state.board;
    let block = spec_for(board.name).named("piles");
    let view = observe(&mut state, 0);
    for (c, &unseen) in state.unseen_counts(0).iter().enumerate() {
        let expected = f64::from(unseen) / f64::from(board.cards_per_type(c as u8));
        let got = f64::from(view[block.at(0, "unseen") + c]);
        assert!(
            (got - expected).abs() < 1e-6,
            "card type {c}: {got} != {expected}"
        );
    }
}

/// Face-up takes are public, so what an opponent certainly holds is not a guess.
#[test]
fn certain_knowledge_of_an_opponent_shows_up() {
    let mut state = in_main(2);
    let block = spec_for(state.board.name).named("opponents");
    let before = observe(&mut state, 0);

    // Seat 1 takes a face-up card; seat 0 now knows one of its cards for certain.
    while state.current_player() != 1 || state.phase != PHASE_MAIN {
        let action = state.legal_actions()[0];
        state
            .step(action)
            .expect("the lowest legal action is legal");
    }
    let card = usize::try_from(state.faceup[0]).expect("face-up slot 0 holds a card");
    let take = state.space.draw(0);
    state.step(take).expect("taking face-up slot 0 is legal");
    let after = observe(&mut state, 0);
    let slot = block.at(0, "certain") + card;
    assert!(after[slot] > before[slot]);
}

#[test]
fn clock_block_tracks_the_endgame() {
    let mut state = in_main(2);
    let block = spec_for(state.board.name).named("clock");
    assert_eq!(observe(&mut state, 0)[block.at(0, "final_triggered")], 0.0);

    state.final_left = 2;
    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(0, "final_triggered")], 1.0);
    assert_eq!(view[block.at(0, "final_countdown")], 1.0);
    assert_eq!(view[block.at(0, "seats")], 1.0, "two seats");
}

#[test]
fn score_rank_and_gap_to_leader() {
    let mut state = in_main(3);
    let block = spec_for(state.board.name).named("clock");
    state.score[..3].copy_from_slice(&[10, 50, 30]);

    let leader = observe(&mut state, 1);
    assert_eq!(leader[block.at(0, "score_rank")], 1.0);
    assert_eq!(leader[block.at(0, "gap_to_leader")], 0.0);

    let last = observe(&mut state, 0);
    assert_eq!(last[block.at(0, "score_rank")], 0.0);
    assert!((f64::from(last[block.at(0, "gap_to_leader")]) - 0.4).abs() < 1e-6);
}

#[test]
fn steiner_block_reports_slack_and_exactness() {
    let mut state = in_main(2);
    let block = spec_for(state.board.name).named("steiner");
    let view = observe(&mut state, 0);
    assert_eq!(view[block.at(0, "exact")], 1.0);
    assert!(view[block.at(0, "remaining_cost")] > 0.0);
    // Full train supply minus the tree cost, so slack starts positive.
    assert!(view[block.at(0, "cost_minus_trains")] < 1.0);
}

#[test]
fn encoding_survives_a_whole_game() {
    for board in all_boards() {
        let mut state = game(board.name, 2).new_initial_state(4);
        let mut rng = policy(4);
        let mut buffer = vec![0.0; observation_size(&state)];
        let mut steps = 0;
        while !state.is_terminal() && steps < 40 {
            let seat = usize::from(state.current_player());
            encode(&mut state, seat, &mut buffer);
            let action = state.sample_legal(&mut rng).expect("a non-terminal state");
            state.step(action).expect("a sampled action is legal");
            steps += 1;
        }
        assert!(steps > 10, "{}", board.name);
    }
}
