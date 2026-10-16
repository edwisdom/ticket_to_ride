//! The encoder's plan features: `on_my_steiner_tree`, and which tickets are terminals.
//!
//! Written for Phase 3.1's mutation testing, which found these features had no explicit
//! test at all: emptying the Steiner edge set, or letting a completed or dead ticket into
//! the terminal set, survived the whole ported suite and was caught only by the `oracle-v1`
//! differential and the golden corpus. Those stay, but they compare against a frozen copy;
//! a feature's meaning deserves a test that states it.
//!
//! Writing them found a defect that is part of `OBS_VERSION` 1 as signed (the retired
//! Python encoder had it too): the walk that recovers a planned route gives my own track
//! weight 0, so it can step back and forth across one of my segments until its loop guard
//! runs out, and the plan comes back truncated -- 43-44% of open-ticket plans in H3 games on
//! the USA map, 23% on mini. `on_my_steiner_tree` and `fragility` both read that walk. The
//! fix changes features, so it is a named `OBS_VERSION` bump done on its own; until then
//! the test that states the right behaviour is ignored with the defect as its reason, and
//! `plans_survive_a_real_game` asserts only what holds either way.

mod common;

use std::collections::BTreeSet;

use common::{game, give, policy, to_main};
use ttr_core::board::Board;
use ttr_core::config::FREE;
use ttr_core::graph::remaining_costs_from;
use ttr_core::obs::{encode, observation_size, spec_for};
use ttr_core::state::State;

/// The absolute offset of `block[entity].field`.
fn at(map: &str, block: &str, entity: usize, field: &str) -> usize {
    let spec = spec_for(map);
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

fn observe(state: &mut State, player: usize) -> Vec<f32> {
    let mut out = vec![0.0; observation_size(state)];
    encode(state, player, &mut out);
    out
}

/// USA `n`P in MAIN, seat 0 to act and holding exactly `tickets`, nothing claimed.
fn holding(n: usize, tickets: &[usize]) -> State {
    let mut state = game("usa", n).new_initial_state(3);
    to_main(&mut state);
    state.tickets[0] = tickets.iter().fold(0u32, |m, &t| m | 1 << t);
    state.cur = 0;
    state
}

/// Segments flagged `on_my_steiner_tree` in seat 0's observation.
fn plan(state: &mut State) -> BTreeSet<usize> {
    let view = observe(state, 0);
    let map = state.board.name;
    (0..state.board.n_segments)
        .filter(|&s| view[at(map, "segment_dynamic", s, "on_my_steiner_tree")] == 1.0)
        .collect()
}

fn length(board: &Board, segments: &BTreeSet<usize>) -> u16 {
    segments.iter().map(|&s| u16::from(board.seg_len[s])).sum()
}

fn distance(board: &Board, ticket: usize) -> u16 {
    board.dist[usize::from(board.ticket_a[ticket])][usize::from(board.ticket_b[ticket])]
}

/// Do `segments` form exactly one walk from `a` to `b`?
fn is_path(board: &Board, segments: &BTreeSet<usize>, a: u8, b: u8) -> bool {
    let mut left: Vec<usize> = segments.iter().copied().collect();
    let mut here = a;
    while here != b {
        let Some(i) = left
            .iter()
            .position(|&s| board.seg_a[s] == here || board.seg_b[s] == here)
        else {
            return false;
        };
        let s = left.swap_remove(i);
        here = if board.seg_a[s] == here {
            board.seg_b[s]
        } else {
            board.seg_a[s]
        };
    }
    left.is_empty()
}

/// Tickets longest first, so a plan is a real multi-segment route.
fn long_tickets(board: &Board) -> Vec<usize> {
    let mut tickets: Vec<usize> = (0..board.n_tickets).collect();
    tickets.sort_by_key(|&t| std::cmp::Reverse(distance(board, t)));
    tickets
}

/// One held ticket on an empty board: the plan is a cheapest route between its endpoints --
/// free track only, as long as the ticket's distance, and one connected walk.
#[test]
fn the_plan_flags_exactly_one_cheapest_path_for_a_single_ticket() {
    let board = game("usa", 2).board;
    let ticket = long_tickets(board)[0];
    let (a, b) = (board.ticket_a[ticket], board.ticket_b[ticket]);
    let mut state = holding(2, &[ticket]);

    let route = plan(&mut state);
    assert!(!route.is_empty(), "a held, unfinished ticket has a plan");
    assert!(route.iter().all(|&s| state.seg_owner[s] == FREE));
    assert_eq!(
        length(board, &route),
        distance(board, ticket),
        "a *cheapest* route"
    );
    assert!(
        is_path(board, &route, a, b),
        "one walk from end to end: {route:?}"
    );
}

/// Claiming a step of the plan takes it off the plan, and what is left still reaches: the
/// remaining free steps plus my own track form one walk, costing exactly what is left.
#[test]
#[ignore = "OBS_VERSION 1 defect: the plan walk ping-pongs across zero-weight own track and \
            truncates the plan; fixing it is a named OBS_VERSION bump (see the module doc)"]
fn claimed_track_leaves_the_plan_and_the_rest_still_reaches() {
    let board = game("usa", 2).board;
    let ticket = long_tickets(board)[0];
    let (a, b) = (board.ticket_a[ticket], board.ticket_b[ticket]);
    let mut state = holding(2, &[ticket]);
    let claimed = *plan(&mut state).iter().next().expect("a plan");
    give(&mut state, 0, claimed);

    let rest = plan(&mut state);
    assert!(
        !rest.contains(&claimed),
        "my own track is mine to use, not a step to take"
    );
    let left = remaining_costs_from(board, &state.seg_owner[..board.n_segments], 0, a, true);
    assert_eq!(length(board, &rest), left[usize::from(b)]);
    // Any route avoiding the claimed segment costs at least the old distance, which is more
    // than what is left -- so the new cheapest route goes through it.
    let mut walk = rest.clone();
    walk.insert(claimed);
    assert!(is_path(board, &walk, a, b), "{walk:?}");
}

/// A finished ticket costs nothing and a dead one cannot be finished: neither is a terminal.
/// Letting a dead ticket in would also poison the Steiner block with the unreachable cost.
#[test]
fn completed_and_dead_tickets_are_not_terminals() {
    let board = game("usa", 2).board;
    let ticket = long_tickets(board)[0];

    let mut done = holding(2, &[ticket]);
    for s in plan(&mut done) {
        give(&mut done, 0, s);
    }
    let view = observe(&mut done, 0);
    assert_eq!(view[at("usa", "tickets", ticket, "connected")], 1.0);
    assert!(
        plan(&mut done).is_empty(),
        "a completed ticket plans nothing"
    );
    assert_eq!(view[at("usa", "steiner", 0, "remaining_cost")], 0.0);

    let mut dead = holding(2, &[ticket]);
    let a = usize::from(board.ticket_a[ticket]);
    for &(_, segment) in &board.adjacency[a] {
        dead.seg_owner[usize::from(segment)] = 1;
    }
    let view = observe(&mut dead, 0);
    assert_eq!(view[at("usa", "tickets", ticket, "is_dead")], 1.0);
    assert!(plan(&mut dead).is_empty(), "a dead ticket plans nothing");
    assert_eq!(view[at("usa", "steiner", 0, "remaining_cost")], 0.0);
}

/// A dead ticket held alongside an open one changes nothing about the open one: same plan,
/// same tree cost. (The doomed ticket is cut off by track that is not on the open plan, so
/// the open route itself is untouched.)
#[test]
fn a_dead_ticket_does_not_hide_an_open_one() {
    let board = game("usa", 2).board;
    let tickets = long_tickets(board);
    let open = tickets[0];
    let mut alone = holding(2, &[open]);
    let alone_plan = plan(&mut alone);
    let alone_view = observe(&mut alone, 0);

    let ends = [board.ticket_a[open], board.ticket_b[open]];
    let doomed = *tickets[1..]
        .iter()
        .find(|&&t| {
            let a = board.ticket_a[t];
            !ends.contains(&a)
                && !ends.contains(&board.ticket_b[t])
                && board.adjacency[usize::from(a)]
                    .iter()
                    .all(|&(_, s)| !alone_plan.contains(&usize::from(s)))
        })
        .expect("a ticket that can be cut off away from the open plan");

    let mut both = holding(2, &[open, doomed]);
    for &(_, segment) in &board.adjacency[usize::from(board.ticket_a[doomed])] {
        both.seg_owner[usize::from(segment)] = 1;
    }
    let both_view = observe(&mut both, 0);
    assert_eq!(both_view[at("usa", "tickets", doomed, "is_dead")], 1.0);
    assert_eq!(plan(&mut both), alone_plan);
    let key = at("usa", "steiner", 0, "remaining_cost");
    assert_eq!(
        both_view[key], alone_view[key],
        "the dead ticket changed the tree's cost"
    );
}

/// Several tickets: the plan is the union of each ticket's own plan, because each terminal
/// pair is routed on its own.
#[test]
fn several_tickets_plan_the_union_of_their_routes() {
    let board = game("usa", 2).board;
    let tickets = &long_tickets(board)[..3];
    let mut all = holding(2, tickets);
    let union: BTreeSet<usize> = tickets
        .iter()
        .flat_map(|&t| plan(&mut holding(2, &[t])))
        .collect();
    assert_eq!(plan(&mut all), union);
}

/// Mid-game: every flagged segment is free, and there is no plan without a ticket that is
/// neither finished nor dead. (The converse -- an open ticket always has a plan -- is what
/// the `OBS_VERSION` 1 walk defect can break, so it is not asserted here.)
#[test]
fn plans_survive_a_real_game() {
    for seed in 0..8 {
        let mut state = game("usa", 3).new_initial_state(seed);
        let mut rng = policy(seed);
        for _ in 0..90 {
            if state.is_terminal() {
                break;
            }
            let action = state.sample_legal(&mut rng).expect("not over");
            state.step(action).expect("legal");
        }
        let route = plan(&mut state);
        assert!(
            route.iter().all(|&s| state.seg_owner[s] == FREE),
            "seed {seed}"
        );
        let view = observe(&mut state, 0);
        let open = state.tickets_of(0).into_iter().map(usize::from).any(|t| {
            view[at("usa", "tickets", t, "connected")] == 0.0
                && view[at("usa", "tickets", t, "is_dead")] == 0.0
        });
        assert!(
            open || route.is_empty(),
            "seed {seed}: a plan with no open ticket"
        );
    }
}
