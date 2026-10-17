//! Union-find, reachability costs, the Steiner tree, and the longest trail.
//!
//! The fast functions also meet slow, obviously-correct references inside the crate, in
//! `src/naive_graph.rs`; this file keeps the hand-checked facts and the closed forms.
//!
//! The longest trail gets an independent oracle here: a brute force over *permutations* of
//! the owned segments, which shares no code with the optimized version. That matters because
//! the optimization has four layers -- component split, Eulerian shortcut, memoized DFS,
//! early exit -- and a bug in any of them would otherwise be invisible.

mod common;

use common::{city, game, policy, segment_named};
use ttr_core::board::{Board, NO_SIBLING, UNREACHABLE, get_board};
use ttr_core::config::{CLOSED, FREE};
use ttr_core::graph::{
    MAX_TRAIL_EDGES, dsu_connected, dsu_find, dsu_union, edge_cost, longest_trail, owned_segments,
    remaining_costs_from, steiner_cost, steiner_cost_exact,
};

fn usa() -> &'static Board {
    get_board("usa").expect("the USA map is generated")
}

fn mini() -> &'static Board {
    get_board("mini").expect("TTR-mini is generated")
}

/// Nobody owns anything.
fn empty(board: &Board) -> Vec<u8> {
    vec![FREE; board.n_segments]
}

/// `player` owns exactly `segments`; everything else is free.
fn own(board: &Board, segments: &[usize], player: u8) -> Vec<u8> {
    let mut owner = empty(board);
    for &s in segments {
        owner[s] = player;
    }
    owner
}

// ---------------------------------------------------------------------------
// Union-find
// ---------------------------------------------------------------------------

#[test]
fn union_find_basics() {
    let mut parent: Vec<u8> = (0..10).collect();
    assert_eq!(dsu_find(&mut parent, 7), 7);
    assert!(!dsu_connected(&mut parent, 1, 2));
    assert!(dsu_union(&mut parent, 1, 2));
    assert!(dsu_connected(&mut parent, 1, 2));
    assert!(
        !dsu_union(&mut parent, 2, 1),
        "a redundant union reports no change"
    );
    dsu_union(&mut parent, 2, 3);
    assert!(
        dsu_connected(&mut parent, 1, 3),
        "connectivity is transitive"
    );
    assert!(!dsu_connected(&mut parent, 1, 4));
}

/// Union by smaller index, so the forest is a function of the claims, not their order.
#[test]
fn union_find_is_deterministic_regardless_of_merge_order() {
    let mut a: Vec<u8> = (0..8).collect();
    let mut b: Vec<u8> = (0..8).collect();
    for (x, y) in [(1, 2), (3, 4), (2, 3)] {
        dsu_union(&mut a, x, y);
    }
    for (x, y) in [(3, 4), (2, 3), (1, 2)] {
        dsu_union(&mut b, x, y);
    }
    for i in 0..8 {
        assert_eq!(dsu_find(&mut a, i), dsu_find(&mut b, i), "city {i}");
    }
}

/// Halving, not full compression: one pointer update per step and no second pass. A single
/// find halves a chain rather than flattening it, so repeated finds converge instead of
/// paying a two-pass walk every time.
#[test]
fn path_halving_shortens_chains() {
    fn depth(parent: &[u8], mut x: u8) -> usize {
        let mut steps = 0;
        while parent[usize::from(x)] != x {
            x = parent[usize::from(x)];
            steps += 1;
        }
        steps
    }

    let mut parent: Vec<u8> = (0..20).collect();
    for i in (1..20u8).rev() {
        parent[usize::from(i)] = i - 1; // a degenerate chain
    }
    assert_eq!(depth(&parent, 19), 19);
    assert_eq!(dsu_find(&mut parent, 19), 0);
    assert!(depth(&parent, 19) < 19, "the chain must get shorter");
    for _ in 0..6 {
        dsu_find(&mut parent, 19);
    }
    assert_eq!(parent[19], 0, "repeated finds converge to a flat forest");
}

// ---------------------------------------------------------------------------
// Edge costs and remaining cost
// ---------------------------------------------------------------------------

#[test]
fn edge_cost_prices_track_by_who_owns_it() {
    let usa = usa();
    let segment = segment_named(usa, "Atlanta", "Miami", 0); // length 5
    let mut owner = empty(usa);
    assert_eq!(
        edge_cost(usa, &owner, 0, segment, true),
        5,
        "free track costs its length"
    );
    owner[segment] = 0;
    assert_eq!(
        edge_cost(usa, &owner, 0, segment, true),
        0,
        "my own track is free"
    );
    owner[segment] = 1;
    assert!(edge_cost(usa, &owner, 0, segment, true) >= UNREACHABLE);
    owner[segment] = CLOSED;
    assert!(edge_cost(usa, &owner, 0, segment, true) >= UNREACHABLE);
}

#[test]
fn edge_cost_blocks_a_double_i_already_half_own_in_four_or_five_player_games() {
    let usa = usa();
    let segment = segment_named(usa, "Boston", "Montreal", 0);
    assert_ne!(
        usa.sibling[segment], NO_SIBLING,
        "Boston-Montreal is a double"
    );
    let sibling = usize::from(usa.sibling[segment]);
    let mut owner = empty(usa);
    owner[segment] = 0;
    assert!(edge_cost(usa, &owner, 0, sibling, false) >= UNREACHABLE);
    assert_eq!(
        edge_cost(usa, &owner, 1, sibling, false),
        u16::from(usa.seg_len[sibling])
    );
}

#[test]
fn remaining_cost_on_an_empty_board_is_the_static_distance() {
    let usa = usa();
    let dist = remaining_costs_from(usa, &empty(usa), 0, 0, true);
    assert_eq!(dist, usa.dist[0]);
}

#[test]
fn owning_track_makes_it_free() {
    let usa = usa();
    let src = city(usa, "Atlanta");
    let dst = usize::from(city(usa, "Miami"));
    let before = remaining_costs_from(usa, &empty(usa), 0, src, true)[dst];
    let owner = own(usa, &[segment_named(usa, "Atlanta", "Miami", 0)], 0);
    let after = remaining_costs_from(usa, &owner, 0, src, true)[dst];
    assert_eq!(before, 5);
    assert_eq!(after, 0);
}

/// `UNREACHABLE` is the `is_dead` signal the ticket features report.
#[test]
fn an_opponent_can_cut_a_connection_dead() {
    let usa = usa();
    let mut owner = empty(usa);
    let vancouver = city(usa, "Vancouver");
    for &(_, segment) in &usa.adjacency[usize::from(vancouver)] {
        owner[usize::from(segment)] = 1;
    }
    let dist = remaining_costs_from(usa, &owner, 0, vancouver, true);
    assert!(dist[usize::from(city(usa, "Denver"))] >= UNREACHABLE);
}

// ---------------------------------------------------------------------------
// Steiner tree
// ---------------------------------------------------------------------------

#[test]
fn steiner_of_fewer_than_two_terminals_is_zero() {
    let usa = usa();
    let owner = empty(usa);
    assert_eq!(steiner_cost(usa, &owner, 0, &[], true), 0);
    assert_eq!(steiner_cost(usa, &owner, 0, &[3], true), 0);
}

#[test]
fn steiner_of_two_terminals_is_the_shortest_path() {
    let usa = usa();
    let owner = empty(usa);
    for (a, b) in [
        ("Denver", "El Paso"),
        ("Seattle", "New York"),
        ("Boston", "Miami"),
    ] {
        let (ia, ib) = (city(usa, a), city(usa, b));
        assert_eq!(
            steiner_cost(usa, &owner, 0, &[ia, ib], true),
            u32::from(usa.dist[usize::from(ia)][usize::from(ib)]),
            "{a}-{b}"
        );
    }
}

/// The whole point: per-ticket Dijkstras double-count the shared trunk line.
#[test]
fn steiner_never_exceeds_the_sum_of_separate_paths() {
    let usa = usa();
    let owner = empty(usa);
    let hub = city(usa, "Chicago");
    let spokes = ["Seattle", "Miami", "Los Angeles"].map(|c| city(usa, c));
    let separate: u32 = spokes
        .iter()
        .map(|&s| u32::from(usa.dist[usize::from(hub)][usize::from(s)]))
        .sum();
    let together = steiner_cost(
        usa,
        &owner,
        0,
        &[hub, spokes[0], spokes[1], spokes[2]],
        true,
    );
    assert!(together <= separate);
    assert!(
        together < separate,
        "these three share the trunk out of Chicago"
    );
}

#[test]
fn steiner_is_zero_once_the_terminals_are_connected() {
    let usa = usa();
    let owner = own(usa, &[segment_named(usa, "Nashville", "Atlanta", 0)], 0);
    let (a, b) = (city(usa, "Nashville"), city(usa, "Atlanta"));
    assert_eq!(steiner_cost(usa, &owner, 0, &[a, b], true), 0);
}

/// Ten terminals take the documented MST fallback. The last two assertions are new in the
/// port: four terminals are under the cap, so exact, and an upper bound on ten terminals can
/// never undercut the exact tree for four of them.
#[test]
fn steiner_falls_back_to_an_upper_bound_past_the_terminal_cap() {
    let usa = usa();
    let owner = empty(usa);
    let terminals: Vec<u8> = (0..10).collect();
    let (cost, exact) = steiner_cost_exact(usa, &owner, 0, &terminals, true);
    assert!(
        !exact,
        "ten terminals must take the documented MST fallback"
    );
    assert!(cost > 0);
    let (tree, tree_exact) = steiner_cost_exact(usa, &owner, 0, &terminals[..4], true);
    assert!(tree > 0);
    assert!(tree_exact);
    assert!(
        tree <= cost,
        "fallback {cost} undercuts a subset's exact tree {tree}"
    );
}

#[test]
fn steiner_reports_unreachable_when_a_terminal_is_walled_off() {
    let usa = usa();
    let mut owner = empty(usa);
    let vancouver = city(usa, "Vancouver");
    for &(_, segment) in &usa.adjacency[usize::from(vancouver)] {
        owner[usize::from(segment)] = 1;
    }
    let cost = steiner_cost(usa, &owner, 0, &[vancouver, city(usa, "Miami")], true);
    assert!(cost >= u32::from(UNREACHABLE));
}

/// A three-terminal Steiner tree has at most one branch vertex, which gives an oracle: the
/// optimum is `min over v of d(v,a) + d(v,b) + d(v,c)` in the shortest-path metric. Any
/// three-terminal tree decomposes into three paths meeting at one vertex, and where two of
/// them overlap the branch vertex slides along the overlap. A genuinely independent check
/// on the subset DP -- it shares no code with it. Every triple on TTR-mini, a sample on USA.
#[test]
fn steiner_of_three_terminals_matches_the_closed_form() {
    fn triples(cities: &[u8]) -> Vec<[u8; 3]> {
        let mut out = Vec::new();
        for (i, &a) in cities.iter().enumerate() {
            for (j, &b) in cities.iter().enumerate().skip(i + 1) {
                for &c in &cities[j + 1..] {
                    out.push([a, b, c]);
                }
            }
        }
        out
    }

    let all_mini: Vec<u8> = (0..mini().n_cities as u8).collect();
    let usa_sample: Vec<u8> = (0..usa().n_cities as u8).step_by(5).collect();
    for (board, cities) in [(mini(), all_mini), (usa(), usa_sample)] {
        let owner = empty(board);
        for [a, b, c] in triples(&cities) {
            let expected = (0..board.n_cities)
                .map(|v| {
                    [a, b, c]
                        .iter()
                        .map(|&t| u32::from(board.dist[v][usize::from(t)]))
                        .sum::<u32>()
                })
                .min()
                .expect("a board has cities");
            let got = steiner_cost(board, &owner, 0, &[a, b, c], true);
            assert_eq!(got, expected, "{} {:?}", board.name, (a, b, c));
        }
    }
}

// ---------------------------------------------------------------------------
// Longest trail -- against an independent oracle
// ---------------------------------------------------------------------------

/// Does this exact sequence of segments form a walk that reuses no segment?
fn is_trail(board: &Board, order: &[usize]) -> bool {
    let first = order[0];
    let (a0, b0) = (board.seg_a[first], board.seg_b[first]);
    [a0, b0].into_iter().any(|start| {
        let mut here = if start == a0 { b0 } else { a0 };
        order[1..].iter().all(|&segment| {
            let (a, b) = (board.seg_a[segment], board.seg_b[segment]);
            if here == a {
                here = b;
                true
            } else if here == b {
                here = a;
                true
            } else {
                false
            }
        })
    })
}

/// Step to the next lexicographic permutation; `false` once the last one has been seen.
fn next_permutation(items: &mut [usize]) -> bool {
    let Some(i) = (1..items.len()).rev().find(|&i| items[i - 1] < items[i]) else {
        return false;
    };
    let j = (i..items.len())
        .rev()
        .find(|&j| items[j] > items[i - 1])
        .expect("items[i] itself qualifies");
    items.swap(i - 1, j);
    items[i..].reverse();
    true
}

/// The oracle: every ordering, longest valid prefix. Shares no code with the engine.
fn brute_force_longest_trail(board: &Board, segments: &[usize]) -> u16 {
    let mut order = segments.to_vec();
    order.sort_unstable(); // the first permutation in lexicographic order
    let mut best = 0;
    loop {
        for size in (1..=order.len()).rev() {
            let prefix = &order[..size];
            let weight: u16 = prefix.iter().map(|&s| u16::from(board.seg_len[s])).sum();
            if weight <= best {
                break;
            }
            if is_trail(board, prefix) {
                best = weight;
                break;
            }
        }
        if !next_permutation(&mut order) {
            return best;
        }
    }
}

#[test]
fn longest_trail_matches_brute_force() {
    let cases: [&[(&str, &str)]; 6] = [
        &[("Nashville", "Atlanta")],
        &[("Nashville", "Atlanta"), ("Atlanta", "Miami")],
        // A triangle: an Eulerian circuit, so the answer is every edge.
        &[
            ("Nashville", "Atlanta"),
            ("Atlanta", "Raleigh"),
            ("Raleigh", "Nashville"),
        ],
        // Two disconnected pieces: the answer is the bigger one, not the sum.
        &[("Nashville", "Atlanta"), ("Vancouver", "Calgary")],
        // A star at Denver: degree 4, so no Eulerian trail and the DFS has to run.
        &[
            ("Denver", "Omaha"),
            ("Denver", "Santa Fe"),
            ("Denver", "Helena"),
            ("Denver", "Phoenix"),
        ],
        // A hub with a tail, mixing the layers.
        &[
            ("Denver", "Omaha"),
            ("Denver", "Santa Fe"),
            ("Denver", "Helena"),
            ("Denver", "Phoenix"),
            ("Santa Fe", "El Paso"),
            ("Omaha", "Chicago"),
        ],
    ];
    let usa = usa();
    for pairs in cases {
        let segments: Vec<usize> = pairs
            .iter()
            .map(|&(a, b)| segment_named(usa, a, b, 0))
            .collect();
        let owner = own(usa, &segments, 0);
        assert_eq!(
            longest_trail(usa, &owner, 0),
            brute_force_longest_trail(usa, &segments),
            "{pairs:?}"
        );
    }
}

/// Positions a random game actually reaches, capped where the oracle stays affordable.
#[test]
fn longest_trail_matches_brute_force_on_real_game_subgraphs() {
    let game = game("usa", 2);
    let board = game.board;
    let mut checked = 0;
    for seed in 0..40u64 {
        let mut state = game.new_initial_state(seed);
        let mut rng = policy(seed);
        while !state.is_terminal() {
            let action = state.sample_legal(&mut rng).expect("a non-terminal state");
            state.step(action).expect("a sampled action is legal");
            for p in 0..2u8 {
                let segments = owned_segments(&state.seg_owner, p, board.n_segments);
                if !(1..=7).contains(&segments.len()) {
                    continue;
                }
                assert_eq!(
                    longest_trail(board, &state.seg_owner, p),
                    brute_force_longest_trail(board, &segments),
                    "seed {seed} seat {p}: {segments:?}"
                );
                checked += 1;
                if checked > 300 {
                    return;
                }
            }
        }
    }
    assert!(checked > 50, "only {checked} subgraphs were checked");
}

#[test]
fn longest_trail_of_nothing_is_zero() {
    let usa = usa();
    assert_eq!(longest_trail(usa, &empty(usa), 0), 0);
}

/// A path (2 odd vertices) and a circuit (0) both admit a trail over every edge.
#[test]
fn the_eulerian_shortcut_fires_and_is_right() {
    let usa = usa();
    let path = [
        segment_named(usa, "Vancouver", "Calgary", 0),
        segment_named(usa, "Calgary", "Helena", 0),
        segment_named(usa, "Helena", "Denver", 0),
    ];
    let owner = own(usa, &path, 0);
    let total: u16 = path.iter().map(|&s| u16::from(usa.seg_len[s])).sum();
    assert_eq!(longest_trail(usa, &owner, 0), total);
}

#[test]
fn longest_trail_ignores_other_players_and_closed_track() {
    let usa = usa();
    let mut owner = empty(usa);
    let mine = segment_named(usa, "Atlanta", "Miami", 0);
    owner[mine] = 0;
    owner[segment_named(usa, "Nashville", "Atlanta", 0)] = 1;
    owner[segment_named(usa, "Charleston", "Atlanta", 0)] = CLOSED;
    assert_eq!(longest_trail(usa, &owner, 0), u16::from(usa.seg_len[mine]));
}

/// Better a clear error than a silently truncated mask. The train supply prevents it.
///
/// Stricter than the Python original, whose `try`/`except` also passed when nothing was
/// raised: these 40 short segments do form a 35-edge component with 8 odd vertices, so the
/// panic is required, not merely allowed.
#[test]
#[should_panic(expected = "mask limit")]
fn a_component_wider_than_the_bitmask_is_rejected_loudly() {
    let usa = usa();
    let mut owner = empty(usa);
    // A long chain plus a hub, forced past the 32-edge limit and made non-Eulerian.
    let chain: Vec<usize> = (0..usa.n_segments)
        .filter(|&s| usa.seg_len[s] <= 2)
        .collect();
    for &s in chain.iter().take(MAX_TRAIL_EDGES + 8) {
        owner[s] = 0;
    }
    longest_trail(usa, &owner, 0);
}
