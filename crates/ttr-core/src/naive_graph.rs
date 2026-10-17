//! Deliberately naive graph references, for differential testing inside the crate.
//!
//! PLAN.md §14.1 step 4. With the Python engine gone nothing else disagrees with
//! [`crate::graph`], so each fast function meets a slow one that shares none of its structure
//! and whose correctness can be checked by reading it:
//!
//! * `remaining_costs_from` (Dijkstra), `all_pairs` (a fixed-width Floyd-Warshall) and
//!   `guided_distance` (A* with one segment priced out) against `floyd_warshall`.
//! * `steiner_cost_exact` (contraction, Dreyfus-Wagner, MST fallback) against
//!   `spanning_trees` + `cheapest_superset`, a brute force over every set of cities -- so on
//!   TTR-mini only.
//! * `longest_trail` (component split, Eulerian shortcut, memoized DFS, early exit) against
//!   `exhaustive_longest_trail`.
//!
//! Track prices come from `edge_cost` on both sides. It is ten lines of rules and is tested
//! directly in `tests/graph.rs`; what these references check is the algorithms built on it.
//! `tests/graph.rs` also holds a second, independent longest-trail oracle -- every
//! permutation of the owned segments -- which is affordable only up to seven segments.
//!
//! The default tests stay small enough for a debug `cargo test`; the `#[ignore]`d sweeps
//! run the same comparisons much wider.

use std::collections::BTreeSet;

use crate::board::{Board, UNREACHABLE, get_board};
use crate::config::{CLOSED, FREE, RuleConfig};
use crate::graph::{
    MAX_STEINER_TERMINALS, MAX_TRAIL_EDGES, all_pairs, crossing_costs, edge_cost, guided_distance,
    longest_trail, remaining_costs_from, steiner_cost_exact,
};
use crate::rng::{Part, Pcg32, stream};
use crate::state::Game;

// ---------------------------------------------------------------------------
// The references
// ---------------------------------------------------------------------------

/// All-pairs "train cars still needed", by Floyd-Warshall over `edge_cost`, with the
/// segment `without` (if any) treated as uncrossable.
///
/// "No route" stays `None` until the very end, so no sentinel ever takes part in a sum.
fn floyd_warshall(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    doubles_locked: bool,
    without: Option<usize>,
) -> Vec<Vec<u16>> {
    let n = board.n_cities;
    let mut d: Vec<Vec<Option<u32>>> = vec![vec![None; n]; n];
    for (v, row) in d.iter_mut().enumerate() {
        row[v] = Some(0);
    }
    for s in 0..board.n_segments {
        let w = edge_cost(board, seg_owner, player, s, doubles_locked);
        if w >= UNREACHABLE || without == Some(s) {
            continue;
        }
        let (a, b) = (usize::from(board.seg_a[s]), usize::from(board.seg_b[s]));
        for (x, y) in [(a, b), (b, a)] {
            d[x][y] = Some(d[x][y].map_or(u32::from(w), |old| old.min(u32::from(w))));
        }
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if let (Some(ik), Some(kj)) = (d[i][k], d[k][j]) {
                    if d[i][j].is_none_or(|ij| ik + kj < ij) {
                        d[i][j] = Some(ik + kj);
                    }
                }
            }
        }
    }
    d.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|c| c.map_or(UNREACHABLE, |c| u16::try_from(c).expect("a real distance")))
                .collect()
        })
        .collect()
}

/// A brute force over vertex sets is exponential in the city count: TTR-mini has 14.
const MAX_BRUTE_FORCE_CITIES: usize = 16;

/// `tree[set]`: the cheapest track spanning the cities in the bitmask `set` -- the minimum
/// spanning tree of the subgraph `set` induces, over passable track -- or `None` when that
/// track does not connect them.
///
/// Some cheapest track connecting the terminals is a tree (drop an edge from any cycle;
/// nothing costs less than zero). That tree spans *some* set of cities U, so it costs at
/// least `tree[U]`; and `tree[U]` connects every city in U. So the Steiner cost of terminals
/// T is exactly the minimum of `tree[U]` over every U containing T, which is
/// [`cheapest_superset`]. Nothing is pruned, bounded or shortcut: no metric closure, no limit
/// on branch points, and no "first size that works", which is wrong because more edges can
/// weigh less (docs/GOTCHAS.md).
fn spanning_trees(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    doubles_locked: bool,
) -> Vec<Option<u32>> {
    let n = board.n_cities;
    assert!(
        n <= MAX_BRUTE_FORCE_CITIES,
        "{}: {n} cities is too many to brute-force",
        board.name
    );
    // Kruskal: passable track, cheapest first.
    let mut track: Vec<(u16, usize, usize)> = (0..board.n_segments)
        .map(|s| {
            (
                edge_cost(board, seg_owner, player, s, doubles_locked),
                usize::from(board.seg_a[s]),
                usize::from(board.seg_b[s]),
            )
        })
        .filter(|&(w, _, _)| w < UNREACHABLE)
        .collect();
    track.sort_unstable();
    (0..1usize << n)
        .map(|set| kruskal(&track, set, n))
        .collect()
}

/// The minimum spanning tree of the subgraph `set` induces, if that subgraph is connected.
fn kruskal(track: &[(u16, usize, usize)], set: usize, n: usize) -> Option<u32> {
    let inside = |v: usize| set >> v & 1 == 1;
    let mut parent: Vec<usize> = (0..n).collect();
    let (mut total, mut joined) = (0u32, 0u32);
    for &(w, a, b) in track {
        if !inside(a) || !inside(b) {
            continue;
        }
        let (ra, rb) = (root(&parent, a), root(&parent, b));
        if ra != rb {
            parent[ra] = rb;
            total += u32::from(w);
            joined += 1;
        }
    }
    let cities = set.count_ones();
    (cities == 0 || joined == cities - 1).then_some(total)
}

/// The cheapest `tree[U]` over every city set U that contains all of `terminals`.
fn cheapest_superset(tree: &[Option<u32>], terminals: &[u8]) -> Option<u32> {
    let need = terminals.iter().fold(0usize, |mask, &t| mask | 1 << t);
    let mut best: Option<u32> = None;
    for (set, &cost) in tree.iter().enumerate() {
        if set & need == need {
            if let Some(c) = cost {
                best = Some(best.map_or(c, |b| b.min(c)));
            }
        }
    }
    best
}

/// Union-find with neither halving nor ranks: a root is wherever the pointers stop.
fn root(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

/// How many separate pieces of `player`'s own track the terminals sit on, a terminal off it
/// counting as a piece of its own. Own track costs nothing, so the Steiner DP runs over these
/// pieces, and their count is what decides exact versus the MST fallback.
fn pieces(board: &Board, seg_owner: &[u8], player: u8, terminals: &[u8]) -> usize {
    let mut parent: Vec<usize> = (0..board.n_cities).collect();
    for s in (0..board.n_segments).filter(|&s| seg_owner[s] == player) {
        let ra = root(&parent, usize::from(board.seg_a[s]));
        let rb = root(&parent, usize::from(board.seg_b[s]));
        parent[ra] = rb;
    }
    terminals
        .iter()
        .map(|&t| root(&parent, usize::from(t)))
        .collect::<BTreeSet<_>>()
        .len()
}

/// The longest trail by exhaustive search: from every city, every walk along unused track.
///
/// No memo, no component split, no Eulerian shortcut, no early exit and no bitmask, so no
/// edge limit either. Exponential, and affordable for anything a seat can own.
fn exhaustive_longest_trail(board: &Board, seg_owner: &[u8], player: u8) -> u16 {
    let owned: Vec<usize> = (0..board.n_segments)
        .filter(|&s| seg_owner[s] == player)
        .collect();
    let mut used = vec![false; owned.len()];
    (0..board.n_cities as u8)
        .map(|city| longest_walk_from(board, &owned, &mut used, city))
        .max()
        .unwrap_or(0)
}

fn longest_walk_from(board: &Board, owned: &[usize], used: &mut [bool], here: u8) -> u16 {
    let mut best = 0;
    for (i, &s) in owned.iter().enumerate() {
        let (a, b) = (board.seg_a[s], board.seg_b[s]);
        if used[i] || (here != a && here != b) {
            continue;
        }
        let there = if here == a { b } else { a };
        used[i] = true;
        best = best.max(u16::from(board.seg_len[s]) + longest_walk_from(board, owned, used, there));
        used[i] = false;
    }
    best
}

// ---------------------------------------------------------------------------
// Positions to compare on
// ---------------------------------------------------------------------------

fn board(name: &str) -> &'static Board {
    get_board(name).expect("a generated map")
}

/// A track layout, and the seats whose view of it gets checked.
struct Layout {
    board: &'static Board,
    seg_owner: Vec<u8>,
    seats: Vec<u8>,
    label: String,
}

fn empty_layout(board: &'static Board) -> Layout {
    Layout {
        board,
        seg_owner: vec![FREE; board.n_segments],
        seats: vec![0],
        label: format!("{} empty", board.name),
    }
}

/// Random track: each segment free with probability `free_percent`, else seat 0's, seat 1's,
/// or closed. Not every layout is reachable in play -- seat 0 may own both halves of a double,
/// or one half beside a free sibling it could never claim in 4-5P -- and the graph functions
/// must not care. Seat 1 sees seat 0's track as a rival's, and the other way round.
fn random_layouts(board: &'static Board, free_percents: &[u32], seed: u64) -> Vec<Layout> {
    let mut rng: Pcg32 = stream(seed, &[Part::Str("naive-graph"), Part::Str(board.name)]);
    free_percents
        .iter()
        .enumerate()
        .map(|(i, &free_percent)| Layout {
            board,
            seg_owner: (0..board.n_segments)
                .map(|_| {
                    if rng.below(100) < free_percent {
                        FREE
                    } else {
                        [0, 1, CLOSED][rng.below(3) as usize]
                    }
                })
                .collect(),
            seats: vec![0, 1],
            label: format!(
                "{} random #{i} ({free_percent}% free) seed {seed}",
                board.name
            ),
        })
        .collect()
}

/// The track layouts a uniformly random game passes through: one after every `every`-th
/// claim, and always the last. Every seat is checked.
fn game_layouts(map: &str, n_players: usize, seed: u64, every: usize) -> Vec<Layout> {
    let cfg = RuleConfig::new(map, n_players).expect("a supported configuration");
    let game = Game::new(cfg).expect("a supported configuration");
    let board = game.board;
    let mut state = game.new_initial_state(seed);
    let mut rng = stream(seed, &[Part::Str("policy")]);
    let mut claims = Vec::new();
    let mut version = state.board_version;
    while !state.is_terminal() {
        let action = state.sample_legal(&mut rng).expect("a non-terminal state");
        state.step(action).expect("a sampled action is legal");
        if state.board_version != version {
            version = state.board_version;
            claims.push(state.seg_owner[..board.n_segments].to_vec());
        }
    }
    let last = claims.len().saturating_sub(1);
    claims
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| (i + 1) % every == 0 || i == last)
        .map(|(i, seg_owner)| Layout {
            board,
            seg_owner,
            seats: (0..n_players as u8).collect(),
            label: format!("{map} {n_players}P seed {seed} after claim {}", i + 1),
        })
        .collect()
}

/// Real games on every map and seat count, `seeds` each.
fn all_game_layouts(seeds: std::ops::Range<u64>, every: usize) -> Vec<Layout> {
    let mut out = Vec::new();
    for b in crate::board::all_boards() {
        for n in usize::from(b.raw.min_players)..=usize::from(b.raw.max_players) {
            for seed in seeds.clone() {
                out.extend(game_layouts(b.name, n, seed, every));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Remaining cost: Dijkstra, all-pairs and A* against Floyd-Warshall
// ---------------------------------------------------------------------------

/// Every `GUIDED_STRIDE`-th crossable segment is priced out in turn for the A* check.
const GUIDED_STRIDE: usize = 7;

/// Compares every source on every layout. Returns how many (source, city) pairs were dead.
///
/// Then prices segments out one at a time, as `fragility` does, and checks the guided
/// search between every pair of cities against Floyd-Warshall without that segment.
fn check_remaining_costs(layouts: &[Layout]) -> usize {
    let mut dead = 0;
    for layout in layouts {
        let board = layout.board;
        let n = board.n_cities;
        for &seat in &layout.seats {
            for doubles_locked in [true, false] {
                let label = format!(
                    "{}: seat {seat}, doubles_locked={doubles_locked}",
                    layout.label
                );
                let expected = floyd_warshall(board, &layout.seg_owner, seat, doubles_locked, None);
                let cost = crossing_costs(board, &layout.seg_owner, seat, doubles_locked);
                let table = all_pairs(board, &cost);
                for source in 0..n {
                    let want = &expected[source];
                    assert_eq!(
                        &remaining_costs_from(
                            board,
                            &layout.seg_owner,
                            seat,
                            source as u8,
                            doubles_locked
                        ),
                        want,
                        "{label}: Dijkstra from city {source}"
                    );
                    assert_eq!(
                        &table[source][..n],
                        &want[..],
                        "{label}: all_pairs row {source}"
                    );
                    dead += want.iter().filter(|&&d| d == UNREACHABLE).count();
                }

                let crossable = (0..board.n_segments).filter(|&s| cost[s] < UNREACHABLE);
                for out in crossable.step_by(GUIDED_STRIDE) {
                    let without =
                        floyd_warshall(board, &layout.seg_owner, seat, doubles_locked, Some(out));
                    let mut priced_out = cost;
                    priced_out[out] = UNREACHABLE;
                    for (source, want) in without.iter().enumerate() {
                        for (target, &d) in want.iter().enumerate() {
                            assert_eq!(
                                guided_distance(
                                    board,
                                    &priced_out,
                                    &table[target],
                                    source as u8,
                                    target as u8
                                ),
                                d,
                                "{label}: A* from {source} to {target} without segment {out}"
                            );
                        }
                    }
                }
            }
        }
    }
    dead
}

/// The searches and Floyd-Warshall share nothing but `edge_cost`. On an empty board the
/// reference must also reproduce the static distance table, which a third piece of code
/// built.
#[test]
fn remaining_costs_match_floyd_warshall() {
    let mut layouts = Vec::new();
    for name in ["usa", "mini"] {
        let b = board(name);
        let empty = empty_layout(b);
        assert_eq!(
            floyd_warshall(b, &empty.seg_owner, 0, true, None),
            b.dist,
            "{name}: Floyd-Warshall disagrees with the static table"
        );
        layouts.push(empty);
        layouts.extend(random_layouts(b, &[90, 75, 60, 45, 30], 0));
    }
    layouts.extend(all_game_layouts(0..1, 3));
    assert!(
        check_remaining_costs(&layouts) > 0,
        "no layout cut a connection dead"
    );
}

#[test]
#[ignore = "sweep; run with --release -- --ignored"]
fn remaining_costs_match_floyd_warshall_sweep() {
    let mut layouts = Vec::new();
    for name in ["usa", "mini"] {
        for seed in 0..20 {
            layouts.extend(random_layouts(
                board(name),
                &[95, 85, 75, 65, 55, 45, 35, 25],
                seed,
            ));
        }
    }
    layouts.extend(all_game_layouts(0..20, 1));
    check_remaining_costs(&layouts);
}

// ---------------------------------------------------------------------------
// Steiner: Dreyfus-Wagner against every vertex set, on TTR-mini
// ---------------------------------------------------------------------------

/// Terminal lists to try on one layout: every pair, every city doubled, random multisets
/// of 3..=MAX_STEINER_TERMINALS (duplicates allowed), and random sets of more than
/// MAX_STEINER_TERMINALS cities, where the fallback can take over.
fn terminal_lists(board: &Board, rng: &mut Pcg32, small: usize, large: usize) -> Vec<Vec<u8>> {
    let n = board.n_cities as u8;
    let mut out: Vec<Vec<u8>> = Vec::new();
    for a in 0..n {
        out.push(vec![a, a]);
        for b in a + 1..n {
            out.push(vec![a, b]);
        }
    }
    for _ in 0..small {
        let len = 3 + rng.below(MAX_STEINER_TERMINALS as u32 - 2) as usize;
        out.push((0..len).map(|_| rng.below(u32::from(n)) as u8).collect());
    }
    for _ in 0..large {
        let mut cities: Vec<u8> = (0..n).collect();
        rng.shuffle(&mut cities);
        let len = MAX_STEINER_TERMINALS
            + 1
            + rng.below(u32::from(n) - MAX_STEINER_TERMINALS as u32) as usize;
        cities.truncate(len);
        out.push(cities);
    }
    out
}

/// Which branch of `steiner_cost_exact` one query exercised.
#[derive(Clone, Copy, Debug)]
enum Outcome {
    /// Some terminal is cut off from the rest.
    Cut,
    /// More than `MAX_STEINER_TERMINALS` pieces: the MST fallback.
    Fallback,
    /// Exact, after own track merged some terminals into one piece.
    Contracted,
    /// Exact, every distinct terminal its own piece.
    Plain,
}

/// One Steiner query against the brute force.
///
/// Cut off: `UNREACHABLE`, reported as exact. More than `MAX_STEINER_TERMINALS` pieces: the
/// documented fallback, flagged inexact, never below the optimum and -- the classic MST
/// bound -- never above twice it. Otherwise: exactly the optimum.
fn check_steiner(
    layout: &Layout,
    seat: u8,
    doubles_locked: bool,
    tree: &[Option<u32>],
    terminals: &[u8],
) -> Outcome {
    let board = layout.board;
    let optimum = cheapest_superset(tree, terminals);
    let got = steiner_cost_exact(board, &layout.seg_owner, seat, terminals, doubles_locked);
    let pieces = pieces(board, &layout.seg_owner, seat, terminals);
    let context = format!(
        "{}: seat {seat}, doubles_locked={doubles_locked}, terminals {terminals:?} \
         ({pieces} pieces), brute force {optimum:?}",
        layout.label
    );
    let Some(best) = optimum else {
        assert_eq!(got, (u32::from(UNREACHABLE), true), "{context}");
        return Outcome::Cut;
    };
    if pieces > MAX_STEINER_TERMINALS {
        let (cost, exact) = got;
        assert!(!exact, "{context}: past the cap, but reported exact");
        assert!(
            best <= cost && cost <= 2 * best,
            "{context}: fallback {cost} is outside [{best}, {}]",
            2 * best
        );
        return Outcome::Fallback;
    }
    assert_eq!(got, (best, true), "{context}");
    if pieces < terminals.iter().collect::<BTreeSet<_>>().len() {
        Outcome::Contracted
    } else {
        Outcome::Plain
    }
}

fn steiner_layouts(seeds: std::ops::Range<u64>, every: usize) -> Vec<Layout> {
    let mini = board("mini");
    let mut layouts = vec![empty_layout(mini)];
    for seed in seeds.clone() {
        layouts.extend(random_layouts(mini, &[85, 65, 45], seed));
    }
    for n in 2..=4 {
        for seed in seeds.clone() {
            layouts.extend(game_layouts("mini", n, seed, every));
        }
    }
    layouts
}

#[test]
fn steiner_matches_a_brute_force_on_mini() {
    let mini = board("mini");
    // The brute force itself, on an empty board, against the static distance table.
    let empty = spanning_trees(mini, &vec![FREE; mini.n_segments], 0, true);
    for a in 0..mini.n_cities as u8 {
        for b in 0..mini.n_cities as u8 {
            assert_eq!(
                cheapest_superset(&empty, &[a, b]),
                Some(u32::from(mini.dist[usize::from(a)][usize::from(b)])),
                "brute force {a}-{b}"
            );
        }
    }

    let mut rng = stream(0, &[Part::Str("naive-steiner")]);
    let mut seen = [0usize; 4];
    for layout in steiner_layouts(0..2, 3) {
        // One seat per layout keeps a debug run short; the sweep checks every seat.
        let seat = layout.seats[rng.below(layout.seats.len() as u32) as usize];
        for doubles_locked in [true, false] {
            let tree = spanning_trees(mini, &layout.seg_owner, seat, doubles_locked);
            for terminals in terminal_lists(mini, &mut rng, 12, 4) {
                seen[check_steiner(&layout, seat, doubles_locked, &tree, &terminals) as usize] += 1;
            }
        }
    }
    // Cut, Fallback, Contracted, Plain: a comparison that never reaches a branch tests nothing.
    assert!(
        seen.iter().all(|&n| n > 0),
        "branches never reached: {seen:?}"
    );
}

/// Every seat, both rules, many more layouts and queries -- and on every twentieth layout,
/// every set of two or more cities there is.
#[test]
#[ignore = "sweep; run with --release -- --ignored"]
fn steiner_matches_a_brute_force_on_mini_sweep() {
    let mini = board("mini");
    let n = mini.n_cities;
    let every_set: Vec<Vec<u8>> = (0..1usize << n)
        .filter(|set| set.count_ones() >= 2)
        .map(|set| (0..n as u8).filter(|&c| set >> c & 1 == 1).collect())
        .collect();
    let mut rng = stream(1, &[Part::Str("naive-steiner")]);
    for (i, layout) in steiner_layouts(0..5, 2).iter().enumerate() {
        for &seat in &layout.seats {
            for doubles_locked in [true, false] {
                let tree = spanning_trees(mini, &layout.seg_owner, seat, doubles_locked);
                let sampled = terminal_lists(mini, &mut rng, 300, 60);
                let lists = if i % 20 == 0 { &every_set } else { &sampled };
                for terminals in lists {
                    check_steiner(layout, seat, doubles_locked, &tree, terminals);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Longest trail: the layered search against an exhaustive one
// ---------------------------------------------------------------------------

fn check_longest_trail(board: &Board, seg_owner: &[u8], player: u8, label: &str) {
    assert_eq!(
        longest_trail(board, seg_owner, player),
        exhaustive_longest_trail(board, seg_owner, player),
        "{label}: seat {player}, segments {:?}",
        (0..board.n_segments)
            .filter(|&s| seg_owner[s] == player)
            .map(|s| board.segment_name(s))
            .collect::<Vec<_>>()
    );
}

/// Shapes picked to drive each layer: stars and spiders whose answer is the two longest
/// legs, figure-eights and thetas that the Eulerian shortcut takes whole until a tail breaks
/// it, both tracks of a double (one seat cannot own both in play, but a parallel pair is
/// exactly what a search keyed on city pairs rather than segments gets wrong), and several
/// components at once.
#[test]
fn longest_trail_matches_an_exhaustive_search_on_hand_built_shapes() {
    // `(a, b, which)`: the `which`-th track between two cities, as in tests/common.
    type Shape = &'static [(&'static str, &'static str, usize)];
    let usa_shapes: [Shape; 7] = [
        // A spider at Denver with four uneven legs: the answer is the two longest.
        &[
            ("Denver", "Helena", 0),
            ("Helena", "Winnipeg", 0),
            ("Denver", "Omaha", 0),
            ("Omaha", "Chicago", 0),
            ("Denver", "Santa Fe", 0),
            ("Santa Fe", "El Paso", 0),
            ("El Paso", "Houston", 0),
            ("Denver", "Phoenix", 0),
        ],
        // Two cycles through Atlanta: an Eulerian circuit, taken whole...
        &[
            ("Atlanta", "New Orleans", 0),
            ("New Orleans", "Little Rock", 0),
            ("Little Rock", "Nashville", 0),
            ("Nashville", "Atlanta", 0),
            ("Atlanta", "Raleigh", 0),
            ("Raleigh", "Charleston", 0),
            ("Charleston", "Atlanta", 0),
        ],
        // ...until two tails make four odd cities and the DFS has to run.
        &[
            ("Atlanta", "New Orleans", 0),
            ("New Orleans", "Little Rock", 0),
            ("Little Rock", "Nashville", 0),
            ("Nashville", "Atlanta", 0),
            ("Atlanta", "Raleigh", 0),
            ("Raleigh", "Charleston", 0),
            ("Charleston", "Atlanta", 0),
            ("Atlanta", "Miami", 0),
            ("Raleigh", "Washington", 0),
        ],
        // A theta, Kansas City to Oklahoma City three ways, plus a tail off Denver.
        &[
            ("Kansas City", "Oklahoma City", 0),
            ("Kansas City", "Saint Louis", 0),
            ("Saint Louis", "Little Rock", 0),
            ("Little Rock", "Oklahoma City", 0),
            ("Kansas City", "Denver", 0),
            ("Denver", "Oklahoma City", 0),
            ("Denver", "Santa Fe", 0),
        ],
        // Both Seattle-Portland tracks, with a spur at each end.
        &[
            ("Seattle", "Portland", 0),
            ("Seattle", "Portland", 1),
            ("Seattle", "Vancouver", 0),
            ("Portland", "San Francisco", 0),
        ],
        // A diamond with three spurs off Oklahoma City.
        &[
            ("Denver", "Kansas City", 0),
            ("Denver", "Omaha", 0),
            ("Kansas City", "Omaha", 0),
            ("Denver", "Oklahoma City", 0),
            ("Kansas City", "Oklahoma City", 0),
            ("Oklahoma City", "Dallas", 0),
            ("Oklahoma City", "Santa Fe", 0),
            ("Oklahoma City", "Little Rock", 0),
        ],
        // Three components, the largest not first: the answer is one of them, never a sum.
        &[
            ("Boston", "New York", 0),
            ("Miami", "New Orleans", 0),
            ("New Orleans", "Houston", 0),
            ("Houston", "Dallas", 0),
            ("Dallas", "Little Rock", 0),
            ("Dallas", "Oklahoma City", 0),
            ("Calgary", "Winnipeg", 0),
        ],
    ];
    let mini_shapes: [Shape; 4] = [
        // The south-west mesh with a spur to Las Vegas: four odd cities.
        &[
            ("El Paso", "Phoenix", 0),
            ("El Paso", "Santa Fe", 0),
            ("Phoenix", "Santa Fe", 0),
            ("Los Angeles", "Phoenix", 0),
            ("El Paso", "Los Angeles", 0),
            ("Denver", "Santa Fe", 0),
            ("Denver", "Phoenix", 0),
            ("Las Vegas", "Los Angeles", 0),
        ],
        // Both north-west doubles in a tangle of triangles, plus a tail to Winnipeg.
        &[
            ("Seattle", "Vancouver", 0),
            ("Seattle", "Vancouver", 1),
            ("Portland", "Seattle", 0),
            ("Portland", "Seattle", 1),
            ("Calgary", "Vancouver", 0),
            ("Calgary", "Seattle", 0),
            ("Calgary", "Helena", 0),
            ("Helena", "Seattle", 0),
            ("Helena", "Winnipeg", 0),
        ],
        // A five-legged star at Salt Lake City.
        &[
            ("Denver", "Salt Lake City", 0),
            ("Helena", "Salt Lake City", 0),
            ("Las Vegas", "Salt Lake City", 0),
            ("Portland", "Salt Lake City", 0),
            ("Salt Lake City", "San Francisco", 0),
        ],
        // Two of San Francisco's doubles, a triangle through Salt Lake City, a tail south.
        &[
            ("Portland", "San Francisco", 0),
            ("Portland", "San Francisco", 1),
            ("Salt Lake City", "San Francisco", 0),
            ("Salt Lake City", "San Francisco", 1),
            ("Portland", "Salt Lake City", 0),
            ("Los Angeles", "San Francisco", 0),
            ("Las Vegas", "Los Angeles", 0),
        ],
    ];
    for (b, shapes) in [
        (board("usa"), &usa_shapes[..]),
        (board("mini"), &mini_shapes[..]),
    ] {
        for (i, shape) in shapes.iter().enumerate() {
            let mut owner = vec![FREE; b.n_segments];
            for &(x, y, which) in shape.iter() {
                owner[segment_named(b, x, y, which)] = 0;
            }
            check_longest_trail(b, &owner, 0, &format!("{} shape #{i}", b.name));
        }
    }
}

/// The `which`-th segment between two cities, by name.
fn segment_named(board: &Board, a: &str, b: &str, which: usize) -> usize {
    let index = |name: &str| {
        board
            .raw
            .cities
            .iter()
            .position(|&c| c == name)
            .unwrap_or_else(|| panic!("{} has no city {name:?}", board.name)) as u8
    };
    let (ia, ib) = (index(a).min(index(b)), index(a).max(index(b)));
    (0..board.n_segments)
        .filter(|&s| (board.seg_a[s], board.seg_b[s]) == (ia, ib))
        .nth(which)
        .unwrap_or_else(|| panic!("no segment #{which} between {a} and {b}"))
}

/// Uniformly random sets of track, seat 0 owning them and seat 1 or `CLOSED` a scattering of
/// the rest; every tenth set is the largest allowed.
///
/// Sizes are capped per map. TTR-mini at 16 of 30 segments is already a tangle of cycles and
/// doubles that no seat could own (9 is the most its train supply buys), and the exhaustive
/// search grows steeply past it. USA goes to [`MAX_TRAIL_EDGES`] = 32, so no random set can
/// form a component wider than the DFS mask, which is a documented panic. (This reference
/// found that `longest_trail` used to number mask bits across *all* the seat's segments
/// rather than within a component, so a 33rd segment overflowed the shift even when every
/// component was small; fixed, and pinned by
/// `graph::tests::a_seat_with_more_than_32_segments_still_gets_its_whole_trail`.)
fn check_random_track(seeds: std::ops::Range<u64>, per_seed: usize) {
    for (name, max_size) in [("usa", MAX_TRAIL_EDGES), ("mini", 16)] {
        let b = board(name);
        for seed in seeds.clone() {
            let mut rng = stream(seed, &[Part::Str("naive-trail"), Part::Str(name)]);
            for i in 0..per_seed {
                let size = if i % 10 == 0 {
                    max_size
                } else {
                    1 + rng.below(max_size as u32) as usize
                };
                let mut segments: Vec<usize> = (0..b.n_segments).collect();
                rng.shuffle(&mut segments);
                let mut owner = vec![FREE; b.n_segments];
                for &s in &segments[..size] {
                    owner[s] = 0;
                }
                for &s in &segments[size..size + 4] {
                    owner[s] = [1, CLOSED][rng.below(2) as usize];
                }
                check_longest_trail(b, &owner, 0, &format!("{name} seed {seed} random #{i}"));
            }
        }
    }
}

#[test]
fn longest_trail_matches_an_exhaustive_search_on_random_track() {
    check_random_track(0..1, 100);
}

/// Everything a seat owns after every claim of real games, on both maps and every seat
/// count: the full subgraphs, not just the small ones a permutation oracle can afford.
#[test]
fn longest_trail_matches_an_exhaustive_search_on_real_game_subgraphs() {
    for layout in all_game_layouts(0..8, 1) {
        for &seat in &layout.seats {
            check_longest_trail(layout.board, &layout.seg_owner, seat, &layout.label);
        }
    }
}

#[test]
#[ignore = "sweep; run with --release -- --ignored"]
fn longest_trail_matches_an_exhaustive_search_sweep() {
    check_random_track(0..40, 500);
    for layout in all_game_layouts(0..200, 1) {
        for &seat in &layout.seats {
            check_longest_trail(layout.board, &layout.seg_owner, seat, &layout.label);
        }
    }
}
