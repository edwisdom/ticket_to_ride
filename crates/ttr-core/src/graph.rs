//! Graph work the engine needs. Everything here is a pure function of its arguments.
//!
//! No caching: callers that want it own it, because they know when it goes stale. The
//! union-find is the only piece `step()` touches; the rest is used by scoring and the
//! observation encoder.
//!
//! Deliberately not a graph library. The two things actually needed are short, run
//! millions of times, and no crate has a longest-*trail* function at all.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::board::{Board, NO_SIBLING, UNREACHABLE};
use crate::config::FREE;

/// The longest-trail DFS keeps used edges in one machine word.
pub const MAX_TRAIL_EDGES: usize = 32;

/// Fewer than two terminals is nothing to connect.
pub const MIN_TERMINALS: usize = 2;

/// Above this many distinct terminal components the subset DP is abandoned for a
/// minimum-spanning-tree upper bound. Held-ticket terminal sets are almost always smaller.
pub const MAX_STEINER_TERMINALS: usize = 8;

/// Path *halving*: one pointer update per step, no second pass, no recursion.
///
/// A clone-based state can compress freely; an undo-log design could not, which is one of
/// the reasons the state is cloned rather than journalled (PLAN.md §5.1).
///
/// Note that halving does **not** flatten a chain in one pass -- a test asserting
/// `parent[x] == root` after a single find on a long chain will fail. Repeated finds
/// converge, and that is the trade being made.
#[inline]
pub fn dsu_find(parent: &mut [u8], mut x: u8) -> u8 {
    while parent[x as usize] != x {
        let grandparent = parent[parent[x as usize] as usize];
        parent[x as usize] = grandparent;
        x = grandparent;
    }
    x
}

/// Union by **smaller index**, so the forest is a deterministic function of the claims
/// rather than of the order they happened to be applied in.
pub fn dsu_union(parent: &mut [u8], a: u8, b: u8) -> bool {
    let (mut ra, mut rb) = (dsu_find(parent, a), dsu_find(parent, b));
    if ra == rb {
        return false;
    }
    if ra > rb {
        std::mem::swap(&mut ra, &mut rb);
    }
    parent[rb as usize] = ra;
    true
}

#[inline]
pub fn dsu_connected(parent: &mut [u8], a: u8, b: u8) -> bool {
    dsu_find(parent, a) == dsu_find(parent, b)
}

// ---------------------------------------------------------------------------
// Remaining cost -- "how many more train cars do I need?"
// ---------------------------------------------------------------------------

/// Cost in train cars for `player` to traverse `segment`, or [`UNREACHABLE`].
///
/// Already mine: free. Unclaimed and still claimable by me: its length. Anyone else's,
/// closed, or the sibling of a track I already own in a 4-5P game: impassable.
#[inline]
pub fn edge_cost(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    segment: usize,
    doubles_locked: bool,
) -> u16 {
    let owner = seg_owner[segment];
    if owner == player {
        return 0;
    }
    if owner != FREE {
        return UNREACHABLE;
    }
    if !doubles_locked {
        // 4-5P: the sibling stays open to *others*, never to me.
        let sibling = board.sibling[segment];
        if sibling != NO_SIBLING && seg_owner[sibling as usize] == player {
            return UNREACHABLE;
        }
    }
    u16::from(board.seg_len[segment])
}

/// Dijkstra from `source` with my track free, free track priced, enemy track blocked.
///
/// The result is "train cars still needed" to reach each city. [`UNREACHABLE`] means the
/// connection is dead -- an opponent has cut every route -- which is exactly the signal a
/// ticket-valuation heuristic needs and the `is_dead` observation feature reports.
pub fn remaining_costs_from(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    source: u8,
    doubles_locked: bool,
) -> Vec<u16> {
    let mut dist = vec![UNREACHABLE; board.n_cities];
    dist[source as usize] = 0;
    let mut queue = BinaryHeap::new();
    queue.push(Reverse((0u16, source)));
    while let Some(Reverse((d, u))) = queue.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for &(nb, segment) in &board.adjacency[u as usize] {
            let w = edge_cost(board, seg_owner, player, segment as usize, doubles_locked);
            if w >= UNREACHABLE {
                continue;
            }
            let nd = d + w;
            if nd < dist[nb as usize] {
                dist[nb as usize] = nd;
                queue.push(Reverse((nd, nb)));
            }
        }
    }
    dist
}

/// Cheapest set of new track connecting every terminal, and whether it is exact.
///
/// Dreyfus-Wagner. Why this and not the sum of per-ticket shortest paths: those
/// double-count shared trunk lines, so a player holding Seattle-New York and
/// Portland-New York looks like it needs two transcontinentals when it needs one and a
/// spur. The Steiner cost is the truth, and it is what the H2+ heuristics plan against.
///
/// Terminals already connected to each other are contracted first, which is what usually
/// keeps the subset DP small: mid-game most held tickets share a component. Above
/// [`MAX_STEINER_TERMINALS`] distinct components the DP is abandoned and a
/// minimum-spanning-tree approximation over the terminal metric is returned instead --
/// documented, upper-bounding, and never silent (the second return value is `false`).
///
/// Widths follow Python's unbounded integers rather than the engine's `u16`: combining
/// eight terminals can sum eight [`UNREACHABLE`] sentinels, which overflows a `u16` and
/// would turn "no such tree" into a small, plausible, wrong number.
pub fn steiner_cost_exact(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    terminals: &[u8],
    doubles_locked: bool,
) -> (u32, bool) {
    if terminals.len() < MIN_TERMINALS {
        return (0, true);
    }

    let mut unique: Vec<u8> = terminals.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let distances: HashMap<u8, Vec<u16>> = unique
        .iter()
        .map(|&t| {
            (
                t,
                remaining_costs_from(board, seg_owner, player, t, doubles_locked),
            )
        })
        .collect();

    // Contract terminals that already cost nothing to reach from one another.
    let mut roots: Vec<u8> = Vec::new();
    for &t in &unique {
        let from_t = &distances[&t];
        if !roots.iter().any(|&r| from_t[r as usize] == 0) {
            roots.push(t);
        }
    }
    if roots.len() < MIN_TERMINALS {
        return (0, true);
    }
    let from_first = &distances[&roots[0]];
    if roots.iter().any(|&r| from_first[r as usize] >= UNREACHABLE) {
        return (u32::from(UNREACHABLE), true);
    }
    if roots.len() > MAX_STEINER_TERMINALS {
        return (mst_bound(&roots, &distances), false);
    }

    let n = board.n_cities;
    let full = (1usize << roots.len()) - 1;
    // dp[mask][v] = cheapest tree spanning the terminals in `mask` plus the vertex v.
    let mut dp = vec![vec![u32::from(UNREACHABLE); n]; full + 1];
    for (i, t) in roots.iter().enumerate() {
        let source = &distances[t];
        for v in 0..n {
            dp[1 << i][v] = u32::from(source[v]);
        }
    }

    for mask in 1..=full {
        if mask & (mask - 1) == 0 {
            continue; // single terminal: already initialized from its Dijkstra
        }
        // Lifted out of the loop: `sub` and `mask ^ sub` are both proper non-empty subsets
        // of `mask`, so neither aliases this row, and taking it once lets the inner loop
        // zip three slices instead of indexing three rows of a `Vec<Vec<_>>`.
        let mut row = std::mem::take(&mut dp[mask]);
        let mut sub = (mask - 1) & mask;
        while sub != 0 {
            let (left, other) = (&dp[sub], &dp[mask ^ sub]);
            for ((cell, &a), &b) in row.iter_mut().zip(left).zip(other) {
                let combined = a + b;
                if combined < *cell {
                    *cell = combined;
                }
            }
            sub = (sub - 1) & mask;
        }
        relax(board, seg_owner, player, &mut row, doubles_locked);
        dp[mask] = row;
    }

    (*dp[full].iter().min().expect("at least one city"), true)
}

/// Cheapest set of new track connecting every terminal. See [`steiner_cost_exact`].
pub fn steiner_cost(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    terminals: &[u8],
    doubles_locked: bool,
) -> u32 {
    steiner_cost_exact(board, seg_owner, player, terminals, doubles_locked).0
}

/// Multi-source Dijkstra in place: let the tree's root slide along cheap track.
fn relax(board: &Board, seg_owner: &[u8], player: u8, dist: &mut [u32], doubles_locked: bool) {
    let mut queue: BinaryHeap<Reverse<(u32, u8)>> = dist
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d < u32::from(UNREACHABLE))
        .map(|(v, &d)| Reverse((d, v as u8)))
        .collect();
    while let Some(Reverse((d, u))) = queue.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for &(nb, segment) in &board.adjacency[u as usize] {
            let w = edge_cost(board, seg_owner, player, segment as usize, doubles_locked);
            if w >= UNREACHABLE {
                continue;
            }
            let nd = d + u32::from(w);
            if nd < dist[nb as usize] {
                dist[nb as usize] = nd;
                queue.push(Reverse((nd, nb)));
            }
        }
    }
}

/// Prim over the terminal metric. Always >= the Steiner cost, never below it.
fn mst_bound(roots: &[u8], distances: &HashMap<u8, Vec<u16>>) -> u32 {
    let mut inside = vec![roots[0]];
    let mut total = 0u32;
    while inside.len() < roots.len() {
        let mut best = UNREACHABLE;
        let mut best_t: Option<u8> = None;
        for &t in roots {
            if inside.contains(&t) {
                continue;
            }
            for &u in &inside {
                if distances[&u][t as usize] < best {
                    best = distances[&u][t as usize];
                    best_t = Some(t);
                }
            }
        }
        match best_t {
            Some(t) => {
                inside.push(t);
                total += u32::from(best);
            }
            None => return u32::from(UNREACHABLE),
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Longest continuous path -- a longest *trail*, weighted in train cars
// ---------------------------------------------------------------------------

/// The longest continuous path bonus, measured in train cars.
///
/// It is a **trail**, not a path and not a segment count: each segment may be used at most
/// once, cities may repeat, and loops are allowed. Four layers, in order, because the
/// naive version has a 126 ms tail that shows up as a p99.9 latency spike in self-play:
///
/// 1. **Split into connected components.** The answer is the max over components; a 6-edge
///    and a 4-edge component are two tiny searches instead of one 10-edge one.
/// 2. **Eulerian shortcut.** A component with 0 or 2 odd-degree vertices admits a trail
///    using *every* edge, so the answer is the total weight with no search at all. This
///    fires on a large fraction of real player subgraphs.
/// 3. **Memoized DFS** on `(vertex, used-edge bitmask)`, the mask in one machine word.
/// 4. **Early exit** as soon as the component's total weight is reached.
///
/// # Panics
/// If a component holds more than [`MAX_TRAIL_EDGES`] edges. The train supply makes that
/// unreachable, and a panic beats silently truncating the mask.
pub fn longest_trail(board: &Board, seg_owner: &[u8], player: u8) -> u16 {
    let owned: Vec<usize> = (0..board.n_segments)
        .filter(|&s| seg_owner[s] == player)
        .collect();
    if owned.is_empty() {
        return 0;
    }

    // city -> [(neighbour, local edge index, weight)]
    let mut adjacency: HashMap<u8, Vec<(u8, usize, u16)>> = HashMap::new();
    for (local, &segment) in owned.iter().enumerate() {
        let (a, b, w) = (
            board.seg_a[segment],
            board.seg_b[segment],
            u16::from(board.seg_len[segment]),
        );
        adjacency.entry(a).or_default().push((b, local, w));
        adjacency.entry(b).or_default().push((a, local, w));
    }

    components(&adjacency)
        .into_iter()
        .map(|component| component_longest_trail(&adjacency, &component))
        .max()
        .unwrap_or(0)
}

fn components(adjacency: &HashMap<u8, Vec<(u8, usize, u16)>>) -> Vec<Vec<u8>> {
    let mut starts: Vec<u8> = adjacency.keys().copied().collect();
    starts.sort_unstable();
    let mut seen: Vec<u8> = Vec::new();
    let mut out = Vec::new();
    for start in starts {
        if seen.contains(&start) {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        seen.push(start);
        while let Some(v) = stack.pop() {
            component.push(v);
            for &(nb, _, _) in &adjacency[&v] {
                if !seen.contains(&nb) {
                    seen.push(nb);
                    stack.push(nb);
                }
            }
        }
        out.push(component);
    }
    out
}

fn component_longest_trail(
    adjacency: &HashMap<u8, Vec<(u8, usize, u16)>>,
    component: &[u8],
) -> u16 {
    let mut edges: HashMap<usize, u16> = HashMap::new();
    for v in component {
        for &(_, local, w) in &adjacency[v] {
            edges.insert(local, w);
        }
    }
    let total: u16 = edges.values().sum();

    let odd = component
        .iter()
        .filter(|v| adjacency[v].len() % 2 == 1)
        .count();
    if odd == 0 || odd == 2 {
        return total; // layer 2: an Eulerian trail uses every edge
    }

    assert!(
        edges.len() <= MAX_TRAIL_EDGES,
        "longest-trail component has {} edges, above the {MAX_TRAIL_EDGES}-bit mask limit; \
         the train supply should make this unreachable",
        edges.len()
    );

    let mut memo: HashMap<(u8, u32), u16> = HashMap::new();
    let mut overall = 0;
    for &start in component {
        overall = overall.max(best_from(adjacency, &mut memo, start, 0));
        if overall == total {
            break; // layer 4: cannot do better than every edge
        }
    }
    overall
}

fn best_from(
    adjacency: &HashMap<u8, Vec<(u8, usize, u16)>>,
    memo: &mut HashMap<(u8, u32), u16>,
    v: u8,
    used: u32,
) -> u16 {
    if let Some(&cached) = memo.get(&(v, used)) {
        return cached;
    }
    let mut best = 0;
    for &(nb, local, w) in &adjacency[&v] {
        let bit = 1u32 << local;
        if used & bit != 0 {
            continue;
        }
        best = best.max(w + best_from(adjacency, memo, nb, used | bit));
    }
    memo.insert((v, used), best);
    best
}

/// Segment ids `player` has claimed. `CLOSED` siblings are nobody's, and excluded.
pub fn owned_segments(seg_owner: &[u8], player: u8, n_segments: usize) -> Vec<usize> {
    (0..n_segments)
        .filter(|&s| seg_owner[s] == player)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(n: u8) -> Vec<u8> {
        (0..n).collect()
    }

    #[test]
    fn union_and_find_agree() {
        let mut p = fresh(10);
        assert!(!dsu_connected(&mut p, 0, 5));
        assert!(dsu_union(&mut p, 0, 5));
        assert!(dsu_connected(&mut p, 0, 5));
        assert!(
            !dsu_union(&mut p, 5, 0),
            "a redundant union must report false"
        );
    }

    #[test]
    fn the_root_is_the_smallest_index_in_the_component() {
        let mut p = fresh(10);
        for (a, b) in [(7, 3), (3, 9), (9, 1)] {
            dsu_union(&mut p, a, b);
        }
        for v in [1, 3, 7, 9] {
            assert_eq!(dsu_find(&mut p, v), 1);
        }
    }

    #[test]
    fn path_halving_does_not_flatten_in_one_pass() {
        // Documented, not a defect: halving updates one pointer per step. A test that
        // asserts full flattening after a single find is testing path *compression*, a
        // different algorithm, and would fail here for the right reason.
        let mut p: Vec<u8> = (0..20).collect();
        for i in (1..20).rev() {
            p[i] = (i - 1) as u8;
        }
        assert_eq!(dsu_find(&mut p, 19), 0);
        assert_ne!(p[19], 0, "one halving pass should not reach the root");
        for _ in 0..10 {
            dsu_find(&mut p, 19);
        }
        assert_eq!(p[19], 0, "repeated finds must converge");
    }
}
