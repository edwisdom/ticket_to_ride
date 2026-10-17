//! Graph work the engine needs. Everything here is a pure function of its arguments.
//!
//! No caching: callers that want it own it, because they know when it goes stale. The
//! union-find is the only piece `step()` touches; the rest is used by scoring, the
//! heuristics and the observation encoder.
//!
//! Nothing here allocates on the hot path: searches run over fixed arrays sized to the
//! largest board, and the all-pairs table ([`all_pairs`]) is what the Steiner DP and the
//! encoder read their distances from. Results are exact and unique -- shortest distances,
//! the minimum Steiner cost, the MST weight -- so they do not depend on search order, and
//! the `naive_graph` references and the `oracle-v1` differential check them as values.
//!
//! Deliberately not a graph library. The two things actually needed are short, run
//! millions of times, and no crate has a longest-*trail* function at all.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::board::{Board, MAX_CITIES, MAX_ROUTE_LEN, MAX_SEGMENTS, NO_SIBLING, UNREACHABLE};
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

/// Every segment's [`edge_cost`] for `player`, once, so the searches below read a table
/// instead of re-deriving ownership and the double-route rule at every relaxation.
pub fn crossing_costs(
    board: &Board,
    seg_owner: &[u8],
    player: u8,
    doubles_locked: bool,
) -> [u16; MAX_SEGMENTS] {
    let mut cost = [UNREACHABLE; MAX_SEGMENTS];
    for (segment, c) in cost.iter_mut().enumerate().take(board.n_segments) {
        *c = edge_cost(board, seg_owner, player, segment, doubles_locked);
    }
    cost
}

/// Dijkstra from `source` over a crossing-cost table, into `dist` (at least `n_cities`).
///
/// A bucket queue (Dial's algorithm), allocation-free: crossing costs are small integers --
/// 0 for my own track, a route's length for free track, at most `max_len` -- so the open
/// distances always lie in a window `max_len + 1` wide, and a ring of that many city
/// bitmasks is the whole priority queue. Zero-cost track lands back in the bucket being
/// drained and is picked up in the same round.
///
/// Exact shortest distances are unique, so this agrees with any other correct search; the
/// `naive_graph` Floyd-Warshall reference checks it.
fn dijkstra(board: &Board, cost: &[u16], source: u8, dist: &mut [u16]) {
    const RING: usize = MAX_ROUTE_LEN + 1;
    debug_assert!(usize::from(board.max_len) < RING);
    let n = board.n_cities;
    dist[..n].fill(UNREACHABLE);
    dist[usize::from(source)] = 0;
    let mut ring = [0u64; RING];
    ring[0] = 1 << source;
    let mut settled = 0u64;
    let mut current: u16 = 0;
    loop {
        let slot = usize::from(current) % RING;
        while ring[slot] != 0 {
            let mut batch = std::mem::take(&mut ring[slot]);
            while batch != 0 {
                let here = batch.trailing_zeros() as usize;
                batch &= batch - 1;
                if settled >> here & 1 == 1 || dist[here] != current {
                    continue;
                }
                settled |= 1 << here;
                for &(nb, segment) in &board.adjacency[here] {
                    let w = cost[usize::from(segment)];
                    if w >= UNREACHABLE {
                        continue;
                    }
                    let nd = current + w;
                    let nb = usize::from(nb);
                    if nd < dist[nb] {
                        dist[nb] = nd;
                        ring[usize::from(nd) % RING] |= 1 << nb;
                    }
                }
            }
        }
        if ring.iter().all(|&b| b == 0) {
            return;
        }
        current += 1;
    }
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
    let cost = crossing_costs(board, seg_owner, player, doubles_locked);
    let mut dist = vec![UNREACHABLE; board.n_cities];
    dijkstra(board, &cost, source, &mut dist);
    dist
}

/// The cheapest crossing cost from `source` to `target` over `cost` ([`UNREACHABLE`] if
/// there is no route), by A* search: every city's distance to `target` over some cheaper
/// cost table is already known (`toward`), and it steers the search.
///
/// `toward` must be a consistent lower bound for `cost`: for every crossable segment from
/// x to y, `toward[x] <= cost + toward[y]`. Distances to `target` over a table that `cost`
/// only raises are exactly that -- the encoder's case: `fragility` prices one segment out
/// of my table and reads `toward` from the all-pairs table of the original. The search then
/// settles cities in order of `g + toward`, which is non-decreasing, and stops at `target`
/// with the exact distance; a city that cannot reach `target` over the cheaper table cannot
/// over `cost` either, and is never queued. The ring is twice as wide as Dijkstra's
/// because a step raises `g + toward` by up to twice a segment's cost.
pub fn guided_distance(board: &Board, cost: &[u16], toward: &[u16], source: u8, target: u8) -> u16 {
    const RING: usize = 2 * MAX_ROUTE_LEN + 1;
    debug_assert!(2 * usize::from(board.max_len) < RING);
    let start = toward[usize::from(source)];
    if start >= UNREACHABLE {
        return UNREACHABLE;
    }
    let mut g = [UNREACHABLE; MAX_CITIES];
    g[usize::from(source)] = 0;
    let mut ring = [0u64; RING];
    ring[usize::from(start) % RING] = 1 << source;
    let mut settled = 0u64;
    let mut current = start;
    loop {
        let slot = usize::from(current) % RING;
        while ring[slot] != 0 {
            let mut batch = std::mem::take(&mut ring[slot]);
            while batch != 0 {
                let here = batch.trailing_zeros() as usize;
                batch &= batch - 1;
                if settled >> here & 1 == 1 || g[here] + toward[here] != current {
                    continue;
                }
                if here == usize::from(target) {
                    return g[here];
                }
                settled |= 1 << here;
                for &(nb, segment) in &board.adjacency[here] {
                    let w = cost[usize::from(segment)];
                    let nb = usize::from(nb);
                    let h = toward[nb];
                    if w >= UNREACHABLE || h >= UNREACHABLE {
                        continue;
                    }
                    let ng = g[here] + w;
                    if ng < g[nb] {
                        debug_assert!(ng + h >= current, "`toward` is not consistent");
                        g[nb] = ng;
                        ring[usize::from(ng + h) % RING] |= 1 << nb;
                    }
                }
            }
        }
        if ring.iter().all(|&b| b == 0) {
            return UNREACHABLE;
        }
        current += 1;
    }
}

/// Rows of the all-pairs table are padded to this many entries, a multiple of the SIMD
/// width, so the min-plus loops below vectorize without a remainder.
pub const DIST_STRIDE: usize = MAX_CITIES.next_multiple_of(8);

/// All-pairs crossing costs: `dist[a][b]` is what [`remaining_costs_from`] from `a` would
/// say about `b`, for every pair, [`UNREACHABLE`] where there is no route.
pub type Distances = [[u16; DIST_STRIDE]; DIST_STRIDE];

/// Floyd-Warshall over a crossing-cost table.
///
/// One pass yields every row the encoder and the Steiner DP would otherwise each run a
/// Dijkstra for -- every ticket endpoint, every terminal -- and the inner loop is a
/// fixed-width min-plus that compiles to straight SIMD. Shortest distances are unique, so
/// this agrees with the Dijkstra in [`remaining_costs_from`] entry for entry.
///
/// Sums stay inside `u16`: a finite distance is under the sentinel, so a finite plus the
/// sentinel is at most `2 * UNREACHABLE - 1`, and taking the minimum against a cell that
/// is at most the sentinel keeps every entry at most the sentinel.
pub fn all_pairs(board: &Board, cost: &[u16]) -> Distances {
    const _: () = assert!(2 * (UNREACHABLE as usize) < u16::MAX as usize);
    let n = board.n_cities;
    let mut dist = [[UNREACHABLE; DIST_STRIDE]; DIST_STRIDE];
    for (city, row) in dist.iter_mut().enumerate().take(n) {
        row[city] = 0;
    }
    for (segment, &w) in cost.iter().enumerate().take(board.n_segments) {
        if w >= UNREACHABLE {
            continue;
        }
        let (a, b) = (
            usize::from(board.seg_a[segment]),
            usize::from(board.seg_b[segment]),
        );
        dist[a][b] = dist[a][b].min(w);
        dist[b][a] = dist[b][a].min(w);
    }
    for k in 0..n {
        let via = dist[k];
        for row in dist.iter_mut().take(n) {
            let to_k = row[k];
            if to_k >= UNREACHABLE {
                continue;
            }
            for (cell, &onward) in row.iter_mut().zip(&via) {
                *cell = (*cell).min(to_k + onward);
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
    let cost = crossing_costs(board, seg_owner, player, doubles_locked);
    steiner_cost_with(board, &all_pairs(board, &cost), terminals)
}

/// [`steiner_cost_exact`] over an all-pairs table the caller already has.
///
/// Every cell of the DP is at most [`UNREACHABLE`]: cells start there and only ever
/// decrease. That is what makes the min-plus relaxation below the same computation as the
/// multi-source Dijkstra it replaces -- seeding from every cell under the sentinel and
/// adding exact distances reaches each city at the same minimum, since a detour through an
/// improved city is never shorter than the direct distance (the triangle inequality), and
/// a sentinel-valued cell can never improve anything.
pub fn steiner_cost_with(board: &Board, dist: &Distances, terminals: &[u8]) -> (u32, bool) {
    if terminals.len() < MIN_TERMINALS {
        return (0, true);
    }
    let mut unique = [0u8; MAX_CITIES];
    let mut n_unique = 0;
    let mut seen = 0u64;
    for &t in terminals {
        if seen >> t & 1 == 0 {
            seen |= 1 << t;
            unique[n_unique] = t;
            n_unique += 1;
        }
    }
    unique[..n_unique].sort_unstable();

    // Contract terminals that already cost nothing to reach from one another.
    let mut roots = [0u8; MAX_CITIES];
    let mut n_roots = 0;
    for &t in &unique[..n_unique] {
        let from_t = &dist[usize::from(t)];
        if !roots[..n_roots]
            .iter()
            .any(|&r| from_t[usize::from(r)] == 0)
        {
            roots[n_roots] = t;
            n_roots += 1;
        }
    }
    let roots = &roots[..n_roots];
    if roots.len() < MIN_TERMINALS {
        return (0, true);
    }
    let from_first = &dist[usize::from(roots[0])];
    if roots
        .iter()
        .any(|&r| from_first[usize::from(r)] >= UNREACHABLE)
    {
        return (u32::from(UNREACHABLE), true);
    }
    if roots.len() > MAX_STEINER_TERMINALS {
        return (mst_bound(roots, dist), false);
    }

    let n = board.n_cities;
    let full = (1usize << roots.len()) - 1;
    // dp[mask][v] = cheapest tree spanning the terminals in `mask` plus the vertex v. In u16
    // exactly: every cell is at most UNREACHABLE, so a sum of two is under u16::MAX (the
    // assertion in `all_pairs`). Rows above `full` are never read, and every row at or below
    // it is written before it is read, since submasks are numerically smaller.
    let mut dp = [[UNREACHABLE; DIST_STRIDE]; 1 << MAX_STEINER_TERMINALS];
    for (i, &t) in roots.iter().enumerate() {
        dp[1 << i] = dist[usize::from(t)];
    }

    for mask in 1..=full {
        if mask & (mask - 1) == 0 {
            continue; // single terminal: already initialized from its distances
        }
        // Every split into two non-empty halves, once: the half holding mask's lowest
        // terminal is `low | s`, the other `rest ^ s`, for each s strictly inside `rest`.
        let low = mask & mask.wrapping_neg();
        let rest = mask ^ low;
        let mut row = [UNREACHABLE; DIST_STRIDE];
        let mut s = rest;
        while s != 0 {
            s = (s - 1) & rest;
            let (left, other) = (&dp[low | s], &dp[rest ^ s]);
            for ((cell, &a), &b) in row.iter_mut().zip(left).zip(other) {
                *cell = (*cell).min(a + b);
            }
        }
        if mask == full {
            // The relaxation below cannot lower the minimum (d(u, u) = 0, distances are
            // non-negative), and the minimum is all that is read of the last row.
            let best = *row[..n].iter().min().expect("at least one city");
            return (u32::from(best), true);
        }
        // Let the tree's root slide along cheap track: min over u of row[u] + d(u, v).
        let mut relaxed = row;
        for (u, &through) in row.iter().enumerate().take(n) {
            if through >= UNREACHABLE {
                continue;
            }
            for (cell, &d) in relaxed.iter_mut().zip(&dist[u]) {
                *cell = (*cell).min(through + d);
            }
        }
        dp[mask] = relaxed;
    }
    unreachable!("the full mask returns from the loop")
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

/// Prim over the terminal metric. Always >= the Steiner cost, never below it.
fn mst_bound(roots: &[u8], dist: &Distances) -> u32 {
    let mut inside = [0u8; MAX_CITIES];
    inside[0] = roots[0];
    let mut n_inside = 1;
    let mut total = 0u32;
    while n_inside < roots.len() {
        let mut best = UNREACHABLE;
        let mut best_t: Option<u8> = None;
        for &t in roots {
            if inside[..n_inside].contains(&t) {
                continue;
            }
            for &u in &inside[..n_inside] {
                let d = dist[usize::from(u)][usize::from(t)];
                if d < best {
                    best = d;
                    best_t = Some(t);
                }
            }
        }
        match best_t {
            Some(t) => {
                inside[n_inside] = t;
                n_inside += 1;
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
    // Every segment the seat owns, and each city's incident ones: (neighbour, local edge
    // index, weight), in segment order.
    let mut owned = [0u16; MAX_SEGMENTS];
    let mut n_owned = 0;
    for (segment, &owner) in seg_owner.iter().enumerate().take(board.n_segments) {
        if owner == player {
            owned[n_owned] = segment as u16;
            n_owned += 1;
        }
    }
    if n_owned == 0 {
        return 0;
    }
    let mut trail = Trail::default();
    for (local, &segment) in owned[..n_owned].iter().enumerate() {
        let s = usize::from(segment);
        let (a, b, w) = (board.seg_a[s], board.seg_b[s], u16::from(board.seg_len[s]));
        trail.connect(a, b, local as u8, w);
        trail.connect(b, a, local as u8, w);
    }

    // Layer 1: components, by flood fill over the cities that have track at all.
    let mut best = 0;
    let mut unvisited = trail.touched;
    while unvisited != 0 {
        let start = unvisited.trailing_zeros() as u8;
        let mut component = 0u64;
        let mut frontier = 1u64 << start;
        while frontier != 0 {
            let city = frontier.trailing_zeros() as usize;
            frontier &= frontier - 1;
            component |= 1 << city;
            for &(nb, _, _) in trail.edges_of(city) {
                if component >> nb & 1 == 0 {
                    frontier |= 1 << nb;
                }
            }
        }
        unvisited &= !component;
        best = best.max(trail.component_longest(component));
    }
    best
}

/// The seat's network as fixed arrays: per city, its incident owned segments.
struct Trail {
    edges: [[(u8, u8, u16); MAX_TRAIL_DEGREE]; MAX_CITIES],
    degree: [u8; MAX_CITIES],
    /// Cities with at least one owned segment.
    touched: u64,
}

/// More owned segments at one city than any board has routes there.
const MAX_TRAIL_DEGREE: usize = 16;

impl Default for Trail {
    fn default() -> Self {
        Self {
            edges: [[(0, 0, 0); MAX_TRAIL_DEGREE]; MAX_CITIES],
            degree: [0; MAX_CITIES],
            touched: 0,
        }
    }
}

impl Trail {
    fn connect(&mut self, from: u8, to: u8, local: u8, w: u16) {
        let f = usize::from(from);
        let d = usize::from(self.degree[f]);
        self.edges[f][d] = (to, local, w);
        self.degree[f] += 1;
        self.touched |= 1 << from;
    }

    fn edges_of(&self, city: usize) -> &[(u8, u8, u16)] {
        &self.edges[city][..usize::from(self.degree[city])]
    }

    /// Layers 2-4 on one component, given as a city bitmask.
    fn component_longest(&self, component: u64) -> u16 {
        // Each edge is seen from both ends; count it from its lower-local end only once by
        // collecting locals in a bitmask (locals are below MAX_SEGMENTS <= 128).
        let mut locals = 0u128;
        let mut total = 0u16;
        let mut odd = 0;
        let mut cities = component;
        while cities != 0 {
            let city = cities.trailing_zeros() as usize;
            cities &= cities - 1;
            let incident = self.edges_of(city);
            if incident.len() % 2 == 1 {
                odd += 1;
            }
            for &(_, local, w) in incident {
                if locals >> local & 1 == 0 {
                    locals |= 1 << local;
                    total += w;
                }
            }
        }
        if odd == 0 || odd == 2 {
            return total; // layer 2: an Eulerian trail uses every edge
        }
        let n_edges = locals.count_ones() as usize;
        assert!(
            n_edges <= MAX_TRAIL_EDGES,
            "longest-trail component has {n_edges} edges, above the {MAX_TRAIL_EDGES}-bit mask \
             limit; the train supply should make this unreachable"
        );

        // Mask bits are numbered within the component, not across every segment the seat
        // owns. Numbering globally let a seat with more than 32 segments in total shift past
        // bit 31 even when every component fits the mask -- a panic in debug builds and a
        // silently short trail in release. Unreachable in play (a seat can own at most 27
        // segments on the USA map), and found by the brute-force reference in
        // `naive_graph`.
        let mut bit_of = [0u32; MAX_SEGMENTS];
        let mut remaining = locals;
        let mut i = 0;
        while remaining != 0 {
            let local = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            bit_of[local] = 1 << i;
            i += 1;
        }

        MEMO.with(|memo| {
            let mut memo = memo.borrow_mut();
            memo.clear();
            let mut overall = 0;
            let mut starts = component;
            while starts != 0 {
                let start = starts.trailing_zeros() as u8;
                starts &= starts - 1;
                overall = overall.max(self.best_from(&bit_of, &mut memo, start, 0));
                if overall == total {
                    break; // layer 4: cannot do better than every edge
                }
            }
            overall
        })
    }

    /// Layer 3: the longest walk from `v` over edges not in `used`, memoized.
    fn best_from(&self, bit_of: &[u32], memo: &mut TrailMemo, v: u8, used: u32) -> u16 {
        let key = u64::from(v) << 32 | u64::from(used);
        if let Some(&cached) = memo.get(&key) {
            return cached;
        }
        let mut best = 0;
        for &(nb, local, w) in self.edges_of(usize::from(v)) {
            let bit = bit_of[usize::from(local)];
            if used & bit != 0 {
                continue;
            }
            best = best.max(w + self.best_from(bit_of, memo, nb, used | bit));
        }
        memo.insert(key, best);
        best
    }
}

/// The longest-trail memo: `(vertex, used-edge mask)` packed into a `u64`.
///
/// A multiplicative hash rather than SipHash -- the keys are not attacker-controlled, and
/// SipHash was a tenth of an observation -- and one table per thread, cleared rather than
/// reallocated per component.
type TrailMemo = HashMap<u64, u16, BuildHasherDefault<PackedKeyHasher>>;

thread_local! {
    static MEMO: RefCell<TrailMemo> = RefCell::new(TrailMemo::default());
}

/// Fibonacci hashing of a single `u64` key.
#[derive(Default)]
struct PackedKeyHasher(u64);

impl Hasher for PackedKeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_u64(&mut self, key: u64) {
        self.0 = (key ^ self.0).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
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
    fn a_seat_with_more_than_32_segments_still_gets_its_whole_trail() {
        // The mask is per component. A seat owning 33 segments, none of its components
        // wider than the mask, used to overflow the shift: a panic in debug builds and a
        // trail of 37 instead of 48 in release. 48 is what the exhaustive reference and the
        // retired Python engine both report.
        let board = crate::board::get_board("usa").unwrap();
        let owned = [
            3, 5, 8, 9, 10, 11, 12, 13, 14, 18, 27, 29, 32, 39, 45, 53, 54, 58, 62, 64, 66, 67, 70,
            72, 73, 75, 80, 82, 83, 85, 88, 95, 97,
        ];
        let mut seg_owner = vec![FREE; board.n_segments];
        for s in owned {
            seg_owner[s] = 0;
        }
        assert_eq!(longest_trail(board, &seg_owner, 0), 48);
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
