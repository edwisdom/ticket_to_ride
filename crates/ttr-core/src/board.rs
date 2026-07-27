//! Derived board tables, built once from the generated constants in [`crate::board_gen`].
//!
//! The twin of `ticket_to_ride/data/board.py`, and deliberately *not* generated for the
//! same reason: everything here is a pure function of the generated data, so emitting it
//! into two languages would double the surface area for no gain. Contract §5 lists
//! adjacency, buckets and distance tables as explicitly **not frozen** -- none of them
//! reaches a hash, so they may be reorganised for speed freely.
//!
//! They exist so the engine's hot loops never do graph work:
//!
//! * [`Board::sibling`] turns the double-route rules into an array lookup.
//! * [`Board::buckets`] turns claim masking into ~45 affordability tests plus a short scan
//!   of only the affordable buckets, instead of a 900-way action scan (PLAN.md §5.3).
//! * [`Board::dist`] is all-pairs shortest path in train cars, the base of every ticket
//!   heuristic and of the observation encoder.
//!
//! One ordering *is* worth matching Python exactly even though nothing forces it: buckets
//! are sorted by `(length, color)` and walked in that order, and each bucket's segments
//! ascend. `legal_actions()` sorts before anyone compares it, but `sample_legal()`
//! deliberately does not -- so matching the unsorted order makes random playouts on a
//! given seed comparable between the two engines, which is a free extra check.

use std::sync::OnceLock;

use crate::board_gen::{MAPS, RawMap};

/// "Any single colour" -- a segment with no required colour. Also the sentinel for "no
/// twin" in [`Board::sibling`]; both are u8 ids and 255 is out of range for either.
pub const GRAY: u8 = 255;

/// Sentinel in [`Board::sibling`]: this segment is not half of a double route.
pub const NO_SIBLING: u16 = u16::MAX;

/// Unreachable, in the distance tables. Large enough to add twice without overflowing.
pub const UNREACHABLE: u16 = 10_000;

/// A double route is exactly two parallel tracks; the sibling table assumes it.
const TRACKS_PER_DOUBLE: usize = 2;

// ---------------------------------------------------------------------------
// Compile-time maxima
// ---------------------------------------------------------------------------
//
// `State` is a fixed-size POD so that cloning is a memcpy (PLAN.md §5.1), which means its
// arrays are sized to the largest board rather than to the board in play. Computing the
// maxima from the generated data with const fns rather than writing literals means a new
// or enlarged map resizes the state automatically instead of silently overflowing it.

/// The largest segment count over every generated map.
pub const MAX_SEGMENTS: usize = {
    let mut best = 0;
    let mut i = 0;
    while i < MAPS.len() {
        let n = MAPS[i].segments.len();
        if n > best {
            best = n;
        }
        i += 1;
    }
    best
};

/// The largest city count over every generated map.
pub const MAX_CITIES: usize = {
    let mut best = 0;
    let mut i = 0;
    while i < MAPS.len() {
        let n = MAPS[i].cities.len();
        if n > best {
            best = n;
        }
        i += 1;
    }
    best
};

/// The largest ticket count over every generated map. Also the ticket ring's capacity.
pub const MAX_TICKETS: usize = {
    let mut best = 0;
    let mut i = 0;
    while i < MAPS.len() {
        let n = MAPS[i].tickets.len();
        if n > best {
            best = n;
        }
        i += 1;
    }
    best
};

/// The largest card-type count (colours + locomotive) over every generated map.
pub const MAX_CARD_TYPES: usize = {
    let mut best = 0;
    let mut i = 0;
    while i < MAPS.len() {
        let n = MAPS[i].color_names.len() + 1;
        if n > best {
            best = n;
        }
        i += 1;
    }
    best
};

/// The largest deck size over every generated map.
pub const MAX_DECK: usize = {
    let mut best = 0;
    let mut i = 0;
    while i < MAPS.len() {
        let m = MAPS[i];
        let n = m.color_names.len() * m.cards_per_color as usize + m.locomotives as usize;
        if n > best {
            best = n;
        }
        i += 1;
    }
    best
};

/// The largest seat count over every generated map.
pub const MAX_PLAYERS: usize = {
    let mut best = 0;
    let mut i = 0;
    while i < MAPS.len() {
        let n = MAPS[i].max_players as usize;
        if n > best {
            best = n;
        }
        i += 1;
    }
    best
};

/// The face-up display is five cards in every published edition.
pub const FACEUP_SLOTS: usize = 5;

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

/// One bucket of the claim-legality scan: every segment sharing a `(length, colour)`.
///
/// Affordability depends only on the pair, so the scan tests each bucket once and walks
/// segments only for the affordable ones.
pub struct Bucket {
    pub length: u8,
    /// A colour index, or [`GRAY`].
    pub color: u8,
    pub segments: Vec<u16>,
}

/// One map's immutable derived tables. Built once and shared by every `State`.
pub struct Board {
    pub raw: &'static RawMap,
    pub name: &'static str,

    pub n_cities: usize,
    pub n_segments: usize,
    pub n_tickets: usize,
    pub n_colors: usize,
    pub n_card_types: usize,
    /// Card type of the locomotive: `n_colors`, **never a hard-coded 8**. TTR-mini has six
    /// colours, so its locomotive is card type 6; hard-coding 8 works on USA and silently
    /// corrupts mini.
    pub locomotive: u8,

    pub seg_a: Vec<u8>,
    pub seg_b: Vec<u8>,
    pub seg_len: Vec<u8>,
    /// Required colour, or [`GRAY`].
    pub seg_color: Vec<u8>,
    pub max_len: u8,
    pub total_spaces: u32,

    pub ticket_a: Vec<u8>,
    pub ticket_b: Vec<u8>,
    pub ticket_points: Vec<u8>,

    /// The deck in **canonical order** -- `cards_per_color` of colour 0, then colour 1,
    /// ... then the locomotives. Shuffled once at construction; see CONTRACT.md §2.1.
    pub deck_composition: Vec<u8>,
    pub deck_size: usize,
    pub deck_composition_counts: Vec<u8>,

    /// The other track of a double route, or [`NO_SIBLING`].
    pub sibling: Vec<u16>,
    pub pair_of_segment: Vec<u16>,
    pub n_pairs: usize,
    /// `(neighbour city, segment id)` per city.
    pub adjacency: Vec<Vec<(u8, u16)>>,

    pub buckets: Vec<Bucket>,
    pub seg_bucket: Vec<u16>,

    /// Shortest path between every city pair in train cars, over the cheapest track of
    /// each pair. [`UNREACHABLE`] where no route exists.
    pub dist: Vec<Vec<u16>>,

    /// Points scored for a route of each length; index 0 unused.
    pub route_points: &'static [u8],
    pub data_hash: &'static str,
}

impl Board {
    fn build(raw: &'static RawMap) -> Self {
        let n_cities = raw.cities.len();
        let n_segments = raw.segments.len();
        let n_tickets = raw.tickets.len();
        let n_colors = raw.color_names.len();

        let seg_a: Vec<u8> = raw.segments.iter().map(|s| s.0).collect();
        let seg_b: Vec<u8> = raw.segments.iter().map(|s| s.1).collect();
        let seg_len: Vec<u8> = raw.segments.iter().map(|s| s.2).collect();
        let seg_color: Vec<u8> = raw.segments.iter().map(|s| s.3).collect();

        let deck_composition: Vec<u8> = (0..n_colors as u8)
            .flat_map(|c| std::iter::repeat_n(c, raw.cards_per_color as usize))
            .chain(std::iter::repeat_n(
                raw.locomotive(),
                raw.locomotives as usize,
            ))
            .collect();
        let mut deck_composition_counts = vec![raw.cards_per_color; n_colors];
        deck_composition_counts.push(raw.locomotives);

        let (sibling, pair_of_segment, n_pairs, adjacency) =
            Self::build_pairs(n_cities, n_segments, &seg_a, &seg_b);
        let (buckets, seg_bucket) = Self::build_buckets(n_segments, &seg_len, &seg_color);
        let dist = all_pairs_distance(n_cities, &adjacency, &seg_len);

        Self {
            raw,
            name: raw.name,
            n_cities,
            n_segments,
            n_tickets,
            n_colors,
            n_card_types: n_colors + 1,
            locomotive: raw.locomotive(),
            max_len: seg_len.iter().copied().max().unwrap_or(0),
            total_spaces: seg_len.iter().map(|&w| u32::from(w)).sum(),
            seg_a,
            seg_b,
            seg_len,
            seg_color,
            ticket_a: raw.tickets.iter().map(|t| t.0).collect(),
            ticket_b: raw.tickets.iter().map(|t| t.1).collect(),
            ticket_points: raw.tickets.iter().map(|t| t.2).collect(),
            deck_size: deck_composition.len(),
            deck_composition,
            deck_composition_counts,
            sibling,
            pair_of_segment,
            n_pairs,
            adjacency,
            buckets,
            seg_bucket,
            dist,
            route_points: raw.route_points,
            data_hash: raw.data_hash,
        }
    }

    /// Sibling table, city-pair ids, and adjacency.
    #[allow(clippy::type_complexity)]
    fn build_pairs(
        n_cities: usize,
        n_segments: usize,
        seg_a: &[u8],
        seg_b: &[u8],
    ) -> (Vec<u16>, Vec<u16>, usize, Vec<Vec<(u8, u16)>>) {
        let mut by_pair: Vec<((u8, u8), Vec<u16>)> = Vec::new();
        for s in 0..n_segments {
            let key = (seg_a[s], seg_b[s]);
            match by_pair.iter_mut().find(|(k, _)| *k == key) {
                Some((_, segs)) => segs.push(s as u16),
                None => by_pair.push((key, vec![s as u16])),
            }
        }
        by_pair.sort_by_key(|(k, _)| *k);

        let mut sibling = vec![NO_SIBLING; n_segments];
        let mut pair_of_segment = vec![0u16; n_segments];
        let mut adjacency: Vec<Vec<(u8, u16)>> = vec![Vec::new(); n_cities];

        for (pair_id, ((a, b), segs)) in by_pair.iter().enumerate() {
            if segs.len() == TRACKS_PER_DOUBLE {
                sibling[segs[0] as usize] = segs[1];
                sibling[segs[1] as usize] = segs[0];
            }
            for &s in segs {
                pair_of_segment[s as usize] = pair_id as u16;
                adjacency[*a as usize].push((*b, s));
                adjacency[*b as usize].push((*a, s));
            }
        }
        (sibling, pair_of_segment, by_pair.len(), adjacency)
    }

    /// Group segments by `(length, colour)`, sorted -- gray (255) sorts last within a
    /// length, matching Python's tuple ordering.
    fn build_buckets(
        n_segments: usize,
        seg_len: &[u8],
        seg_color: &[u8],
    ) -> (Vec<Bucket>, Vec<u16>) {
        let mut grouped: Vec<((u8, u8), Vec<u16>)> = Vec::new();
        for s in 0..n_segments {
            let key = (seg_len[s], seg_color[s]);
            match grouped.iter_mut().find(|(k, _)| *k == key) {
                Some((_, segs)) => segs.push(s as u16),
                None => grouped.push((key, vec![s as u16])),
            }
        }
        grouped.sort_by_key(|(k, _)| *k);

        let buckets: Vec<Bucket> = grouped
            .into_iter()
            .map(|((length, color), segments)| Bucket {
                length,
                color,
                segments,
            })
            .collect();

        let mut seg_bucket = vec![0u16; n_segments];
        for (i, bucket) in buckets.iter().enumerate() {
            for &s in &bucket.segments {
                seg_bucket[s as usize] = i as u16;
            }
        }
        (buckets, seg_bucket)
    }

    /// How many of this card type are printed. The locomotive count differs from a colour's.
    pub fn cards_per_type(&self, card_type: u8) -> u8 {
        self.deck_composition_counts[card_type as usize]
    }

    pub fn color_name(&self, c: u8) -> &'static str {
        if c == GRAY {
            "gray"
        } else if c == self.locomotive {
            "loco"
        } else {
            self.raw.color_names[c as usize]
        }
    }

    pub fn segment_name(&self, s: usize) -> String {
        format!(
            "{}-{}:{}{}",
            self.raw.cities[self.seg_a[s] as usize],
            self.raw.cities[self.seg_b[s] as usize],
            self.seg_len[s],
            self.color_name(self.seg_color[s]),
        )
    }

    pub fn ticket_name(&self, t: usize) -> String {
        format!(
            "{}-{}({})",
            self.raw.cities[self.ticket_a[t] as usize],
            self.raw.cities[self.ticket_b[t] as usize],
            self.ticket_points[t],
        )
    }
}

/// Shortest path between every city pair, in train cars, over the cheapest track of each
/// pair. Dense Dijkstra from each source: 36 nodes makes the priority queue pure overhead.
fn all_pairs_distance(
    n_cities: usize,
    adjacency: &[Vec<(u8, u16)>],
    seg_len: &[u8],
) -> Vec<Vec<u16>> {
    let mut best_edge = vec![vec![UNREACHABLE; n_cities]; n_cities];
    for (city, row) in adjacency.iter().enumerate() {
        for &(nb, seg) in row {
            let w = u16::from(seg_len[seg as usize]);
            if w < best_edge[city][nb as usize] {
                best_edge[city][nb as usize] = w;
            }
        }
    }

    let mut out = Vec::with_capacity(n_cities);
    for src in 0..n_cities {
        let mut dist = vec![UNREACHABLE; n_cities];
        dist[src] = 0;
        let mut done = vec![false; n_cities];
        for _ in 0..n_cities {
            let mut u = usize::MAX;
            let mut best = UNREACHABLE;
            for v in 0..n_cities {
                if !done[v] && dist[v] < best {
                    u = v;
                    best = dist[v];
                }
            }
            if u == usize::MAX {
                break;
            }
            done[u] = true;
            for v in 0..n_cities {
                let w = best_edge[u][v];
                if w < UNREACHABLE && best + w < dist[v] {
                    dist[v] = best + w;
                }
            }
        }
        out.push(dist);
    }
    out
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

fn boards() -> &'static Vec<Board> {
    static BOARDS: OnceLock<Vec<Board>> = OnceLock::new();
    BOARDS.get_or_init(|| MAPS.iter().map(|raw| Board::build(raw)).collect())
}

/// Look up a board by name.
pub fn get_board(name: &str) -> Option<&'static Board> {
    boards().iter().find(|b| b.name == name)
}

/// Every generated board, in generation order.
pub fn all_boards() -> &'static [Board] {
    boards()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_state_maxima_cover_every_map() {
        // The whole point of computing these with const fns: a new map cannot silently
        // overflow the fixed-size State. This asserts the const fns actually did it.
        for board in all_boards() {
            assert!(board.n_segments <= MAX_SEGMENTS, "{}", board.name);
            assert!(board.n_cities <= MAX_CITIES, "{}", board.name);
            assert!(board.n_tickets <= MAX_TICKETS, "{}", board.name);
            assert!(board.n_card_types <= MAX_CARD_TYPES, "{}", board.name);
            assert!(board.deck_size <= MAX_DECK, "{}", board.name);
            assert!(
                board.raw.max_players as usize <= MAX_PLAYERS,
                "{}",
                board.name
            );
        }
        assert_eq!((MAX_SEGMENTS, MAX_CITIES, MAX_TICKETS), (100, 36, 30));
        assert_eq!((MAX_CARD_TYPES, MAX_DECK, MAX_PLAYERS), (9, 110, 5));
    }

    #[test]
    fn the_locomotive_is_not_hard_coded_to_eight() {
        // The trap called out in CONTRACT.md §0: hard-coding 8 works on USA and silently
        // corrupts mini, whose locomotive is card type 6.
        assert_eq!(get_board("usa").unwrap().locomotive, 8);
        assert_eq!(get_board("mini").unwrap().locomotive, 6);
    }

    #[test]
    fn data_hashes_match_the_contract() {
        assert_eq!(
            get_board("usa").unwrap().data_hash,
            "1e5e154018541e483aba9c4ba702396c"
        );
        assert_eq!(
            get_board("mini").unwrap().data_hash,
            "103c6ccf956f42fa082881b0b3e9f9f9"
        );
    }

    #[test]
    fn usa_board_invariants() {
        let b = get_board("usa").unwrap();
        assert_eq!((b.n_cities, b.n_segments, b.n_tickets), (36, 100, 30));
        assert_eq!((b.n_pairs, b.total_spaces), (78, 309));
        assert_eq!(b.deck_size, 110);
        // 22 double routes: 44 segments have a sibling.
        assert_eq!(b.sibling.iter().filter(|&&s| s != NO_SIBLING).count(), 44);
        // PLAN.md §5.3 guessed 33 buckets during planning; 45 is the measured count.
        assert_eq!(b.buckets.len(), 45);
    }

    #[test]
    fn buckets_partition_the_segments_and_stay_sorted() {
        for board in all_boards() {
            let total: usize = board.buckets.iter().map(|b| b.segments.len()).sum();
            assert_eq!(total, board.n_segments, "{}", board.name);
            let mut keys: Vec<(u8, u8)> =
                board.buckets.iter().map(|b| (b.length, b.color)).collect();
            let sorted = {
                let mut k = keys.clone();
                k.sort_unstable();
                k
            };
            assert_eq!(keys, sorted, "{} buckets are not sorted", board.name);
            keys.dedup();
            assert_eq!(
                keys.len(),
                board.buckets.len(),
                "{} has duplicate buckets",
                board.name
            );
            // Segments ascend within a bucket: the legality scan emits action ids in this
            // order, and matching Python's unsorted order keeps random playouts comparable.
            for bucket in &board.buckets {
                assert!(
                    bucket.segments.windows(2).all(|w| w[0] < w[1]),
                    "{}",
                    board.name
                );
            }
        }
    }

    #[test]
    fn siblings_are_symmetric_and_share_endpoints() {
        for board in all_boards() {
            for s in 0..board.n_segments {
                let twin = board.sibling[s];
                if twin == NO_SIBLING {
                    continue;
                }
                assert_eq!(board.sibling[twin as usize], s as u16, "{}", board.name);
                assert_eq!(board.seg_a[s], board.seg_a[twin as usize]);
                assert_eq!(board.seg_b[s], board.seg_b[twin as usize]);
            }
        }
    }

    #[test]
    fn the_deck_composition_is_canonical_and_complete() {
        for board in all_boards() {
            let raw = board.raw;
            let mut counts = vec![0usize; board.n_card_types];
            for &card in &board.deck_composition {
                counts[card as usize] += 1;
            }
            for (c, &count) in counts.iter().take(board.n_colors).enumerate() {
                assert_eq!(
                    count, raw.cards_per_color as usize,
                    "{} colour {c}",
                    board.name
                );
            }
            assert_eq!(counts[board.locomotive as usize], raw.locomotives as usize);
            // Canonical order: all of colour 0, then colour 1, ... then locomotives. The
            // permutation must come from the seed stream alone, never from this order.
            assert!(
                board.deck_composition.windows(2).all(|w| w[0] <= w[1]),
                "{}",
                board.name
            );
        }
    }

    #[test]
    fn every_board_is_connected() {
        for board in all_boards() {
            for a in 0..board.n_cities {
                for b in 0..board.n_cities {
                    assert!(board.dist[a][b] < UNREACHABLE, "{} {a}-{b}", board.name);
                }
            }
        }
    }

    #[test]
    fn mini_ticket_points_are_shortest_path_costs() {
        // A design rule for TTR-mini specifically -- USA's ticket values come from the
        // rulebook and are not shortest paths. Asserted rather than trusted, because the
        // mini values were computed and a transcription slip would quietly change what
        // the agent is optimising. It also cross-checks `dist` against an independent
        // source, which is why it lives here rather than only on the Python side.
        let mini = get_board("mini").unwrap();
        for t in 0..mini.n_tickets {
            let a = mini.ticket_a[t] as usize;
            let b = mini.ticket_b[t] as usize;
            assert_eq!(
                u16::from(mini.ticket_points[t]),
                mini.dist[a][b],
                "{}",
                mini.ticket_name(t)
            );
        }
    }
}
