//! Board invariants -- the generated constants and every table the engine derives from them.
//!
//! The checks on RULES.md and on the generator's freshness stay in Python with
//! `tools/gen_board.py` (`tests/unit/data/`). This file checks what `ttr_core::board` builds
//! from `board_gen.rs`: that the derived tables agree with the raw data, and that the
//! numbers the engine is sized against hold.

use std::collections::{BTreeMap, HashSet};

use ttr_core::board::{Board, GRAY, NO_SIBLING, UNREACHABLE, all_boards, get_board};
use ttr_core::config::RuleConfig;

fn usa() -> &'static Board {
    get_board("usa").expect("usa is generated")
}

fn mini() -> &'static Board {
    get_board("mini").expect("mini is generated")
}

/// Segments of exactly this colour (or [`GRAY`]).
fn segments_of_color(board: &Board, color: u8) -> usize {
    board.seg_color.iter().filter(|&&c| c == color).count()
}

/// Double routes: each is two segments that name each other.
fn doubles(board: &Board) -> usize {
    board.sibling.iter().filter(|&&s| s != NO_SIBLING).count() / 2
}

// ---------------------------------------------------------------------------
// Structure, on every map
// ---------------------------------------------------------------------------

/// Sorted, deduplicated, and endpoint-normalized, so a re-transcription is stable.
#[test]
fn indices_are_canonically_ordered() {
    for board in all_boards() {
        let raw = board.raw;
        let name = board.name;
        assert!(raw.cities.windows(2).all(|w| w[0] < w[1]), "{name} cities");
        assert!(
            raw.color_names.windows(2).all(|w| w[0] < w[1]),
            "{name} colours"
        );
        assert!(raw.segments.is_sorted(), "{name} segments");
        assert!(raw.tickets.is_sorted(), "{name} tickets");
        for s in 0..board.n_segments {
            assert!(board.seg_a[s] < board.seg_b[s], "{name} segment {s}");
        }
        for t in 0..board.n_tickets {
            assert!(board.ticket_a[t] < board.ticket_b[t], "{name} ticket {t}");
        }
    }
}

/// Getting double routes wrong is the most common TTR implementation bug.
#[test]
fn siblings_are_mutual_and_agree() {
    for board in all_boards() {
        let name = board.name;
        for (s, &sib) in board.sibling.iter().enumerate() {
            if sib == NO_SIBLING {
                continue;
            }
            let sib = usize::from(sib);
            assert_eq!(
                usize::from(board.sibling[sib]),
                s,
                "{name}: sibling relation is not mutual"
            );
            assert_ne!(sib, s, "{name}: segment {s} is its own sibling");
            assert_eq!(
                (board.seg_a[s], board.seg_b[s]),
                (board.seg_a[sib], board.seg_b[sib]),
                "{name} segment {s}"
            );
            assert_eq!(board.seg_len[s], board.seg_len[sib], "{name} segment {s}");
            assert_eq!(
                board.pair_of_segment[s], board.pair_of_segment[sib],
                "{name} segment {s}"
            );
        }
        assert_eq!(board.n_segments - doubles(board), board.n_pairs, "{name}");
    }
}

#[test]
fn buckets_partition_the_segments() {
    for board in all_boards() {
        let name = board.name;
        let mut seen = HashSet::new();
        for (i, bucket) in board.buckets.iter().enumerate() {
            assert!(!bucket.segments.is_empty(), "{name}: empty bucket {i}");
            for &s in &bucket.segments {
                let s = usize::from(s);
                assert_eq!(
                    (board.seg_len[s], board.seg_color[s]),
                    (bucket.length, bucket.color),
                    "{name} segment {s}"
                );
                assert_eq!(usize::from(board.seg_bucket[s]), i, "{name} segment {s}");
                assert!(seen.insert(s), "{name}: segment {s} is in two buckets");
            }
        }
        assert_eq!(
            seen,
            (0..board.n_segments).collect::<HashSet<_>>(),
            "{name}"
        );
    }
}

#[test]
fn adjacency_matches_the_segment_table() {
    for board in all_boards() {
        for (city, row) in board.adjacency.iter().enumerate() {
            for &(nb, seg) in row {
                let seg = usize::from(seg);
                let mut ends = [city as u8, nb];
                ends.sort_unstable();
                assert_eq!(
                    ends,
                    [board.seg_a[seg], board.seg_b[seg]],
                    "{} segment {seg}",
                    board.name
                );
            }
        }
        let entries: usize = board.adjacency.iter().map(Vec::len).sum();
        assert_eq!(entries, 2 * board.n_segments, "{}", board.name);
    }
}

#[test]
fn distance_table_is_a_metric_and_finite() {
    for board in all_boards() {
        let (name, dist) = (board.name, &board.dist);
        let n = board.n_cities;
        for i in 0..n {
            assert_eq!(dist[i][i], 0, "{name} city {i}");
            for j in 0..n {
                assert_eq!(dist[i][j], dist[j][i], "{name}: distance is not symmetric");
                assert!(dist[i][j] < UNREACHABLE, "{name}: map is not connected");
                for k in 0..n {
                    assert!(
                        dist[i][j] <= dist[i][k] + dist[k][j],
                        "{name}: triangle inequality fails at {i}-{k}-{j}"
                    );
                }
            }
        }
    }
}

#[test]
fn deck_composition() {
    for board in all_boards() {
        let (name, raw, deck) = (board.name, board.raw, &board.deck_composition);
        let per_color = usize::from(raw.cards_per_color);
        assert_eq!(
            deck.len(),
            per_color * board.n_colors + usize::from(raw.locomotives),
            "{name}"
        );
        let count = |card: u8| deck.iter().filter(|&&c| c == card).count();
        for c in 0..board.n_colors as u8 {
            assert_eq!(count(c), per_color, "{name} colour {c}");
        }
        assert_eq!(
            count(board.locomotive),
            usize::from(raw.locomotives),
            "{name}"
        );
        assert!(deck.is_sorted(), "{name}: canonical order");
        // A route must be payable from a single colour run, or it is unclaimable by design.
        assert!(raw.cards_per_color >= board.max_len, "{name}");
    }
}

#[test]
fn tickets_are_reachable_and_distinct() {
    for board in all_boards() {
        let name = board.name;
        let pairs: HashSet<(u8, u8)> = board
            .ticket_a
            .iter()
            .copied()
            .zip(board.ticket_b.iter().copied())
            .collect();
        assert_eq!(pairs.len(), board.n_tickets, "{name}: duplicate tickets");
        for t in 0..board.n_tickets {
            let (a, b) = (board.ticket_a[t], board.ticket_b[t]);
            assert_ne!(a, b, "{name} ticket {t}");
            assert!(
                board.dist[usize::from(a)][usize::from(b)] < UNREACHABLE,
                "{name} ticket {t}"
            );
            assert!(board.ticket_points[t] > 0, "{name} ticket {t}");
        }
    }
}

/// Setup deals `initial_deal` tickets per player; running the deck dry there is a bug.
#[test]
fn enough_tickets_for_a_full_table() {
    for board in all_boards() {
        let needed =
            usize::from(board.raw.max_players) * usize::from(board.raw.initial_ticket_deal);
        assert!(
            board.n_tickets >= needed,
            "{}: {} tickets < {needed}",
            board.name,
            board.n_tickets
        );
    }
}

/// Every seat must be able to spend its trains without the board running out.
#[test]
fn enough_track_for_a_full_table() {
    for board in all_boards() {
        // `total_spaces >= 0.9 * seats * trains`, kept in integers.
        let supply = u32::from(board.raw.max_players) * u32::from(board.raw.trains_per_player);
        assert!(
            10 * board.total_spaces >= 9 * supply,
            "{}: {} spaces for {supply} trains",
            board.name,
            board.total_spaces
        );
    }
}

// ---------------------------------------------------------------------------
// USA specifics -- the numbers the whole engine is sized against
// ---------------------------------------------------------------------------

#[test]
fn usa_headline_counts() {
    let usa = usa();
    assert_eq!(
        (usa.n_cities, usa.n_pairs, usa.n_segments, usa.total_spaces),
        (36, 78, 100, 309)
    );
    assert_eq!((usa.n_tickets, usa.deck_size, usa.n_colors), (30, 110, 8));
    assert_eq!(usa.raw.trains_per_player, 45);
    assert_eq!(doubles(usa), 22);
}

/// 44 gray plus exactly 7 segments of each of the 8 colours.
#[test]
fn usa_colour_balance() {
    let usa = usa();
    assert_eq!(segments_of_color(usa, GRAY), 44);
    for c in 0..usa.n_colors as u8 {
        assert_eq!(segments_of_color(usa, c), 7, "{}", usa.color_name(c));
    }
}

#[test]
fn usa_length_histogram() {
    let mut hist: BTreeMap<u8, usize> = BTreeMap::new();
    for &len in &usa().seg_len {
        *hist.entry(len).or_default() += 1;
    }
    assert_eq!(
        hist,
        BTreeMap::from([(1, 9), (2, 36), (3, 20), (4, 16), (5, 10), (6, 9)])
    );
}

/// Denver, Helena and Pittsburgh are the degree-7 hubs the feature spec assumes.
#[test]
fn usa_degree_hubs() {
    let usa = usa();
    let degree: Vec<usize> = usa
        .adjacency
        .iter()
        .map(|row| row.iter().map(|&(nb, _)| nb).collect::<HashSet<_>>().len())
        .collect();
    let hubs: HashSet<&str> = (0..usa.n_cities)
        .filter(|&c| degree[c] == 7)
        .map(|c| usa.raw.cities[c])
        .collect();
    assert_eq!(hubs, HashSet::from(["Denver", "Helena", "Pittsburgh"]));
    assert_eq!(degree.iter().max(), Some(&7));
}

// ---------------------------------------------------------------------------
// TTR-mini specifics
// ---------------------------------------------------------------------------

/// The prior art's mini map dropped to 4 colours / max length 3 / no doubles.
#[test]
fn mini_keeps_what_makes_ttr_interesting() {
    let mini = mini();
    assert_eq!(mini.n_colors, 6);
    assert!(mini.max_len >= 5);
    assert_eq!(doubles(mini), 6);
    assert!((12..=16).contains(&mini.n_cities));
    assert_eq!(
        mini.n_tickets, mini.n_cities,
        "#tickets should track #cities"
    );
    // Mean degree `2 * pairs / cities > 3`, kept in integers.
    assert!(2 * mini.n_pairs > 3 * mini.n_cities, "mean degree too low");
}

#[test]
fn mini_colour_balance() {
    let mini = mini();
    for c in 0..mini.n_colors as u8 {
        assert_eq!(segments_of_color(mini, c), 3, "{}", mini.color_name(c));
    }
    assert_eq!(segments_of_color(mini, GRAY), 12);
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

/// One shared instance per map, and a name that is not a map is refused -- with the reason
/// spelled out where a name actually arrives from a user, a config.
#[test]
fn get_board_rejects_unknown_maps() {
    let usa = usa();
    assert_eq!(usa.name, "usa");
    assert!(std::ptr::eq(
        usa,
        get_board("usa").expect("usa is generated")
    ));
    assert!(get_board("europe").is_none());
    let err = RuleConfig::new("europe", 2).expect_err("an unknown map was accepted");
    assert!(err.to_string().contains("unknown map"), "{err}");
}

#[test]
fn data_hashes_are_distinct_and_well_formed() {
    let hashes: Vec<&str> = all_boards().iter().map(|b| b.data_hash).collect();
    let distinct: HashSet<&str> = hashes.iter().copied().collect();
    assert_eq!(distinct.len(), hashes.len(), "{hashes:?}");
    for h in hashes {
        assert_eq!(h.len(), 32, "{h}");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()), "{h}");
    }
}

#[test]
fn names_render() {
    let usa = usa();
    assert!(usa.segment_name(0).contains('-'));
    assert_eq!(usa.color_name(GRAY), "gray");
    assert_eq!(usa.color_name(usa.locomotive), "loco");
    assert_eq!(usa.color_name(0), "black");
    assert!(usa.ticket_name(0).contains('('));
    assert!(format!("{usa:?}").contains("usa"));
}
