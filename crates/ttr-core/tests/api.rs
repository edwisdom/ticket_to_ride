//! The query surface: accessors, the action-space helpers, and configuration errors.
//!
//! Ported from `tests/unit/engine/test_api.py`. Everything here is what the PyO3 shim exposes
//! and Phase 5's search calls, so it is pinned directly rather than discovered to be subtly
//! wrong from inside a search tree.

mod common;

use common::{Rig, color, force, game, give, play_out, policy, to_main};
use ttr_core::CONTRACT_VERSION;
use ttr_core::actions::ACTION_SPACE_VERSION;
use ttr_core::config::{ChanceMode, ConfigError, EMPTY_SLOT, PHASE_MAIN, RuleConfig, WildPolicy};
use ttr_core::obs::obs_version;
use ttr_core::state::Game;

/// The message of the panic `f` raises, or `None` if it returns normally.
fn panic_message<R>(f: impl FnOnce() -> R) -> Option<String> {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default(),
    )
}

/// Three independent version numbers; conflating them is how stale checkpoints load.
/// (Unsigned, so `!= 0` is Python's `>= 1`.)
#[test]
fn versions_are_present_and_separate() {
    assert_ne!(CONTRACT_VERSION, 0);
    assert_ne!(ACTION_SPACE_VERSION, 0);
    assert_ne!(obs_version(), 0);
}

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------

#[test]
fn num_distinct_actions_and_repr() {
    let g = game("usa", 2);
    assert_eq!(g.num_distinct_actions(), 915);
    let shown = format!("{g:?}");
    assert!(shown.contains("usa") && shown.contains("915"), "{shown}");
    let state = format!("{:?}", g.new_initial_state(1));
    assert!(state.contains("usa"), "{state}");
}

#[test]
fn default_config_is_a_two_player_usa_game() {
    let g = Game::new(RuleConfig::default()).expect("the default configuration is supported");
    assert_eq!((g.board.name, g.rules.n_players), ("usa", 2));
}

/// Checked both where a config is built (`RuleConfig::new`) and where a hand-built one is
/// used (`Game::new`), since a struct literal skips the first.
#[test]
fn seat_count_is_validated_against_the_map() {
    for (map, n, expected) in [
        ("usa", 6, "supports 2-5 players"),
        ("mini", 5, "supports 2-4 players"),
    ] {
        let err = RuleConfig::new(map, n).expect_err("an unsupported seat count was accepted");
        assert!(matches!(err, ConfigError::PlayerCount { .. }), "{err:?}");
        assert!(err.to_string().contains(expected), "{err}");

        let cfg = RuleConfig {
            map_name: map.to_string(),
            n_players: n,
            ..RuleConfig::default()
        };
        let err = Game::new(cfg).expect_err("Game::new skipped validation");
        assert!(err.to_string().contains(expected), "{err}");
    }
}

/// Python's third case, `initial_hand=-1`, is unrepresentable: the field is a `u8`.
#[test]
fn unimplemented_variants_say_so() {
    let cases = [
        (
            RuleConfig {
                wild_policy: WildPolicy::Explicit,
                ..RuleConfig::default()
            },
            "wild_policy",
        ),
        (
            RuleConfig {
                chance_mode: ChanceMode::Explicit,
                ..RuleConfig::default()
            },
            "chance_mode",
        ),
    ];
    for (cfg, name) in cases {
        let err = cfg
            .validate()
            .expect_err("an unimplemented variant was accepted");
        assert!(matches!(err, ConfigError::NotImplemented(_)), "{err:?}");
        assert!(err.to_string().contains(name), "{err}");
        let err = Game::new(cfg).expect_err("Game::new skipped validation");
        assert!(err.to_string().contains(name), "{err}");
    }
}

// ---------------------------------------------------------------------------
// Action space
// ---------------------------------------------------------------------------

/// Seven keep actions, not eight: keeping nothing is never legal.
///
/// Python's `keep()` raised on a bad mask in every build. Rust's sits on the hot path behind a
/// `debug_assert!`, so that refusal is checked in debug builds; the engine's own refusal of the
/// empty keep, in `step`, holds in every build.
#[test]
fn keep_masks_are_one_based() {
    let g = game("usa", 2);
    let space = g.space;
    assert_eq!(space.keep(1) - space.keep_base(), 1);

    if cfg!(debug_assertions) {
        for bad in [0u16, 8] {
            let message = panic_message(|| space.keep(bad))
                .unwrap_or_else(|| panic!("keep({bad}) was accepted"));
            assert!(message.contains("keep mask must be 1..7"), "{message}");
        }
    }

    let mut state = g.new_initial_state(2);
    assert!(
        state.step(space.keep_base()).is_err(),
        "the empty keep was accepted"
    );
}

#[test]
fn decoders_round_trip() {
    let space = game("usa", 2).space;
    assert_eq!(space.decode_draw(space.draw(3)), 3);
    assert_eq!(space.decode_keep(space.keep(5)), 5);
    assert!(space.is_claim(0));
    assert!(!space.is_claim(space.draw_base));
}

#[test]
fn action_names_cover_every_range() {
    let g = game("usa", 2);
    let (space, board) = (g.space, g.board);
    let name = |action| space.to_string(board, action);
    assert!(name(space.claim(0, 0)).starts_with("CLAIM"));
    assert_eq!(name(space.draw(2)), "DRAW faceup[2]");
    assert_eq!(name(space.draw(5)), "DRAW blind");
    assert_eq!(name(space.draw_tickets), "DRAW_TICKETS");
    assert_eq!(name(space.keep(0b101)), "KEEP offer[0, 2]");
    assert_eq!(name(space.pass_action), "PASS");
    let message = panic_message(|| space.to_string(board, space.n))
        .expect("an id outside the space was named");
    assert!(message.contains("outside"), "{message}");
}

#[test]
fn claim_names_show_the_route_and_the_payment() {
    let g = game("usa", 2);
    let (space, board) = (g.space, g.board);
    let blue = color(board, "blue");
    let segment = (0..board.n_segments)
        .find(|&s| board.seg_color[s] == blue)
        .expect("a blue route");
    let name = space.to_string(
        board,
        space.claim(segment as u16, u16::from(board.locomotive)),
    );
    assert!(name.contains("blue") && name.contains("pay loco"), "{name}");
}

// ---------------------------------------------------------------------------
// State queries
// ---------------------------------------------------------------------------

#[test]
fn hand_accessors_agree_with_the_flat_array() {
    let g = game("usa", 3);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let k = g.board.n_card_types;
    for p in 0..3 {
        assert_eq!(s.hand_of(p), &s.hand[p * k..(p + 1) * k]);
        let held: u32 = s.hand_of(p).iter().map(|&c| u32::from(c)).sum();
        assert_eq!(s.hand_size(p), held);
        let known: u32 = s.certain_of(p).iter().map(|&c| u32::from(c)).sum();
        assert_eq!(known + u32::from(s.unknown[p]), s.hand_size(p));
    }
}

#[test]
fn ticket_queries() {
    let g = game("usa", 2);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let held = s.tickets_of(0);
    assert!(
        !held.is_empty(),
        "the opening keep holds at least two tickets"
    );
    assert_eq!(held.len() as u32, s.tickets[0].count_ones());
    assert!(held.iter().all(|&t| usize::from(t) < g.board.n_tickets));
    assert_eq!(s.tickets_remaining(), s.tdeck_len);
    // Nothing is claimed yet, so no ticket can be complete.
    assert!(!held.iter().any(|&t| s.ticket_complete(0, usize::from(t))));
}

#[test]
fn deck_counts_sum_to_the_undrawn_deck() {
    let g = game("usa", 2);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let counts = s.deck_counts();
    let undrawn = &s.deck[usize::from(s.deck_pos)..usize::from(s.deck_len)];
    let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
    assert_eq!(total, undrawn.len());
    for (card_type, &count) in counts.iter().enumerate() {
        let in_deck = undrawn
            .iter()
            .filter(|&&c| usize::from(c) == card_type)
            .count();
        assert_eq!(in_deck, usize::from(count), "card type {card_type}");
    }
}

/// Everything the observer cannot account for: the deck plus opponents' blind draws.
#[test]
fn unseen_counts_are_what_a_determinization_would_deal() {
    let g = game("usa", 3);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let mut rng = policy(2);
    for _ in 0..40 {
        if s.is_terminal() {
            break;
        }
        let action = s.sample_legal(&mut rng).expect("a non-terminal state");
        s.step(action).expect("a sampled action is legal");
    }

    for observer in 0..3 {
        let unseen = s.unseen_counts(observer);
        assert!(unseen.iter().all(|&c| c >= 0), "{unseen:?}");
        let hidden: i64 = (0..3)
            .filter(|&p| p != observer)
            .map(|p| {
                let known: i64 = s.certain_of(p).iter().map(|&c| i64::from(c)).sum();
                i64::from(s.hand_size(p)) - known
            })
            .sum();
        let undrawn = i64::from(s.deck_len - s.deck_pos);
        let total: i64 = unseen.iter().map(|&c| i64::from(c)).sum();
        assert_eq!(total, undrawn + hidden, "observer {observer}");
    }
}

#[test]
fn history_is_recorded_when_tracking_is_on() {
    let g = game("usa", 2);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let before = s.history.len();
    let action = s
        .sample_legal(&mut policy(2))
        .expect("a non-terminal state");
    s.step(action).expect("a sampled action is legal");
    assert_eq!(s.history.len(), before + 1);
    assert_eq!(s.history.last(), Some(&action));
}

/// Copying the action list dominates a clone, so search and benchmarks turn it off.
#[test]
fn history_is_skipped_when_tracking_is_off() {
    let cfg = RuleConfig {
        track_history: false,
        ..RuleConfig::new("usa", 2).expect("a supported configuration")
    };
    let g = Game::new(cfg).expect("a supported configuration");
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let action = s
        .sample_legal(&mut policy(2))
        .expect("a non-terminal state");
    s.step(action).expect("a sampled action is legal");
    assert!(s.history.is_empty(), "{:?}", s.history);
    assert!(s.clone().history.is_empty());
}

#[test]
fn stepping_a_finished_game_is_rejected() {
    let g = game("usa", 2);
    let mut s = g.new_initial_state(2);
    play_out(&mut s, &mut policy(2));
    let err = s.step(0).expect_err("a finished game accepted an action");
    assert!(err.0.contains("game is over"), "{err}");
    assert!(s.legal_actions().is_empty());
}

#[test]
fn current_player_tracks_the_turn() {
    let g = game("usa", 3);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let mut seats: Vec<u8> = Vec::new();
    for _ in 0..6 {
        let seat = s.current_player();
        seats.push(seat);
        while s.phase != PHASE_MAIN || s.current_player() == seat {
            if s.is_terminal() {
                break;
            }
            let action = s
                .sample_legal(&mut policy(seats.len() as u64))
                .expect("a non-terminal state");
            s.step(action).expect("a sampled action is legal");
        }
    }
    assert_eq!(seats[..3], [0, 1, 2]);
}

/// No USA ticket connects two adjacent cities, so completion needs a whole path.
#[test]
fn ticket_completion_follows_the_union_find() {
    let g = game("usa", 2);
    let board = g.board;
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let dist = |a: u8, b: u8| board.dist[usize::from(a)][usize::from(b)];
    let ticket = (0..board.n_tickets)
        .min_by_key(|&t| dist(board.ticket_a[t], board.ticket_b[t]))
        .expect("the board has tickets");
    force(
        &mut s,
        Rig {
            tickets: Some(vec![1 << ticket, 0]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert!(!s.ticket_complete(0, ticket));

    // Walk a shortest path, claiming as we go; the ticket stays open until the last step.
    let mut here = board.ticket_a[ticket];
    let target = board.ticket_b[ticket];
    while here != target {
        assert!(!s.ticket_complete(0, ticket), "complete before the path is");
        let &(nb, segment) = board.adjacency[usize::from(here)]
            .iter()
            .min_by_key(|&&(nb, seg)| u16::from(board.seg_len[usize::from(seg)]) + dist(nb, target))
            .expect("every city has a route");
        give(&mut s, 0, usize::from(segment));
        here = nb;
    }
    assert!(s.ticket_complete(0, ticket));
}

#[test]
fn faceup_reports_empty_slots_as_minus_one() {
    let g = game("usa", 2);
    let mut s = g.new_initial_state(2);
    to_main(&mut s);
    let counts = g.board.deck_composition_counts.clone();
    let k = counts.len();
    force(
        &mut s,
        Rig {
            hands: Some(vec![counts, vec![0; k]]),
            faceup: Some(vec![]),
            deck: Some(vec![]),
            cur: Some(0),
            ..Rig::default()
        },
    );
    assert_eq!(EMPTY_SLOT, -1);
    assert_eq!(s.faceup, [EMPTY_SLOT; 5]);
}
