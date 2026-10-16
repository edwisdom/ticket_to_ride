//! The reward, as exact values.
//!
//! Phase 3.1's mutation testing found that the 3-5P `returns()` formula had no test that
//! pinned its *values*: the ported tests check that returns sum to zero and that the winner
//! gets the most, and most mutations of the formula -- a win share multiplied instead of
//! divided, ranks scaled by 2 instead of halved -- keep both properties. The Python
//! differential used to pin returns bit for bit; these pin them by hand, from the formula
//! in `scoring::returns`: `0.75 * (win - mean win) + 0.25 * (rank - mean rank)`, win shared
//! equally among tied leaders, rank 0..1 from worst to best and averaged over ties.
//!
//! Every position here has no tickets and no track, so each seat's final key is just its
//! banked route score.

mod common;

use common::{game, to_main};
use ttr_core::scoring::{returns, winners};

/// Returns for a finished position whose seats banked `scores` and nothing else.
fn returns_for(scores: &[i16]) -> Vec<f64> {
    let mut state = game("usa", scores.len()).new_initial_state(1);
    to_main(&mut state);
    for (seat, &score) in scores.iter().enumerate() {
        state.score[seat] = score;
        state.tickets[seat] = 0;
    }
    returns(&mut state)
}

fn assert_close(got: &[f64], want: &[f64], context: &str) {
    assert_eq!(got.len(), want.len());
    for (seat, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "{context}: seat {seat} got {g}, want {w}"
        );
    }
}

#[test]
fn three_players_in_a_clear_order() {
    // win [0,0,1] mean 1/3; rank [0, 1/2, 1] mean 1/2.
    assert_close(
        &returns_for(&[10, 20, 30]),
        &[-0.375, -0.25, 0.625],
        "3P 10/20/30",
    );
}

#[test]
fn a_tie_for_first_shares_the_win_and_the_rank() {
    // win [1/2, 1/2, 0]; the tied pair share ranks 1/2 and 1, so 3/4 each.
    assert_close(
        &returns_for(&[30, 30, 10]),
        &[0.1875, 0.1875, -0.375],
        "3P 30/30/10",
    );
}

#[test]
fn a_tie_for_last_shares_the_bottom_ranks() {
    // The two trailing seats share ranks 0 and 1/2, so 1/4 each.
    assert_close(
        &returns_for(&[30, 10, 10]),
        &[0.625, -0.3125, -0.3125],
        "3P 30/10/10",
    );
}

#[test]
fn four_players_in_a_clear_order() {
    // win [1,0,0,0] mean 1/4; rank [1, 2/3, 1/3, 0] mean 1/2.
    assert_close(
        &returns_for(&[40, 30, 20, 10]),
        &[0.6875, -0.1875 + 0.25 / 6.0, -0.1875 - 0.25 / 6.0, -0.3125],
        "4P 40/30/20/10",
    );
}

#[test]
fn five_players_all_tied_get_nothing() {
    assert_close(&returns_for(&[12; 5]), &[0.0; 5], "5P all tied");
    let mut state = game("usa", 5).new_initial_state(1);
    to_main(&mut state);
    for seat in 0..5 {
        state.score[seat] = 12;
        state.tickets[seat] = 0;
    }
    assert_eq!(winners(&mut state), vec![0, 1, 2, 3, 4]);
}

#[test]
fn five_players_with_a_three_way_tie_in_the_middle() {
    // win [1,0,0,0,0] mean 1/5; ranks: last 0, the middle three share 1/4..3/4 = 1/2,
    // first 1; mean 1/2.
    let want_mid = 0.75 * (0.0 - 0.2);
    assert_close(
        &returns_for(&[50, 20, 20, 20, 5]),
        &[
            0.75 * 0.8 + 0.25 * 0.5,
            want_mid,
            want_mid,
            want_mid,
            0.75 * (0.0 - 0.2) + 0.25 * (0.0 - 0.5),
        ],
        "5P 50/20/20/20/5",
    );
}
