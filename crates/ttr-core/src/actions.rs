//! The flat, maskable action space. 915 on the USA map, 225 on TTR-mini.
//!
//! | Range | Size | Meaning |
//! | --- | --- | --- |
//! | `0 .. S*K-1` | `S*K` | `CLAIM`: `segment*K + pay`; pay `0..n_colors-1` = that colour, `n_colors` = pay entirely with locomotives |
//! | next 6 | 6 | `DRAW`: face-up slot 0-4, or 5 = blind |
//! | next 1 | 1 | `DRAW_TICKETS` |
//! | next 7 | 7 | `KEEP`: `bitmask - 1`, mask in 1..7 |
//! | next 1 | 1 | `PASS` |
//!
//! `S` is the segment count and `K` the card-type count (`n_colors + 1`). USA:
//! `100*9 + 15 = 915`.
//!
//! **The keep mask starts at 1, so there are 7 keep actions, not 8.** Keeping *nothing* is
//! never legal, and an off-by-one there shifts every action index above it, silently.
//!
//! **Locomotive payment is canonical**, which is what collapses the naive `100*9*7 = 6300`
//! space to 900. For a route of length `L` in colour `c`, let `k = min(hand[c], L)`; paying
//! `k` coloured plus `L-k` locomotives weakly dominates paying fewer coloured cards,
//! because the two resulting hands differ only by trading coloured cards for locomotives,
//! and a locomotive substitutes for `c` in every legal claim but not conversely. The usual
//! objection -- hoarding locomotives is strategically costly -- does not apply to base TTR,
//! which has no hand limit.

use crate::board::{Board, GRAY};

/// Bumped when the layout above changes. Baked into every checkpoint.
pub const ACTION_SPACE_VERSION: u32 = 1;

/// 5 face-up slots plus the blind draw.
pub const N_DRAW_ACTIONS: u16 = 6;

/// The blind draw's slot index.
pub const BLIND_SLOT: u16 = 5;

/// Keep masks 1..7 over a 3-ticket offer. Never 8: the empty keep is illegal.
pub const N_KEEP_ACTIONS: u16 = 7;

/// 6 draw + 1 draw-tickets + 7 keep + 1 pass.
pub const N_NON_CLAIM_ACTIONS: u16 = N_DRAW_ACTIONS + 1 + N_KEEP_ACTIONS + 1;

/// Action id arithmetic for one board. Immutable; built once and shared by every state.
#[derive(Clone, Copy, Debug)]
pub struct ActionSpace {
    pub k: u16,
    pub claim_end: u16,
    pub draw_base: u16,
    pub draw_tickets: u16,
    pub n: u16,
    pub pass_action: u16,
}

impl ActionSpace {
    pub fn new(board: &Board) -> Self {
        let k = board.n_card_types as u16;
        let claim_end = board.n_segments as u16 * k;
        let n = claim_end + N_NON_CLAIM_ACTIONS;
        Self {
            k,
            claim_end,
            draw_base: claim_end,
            draw_tickets: claim_end + N_DRAW_ACTIONS,
            n,
            pass_action: n - 1,
        }
    }

    // -- encode ------------------------------------------------------------

    #[inline]
    pub fn claim(&self, segment: u16, pay: u16) -> u16 {
        segment * self.k + pay
    }

    #[inline]
    pub fn draw(&self, slot: u16) -> u16 {
        self.draw_base + slot
    }

    /// `keep(mask) == keep_base() + mask` for `mask` in 1..=7.
    #[inline]
    pub fn keep(&self, mask: u16) -> u16 {
        debug_assert!(
            (1..=N_KEEP_ACTIONS).contains(&mask),
            "keep mask must be 1..7"
        );
        self.draw_tickets + mask
    }

    // -- decode ------------------------------------------------------------

    /// The id of the (always illegal) empty keep, so keep ids are `keep_base + mask`.
    #[inline]
    pub fn keep_base(&self) -> u16 {
        self.draw_tickets
    }

    #[inline]
    pub fn is_claim(&self, action: u16) -> bool {
        action < self.claim_end
    }

    #[inline]
    pub fn decode_claim(&self, action: u16) -> (u16, u16) {
        (action / self.k, action % self.k)
    }

    #[inline]
    pub fn decode_draw(&self, action: u16) -> u16 {
        action - self.draw_base
    }

    #[inline]
    pub fn decode_keep(&self, action: u16) -> u16 {
        action - self.draw_tickets
    }

    /// Human-readable, for the terminal client, replays and test failure messages.
    pub fn to_string(&self, board: &Board, action: u16) -> String {
        assert!(action < self.n, "action {action} outside 0..{}", self.n - 1);
        if action < self.claim_end {
            let (segment, pay) = self.decode_claim(action);
            let required = board.seg_color[segment as usize];
            let required_name = if required == GRAY {
                "gray"
            } else {
                board.raw.color_names[required as usize]
            };
            let paid = if pay as u8 == board.locomotive {
                "loco"
            } else {
                board.raw.color_names[pay as usize]
            };
            return format!(
                "CLAIM {}-{}[{}{}] pay {}",
                board.raw.cities[board.seg_a[segment as usize] as usize],
                board.raw.cities[board.seg_b[segment as usize] as usize],
                board.seg_len[segment as usize],
                required_name,
                paid,
            );
        }
        if action < self.draw_tickets {
            let slot = self.decode_draw(action);
            return if slot == BLIND_SLOT {
                "DRAW blind".to_string()
            } else {
                format!("DRAW faceup[{slot}]")
            };
        }
        if action == self.draw_tickets {
            return "DRAW_TICKETS".to_string();
        }
        if action < self.pass_action {
            let mask = self.decode_keep(action);
            let kept: Vec<usize> = (0..3).filter(|i| mask & (1 << i) != 0).collect();
            return format!("KEEP offer{kept:?}");
        }
        "PASS".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{all_boards, get_board};

    #[test]
    fn usa_is_nine_hundred_and_fifteen() {
        let space = ActionSpace::new(get_board("usa").unwrap());
        assert_eq!(space.n, 915);
        assert_eq!(space.claim_end, 900);
        assert_eq!(space.draw_base, 900);
        assert_eq!(space.draw_tickets, 906);
        assert_eq!(space.pass_action, 914);
        // 900 claim + 6 draw + 1 draw-tickets + 7 keep + 1 pass.
        assert_eq!(space.keep(1), 907);
        assert_eq!(space.keep(7), 913);
    }

    #[test]
    fn mini_is_two_hundred_and_twenty_five() {
        let space = ActionSpace::new(get_board("mini").unwrap());
        assert_eq!(space.n, 30 * 7 + 15);
        assert_eq!(space.n, 225);
    }

    #[test]
    fn there_are_seven_keep_actions_not_eight() {
        // Keeping nothing is never legal, so masks run 1..=7. An off-by-one here shifts
        // PASS and every index above it, silently.
        for board in all_boards() {
            let space = ActionSpace::new(board);
            assert_eq!(space.keep(N_KEEP_ACTIONS) + 1, space.pass_action);
            assert_eq!(space.keep_base() + 1, space.keep(1));
        }
    }

    #[test]
    fn claim_ids_round_trip() {
        for board in all_boards() {
            let space = ActionSpace::new(board);
            for segment in 0..board.n_segments as u16 {
                for pay in 0..space.k {
                    let id = space.claim(segment, pay);
                    assert!(space.is_claim(id));
                    assert_eq!(space.decode_claim(id), (segment, pay));
                }
            }
        }
    }

    #[test]
    fn the_ranges_tile_the_space_without_gaps_or_overlap() {
        for board in all_boards() {
            let space = ActionSpace::new(board);
            let mut seen = vec![false; space.n as usize];
            let mut mark = |id: u16| {
                assert!(
                    !seen[id as usize],
                    "{} action {id} claimed twice",
                    board.name
                );
                seen[id as usize] = true;
            };
            for segment in 0..board.n_segments as u16 {
                for pay in 0..space.k {
                    mark(space.claim(segment, pay));
                }
            }
            for slot in 0..N_DRAW_ACTIONS {
                mark(space.draw(slot));
            }
            mark(space.draw_tickets);
            for mask in 1..=N_KEEP_ACTIONS {
                mark(space.keep(mask));
            }
            mark(space.pass_action);
            assert!(
                seen.iter().all(|&s| s),
                "{} has unreachable ids",
                board.name
            );
        }
    }

    #[test]
    fn names_are_produced_for_every_action() {
        for board in all_boards() {
            let space = ActionSpace::new(board);
            for action in 0..space.n {
                assert!(!space.to_string(board, action).is_empty());
            }
        }
    }
}
