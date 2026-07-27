//! Determinization: sample a full state consistent with what one seat can actually see.
//!
//! This is the primitive ISMCTS is built on (PLAN.md §5.5), and it is the reason the state
//! carries `certain` and `unknown` at all. **It is built now rather than in Phase 5
//! deliberately**: it is the one search hook that constrains the *state layout*, and if
//! the unseen-pool arithmetic did not close against the fields the engine keeps, that is a
//! second-port-pass problem best discovered while the layout is still free (§8.1
//! mitigation 2).
//!
//! Far less is hidden in Ticket to Ride than people assume. Face-up takes are public, and
//! cards spent on a claim are publicly discarded. The invariant the engine maintains is
//! `hand[p] == certain[p] + unknown[p] blind-drawn cards`, so determinization is:
//!
//! 1. the unseen pool is the printed deck minus my hand, minus the discard, minus the
//!    face-up display, minus every opponent's `certain`;
//! 2. deal `unknown[q]` cards to each opponent by multivariate hypergeometric;
//! 3. **the remainder is the deck.**
//!
//! ## Two approximations, stated so nobody "fixes" them
//!
//! * **Uniform resampling ignores correlation with reshuffle boundaries.** A card that
//!   went into the discard and came back around is treated as interchangeable with one
//!   that never left the deck. Modelling that exactly would require reconstructing when
//!   each reshuffle happened relative to each blind draw.
//! * **The ticket deck's bottom-return ordering is not reconstructed.** Returned tickets
//!   go to the bottom and can come back around, which is real information; the sampler
//!   leaves the ticket ring exactly as it is rather than guessing an order.
//!
//! Both are deliberate and both are documented in PLAN.md §5.5. Neither affects the
//! conservation laws [`assert_consistent`] checks.
//!
//! ## Not frozen, and not cross-checked against Python
//!
//! docs/CONTRACT.md freezes the PRNG, the draw procedure and `state_hash()`. The sampler
//! is none of those, so there is no requirement that it produce the same particles as some
//! future Python twin -- and writing one purely to compare against would be weak evidence
//! anyway, since two implementations agreeing is exactly what a shared misunderstanding
//! looks like. [`assert_consistent`] is the stronger check: it verifies the sample against
//! the *public state*, so it catches both sides being wrong.

use crate::board::FACEUP_SLOTS;
use crate::config::EMPTY_SLOT;
use crate::rng::Pcg32;
use crate::state::State;

/// A conservation law the sample broke.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inconsistent(pub String);

impl std::fmt::Display for Inconsistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Inconsistent {}

/// Sample a state consistent with everything `observer` can see.
///
/// The observer's own hand, the discard, the face-up display, every seat's ticket holdings
/// and every public counter are reproduced exactly. Opponents' hidden cards and the
/// undrawn deck are resampled.
///
/// In debug builds every sample is checked with [`assert_consistent`] before it is
/// returned, because determinization bugs are silent and devastating: a sampler that leaks
/// a card produces plausible-looking particles and a search that is confidently wrong.
///
/// # Panics
/// If the public state is itself inconsistent, which would mean the *engine* has a bug
/// rather than the sampler.
pub fn resample_from_infoset(state: &State, observer: usize, rng: &mut Pcg32) -> State {
    let board = state.board;
    let k = board.n_card_types;
    let n = state.n_players();

    // The unseen pool: what the observer cannot account for.
    let unseen = state.unseen_counts(observer);
    let mut pool: Vec<u8> = unseen
        .iter()
        .map(|&c| {
            u8::try_from(c).expect("the unseen pool went negative; the engine's bookkeeping broke")
        })
        .collect();

    let mut sampled = state.clone();

    // Deal each opponent its blind-drawn count by multivariate hypergeometric: draw one
    // card at a time from the pool weighted by what is left, which is exactly sampling
    // without replacement from the multiset.
    for seat in 0..n {
        if seat == observer {
            continue;
        }
        let blind = state.unknown[seat] as usize;
        // Keep what is publicly known; resample only the blind draws.
        for c in 0..k {
            sampled.hand[seat * k + c] = state.certain[seat * k + c];
        }
        for _ in 0..blind {
            let card = draw_from_pool(&mut pool, rng)
                .expect("the unseen pool ran dry before every blind draw was dealt");
            sampled.hand[seat * k + card] += 1;
        }
    }

    // Whatever is left *is* the deck. Rebuilt in canonical order and shuffled, exactly as a
    // reshuffle does -- the observer knows the multiset, never the order.
    let mut deck: Vec<u8> = Vec::with_capacity(pool.iter().map(|&c| c as usize).sum());
    for (card_type, &count) in pool.iter().enumerate() {
        deck.extend(std::iter::repeat_n(card_type as u8, count as usize));
    }
    rng.shuffle(&mut deck);
    sampled.deck[..deck.len()].copy_from_slice(&deck);
    sampled.deck_len = deck.len() as u16;
    // The consumed prefix is gone: a determinization is a *position*, not a replay, and
    // `position_hash` zeroes that prefix for exactly this reason.
    sampled.deck_pos = 0;

    debug_assert!(
        assert_consistent(&sampled, state, observer).is_ok(),
        "{}",
        assert_consistent(&sampled, state, observer).unwrap_err()
    );
    sampled
}

/// Draw one card from a count vector, weighted by what remains. Returns `None` if empty.
fn draw_from_pool(pool: &mut [u8], rng: &mut Pcg32) -> Option<usize> {
    let total: u32 = pool.iter().map(|&c| u32::from(c)).sum();
    if total == 0 {
        return None;
    }
    let mut pick = rng.below(total);
    for (card_type, count) in pool.iter_mut().enumerate() {
        let have = u32::from(*count);
        if pick < have {
            *count -= 1;
            return Some(card_type);
        }
        pick -= have;
    }
    unreachable!("the weighted pick fell past the end of a non-empty pool")
}

/// Check a sample against the public state it claims to be consistent with.
///
/// Run on **every** sample in debug builds. Determinization bugs are silent: a sampler
/// that quietly loses a card still produces states that play, and the search built on them
/// is confidently wrong with nothing in the logs.
///
/// Checks, in the order PLAN.md §5.5 lists them: per-type card conservation against the
/// printed deck, every seat's hand *size*, `opp_hand ⊇ certain`, ticket disjointness and
/// counts, and exact reproduction of all public state.
pub fn assert_consistent(
    sample: &State,
    public: &State,
    observer: usize,
) -> Result<(), Inconsistent> {
    let board = sample.board;
    let k = board.n_card_types;
    let n = sample.n_players();

    macro_rules! fail {
        ($($arg:tt)*) => { return Err(Inconsistent(format!($($arg)*))) };
    }

    // 1. Per-type conservation against what is printed: {cards_per_color x n_colors, locos}.
    for c in 0..k {
        let held: u32 = (0..n).map(|p| u32::from(sample.hand[p * k + c])).sum();
        let faceup = sample
            .faceup
            .iter()
            .filter(|&&x| x != EMPTY_SLOT && x as usize == c)
            .count() as u32;
        let in_deck = sample.deck[sample.deck_pos as usize..sample.deck_len as usize]
            .iter()
            .filter(|&&x| x as usize == c)
            .count() as u32;
        let total = held + faceup + u32::from(sample.discard[c]) + in_deck;
        let printed = u32::from(board.cards_per_type(c as u8));
        if total != printed {
            fail!(
                "card type {} ({}): {total} in the sample, {printed} printed",
                c,
                board.color_name(c as u8)
            );
        }
    }

    // 2. Every seat's hand size is public, and the observer's hand is public *exactly*.
    for p in 0..n {
        if sample.hand_size(p) != public.hand_size(p) {
            fail!(
                "seat {p} holds {} cards in the sample, {} publicly",
                sample.hand_size(p),
                public.hand_size(p)
            );
        }
        if p == observer && sample.hand_of(p) != public.hand_of(p) {
            fail!("the observer's own hand was resampled");
        }
        // 3. opp_hand contains certain: nothing publicly known may go missing.
        for c in 0..k {
            let known = public.certain[p * k + c];
            if sample.hand[p * k + c] < known {
                fail!(
                    "seat {p} is publicly known to hold {known} {} but the sample gives it {}",
                    board.color_name(c as u8),
                    sample.hand[p * k + c]
                );
            }
        }
    }

    // 4. Tickets: disjoint across seats, and counts unchanged.
    let mut seen = 0u32;
    for p in 0..n {
        if sample.tickets[p] != public.tickets[p] {
            fail!("seat {p}'s tickets were resampled; ticket holdings are not resampled");
        }
        if seen & sample.tickets[p] != 0 {
            fail!("seat {p} holds a ticket another seat also holds");
        }
        seen |= sample.tickets[p];
    }
    let in_hands = seen.count_ones();
    if in_hands + u32::from(sample.tdeck_len) + u32::from(sample.offer_len)
        != board.n_tickets as u32
    {
        fail!(
            "tickets leaked: {in_hands} held + {} in deck",
            sample.tdeck_len
        );
    }

    // 5. Every piece of public state reproduced exactly.
    if sample.discard[..k] != public.discard[..k] {
        fail!("the discard is public and was changed");
    }
    if sample.faceup[..FACEUP_SLOTS] != public.faceup[..FACEUP_SLOTS] {
        fail!("the face-up display is public and was changed");
    }
    if sample.seg_owner[..board.n_segments] != public.seg_owner[..board.n_segments] {
        fail!("the board is public and was changed");
    }
    for p in 0..n {
        if sample.trains[p] != public.trains[p] || sample.score[p] != public.score[p] {
            fail!("seat {p}'s trains or banked score changed");
        }
    }
    if (
        sample.phase,
        sample.cur,
        sample.turn,
        sample.final_left,
        sample.pass_streak,
    ) != (
        public.phase,
        public.cur,
        public.turn,
        public.final_left,
        public.pass_streak,
    ) {
        fail!("a public turn counter changed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::rng::{Part, stream};
    use crate::state::Game;

    fn mid_game(map: &str, n: usize, seed: u64, steps: usize) -> State {
        let game = Game::new(RuleConfig::new(map, n).unwrap()).unwrap();
        let mut state = game.new_initial_state(seed);
        let mut rng = stream(seed, &[Part::Str("resample-setup")]);
        let mut scratch = Vec::new();
        for _ in 0..steps {
            if state.is_terminal() {
                break;
            }
            let Some(a) = state.sample_legal_into(&mut rng, &mut scratch) else {
                break;
            };
            state.step(a).unwrap();
        }
        state
    }

    #[test]
    fn samples_are_consistent_across_maps_seats_and_depths() {
        for (map, seats) in [("usa", 2..=5), ("mini", 2..=4)] {
            for n in seats {
                for seed in 0..4u64 {
                    for depth in [0usize, 20, 80, 200] {
                        let state = mid_game(map, n, seed, depth);
                        for observer in 0..n {
                            let mut rng = stream(seed, &[Part::Str("resample"), Part::Int(7)]);
                            for _ in 0..5 {
                                let sample = resample_from_infoset(&state, observer, &mut rng);
                                assert_consistent(&sample, &state, observer).unwrap_or_else(|e| {
                                    panic!(
                                        "{map} {n}P seed {seed} depth {depth} obs {observer}: {e}"
                                    )
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_sample_is_a_playable_state() {
        // Consistency is not enough: the particle has to be steppable, or search would
        // build a tree over positions the engine rejects.
        let state = mid_game("usa", 3, 2, 60);
        let mut rng = stream(2, &[Part::Str("play")]);
        let mut sample = resample_from_infoset(&state, 0, &mut rng);
        sample
            .validate()
            .expect("a sample must satisfy the engine's own invariants");
        let mut scratch = Vec::new();
        for _ in 0..50 {
            if sample.is_terminal() {
                break;
            }
            let a = sample.sample_legal_into(&mut rng, &mut scratch).unwrap();
            sample.step(a).unwrap();
            sample.validate().unwrap();
        }
    }

    #[test]
    fn resampling_actually_varies_the_hidden_cards() {
        // A sampler that returned the true state every time would pass every consistency
        // check and be useless. This is the guard on that.
        let state = mid_game("usa", 2, 5, 60);
        let mut rng = stream(5, &[Part::Str("vary")]);
        let mut hashes = std::collections::HashSet::new();
        for _ in 0..25 {
            hashes.insert(resample_from_infoset(&state, 0, &mut rng).position_hash());
        }
        assert!(
            hashes.len() > 1,
            "every determinization was identical; the sampler is not sampling"
        );
    }

    #[test]
    fn the_observer_sees_its_own_hand_unchanged() {
        for observer in 0..3 {
            let state = mid_game("usa", 3, 9, 70);
            let mut rng = stream(9, &[Part::Str("own")]);
            let sample = resample_from_infoset(&state, observer, &mut rng);
            assert_eq!(sample.hand_of(observer), state.hand_of(observer));
        }
    }

    #[test]
    fn the_consistency_check_is_not_vacuous() {
        // Guard the guard: hand-corrupt a sample in each of the ways that matter and
        // confirm each is caught. A checker that passes everything is worse than none.
        let state = mid_game("usa", 2, 4, 60);
        let mut rng = stream(4, &[Part::Str("guard")]);
        let good = resample_from_infoset(&state, 0, &mut rng);
        assert!(assert_consistent(&good, &state, 0).is_ok());

        let k = state.board.n_card_types;

        // A card conjured into an opponent's hand.
        let mut extra = good.clone();
        extra.hand[k] += 1;
        assert!(
            assert_consistent(&extra, &state, 0).is_err(),
            "an extra card passed"
        );

        // The observer's own hand quietly resampled.
        let mut mine = good.clone();
        let c = (0..k)
            .find(|&c| mine.hand[c] > 0)
            .expect("the observer holds something");
        mine.hand[c] -= 1;
        assert!(
            assert_consistent(&mine, &state, 0).is_err(),
            "a changed own hand passed"
        );

        // The public discard altered.
        let mut discarded = good.clone();
        discarded.discard[0] += 1;
        assert!(
            assert_consistent(&discarded, &state, 0).is_err(),
            "a changed discard passed"
        );

        // Publicly-known cards taken away from an opponent.
        let mut robbed = good.clone();
        if let Some(c) = (0..k).find(|&c| state.certain[k + c] > 0) {
            robbed.hand[k + c] -= 1;
            assert!(
                assert_consistent(&robbed, &state, 0).is_err(),
                "losing a publicly-known card passed"
            );
        }
    }
}
