//! Comparing two engines position by position, and saying *where* they part.
//!
//! The successor to the Python differential harness. Terminal-only comparison tells you
//! *that* two engines diverged, never *where*; so every quantity is compared after every
//! step, and a hash mismatch is resolved to the first serialized **field** that differs --
//! the difference between a two-minute fix and an afternoon.

use std::fmt::Write as _;

use ttr_core::board::{Board, get_board};

use crate::corpus::ObsMask;
use crate::engine::Engine;

/// Phase 3.2's contract allows continuous observation slots to move by up to 1e-6. Held
/// at exact equality until a change actually needs the allowance -- loosening it is a
/// decision to record in the worklog, not a default to inherit.
pub const CONTINUOUS_TOLERANCE: f32 = 0.0;

/// The field spans of docs/CONTRACT.md §3.1, in order.
///
/// `deck_len` is a field of the image rather than a constant: a reshuffle installs a
/// shorter deck, so every span after it moves. Read it from the bytes.
fn spans(board: &Board, n: usize, deck_len: usize) -> Vec<(&'static str, usize, usize)> {
    let k = board.n_card_types;
    let fields: [(&'static str, usize); 27] = [
        ("contract_version", 1),
        ("n_players", 1),
        ("phase", 1),
        ("current_player", 1),
        ("draws_left", 1),
        ("final_left", 1),
        ("pass_streak", 1),
        ("turn", 2),
        ("seg_owner", board.n_segments),
        ("hand", n * k),
        ("trains", n),
        ("score", 2 * n),
        ("tickets", 4 * n),
        ("deck_cursor", 2),
        ("deck_len", 2),
        ("deck", deck_len),
        ("discard", k),
        ("faceup", 5),
        ("tdeck_head", 1),
        ("tdeck_len", 1),
        ("tdeck", board.n_tickets),
        ("certain", n * k),
        ("unknown", n),
        ("offer_len", 1),
        ("offer", 3),
        ("rng.state", 8),
        ("rng.inc", 8),
    ];
    let mut at = 0;
    fields
        .iter()
        .map(|&(name, len)| {
            let span = (name, at, len);
            at += len;
            span
        })
        .collect()
}

fn deck_len_of(board: &Board, n: usize, image: &[u8]) -> usize {
    let k = board.n_card_types;
    let at = 9 + board.n_segments + n * k + n + 2 * n + 4 * n + 2;
    usize::from(u16::from_le_bytes([image[at], image[at + 1]]))
}

/// Name the first serialized field in which two images differ.
///
/// Field-wise, each image measured with its *own* `deck_len`: a reshuffle at a different
/// moment changes the image length, and a byte-offset diff would then blame the layout for
/// what is a value difference several fields earlier.
pub fn locate(map: &str, n_players: usize, mine: &[u8], theirs: &[u8]) -> String {
    let board = get_board(map).expect("a known board");
    let a_spans = spans(board, n_players, deck_len_of(board, n_players, mine));
    let b_spans = spans(board, n_players, deck_len_of(board, n_players, theirs));
    for (&(name, a_at, a_len), &(_, b_at, b_len)) in a_spans.iter().zip(&b_spans) {
        let a = &mine[a_at..a_at + a_len];
        let b = &theirs[b_at..b_at + b_len];
        if a == b {
            continue;
        }
        if a.len() != b.len() {
            return format!(
                "field {name:?} has different lengths: {} vs {} (implied by an earlier length field)",
                a.len(),
                b.len()
            );
        }
        let i = a
            .iter()
            .zip(b)
            .position(|(x, y)| x != y)
            .expect("they differ");
        return format!(
            "first differing field is {name:?}[{i}] (byte {}): {} vs {}",
            a_at + i,
            a[i],
            b[i]
        );
    }
    if mine.len() != theirs.len() {
        return format!(
            "every known field matches but the images differ in length: {} vs {}",
            mine.len(),
            theirs.len()
        );
    }
    "the images are byte-identical, so the mismatch is not in the serialization".into()
}

/// Everything two engines must agree on at one position, observations aside.
pub fn compare_position<A: Engine, B: Engine>(
    a: &A,
    sa: &A::State,
    b: &B,
    sb: &B::State,
) -> Result<(), String> {
    let (map, n) = (a.map_name(), a.n_players());
    if a.is_terminal(sa) != b.is_terminal(sb) {
        return Err(format!(
            "is_terminal: {} vs {}",
            a.is_terminal(sa),
            b.is_terminal(sb)
        ));
    }
    let (ha, hb) = (a.state_hash(sa), b.state_hash(sb));
    if ha != hb {
        return Err(format!(
            "state_hash {ha:016x} != {hb:016x}; {}",
            locate(map, n, &a.serialize(sa, false), &b.serialize(sb, false))
        ));
    }
    // The MCTS transposition key zeroes different bytes; a bug in the zeroing is invisible
    // to state_hash alone.
    let (pa, pb) = (a.position_hash(sa), b.position_hash(sb));
    if pa != pb {
        return Err(format!(
            "position_hash {pa:016x} != {pb:016x}; {}",
            locate(map, n, &a.serialize(sa, true), &b.serialize(sb, true))
        ));
    }
    let (la, lb) = (a.legal(sa), b.legal(sb));
    if la != lb {
        let only_a: Vec<u16> = la.iter().filter(|x| !lb.contains(x)).copied().collect();
        let only_b: Vec<u16> = lb.iter().filter(|x| !la.contains(x)).copied().collect();
        return Err(format!(
            "legal actions differ: {} vs {}; only first {:?}, only second {:?}",
            la.len(),
            lb.len(),
            &only_a[..only_a.len().min(8)],
            &only_b[..only_b.len().min(8)]
        ));
    }
    if !a.is_terminal(sa) && a.current_player(sa) != b.current_player(sb) {
        return Err(format!(
            "current_player: {} vs {}",
            a.current_player(sa),
            b.current_player(sb)
        ));
    }
    Ok(())
}

/// One seat's observation from both engines: discrete slots bit-exact, continuous slots
/// within [`CONTINUOUS_TOLERANCE`].
#[allow(clippy::too_many_arguments)]
pub fn compare_observation<A: Engine, B: Engine>(
    a: &A,
    sa: &mut A::State,
    b: &B,
    sb: &mut B::State,
    player: usize,
    mask: &ObsMask,
    buf_a: &mut [f32],
    buf_b: &mut [f32],
) -> Result<(), String> {
    a.encode(sa, player, buf_a);
    b.encode(sb, player, buf_b);
    let mut bad = 0usize;
    let mut first = String::new();
    for (slot, (&x, &y)) in buf_a.iter().zip(buf_b.iter()).enumerate() {
        let ok = if mask.continuous[slot] {
            x.to_bits() == y.to_bits() || (x - y).abs() <= CONTINUOUS_TOLERANCE
        } else {
            x.to_bits() == y.to_bits()
        };
        if !ok {
            if bad == 0 {
                let _ = write!(
                    first,
                    "slot {slot} ({}) {x:?} vs {y:?}",
                    slot_name(a.map_name(), slot)
                );
            }
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(format!(
            "seat {player} observation: {bad} of {} slots differ; first is {first}",
            buf_a.len()
        ));
    }
    Ok(())
}

/// `block[entity].field[i]` for an observation slot, from the spec.
pub fn slot_name(map: &str, slot: usize) -> String {
    let spec = ttr_core::obs::spec_for(map);
    for block in spec.blocks {
        if slot < block.offset || slot >= block.offset + block.count * block.stride {
            continue;
        }
        let within = slot - block.offset;
        let (entity, inside) = (within / block.stride, within % block.stride);
        for field in block.fields {
            if (field.offset..field.offset + field.width).contains(&inside) {
                return if block.count > 1 {
                    format!(
                        "{}[{entity}].{}[{}]",
                        block.name,
                        field.name,
                        inside - field.offset
                    )
                } else {
                    format!("{}.{}[{}]", block.name, field.name, inside - field.offset)
                };
            }
        }
    }
    "<past every block>".into()
}
