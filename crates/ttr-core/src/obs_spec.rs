//! The observation feature spec: **one declarative table**, resolved per board.
//!
//! Observation encoding is the most churn-prone code in the project, so its layout is
//! declared once here and every offset is derived from it. Adding a feature is a table edit
//! plus one write in [`crate::obs`], not a hand-maintained offset list. Python reads it back
//! through `ttr_rust.obs_spec()`.
//!
//! [`OBS_VERSION`] is baked into every checkpoint. Loading a checkpoint whose version does
//! not match is a hard error, which over a months-long league is the difference between
//! "old checkpoints fail loudly" and "old checkpoints silently play garbage and corrupt
//! the Elo table". **Any change to this table -- a field, a width, an order, a kind -- is
//! an `OBS_VERSION` bump**, and re-signs the golden corpus.
//!
//! Widths are symbolic where they depend on the board ([`Width`]), so one table describes
//! the 36-city USA map and 14-city TTR-mini. Slot counts the *network* must see as fixed --
//! opponent slots above all -- are pinned to the maximum seat count so one net plays any
//! table size.
//!
//! Design notes worth keeping next to the table:
//!
//! * **Perspective-normalized.** Everything is from the acting seat's point of view, with
//!   opponents ordered by distance in turn order (next to act = slot 0). Seat-*relative*
//!   rather than permutation-invariant, which is correct: "who moves before me" decides
//!   games. It is also a free N-times data multiplier.
//! * **Connectivity is computed, not learned.** `remaining_cost` is the single most
//!   important engineered feature in the project. Message passing does connectivity badly --
//!   a hop diameter around 10 would need ten layers and over-squash through the hubs -- so
//!   the environment computes it exactly and hands the network the answer.
//! * **Segment colour is an input feature, not part of the ID embedding.** The
//!   colour-symmetry regularizer permutes the eight non-locomotive colours across hand,
//!   display, discard and required colour to produce an isomorphic game; that only works if
//!   the required colour is something the permutation can reach.
//! * **Every field has a [`Kind`].** A flag holds exactly 0.0 or 1.0; a ratio is an integer
//!   over a constant. The golden corpus digests the two separately, because Phase 3.2's
//!   contract is "flags bit-exact, ratios within 1e-6".

use std::sync::OnceLock;

use crate::board::{Board, FACEUP_SLOTS, MAX_PLAYERS, all_boards};

/// Bumped whenever the table below changes shape or meaning. Baked into every checkpoint.
///
/// - **1**: the table as the Python encoder signed it (Phase 3.1's golden corpus).
/// - **2**: the same layout, three values corrected. The plan (`on_my_steiner_tree`,
///   `fragility`) follows a breadth-first route that cannot be truncated by my own
///   zero-cost track; `extends_my_chain` means touching a city my track reaches (1 read the
///   union-find, and missed each component's root city); and own-hand counts divide each
///   card type by its own printed count (1 divided locomotives by the colour count).
pub const OBS_VERSION: u32 = 2;

/// Opponent slots are fixed at the maximum table size so one network plays any seat count;
/// unused slots are zeroed and carry a `present` flag.
pub const OPPONENT_SLOTS: usize = MAX_PLAYERS - 1;

/// Thermometer bucket edges. A thermometer beats a raw count for a network because "at
/// least three" is a linearly separable question and "exactly three" is not.
pub const HAND_BUCKETS: [u8; 7] = [1, 2, 3, 4, 5, 6, 7];
pub const COST_BUCKETS: [u16; 6] = [1, 3, 5, 8, 12, 17];
pub const TRAIN_BUCKETS: [u8; 7] = [1, 3, 6, 10, 15, 25, 35];

/// What a field's slots can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Exactly 0.0 or 1.0: one-hots, booleans, thermometers.
    Flag,
    /// An integer divided by a constant. May exceed 1 where the constant is a rough scale.
    Ratio,
}

/// A width, either fixed or resolved against the board.
#[derive(Clone, Copy, Debug)]
pub enum Width {
    Fixed(usize),
    /// Card types, locomotive last: USA 9, mini 7.
    CardTypes,
    /// Card types plus "empty slot": USA 10, mini 8.
    CardTypesOrEmpty,
    /// Segment colours plus gray: USA 9, mini 7.
    ColorsOrGray,
    /// Free, me, then every opponent slot.
    Owner,
}

/// Which entities a block repeats over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    Once,
    Segments,
    Tickets,
    Opponents,
    Faceup,
}

/// One named run of floats inside a block.
#[derive(Clone, Copy, Debug)]
pub struct FieldDef {
    pub name: &'static str,
    pub width: Width,
    pub kind: Kind,
    pub doc: &'static str,
}

/// A group of fields, optionally repeated once per entity.
#[derive(Clone, Copy, Debug)]
pub struct BlockDef {
    pub name: &'static str,
    pub repeat: Repeat,
    pub fields: &'static [FieldDef],
}

const fn f(name: &'static str, width: Width, kind: Kind, doc: &'static str) -> FieldDef {
    FieldDef {
        name,
        width,
        kind,
        doc,
    }
}

use Kind::{Flag, Ratio};
use Width::{CardTypes, CardTypesOrEmpty, ColorsOrGray, Fixed, Owner};

/// The table. Order is layout: blocks in sequence, fields in sequence within a block.
pub const FEATURE_SPEC: &[BlockDef] = &[
    BlockDef {
        name: "segment_static",
        repeat: Repeat::Segments,
        fields: &[f(
            "required_color",
            ColorsOrGray,
            Flag,
            "One-hot over the colours plus gray. An input feature rather than part of the \
             segment-ID embedding, so the colour-permutation regularizer can reach it.",
        )],
    },
    BlockDef {
        name: "segment_dynamic",
        repeat: Repeat::Segments,
        fields: &[
            f(
                "owner",
                Owner,
                Flag,
                "One-hot: free, me, then opponents in turn order.",
            ),
            f(
                "closed",
                Fixed(1),
                Flag,
                "Blocked for everyone: a 2-3P double whose twin is claimed.",
            ),
            f(
                "twin_locked",
                Fixed(1),
                Flag,
                "I own the other track of this double route.",
            ),
            f(
                "can_afford_now",
                Fixed(1),
                Flag,
                "A legal claim for me this turn.",
            ),
            f(
                "cards_short",
                Fixed(HAND_BUCKETS.len()),
                Flag,
                "Thermometer over cards still needed.",
            ),
            f(
                "on_my_steiner_tree",
                Fixed(1),
                Flag,
                "Free, and on my plan: the union of each open ticket's cheapest route (the \
                 fewest segments among the cheapest), standing in for the Steiner tree.",
            ),
            f(
                "extends_my_chain",
                Fixed(1),
                Flag,
                "Free, and touches a city my own track already reaches.",
            ),
        ],
    },
    BlockDef {
        name: "own_hand",
        repeat: Repeat::Once,
        fields: &[
            f(
                "counts",
                CardTypes,
                Ratio,
                "Card counts, each over that type's printed count; locomotive last.",
            ),
            f("thermometer", CardTypes, Flag, "Per colour: at least one."),
            f("thermometer2", CardTypes, Flag, "Per colour: at least two."),
            f(
                "thermometer3",
                CardTypes,
                Flag,
                "Per colour: at least three.",
            ),
            f(
                "thermometer4",
                CardTypes,
                Flag,
                "Per colour: at least four.",
            ),
            f(
                "thermometer5",
                CardTypes,
                Flag,
                "Per colour: at least five.",
            ),
            f("thermometer6", CardTypes, Flag, "Per colour: at least six."),
        ],
    },
    BlockDef {
        name: "tickets",
        repeat: Repeat::Tickets,
        fields: &[
            f("held", Fixed(1), Flag, "In my hand."),
            f(
                "connected",
                Fixed(1),
                Flag,
                "Already completed by my network.",
            ),
            f(
                "remaining_cost",
                Fixed(1),
                Ratio,
                "Train cars still needed, normalized. The single most important engineered \
                 feature in the project.",
            ),
            f(
                "cost_thermometer",
                Fixed(COST_BUCKETS.len()),
                Flag,
                "Buckets over remaining cost.",
            ),
            f(
                "is_dead",
                Fixed(1),
                Flag,
                "An opponent has cut every route; the points are lost.",
            ),
            f("points", Fixed(1), Ratio, "Ticket value, normalized."),
            f(
                "fragility",
                Fixed(1),
                Ratio,
                "Worst-case cost increase if one enemy claim lands on my best path. \
                 Separates a safe connection from one hanging on a single contested edge.",
            ),
        ],
    },
    BlockDef {
        name: "steiner",
        repeat: Repeat::Once,
        fields: &[
            f(
                "remaining_cost",
                Fixed(1),
                Ratio,
                "Cheapest tree spanning my held tickets, normalized.",
            ),
            f(
                "cost_minus_trains",
                Fixed(1),
                Ratio,
                "Slack: negative means I cannot finish.",
            ),
            f(
                "exact",
                Fixed(1),
                Flag,
                "0 when the terminal cap forced the MST upper bound.",
            ),
        ],
    },
    BlockDef {
        name: "faceup",
        repeat: Repeat::Faceup,
        fields: &[f(
            "card",
            CardTypesOrEmpty,
            Flag,
            "One-hot per slot, last position = empty. Slot order is action order.",
        )],
    },
    BlockDef {
        name: "piles",
        repeat: Repeat::Once,
        fields: &[
            f("deck_size", Fixed(1), Ratio, "Undrawn cards, normalized."),
            f(
                "discard_size",
                Fixed(1),
                Ratio,
                "Discarded cards, normalized.",
            ),
            f(
                "ticket_deck_size",
                Fixed(1),
                Ratio,
                "Tickets left, normalized.",
            ),
            f(
                "discard_composition",
                CardTypes,
                Ratio,
                "What is in the discard, normalized.",
            ),
            f(
                "unseen",
                CardTypes,
                Ratio,
                "What I cannot account for: deck plus opponents' blind draws.",
            ),
        ],
    },
    BlockDef {
        name: "opponents",
        repeat: Repeat::Opponents,
        fields: &[
            f(
                "present",
                Fixed(1),
                Flag,
                "0 for an unused slot at a smaller table.",
            ),
            f("trains", Fixed(1), Ratio, "Trains left, normalized."),
            f(
                "trains_thermometer",
                Fixed(TRAIN_BUCKETS.len()),
                Flag,
                "Buckets over trains left.",
            ),
            f("score", Fixed(1), Ratio, "Banked route points, normalized."),
            f("hand_size", Fixed(1), Ratio, "Cards held, normalized."),
            f("ticket_count", Fixed(1), Ratio, "Tickets held, normalized."),
            f(
                "blind_draws",
                Fixed(1),
                Ratio,
                "Cards drawn blind and not yet spent, normalized.",
            ),
            f(
                "segments_claimed",
                Fixed(1),
                Ratio,
                "Routes claimed, normalized.",
            ),
            f(
                "longest_chain",
                Fixed(1),
                Ratio,
                "Longest trail so far, normalized.",
            ),
            f(
                "certain",
                CardTypes,
                Ratio,
                "Cards this seat is publicly known to hold.",
            ),
            f(
                "max_possible",
                CardTypes,
                Ratio,
                "Upper bound on this seat's holding of each colour.",
            ),
        ],
    },
    BlockDef {
        name: "clock",
        repeat: Repeat::Once,
        fields: &[
            f("phase", Fixed(5), Flag, "One-hot over the five phases."),
            f(
                "seats",
                Fixed(OPPONENT_SLOTS + 1),
                Flag,
                "One-hot over table size, 2..5.",
            ),
            f(
                "draws_left",
                Fixed(1),
                Flag,
                "1 when a second card draw is owed.",
            ),
            f(
                "final_triggered",
                Fixed(1),
                Flag,
                "The end has been triggered.",
            ),
            f(
                "final_countdown",
                Fixed(1),
                Ratio,
                "Turns left once triggered, normalized.",
            ),
            f("turn", Fixed(1), Ratio, "Turn index, normalized."),
            f("my_trains", Fixed(1), Ratio, "My trains left, normalized."),
            f(
                "my_trains_thermometer",
                Fixed(TRAIN_BUCKETS.len()),
                Flag,
                "Buckets over my trains.",
            ),
            f(
                "my_score",
                Fixed(1),
                Ratio,
                "My banked route points, normalized.",
            ),
            f(
                "my_ticket_count",
                Fixed(1),
                Ratio,
                "Tickets I hold, normalized.",
            ),
            f(
                "score_rank",
                Fixed(1),
                Ratio,
                "My rank on banked points, 1.0 = leading.",
            ),
            f(
                "gap_to_leader",
                Fixed(1),
                Ratio,
                "Points behind the leader, normalized.",
            ),
        ],
    },
];

// ---------------------------------------------------------------------------
// Resolved against a board
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsField {
    pub name: &'static str,
    /// Offset within one entity of the enclosing block.
    pub offset: usize,
    pub width: usize,
    pub kind: Kind,
    pub doc: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsBlock {
    pub name: &'static str,
    /// Entities this block repeats over; 1 for a singleton.
    pub count: usize,
    /// Floats per entity.
    pub stride: usize,
    /// Absolute offset of entity 0.
    pub offset: usize,
    pub fields: Vec<ObsField>,
}

impl ObsBlock {
    pub fn field(&self, name: &str) -> Option<&ObsField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsSpec {
    pub map: &'static str,
    pub size: usize,
    pub blocks: Vec<ObsBlock>,
}

impl ObsSpec {
    pub fn block(&self, name: &str) -> Option<&ObsBlock> {
        self.blocks.iter().find(|b| b.name == name)
    }
}

fn resolve_width(width: Width, board: &Board) -> usize {
    match width {
        Width::Fixed(n) => n,
        Width::CardTypes => board.n_card_types,
        Width::CardTypesOrEmpty => board.n_card_types + 1,
        Width::ColorsOrGray => board.n_colors + 1,
        Width::Owner => 2 + OPPONENT_SLOTS,
    }
}

/// The table, resolved against one board: concrete widths and offsets.
pub fn resolve(board: &Board) -> ObsSpec {
    let mut blocks = Vec::with_capacity(FEATURE_SPEC.len());
    let mut offset = 0;
    for def in FEATURE_SPEC {
        let mut stride = 0;
        let fields: Vec<ObsField> = def
            .fields
            .iter()
            .map(|field| {
                let width = resolve_width(field.width, board);
                let resolved = ObsField {
                    name: field.name,
                    offset: stride,
                    width,
                    kind: field.kind,
                    doc: field.doc,
                };
                stride += width;
                resolved
            })
            .collect();
        let count = match def.repeat {
            Repeat::Once => 1,
            Repeat::Segments => board.n_segments,
            Repeat::Tickets => board.n_tickets,
            Repeat::Opponents => OPPONENT_SLOTS,
            Repeat::Faceup => FACEUP_SLOTS,
        };
        blocks.push(ObsBlock {
            name: def.name,
            count,
            stride,
            offset,
            fields,
        });
        offset += count * stride;
    }
    ObsSpec {
        map: board.name,
        size: offset,
        blocks,
    }
}

/// The resolved spec for a board, computed once.
///
/// # Panics
/// For a map with no board.
pub fn spec_for(map: &str) -> &'static ObsSpec {
    static SPECS: OnceLock<Vec<ObsSpec>> = OnceLock::new();
    SPECS
        .get_or_init(|| all_boards().iter().map(resolve).collect())
        .iter()
        .find(|s| s.map == map)
        .unwrap_or_else(|| panic!("no board named {map:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::get_board;
    use crate::hashing::hash128_hex;

    /// The layout as blake2b-128 over `map size` / `block count stride offset` /
    /// `field offset width` lines, and the versions signed under it. Taken from the
    /// generated table this module replaced, so the move from Python-generated to
    /// hand-maintained is proven layout-preserving; version 2 changed values, not layout.
    const LAYOUT: &str = "ebdd9a514988b2eb6533c1921c2602d3";
    const LAYOUT_VERSIONS: &[u32] = &[1, 2];

    #[test]
    fn the_layout_is_the_one_the_corpus_was_signed_under() {
        assert!(
            LAYOUT_VERSIONS.contains(&OBS_VERSION),
            "a version bump must re-pin the layout digest"
        );
        let mut text = String::new();
        for map in ["usa", "mini"] {
            let spec = spec_for(map);
            text.push_str(&format!("{} {}\n", spec.map, spec.size));
            for b in &spec.blocks {
                text.push_str(&format!(
                    "{} {} {} {}\n",
                    b.name, b.count, b.stride, b.offset
                ));
                for f in &b.fields {
                    text.push_str(&format!("  {} {} {}\n", f.name, f.offset, f.width));
                }
            }
        }
        assert_eq!(
            hash128_hex(text.as_bytes()),
            LAYOUT,
            "the observation layout moved without an OBS_VERSION bump:\n{text}"
        );
    }

    #[test]
    fn names_are_unique_and_every_field_is_documented() {
        let mut blocks: Vec<&str> = FEATURE_SPEC.iter().map(|b| b.name).collect();
        blocks.sort_unstable();
        blocks.dedup();
        assert_eq!(blocks.len(), FEATURE_SPEC.len(), "duplicate block names");
        for block in FEATURE_SPEC {
            let mut fields: Vec<&str> = block.fields.iter().map(|f| f.name).collect();
            fields.sort_unstable();
            fields.dedup();
            assert_eq!(
                fields.len(),
                block.fields.len(),
                "duplicate field in {}",
                block.name
            );
            for field in block.fields {
                assert!(
                    !field.doc.trim().is_empty(),
                    "{}.{} is undocumented; the doc is the only description of a feature",
                    block.name,
                    field.name
                );
            }
        }
    }

    #[test]
    fn symbolic_widths_and_counts_follow_the_board() {
        // The check the retired generator's staleness test used to make for free: a board's
        // spec is sized by that board, so a new map gets a correct layout or a failing test.
        for board in all_boards() {
            let spec = spec_for(board.name);
            let width = |block: &str, field: &str| {
                spec.block(block)
                    .and_then(|b| b.field(field))
                    .unwrap_or_else(|| panic!("{block}.{field}"))
                    .width
            };
            let count = |block: &str| spec.block(block).unwrap().count;
            assert_eq!(count("segment_static"), board.n_segments);
            assert_eq!(count("segment_dynamic"), board.n_segments);
            assert_eq!(count("tickets"), board.n_tickets);
            assert_eq!(count("opponents"), OPPONENT_SLOTS);
            assert_eq!(count("faceup"), FACEUP_SLOTS);
            assert_eq!(
                width("segment_static", "required_color"),
                board.n_colors + 1
            );
            assert_eq!(width("segment_dynamic", "owner"), 2 + OPPONENT_SLOTS);
            assert_eq!(width("faceup", "card"), board.n_card_types + 1);
            assert_eq!(width("own_hand", "counts"), board.n_card_types);
            assert_eq!(width("segment_dynamic", "cards_short"), HAND_BUCKETS.len());
            assert_eq!(width("tickets", "cost_thermometer"), COST_BUCKETS.len());
            assert_eq!(
                width("opponents", "trains_thermometer"),
                TRAIN_BUCKETS.len()
            );
            assert_eq!(width("clock", "seats"), MAX_PLAYERS);

            let mut at = 0;
            for block in &spec.blocks {
                assert_eq!(
                    block.offset, at,
                    "{} {} does not start where the last ended",
                    board.name, block.name
                );
                assert_eq!(
                    block.stride,
                    block.fields.iter().map(|f| f.width).sum::<usize>()
                );
                at += block.count * block.stride;
            }
            assert_eq!(at, spec.size);
        }
        assert!(get_board("usa").is_some() && spec_for("usa").size > spec_for("mini").size);
    }
}
