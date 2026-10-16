//! One interface over "an engine", so a drive written once runs against the live crate and
//! against the frozen `oracle-v1` build.
//!
//! The two are different crates with different types, so the interface is a trait and the
//! implementation is a macro instantiated once per crate: [`Live`] here, and the oracle in
//! the tests that depend on it. A macro rather than two hand-written impls because the two
//! must be the same code -- an impl that called `final_scores` on one side and summed the
//! breakdown on the other would be comparing two different questions.

/// Everything a finished game settles, in the order the corpus folds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Terminal {
    pub scores: Vec<i16>,
    /// Per seat: routes, tickets made, tickets missed, ticket points, longest-path bonus,
    /// longest trail, completed tickets, total -- `scoring::Breakdown` in field order.
    pub breakdown: Vec<[i32; 8]>,
    pub returns: Vec<f64>,
    pub winners: Vec<usize>,
    pub trails: Vec<u16>,
}

impl Terminal {
    /// Seats tied for first, as a bitmask.
    pub fn winners_mask(&self) -> u8 {
        self.winners.iter().fold(0u8, |mask, &p| mask | 1 << p)
    }
}

/// The slice of an engine the golden-artifact drives touch.
pub trait Engine: Sync {
    type State: Clone + Send;

    fn n_players(&self) -> usize;
    fn map_name(&self) -> &'static str;
    fn obs_size(&self) -> usize;
    fn initial(&self, seed: u64) -> Self::State;
    /// Panics on an illegal action: every caller drives from the legal list.
    fn step(&self, state: &mut Self::State, action: u16);
    /// Sorted ascending.
    fn legal(&self, state: &Self::State) -> Vec<u16>;
    fn state_hash(&self, state: &Self::State) -> u64;
    fn position_hash(&self, state: &Self::State) -> u64;
    fn serialize(&self, state: &Self::State, canonical: bool) -> Vec<u8>;
    fn is_terminal(&self, state: &Self::State) -> bool;
    fn current_player(&self, state: &Self::State) -> usize;
    fn encode(&self, state: &mut Self::State, player: usize, out: &mut [f32]);
    fn terminal(&self, state: &mut Self::State) -> Terminal;
}

/// Implement [`Engine`] for one crate that has `ttr-core`'s module layout.
///
/// `engine_for!(Oracle, ttr_oracle)` defines `pub struct Oracle` over that crate's `Game`.
#[macro_export]
macro_rules! engine_for {
    ($(#[$meta:meta])* $name:ident, $krate:ident) => {
        $(#[$meta])*
        pub struct $name {
            game: $krate::state::Game,
        }

        impl $name {
            pub fn new(map: &str, n_players: usize) -> Self {
                let cfg = $krate::config::RuleConfig::new(map, n_players)
                    .unwrap_or_else(|e| panic!("{map} {n_players}P: {e}"));
                let game = $krate::state::Game::new(cfg)
                    .unwrap_or_else(|e| panic!("{map} {n_players}P: {e}"));
                Self { game }
            }

            pub fn game(&self) -> &$krate::state::Game {
                &self.game
            }
        }

        impl $crate::engine::Engine for $name {
            type State = $krate::state::State;

            fn n_players(&self) -> usize {
                usize::from(self.game.rules.n_players)
            }

            fn map_name(&self) -> &'static str {
                self.game.board.name
            }

            fn obs_size(&self) -> usize {
                $krate::obs::spec_for(self.game.board.name).size
            }

            fn initial(&self, seed: u64) -> Self::State {
                self.game.new_initial_state(seed)
            }

            fn step(&self, state: &mut Self::State, action: u16) {
                state
                    .step(action)
                    .unwrap_or_else(|e| panic!("action {action} was refused: {e}"));
            }

            fn legal(&self, state: &Self::State) -> Vec<u16> {
                state.legal_actions()
            }

            fn state_hash(&self, state: &Self::State) -> u64 {
                state.state_hash()
            }

            fn position_hash(&self, state: &Self::State) -> u64 {
                state.position_hash()
            }

            fn serialize(&self, state: &Self::State, canonical: bool) -> Vec<u8> {
                state.serialize(canonical)
            }

            fn is_terminal(&self, state: &Self::State) -> bool {
                state.is_terminal()
            }

            fn current_player(&self, state: &Self::State) -> usize {
                usize::from(state.current_player())
            }

            fn encode(&self, state: &mut Self::State, player: usize, out: &mut [f32]) {
                $krate::obs::encode(state, player, out);
            }

            fn terminal(&self, state: &mut Self::State) -> $crate::engine::Terminal {
                let breakdown = $krate::scoring::score_breakdown(state);
                $crate::engine::Terminal {
                    scores: breakdown.iter().map(|b| b.total).collect(),
                    breakdown: breakdown
                        .iter()
                        .map(|b| {
                            [
                                i32::from(b.routes),
                                i32::from(b.tickets_made),
                                i32::from(b.tickets_missed),
                                i32::from(b.ticket_points),
                                i32::from(b.longest_bonus),
                                i32::from(b.longest_trail),
                                i32::from(b.completed),
                                i32::from(b.total),
                            ]
                        })
                        .collect(),
                    returns: $krate::scoring::returns(state),
                    winners: $krate::scoring::winners(state),
                    trails: $krate::scoring::longest_trails(state),
                }
            }
        }
    };
}

engine_for!(
    /// The engine this workspace builds.
    Live,
    ttr_core
);
