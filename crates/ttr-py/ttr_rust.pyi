"""Type stubs for the Rust core.

A compiled extension carries no type information, so without this every consumer --
`ttr bench`, the arena, and everything Phases 5-8 build on top -- is untyped from `ty`'s
point of view, and a typo in a method name is a runtime failure instead of a build
failure.

Kept next to `Cargo.toml` and shipped by maturin. It is hand-written, so it can drift from
`src/lib.rs`; `tests/unit/rust/test_stub_matches_module.py` asserts the two agree.
"""

from collections.abc import Buffer
from typing import Any

CONTRACT_VERSION: int

class IllegalAction(Exception):  # noqa: N818 - mirrors engine.state.IllegalAction
    """An action that is not legal in the current state."""

def contract_version() -> int: ...
def obs_version() -> int: ...
def obs_spec(
    map_name: str = "usa",
) -> list[tuple[str, int, int, int, list[tuple[str, int, int, str, str]]]]:
    """The observation layout: `(block, count, stride, offset, fields)` per block, each field
    `(name, offset, width, kind, doc)` with `kind` in `"flag"` / `"ratio"`."""

def performance_threads() -> int: ...
def default_params() -> dict[str, float | int | bool]:
    """The Elo anchor's heuristic constants. The one place Python reads them from."""

def tier_names() -> list[str]:
    """Every heuristic tier's name, weakest first."""

def params_hash(tier: str, overrides: dict[str, float | int | bool] | None = None) -> str:
    """blake2b-128 over the parameters `tier` reads. Provenance -- see `behaviour_hash` for
    identity. A field added for one tier never reaches another tier's hash."""

def behaviour_hash(tier: str, overrides: dict[str, float | int | bool] | None = None) -> str:
    """A content address for an agent: what it plays over a frozen probe set.

    This is what pins the Elo anchor. A params hash catches an edited constant and misses a
    changed tiebreak or a fixed bug; both move H3's play and would silently re-base every
    rating recorded against it.
    """

def run_arena(
    map_name: str,
    n_players: int,
    agents: list[tuple[str, dict[str, float | int | bool] | None, int]],
    lineup: list[int],
    seed_root: int = 0,
    blocks: int = 100,
    threads: int = 0,
) -> tuple[ArenaGames, ArenaSeats]:
    """Play every rotation of `blocks` seed blocks. One call for the whole schedule."""

def random_playouts(
    map_name: str,
    n_players: int,
    games: int,
    base_seed: int = 0,
    threads: int = 1,
) -> tuple[int, int, float]:
    """Play `games` random games entirely inside Rust. Returns `(games, steps, seconds)`."""

class Rng:
    """A PCG-XSH-RR 64/32 stream. Hold one and advance it across determinizations rather
    than reseeding per call, which would make consecutive samples correlated."""

    def __init__(self, state: int, inc: int) -> None: ...
    @staticmethod
    def seeded(initstate: int, initseq: int) -> Rng: ...
    @staticmethod
    def stream(root: int, *parts: str | int) -> Rng: ...
    def next_u32(self) -> int: ...
    def below(self, bound: int) -> int: ...
    @property
    def state(self) -> int: ...
    @property
    def inc(self) -> int: ...

class Game:
    def __init__(
        self, map_name: str = ..., n_players: int = ..., *, track_history: bool = ...
    ) -> None: ...
    @property
    def num_distinct_actions(self) -> int: ...
    @property
    def n_players(self) -> int: ...
    @property
    def map_name(self) -> str: ...
    @property
    def data_hash(self) -> str: ...
    @property
    def rules_hash(self) -> str: ...
    def new_initial_state(self, seed: int) -> State: ...
    def action_to_string(self, action: int) -> str: ...
    def decode_action(self, action: int) -> DecodedAction:
        """What an action id means, from the one place the layout lives."""

class DecodedAction:
    """An action id decoded. Fields that do not apply to `kind` are `None` (or empty)."""

    #: `"claim"`, `"draw"`, `"draw_tickets"`, `"keep"` or `"pass"`.
    kind: str
    segment: int | None
    #: The card type paid with: a colour, or the locomotive for an all-locomotive claim.
    pay: int | None
    #: A face-up draw's slot, 0-4; `None` for a blind draw.
    slot: int | None
    blind: bool
    #: A keep's mask, 1-7.
    mask: int | None
    #: A keep's offer positions, ascending: the bits of `mask`.
    offer_indices: list[int]

class State:
    def step(self, action: int) -> None: ...
    def legal_actions(self) -> list[int]: ...
    def legal_action_mask(self) -> bytes: ...
    def state_hash(self) -> int: ...
    def position_hash(self) -> int: ...
    def serialize(self, canonical: bool = False) -> bytes: ...
    def is_terminal(self) -> bool: ...
    def current_player(self) -> int: ...
    def clone(self) -> State: ...
    def clone_into(self, dst: State) -> None: ...
    def validate(self) -> None: ...
    def history(self) -> list[int]: ...
    # These five, and the `trains` and `seg_owner` properties, come back as `bytes`: PyO3
    # hands any `Vec<u8>` over that way. Indexing and iteration give ints.
    def hand_of(self, player: int) -> bytes: ...
    def certain_of(self, player: int) -> bytes: ...
    def tickets_of(self, player: int) -> bytes: ...
    def deck_counts(self) -> bytes: ...
    def unseen_counts(self, observer: int) -> list[int]: ...
    def offer(self) -> list[int]:
        """The tickets on offer during a keep, in offer order: KEEP bit `i` is `offer()[i]`."""

    def discard(self) -> list[int]:
        """The discard pile's composition, by card type."""

    def ticket_progress(self, seat: int) -> list[tuple[int, bool, int | None]]:
        """`(ticket, connected, remaining train cars or None if unreachable)` per ticket held."""

    def claim_payment(self, action: int) -> tuple[int, int]:
        """`(coloured cards, locomotives)` the current player would pay for claim `action`."""

    def final_scores(self) -> list[int]: ...
    def score_breakdown(self) -> list[tuple[int, int, int, int, int, int, int, int]]: ...
    def returns(self) -> list[float]: ...
    def winners(self) -> list[int]: ...
    def ranks(self) -> list[int]:
        """Competition ranks under the full tiebreak chain, 1 for first."""

    def longest_trails(self) -> list[int]: ...
    def resample_from_infoset(self, observer: int, rng: Rng) -> State: ...
    def assert_consistent(self, public: State, observer: int) -> None: ...
    def observation(self, player: int) -> list[float]: ...
    @property
    def observation_size(self) -> int: ...
    @property
    def phase(self) -> int: ...
    @property
    def phase_name(self) -> str: ...
    @property
    def turn(self) -> int: ...
    @property
    def cur(self) -> int: ...
    @property
    def draws_left(self) -> int: ...
    @property
    def final_left(self) -> int: ...
    @property
    def pass_streak(self) -> int: ...
    @property
    def trains(self) -> bytes: ...
    @property
    def score(self) -> list[int]: ...
    @property
    def faceup(self) -> list[int]: ...
    @property
    def seg_owner(self) -> bytes: ...
    @property
    def deck_pos(self) -> int: ...
    @property
    def tickets_remaining(self) -> int: ...
    @property
    def seed(self) -> int: ...

class Heuristic:
    """A scripted agent, H0-H4, deciding one position at a time.

    `explain` is `act` with its reasoning recorded -- one scorer, a trace sink threaded
    through it -- so it returns the action `act` would and leaves the agent as `act` would.
    Call one or the other for a decision, never both: H0 would draw twice.
    """

    def __init__(
        self, tier: str, params: dict[str, float | int | bool] | None = None, seed: int = 0
    ) -> None: ...
    @property
    def tier(self) -> str: ...
    def begin_game(self, seat: int, game_id: int) -> None: ...
    def act(self, state: State) -> int: ...
    def explain(self, state: State, top_k: int = 8) -> Explanation: ...

class Explanation:
    """One decision, explained. `candidates` are best first; for H1-H4 the first is `chosen`."""

    tier: str
    seat: int
    phase: str
    chosen: int
    #: The chosen candidate's rule: H1's cascade step, "forced", "uniform" (H0).
    rule: str | None
    context: list[tuple[str, float]]
    candidates: list[Candidate]

class Candidate:
    """One scored action: its total (`-inf` when ruled out), additive terms in points that
    sum to it, and the inputs behind them."""

    action: int
    total: float
    rule: str | None
    terms: list[tuple[str, float]]
    facts: list[tuple[str, float]]

class ArenaGames:
    """One row per finished game, columnar."""

    block_seed: list[int]
    rotation: list[int]
    turns: list[int]
    #: Decisions across every seat; the denominator for microseconds-per-decision.
    decisions: list[int]
    #: The final `state_hash`, so a re-run is verifiable rather than merely repeatable.
    final_hash: list[int]
    seconds: float

class ArenaSeats:
    """One row per (game, seat), columnar. `game` indexes into `ArenaGames`."""

    game: list[int]
    seat: list[int]
    #: Index into the arena's agent list -- **not** the seat.
    agent: list[int]
    score: list[int]
    ret: list[float]
    rank: list[int]
    won: list[int]
    tickets_kept: list[int]
    tickets_made: list[int]
    ticket_points: list[int]
    routes_claimed: list[int]
    trains_left: list[int]
    cards_left: list[int]
    longest_trail: list[int]
    longest_bonus: list[int]
    n_claim: list[int]
    n_claim_wild: list[int]
    n_claim_double: list[int]
    n_draw_faceup: list[int]
    n_draw_blind: list[int]
    n_draw_tickets: list[int]
    n_keep: list[int]
    n_keep_extra: list[int]
    n_pass: list[int]

class VecEnv:
    def __init__(
        self,
        map_name: str = ...,
        n_players: int = ...,
        n_envs: int = ...,
        base_seed: int = ...,
        *,
        track_history: bool = ...,
    ) -> None: ...
    @property
    def n_envs(self) -> int: ...
    @property
    def obs_size(self) -> int: ...
    @property
    def n_actions(self) -> int: ...
    @property
    def seeds(self) -> list[int]: ...
    def step(self, actions: list[int]) -> None: ...
    def auto_reset(self) -> int: ...
    def observe_current(self, out: Buffer) -> None:
        """Fill `out` (`n_envs * obs_size` float32s) in place. Nothing is copied."""

    def observe_seat(self, player: int, out: Buffer) -> None: ...
    def legal_masks(self, out: Buffer) -> None:
        """Fill `out` (`n_envs * n_actions` bytes) in place. Nothing is copied."""

    def current_players(self) -> bytes:
        """One byte per environment (PyO3 hands a `Vec<u8>` back as `bytes`)."""

    def terminal_flags(self) -> bytes:
        """One byte per environment, 1 if the game is over."""

    def state_hashes(self) -> list[int]: ...
    @property
    def n_players(self) -> int: ...
    @property
    def n_tickets(self) -> int: ...
    def assign(self, env: int, seats: list[str]) -> None:
        """Seats to "learner" or "h0".."h4"; scripted seats then move inside Rust."""

    def returns(self, out: Buffer) -> None:
        """Fill `out` (`n_envs * n_players` float32s); finished games only, else zero."""

    def final_scores(self) -> list[int]: ...
    def ticket_status(self, out: Buffer) -> None:
        """Fill `out` (`n_envs * n_players * n_tickets` bytes): 0 none, 1 open, 2 done."""

    def finished_games(self) -> list[tuple[int, int, list[int]]]: ...

def board_names() -> list[str]: ...
def board_summary(map_name: str = ...) -> dict[str, Any]:
    """A board's data and derived invariants, as `ttr map` prints them."""

def board_data(map_name: str = ...) -> dict[str, Any]:
    """A board as the engine plays it: its own arrays, in its own order. The web client's
    only source of board data (mini's TOML is not in engine order)."""
