# Reproducibility

Three levels, stated honestly, because **pretending L2 is L1 costs a day chasing ghosts**.
Every run records which level it achieved.

| Level | Promise | Scope | Enforced by |
| --- | --- | --- | --- |
| **L0 — Replay** | Bitwise, always | A recorded game replays to the same `state_hash()` and the same scores | [`engine/replay.py`](../ticket_to_ride/engine/replay.py), the 84-game corpus in `tests/golden/replays.bin`, and `tests/integration/test_golden_replays.py` |
| **L1 — Rerun** | Bitwise, given the same seed + `uv.lock` + git sha | Engine, heuristics and (from Phase 5) search | `test_the_same_seed_replays_bitwise` runs a seed twice and diffs the action logs |
| **L2 — Statistical** | Same distribution, same curve within noise | Neural-network training | Not bitwise, and not claimed to be |

## Why L2 is not L1

Neural-network training on MPS is not bitwise reproducible across torch versions, and often
not across runs on the same version: reduction orders differ, and `PYTORCH_ENABLE_MPS_FALLBACK`
can silently move an op to CPU. The honest claim is "same distribution, same curve within
noise", and the honest check is a CPU-versus-MPS comparison of one training iteration at the
same seed to 1e-3 — not a hash.

## Seeding discipline

One `seed` in the config; every other stream **derived, never drawn**:

```
derive(root, *parts) = blake2b-128(root ‖ 0x1f ‖ part ‖ 0x1f ‖ part ‖ …) → (initstate, initseq)
```

The exact encoding is frozen in [CONTRACT.md §1.3](CONTRACT.md). Four non-negotiable rules:

1. **The engine never touches global `random` or `np.random`.** This is the precondition
   that makes mirrored paired evaluation possible at all.
2. **Environment randomness is fully materialized at game construction.** The deck is a
   shuffled permutation, not a lazy sample from card counts. If it were sampled lazily, an
   agent that drew three cards and one that drew five would realize *different shuffles*
   from the same seed, and the entire variance-reduction scheme would evaporate with no
   error anywhere. `test_environment_randomness_does_not_depend_on_agent_behaviour` pins it.
3. **Agent stochasticity is a separate per-seat stream**, `("agent", seat, game_id)`, so
   draining one agent's randomness cannot move the environment's.
4. **Worker streams come from a spawn**, so results do not depend on scheduling order.

Named streams currently in use:

| Stream | Purpose |
| --- | --- |
| `("env", "deck")` | the train-car deck permutation |
| `("env", "tickets")` | the ticket deck permutation |
| `("env", "reshuffle")` | seeds `State.rng`; advanced by reshuffles and nothing else |
| `("agent", seat, game_id)` | one agent's decisions in one game |

## What invalidates a replay

By design, loudly rather than silently:

| Change | Detected by |
| --- | --- |
| The board | `DATA_HASH` mismatch |
| The rules (`RuleConfig`) | `rules_hash` mismatch |
| The PRNG, the draw procedure, or the state serialization | `CONTRACT_VERSION` mismatch |
| Anything else that changes play | the final `state_hash()` |
| A scoring bug that leaves the position intact | the final scores, stored separately |

## Not yet written

Run directories, manifests and checkpoint version gating (`OBS_VERSION`,
`ACTION_SPACE_VERSION`) arrive with Phase 6; see PLAN.md §10.
