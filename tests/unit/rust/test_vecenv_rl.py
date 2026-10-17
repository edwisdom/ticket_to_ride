"""The `VecEnv` additions Phase 3.4 needs: scripted seats, returns, aux targets, records."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import ttr_rust


def finish(env: ttr_rust.VecEnv, rng: np.random.Generator) -> None:
    mask = np.zeros((env.n_envs, env.n_actions), dtype=np.uint8)
    while not all(env.terminal_flags()):
        env.legal_masks(mask)
        env.step([int(rng.choice(np.flatnonzero(row))) if row.any() else 0 for row in mask])


def test_python_only_sees_the_learner_and_reads_results_before_reset(rust: ModuleType) -> None:
    env = rust.VecEnv("mini", 2, 4, 0, track_history=True)
    for i in range(4):
        env.assign(i, ["learner", "h1"] if i % 2 == 0 else ["h0", "learner"])
    seats = np.arange(4) % 2
    assert all(
        p == s
        for p, s, done in zip(env.current_players(), seats, env.terminal_flags(), strict=True)
        if not done
    )
    finish(env, np.random.default_rng(0))

    returns = np.zeros((4, 2), dtype=np.float32)
    env.returns(returns)
    assert set(np.abs(returns).ravel()) <= {0.0, 1.0}
    assert np.all(returns.sum(axis=1) == 0)
    scores = np.array(env.final_scores()).reshape(4, 2)
    assert np.all((scores[:, 0] > scores[:, 1]) == (returns[:, 0] > returns[:, 1]))

    status = np.zeros((4, 2, env.n_tickets), dtype=np.uint8)
    env.ticket_status(status)
    assert set(np.unique(status)) <= {0, 1, 2}
    assert np.all((status > 0).sum(axis=2) >= 1)  # everyone keeps a ticket

    records = env.finished_games()
    assert [r[0] for r in records] == [0, 1, 2, 3]
    assert all(seed == s for (_, seed, _), s in zip(records, env.seeds, strict=True))
    assert all(len(actions) > 0 for _, _, actions in records)

    env.auto_reset()
    env.returns(returns)
    assert not returns.any()


def test_a_bad_assignment_is_refused(rust: ModuleType) -> None:
    env = rust.VecEnv("mini", 2, 1, 0)
    with pytest.raises(ValueError, match="unknown seat"):
        env.assign(0, ["learner", "h9"])
    with pytest.raises(ValueError, match="2-player"):
        env.assign(0, ["learner"])
