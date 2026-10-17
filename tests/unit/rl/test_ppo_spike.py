"""The Phase 3.4 PPO spike runs end to end and never plays a masked move."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.torch


def test_a_tiny_run_trains_and_evaluates() -> None:
    pytest.importorskip("torch")
    from ticket_to_ride.rl.ppo_spike import Config, train  # noqa: PLC0415 -- needs torch

    result = train(Config(steps=2 * 16 * 8, n_envs=16, horizon=8, eval_games=8, device="cpu"))
    assert result["mask_violations"] == 0
    assert result["learner_steps"] == 256
    vs = result["vs"]
    assert isinstance(vs, dict)
    assert set(vs) == {"h0", "h1"}
