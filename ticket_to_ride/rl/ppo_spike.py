"""Phase 3.4's PPO spike on TTR-mini 2P: throwaway by design (PLAN.md §14.1).

Does PPO learn on this observation, and what does the whole loop sustain? A small MLP with
a masked categorical policy and a value head, trained against scripted opponents that move
inside Rust (`VecEnv.assign`), so every step Python sees is a learner decision. One learner
per environment, so per-seat GAE is per-environment GAE; the reward is the learner's
terminal return. Phase 6 replaces all of it; the `VecEnv` additions it uses are kept.

    uv run --extra rl python -m ticket_to_ride.rl.ppo_spike --steps 2000000

Learner seats alternate across environments, so both seatings are trained and measured.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import ttr_rust
from torch import nn

from ticket_to_ride.config.device import resolve_device


@dataclass
class Config:
    map: str = "mini"
    steps: int = 2_000_000
    n_envs: int = 256
    horizon: int = 64
    opponents: tuple[str, ...] = ("h0", "h1")
    hidden: int = 256
    lr: float = 3e-4
    epochs: int = 2
    minibatches: int = 4
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    eval_games: int = 400
    seed: int = 0
    device: str = "auto"


class Policy(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.logits = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        logits = self.logits(h).masked_fill(~mask, -1e9)
        return logits, self.value(h).squeeze(-1)


def make_env(
    n_envs: int, opponents: tuple[str, ...], seed: int, map_name: str = "mini"
) -> ttr_rust.VecEnv:
    """Learner in seat `i % 2`, opponent cycling through `opponents` by env."""
    env = ttr_rust.VecEnv(map_name, 2, n_envs, seed)
    for i in range(n_envs):
        opponent = opponents[(i // 2) % len(opponents)]
        env.assign(i, ["learner", opponent] if i % 2 == 0 else [opponent, "learner"])
    return env


def terminal(env: ttr_rust.VecEnv) -> np.ndarray:
    """`terminal_flags()` is `bytes`; `np.array(bytes, dtype=bool)` would be one scalar."""
    return np.frombuffer(env.terminal_flags(), dtype=np.uint8).astype(bool)


def sample(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-max over masked logits. `Categorical.sample` on MPS returned masked actions
    (an illegal move, measured); argmax never picks a -1e9 logit, on any device."""
    noise = torch.rand_like(logits).clamp_(1e-12, 1.0 - 1e-7)
    return (logits - torch.log(-torch.log(noise))).argmax(-1)


def learner_seats(env: ttr_rust.VecEnv) -> np.ndarray:
    return np.arange(env.n_envs) % 2


@torch.no_grad()
def evaluate(policy: Policy, cfg: Config, opponent: str, device: str) -> float:
    """Score per game against one opponent, both seatings: win 1, draw 0.5, loss 0."""
    n = cfg.eval_games
    env = make_env(n, (opponent,), 10_000_000 + cfg.seed, cfg.map)
    obs = np.zeros((n, env.obs_size), dtype=np.float32)
    mask = np.zeros((n, env.n_actions), dtype=np.uint8)
    returns = np.zeros((n, 2), dtype=np.float32)
    seat = learner_seats(env)
    done = np.zeros(n, dtype=bool)
    score = np.zeros(n)
    while not done.all():
        env.observe_current(obs)
        env.legal_masks(mask)
        logits, _ = policy(
            torch.from_numpy(obs).to(device), torch.from_numpy(mask).bool().to(device)
        )
        actions = logits.argmax(-1).cpu().numpy()
        env.step(actions.tolist())
        flags = terminal(env)
        env.returns(returns)
        newly = flags & ~done
        score[newly] = (returns[np.arange(n), seat][newly] + 1) / 2
        done |= flags
    return float(score.mean())


@torch.no_grad()
def bootstrap(policy: Policy, env: ttr_rust.VecEnv, device: str) -> np.ndarray:
    """The value of each environment's current position, closing an unfinished rollout."""
    obs = np.zeros((env.n_envs, env.obs_size), dtype=np.float32)
    mask = np.zeros((env.n_envs, env.n_actions), dtype=np.uint8)
    env.observe_current(obs)
    env.legal_masks(mask)
    _, value = policy(torch.from_numpy(obs).to(device), torch.from_numpy(mask).bool().to(device))
    return value.cpu().numpy()


def advantages(
    cfg: Config,
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
) -> np.ndarray:
    """GAE with gamma = 1: the only reward is the terminal return."""
    t = values.shape[0]
    adv = np.zeros_like(values)
    last = np.zeros_like(next_value)
    for k in reversed(range(t)):
        following = next_value if k == t - 1 else values[k + 1]
        live = 1.0 - dones[k]
        delta = rewards[k] + live * following - values[k]
        last = delta + cfg.gae_lambda * live * last
        adv[k] = last
    return adv


def update(
    policy: Policy,
    optim: torch.optim.Optimizer,
    *,
    cfg: Config,
    rng: np.random.Generator,
    device: str,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    adv: np.ndarray,
    values: np.ndarray,
) -> None:
    """Clipped PPO over one rollout: `epochs` passes of `minibatches` shuffled slices."""
    obs, masks, acts, logps = batch
    flat = adv.size
    b_obs = torch.from_numpy(obs.reshape(flat, -1)).to(device)
    b_mask = torch.from_numpy(masks.reshape(flat, -1)).bool().to(device)
    b_act = torch.from_numpy(acts.reshape(flat)).to(device)
    b_logp = torch.from_numpy(logps.reshape(flat)).to(device)
    b_ret = torch.from_numpy((adv + values).reshape(flat)).to(device)
    b_adv = torch.from_numpy(adv.reshape(flat)).to(device)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    size = flat // cfg.minibatches
    for _ in range(cfg.epochs):
        order = torch.from_numpy(rng.permutation(flat)).to(device)
        for start in range(0, flat, size):
            idx = order[start : start + size]
            logits, value = policy(b_obs[idx], b_mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
            pg = -torch.min(
                ratio * b_adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * b_adv[idx]
            ).mean()
            loss = (
                pg
                + cfg.value_coef * (value - b_ret[idx]).pow(2).mean()
                - cfg.entropy * dist.entropy().mean()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()


def train(cfg: Config) -> dict[str, object]:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = make_env(cfg.n_envs, cfg.opponents, cfg.seed, cfg.map)
    n, t = cfg.n_envs, cfg.horizon
    policy = Policy(env.obs_size, env.n_actions, cfg.hidden).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    seat = learner_seats(env)

    obs_buf = np.zeros((t, n, env.obs_size), dtype=np.float32)
    mask_buf = np.zeros((t, n, env.n_actions), dtype=np.uint8)
    act_buf = np.zeros((t, n), dtype=np.int64)
    logp_buf = np.zeros((t, n), dtype=np.float32)
    val_buf = np.zeros((t, n), dtype=np.float32)
    rew_buf = np.zeros((t, n), dtype=np.float32)
    done_buf = np.zeros((t, n), dtype=np.float32)
    returns = np.zeros((n, 2), dtype=np.float32)

    steps, games, mask_violations, history = 0, 0, 0, []
    # Where the time goes, in seconds: the engine (observe, mask, step including scripted
    # replies), the network's forward pass and copies, the rest of the rollout loop
    # (rewards, resets), and the PPO update.
    timing = dict.fromkeys(("env", "network", "bookkeeping", "update"), 0.0)
    started = time.perf_counter()
    while steps < cfg.steps:
        with torch.no_grad():
            for k in range(t):
                tick = time.perf_counter()
                env.observe_current(obs_buf[k])
                env.legal_masks(mask_buf[k])
                tock = time.perf_counter()
                timing["env"] += tock - tick
                obs_t = torch.from_numpy(obs_buf[k]).to(device)
                mask_t = torch.from_numpy(mask_buf[k]).bool().to(device)
                logits, value = policy(obs_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = sample(logits)
                act_buf[k] = action.cpu().numpy()
                logp_buf[k] = dist.log_prob(action).cpu().numpy()
                val_buf[k] = value.cpu().numpy()
                tick = time.perf_counter()
                timing["network"] += tick - tock
                mask_violations += int((mask_buf[k][np.arange(n), act_buf[k]] == 0).sum())
                actions = act_buf[k].tolist()
                tock = time.perf_counter()
                env.step(actions)
                tick = time.perf_counter()
                timing["env"] += tick - tock
                flags = terminal(env)
                env.returns(returns)
                rew_buf[k] = np.where(flags, returns[np.arange(n), seat], 0.0)
                done_buf[k] = flags
                games += int(flags.sum())
                env.auto_reset()
                timing["bookkeeping"] += time.perf_counter() - tick

        tick = time.perf_counter()
        update(
            policy,
            optim,
            cfg=cfg,
            rng=rng,
            device=device,
            batch=(obs_buf, mask_buf, act_buf, logp_buf),
            adv=advantages(cfg, val_buf, rew_buf, done_buf, bootstrap(policy, env, device)),
            values=val_buf,
        )
        timing["update"] += time.perf_counter() - tick
        steps += t * n
        elapsed = time.perf_counter() - started
        history.append({"steps": steps, "games": games, "seconds": round(elapsed, 1)})
        print(
            f"{steps:>9,} learner steps  {games:>7,} games  {steps / elapsed:>8,.0f} steps/s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    policy.eval()
    return {
        "config": asdict(cfg),
        "device": device,
        "learner_steps": steps,
        "games": games,
        "seconds": round(elapsed, 1),
        "steps_per_second": round(steps / elapsed),
        "mask_violations": mask_violations,
        "us_per_step": {k: round(v / steps * 1e6, 2) for k, v in timing.items()},
        "vs": {opponent: evaluate(policy, cfg, opponent, device) for opponent in ("h0", "h1")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    for name, value in asdict(Config()).items():
        if isinstance(value, tuple):
            parser.add_argument(f"--{name}", default=",".join(value))
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    args = vars(parser.parse_args())
    args["opponents"] = tuple(str(args["opponents"]).split(","))
    print(json.dumps(train(Config(**args)), indent=2))


if __name__ == "__main__":
    main()
