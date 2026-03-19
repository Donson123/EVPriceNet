from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from benchmark.models.base import TrainOutput


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Networks (DDPG-style)
# ---------------------------

class Actor(nn.Module):
    # Outputs normalized actions in [0,1] (Sigmoid), later scaled to [0,p_max]
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], act_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    # Q(s,a)
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


# ---------------------------
# Rank-based Prioritized Replay Buffer (PER)
# Paper uses rank-based priority and IS weights driven by TD error magnitude.
# ---------------------------

class RankBasedPER:
    """
    Stores (s, a, r, s2, done) and priorities based on |TD error|.
    Sampling probability uses rank-based priorities:
      p_n = 1 / rank_n
      P(n) ∝ p_n^beta
    plus importance-sampling weights Wn.
    """
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        beta: float = 0.6,
        is_exponent: float = 0.4,
        eps: float = 1e-6,
        seed: int = 0,
        refresh_ranks_every: int = 256,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.beta = float(beta)
        self.is_exponent = float(is_exponent)
        self.eps = float(eps)

        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)

        # priority proxy: abs(td_error) + eps
        self.prio = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

        # rank cache
        self._rank_idx_desc = None  # indices sorted by prio desc
        self._prob_cache = None
        self._refresh_counter = 0
        self.refresh_ranks_every = int(refresh_ranks_every)

    def add(self, s, a, r, s2, done, init_prio: float = 1.0) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = float(r)
        self.s2[i] = s2
        self.d[i] = float(done)

        self.prio[i] = float(max(self.eps, init_prio))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # invalidate caches cheaply
        self._refresh_counter += 1
        if self._refresh_counter >= self.refresh_ranks_every:
            self._rank_idx_desc = None
            self._prob_cache = None
            self._refresh_counter = 0

    def _ensure_rank_probs(self) -> None:
        if self._rank_idx_desc is not None and self._prob_cache is not None:
            return

        n = self.size
        pr = self.prio[:n]

        # sort by priority descending
        idx = np.argsort(-pr, kind="mergesort")
        self._rank_idx_desc = idx

        # rank: 1..n
        ranks = np.empty(n, dtype=np.int32)
        ranks[idx] = np.arange(1, n + 1, dtype=np.int32)

        p = (1.0 / ranks.astype(np.float32)) ** self.beta
        p = p / np.sum(p)
        self._prob_cache = p  # aligned with buffer index 0..n-1

    def sample(self, batch_size: int):
        self._ensure_rank_probs()
        n = self.size
        p = self._prob_cache  # shape (n,)

        idx = self.rng.choice(n, size=batch_size, replace=True, p=p)

        # importance sampling weights
        # w_i = (N * P(i))^(-is_exponent), normalized by max(w)
        w = (n * p[idx]) ** (-self.is_exponent)
        w = w / np.max(w)

        batch = (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
            idx,
            w.astype(np.float32),
        )
        return batch

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray) -> None:
        td_abs = np.asarray(td_abs, dtype=np.float32)
        self.prio[idx] = np.maximum(self.eps, td_abs)

        # invalidate caches
        self._rank_idx_desc = None
        self._prob_cache = None
        self._refresh_counter = 0


# ---------------------------
# PDDPG: DDPG + PER
# ---------------------------

@dataclass
class PDDPG:
    env_kwargs: Dict[str, Any]
    epochs: int

    # Paper-style defaults (adaptable)
    gamma: float = 0.7
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    critic_weight_decay: float = 1e-2  # L2 regularization for critic

    tau: float = 0.001  # soft update rate (paper uses very small tau) :contentReference[oaicite:3]{index=3}
    batch_size: int = 128

    # PER params
    per_beta: float = 0.6
    is_exponent: float = 0.4
    buffer_capacity: int = 50_000

    # exploration (Gaussian in normalized [0,1] space)
    noise_std: float = 0.10

    warmup_steps: int = 365
    grad_clip: float = 5.0

    seed: int = 42

    def __post_init__(self):
        self.day_steps = int(self.env_kwargs.get("day_steps", 24))
        self.p_max = float(self.env_kwargs.get("p_max", 1.0))

        # obs encoding: 4 + 3*24
        self.obs_dim = int(self.env_kwargs.get("obs_dim", 4 + 3 * self.day_steps))
        self.act_dim = int(self.env_kwargs.get("act_dim", self.day_steps))

        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic = Critic(self.obs_dim, self.act_dim).to(device)
        self.actor_t = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic_t = Critic(self.obs_dim, self.act_dim).to(device)

        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_c = optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            weight_decay=self.critic_weight_decay,
        )

        self.rb = RankBasedPER(
            capacity=self.buffer_capacity,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            beta=self.per_beta,
            is_exponent=self.is_exponent,
            seed=self.seed,
        )

        self.rng: Optional[np.random.Generator] = None
        self.total_steps = 0

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _day_to_obs(self, day) -> np.ndarray:
        # Must match your environment's day record fields.
        dow = float(day.dow)
        doy = float(day.day_of_year)

        sin_dow = np.sin(2.0 * np.pi * dow / 7.0)
        cos_dow = np.cos(2.0 * np.pi * dow / 7.0)
        sin_y = np.sin(2.0 * np.pi * doy / 365.0)
        cos_y = np.cos(2.0 * np.pi * doy / 365.0)

        obs = np.concatenate([
            np.array([sin_dow, cos_dow, sin_y, cos_y], dtype=np.float32),
            day.wholesale_24.astype(np.float32),
            day.expected_arrivals_24.astype(np.float32),
            day.congestion_24.astype(np.float32),
        ]).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}")
        return obs

    def act(self, day, *, eval_mode: bool = False) -> np.ndarray:
        obs = self._day_to_obs(day)
        x = torch.from_numpy(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            a01 = self.actor(x).squeeze(0).cpu().numpy().astype(np.float32)  # in [0,1]

        if not eval_mode:
            rng = self.rng if self.rng is not None else np.random.default_rng(0)
            a01 = a01 + rng.normal(0.0, self.noise_std, size=a01.shape).astype(np.float32)
            a01 = np.clip(a01, 0.0, 1.0)

        return (a01 * self.p_max).astype(np.float32)

    @torch.no_grad()
    def _soft_update(self, net: nn.Module, net_t: nn.Module) -> None:
        for p, pt in zip(net.parameters(), net_t.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def observe(self, day, action_24: np.ndarray, reward: float, next_day, done: bool) -> Dict[str, float]:
        """
        IMPORTANT: you must pass next_day (or next_obs) from the runner to make (s,a,r,s') transitions.
        """
        s = self._day_to_obs(day)
        s2 = self._day_to_obs(next_day) if next_day is not None else s.copy()

        a01 = np.clip(action_24.astype(np.float32) / self.p_max, 0.0, 1.0)
        r = float(reward)
        d = float(done)

        # init prio high so new transitions get sampled at least a few times
        self.rb.add(s, a01, r, s2, d, init_prio=1.0)

        self.total_steps += 1
        if self.rb.size < max(self.warmup_steps, self.batch_size):
            return {}

        # ----- PER sample -----
        bs = self.batch_size
        s_b, a_b, r_b, s2_b, d_b, idx, w_b = self.rb.sample(bs)

        s_t = torch.from_numpy(s_b).to(device)
        a_t = torch.from_numpy(a_b).to(device)
        r_t = torch.from_numpy(r_b).to(device)          # [B]
        s2_t = torch.from_numpy(s2_b).to(device)
        d_t = torch.from_numpy(d_b).to(device)          # [B]
        w_t = torch.from_numpy(w_b).to(device)          # [B]

        # ----- Critic TD target: y = r + gamma * Q'(s2, mu'(s2)) -----
        with torch.no_grad():
            a2 = self.actor_t(s2_t)
            q2 = self.critic_t(s2_t, a2).squeeze(1)     # [B]
            y = r_t + (1.0 - d_t) * self.gamma * q2     # [B]

        q = self.critic(s_t, a_t).squeeze(1)            # [B]
        td = y - q                                      # [B]
        td_abs = torch.abs(td).detach().cpu().numpy()

        # weighted critic loss (IS weights) as in PER formulation
        critic_loss = torch.mean(w_t * td.pow(2))

        self.opt_c.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_c.step()

        # ----- Actor update: maximize Q(s, mu(s)) -----
        a_pred = self.actor(s_t)
        actor_loss = -self.critic(s_t, a_pred).mean()

        self.opt_a.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_a.step()

        # ----- Update priorities from |TD error| -----
        self.rb.update_priorities(idx, td_abs)

        # ----- Soft update target nets -----
        with torch.no_grad():
            self._soft_update(self.actor, self.actor_t)
            self._soft_update(self.critic, self.critic_t)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mean_abs_td": float(np.mean(td_abs)),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic_t": self.critic_t.state_dict(),
            "total_steps": self.total_steps,
            # replay buffer persistence optional; usually omitted
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_t.load_state_dict(state["actor_t"])
        self.critic_t.load_state_dict(state["critic_t"])
        self.total_steps = int(state.get("total_steps", 0))

    def train(self, *args, **kwargs) -> TrainOutput:
        return TrainOutput(model_state=self.state_dict(), train_logs=[])
