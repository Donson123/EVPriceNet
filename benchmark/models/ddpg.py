from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], act_dim),
            nn.Sigmoid(),  # normalized actions in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


@dataclass
class DDPG:
    env_kwargs: Dict[str, Any]
    epochs: int

    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    warmup: int = 365
    tau: float = 0.005

    # exploration noise in normalized [0,1] action space
    noise_std: float = 0.10

    seed: int = 42

    def __post_init__(self):
        self.day_steps = int(self.env_kwargs.get("day_steps", 24))
        self.p_max = float(self.env_kwargs.get("p_max", 1.0))
        self.obs_dim = 4 + 3 * self.day_steps
        self.act_dim = self.day_steps

        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic = Critic(self.obs_dim, self.act_dim).to(device)
        self.actor_t = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic_t = Critic(self.obs_dim, self.act_dim).to(device)

        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Replay stores (s, a_norm, R)
        self.rb_s = np.zeros((self.replay_size, self.obs_dim), dtype=np.float32)
        self.rb_a = np.zeros((self.replay_size, self.act_dim), dtype=np.float32)
        self.rb_r = np.zeros((self.replay_size,), dtype=np.float32)
        self.rb_pos = 0
        self.rb_full = False

        self.rng: Optional[np.random.Generator] = None

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _size(self) -> int:
        return self.replay_size if self.rb_full else self.rb_pos

    def _push(self, s: np.ndarray, a01: np.ndarray, r: float) -> None:
        i = self.rb_pos
        self.rb_s[i] = s
        self.rb_a[i] = a01
        self.rb_r[i] = r
        self.rb_pos = (self.rb_pos + 1) % self.replay_size
        if self.rb_pos == 0:
            self.rb_full = True

    def _day_to_obs(self, day) -> np.ndarray:
        # Same encoding as your env observation design
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
        obs0 = self._day_to_obs(day)
        x = torch.from_numpy(obs0).unsqueeze(0).to(device)

        with torch.no_grad():
            a01 = self.actor(x).squeeze(0).cpu().numpy().astype(np.float32)  # [0,1]

        if not eval_mode:
            rng = self.rng if self.rng is not None else np.random.default_rng(0)
            a01 = a01 + rng.normal(0.0, self.noise_std, size=a01.shape).astype(np.float32)
            a01 = np.clip(a01, 0.0, 1.0)

        return (a01 * self.p_max).astype(np.float32)

    def observe(self, day, action_24: np.ndarray, reward: float) -> Dict[str, float]:
        obs0 = self._day_to_obs(day)
        a01 = np.clip(action_24.astype(np.float32) / self.p_max, 0.0, 1.0)
        R = float(reward)

        self._push(obs0, a01, R)

        if self._size() < max(self.warmup, self.batch_size):
            return {}

        rng = self.rng if self.rng is not None else np.random.default_rng(0)
        idx = rng.integers(0, self._size(), size=self.batch_size)

        s = torch.from_numpy(self.rb_s[idx]).to(device)
        a = torch.from_numpy(self.rb_a[idx]).to(device)
        r = torch.from_numpy(self.rb_r[idx]).to(device)  # shape [B]

        # ===== Critic update: regress Q(s,a) -> R_day =====
        q = self.critic(s, a).squeeze(1)                 # [B]
        y = r                                            # [B]
        loss_c = (q - y).pow(2).mean()

        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_c.step()

        # ===== Actor update: maximize Q(s, pi(s)) =====
        a_pred = self.actor(s)
        loss_a = -self.critic(s, a_pred).mean()

        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_a.step()

        # ===== Soft target updates =====
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {"critic_loss": float(loss_c.item()), "actor_loss": float(loss_a.item())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic_t": self.critic_t.state_dict(),
            "rb_pos": self.rb_pos,
            "rb_full": self.rb_full,
            "rb_s": self.rb_s,
            "rb_a": self.rb_a,
            "rb_r": self.rb_r,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_t.load_state_dict(state["actor_t"])
        self.critic_t.load_state_dict(state["critic_t"])
        self.rb_pos = int(state.get("rb_pos", 0))
        self.rb_full = bool(state.get("rb_full", False))
        # Optional: restore replay if you persist it
        if "rb_s" in state: self.rb_s = state["rb_s"]
        if "rb_a" in state: self.rb_a = state["rb_a"]
        if "rb_r" in state: self.rb_r = state["rb_r"]
