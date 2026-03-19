from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from benchmark.models.base import TrainOutput  # mag blijven, maar train() is optioneel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], out_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQN:
    env_kwargs: Dict[str, Any]
    epochs: int  # runner gebruikt dit misschien; verder niet nodig

    # actie-discretisatie per uur
    n_levels: int = 101  # bv 0..p_max in 11 stapjes

    # learning
    lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    warmup: int = 365
    target_update: int = 500

    # epsilon-greedy (runner mag set_eps doen)
    eps_start: float = 0.25
    eps_end: float = 0.05

    # gamma
    gamma: float = 0.05

    def __post_init__(self):
        self.day_steps = int(self.env_kwargs.get("day_steps", 24))
        self.p_max = float(self.env_kwargs.get("p_max", 1.0))

        # obs: [sin_dow, cos_dow, sin_y, cos_y, wholesale[24], expected_arrivals[24], congestion[24]]
        self.obs_dim = 4 + 3 * self.day_steps

        # levels in [0, p_max]
        self.levels = np.linspace(0.0, self.p_max, self.n_levels, dtype=np.float32)

        out_dim = self.day_steps * self.n_levels  # per hour L Q-values
        self.online = MLP(self.obs_dim, out_dim).to(device)
        self.target = MLP(self.obs_dim, out_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # replay buffer: store (s, a_idx_vec[24], r)
        self.rb_s = np.zeros((self.replay_size, self.obs_dim), dtype=np.float32)
        self.rb_a = np.zeros((self.replay_size, self.day_steps), dtype=np.int64)
        self.rb_r = np.zeros((self.replay_size,), dtype=np.float32)
        self.rb_pos = 0
        self.rb_full = False

        self.step = 0
        self.eps = float(self.eps_start)
        self.rng: Optional[np.random.Generator] = None

    # --------- utils ---------
    def _size(self) -> int:
        return self.replay_size if self.rb_full else self.rb_pos

    def _push(self, s: np.ndarray, a_idx: np.ndarray, r: float) -> None:
        i = self.rb_pos
        self.rb_s[i] = s
        self.rb_a[i] = a_idx
        self.rb_r[i] = r
        self.rb_pos = (self.rb_pos + 1) % self.replay_size
        if self.rb_pos == 0:
            self.rb_full = True

    def _q_reshape(self, q_flat: torch.Tensor) -> torch.Tensor:
        # (B, T*L) -> (B, T, L)
        return q_flat.view(-1, self.day_steps, self.n_levels)

    def _indices_to_action(self, a_idx: np.ndarray) -> np.ndarray:
        return self.levels[a_idx].astype(np.float32, copy=True)

    def _action_to_indices(self, a24: np.ndarray) -> np.ndarray:
        # map continuous prices -> nearest discrete level per hour
        a = np.clip(a24.astype(np.float32), 0.0, self.p_max)
        # compute nearest level index
        # shape: (24, L) -> argmin over L
        diffs = np.abs(a[:, None] - self.levels[None, :])
        return np.argmin(diffs, axis=1).astype(np.int64)

    def _day_to_obs(self, day) -> np.ndarray:
        # bouw obs zonder env.reset: consistent met "4 + 3*24"
        # dow: 0..6
        dow = float(day.dow)
        # day_of_year: 1..365/366 (maakt weinig uit, we nemen 365 schaal)
        doy = float(day.day_of_year)

        sin_dow = np.sin(2.0 * np.pi * dow / 7.0)
        cos_dow = np.cos(2.0 * np.pi * dow / 7.0)
        sin_y = np.sin(2.0 * np.pi * doy / 365.0)
        cos_y = np.cos(2.0 * np.pi * doy / 365.0)

        wholesale = day.wholesale_24.astype(np.float32)
        expected_arrivals = day.expected_arrivals_24.astype(np.float32)
        congestion = day.congestion_24.astype(np.float32)

        obs = np.concatenate([
            np.array([sin_dow, cos_dow, sin_y, cos_y], dtype=np.float32),
            wholesale, expected_arrivals, congestion
        ]).astype(np.float32)

        # safety
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}")
        return obs

    # --------- runner interface ---------
    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def set_eps(self, eps: float) -> None:
        self.eps = float(eps)

    def load_state(self, state: Any) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])

    def state_dict(self) -> Dict[str, Any]:
        return {"online": self.online.state_dict(), "target": self.target.state_dict()}

    def act(self, day, *, eval_mode: bool = False) -> np.ndarray:
        obs0 = self._day_to_obs(day)

        # epsilon-greedy per uur (alleen tijdens train)
        rng = self.rng if self.rng is not None else np.random.default_rng(0)
        if (not eval_mode) and (rng.random() < self.eps):
            a_idx = rng.integers(0, self.n_levels, size=self.day_steps, dtype=np.int64)
            return self._indices_to_action(a_idx)

        x = torch.from_numpy(obs0).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self._q_reshape(self.online(x)).squeeze(0)  # (T,L)
        a_idx = torch.argmax(q, dim=-1).to(torch.int64).cpu().numpy()
        return self._indices_to_action(a_idx)

    def observe(self, day, action_24: np.ndarray, reward: float) -> None:
        # store transition (s, a, r) and do one gradient step if possible
        obs0 = self._day_to_obs(day)
        a_idx = self._action_to_indices(action_24)
        R = float(reward)

        self._push(obs0, a_idx, R)

        if self._size() < max(self.warmup, self.batch_size):
            return

        rng = self.rng if self.rng is not None else np.random.default_rng(0)
        idx = rng.integers(0, self._size(), size=self.batch_size)

        s = torch.from_numpy(self.rb_s[idx]).to(device)              # (B,obs)
        a = torch.from_numpy(self.rb_a[idx]).long().to(device)       # (B,T)
        r = torch.from_numpy(self.rb_r[idx]).to(device)              # (B,)

        q_all = self._q_reshape(self.online(s))                      # (B,T,L)
        q_chosen = q_all.gather(2, a.unsqueeze(-1)).squeeze(-1)      # (B,T)
        q_sa = q_chosen.sum(dim=1)                                   # (B,)

        # terminal bandit target
        y = r

        loss = self.loss_fn(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.opt.step()

        if (self.step % self.target_update) == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.step += 1

    # optioneel: als jouw runner dit verwacht
    def train(self, *args, **kwargs) -> TrainOutput:
        # Sommige benchmark frameworks verwachten train() te bestaan.
        # Deze DQN leert via observe(); train() kan dan leeg zijn.
        return TrainOutput(model_state=self.state_dict(), train_logs=[])
