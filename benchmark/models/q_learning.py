from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from benchmark.models.base import TrainOutput  # mag blijven


@dataclass
class QLearning:
    env_kwargs: Dict[str, Any]
    epochs: int

    alpha_q: float = 0.25
    eps_start: float = 0.25
    eps_end: float = 0.05

    obs_round: int = 2
    act_round: int = 2
    max_actions_per_state: int = 200

    def __post_init__(self):
        self.Q: Dict[bytes, Dict[bytes, float]] = {}
        self.best_action: Dict[bytes, np.ndarray] = {}

        self.p_max: float = float(self.env_kwargs.get("p_max", 1.0))
        self.eps: float = float(self.eps_start)
        self.rng: Optional[np.random.Generator] = None

    # ----- runner hooks -----
    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def set_eps(self, eps: float) -> None:
        self.eps = float(eps)

    def load_state(self, state: Any) -> None:
        self.Q = state.get("Q", {})
        self.best_action = state.get("best_action", {})

    def state_dict(self) -> Dict[str, Any]:
        return {"Q": self.Q, "best_action": self.best_action}

    # ----- hashing -----
    def _obs_key_from_day(self, day) -> bytes:
        # Runner geeft geen env-obs mee, dus we bouwen een deterministische key uit day.
        x = np.concatenate([
            np.array([float(day.dow), float(day.day_of_year)], dtype=np.float32),
            day.wholesale_24.astype(np.float32),
            day.expected_arrivals_24.astype(np.float32),
            day.congestion_24.astype(np.float32),
        ])
        x = np.round(x, self.obs_round).astype(np.float32)
        return x.tobytes()

    def _act_key(self, a24: np.ndarray) -> bytes:
        a = np.round(a24.astype(np.float32), self.act_round).astype(np.float32)
        return a.tobytes()

    def _sample_action(self, rng: np.random.Generator) -> np.ndarray:
        return (rng.random(24, dtype=np.float32) * float(self.p_max)).astype(np.float32)

    # ----- policy -----
    def act(self, day, *, eval_mode: bool = False) -> np.ndarray:
        s_key = self._obs_key_from_day(day)
        state_actions = self.Q.get(s_key, {})

        rng = self.rng if self.rng is not None else np.random.default_rng(0)

        # If state is unseen, always explore (also in eval)
        if len(state_actions) == 0:
            return self._sample_action(rng)

        # Known state
        if eval_mode:
            best_key = max(state_actions.items(), key=lambda kv: kv[1])[0]
            a24 = np.frombuffer(best_key, dtype=np.float32).copy()
            return np.clip(a24, 0.0, self.p_max).astype(np.float32)

        # Train: epsilon-greedy
        if rng.random() < self.eps:
            return self._sample_action(rng)

        best_key = max(state_actions.items(), key=lambda kv: kv[1])[0]
        a24 = np.frombuffer(best_key, dtype=np.float32).copy()
        return np.clip(a24, 0.0, self.p_max).astype(np.float32)


    # ----- learning step -----
    def observe(self, day, action_24: np.ndarray, reward: float) -> None:
        s_key = self._obs_key_from_day(day)
        a24 = np.clip(action_24.astype(np.float32), 0.0, self.p_max)
        a_key = self._act_key(a24)
        R = float(reward)

        state_actions = self.Q.setdefault(s_key, {})

        q_old = float(state_actions.get(a_key, 0.0))
        q_new = (1.0 - self.alpha_q) * q_old + self.alpha_q * R
        state_actions[a_key] = q_new

        # cache best action
        if (s_key not in self.best_action) or (q_new >= max(state_actions.values())):
            self.best_action[s_key] = a24.copy()

        # cap per-state memory
        if len(state_actions) > self.max_actions_per_state:
            worst = sorted(state_actions.items(), key=lambda kv: kv[1])[: len(state_actions) - self.max_actions_per_state]
            for k_rm, _ in worst:
                state_actions.pop(k_rm, None)

    # optional compat
    def train(self, *args, **kwargs) -> TrainOutput:
        return TrainOutput(model_state=self.state_dict(), train_logs=[])

