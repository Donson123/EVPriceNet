from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class FlatTariff:
    """
     baseline: fixed flat tariff for the whole day.

    - At each episode/day, returns a length-24 price vector with a constant price.
    - Compatible with the  runners that expect .reset(), .act(), .observe(), .state_dict().

    Notes:
    - Prices are clipped to [0, p_max].
    - No learning: observe() is a no-op.
    """
    env_kwargs: Dict[str, Any]
    epochs: int = 0

    flat_price: float = 0.45  # EUR/kWh
    seed: int = 42

    def __post_init__(self):
        self.day_steps = int(self.env_kwargs.get("day_steps", 24))
        self.p_max = float(self.env_kwargs.get("p_max", 1.0))
        self.rng: Optional[np.random.Generator] = None

        # Precompute the constant action vector (still clip for safety)
        p = float(np.clip(self.flat_price, 0.0, self.p_max))
        self._action_24 = np.full((self.day_steps,), p, dtype=np.float32)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, day, *, eval_mode: bool = False) -> np.ndarray:
        # day is unused; kept for a consistent interface
        return self._action_24.copy()

    def observe(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def state_dict(self) -> Dict[str, Any]:
        return {"flat_price": float(self.flat_price)}

    def load_state(self, state: Dict[str, Any]) -> None:
        if "flat_price" in state:
            self.flat_price = float(state["flat_price"])
            p = float(np.clip(self.flat_price, 0.0, self.p_max))
            self._action_24 = np.full((self.day_steps,), p, dtype=np.float32)
