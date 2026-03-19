from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
import numpy as np

from benchmark.dataset import DayRecord

@dataclass
class TrainOutput:
    model_state: Any
    train_logs: List[Dict[str, float]]  # per epoch: train/val metrics

class MacroModelAdapter(Protocol):
    """
    Macro-action adapter:
      - input state: obs0 from env (76-dim)
      - output action: 24-dim vector in [0,1]
      - one action per episode, but env returns 24 hourly rewards
    """
    def train(self, train_days: List[DayRecord], val_days: List[DayRecord], *, seed: int) -> TrainOutput: ...
    def act(self, obs0: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...
    def load_state(self, state: Any) -> None: ...
