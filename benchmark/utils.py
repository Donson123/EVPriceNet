from __future__ import annotations
import os
import json
import inspect
import numpy as np
import pandas as pd
from typing import Any, Dict, List

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)

def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs accepted by fn. Prevents crashes when env signature changes.
    """
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}

def make_session_energy_data(
    rng: np.random.Generator,
    n: int,
    mean_kwh: float,
    std_kwh: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """
    Generate lognormal samples matching dataset-level mean/std (as in your year script).
    """
    mean_kwh = float(mean_kwh)
    std_kwh = float(std_kwh)

    mu = np.log(mean_kwh**2 / np.sqrt(std_kwh**2 + mean_kwh**2))
    sigma = np.sqrt(np.log(1 + (std_kwh**2 / mean_kwh**2)))

    samples = rng.lognormal(mean=mu, sigma=sigma, size=int(n))
    return np.clip(samples, float(clip_min), float(clip_max)).astype(float)
