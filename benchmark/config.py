from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkConfig:
    # Inputs
    sessions_csv: str
    prices_csv: str
    ev_scaling_csv: str

    # Splits
    train_years: Tuple[int, int] = (2018, 2024)   # inclusive
    val_year: Optional[int] = None                # set to e.g. 2023 if you want validation
    test_year: int = 2025

    # Runs
    n_runs: int = 20
    seeds: Optional[List[int]] = None

    # Output
    out_dir: str = "results_cash"

    # Environment config
    day_steps: int = 24
    dt_hours: float = 1.0
    p_max: float = 1.00

    # Graph / network
    n_nodes: int = 50
    competitor_share: float = 0.5
    base_radius: float = 0.35
    max_degree: int = 3
    graph_seed: int = 42

    # Station / charging capacity
    n_chargers: int = 50
    charger_power_kW: float = 11
    sessions_per_node: int = 2

    # Demand modelling
    lambda_kappa: float = 24.0
    lambda_fallback_rate: float = 0.01
    normalize_lambda_by: str = "n_nodes"  # "n_nodes" or "n_unique_stations"

    # Competitor pricing
    competitor_fixed_price: float = 0.45

    # Session energy distribution
    session_energy_col: str = "energy_kWh"
    session_energy_n_samples: int = 180_000
    session_energy_clip_min: float = 4.0
    session_energy_clip_quantile: float = 0.995  # upper clip = q-quantile of empirical data

    # Choice model parameters
    k_price_comp: float = 2.5
    k_price_delay: float = 2.5
    mid_diff: float = 0.05

    # Reward weights
    w_rev: float = 0.3
    w_cong: float = 0.25
    w_short: float = 0.1
    w_lost: float = 0.05
    w_vol: float = 0.1
    w_comp: float = 0.2

    # Congestion hours (end-exclusive)
    congestion_hours: Tuple[int, int] = (16, 21)

    # Training schedule
    epochs: int = 1

    def __post_init__(self):
        if self.seeds is None:
            object.__setattr__(self, "seeds", list(range(42, 42 + self.n_runs)))

        if len(self.seeds) != self.n_runs:
            raise ValueError("len(seeds) must equal n_runs")

        if self.normalize_lambda_by not in ("n_nodes", "n_unique_stations"):
            raise ValueError("normalize_lambda_by must be 'n_nodes' or 'n_unique_stations'")

        if self.val_year is not None:
            if not (self.train_years[0] <= self.val_year <= self.test_year):
                raise ValueError("val_year must be within the overall simulated range")

        if self.train_years[1] >= self.test_year:
            raise ValueError("train_years must end before test_year (no leakage)")

        if self.session_energy_n_samples <= 0:
            raise ValueError("session_energy_n_samples must be > 0")

        if self.session_energy_clip_min < 0:
            raise ValueError("session_energy_clip_min must be >= 0")

        q = float(self.session_energy_clip_quantile)
        if not (0.90 <= q <= 0.9999):
            raise ValueError("session_energy_clip_quantile must be in [0.90, 0.9999]")

    def sample_session_energy_data(self, rng: np.random.Generator) -> np.ndarray:
        """
        Empirical-only session energy sampling:
        - loads sessions_csv
        - reads column session_energy_col (expected: 'energy_kWh')
        - filters invalid/non-positive values
        - clips to [clip_min, quantile(clip_q)]
        - draws session_energy_n_samples with replacement
        """
        df = pd.read_csv(self.sessions_csv)

        if self.session_energy_col not in df.columns:
            raise ValueError(
                f"sessions_csv missing column '{self.session_energy_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        x = pd.to_numeric(df[self.session_energy_col], errors="coerce").dropna().to_numpy(dtype=float)
        x = x[x > 0.0]

        if x.size == 0:
            raise ValueError("No valid positive session energies found in sessions_csv")

        lo = float(self.session_energy_clip_min)
        hi = float(np.quantile(x, float(self.session_energy_clip_quantile)))

        if hi <= lo:
            raise ValueError(f"Invalid clipping range: hi ({hi:.3f}) <= lo ({lo:.3f})")

        x = np.clip(x, lo, hi)

        idx = rng.integers(0, x.size, size=int(self.session_energy_n_samples))
        return x[idx]
