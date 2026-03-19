from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class EVScalingTable:
    factors: Dict[Tuple[int, int], float]  # (year, month) -> factor

    def factor_for(self, year: int, month: int) -> float:
        key = (int(year), int(month))
        if key in self.factors:
            return float(self.factors[key])

        candidates = [k for k in self.factors.keys() if k <= key]
        if not candidates:
            return float(self.factors[min(self.factors.keys())])

        best = max(candidates)
        return float(self.factors[best])

    
    def get(self, key, default=1.0) -> float:
        year, month = key
        try:
            return float(self.factor_for(int(year), int(month)))
        except Exception:
            return float(default)


def load_ev_scaling(path: str) -> EVScalingTable:
    df = pd.read_csv(path)
    req = {"year", "month", "factor"}
    if not req.issubset(df.columns):
        raise ValueError(f"EV scaling CSV must contain {sorted(req)}. Found: {list(df.columns)}")

    factors: Dict[Tuple[int, int], float] = {}
    for _, r in df.iterrows():
        y = int(r["year"]); m = int(r["month"]); f = float(r["factor"])
        factors[(y, m)] = f
    return EVScalingTable(factors=factors)
