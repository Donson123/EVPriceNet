from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

TZ = "Europe/Amsterdam"

START_COL_CANDIDATES = [
    "arrival", "start", "session_start", "started_at", "start_time", "timestamp_start"
]

STATION_ID_CANDIDATES = [
    "station_id", "station", "cp_id", "charger_id", "location_id", "site_id"
]

@dataclass(frozen=True)
class DayRecord:
    date_local: pd.Timestamp               # midnight local
    dow: int
    day_of_year: int
    wholesale_24: np.ndarray               # (24,)
    competitor_24: np.ndarray              # (24,)
    expected_arrivals_24: np.ndarray       # (24,) per node expected rate
    congestion_24: np.ndarray              # (24,) 0/1 flags

def date_range_compat(*, start, end, freq: str, inclusive: str = "left"):
    try:
        return pd.date_range(start=start, end=end, freq=freq, inclusive=inclusive)
    except TypeError:
        # older pandas fallback
        closed_map = {"left": "left", "right": "right"}
        closed = closed_map.get(inclusive, None)
        if closed is None:
            # fallback: generate full range and trim
            dr = pd.date_range(start=start, end=end, freq=freq)
            if inclusive == "both":
                return dr
            if inclusive == "neither":
                return dr[(dr > pd.Timestamp(start)) & (dr < pd.Timestamp(end))]
            # default safest
            return dr
        return pd.date_range(start=start, end=end, freq=freq, closed=closed)

def _find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def _parse_ts_utc(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.isna().mean() > 0.5:
        # fallback: parse as local
        ts_local = pd.to_datetime(s, errors="coerce")
        if getattr(ts_local.dt, "tz", None) is None:
            ts_local = ts_local.dt.tz_localize(TZ)
        else:
            ts_local = ts_local.dt.tz_convert(TZ)
        ts = ts_local.dt.tz_convert("UTC")
    return ts

def load_sessions_index(path: str) -> Tuple[pd.DataFrame, Optional[str], int]:
    """
    Returns:
      df_sessions with columns:
        - start_local
        - date, month, day, hour
      station_col (if present)
      n_unique_stations (>=1)
    """
    df = pd.read_csv(path)

    start_col = _find_first_existing_col(df, START_COL_CANDIDATES)
    if start_col is None:
        raise ValueError(f"Could not infer start/arrival column from: {list(df.columns)}")

    station_col = _find_first_existing_col(df, STATION_ID_CANDIDATES)

    start_utc = _parse_ts_utc(df[start_col])
    start_local = start_utc.dt.tz_convert(TZ)

    out = pd.DataFrame({"start_local": start_local}).dropna()
    out["date"] = out["start_local"].dt.date
    out["month"] = out["start_local"].dt.month.astype(int)
    out["day"] = out["start_local"].dt.day.astype(int)
    out["hour"] = out["start_local"].dt.hour.astype(int)

    n_unique = 1
    if station_col is not None:
        out[station_col] = df.loc[out.index, station_col].values
        n_unique = int(pd.Series(out[station_col]).nunique(dropna=True))
        n_unique = max(n_unique, 1)

    return out, station_col, n_unique

def compute_lambda_profile_for_month_day(
    df_sessions: pd.DataFrame,
    month: int,
    day: int,
    *,
    kappa: float,
    fallback_rate: float,
    normalize_by: str,
    n_nodes: int,
    n_unique_stations: int,
) -> np.ndarray:
    """
    24-length λ_t profile, interpreted as per-node expected arrivals rate for hour t.

    normalize_by:
      - "n_nodes": divide network totals by n_nodes
      - "n_unique_stations": divide by n_unique_stations (if station IDs are meaningful)
    """
    sub = df_sessions[(df_sessions["month"] == month) & (df_sessions["day"] == day)]
    if sub.empty:
        return np.ones(24, dtype=float) * float(fallback_rate)

    # count starts per (date, hour)
    per_day_hour = (
        sub.groupby(["date", "hour"]).size().unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )

    mean_network_starts_per_hour = per_day_hour.mean(axis=0).to_numpy(dtype=float)

    # global prior: mean per-hour across all dates
    global_per_hour = (
        df_sessions.groupby("hour").size().reindex(range(24), fill_value=0).to_numpy(dtype=float)
        / float(df_sessions["date"].nunique())
    )

    if normalize_by == "n_unique_stations":
        denom = float(max(n_unique_stations, 1))
    else:
        denom = float(max(n_nodes, 1))

    lam_per_node = mean_network_starts_per_hour / denom
    global_per_node = global_per_hour / denom

    if global_per_node.sum() <= 0:
        global_per_node = np.ones(24, dtype=float) * float(fallback_rate)

    # smoothing
    w = float(kappa)
    lam = (lam_per_node + (w / 24.0) * global_per_node) / (1.0 + (w / 24.0))
    lam = np.clip(lam, 0.0, None)

    return lam.astype(float)

def load_hourly_prices(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if "timestamp_europe_amsterdam" in df.columns:
        ts = pd.to_datetime(df["timestamp_europe_amsterdam"].astype(str).str.strip(), errors="coerce", utc=True)
        ts_local = ts.dt.tz_convert(TZ)
    elif "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"].astype(str).str.strip(), errors="coerce", utc=True)
        ts_local = ts.dt.tz_convert(TZ)
    else:
        raise ValueError("Prices CSV must have timestamp_europe_amsterdam or timestamp_utc")

    if "price_eur_per_mwh" in df.columns:
        price = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce") / 1000.0
    elif "price_eur_per_kwh" in df.columns:
        price = pd.to_numeric(df["price_eur_per_kwh"], errors="coerce")
    else:
        raise ValueError("Prices CSV must have price_eur_per_mwh or price_eur_per_kwh")

    s = pd.Series(price.to_numpy(dtype=float), index=ts_local)
    s = s[~s.index.isna()].sort_index()
    s = s[~s.isna()]

    # fill missing hours
    full = pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h"), freq="h", tz=TZ)
    s = s.reindex(full).interpolate(limit_direction="both")
    return s

def _get_wholesale_for_day(prices_hourly: pd.Series, day: pd.Timestamp) -> np.ndarray:
    idx = pd.date_range(day, day + pd.Timedelta(hours=23), freq="h", tz=TZ)
    sub = prices_hourly.reindex(idx)
    if sub.isna().any():
        sub = sub.ffill().bfill()
    return sub.to_numpy(dtype=float)

import pandas as pd


def make_day_records(
    prices_hourly: pd.Series,
    df_sessions: pd.DataFrame,
    n_unique_stations: int,
    evtab: Dict[Tuple[int, int], float],
    *,
    year_start: int,
    year_end: int,
    n_nodes: int,
    competitor_fixed_price: float,
    congestion_hours: Tuple[int, int],
    kappa: float,
    fallback_rate: float,
    normalize_by: str,
) -> Dict[int, List[DayRecord]]:
    """
    Build DayRecord lists per year in [year_start, year_end], inclusive.
    expected_arrivals_24 is per-node λ_t, scaled by EV factor for (year, month).
    """
    out: Dict[int, List[DayRecord]] = {}
    for year in range(year_start, year_end + 1):
        days = pd.date_range(
            f"{year}-01-01", f"{year+1}-01-01",
            freq="1D", tz=TZ, inclusive="left"
        )
        recs: List[DayRecord] = []
        for day in days:
            wholesale_24 = _get_wholesale_for_day(prices_hourly, day)

            m = int(day.month)
            d = int(day.day)

            lam = compute_lambda_profile_for_month_day(
                df_sessions=df_sessions,
                month=m,
                day=d,
                kappa=kappa,
                fallback_rate=fallback_rate,
                normalize_by=normalize_by,
                n_nodes=n_nodes,
                n_unique_stations=n_unique_stations,
            )

            # EV scaling factor (carry-forward already handled in load_ev_scaling)
            factor = float(evtab.get((year, m), 1.0))
            expected_arrivals_24 = lam * factor

            competitor_24 = np.full(24, float(competitor_fixed_price), dtype=float)

            cong_start, cong_end = congestion_hours
            congestion_24 = np.zeros(24, dtype=float)
            congestion_24[cong_start:cong_end] = 1.0

            recs.append(DayRecord(
                date_local=day,
                dow=int(day.dayofweek),
                day_of_year=int(day.dayofyear),
                wholesale_24=wholesale_24,
                competitor_24=competitor_24,
                expected_arrivals_24=expected_arrivals_24,
                congestion_24=congestion_24,
            ))
        out[year] = recs
    return out
