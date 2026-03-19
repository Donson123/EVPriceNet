from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np

from dataset import DayRecord
from utils import filter_kwargs_for_callable
from envs.dynamic_pricing_env import DynamicPricingEnv


def make_env(env_kwargs: Dict[str, Any]) -> DynamicPricingEnv:
    filtered = filter_kwargs_for_callable(DynamicPricingEnv.__init__, env_kwargs)
    return DynamicPricingEnv(**filtered)


def eval_policy(env: DynamicPricingEnv, days, policy_fn, *, seed: int) -> Tuple[float, list[float]]:
    rewards: list[float] = []
    for day in days:
        action_24 = policy_fn(day, eval_mode=True)
        out = run_one_episode(env, day, action_24, seed=seed)
        rewards.append(float(out["reward"]))
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    return mean_reward, rewards


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _hour_encoding(t: int, T: int = 24) -> Tuple[float, float]:
    ang = 2.0 * np.pi * float(t) / float(T)
    return float(np.sin(ang)), float(np.cos(ang))



def run_one_episode_with_traj(
    env: DynamicPricingEnv, day: DayRecord, action_24: np.ndarray, *, seed: int
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Runs one day and returns:
      - out: aggregated metrics for that day
      - traj: arrays with:
          actions_24, wholesale_24, competitor_24, expected_arrivals_24,
          dow, doy, reward_day, congestion_24
    Also includes "other metrics" keys in out.
    """
    action_24 = np.asarray(action_24, dtype=np.float32).reshape(24)

    obs, info = env.reset(
        options={
            "dow": day.dow,
            "day_of_year": day.day_of_year,
            "wholesale": day.wholesale_24,
            "competitor": day.competitor_24,
            "expected_arrivals": day.expected_arrivals_24,
            "congestion_flags": day.congestion_24,
        },
    )

    wholesale_24 = np.asarray(day.wholesale_24, dtype=np.float32).reshape(24)
    competitor_24 = np.asarray(day.competitor_24, dtype=np.float32).reshape(24)
    expected_arrivals_24 = np.asarray(day.expected_arrivals_24, dtype=np.float32).reshape(24)
    congestion_24 = np.asarray(day.congestion_24, dtype=np.int64).reshape(24)

    reward_day = np.zeros((24,), dtype=np.float32)

    # Aggregates (kWh, EUR)
    total_reward = 0.0
    total_margin_eur = 0.0

    total_delivered_kwh = 0.0
    total_competitor_kwh = 0.0

    total_agent_revenue_eur = 0.0
    total_competitor_revenue_eur = 0.0

    total_agent_sessions = 0.0
    total_competitor_sessions = 0.0
    total_lost_sessions = 0.0

    total_shortfall = 0.0

    agent_cong_kwh = 0.0
    competitor_cong_kwh = 0.0

    for t in range(24):
        obs, r, terminated, truncated, inf = env.step(action_24)

        rr = _as_float(r)
        reward_day[t] = rr
        total_reward += rr

        if isinstance(inf, dict):
            p_t = _as_float(inf.get("p_t", 0.0))
            p_comp_t = _as_float(inf.get("p_comp_t", 0.0))

            e_del_t = _as_float(inf.get("E_delivered_t", 0.0))
            e_comp_t = _as_float(inf.get("E_competitor_t", 0.0))

            total_delivered_kwh += e_del_t
            total_competitor_kwh += e_comp_t

            total_margin_eur += _as_float(inf.get("margin", 0.0))

            # "Revenue" (EUR) computed as price * energy
            total_agent_revenue_eur += p_t * e_del_t
            total_competitor_revenue_eur += p_comp_t * e_comp_t

            total_agent_sessions += _as_float(inf.get("sessions_ours_new", 0.0))
            total_competitor_sessions += _as_float(inf.get("sessions_comp_new", 0.0))
            total_lost_sessions += _as_float(inf.get("sessions_lost", 0.0))

            total_shortfall += _as_float(inf.get("shortfall_t", 0.0))

            if int(congestion_24[t]) == 1:
                agent_cong_kwh += e_del_t
                competitor_cong_kwh += e_comp_t

        if terminated or truncated:
            break

    agent_vol_mwh = total_delivered_kwh / 1000.0
    comp_vol_mwh = total_competitor_kwh / 1000.0
    agent_cong_mwh = agent_cong_kwh / 1000.0
    comp_cong_mwh = competitor_cong_kwh / 1000.0

    out = dict(
        # single agent metrics
        reward=float(total_reward),
        margin=float(total_margin_eur),
        delivered_kWh=float(total_delivered_kwh),
        competitor_kWh=float(total_competitor_kwh),
        shortfall=float(total_shortfall),
        lost_sessions=float(total_lost_sessions),

        # multi agent full metrics
        agent_revenue_eur=float(total_agent_revenue_eur),
        competitor_revenue_eur=float(total_competitor_revenue_eur),
        agent_sessions=float(total_agent_sessions),
        competitor_sessions=float(total_competitor_sessions),
        agent_volume_mwh=float(agent_vol_mwh),
        competitor_volume_mwh=float(comp_vol_mwh),
        agent_congestion_mwh=float(agent_cong_mwh),
        competitor_congestion_mwh=float(comp_cong_mwh),
        total_congestion_mwh=float(agent_cong_mwh + comp_cong_mwh),
        shortfall_kWh=float(total_shortfall),
    )

    traj = dict(
        actions_24=action_24,
        wholesale_24=wholesale_24,
        competitor_24=competitor_24,
        expected_arrivals_24=expected_arrivals_24,
        dow=np.full((24,), int(day.dow), dtype=np.int64),
        doy=np.full((24,), int(day.day_of_year), dtype=np.int64),
        reward_day=reward_day,
        congestion_24=congestion_24,
    )

    return out, traj


def run_one_episode(env: DynamicPricingEnv, day: DayRecord, action_24: np.ndarray, *, seed: int) -> Dict[str, float]:
    out, _traj = run_one_episode_with_traj(env, day, action_24, seed=seed)
    return out

def run_one_episode_metrics(env: DynamicPricingEnv, day: DayRecord, action_24: np.ndarray, *, seed: int) -> Dict[str, float]:
    """
    Runs one day episode and returns detailed metrics needed for thesis tables.
    Aggregates both agent and competitor quantities over the day.
    """
    obs, info = env.reset( options={
        "dow": day.dow,
        "day_of_year": day.day_of_year,
        "wholesale": day.wholesale_24,
        "competitor": day.competitor_24,
        "expected_arrivals": day.expected_arrivals_24,
        "congestion_flags": day.congestion_24,
    })

    totals = {
        "reward": 0.0,

        # net revenue
        "revenue_agent": 0.0,
        "revenue_competitor": 0.0,

        # delivered energy
        "volume_agent_kWh": 0.0,
        "volume_competitor_kWh": 0.0,

        # sessions started
        "sessions_agent": 0.0,
        "sessions_competitor": 0.0,

        # congestion-hour delivered energy
        "congestion_agent_kWh": 0.0,
        "congestion_competitor_kWh": 0.0,
        "congestion_total_kWh": 0.0,

        # extras
        "shortfall_kWh": 0.0,
        "lost_sessions": 0.0,
    }

    t = 0
    while True:
        obs, r, terminated, truncated, inf = env.step(action_24)
        totals["reward"] += float(r)

        if isinstance(inf, dict):
            p_t = float(inf.get("p_t", 0.0))
            p_comp_t = float(inf.get("p_comp_t", 0.0))
            pi_wh_t = float(inf.get("pi_wh_t", 0.0))

            e_total = float(inf.get("E_delivered_t", 0.0))
            e_comp = float(inf.get("E_competitor_t", 0.0))
            e_agent = e_total - e_comp

            totals["volume_agent_kWh"] += e_agent
            totals["volume_competitor_kWh"] += e_comp

            totals["revenue_agent"] += (p_t - pi_wh_t) * e_agent
            totals["revenue_competitor"] += (p_comp_t - pi_wh_t) * e_comp

            totals["sessions_agent"] += float(inf.get("sessions_ours_new", 0.0))
            totals["sessions_competitor"] += float(inf.get("sessions_comp_new", 0.0))

            if day.congestion_24[t] > 0:
                totals["congestion_agent_kWh"] += e_agent
                totals["congestion_competitor_kWh"] += e_comp
                totals["congestion_total_kWh"] += e_total

            totals["shortfall_kWh"] += float(inf.get("shortfall_t", 0.0))
            totals["lost_sessions"] += float(inf.get("sessions_lost", 0.0))

        t += 1
        if terminated or truncated:
            break

    # convenience MWh versions
    totals["congestion_agent_MWh"] = totals["congestion_agent_kWh"] / 1000.0
    totals["congestion_competitor_MWh"] = totals["congestion_competitor_kWh"] / 1000.0
    totals["congestion_total_MWh"] = totals["congestion_total_kWh"] / 1000.0

    return totals