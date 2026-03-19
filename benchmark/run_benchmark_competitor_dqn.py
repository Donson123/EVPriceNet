from __future__ import annotations

import os
import sys
import inspect
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(sys.executable)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.config import BenchmarkConfig
from benchmark.ev_scaling import load_ev_scaling
from benchmark.dataset import load_sessions_index, load_hourly_prices, make_day_records, DayRecord
from benchmark.utils import ensure_dir, save_json, save_csv, set_global_seeds
from benchmark.registry import all_models
from benchmark.evaluate import make_env
from benchmark.models import register_all
from envs.charging_network_graph import generate_station_graph

def _empty_test_totals():
    """
    Helper function that intializes all metrics at 0.0

    """
    return {
        "reward": 0.0,
        "revenue_agent": 0.0,
        "revenue_competitor": 0.0,
        "volume_agent_kWh": 0.0,
        "volume_competitor_kWh": 0.0,
        "sessions_agent": 0.0,
        "sessions_competitor": 0.0,
        "congestion_agent_kWh": 0.0,
        "congestion_competitor_kWh": 0.0,
        "congestion_total_kWh": 0.0,
        "congestion_agent_MWh": 0.0,
        "congestion_competitor_MWh": 0.0,
        "congestion_total_MWh": 0.0,
        "shortfall_kWh": 0.0,
        "lost_sessions": 0.0,
    }


def call_observe(model, day, action_24, reward, next_day, done):
    """
    Call a model's `observe` method if it exists, supporting multiple signatures.

    This helper supports two common observe signatures:
        - observe(day, action_24, reward)
        - observe(day, action_24, reward, next_day, done)

    If signature introspection fails or the signature is unexpected, it falls back
    to trying both call patterns.

    Parameters
        model
            Object that may implement an `observe` method.
        day
            Current day identifier or state used by the model.
        action_24
            Daily action vector (e.g., 24-hour price schedule).
        reward
            Realized reward for the current day.
        next_day
            Next day identifier or state (used by some models).
        done
            Episode termination flag (used by some models).

    Returns
        None
            This function is called for side effects only.
    """
    obs_fn = getattr(model, "observe", None)
    if obs_fn is None:
        return

    try:
        sig = inspect.signature(obs_fn)
        n_params = len(sig.parameters)
    except Exception:
        try:
            model.observe(day, action_24, reward)
            return
        except TypeError:
            model.observe(day, action_24, reward, next_day, done)
            return

    if n_params == 3:
        model.observe(day, action_24, reward)
    elif n_params == 5:
        model.observe(day, action_24, reward, next_day, done)
    else:
        try:
            model.observe(day, action_24, reward)
        except TypeError:
            model.observe(day, action_24, reward, next_day, done)


def _make_competitor(env_kwargs_base, epochs: int, congestion_hours):
    """
    Try to get DQN, otherwise import flat competitor
    """
    try:
        from benchmark.models.dqn import DQN
        return DQN(env_kwargs=env_kwargs_base, epochs=epochs)
    except Exception:
        from benchmark.models.flat_tariff import FlatTariff 
        return DQN(env_kwargs=env_kwargs_base, epochs=epochs)


def run_one_episode_dual(env, day: DayRecord, action_agent_24: np.ndarray, *, seed: int) -> Tuple[dict, float]:
    """
    Runs one episode and returns:
      out_agent: dict with daily aggregates (reward key is 'reward')
      reward_comp: float proxy competitor reward (margin proxy)
    """
    obs, info = env.reset(seed=seed, options={
        "dow": day.dow,
        "day_of_year": day.day_of_year,
        "wholesale": day.wholesale_24,
        "competitor": day.competitor_24,
        "expected_arrivals": day.expected_arrivals_24,
    })

    total_reward_agent = 0.0
    total_reward_comp = 0.0

    total_margin_agent = 0.0
    total_delivered = 0.0
    total_competitor_kwh = 0.0
    total_shortfall = 0.0
    total_lost = 0.0

    for t in range(24):
        obs, r_t, terminated, truncated, inf = env.step(action_agent_24)
        total_reward_agent += float(r_t)

        if isinstance(inf, dict):
            total_margin_agent += float(inf.get("margin", 0.0))
            total_delivered += float(inf.get("E_delivered_t", 0.0))
            total_competitor_kwh += float(inf.get("E_competitor_t", 0.0))
            total_shortfall += float(inf.get("shortfall_t", 0.0))
            total_lost += float(inf.get("sessions_lost", inf.get("sessions_lost_t", 0.0)))

            E_comp_t = float(inf.get("E_competitor_t", 0.0))
            p_comp_t = float(inf.get("p_comp_t", day.competitor_24[t] if t < len(day.competitor_24) else 0.0))
            pi_wh_t = float(inf.get("pi_wh_t", day.wholesale_24[t] if t < len(day.wholesale_24) else 0.0))
            total_reward_comp += (p_comp_t - pi_wh_t) * E_comp_t

        if terminated or truncated:
            break

    out_agent = dict(
        reward=float(total_reward_agent),
        margin=float(total_margin_agent),
        delivered_kWh=float(total_delivered),
        competitor_kWh=float(total_competitor_kwh),
        shortfall=float(total_shortfall),
        lost_sessions=float(total_lost),
    )
    return out_agent, float(total_reward_comp)


def run_one_episode_dual_metrics(env, day: DayRecord, action_agent_24: np.ndarray, *, seed: int) -> dict:
    """
    Runs one dual-competitor episode and returns detailed totals
    in the same format as the flat-competitor benchmark.
    """
    obs, info = env.reset(seed=seed, options={
        "dow": day.dow,
        "day_of_year": day.day_of_year,
        "wholesale": day.wholesale_24,
        "competitor": day.competitor_24,
        "expected_arrivals": day.expected_arrivals_24,
    })

    totals = _empty_test_totals()

    for t in range(24):
        obs, r_t, terminated, truncated, inf = env.step(action_agent_24)
        totals["reward"] += float(r_t)

        if isinstance(inf, dict):
            p_t = float(inf.get("p_t", 0.0))
            p_comp_t = float(inf.get("p_comp_t", day.competitor_24[t] if t < len(day.competitor_24) else 0.0))
            pi_wh_t = float(inf.get("pi_wh_t", day.wholesale_24[t] if t < len(day.wholesale_24) else 0.0))

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
            totals["lost_sessions"] += float(inf.get("sessions_lost", inf.get("sessions_lost_t", 0.0)))

        if terminated or truncated:
            break

    totals["congestion_agent_MWh"] = totals["congestion_agent_kWh"] / 1000.0
    totals["congestion_competitor_MWh"] = totals["congestion_competitor_kWh"] / 1000.0
    totals["congestion_total_MWh"] = totals["congestion_total_kWh"] / 1000.0

    return totals



def _eval_days(env, days, agent_model, competitor_model, *, seed: int) -> float:
    rewards = []
    for day in days:
        comp_24 = competitor_model.act(day, eval_mode=True)
        day_with_comp = replace(day, competitor_24=comp_24)

        a24 = agent_model.act(day_with_comp, eval_mode=True)
        out, _rcomp = run_one_episode_dual(env, day_with_comp, a24, seed=seed)
        rewards.append(out["reward"])
    return float(np.mean(rewards)) if rewards else float("nan")


def main(cfg: BenchmarkConfig) -> None:
    ensure_dir(cfg.out_dir)

    evtab = load_ev_scaling(cfg.ev_scaling_csv)
    df_sessions, _station_col, n_unique_stations = load_sessions_index(cfg.sessions_csv)
    prices_hourly = load_hourly_prices(cfg.prices_csv)

    year_min = int(cfg.train_years[0])
    cands = [int(cfg.train_years[1]), int(cfg.test_year)]
    if getattr(cfg, "val_year", None) is not None:
        cands.append(int(cfg.val_year))
    year_max = max(cands)

    by_year = make_day_records(
        prices_hourly=prices_hourly,
        df_sessions=df_sessions,
        n_unique_stations=n_unique_stations,
        evtab=evtab,
        year_start=year_min,
        year_end=year_max,
        n_nodes=cfg.n_nodes,
        competitor_fixed_price=cfg.competitor_fixed_price,
        congestion_hours=cfg.congestion_hours,
        kappa=cfg.lambda_kappa,
        fallback_rate=cfg.lambda_fallback_rate,
        normalize_by=cfg.normalize_lambda_by,
    )

    train_days = []
    for y in range(int(cfg.train_years[0]), int(cfg.train_years[1]) + 1):
        train_days.extend(by_year[int(y)])

    val_year = getattr(cfg, "val_year", None)
    val_days = by_year[int(val_year)] if val_year is not None else []
    test_days = by_year[int(cfg.test_year)]

    G = generate_station_graph(
        n_nodes=cfg.n_nodes,
        area_size=(2.0, 2.0),
        base_radius=cfg.base_radius,
        max_degree=cfg.max_degree,
        competitor_share=cfg.competitor_share,
        seed=cfg.graph_seed,
    )

    env_kwargs_base = dict(
        day_steps=cfg.day_steps,
        dt_hours=cfg.dt_hours,
        p_max=cfg.p_max,
        n_chargers=cfg.n_chargers,
        charger_power_kW=cfg.charger_power_kW,
        sessions_per_node=cfg.sessions_per_node,
        k_price_comp=cfg.k_price_comp,
        k_price_delay=cfg.k_price_delay,
        mid_diff=cfg.mid_diff,

        w_rev=cfg.w_rev,
        w_cong=cfg.w_cong,
        w_short=cfg.w_short,
        w_lost=cfg.w_lost,
        w_vol=cfg.w_vol,
        w_comp=cfg.w_comp,

        congestion_hours=cfg.congestion_hours,
        charging_graph=G,
    )

    register_all(env_kwargs_base, cfg.epochs, cfg.congestion_hours)

    results_rows = []
    summary_rows = []

    train_competitor_online = True

    for model_name, spec in all_models().items():
        model_dir = os.path.join(cfg.out_dir, f"{model_name}__vs_dqn_comp")
        ensure_dir(model_dir)

        run_rewards_test = []
        run_rewards_val = []

        for run_i, seed in enumerate(cfg.seeds):
            set_global_seeds(seed)
            rng = np.random.default_rng(seed)

            session_energy_data = cfg.sample_session_energy_data(rng)
            env_kwargs = dict(env_kwargs_base)
            env_kwargs["session_energy_data"] = session_energy_data

            env = make_env(env_kwargs)

            agent = spec.adapter_cls()
            agent.reset(seed=seed)

            competitor = _make_competitor(env_kwargs_base, cfg.epochs, cfg.congestion_hours)
            competitor.reset(seed=seed)

            train_curve = []

            # Train
            for epoch in range(cfg.epochs):
                for i, day in enumerate(train_days):
                    next_day = train_days[i + 1] if (i + 1) < len(train_days) else train_days[0]
                    done = (i + 1) == len(train_days)

                    comp_24 = competitor.act(day, eval_mode=not train_competitor_online)
                    day_with_comp = replace(day, competitor_24=comp_24)

                    a24 = agent.act(day_with_comp, eval_mode=False)
                    out_agent, r_comp = run_one_episode_dual(env, day_with_comp, a24, seed=seed)

                    call_observe(agent, day_with_comp, a24, out_agent["reward"], next_day, done)
                    train_curve.append(out_agent["reward"])

                    if train_competitor_online:
                        call_observe(competitor, day, comp_24, r_comp, next_day, done)

            # Validation
            if val_days:
                val_mean = _eval_days(env, val_days, agent, competitor, seed=seed)
                run_rewards_val.append(val_mean)
            else:
                val_mean = float("nan")

            # Test with detailed metrics
            tests = []
            test_totals = _empty_test_totals()

            for day in test_days:
                comp_24 = competitor.act(day, eval_mode=True)
                day_with_comp = replace(day, competitor_24=comp_24)

                a24 = agent.act(day_with_comp, eval_mode=True)
                out = run_one_episode_dual_metrics(env, day_with_comp, a24, seed=seed)

                tests.append(out["reward"])
                for k in test_totals.keys():
                    test_totals[k] += float(out.get(k, 0.0))

            test_mean = float(np.mean(tests)) if tests else float("nan")
            run_rewards_test.append(test_mean)

            save_json(
                os.path.join(model_dir, f"run_{run_i}_seed_{seed}.json"),
                {
                    "model": model_name,
                    "seed": seed,
                    "val_mean_reward": val_mean,
                    "test_mean_reward": test_mean,
                    "epochs": cfg.epochs,
                    "competitor": "DQN (trained online)" if train_competitor_online else "DQN (fixed)",
                },
            )

            save_json(
                os.path.join(model_dir, f"true_test_metrics_run_{run_i}_seed_{seed}.json"),
                {
                    "model": model_name,
                    "seed": seed,
                    "run": run_i,
                    "test_mean_reward": test_mean,
                    "n_test_days": len(test_days),
                    "competitor": "DQN (trained online)" if train_competitor_online else "DQN (fixed)",
                    "test_totals": test_totals,
                },
            )

            save_csv(
                os.path.join(model_dir, f"train_curve_run_{run_i}.csv"),
                [{"step": j, "reward": float(r)} for j, r in enumerate(train_curve)],
            )

            results_rows.append(
                dict(
                    model=model_name,
                    run=run_i,
                    seed=seed,
                    val_mean_reward=val_mean,
                    test_mean_reward=test_mean,
                )
            )

        summary_rows.append(
            dict(
                model=model_name,
                val_mean_reward=float(np.mean(run_rewards_val)) if run_rewards_val else float("nan"),
                val_std_reward=float(np.std(run_rewards_val)) if run_rewards_val else float("nan"),
                test_mean_reward=float(np.mean(run_rewards_test)) if run_rewards_test else float("nan"),
                test_std_reward=float(np.std(run_rewards_test)) if run_rewards_test else float("nan"),
                n_runs=cfg.n_runs,
                epochs=cfg.epochs,
            )
        )

    save_csv(os.path.join(cfg.out_dir, "runs_vs_dqn_comp.csv"), results_rows)
    save_csv(os.path.join(cfg.out_dir, "summary_vs_dqn_comp.csv"), summary_rows)
    save_json(os.path.join(cfg.out_dir, "config_used_vs_dqn_comp.json"), cfg.__dict__)
    print("Done. Results in:", cfg.out_dir)


if __name__ == "__main__":
    cfg = BenchmarkConfig(
        sessions_csv="data/sessions_all.csv",
        prices_csv="data/dayahead_nl_2018_2025_filled.csv",
        ev_scaling_csv="data/ev_scaling.csv",
        out_dir="results",
        epochs=1,
        train_years=(2018, 2024),
        val_year=None,
        test_year=2025,
    )
    main(cfg)