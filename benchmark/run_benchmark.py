from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(sys.executable)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import inspect
import numpy as np

from benchmark.config import BenchmarkConfig
from benchmark.ev_scaling import load_ev_scaling
from benchmark.dataset import load_sessions_index, load_hourly_prices, make_day_records
from benchmark.utils import (
    ensure_dir,
    save_json,
    save_csv,
    set_global_seeds
)
from benchmark.registry import all_models
from benchmark.evaluate import (
    make_env,
    run_one_episode,
    run_one_episode_with_traj,
    run_one_episode_metrics,
)

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


def main(cfg: BenchmarkConfig):
    ensure_dir(cfg.out_dir)

    # Load EV scaling (carry-forward in load_ev_scaling)
    evtab = load_ev_scaling(cfg.ev_scaling_csv)

    # Sessions index + optional station stats
    df_sessions, station_col, n_unique_stations = load_sessions_index(cfg.sessions_csv)

    # Prices
    prices_hourly = load_hourly_prices(cfg.prices_csv)

    # Years we need
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
        train_days.extend(by_year[y])

    val_year = getattr(cfg, "val_year", None)
    val_days = by_year[val_year] if val_year is not None else []
    test_days = by_year[int(cfg.test_year)]

    # Build graph once (fixed topology)
    G = generate_station_graph(
        n_nodes=cfg.n_nodes,
        area_size=(2.0, 2.0),
        base_radius=cfg.base_radius,
        max_degree=cfg.max_degree,
        competitor_share=cfg.competitor_share,
        seed=cfg.graph_seed,
    )

    # Base env kwargs (session_energy_data is seed-specific, set per run)
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

        # reward weights
        w_rev=cfg.w_rev,
        w_cong=cfg.w_cong,
        w_short=cfg.w_short,
        w_lost=cfg.w_lost,
        w_vol=cfg.w_vol,
        w_comp=cfg.w_comp,

        congestion_hours=cfg.congestion_hours,
        charging_graph=G,
    )

    # Register models
    register_all(env_kwargs_base, cfg.epochs, cfg.congestion_hours)

    results_rows = []
    summary_rows = []

    for model_name, spec in all_models().items():
        model_dir = os.path.join(cfg.out_dir, model_name)
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
            env.rng = np.random.default_rng(seed)

            model = spec.adapter_cls()
            model.reset(seed=seed)
            train_curve = []

            # Train
            for epoch in range(cfg.epochs):
                for i, day in enumerate(train_days):
                    next_day = train_days[i + 1] if (i + 1) < len(train_days) else train_days[0]
                    done = (i + 1) == len(train_days)

                    action_24 = model.act(day, eval_mode=False)
                    out = run_one_episode(env, day, action_24, seed=seed)

                    call_observe(model, day, action_24, out["reward"], next_day, done)
                    train_curve.append(out["reward"])

                    
            # Validation (optional)
            if val_days:
                vals = []
                for day in val_days:
                    action_24 = model.act(day, eval_mode=True)
                    out = run_one_episode(env, day, action_24, seed=seed)
                    vals.append(out["reward"])
                val_mean = float(np.mean(vals))
                run_rewards_val.append(val_mean)
            else:
                val_mean = float("nan")

            # Test
            tests = []
            test_totals = _empty_test_totals()

            for day in test_days:
                action_24 = model.act(day, eval_mode=True)

                # detailed metrics
                out = run_one_episode_metrics(env, day, action_24, seed=seed)

                tests.append(out["reward"])
                for k in test_totals.keys():
                    test_totals[k] += float(out.get(k, 0.0))

            test_mean = float(np.mean(tests)) if tests else 0.0
            run_rewards_test.append(test_mean)

            save_json(
                os.path.join(model_dir, f"run_{run_i}_seed_{seed}.json"),
                {
                    "model": model_name,
                    "seed": seed,
                    "val_mean_reward": val_mean,
                    "test_mean_reward": test_mean,
                    "epochs": cfg.epochs,
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
                    "test_totals": test_totals,
                },
            )
            save_csv(
                os.path.join(model_dir, f"train_curve_run_{run_i}.csv"),
                [{"step": i, "reward": float(r)} for i, r in enumerate(train_curve)],
            )

            results_rows.append(
                {
                    "model": model_name,
                    "run": run_i,
                    "seed": seed,
                    "val_mean_reward": val_mean,
                    "test_mean_reward": test_mean,
                }
            )

        summary_rows.append(
            {
                "model": model_name,
                "val_mean_reward": float(np.mean(run_rewards_val)) if run_rewards_val else float("nan"),
                "val_std_reward": float(np.std(run_rewards_val)) if run_rewards_val else float("nan"),
                "test_mean_reward": float(np.mean(run_rewards_test)) if run_rewards_test else 0.0,
                "test_std_reward": float(np.std(run_rewards_test)) if run_rewards_test else 0.0,
                "n_runs": cfg.n_runs,
                "epochs": cfg.epochs,
            }
        )

    save_csv(os.path.join(cfg.out_dir, "runs.csv"), results_rows)
    save_csv(os.path.join(cfg.out_dir, "summary.csv"), summary_rows)
    save_json(os.path.join(cfg.out_dir, "config_used.json"), cfg.__dict__)
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
