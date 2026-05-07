# EVPriceNet

EVPriceNet is a simulation framework for evaluating reinforcement learning (RL) approaches for dynamic electric vehicle (EV) charging pricing under congestion constraints.

The environment simulates:
- stochastic EV arrivals
- charging station competition
- congestion and grid constraints
- user price sensitivity
- charging session dynamics
- day-ahead electricity prices

The goal is to train RL agents that generate day-ahead charging prices while balancing revenue, congestion mitigation, service quality, and competitive positioning.

---

# Features

- Dynamic EV charging pricing environment
- Multi-objective reward optimization
- Reinforcement learning baselines:
  - Q-Learning
  - DQN
  - DDPG
  - PDDPG
- Semi-realistic charging network simulation
- Congestion-aware energy delivery
- Competitor charging stations
- Historical Dutch wholesale electricity prices
- EV demand scaling based on EV adoption

---

# Project Structure

```text
.
├── dynamic_pricing_env.py
├── Charging_Network_Graph.py
├── config.py
├── evaluate.py
├── run_benchmark.py
├── dataset.py
├── ev_scaling.py
├── q_learning_macro.py
├── dqn_macro.py
├── ddpg_macro.py
├── pddpg_macro.py
├── flat_tariff_macro.py
├── utils.py
├── registry.py
├── ev_scaling.csv
└── dayahead_nl_2018_2025.csv
```

---

# Environment Overview

Each episode represents one full day with 24 hourly timesteps.

At the start of the episode, the agent submits a full 24-hour charging price schedule:

```python
a = [p_0, p_1, ..., p_23]
```

The environment then simulates:
- EV arrivals
- charging demand
- station competition
- congestion effects
- delayed charging behaviour

The action remains fixed for the entire simulated day.

---

# Observation Space

The observation contains:
- cyclical day-of-week encoding
- cyclical day-of-year encoding
- 24-hour wholesale electricity forecast
- expected EV arrivals
- congestion indicators

Observation size:

```python
4 + (3 * 24) = 76 dimensions
```

---

# Reward Function

The reward balances multiple operational objectives:

```text
Reward =
+ revenue
- congestion penalty
- shortfall penalty
- lost sessions penalty
- price volatility penalty
- competitor energy penalty
```

The objective is not only maximizing revenue, but also:
- reducing congestion
- improving service quality
- reducing volatility
- remaining competitive

---

# Data Sources

## EV Charging Demand

Synthetic charging profiles generated using:
- Elaad charging profile simulator

## Electricity Prices

Historical Dutch day-ahead wholesale electricity prices from:
- ENTSO-E

## EV Adoption Scaling

Historical EV adoption statistics from:
- Dutch government vehicle fleet statistics

---

# Running Experiments

## Run benchmark experiments

```bash
python run_benchmark.py
```

---

# Dependencies

Main libraries used:

```text
numpy
pandas
torch
gymnasium
networkx
matplotlib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Reinforcement Learning Models

The following RL approaches are implemented:

- Q-Learning
- Deep Q-Network (DQN)
- Deep Deterministic Policy Gradient (DDPG)
- Prioritized Deep Deterministic Policy Gradient (PDDPG)

All models are evaluated under identical simulation settings.

---

# License

This repository is intended for academic and research purposes.
