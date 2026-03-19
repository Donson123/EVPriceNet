import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx


class DynamicPricingEnv(gym.Env):
    """
    Environment for dynamic EV charging pricing.

    Episode structure:
    - One episode = one day with `day_steps` hourly timesteps (24).
    - At t=0, the agent provides one action: a vector a in [0,1] of length 24.
      This vector is stored as self.action_series.
    - At hour t, the environment uses a_t = self.action_series[t] to compute
      the price p_t for that hour.
    - The daily price schedule is fixed for the whole episode.

    Observation (76-dim vector for day_steps=24):
    
        [sin_dow, cos_dow, sin_y, cos_y,
         wholesale_prices[24],
         expected_arrivals[24],
         congestion_flags[24]]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        day_steps: int = 24,
        dt_hours: float = 1.0,
        p_max: float = 1.00,

        # Session-level parameters
        session_energy_data: Optional[np.ndarray] = None,
        min_stay_hours: int = 1,
        max_stay_hours: int = 8,

        # Site / charging configuration
        n_chargers: int = 20,
        charger_power_kW: float = 11,
        sessions_per_node: int = 2,
        max_site_energy_per_hour: float = 11 * 20 / 2,  

        # Behavioral and reward parameters
        k_price_comp: float = 2.5,
        k_price_delay: float = 2.5,
        mid_diff: float = 0.05,
        w_rev: float = 0.3,
        w_cong: float = 0.25,
        w_short: float = 0.1,
        w_lost: float = 0.05,
        w_vol: float = 0.1,
        w_comp: float = 0.20,
        congestion_hours: Tuple[int, int] = (16, 21),

        # Network structure
        charging_graph: Optional[nx.Graph] = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(42)

        # Time & price settings
        self.day_steps = int(day_steps)
        self.dt_hours = float(dt_hours)
        self.p_max = float(p_max)

        # Session parameters
        self.session_energy_mean = float(np.mean(session_energy_data))
        self.session_energy_std = float(np.std(session_energy_data))
        self.min_stay_hours = int(min_stay_hours)
        self.max_stay_hours = int(max_stay_hours)

        if self.min_stay_hours <= 0:
            raise ValueError("min_stay_hours must be >= 1")
        if self.max_stay_hours < self.min_stay_hours:
            raise ValueError("max_stay_hours must be >= min_stay_hours")

        # Site configuration
        self.n_chargers = int(n_chargers)
        self.charger_power_kW = float(charger_power_kW)
        self.max_site_energy_per_hour = float(max_site_energy_per_hour)
        self.sessions_per_node = int(sessions_per_node)
        if self.sessions_per_node <= 0:
            raise ValueError("sessions_per_node must be >= 1")

        # Reward & behavioral parameters
        self.k_price_comp = float(k_price_comp)
        self.k_price_delay = float(k_price_delay)
        self.mid_diff = float(mid_diff)
        self.w_rev = float(w_rev)
        self.w_cong = float(w_cong)
        self.w_short = float(w_short)
        self.w_lost = float(w_lost)
        self.w_vol = float(w_vol)
        self.w_comp = float(w_comp)
        self.congestion_hours = (int(congestion_hours[0]), int(congestion_hours[1]))

        # Graph / network state
        self.charging_graph: Optional[nx.Graph] = charging_graph
        self._components: List[Dict[str, Any]] = []
        self._n_nodes: int = 0
        self._competitor_fraction: float = 0.0

        # Time series (set in reset)
        self.wholesale: Optional[np.ndarray] = None
        self.competitor: Optional[np.ndarray] = None
        self.expected_arrivals: Optional[np.ndarray] = None

        # Runtime state
        self.day_of_year = 0
        self.t: int = 0 # time in day
        self.global_t: int = 0 # global timestep
        self.p_prev: float = 0.0
        self.dow: int = 0
        self.last_obs: Optional[np.ndarray] = None

        # Daily price action vector: a_t in [0,1] for each hour t
        self.action_series: Optional[np.ndarray] = None

        # Active sessions (our stations)
        # Each entry: {"remaining_kwh": float, "dep_step": int}
        self.active_sessions: List[Dict[str, Any]] = []

        # Gym spaces: action is full daily price schedule
        self.action_space = spaces.Box(
            low=0.0,
            high=self.p_max,
            shape=(self.day_steps,),
            dtype=np.float32,
        )

        # Observation: [sin_dow, cos_dow, sin_y, cos_y, wholesale[day_steps],arrivals[day_steps],congestion[day_steps]]
        obs_dim = 4 + 3 * self.day_steps
        self.observation_space = spaces.Box(
            low=-10.0,
            high=1e4,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        # Prepare components if a graph was passed
        self._prepare_components()

    #########################################################################################################
    # Network preprocessing


    def _prepare_components(self):
        """
        Precompute connected components and type counts from the charging graph.

         Returns
            None
                Updates internal fields:
                    - self._components
                    - self._n_nodes
                    - self._competitor_fraction
                    - self.max_site_energy_per_hour (scaled by node count)
        
        """
        G = self.charging_graph
        self._components = []
        self._n_nodes = 0
        self._competitor_fraction = 0.0

        if not isinstance(G, nx.Graph) or G.number_of_nodes() == 0:
            # No valid graph
            self._n_nodes = 1
            self._components = [{
                "size": 1,
                "n_ours": 1,
                "n_comp": 0,
            }]
            self._competitor_fraction = 0.0
            return

        self._n_nodes = G.number_of_nodes()

        total_competitors = 0
        for comp_nodes in nx.connected_components(G):
            nodes = list(comp_nodes)
            n_ours = 0
            n_comp = 0
            for n in nodes:
                is_comp = bool(G.nodes[n].get("is_competitor", False))
                if is_comp:
                    n_comp += 1
                else:
                    n_ours += 1
            self._components.append({
                "size": len(nodes),
                "n_ours": n_ours,
                "n_comp": n_comp,
                "nodes": nodes,
            })
            total_competitors += n_comp

        self._competitor_fraction = (
            float(total_competitors) / float(self._n_nodes)
            if self._n_nodes > 0 else 0.0
        )

        # Update site capacity based on number of nodes:
        if self._n_nodes > 0:
            self.max_site_energy_per_hour = self.charger_power_kW * (self._n_nodes / 2.0)
        else:
            self.max_site_energy_per_hour = self.charger_power_kW * 0.5
    ###############################################################################################################

    # Helper functions

    def _get_neighbors(self, node_id: int) -> List[int]:
        """
        Return neighbors of a node in the charging graph.

        Parameters
            node_id : int
                Node index.

        Returns
            List[int]
                Neighbor node indices. Returns an empty list if no valid graph exists.
        """
        
        if not isinstance(self.charging_graph, nx.Graph):
            return []
        return list(self.charging_graph.neighbors(node_id))

    def time_encoders(self, t: int, dow: int):
        """
        Compute cyclic encodings for time-of-day and day-of-week.

        Parameters
            t : int
                Hour index in [0, 23].
            dow : int
                Day-of-week index in [0, 6].

        Returns
            tuple[float, float, float, float]
                (sin_hour, cos_hour, sin_dow, cos_dow).
        """

        sin_hour = math.sin(2 * math.pi * t / 24)
        cos_hour = math.cos(2 * math.pi * t / 24)
        sin_dow = math.sin(2 * math.pi * dow / 7)
        cos_dow = math.cos(2 * math.pi * dow / 7)
        return sin_hour, cos_hour, sin_dow, cos_dow

    def is_congestion(self, t: int) -> bool:
        """
        Check whether hour t is within the congestion window.

        Parameters
            t : int
                Hour index in [0, 23].

        Returns
            bool
                True if t is in [congestion_hours[0], congestion_hours[1]).
        """

        return self.congestion_hours[0] <= t < self.congestion_hours[1]

    def _prob_choose_ours(self, p: float, p_comp: float) -> float:
        """
        Compute probability of choosing an agent station in a mixed component.

        Parameters
            p : float
                Agent retail price at hour t.
            p_comp : float
                Competitor retail price at hour t.

        Returns
            float
                Probability in [0, 1] of selecting an agent-owned node.
        """

        p_rel = p - p_comp - self.mid_diff
        x = p_rel * self.k_price_comp
        return float(1.0 / (1.0 + np.exp(x)))
    
    def _prob_delay(self, p_now: float, p_next: float) -> float:
        """
        Compute probability of delaying charging by one hour.

        Parameters
            p_now : float
                Current hour price.
            p_next : float
                Next hour price.

        Returns
            float
                Probability in [0, 1] of delaying by exactly one hour.
        """

        p_rel = p_now - p_next - self.mid_diff
        x = p_rel * self.k_price_delay
        return float(1.0 / (1.0 + np.exp(x)))

    def _sample_session_energy(self):

        mu = math.log(self.session_energy_mean**2 / math.sqrt(self.session_energy_std**2 + self.session_energy_mean**2))
        sigma = math.sqrt(math.log(1 + (self.session_energy_std**2 / self.session_energy_mean**2)))

        e = self.rng.lognormal(mean=mu, sigma=sigma)
        return float(np.clip(e, 2.0, 300.0))


    def _sample_dwell_hours(self):
        """
        Sample dwell time in hours for a new session.

        Returns
            int
                Dwell time in hours. Sessions starting after 20:00 are forced to 8 hours.
        """
        if (self.global_t % 24) >= 20:
            return 8

        # Skewed discrete distribution for dwell times
        dwell_values = np.array([1,2,3,4,5,6,7,8])
        dwell_probs = np.array([0.05, 0.25, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02])

        # normalize (safety)
        dwell_probs = dwell_probs / dwell_probs.sum()

        return int(self.rng.choice(dwell_values, p=dwell_probs))

    def _observe(self) -> np.ndarray:
        """
        Build the day-ahead observation vector.

        Returns
            np.ndarray
                Observation consisting of:
                    - weekly + yearly cyclic encodings
                    - day-ahead wholesale price vector
                    - day-ahead expected arrivals vector
                    - congestion indicator vector
        """

        # weekly cycle
        sin_dow = math.sin(2 * math.pi * self.dow / 7)
        cos_dow = math.cos(2 * math.pi * self.dow / 7)

        # yearly cycle
        sin_y = math.sin(2 * math.pi * self.day_of_year / 365)
        cos_y = math.cos(2 * math.pi * self.day_of_year / 365)

        # wholesale and arrivals (24 each)
        wh = self.wholesale.astype(np.float32)
        arr = self.expected_arrivals.astype(np.float32)

        # congestion vector: 1 for t in window, else 0
        cong_vec = np.zeros(self.day_steps, dtype=np.float32)
        start, end = self.congestion_hours
        cong_vec[start:end] = 1.0

        # final observation:
        # [sin_dow, cos_dow, sin_y, cos_y,
        #  24 wholesale prices,
        #  24 arrival forecasts,
        #  24 congestion indicators]
        obs = np.concatenate([
            np.array([sin_dow, cos_dow, sin_y, cos_y], dtype=np.float32),
            wh,
            arr,
            cong_vec
        ])

        return obs

    
    def _node_is_occupied(self, node_id: int) -> bool:
        """
        Check whether a node has reached its concurrent session limit.

        Parameters
            node_id : int
                Node index.

        Returns
            bool
                True if active sessions at node_id >= sessions_per_node.
        """
        count = 0
        for sess in self.active_sessions:
            if sess["node_id"] == node_id:
                count += 1
                if count >= self.sessions_per_node:
                    return True
        return False


    #########################################################################################################
    # Gym API

    def reset(self, *, seed=None, options: Optional[dict] = None):
        """
        Reset environment and inject external time series via `options`:

        Expected keys in options:
            - "dow":           integer day of week (0=Monday, ..., 6=Sunday)
            - "wholesale":     np.ndarray shape (day_steps,)
            - "competitor":    np.ndarray shape (day_steps,)
            - "expected_arrivals": np.ndarray shape (day_steps,)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        if options is None:
            options = {}

        # Allow graph to be updated before each episode if needed
        self._prepare_components()

        # Define action series
        self.action_series = None

        # Day-of-week
        self.dow = int(options.get("dow", 0)) % 7

        # Time series (required)
        self.wholesale = np.asarray(options.get("wholesale"), dtype=float)
        self.competitor = np.asarray(options.get("competitor"), dtype=float)
        self.expected_arrivals = np.asarray(options.get("expected_arrivals"), dtype=float)
        self.day_of_year = int(options.get("day_of_year", 0)) % 365

        assert len(self.wholesale) == self.day_steps, "wholesale must have length day_steps"
        assert len(self.competitor) == self.day_steps, "competitor must have length day_steps"
        assert len(self.expected_arrivals) == self.day_steps, "expected_arrivals must have length day_steps"

        # Initialize time and previous price
        self.t = 0
        self.p_prev = float(
            max(self.wholesale[0],
                min(self.p_max, self.wholesale[0] + 0.05))
        )

        obs = self._observe()
        self.last_obs = obs.copy()
        info = {
            "step": self.t,
            "competitor_fraction": self._competitor_fraction,
            "n_components": len(self._components),
            "n_nodes": self._n_nodes,
            "active_sessions": len(self.active_sessions),
        }
        return obs, info

    def step(self, action: np.ndarray):
        """
        One hourly step:
        - At t=0, interpret `action` as the full daily price schedule.
        - Simulate arrivals, relocation, operator choice, delay choice.
        - Start new sessions (agent + competitor).
        - Allocate site capacity proportionally across all sessions.
        - Compute reward.
        """

        # 1. Handle daily price schedule
        if self.action_series is None:
            a = np.asarray(action, dtype=float).reshape(-1)
            if a.shape[0] != self.day_steps:
                raise ValueError(
                    f"Action must have length {self.day_steps}, got {a.shape[0]}"
                )
            self.action_series = np.clip(a, 0.0, self.p_max)
        else:
            a = self.action_series

        # Compute agent price for current hour
        pi_wh_t = float(self.wholesale[self.t])
        p_comp_t = float(self.competitor[self.t])

        a_t = float(self.action_series[self.t])
        p_t = float(np.clip(a_t, 0.0, self.p_max))
        # 1. Sample arrivals

        lambda_per_node = float(self.expected_arrivals[self.t])
        lambda_total = self._n_nodes * lambda_per_node
        n_arrivals_total = int(self.rng.poisson(lambda_total))

        sessions_ours_new = 0
        sessions_comp_new = 0
        sessions_lost = 0
        prob_ours_mixed = self._prob_choose_ours(p_t, p_comp_t)
        delay_prob = 0.0

        # Component sampling probabilities
        if self._n_nodes > 0 and len(self._components) > 0:
            comp_sizes = np.array([c["size"] for c in self._components], dtype=float)
            comp_probs = comp_sizes / comp_sizes.sum()
        else:
            comp_probs = np.array([1.0])

        # 2. Process arrivals

        for _ in range(n_arrivals_total):

            # Choose connected component
            idx = int(self.rng.choice(len(self._components), p=comp_probs))
            comp = self._components[idx]
            nodes_in_comp = comp["nodes"]

            # Pick start node
            start_node = int(self.rng.choice(nodes_in_comp))
            start_node_data = self.charging_graph.nodes[start_node]
            is_competitor_node = bool(start_node_data.get("is_competitor", False))

            # If arrival node is occupied then try edge-neighbors
            if self._node_is_occupied(start_node):
                neighbors = self._get_neighbors(start_node)
                free_neighbors = [n for n in neighbors if not self._node_is_occupied(n)]

                if len(free_neighbors) > 0:
                    start_node = int(self.rng.choice(free_neighbors))
                    start_node_data = self.charging_graph.nodes[start_node]
                    is_competitor_node = bool(start_node_data.get("is_competitor", False))
                else:
                    # No free neighboring nodes then try other components 50/50
                    if self.rng.random() < 0.5:
                        other_components = [c for c in self._components if c is not comp]
                        if len(other_components) > 0:
                            new_comp = self.rng.choice(other_components)
                            start_node = int(self.rng.choice(new_comp["nodes"]))
                            start_node_data = self.charging_graph.nodes[start_node]
                            is_competitor_node = bool(start_node_data.get("is_competitor", False))
                        else:
                            sessions_lost += 1
                            continue
                    else:
                        sessions_lost += 1
                        continue

            # Candidate nodes = start + neighbors
            candidate_nodes = [start_node]
            if isinstance(self.charging_graph, nx.Graph):
                candidate_nodes += list(self.charging_graph.neighbors(start_node))
            candidate_nodes = list(dict.fromkeys(candidate_nodes))

            # Ownership sets
            owners = {
                nid: bool(self.charging_graph.nodes[nid].get("is_competitor", False))
                for nid in candidate_nodes
            }
            our_nodes = [nid for nid in candidate_nodes if not owners[nid]]
            comp_nodes = [nid for nid in candidate_nodes if owners[nid]]

            # Check if alternatives exist
            alternatives = [nid for nid in candidate_nodes if nid != start_node]
            if len(alternatives) == 0:
                # No alternatives then must stay at start node
                if is_competitor_node:
                    sessions_comp_new += 1

                    # Create competitor session
                    E_req = self._sample_session_energy()
                    dwell = self._sample_dwell_hours()
                    dep = self.global_t + dwell
                    self.active_sessions.append({
                        "remaining_kwh": E_req,
                        "dep_step": dep,
                        "node_id": start_node,
                        "delayed_until": self.global_t,
                        "is_competitor": True
                    })

                else:
                    sessions_ours_new += 1

                    E_req = self._sample_session_energy()
                    dwell = self._sample_dwell_hours()
                    dep = self.global_t + dwell
                    self.active_sessions.append({
                        "remaining_kwh": E_req,
                        "dep_step": dep,
                        "node_id": start_node,
                        "delayed_until": self.global_t,
                        "is_competitor": False
                    })
                continue

            # Choice from all available nodes
            available_ours = [nid for nid in our_nodes if not self._node_is_occupied(nid)]
            available_comp = [nid for nid in comp_nodes if not self._node_is_occupied(nid)]

            has_ours = len(available_ours) > 0
            has_comp = len(available_comp) > 0

            if has_ours and has_comp:
                choose_ours = (self.rng.random() < prob_ours_mixed)
            elif has_ours and not has_comp:
                choose_ours = True
            elif has_comp and not has_ours:
                choose_ours = False
            else:
                # All candidate nodes occupied = lost
                sessions_lost += 1
                continue

            # 3. Session creation
            
            if choose_ours:
                # Agent side
                if not is_competitor_node:
                    chosen_node = start_node
                else:
                    chosen_node = int(self.rng.choice(available_ours))

                # Delay decision only for agent EVs
                delay = 0
                if self.t < self.day_steps - 1:
                    p_now = p_t
                    pi_wh_next = float(self.wholesale[self.t + 1])
                    a_next = float(self.action_series[self.t + 1])
                    p_next = pi_wh_next + a_next * (self.p_max - pi_wh_next)
                    delay_prob = self._prob_delay(p_now, p_next)

                    if self.rng.random() < delay_prob:
                        delay = 1

                sessions_ours_new += 1
                E_req = self._sample_session_energy()
                dwell = self._sample_dwell_hours()
                dep = self.global_t + dwell

                self.active_sessions.append({
                    "remaining_kwh": E_req,
                    "dep_step": dep,
                    "node_id": chosen_node,
                    "delayed_until": self.global_t + delay,
                    "is_competitor": False
                })

            else:
                # Competitor side
                if is_competitor_node:
                    chosen_node = start_node
                else:
                    chosen_node = int(self.rng.choice(available_comp))

                sessions_comp_new += 1
                E_req = self._sample_session_energy()
                dwell = self._sample_dwell_hours()
                dep = self.global_t + dwell

                self.active_sessions.append({
                    "remaining_kwh": E_req,
                    "dep_step": dep,
                    "node_id": chosen_node,
                    "delayed_until": self.global_t,
                    "is_competitor": True
                })

        # 4. proportional charging

        cap_site_kWh = self.max_site_energy_per_hour
        per_session_cap = self.charger_power_kW * self.dt_hours

        # Sessions able to charge now
        eligible = [
            sess for sess in self.active_sessions
            if sess["remaining_kwh"] > 1e-6 and sess["delayed_until"] <= self.global_t
        ]

        desired = [min(sess["remaining_kwh"], per_session_cap) for sess in eligible]
        total_demand = sum(desired)

        if total_demand <= cap_site_kWh:
            scale = 1.0
        else:
            scale = cap_site_kWh / total_demand

        E_delivered_t = 0.0
        E_competitor_t = 0.0

        for sess, want in zip(eligible, desired):
            delivered = want * scale
            sess["remaining_kwh"] -= delivered
            E_delivered_t += delivered
            if sess.get("is_competitor", False):
                E_competitor_t += delivered

        # 5. SHORTFALL + session cleanup

        shortfall_t = 0.0
        new_sessions = []

        for sess in self.active_sessions:
            if sess["dep_step"] <= self.global_t + 1:
                if sess["remaining_kwh"] > 1e-6 and not sess.get("is_competitor", False):
                    shortfall_t += sess["remaining_kwh"]
            else:
                new_sessions.append(sess)

        self.active_sessions = new_sessions

        # 6. reward

        margin = (p_t - pi_wh_t) * (E_delivered_t - E_competitor_t) 
        cong_pen = self.w_cong * E_delivered_t * (1.0 if self.is_congestion(self.t) else 0.0)
        short_pen = self.w_short * shortfall_t
        lost_pen = self.w_lost * sessions_lost
        vol_pen = self.w_vol * abs(p_t - self.p_prev)
        comp_pen = self.w_comp * E_competitor_t

        R_t = (self.w_rev * margin)  - cong_pen - short_pen - lost_pen - vol_pen - comp_pen
        # 7. Update time

        self.p_prev = p_t
        self.global_t += 1
        self.t += 1
        terminated = False
        truncated = self.t >= self.day_steps
        obs = self.last_obs

        info = {
            "p_t": p_t,
            "a_t": a_t,
            "pi_wh_t": pi_wh_t,
            "p_comp_t": p_comp_t,
            "sessions_ours_new": float(sessions_ours_new),
            "sessions_comp_new": float(sessions_comp_new),
            "sessions_lost": float(sessions_lost),
            "active_sessions": len(self.active_sessions),
            "E_delivered_t": E_delivered_t,
            "E_competitor_t": E_competitor_t,
            "shortfall_t": shortfall_t,
            "margin": margin,
            "prob_ours_mixed": prob_ours_mixed,
            "prob_delay": delay_prob,
            "competitor_fraction": self._competitor_fraction,
        }

        return obs, float(R_t), terminated, truncated, info
