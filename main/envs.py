import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple

from market import OptionBook
from utils import black_scholes_metrics
import config

# ==========================================
# 1. Discrete Control Environment (Market Making)
# ==========================================
class DiscreteMarketMakingEnv(gym.Env):
    """
    RL Environment for a Market Maker managing an option inventory.
    Action Space: Discrete (Buy/Sell specific contracts or Hold).
    """
    def __init__(self, env_config: Dict):
        super().__init__()
        
        self.cfg = env_config
        self.initial_spot = env_config.get('S0', 100000.0)
        self.max_inventory = env_config.get('I_max', 10.0)
        self.cost_rate = env_config.get('cost_rate', 0.0005)
        self.lot_size = env_config.get('opt_lot_size', 1.0)
        self.hedge_interval = env_config.get('hedge_interval', 12)  # Steps to simulate between RL actions

        # Risk Aversion Parameters
        self.lambda_delta = env_config.get('lambda_delta', 0.1)
        self.lambda_vega = env_config.get('lambda_vega', 0.01)

        # Action Space: 0=NoOp, 1-12=Buy, 13-24=Sell (Total 25 actions)
        self.action_space = spaces.Discrete(25)
        
        # Observation: [Norm_Spot, Norm_Inventory, Time, Delta, Vega, Volatility]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.iv_model = env_config.get('iv_func', lambda t, s, k, T, vol: max(0.2, vol))
        self.option_book: Optional[OptionBook] = None
        self.current_step = 0
        self.mm_inventory = 0.0
        self.cash_balance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Load path data
        path_pool = self.cfg.get('path_pool')
        if path_pool is not None:
            n_available = path_pool['mids'].shape[1]
            idx = np.random.randint(0, n_available)
            self.spot_path = path_pool['mids'][:, idx]
            self.high_path = path_pool['highs'][:, idx]
            self.low_path = path_pool['lows'][:, idx]
        else:
            # Fallback for testing without data
            steps = 4032
            self.spot_path = np.linspace(self.initial_spot, self.initial_spot, steps)
            self.high_path = self.spot_path * 1.001
            self.low_path = self.spot_path * 0.999

        self.max_steps = len(self.spot_path)
        self.current_step = 0
        self.mm_inventory = 0.0
        self.cash_balance = 0.0
        
        # Initialize Order Book
        self.option_book = OptionBook(self.initial_spot, start_time=0.0)

        return self._get_obs(), {}

    def step(self, action: int):
        # 1. Execute Agent Action (Option Trade)
        if self.option_book is None:
            raise RuntimeError("Environment must be reset before stepping.")

        transaction_cost = self.option_book.process_order(action, self.lot_size, self.cost_rate)

        # 2. Simulate Market Dynamics (High-Freq Loop)
        # The agent acts every `hedge_interval` steps. We simulate the MM logic in between.
        start_t = self.current_step
        end_t = min(self.current_step + self.hedge_interval, self.max_steps - 1)
        
        equity_before = self._calculate_equity(self.spot_path[start_t])

        # --- Sub-period Simulation (Avellaneda-Stoikov Logic) ---
        for t in range(start_t, end_t):
            S_t = self.spot_path[t]
            H_next = self.high_path[t+1]
            L_next = self.low_path[t+1]

            # Inventory skew
            q_norm = self.mm_inventory / self.max_inventory
            spread = 0.0005
            skew = 0.01 * q_norm

            bid_px = S_t * (1 - spread - skew)
            ask_px = S_t * (1 + spread + skew)

            # Limit Order Execution Logic
            buy_fill = (L_next <= bid_px)
            sell_fill = (H_next >= ask_px)
            qty = 1.0 # Standard MM unit

            if buy_fill:
                self.mm_inventory += qty
                self.cash_balance -= qty * bid_px
            if sell_fill:
                self.mm_inventory -= qty
                self.cash_balance += qty * ask_px
        # --------------------------------------------------------

        self.current_step = end_t
        current_S = self.spot_path[self.current_step]
        current_time = self.current_step * config.DT_YEAR_5MIN

        # 3. Update Option Book Valuations
        current_vol = 0.5 # Simplified constant vol for this env
        port_val, net_delta, net_vega = self.option_book.update_valuations(
            current_time, current_S, self.iv_model, current_vol
        )

        # 4. Calculate Reward
        equity_after = self._calculate_equity(current_S)
        pnl_period = equity_after - equity_before
        
        # Risk Penalties
        exposure_delta = self.mm_inventory + net_delta
        penalty = self.lambda_delta * (exposure_delta ** 2) + self.lambda_vega * (net_vega ** 2)
        
        reward = pnl_period - transaction_cost - penalty

        terminated = (self.current_step >= self.max_steps - 1)
        truncated = False
        
        info = {
            "pnl": pnl_period,
            "cost": transaction_cost,
            "net_delta": exposure_delta,
            "inventory": self.mm_inventory
        }

        return self._get_obs(current_S, exposure_delta, net_vega, current_vol, current_time), reward, terminated, truncated, info

    def _calculate_equity(self, spot_price):
        options_value = sum([c['position'] * c['price'] for c in self.option_book.contracts])
        return self.cash_balance + self.mm_inventory * spot_price + options_value

    def _get_obs(self, S=None, delta=0.0, vega=0.0, vol=0.5, t=0.0):
        if S is None: S = self.initial_spot
        return np.array([
            S / self.initial_spot,
            self.mm_inventory / self.max_inventory,
            t,
            delta,
            vega,
            vol
        ], dtype=np.float32)


# ==========================================
# 2. Continuous Control Environment (Delta Hedging)
# ==========================================
class ContinuousDeltaHedgingEnv(gym.Env):
    """
    Simplified environment for learning Delta Hedging strategies.
    Action: Target Hedge Ratio (0.0 to 1.0).
    """
    def __init__(self, price_paths: np.ndarray, maturity_days: int = 14, dt_min: int = 5, cost_bps: float = 0.0005):
        super().__init__()
        self.paths = price_paths.astype(np.float32)
        self.n_paths, self.n_steps_total = self.paths.shape
        self.dt = dt_min / (365 * 24 * 60)
        
        self.strike_pct = 1.0
        self.maturity_days = maturity_days
        self.cost_factor = cost_bps
        
        # Action: Desired hedge ratio (0% to 100%)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: [LogMoneyness, TimeToMat, CurrentPos, BS_Delta]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'path_idx' in options:
            self.path_idx = options['path_idx']
        else:
            self.path_idx = np.random.randint(0, self.n_paths)

        self.S_path = self.paths[self.path_idx]
        self.t_step = 0
        self.tau = self.maturity_days / 365.0
        
        self.S0 = float(self.S_path[0])
        self.K = self.S0 * self.strike_pct
        
        self.hedge_ratio = 0.0
        self.cash = 0.0

        # Initial Portfolio Value
        # Portfolio = Short Option + Hedge Asset + Cash
        init_opt_price, _, _ = black_scholes_metrics(self.S0, self.K, self.tau, config.FIXED_VOLATILITY)
        self.pf_value = -init_opt_price + self.hedge_ratio * self.S0 + self.cash
        self.prev_pf_value = self.pf_value
        
        return self._construct_obs(), {}

    def _construct_obs(self):
        S = float(self.S_path[self.t_step])
        log_moneyness = float(np.log(S / (self.K + 1e-9)))
        norm_tau = float(self.tau / (self.maturity_days / 365.0))
        
        _, bs_delta, _ = black_scholes_metrics(S, self.K, max(0, self.tau), config.FIXED_VOLATILITY)
        
        # Safety check for infinity
        obs = [log_moneyness, norm_tau, float(self.hedge_ratio), float(bs_delta)]
        return np.array([x if np.isfinite(x) else 0.0 for x in obs], dtype=np.float32)

    def step(self, action):
        S_t = float(self.S_path[self.t_step])
        
        # Action Processing
        raw_action = float(action.item()) if hasattr(action, 'item') else float(action)
        target_ratio = np.clip(raw_action, 0.0, 1.0)

        # Rebalancing
        trade_size = target_ratio - self.hedge_ratio
        txn_cost = abs(trade_size) * S_t * self.cost_factor
        
        self.cash -= (trade_size * S_t + txn_cost)
        self.hedge_ratio = target_ratio

        # Time Step
        self.t_step += 1
        self.tau -= self.dt

        # Termination Check
        terminated = bool(self.t_step >= self.n_steps_total - 1)
        truncated = bool(self.tau <= 1e-7)
        
        S_next = float(self.S_path[self.t_step]) if not terminated else S_t

        # PnL Calculation
        opt_price_next, bs_delta_next, _ = black_scholes_metrics(
            S_next, self.K, max(0, self.tau), config.FIXED_VOLATILITY
        )
        
        current_pf_val = -opt_price_next + self.hedge_ratio * S_next + self.cash
        period_pnl = current_pf_val - self.prev_pf_value
        self.prev_pf_value = current_pf_val

        # Reward Engineering (Squared PnL Minimization + Cost Penalty)
        # Normalized by initial price to keep reward scale consistent
        norm_pnl = (period_pnl / self.S0) * 100
        norm_cost = (txn_cost / self.S0) * 100
        
        reward = - (norm_pnl ** 2) - 0.5 * norm_cost

        info = {
            "step_pnl": float(period_pnl),
            "total_value": float(current_pf_val),
            "current_hedge": float(self.hedge_ratio),
            "ideal_bs_delta": float(bs_delta_next)
        }
        
        return self._construct_obs(), float(reward), terminated, truncated, info