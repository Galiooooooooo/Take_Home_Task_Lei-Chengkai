import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from utils import black_scholes_metrics

# ==========================================
# 1. Volatility Surface Utilities
# ==========================================
def estimate_local_volatility(price_series: pd.Series, window_hours: int = 6, dt_min: int = 5) -> pd.Series:
    """Calculates rolling annualized volatility."""
    window_steps = int(window_hours * 60 / dt_min)
    log_returns = np.log(price_series / price_series.shift(1))
    
    rolling_std = log_returns.rolling(window=window_steps).std()
    
    # Annualize
    steps_per_year = (60 / dt_min) * 24 * 365
    local_vol = rolling_std * np.sqrt(steps_per_year)
    
    return local_vol.bfill().fillna(0.5)

def parametric_iv_surface(t: float, S: float, K: float, T: float, local_vol: float) -> float:
    """
    Constructs a stylized IV surface based on moneyness and local volatility.
    """
    time_to_maturity = max(T - t, 0.0)
    moneyness = np.log(K / S)
    
    # Clip base volatility to realistic crypto bounds
    base_vol = np.clip(local_vol, 0.20, 3.0)
    
    # Simple smile curve: IV increases as we move away from ATM
    smile_impact = 1.0 + 2.0 * (moneyness ** 2)
    
    return base_vol * smile_impact

# ==========================================
# 2. Option Portfolio Manager
# ==========================================
class OptionBook:
    """
    Manages a portfolio of options, calculating valuations and risk metrics (Greeks).
    """
    def __init__(self, spot_ref: float, start_time: float = 0.0):
        self.contracts: List[Dict] = []
        
        # Portfolio Definition: 12 contracts
        # Strikes relative to S0: 90%, 100%, 110%
        # Maturities: 1 day, 7 days
        self.strikes_pct = [0.9, 1.0, 1.1]
        self.maturities_days = [1.0, 7.0]

        idx = 0
        for days in self.maturities_days:
            T_abs = start_time + days / 365.0
            for k_pct in self.strikes_pct:
                K = k_pct * spot_ref
                for opt_type in ['call', 'put']:
                    self.contracts.append({
                        'id': idx,
                        'name': f"{opt_type}_{days}d_{k_pct}S",
                        'type': opt_type,
                        'K': K,
                        'T_abs': T_abs,
                        'position': 0.0,
                        'price': 0.0,
                        'delta': 0.0,
                        'vega': 0.0
                    })
                    idx += 1
        
    def update_valuations(self, t: float, S: float, iv_function: Callable, local_vol: float):
        """Re-prices all options based on current market conditions."""
        portfolio_value = 0.0
        portfolio_delta = 0.0
        portfolio_vega = 0.0

        for c in self.contracts:
            tau = max(c['T_abs'] - t, 0.0)
            
            # Get IV
            if tau < 1e-6:
                iv = 0.0
            else:
                iv = iv_function(t, S, c['K'], c['T_abs'], local_vol)

            # Pricing
            price, delta, vega = black_scholes_metrics(S, c['K'], tau, iv, c['type'])

            # Update Contract State
            c['price'] = price
            c['delta'] = delta
            c['vega'] = vega

            # Aggregates
            portfolio_value += c['position'] * price
            portfolio_delta += c['position'] * delta
            portfolio_vega += c['position'] * vega

        return portfolio_value, portfolio_delta, portfolio_vega

    def process_order(self, action_id: int, quantity: float, cost_rate: float) -> float:
        """
        Executes a trade based on discrete action ID.
        Map: 0=Hold, 1-12=Buy, 13-24=Sell
        """
        if action_id == 0:
            return 0.0

        is_buy = (action_id <= 12)
        # Adjust index to 0-11 range
        contract_idx = (action_id - 1) if is_buy else (action_id - 13)

        contract = self.contracts[contract_idx]
        price = contract['price']
        
        notional_value = price * quantity
        txn_cost = notional_value * cost_rate

        if is_buy:
            contract['position'] += quantity
        else:
            contract['position'] -= quantity

        return txn_cost