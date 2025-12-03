import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from market import estimate_local_volatility, parametric_iv_surface
from utils import black_scholes_metrics, ExperimentLogger

def run_viz():
    logger = ExperimentLogger("03_market_dynamics_viz")

    # 1. Create a Synthetic Regime Shift Scenario
    # Simulate a price path that goes from calm to volatile
    steps = 300
    prices = np.zeros(steps)
    prices[0] = 50000.0
    
    np.random.seed(101)
    for i in range(1, steps):
        # Regime switch at step 150
        vol_regime = 0.3 if i < 150 else 1.2 
        ret = np.random.normal(0, vol_regime * np.sqrt(config.DT_YEAR_5MIN))
        prices[i] = prices[i-1] * np.exp(ret)

    s_series = pd.Series(prices)
    local_vol_series = estimate_local_volatility(s_series, window_hours=1)

    # 2. Slice Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    moneyness_range = np.linspace(0.8, 1.2, 50)
    
    # Analyze two distinct points
    scenarios = [
        {'step': 100, 'name': 'Calm Regime', 'color': 'green'},
        {'step': 250, 'name': 'High Volatility', 'color': 'red'}
    ]

    for sc in scenarios:
        idx = sc['step']
        spot = prices[idx]
        l_vol = local_vol_series.iloc[idx]
        
        # Look at a 7-day option
        T_target = 7.0 / 365.0
        
        iv_curve = []
        option_prices = []

        for m in moneyness_range:
            K = m * spot
            # Get Implied Vol
            iv = parametric_iv_surface(0, spot, K, T_target, l_vol)
            # Get Price
            p, _, _ = black_scholes_metrics(spot, K, T_target, iv, 'call')
            
            iv_curve.append(iv)
            option_prices.append(p)

        # Plot IV Smile
        axes[0].plot(moneyness_range, iv_curve, label=f"{sc['name']} (Base={l_vol:.2f})", color=sc['color'])
        
        # Plot Option Prices
        axes[1].plot(moneyness_range, option_prices, label=f"{sc['name']}", color=sc['color'])

    axes[0].set_title("Implied Volatility Surface (Cross-Section)")
    axes[0].set_xlabel("Moneyness (K/S)")
    axes[0].set_ylabel("IV")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Call Option Pricing")
    axes[1].set_xlabel("Moneyness (K/S)")
    axes[1].set_ylabel("Premium ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    logger.save_figure(fig, "regime_comparison")
    logger.info("Visualization artifacts generated.")

if __name__ == "__main__":
    run_viz()