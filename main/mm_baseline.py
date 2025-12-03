import numpy as np
import matplotlib.pyplot as plt
import pickle
import config
from simulation import generate_market_paths, construct_ohlc_candles
from utils import ExperimentLogger, seed_everything

# Strategy Hyperparameters (Avellaneda-Stoikov style)
STRATEGY_CONFIG = {
    'spread_base': 0.0005,      # s0
    'order_qty': 1.0,           # q
    'risk_factor': 0.01,        # gamma/k
    'max_inventory': 10.0,
    'stop_loss': -50000.0,
    'initial_capital': 100000.0
}

def execute_inventory_strategy(mid_prices, highs, lows, params):
    """
    Simulates a high-frequency market making strategy on historical/synthetic data.
    """
    n_steps = len(mid_prices)
    
    # Unpack parameters
    s0 = params['spread_base']
    qty = params['order_qty']
    gamma = params['risk_factor']
    limit_inv = params['max_inventory']
    stop_loss = params['stop_loss']
    
    # State arrays
    inventory = np.zeros(n_steps)
    cash = np.zeros(n_steps)
    equity = np.zeros(n_steps)
    
    # Initialization
    cash[0] = params['initial_capital']
    equity[0] = cash[0] # Inventory is 0
    
    # Execution Stats
    fills_bid = np.zeros(n_steps)
    fills_ask = np.zeros(n_steps)

    for t in range(n_steps - 1):
        mid_t = mid_prices[t]
        inv_t = inventory[t]
        cash_t = cash[t]

        # 1. Quote Calculation
        # Inventory Skew: Adjust quotes based on current holding
        inv_ratio = inv_t / limit_inv
        skew_adjust = gamma * inv_ratio
        
        bid_px = mid_t * (1 - s0 - skew_adjust)
        ask_px = mid_t * (1 + s0 + skew_adjust)

        # 2. Order Sizing logic (Stop quoting if full)
        q_bid = qty * max(0.0, 1.0 - inv_ratio) if inv_t < limit_inv else 0.0
        q_ask = qty * max(0.0, 1.0 + inv_ratio) if inv_t > -limit_inv else 0.0

        # 3. Market Interaction (Next candle determines fills)
        high_next = highs[t + 1]
        low_next = lows[t + 1]

        # Limit order logic: 
        # Buy if market Low drops below our Bid
        # Sell if market High rises above our Ask
        is_bid_filled = (low_next <= bid_px)
        is_ask_filled = (high_next >= ask_px)

        # 4. Update State
        inv_next = inv_t
        cash_next = cash_t

        if is_bid_filled and q_bid > 0:
            inv_next += q_bid
            cash_next -= q_bid * bid_px
            fills_bid[t] = q_bid

        if is_ask_filled and q_ask > 0:
            inv_next -= q_ask
            cash_next += q_ask * ask_px
            fills_ask[t] = q_ask

        # 5. Mark-to-Market
        inventory[t + 1] = inv_next
        cash[t + 1] = cash_next
        equity[t + 1] = cash_next + inv_next * mid_prices[t + 1]

        # Stop Loss Check
        pnl = equity[t + 1] - params['initial_capital']
        if pnl < stop_loss:
            # Flatten everything (simplified)
            inventory[t+1:] = inv_next
            equity[t+1:] = equity[t+1]
            break

    return inventory, equity

def run_baseline():
    seed_everything(config.GLOBAL_SEED)
    logger = ExperimentLogger("04_baseline_benchmark")

    # 1. Load Market Model Parameters
    param_file = config.DIR_MODELS / "calibrated_market_model.pkl"
    try:
        with open(param_file, "rb") as f:
            market_model = pickle.load(f)
        logger.info("Market parameters loaded successfully.")
    except FileNotFoundError:
        logger.info("[Error] Run calibration (step 2) first.")
        return

    # 2. Generate Evaluation Scenarios
    N_PATHS = 50
    DAYS = 14
    STEPS = int(DAYS * 24 * 12) # 5-min steps

    logger.info(f"Generating {N_PATHS} synthetic paths for benchmarking...")
    
    sim_mids = generate_market_paths(
        N_PATHS, STEPS, 
        market_model['meta']['ref_price'],
        market_model['qed_params'],
        market_model['hawkes_params'],
        market_model['jump_distributions'],
        config.DT_YEAR_5MIN,
        market_model['meta']['scale_factor']
    )
    sim_highs, sim_lows = construct_ohlc_candles(sim_mids)

    # Save for RL training consistency
    data_cache = config.DIR_DATA / "training_scenarios.npz"
    np.savez(data_cache, mids=sim_mids, highs=sim_highs, lows=sim_lows)
    logger.info(f"Scenarios saved to {data_cache}")

    # 3. Execute Baseline Strategy
    logger.info("Running Avellaneda-Stoikov Baseline...")
    
    all_equity = []
    all_inventory = []

    for i in range(N_PATHS):
        inv_path, eq_path = execute_inventory_strategy(
            sim_mids[:, i], sim_highs[:, i], sim_lows[:, i], STRATEGY_CONFIG
        )
        all_equity.append(eq_path)
        all_inventory.append(inv_path)

    # 4. Analysis & Visualization
    equity_matrix = np.array(all_equity).T
    pnl_matrix = equity_matrix - STRATEGY_CONFIG['initial_capital']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # PnL Paths
    axes[0].plot(pnl_matrix, color='gray', alpha=0.3, linewidth=0.5)
    axes[0].plot(np.mean(pnl_matrix, axis=1), color='blue', linewidth=2, label='Mean PnL')
    axes[0].set_title(f"Baseline PnL over {DAYS} Days ({N_PATHS} Scenarios)")
    axes[0].set_ylabel("PnL ($)")
    axes[0].legend()

    # Inventory Heatmap/Lines
    axes[1].plot(np.array(all_inventory).T, color='purple', alpha=0.1)
    axes[1].axhline(STRATEGY_CONFIG['max_inventory'], color='r', linestyle='--')
    axes[1].axhline(-STRATEGY_CONFIG['max_inventory'], color='r', linestyle='--')
    axes[1].set_title("Inventory Exposure")
    axes[1].set_ylabel("Contracts")

    # Final PnL Distribution
    final_pnls = pnl_matrix[-1, :]
    axes[2].hist(final_pnls, bins=20, color='#3498db', edgecolor='black', alpha=0.8)
    axes[2].axvline(np.mean(final_pnls), color='k', linestyle='dashed', linewidth=1)
    axes[2].set_title(f"Terminal PnL Distribution (Mean: ${np.mean(final_pnls):.2f})")

    plt.tight_layout()
    logger.save_figure(fig, "baseline_performance")

    # Metrics
    metrics = {
        "mean_pnl": float(np.mean(final_pnls)),
        "std_pnl": float(np.std(final_pnls)),
        "sharpe_proxy": float(np.mean(final_pnls) / (np.std(final_pnls) + 1e-5)),
        "win_rate": float(np.sum(final_pnls > 0) / N_PATHS)
    }
    logger.dump_metrics(metrics)

if __name__ == "__main__":
    run_baseline()