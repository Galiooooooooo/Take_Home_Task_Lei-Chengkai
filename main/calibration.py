import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize
from pathlib import Path

import config
from simulation import (
    qed_negative_log_likelihood,
    hawkes_nll_joint,
    generate_market_paths
)
from utils import ExperimentLogger, seed_everything

def run_calibration():
    # 1. Setup
    seed_everything(config.GLOBAL_SEED)
    logger = ExperimentLogger("02_calibration")
    
    # 2. Load Data
    input_file = config.DIR_DATA / "btc_5min_preprocessed.parquet"
    if not input_file.exists():
        logger.info(f"[Error] Data file not found at {input_file}. Run analysis script first.")
        return

    df = pd.read_parquet(input_file)
    
    # ==========================================
    # Phase 1: Calibrate QED Diffusion
    # ==========================================
    logger.info("\n>>> Phase 1: Calibrating QED Diffusion Model...")

    # [FIX] Ensure the index is a DatetimeIndex before resampling
    # Even if it looks like a date, strictly convert it to avoid TypeError
    df.index = pd.to_datetime(df.index)

    # Resample to hourly for stable drift estimation
    # Note: We use the index directly now
    df_hourly = df['close'].resample('1h').last().dropna()
    
    X_series = df_hourly.values
    if len(X_series) == 0:
        logger.info("[Error] Hourly resampling resulted in empty data. Check timestamp format.")
        return

    scale_factor = X_series[0]
    X_normalized = X_series / scale_factor
    dt_hourly = 1.0 / (365 * 24)

    # Optimization
    # Initial Guess: [Theta, Kappa, g, Sigma]
    init_guess = [0.1, 0.5, 0.1, 0.8]
    bounds = [(None, None), (None, None), (None, None), (0.01, None)]
    
    result_qed = minimize(
        qed_negative_log_likelihood, 
        init_guess, 
        args=(X_normalized, dt_hourly),
        method='L-BFGS-B',
        bounds=bounds
    )

    p_theta, p_kappa, p_g, p_sigma = result_qed.x
    logger.info(f"QED Parameters: Theta={p_theta:.4f}, Kappa={p_kappa:.4f}, g={p_g:.4f}, Sigma={p_sigma:.4f}")

    # Visualization check
    logger.info("Generating QED fit check...")
    drift_pred = (p_theta * X_normalized[:-1] - p_kappa * X_normalized[:-1]**2 - p_g * X_normalized[:-1]**3) * dt_hourly
    mu_pred = (X_normalized[:-1] + drift_pred) * scale_factor

    fig1 = plt.figure(figsize=(12, 4))
    plt.plot(df_hourly.index[:200], X_series[:200], 'k-', label='Actual Price', lw=1)
    plt.plot(df_hourly.index[1:201], mu_pred[:200], 'b--', label='QED Expected Mean')
    plt.title("QED Model Fit (First 200 Hours)")
    plt.legend()
    logger.save_figure(fig1, "qed_fit_check")

    # ==========================================
    # Phase 2: Jump Detection & Hawkes Calibration
    # ==========================================
    logger.info("\n>>> Phase 2: Calibrating Hawkes Process (Jump Dynamics)...")

    # Detect Jumps on 5-min data
    prices = df['close'].values
    log_prices = np.log(prices)
    returns_5m = np.diff(log_prices)
    
    vol_5m = np.std(returns_5m)
    threshold = 4 * vol_5m
    logger.info(f"Jump Detection Threshold (4-sigma): {threshold:.6f}")

    is_up_jump = returns_5m > threshold
    is_down_jump = returns_5m < -threshold

    jumps_up_vals = returns_5m[is_up_jump]
    jumps_down_vals = -returns_5m[is_down_jump] # Store magnitude

    logger.info(f"Counted Up-Jumps: {len(jumps_up_vals)}")
    logger.info(f"Counted Down-Jumps: {len(jumps_down_vals)}")

    # Hawkes Optimization
    # Params: [Base_P, Base_M, Alpha_P, Alpha_M, Beta]
    init_hawkes = [100.0, 100.0, 2000.0, 2000.0, 4000.0]
    # Bounds: All positive, Alphas must be reasonable relative to Beta
    h_bounds = [(1.0, None)] * 5
    
    result_hawkes = minimize(
        hawkes_nll_joint,
        init_hawkes,
        args=(is_up_jump.astype(int), is_down_jump.astype(int), config.DT_YEAR_5MIN),
        method='L-BFGS-B',
        bounds=h_bounds
    )

    h_params = result_hawkes.x
    logger.info(f"Hawkes Params: {h_params}")
    logger.info(f"Branching Ratio (+): {h_params[2]/h_params[4]:.4f}")
    logger.info(f"Branching Ratio (-): {h_params[3]/h_params[4]:.4f}")

    # ==========================================
    # Phase 3: Validation Simulation
    # ==========================================
    logger.info("\n>>> Phase 3: Validating via Simulation...")

    SIM_DAYS = 14
    N_STEPS = int(SIM_DAYS * 24 * 12)
    N_PATHS = 10
    S_LAST = df['close'].iloc[-1]

    sim_prices = generate_market_paths(
        N_PATHS, N_STEPS, S_LAST,
        (p_theta, p_kappa, p_g, p_sigma),
        tuple(h_params),
        (jumps_up_vals, jumps_down_vals),
        config.DT_YEAR_5MIN,
        scale_factor
    )

    # Plot Simulation vs Reality
    fig2 = plt.figure(figsize=(14, 6))
    # Ensure we don't plot more than we have history for
    hist_len = min(len(df), N_STEPS)
    plt.plot(df['close'].iloc[-hist_len:].values, 'k', lw=2, label='History (Last 14d)', alpha=0.3)
    for i in range(N_PATHS):
        plt.plot(sim_prices[:, i], alpha=0.6, lw=1)
    plt.title("Synthetic Paths vs Historical Context")
    plt.legend()
    logger.save_figure(fig2, "simulation_validation")

    # ==========================================
    # Phase 4: Save Model
    # ==========================================
    model_payload = {
        "qed_params": (p_theta, p_kappa, p_g, p_sigma),
        "hawkes_params": tuple(h_params),
        "jump_distributions": (jumps_up_vals, jumps_down_vals),
        "meta": {
            "dt_year": config.DT_YEAR_5MIN,
            "ref_price": S_LAST,
            "scale_factor": scale_factor
        }
    }

    out_path = config.DIR_MODELS / "calibrated_market_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_payload, f)
    
    logger.info(f"\nCalibration complete. Model saved to {out_path}")

if __name__ == "__main__":
    run_calibration()