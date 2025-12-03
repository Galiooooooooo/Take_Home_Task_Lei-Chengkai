import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

import config
from envs import ContinuousDeltaHedgingEnv
from utils import ExperimentLogger, seed_everything

# ===========================
# Policy Definitions
# ===========================
def policy_no_hedge(obs):
    return 0.0

def policy_bs_delta(obs):
    # Obs index 3 is explicitly the BS Delta in our environment
    return obs[3]

def policy_rl_agent(obs, model):
    action, _ = model.predict(obs, deterministic=True)
    return action

# ===========================
# Main Evaluation Logic
# ===========================
def run_backtest():
    seed_everything(config.GLOBAL_SEED)
    logger = ExperimentLogger("06_final_evaluation")

    # 1. Prepare Test Data (Out-of-Sample)
    test_file = config.DIR_DATA / "test_paths_oos.npz"
    
    if test_file.exists():
        logger.info(f"Loading existing test set: {test_file.name}")
        test_paths = np.load(test_file)['mids']
    else:
        logger.info("Generating NEW Out-of-Sample test set (500 paths)...")
        # Generate new random paths (different seed implicit by invocation)
        rng = np.random.default_rng(config.GLOBAL_SEED + 999)
        test_paths = 10000 * np.exp(np.cumsum(rng.normal(0, 0.001, (500, 4000)), axis=1))
        np.savez(test_file, mids=test_paths)
    
    logger.info(f"Test Set Shape: {test_paths.shape}")

    # 2. Load Agent
    model_path = config.DIR_MODELS / "ppo_agent" / "best_model.zip"
    if not model_path.exists():
        logger.info("RL Model not found. Skipping RL evaluation.")
        return
    
    agent = PPO.load(model_path)
    env = ContinuousDeltaHedgingEnv(test_paths, cost_bps=config.COST_BPS)

    # 3. Run Comparison Loop
    N_EPISODES = len(test_paths)
    results = {"Unhedged": [], "BS_Delta": [], "RL_Agent": []}

    logger.info(f"Backtesting strategies on {N_EPISODES} episodes...")

    for i in range(N_EPISODES):
        # --- Strategy A: Unhedged ---
        obs, _ = env.reset(options={'path_idx': i})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy_no_hedge(obs))
            done = term or trunc
            if done: results["Unhedged"].append(info['total_value'])

        # --- Strategy B: Classical BS ---
        obs, _ = env.reset(options={'path_idx': i})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy_bs_delta(obs))
            done = term or trunc
            if done: results["BS_Delta"].append(info['total_value'])

        # --- Strategy C: RL Agent ---
        obs, _ = env.reset(options={'path_idx': i})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy_rl_agent(obs, agent))
            done = term or trunc
            if done: results["RL_Agent"].append(info['total_value'])

        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{N_EPISODES}...")

    # 4. Metric Calculation
    df_res = pd.DataFrame(results)
    
    std_unhedged = df_res['Unhedged'].std()
    std_rl = df_res['RL_Agent'].std()
    risk_reduction = (std_unhedged - std_rl) / std_unhedged

    metrics = {
        "count": N_EPISODES,
        "mean_pnl": df_res.mean().to_dict(),
        "std_pnl": df_res.std().to_dict(),
        "risk_reduction_ratio": risk_reduction,
        "raw_stats": df_res.describe().to_dict()
    }
    
    logger.info("\n=== Final Results ===")
    logger.info(f"Risk Reduction (RL vs Unhedged): {risk_reduction*100:.2f}%")
    logger.info(f"RL Std Dev: {std_rl:.2f} | BS Std Dev: {df_res['BS_Delta'].std():.2f}")
    
    logger.dump_metrics(metrics)

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # KDE Plot
    sns.kdeplot(data=df_res, x="Unhedged", ax=axes[0], fill=True, label='Unhedged', alpha=0.3)
    sns.kdeplot(data=df_res, x="RL_Agent", ax=axes[0], fill=True, label='RL Agent', alpha=0.3)
    axes[0].set_title("PnL Distribution Density")
    axes[0].set_xlabel("Final PnL ($)")
    axes[0].legend()

    # Box Plot
    sns.boxplot(data=df_res, ax=axes[1], palette="Set2")
    axes[1].set_title("PnL Spread Comparison")
    axes[1].set_ylabel("PnL ($)")
    axes[1].grid(True, alpha=0.3)

    logger.save_figure(fig, "strategy_comparison_final")

if __name__ == "__main__":
    run_backtest()