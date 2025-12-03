import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from envs import ContinuousDeltaHedgingEnv
from utils import ExperimentLogger, seed_everything

# Training Hyperparameters
RL_CONFIG = {
    'total_timesteps': 100_000,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'ent_coef': 0.00,
    'device': "auto" # Will use CUDA/MPS if available
}

def get_training_data(logger):
    """Loads simulated market paths or generates fallback data."""
    data_path = config.DIR_DATA / "training_scenarios.npz"
    try:
        data = np.load(data_path)
        # Transpose to (n_paths, n_steps) for consistency
        paths = data['mids'].T 
        logger.info(f"Loaded training data: {paths.shape}")
        return paths
    except FileNotFoundError:
        logger.info("Training data not found. Generating simple GBM fallback.")
        # Fallback: simple GBM
        return 10000 * np.exp(np.cumsum(np.random.normal(0, 0.001, (50, 4000)), axis=1))

def train_agent():
    seed_everything(config.GLOBAL_SEED)
    logger = ExperimentLogger("05_rl_training")

    # 1. Prepare Environment
    market_paths = get_training_data(logger)
    
    # Wrap in DummyVecEnv for SB3 compatibility
    env_creator = lambda: ContinuousDeltaHedgingEnv(market_paths)
    vec_env = DummyVecEnv([env_creator])

    # 2. Initialize PPO Agent
    model_dir = config.DIR_MODELS / "ppo_agent"
    model_dir.mkdir(exist_ok=True)
    
    logger.info(f"Initializing PPO on device: {RL_CONFIG['device']}")
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=RL_CONFIG['learning_rate'],
        batch_size=RL_CONFIG['batch_size'],
        n_epochs=RL_CONFIG['n_epochs'],
        gamma=RL_CONFIG['gamma'],
        device=RL_CONFIG['device'],
        seed=config.GLOBAL_SEED
    )

    # 3. Training Loop
    logger.info(f"Starting training for {RL_CONFIG['total_timesteps']} steps...")
    model.learn(total_timesteps=RL_CONFIG['total_timesteps'])

    # 4. Save Artifacts
    save_path = model_dir / "best_model"
    model.save(save_path)
    logger.info(f"Model saved to {save_path}.zip")

def sanity_check():
    """Runs a quick single-episode check to verify the agent learned something."""
    logger = ExperimentLogger("05_rl_sanity_check")
    
    # Load
    model_path = config.DIR_MODELS / "ppo_agent" / "best_model.zip"
    if not model_path.exists():
        logger.info("No model found. Train first.")
        return

    model = PPO.load(model_path)
    market_paths = get_training_data(logger)
    
    # Run 1 episode
    env = ContinuousDeltaHedgingEnv(market_paths)
    obs, _ = env.reset(options={'path_idx': 0})
    
    history = {'spot': [], 'hedge': [], 'bs_delta': [], 'pnl': []}
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        
        # Log (using info dict from env)
        history['spot'].append(env.S_path[env.t_step-1])
        history['hedge'].append(info['current_hedge'])
        history['bs_delta'].append(info['ideal_bs_delta'])
        history['pnl'].append(info['total_value'])

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(history['spot'], 'k', label='Spot Price')
    ax1.set_title("Training Scenario #0")
    ax1.legend()
    
    ax2.plot(history['bs_delta'], 'g--', label='Black-Scholes Delta (Target)')
    ax2.plot(history['hedge'], 'b', lw=2, label='RL Agent Action')
    ax2.set_title("Hedging Behavior Check")
    ax2.set_ylabel("Hedge Ratio")
    ax2.legend()
    
    logger.save_figure(fig, "post_train_behavior")
    
    mse = np.mean((np.array(history['hedge']) - np.array(history['bs_delta']))**2)
    logger.info(f"Agent vs BS-Delta MSE: {mse:.5f}")

if __name__ == "__main__":
    train_agent()
    sanity_check()