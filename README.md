# Take_Home_Task_Lei-Chengkai
RL Agent for Hedging BTC Perpetual Inventory - Quant Finance Assignment

## 📖 Project Overview

This project implements a comparative analysis between **Traditional Market Making** strategies (Avellaneda-Stoikov) and **Deep Reinforcement Learning (PPO)** for inventory management and delta hedging in the high-frequency Bitcoin (BTC) market.

It features a complete quantitative pipeline including:
1.  **Stochastic Modeling:** Calibrating QED (Quadratic Exponential Diffusion) for volatility and Hawkes Processes for jump clustering.
2.  **Market Simulation:** A realistic limit order book environment with transaction costs and slippage.
3.  **Reinforcement Learning:** Training a PPO agent using Stable-Baselines3.
4.  **Backtesting:** Out-of-sample performance comparison on risk-adjusted returns (Sharpe Ratio) and PnL distribution.

---
## 📂 Directory Structure

```text
project_root/
├── data/                # Data storage (Input CSV & Intermediate Parquet/NPZ)
│   ├── BTC_5m.csv       # Raw input data
│   └── ...
├── main/                # Core Library & Executable Scripts
│   ├── config.py        # Global configuration
│   ├── market.py        # Financial logic (Option pricing, Order Book)
│   ├── simulation.py    # Stochastic math (QED, Hawkes)
│   ├── envs.py          # Gymnasium RL Environments
│   ├── utils.py         # Helper functions
│   ├── data_analysis.py # [Script] EDA
│   ├── calibration.py   # [Script] Model Calibration
│   ├── mm_baseline.py   # [Script] Avellaneda-Stoikov Baseline
│   ├── rl_training.py   # [Script] PPO Training
│   └── rl_evaluation.py # [Script] Final Backtest
├── models/              # Saved Model Weights (.pkl, .zip)
├── notebooks/           # Interactive Jupyter Notebooks (for experimentation)
├── results/             # Output Plots, Logs, and Metrics
└── requirements.txt     # Python Dependencies
```
## ⚡ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd btc_hedging_strategy
2. **Create a virtual environment (Recommended):**
   ```bash
   conda create -n btc_rl python=3.9
   conda activate btc_rl
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## 🚀 Usage: Script Mode (Production Pipeline)

Run the pipeline steps in the following order. All artifacts (models, data, plots) will be saved automatically.

### 1. Data Preprocessing
Clean raw data and compute statistical features.
```bash
python main/data_analysis.py

### 2. Model Calibration
Calibrate QED and Hawkes parameters to historical data.
```bash
python main/calibration.py

### 3. Baseline Benchmark
Run the Avellaneda-Stoikov strategy to generate training scenarios and establish a baseline.
```bash
python main/mm_baseline.py

### 4. Train RL Agent
Train the PPO agent on 100,000 timesteps (uses CPU/GPU automatically).
```bash
python main/rl_training.py

### 5. Final Evaluation
Perform Out-of-Sample backtesting comparing Unhedged, Baseline, and RL strategies.
```bash
python main/rl_evaluation.py

## 📓 Usage: Notebook Mode (Interactive Research)

For visualization, debugging, and step-by-step analysis, use the notebooks located in `notebooks/`.

> **Important:** The notebooks import modules from the `main/` directory. Ensure you run Jupyter from the project root or add `main/` to your PYTHONPATH.

1.  **`01_data_and_stylized_facts.ipynb`**: Deep dive into BTC returns, volatility clustering, and fat tails.

2.  **`02_model_calibration.ipynb`**: Interactive calibration of QED/Hawkes with diagnostic plots.

3.  **`03_option_market_layer.ipynb`**: Visualizing the Implied Volatility Surface and option pricing logic.

4.  **`04_baseline_mm_strategy.ipynb`**: Detailed breakdown of the Avellaneda-Stoikov inventory behavior.

5.  **`05_rl_training_ppo.ipynb`**: Monitor RL training progress and view single-episode sanity checks.

6.  **`06_strategy_backtest.ipynb`**: Detailed performance reporting and distribution analysis.

7.  **`07_rl_vs_baseline_evaluation.ipynb`**: The "Grand Finale" comparison on test data.

---

## 🧠 Methodology

* **Simulation Engine:** We do not use simple Geometric Brownian Motion. Instead, we combine a **QED** (Quadratic Exponential Diffusion) process for stochastic volatility with a **Hawkes Process** to simulate realistic market jumps and flash crashes.
* **Hedging Logic:** The agent is trained in a continuous action space ($\in [0, 1]$ hedge ratio) environment. It receives rewards for minimizing PnL variance while avoiding excessive transaction costs.

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Cryptocurrency markets are highly volatile; use these strategies at your own risk.

---
**Author:** [Lei Chengkai/25214020009]
**Last Updated:** 2025-12-3
