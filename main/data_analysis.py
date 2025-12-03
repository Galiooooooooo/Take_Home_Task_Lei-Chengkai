import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import config
from utils import ExperimentLogger

def run_eda():
    logger = ExperimentLogger("01_data_analysis")
    
    file_path = config.DIR_DATA / "BTC_5m.csv"
    logger.info(f"Reading dataset from: {file_path}")

    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]

        # Timestamp parsing
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        logger.info(f"Successfully loaded {len(df)} records.")

    except FileNotFoundError:
        logger.info(f"Warning: {file_path} not found. Creating synthetic random walk for demonstration.")
        dates = pd.date_range(start='2024-01-01', periods=5000, freq='5min')
        df = pd.DataFrame({'close': 10000 * np.exp(np.cumsum(np.random.normal(0, 0.001, 5000)))}, index=dates)

    # 2. Feature Engineering
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    # 3. Visualization: Price & Returns
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1.plot(df.index, df['close'], color='#2c3e50', label='Close Price')
    ax1.set_title('BTC 5-Minute Price Action')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df.index, df['log_returns'], color='#2980b9', alpha=0.7, lw=0.5)
    ax2.set_title('Logarithmic Returns')
    ax2.grid(True, alpha=0.3)
    
    logger.save_figure(fig1, "time_series_overview")

    # 4. Statistical Summary
    returns = df['log_returns'].values
    mu, std = np.mean(returns), np.std(returns)
    skewness = stats.skew(returns)
    kurt = stats.kurtosis(returns)

    logger.info("-" * 40)
    logger.info(f"Statistics for 5-min Log Returns:")
    logger.info(f"Mean:      {mu:.8f}")
    logger.info(f"Std Dev:   {std:.8f}")
    logger.info(f"Skewness:  {skewness:.4f}")
    logger.info(f"Kurtosis:  {kurt:.4f} (Excess)")

    # 5. Distribution Analysis (Fat Tails check)
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram vs Normal PDF
    axes[0].hist(returns, bins=100, density=True, alpha=0.6, color='#e74c3c', label='Empirical Data')
    x_domain = np.linspace(returns.min(), returns.max(), 1000)
    pdf_normal = stats.norm.pdf(x_domain, mu, std)
    axes[0].plot(x_domain, pdf_normal, 'k--', linewidth=2, label='Normal Dist')
    axes[0].set_yscale('log')
    axes[0].set_title('Return Distribution (Log-Scale Y)')
    axes[0].legend()

    # Q-Q Plot
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot vs Normal')
    
    logger.save_figure(fig2, "distribution_analysis")

    # 6. Save Artifacts
    output_path = config.DIR_DATA / "btc_5min_preprocessed.parquet"
    df.to_parquet(output_path)
    logger.info(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    run_eda()