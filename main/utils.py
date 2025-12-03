import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from typing import Tuple, Dict, Any, Union

import config

# ==========================================
# 1. Reproducibility
# ==========================================
def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch 
    to ensure reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Torch not installed or not needed

    print(f"[System] Global seed set to: {seed}")


# ==========================================
# 2. Financial Math (Black-Scholes)
# ==========================================
def black_scholes_metrics(
    S: Union[float, np.ndarray], 
    K: Union[float, np.ndarray], 
    T: Union[float, np.ndarray], 
    sigma: float, 
    option_type: str = 'call'
) -> Tuple[Any, Any, Any]:
    """
    Calculates European option price and Greeks (Delta, Vega) 
    using the Black-Scholes-Merton formula.
    """
    # Handle scalar inputs wrapped in arrays
    if isinstance(S, np.ndarray) and S.size == 1: S = S.item()
    if isinstance(K, np.ndarray) and K.size == 1: K = K.item()
    if isinstance(T, np.ndarray) and T.size == 1: T = T.item()

    # Handle expiration or near-expiration
    if T <= 1e-7:
        intrinsic_val = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return intrinsic_val, delta, 0.0

    sigma = max(sigma, 1e-4) # Avoid division by zero

    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0

    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    return price, delta, vega


# ==========================================
# 3. Experiment Logging
# ==========================================
class ExperimentLogger:
    """
    Handles logging of console output, saving of plots, 
    and serializing metrics to JSON.
    """
    def __init__(self, context_name: str):
        self.context_name = context_name
        self.log_file = config.DIR_RESULTS / f"{context_name}_log.txt"
        
        # Initialize log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Execution Log: {context_name} ===\n")

    def info(self, message: Any):
        """Prints to console and appends to file."""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(str(message) + "\n")

    def save_figure(self, fig: plt.Figure, tag: str):
        """Saves a Matplotlib figure to the results directory."""
        filename = f"{self.context_name}_{tag}.png"
        path = config.DIR_RESULTS / filename
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[IO] Figure saved: {path.name}")
        plt.close(fig)

    def dump_metrics(self, metrics: Dict[str, Any]):
        """Updates the central metrics.json file."""
        json_path = config.DIR_RESULTS / "performance_metrics.json"
        
        current_data = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    current_data = json.load(f)
            except json.JSONDecodeError:
                pass

        current_data[self.context_name] = metrics

        with open(json_path, 'w') as f:
            json.dump(current_data, f, indent=4)
        print(f"[IO] Metrics updated in: {json_path.name}")