import numpy as np
from typing import Tuple, List

# ==========================================
# 1. QED (Quadratic Exponential Diffusion) Model
# ==========================================
def qed_negative_log_likelihood(params: List[float], X: np.ndarray, dt: float) -> float:
    """
    Computes the Negative Log-Likelihood (NLL) for the QED model.
    Dynamics: dX = (theta*X - kappa*X^2 - g*X^3)dt + sigma*X dW
    """
    theta, kappa, g, sigma = params
    
    # Stability constraint
    if sigma <= 0: return 1e10

    X_curr = X[:-1]
    X_next = X[1:]

    # Discretized drift and variance
    drift_term = theta * X_curr - kappa * (X_curr ** 2) - g * (X_curr ** 3)
    mean_next = X_curr + drift_term * dt
    var_next = (sigma ** 2) * (X_curr ** 2) * dt

    # Avoid numerical instability
    var_next = np.maximum(var_next, 1e-9)
    
    # Gaussian Likelihood
    nll_terms = np.log(var_next) + (X_next - mean_next) ** 2 / var_next
    return 0.5 * np.sum(nll_terms)


# ==========================================
# 2. Hawkes Process (Jump Diffusion)
# ==========================================
def compute_hawkes_intensity(events: np.ndarray, baseline: float, alpha: float, beta: float, dt: float) -> np.ndarray:
    """
    Recursive calculation of Hawkes intensity lambda(t).
    lambda(t) = mu + alpha * sum(exp(-beta * (t - ti)))
    """
    T = len(events)
    intensity = np.zeros(T)
    decay_factor = np.exp(-beta * dt)
    
    current_decay = 0.0
    for t in range(T):
        intensity[t] = baseline + alpha * current_decay
        # Update decay state with current event
        current_decay = decay_factor * (current_decay + events[t])
        
    return intensity

def hawkes_nll_single(params: List[float], events: np.ndarray, dt: float) -> float:
    baseline, alpha, beta = params
    
    # Constraints: Positive parameters and stationarity (alpha < beta)
    if baseline <= 0 or alpha < 0 or beta <= 0: return 1e10
    if alpha >= beta: return 1e10

    lambdas = compute_hawkes_intensity(events, baseline, alpha, beta, dt)
    
    # Poisson probability approximation for small dt
    prob_event = lambdas * dt
    prob_event = np.clip(prob_event, 1e-9, 1.0 - 1e-9)
    
    # Binary Cross Entropy / Log Likelihood
    log_likelihood = events * np.log(prob_event) + (1 - events) * np.log(1 - prob_event)
    return -np.sum(log_likelihood)

def hawkes_nll_joint(params: List[float], jumps_pos: np.ndarray, jumps_neg: np.ndarray, dt: float) -> float:
    """
    Joint NLL for positive and negative jump processes sharing a beta decay.
    """
    base_p, base_m, alpha_p, alpha_m, beta = params
    nll_pos = hawkes_nll_single([base_p, alpha_p, beta], jumps_pos, dt)
    nll_neg = hawkes_nll_single([base_m, alpha_m, beta], jumps_neg, dt)
    return nll_pos + nll_neg


# ==========================================
# 3. Monte Carlo Path Generator
# ==========================================
def generate_market_paths(
    n_paths: int, 
    n_steps: int, 
    s0: float, 
    qed_params: Tuple, 
    hawkes_params: Tuple, 
    jump_sizes: Tuple[np.ndarray, np.ndarray], 
    dt_year: float, 
    scaling_factor: float = 1.0
) -> np.ndarray:
    """
    Simulates price paths combining QED diffusion and Hawkes-driven jumps.
    """
    # Unpack parameters
    theta, kappa, g, sigma = qed_params
    base_p, base_m, alpha_p, alpha_m, beta = hawkes_params
    jumps_up_dist, jumps_down_dist = jump_sizes

    # Initialization (Log prices for stability)
    x0_normalized = s0 / scaling_factor
    log_prices = np.zeros((n_steps, n_paths))
    log_prices[0, :] = np.log(x0_normalized)
    
    # Hawkes state variables
    decay_state_p = np.zeros(n_paths)
    decay_state_m = np.zeros(n_paths)
    decay_rate = np.exp(-beta * dt_year)
    sqrt_dt = np.sqrt(dt_year)

    for t in range(n_steps - 1):
        # 1. Continuous Diffusion (QED)
        log_y_curr = log_prices[t, :]
        S_curr = np.exp(log_y_curr)
        
        # Ito's Lemma drift correction for log-process
        drift = (theta - kappa * S_curr - g * S_curr ** 2) - 0.5 * sigma ** 2
        diffusion = sigma * np.random.normal(size=n_paths)
        log_y_cont = log_y_curr + drift * dt_year + diffusion * sqrt_dt

        # 2. Jump Process (Hawkes)
        lambda_p = base_p + alpha_p * decay_state_p
        lambda_m = base_m + alpha_m * decay_state_m
        
        # Event occurrence probabilities
        prob_p = np.clip(lambda_p * dt_year, 0, 1)
        prob_m = np.clip(lambda_m * dt_year, 0, 1)

        has_jump_p = np.random.rand(n_paths) < prob_p
        has_jump_m = np.random.rand(n_paths) < prob_m
        
        # Mutually exclusive jump assumption (simplified)
        has_jump_m[has_jump_p & has_jump_m] = False

        # Sample jump sizes
        jump_vals_p = np.zeros(n_paths)
        jump_vals_m = np.zeros(n_paths)
        
        if np.any(has_jump_p):
            jump_vals_p[has_jump_p] = np.random.choice(jumps_up_dist, size=np.sum(has_jump_p), replace=True)
        if np.any(has_jump_m):
            jump_vals_m[has_jump_m] = np.random.choice(jumps_down_dist, size=np.sum(has_jump_m), replace=True)

        # 3. Update State
        log_prices[t + 1, :] = log_y_cont + jump_vals_p - jump_vals_m
        
        decay_state_p = decay_rate * (decay_state_p + has_jump_p.astype(float))
        decay_state_m = decay_rate * (decay_state_m + has_jump_m.astype(float))

    return np.exp(log_prices) * scaling_factor


def construct_ohlc_candles(close_prices: np.ndarray, volatility_proxy: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic generation of High/Low prices from Close prices.
    """
    n_steps, n_paths = close_prices.shape
    highs = np.zeros_like(close_prices)
    lows = np.zeros_like(close_prices)
    
    # Scale volatility to step size (assuming daily vol input)
    step_vol = volatility_proxy / np.sqrt(24 * 12) 

    for t in range(1, n_steps):
        prev_close = close_prices[t - 1, :]
        curr_close = close_prices[t, :]
        
        base_max = np.maximum(prev_close, curr_close)
        base_min = np.minimum(prev_close, curr_close)

        # Random wicks
        wick_up = np.abs(np.random.normal(0, step_vol, n_paths)) * curr_close
        wick_down = np.abs(np.random.normal(0, step_vol, n_paths)) * curr_close

        highs[t, :] = base_max + wick_up
        lows[t, :] = base_min - wick_down

    # Initial condition
    highs[0, :] = close_prices[0, :]
    lows[0, :] = close_prices[0, :]
    
    return highs, lows