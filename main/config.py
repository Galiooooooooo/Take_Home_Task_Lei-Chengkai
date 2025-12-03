import os
from pathlib import Path

# ==========================================
# Project Path Configuration
# ==========================================
# Resolve the project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define standard directories
DIR_DATA = PROJECT_ROOT / "data"
DIR_MODELS = PROJECT_ROOT / "models"
DIR_RESULTS = PROJECT_ROOT / "results"

# Ensure directories exist
for path in [DIR_DATA, DIR_MODELS, DIR_RESULTS]:
    path.mkdir(parents=True, exist_ok=True)

# ==========================================
# Global Physics & Simulation Constants
# ==========================================
# Time step: 5 minutes expressed in years
# Calculation: 5 / (365 days * 24 hours * 60 minutes)
DT_YEAR_5MIN = 5.0 / (365 * 24 * 60)

# Fixed volatility assumption for RL environment baseline
FIXED_VOLATILITY = 0.60

# Global Random Seed for reproducibility across modules
GLOBAL_SEED = 2025

# Transaction costs (bps)
COST_BPS = 0.0005