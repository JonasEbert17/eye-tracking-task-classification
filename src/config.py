"""Central configuration: file paths, random seed and experiment modes."""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "all_data_analysis_komplett.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Experiment modes (which tasks are included and how the data are split)
# -----------------------------------------------------------------------------
MODE_ALL = "all"                                  # Experiment 1
MODE_SEARCH_VS_MEMO = "search_vs_memorization"    # Experiment 2
MODE_BALANCE = "balance_classification"           # Experiment 3

MODES = [MODE_ALL, MODE_SEARCH_VS_MEMO, MODE_BALANCE]

# Identifies a single trial in the data
TRIAL_KEYS = ["subject", "trial_Cntr"]
