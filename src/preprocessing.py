"""
Event-level preprocessing of the eye-tracking data.

The raw data contain one row per eye-movement event (foveation or blink).
This module takes care of everything that happens *before* the events are
aggregated to trial-level features:

1. loading and cleaning the raw data
2. selecting the tasks for an experiment
3. splitting into train and test set
4. handling implausible values and outliers
5. imputing missing values
"""

import random

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from src.config import MODE_ALL, MODE_BALANCE, MODE_SEARCH_VS_MEMO, TRIAL_KEYS

# =============================================================================
# Constants
# =============================================================================

RAW_COLUMNS = [
    "subject", "trial_id", "trial_Cntr", "task", "ans_cor",
    "gondola", "num_houses", "train",
    "duration_ms", "event",
    "x_start", "x_end", "y_start", "y_end",
    "fov_velo_abs_mean", "fov_p_median", "fov_p_mean", "fov_p_min", "fov_p_max",
    "fov_start", "fov_end", "fov_saliency_mean",
    "sac_velo_abs_mean", "sac_amp_dva", "sac_angle_h", "sac_angle_p",
    "gt_object", "gt_object_category", "fov_category",
]

# The first five trials of every task block are treated as practice trials
PRACTICE_TRIAL_RANGES = [(0, 4), (42, 46), (84, 88), (126, 130)]

MIN_EVENTS_PER_TRIAL = 10
MAX_FOVEATION_DURATION_MS = 5000

# Stimulus videos for which the foveation category is missing
VIDEOS_WITHOUT_FOV_CATEGORY = [
    "3_ti26_tp11_td1_g0_h232_c2_t1_l1.mp4",
    "3_ti26_tp11_td1_g0_h232_c2_t1_l0.mp4",
    "6_ti5_tp9_td0_g1_h14_c0_t31_l1.mp4",
]

PUPIL_COLUMNS = ["fov_p_median", "fov_p_min", "fov_p_max"]

# Physiologically plausible value ranges; values outside are set to NaN
PLAUSIBLE_RANGES = {
    "fov_velo_abs_mean": (0, 140),
    "sac_velo_abs_mean": (0, 1000),
}

# Columns that are winsorized at the IQR bounds (learned on the training set)
OUTLIER_COLUMNS = [
    "duration_ms", "fov_velo_abs_mean",
    "fov_p_median", "fov_p_min", "fov_p_max",
    "fov_saliency_mean", "sac_velo_abs_mean", "sac_amp_dva",
]

# Columns whose missing values are imputed
IMPUTE_COLUMNS = [
    "x_start", "x_end", "y_start", "y_end",
    "fov_velo_abs_mean", "fov_saliency_mean",
    "sac_velo_abs_mean", "sac_amp_dva", "sac_angle_h",
]

# Categorical columns used (one-hot encoded) in the KNN distance calculation
IMPUTE_CATEGORICAL_COLUMNS = ["event", "fov_category", "gt_object_category"]


# =============================================================================
# 1. Loading and cleaning
# =============================================================================

def load_raw_data(path):
    """Load the raw event-level data and keep only the relevant columns."""
    return pd.read_csv(path)[RAW_COLUMNS]


def clean_data(df):
    """Apply all trial exclusion criteria and basic transformations."""
    df = df.copy()

    # Both balance variants are treated as one task
    df["task"] = df["task"].replace(["balance_memorization", "balance_search"], "balance")

    # Remove trials with incorrect answers
    df = df[~df["ans_cor"].isin(["False", "0.0"])].drop(columns="ans_cor")

    # Remove practice trials at the beginning of each task block
    for first, last in PRACTICE_TRIAL_RANGES:
        df = df[~df["trial_Cntr"].between(first, last)]

    # Remove trials with too few events
    n_events = df.groupby(TRIAL_KEYS)["event"].transform("size")
    df = df[n_events >= MIN_EVENTS_PER_TRIAL]

    # Remove trials containing an implausibly long foveation
    has_long_foveation = (
        df.assign(is_long=df["duration_ms"] > MAX_FOVEATION_DURATION_MS)
          .groupby(TRIAL_KEYS)["is_long"].transform("any")
    )
    df = df[~has_long_foveation]

    # 'Ground' is stored as an object but belongs to the object categories
    df.loc[df["gt_object"] == "Ground", "gt_object_category"] = "Ground"

    # Remove videos without foveation category
    df = df[~df["trial_id"].isin(VIDEOS_WITHOUT_FOV_CATEGORY)].drop(columns="trial_id")

    # Pupil size relative to each participant's mean pupil size (baseline)
    pupil_baseline = df.groupby("subject")["fov_p_mean"].transform("mean")
    for col in PUPIL_COLUMNS:
        df[col] = df[col] - pupil_baseline
    df = df.drop(columns="fov_p_mean")

    return df.reset_index(drop=True)


# =============================================================================
# 2. Task selection
# =============================================================================

def select_tasks(df, mode):
    """Keep only the tasks that are needed for the given experiment mode."""
    if mode == MODE_ALL:
        return df
    if mode == MODE_BALANCE:
        return df[df["task"] != "freeviewing"]
    if mode == MODE_SEARCH_VS_MEMO:
        return df[df["task"].isin(["search", "memorization"])]
    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# 3. Train/test split
# =============================================================================

def split_train_test(df, mode, train_fraction=0.8, seed=42):
    """
    Split the data into train and test set on trial level.

    - Balance classification: train = search + memorization, test = balance.
    - All other modes: for every subject × task combination, a random
      `train_fraction` of the trials goes into the training set.
    """
    if mode == MODE_BALANCE:
        return df[df["task"] != "balance"], df[df["task"] == "balance"]

    rng = random.Random(seed)
    train_parts, test_parts = [], []

    for subject in df["subject"].unique():
        for task in df["task"].unique():
            df_sub = df[(df["subject"] == subject) & (df["task"] == task)]

            trials = df_sub["trial_Cntr"].unique().tolist()
            n_train = int(len(trials) * train_fraction)
            train_trials = rng.sample(trials, n_train)

            is_train = df_sub["trial_Cntr"].isin(train_trials)
            train_parts.append(df_sub[is_train])
            test_parts.append(df_sub[~is_train])

    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True))


# =============================================================================
# 4. Outlier handling
# =============================================================================

def _iqr_bounds(series, threshold):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - threshold * iqr, q3 + threshold * iqr


def handle_outliers(df, iqr_bounds=None, threshold=1.5, verbose=True):
    """
    Replace implausible values with NaN and winsorize outliers.

    If `iqr_bounds` is None (training set), the IQR bounds are learned from
    `df`. For the test set, pass the bounds learned on the training set.

    Returns
    -------
    df_clean : pd.DataFrame
    iqr_bounds : dict  {column: (lower, upper)}
    """
    df = df.copy()
    learn_bounds = iqr_bounds is None
    if learn_bounds:
        iqr_bounds = {}

    # Implausible values -> NaN (imputed later)
    for col, (min_val, max_val) in PLAUSIBLE_RANGES.items():
        mask = (df[col] < min_val) | (df[col] > max_val)
        if verbose:
            print(f"  Implausible values set to NaN in {col}: "
                  f"{mask.sum()} ({mask.mean() * 100:.2f}%)")
        df.loc[mask, col] = np.nan

    # Winsorize at IQR bounds
    for col in OUTLIER_COLUMNS:
        if learn_bounds:
            iqr_bounds[col] = _iqr_bounds(df[col], threshold)
        lower, upper = iqr_bounds[col]

        mask = (df[col] < lower) | (df[col] > upper)
        if verbose:
            print(f"  Outliers winsorized in {col}: "
                  f"{mask.sum()} ({mask.mean() * 100:.2f}%)")
        df[col] = df[col].clip(lower, upper)

    return df, iqr_bounds


# =============================================================================
# 5. Imputation
# =============================================================================

def impute_missing_values(df, distance_columns, n_neighbors=3):
    """
    Impute missing values with a KNN imputer, separately for every trial.

    Distances are computed on `distance_columns` (numerical) plus the one-hot
    encoded categorical columns. Only `IMPUTE_COLUMNS` are overwritten.
    """
    df = df.copy()

    for _, trial in df.groupby(TRIAL_KEYS):
        idx = trial.index

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        categorical_encoded = pd.DataFrame(
            encoder.fit_transform(trial[IMPUTE_CATEGORICAL_COLUMNS]),
            index=idx,
            columns=encoder.get_feature_names_out(IMPUTE_CATEGORICAL_COLUMNS),
        )
        X_dist = pd.concat([trial[distance_columns], categorical_encoded], axis=1)

        if not X_dist.isna().any().any():
            continue

        imputer = KNNImputer(n_neighbors=n_neighbors)
        X_filled = pd.DataFrame(imputer.fit_transform(X_dist),
                                index=idx, columns=X_dist.columns)
        df.loc[idx, IMPUTE_COLUMNS] = X_filled[IMPUTE_COLUMNS]

    return df
