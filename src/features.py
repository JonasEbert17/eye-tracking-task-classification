"""
Trial-level feature engineering.

The preprocessed data contain one row per eye-movement event. The functions
in this module aggregate these events to one row per trial (unique
subject × trial_Cntr combination) with engineered features.
"""

from functools import reduce

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import TRIAL_KEYS

# =============================================================================
# Constants
# =============================================================================

N_TIME_BINS = 4
TRIAL_DURATION_MS = 10_000

FOV_CATEGORY_NAMES = {"B": "fov_B", "D": "fov_D", "I": "fov_I", "R": "fov_R"}
FOV_CATEGORY_DURATION_NAMES = {
    "B": "fov_B_dur_mean", "D": "fov_D_dur_mean",
    "I": "fov_I_dur_mean", "R": "fov_R_dur_mean",
}
OBJECT_CATEGORY_NAMES = {
    "animal": "fovs_animals", "gondola": "fovs_gondola", "house": "fovs_house",
    "person": "fovs_persons", "plant": "fovs_plant", "train": "fovs_train",
    "Ground": "fovs_Ground",
}

# Summary statistics computed over all events of a trial
AGGREGATIONS = {
    "duration_ms":       ["min", "max", "mean", "var"],
    "fov_velo_abs_mean": ["mean", "var", "min", "max"],
    "fov_saliency_mean": ["mean", "var"],
    "sac_velo_abs_mean": ["mean", "var", "min", "max"],
    "fov_p_median":      ["median", "min", "max", "var"],
    "fov_p_min":         ["median", "min", "max", "var"],
    "fov_p_max":         ["median", "min", "max", "var"],
    "sac_amp_dva":       ["mean", "var", "min", "max"],
    "sac_angle_h":       ["mean", "var"],
    "sac_angle_p":       ["mean", "var"],
}

# Features whose development over time is described per time bin
TIME_BIN_FEATURES = ["duration_ms", "sac_amp_dva"]

# Count-like features that depend on the length of the analysed time window.
# When a trial is split in half, these are multiplied by 2 so that the parts
# are on the same scale as full-length trials.
COUNT_FEATURES = [
    "fovs_count", "blinks", "fov_B", "fov_D", "fov_I", "fov_R",
    "fovs_Ground", "fovs_animals", "fovs_gondola", "fovs_house",
    "fovs_person", "fovs_plant", "fovs_train", "fovs_engaging",
    "scanpath_length",
]


# =============================================================================
# Helper features (one row per trial)
# =============================================================================

def compute_first_gaze_duration(df):
    """
    Duration of the first gaze on an object: the first 'D' foveation plus
    all directly following 'I' foveations.
    """
    rows = []
    for (subject, trial), events in df.groupby(TRIAL_KEYS):
        categories = events["fov_category"].to_numpy()
        durations = events["duration_ms"].to_numpy()

        duration = np.nan
        first_d = np.flatnonzero(categories == "D")
        if first_d.size > 0:
            start = first_d[0]
            duration = durations[start]
            for category, dur in zip(categories[start + 1:], durations[start + 1:]):
                if category != "I":
                    break
                duration += dur

        rows.append({"subject": subject, "trial_Cntr": trial,
                     "first_gaze_duration": duration})
    return pd.DataFrame(rows)


def compute_change_rate(df, feature):
    """Linear slope of `feature` over the time bins (1–4) of each trial."""
    rows = []
    for (subject, trial), events in df.groupby(TRIAL_KEYS):
        events = events.dropna(subset=[feature, "time_bin"])
        if events.empty:
            continue

        # 'bin1' ... 'bin4'  ->  1 ... 4
        x = (events["time_bin"].cat.codes.to_numpy() + 1).astype(float).reshape(-1, 1)
        y = events[feature].to_numpy()

        slope = np.nan
        if len(x) >= 2:
            slope = LinearRegression().fit(x, y).coef_[0]

        rows.append({"subject": subject, "trial_Cntr": trial,
                     f"{feature}_linear_slope": slope})
    return pd.DataFrame(rows)


def compute_scanned_area(df):
    """Area of the bounding box around all gaze positions of a trial."""
    rows = []
    for (subject, trial), events in df.groupby(TRIAL_KEYS):
        x = np.concatenate([events["x_start"].to_numpy(), events["x_end"].to_numpy()])
        y = np.concatenate([events["y_start"].to_numpy(), events["y_end"].to_numpy()])
        area = (x.max() - x.min()) * (y.max() - y.min())
        rows.append({"subject": subject, "trial_Cntr": trial, "scanned_area": area})
    return pd.DataFrame(rows)


def count_unique_objects(df):
    """Number of distinct objects looked at in a trial (ground excluded)."""
    rows = []
    for (subject, trial), events in df.groupby(TRIAL_KEYS):
        n_objects = events.loc[events["gt_object"] != "Ground", "gt_object"].nunique()
        rows.append({"subject": subject, "trial_Cntr": trial,
                     "unique_object_count": n_objects})
    return pd.DataFrame(rows)


# =============================================================================
# Main feature engineering
# =============================================================================

def engineer_features(df):
    """
    Aggregate event-level data to trial level.

    Returns a DataFrame with one row per trial, containing the trial keys,
    the task label and all engineered features.
    """
    df = df.copy()

    # --- Time bins (equal-width, over the whole recording) -------------------
    bin_edges = np.linspace(0, df["fov_end"].max(), N_TIME_BINS + 1)
    bin_labels = [f"bin{i}" for i in range(1, N_TIME_BINS + 1)]
    df["time_bin"] = pd.cut(df["fov_start"], bins=bin_edges,
                            labels=bin_labels, include_lowest=True)

    grouped = df.groupby(TRIAL_KEYS)

    # --- Task label ----------------------------------------------------------
    task = grouped["task"].first().reset_index()

    # --- Event counts --------------------------------------------------------
    fovs_count = (df[df["event"] == "FOV"].groupby(TRIAL_KEYS).size()
                  .rename("fovs_count").reset_index())
    blinks = (grouped["event"].apply(lambda e: (e == "BLINK").sum())
              .rename("blinks").reset_index())

    # --- Foveation categories: counts and mean durations ---------------------
    df["fov_category"] = df["fov_category"].replace("-", np.nan)

    fov_category_counts = (
        pd.crosstab(index=[df["subject"], df["trial_Cntr"]], columns=df["fov_category"])
          .reset_index()
          .rename(columns=FOV_CATEGORY_NAMES)
    )
    fov_category_durations = (
        df.groupby(TRIAL_KEYS + ["fov_category"])["duration_ms"].mean()
          .unstack(fill_value=0)
          .reset_index()
          .rename(columns=FOV_CATEGORY_DURATION_NAMES)
    )

    # --- Object categories ---------------------------------------------------
    object_category_counts = (
        pd.crosstab(index=[df["subject"], df["trial_Cntr"]], columns=df["gt_object_category"])
          .reset_index()
          .rename(columns=OBJECT_CATEGORY_NAMES)
    )

    # Foveations on "engaging" objects (gondolas, houses, trains), normalised
    # by how many of these objects are present in the video (-1 if none)
    engaging_present = grouped[["gondola", "num_houses", "train"]].first()
    engaging_present["engaging_obj_present"] = (
        engaging_present["gondola"]
        + engaging_present["num_houses"]
        + engaging_present["train"]
    )
    engaging_present = engaging_present[["engaging_obj_present"]].reset_index()

    object_category_counts["fovs_engaging_abs"] = (
        object_category_counts.get("fovs_gondola", 0)
        + object_category_counts.get("fovs_house", 0)
        + object_category_counts.get("fovs_train", 0)
    )
    object_category_counts = object_category_counts.merge(
        engaging_present, on=TRIAL_KEYS, how="left")
    object_category_counts["fovs_engaging"] = np.where(
        object_category_counts["engaging_obj_present"] > 0,
        object_category_counts["fovs_engaging_abs"]
        / object_category_counts["engaging_obj_present"],
        -1,
    )

    # --- Summary statistics --------------------------------------------------
    summary_stats = grouped.agg(AGGREGATIONS)
    summary_stats.columns = ["_".join(col) for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()

    scanpath_length = (grouped["sac_amp_dva"].sum()
                       .rename("scanpath_length").reset_index())

    # --- Temporal features ---------------------------------------------------
    time_bin_means = [
        df.groupby(TRIAL_KEYS + ["time_bin"], observed=False)[feat].mean()
          .unstack("time_bin")
          .add_prefix(f"{feat}_")
          .reset_index()
        for feat in TIME_BIN_FEATURES
    ]

    first_bin = df[df["time_bin"] == "bin1"]
    first_bin_variances = [
        first_bin.groupby(TRIAL_KEYS)[feat].var()
                 .rename(f"{feat}_var_bin1").reset_index()
        for feat in TIME_BIN_FEATURES
    ]

    slopes = [compute_change_rate(df, feat) for feat in TIME_BIN_FEATURES]

    # --- Remaining trial-level features --------------------------------------
    first_gaze = compute_first_gaze_duration(df)
    scanned_area = compute_scanned_area(df)
    unique_objects = count_unique_objects(df)

    # --- Merge everything into one table -------------------------------------
    feature_tables = [
        task,
        summary_stats,
        fovs_count,
        blinks,
        fov_category_counts,
        fov_category_durations,
        object_category_counts,
        scanpath_length,
        *time_bin_means,
        *first_bin_variances,
        first_gaze,
        *slopes,
        scanned_area,
        unique_objects,
    ]
    return reduce(lambda left, right: left.merge(right, on=TRIAL_KEYS, how="left"),
                  feature_tables)


def engineer_split_features(df, split_time_ms=TRIAL_DURATION_MS / 2):
    """
    Split every balance trial at `split_time_ms` and compute features for
    both halves separately.

    The halves are labelled 'balance_part1' and 'balance_part2'. Timestamps of
    the second half are shifted to start at 0, and count-like features are
    multiplied by 2 to match the scale of full-length trials.
    """
    parts = []
    for (_, trial), events in df.groupby(TRIAL_KEYS):
        first_half = events[events["fov_start"] < split_time_ms].copy()
        second_half = events[events["fov_start"] >= split_time_ms].copy()

        if not first_half.empty and first_half["task"].iloc[0] == "balance":
            first_half["task"] = "balance_part1"
            first_half["trial_Cntr"] = f"{trial}_part1"

        if not second_half.empty and second_half["task"].iloc[0] == "balance":
            second_half["task"] = "balance_part2"
            second_half["trial_Cntr"] = f"{trial}_part2"
            second_half["fov_start"] -= split_time_ms
            second_half["fov_end"] -= split_time_ms

        parts.extend([first_half, second_half])

    features = engineer_features(pd.concat(parts, ignore_index=True))

    is_part = features["task"].isin(["balance_part1", "balance_part2"])
    for feat in COUNT_FEATURES:
        if feat in features.columns:
            features.loc[is_part, feat] *= 2

    return features
