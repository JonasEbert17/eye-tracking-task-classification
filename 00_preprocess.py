"""
Step 0 – Preprocessing and feature engineering.

Turns the raw event-level eye-tracking data into scaled trial-level feature
tables for one of the three experiments.

Usage
-----
    python 00_preprocess.py --mode all
    python 00_preprocess.py --mode search_vs_memorization
    python 00_preprocess.py --mode balance_classification

Output: CSV files in data/processed/<mode>/
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.config import (MODE_BALANCE, MODES, PROCESSED_DIR, RANDOM_SEED,
                        RAW_DATA_PATH, TRIAL_KEYS)
from src.features import engineer_features, engineer_split_features
from src.preprocessing import (clean_data, handle_outliers, impute_missing_values,
                               load_raw_data, select_tasks, split_train_test)

# Columns removed after feature engineering
COLUMNS_TO_DROP = [
    *TRIAL_KEYS,                                            # identifiers
    "fovs_gondola", "fovs_house", "fovs_train",             # summarised in fovs_engaging
    "fovs_engaging_abs", "engaging_obj_present",            # helper columns
    "duration_ms_bin4", "sac_amp_dva_bin4",                 # last time bin
]

# Trials with this many blinks or more are excluded
BLINK_LIMIT = 9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="Which experiment to prepare the data for.")
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA_PATH,
                        help="Path to the raw event-level CSV file.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for the train/test split.")
    return parser.parse_args()


def build_feature_table(events, feature_function):
    """Engineer features, drop helper columns and exclude invalid trials."""
    features = feature_function(events).drop(columns=COLUMNS_TO_DROP)
    features = features[features["blinks"] < BLINK_LIMIT]
    return features.dropna()


def split_X_y(features):
    return features.drop(columns="task"), features["task"]


def scale(scaler, X):
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def main():
    args = parse_args()
    if not args.raw_data.exists():
        raise SystemExit(f"Raw data not found: {args.raw_data}\n"
                         "See data/README.md for where to place the data.")

    output_dir = PROCESSED_DIR / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load, clean, select tasks, split ------------------------------------
    print(f"Loading raw data from {args.raw_data}")
    events = select_tasks(clean_data(load_raw_data(args.raw_data)), args.mode)
    events_train, events_test = split_train_test(events, args.mode, seed=args.seed)

    # --- Outliers (bounds learned on train set only) -------------------------
    print("\nOutlier handling – train set")
    events_train, iqr_bounds = handle_outliers(events_train)
    print("\nOutlier handling – test set")
    events_test, _ = handle_outliers(events_test, iqr_bounds=iqr_bounds)

    # --- Imputation ----------------------------------------------------------
    print("\nImputing missing values ...")
    distance_columns = (events_train.select_dtypes(include="number")
                        .columns.difference(TRIAL_KEYS))
    events_train = impute_missing_values(events_train, distance_columns)
    events_test = impute_missing_values(events_test, distance_columns)

    # --- Feature engineering -------------------------------------------------
    print("Engineering features ...")
    X_train, y_train = split_X_y(build_feature_table(events_train, engineer_features))

    if args.mode == MODE_BALANCE:
        # Full balance trials and balance trials split into two halves
        X_test_full, _ = split_X_y(build_feature_table(events_test, engineer_features))
        y_test_full = pd.Series("balance", index=X_test_full.index, name="task")
        test_sets = {
            "test_full": (X_test_full, y_test_full),
            "test_split": split_X_y(build_feature_table(events_test,
                                                        engineer_split_features)),
        }
    else:
        test_sets = {"test": split_X_y(build_feature_table(events_test,
                                                           engineer_features))}

    # --- Scaling (fitted on train set only) and saving -----------------------
    scaler = RobustScaler().fit(X_train)
    datasets = {"train": (X_train, y_train), **test_sets}

    for name, (X, y) in datasets.items():
        scale(scaler, X).to_csv(output_dir / f"X_{name}.csv", index=False)
        y.to_csv(output_dir / f"y_{name}.csv", index=False)
        print(f"  {name:<10}: {len(X):>5} trials")

    print(f"\nData saved to {output_dir}")


if __name__ == "__main__":
    main()
